import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_config, load_jsonl
from src.diplan.failure_memory import build_failure_memory
from src.diplan.metrics import (
    aggregate_method_metrics,
    first_error_step,
    plan_execution_consistency,
    recovery_at_error,
    trap_at_1,
)
from src.diplan.plan_schema import path_to_structured_plan
from src.diplan.tool_decoder import path_to_tool_calls
from src.diplan.torch_pipeline import (
    EOS,
    PAD,
    DiffusionPlanner,
    MLPPlanner,
    PathAutoencoder,
    ValueRanker,
    collate_value,
    load_vocab,
    pad_2d,
    sample_latent,
)


def _build_memory_index(
    train_rows: List[Dict],
    max_postings_per_token: int,
) -> Tuple[Dict[str, List[int]], List[List[str]]]:
    token_to_ids: Dict[str, List[int]] = defaultdict(list)
    path_bank: List[List[str]] = []
    for i, row in enumerate(train_rows):
        path = row.get("oracle_path", [])
        if not isinstance(path, list) or not path:
            continue
        path_bank.append(path)
        path_id = len(path_bank) - 1
        seen = set()
        for t in row.get("query_tokens", []):
            if not isinstance(t, str):
                continue
            tok = t.strip().lower()
            if not tok or tok in seen:
                continue
            seen.add(tok)
            posting = token_to_ids[tok]
            if len(posting) < max_postings_per_token:
                posting.append(path_id)
    return token_to_ids, path_bank


def _retrieve_memory_candidates(
    query_tokens: List[str],
    token_to_ids: Dict[str, List[int]],
    path_bank: List[List[str]],
    top_k: int,
) -> List[List[str]]:
    score = defaultdict(int)
    for t in set(x.strip().lower() for x in query_tokens if isinstance(x, str) and x.strip()):
        for path_id in token_to_ids.get(t, []):
            score[path_id] += 1
    if not score:
        return []
    ranked = sorted(
        score.items(),
        key=lambda x: (-x[1], abs(len(path_bank[x[0]]) - max(1, len(query_tokens)))),
    )
    out: List[List[str]] = []
    seen = set()
    for path_id, _ in ranked:
        p = path_bank[path_id]
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(p))
        if len(out) >= top_k:
            break
    return out


def _merge_candidates(generated: List[List[str]], memory: List[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    seen = set()
    for src in (generated, memory):
        for c in src:
            key = tuple(c)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out


def _memory_rank_scores(memory: List[List[str]]) -> Dict[Tuple[str, ...], float]:
    scores: Dict[Tuple[str, ...], float] = {}
    for rank, c in enumerate(memory):
        key = tuple(c)
        if key not in scores:
            scores[key] = 1.0 / float(rank + 1)
    return scores


def _filter_feasible_candidates(candidates: List[List[str]], constraints: Dict) -> Tuple[List[List[str]], int]:
    kept: List[List[str]] = []
    dropped = 0
    for c in candidates:
        feasible, _ = _is_feasible(c, constraints)
        if feasible:
            kept.append(c)
        else:
            dropped += 1
    return kept, dropped


def _is_feasible(path: List[str], constraints: Dict) -> Tuple[bool, List[str]]:
    violations = []
    if len(path) > int(constraints.get("max_steps", 8)):
        violations.append("max_steps_exceeded")
    banned = set(constraints.get("banned_relations", []))
    for rel in path:
        if rel in banned:
            violations.append("banned_relation")

    def _action_stage(token: str) -> str:
        if not isinstance(token, str):
            return ""
        if "::" in token:
            return token.split("::", 1)[0].strip().upper()
        if "(" in token:
            return token.split("(", 1)[0].strip().upper()
        if "_" in token:
            return token.split("_", 1)[0].strip().upper()
        return token.strip().upper()

    def _matches(token: str, pattern: str) -> bool:
        t = str(token).strip().upper()
        p = str(pattern).strip().upper()
        if not p:
            return False
        return t == p or t.startswith(p) or _action_stage(t) == p

    # Generic forbidden action constraints (supports exact/prefix/stage-level patterns).
    forbidden_actions = constraints.get("forbidden_actions", [])
    if isinstance(forbidden_actions, list):
        for a in path:
            for ptn in forbidden_actions:
                if _matches(a, str(ptn)):
                    violations.append("forbidden_action")
                    break

    # Stage order constraints for long-horizon tasks (e.g., clinical workflow).
    required_stage_order = constraints.get("required_stage_order", [])
    if isinstance(required_stage_order, list) and required_stage_order:
        order_idx = {str(s).strip().upper(): i for i, s in enumerate(required_stage_order)}
        last = -1
        for a in path:
            st = _action_stage(a)
            if st in order_idx:
                cur = order_idx[st]
                if cur < last:
                    violations.append("stage_order_violation")
                    break
                last = cur

    # Pairwise precedence constraints: first must happen before second (if second appears).
    must_precede = constraints.get("must_precede", [])
    if isinstance(must_precede, list):
        for rule in must_precede:
            if not isinstance(rule, dict):
                continue
            first = str(rule.get("first", "")).strip()
            second = str(rule.get("second", "")).strip()
            if not first or not second:
                continue
            first_pos = [i for i, a in enumerate(path) if _matches(a, first)]
            second_pos = [i for i, a in enumerate(path) if _matches(a, second)]
            if second_pos:
                if (not first_pos) or (min(second_pos) < min(first_pos)):
                    violations.append("precedence_violation")
                    break

    # For a target action, prerequisite actions must already exist before it.
    required_before = constraints.get("required_before", {})
    if isinstance(required_before, dict):
        for tgt, prereqs in required_before.items():
            if not isinstance(prereqs, list):
                continue
            tgt_pos = [i for i, a in enumerate(path) if _matches(a, str(tgt))]
            if not tgt_pos:
                continue
            first_tgt = min(tgt_pos)
            for pre in prereqs:
                pre_pos = [i for i, a in enumerate(path) if _matches(a, str(pre))]
                if (not pre_pos) or (min(pre_pos) > first_tgt):
                    violations.append("prerequisite_missing")
                    break
            if "prerequisite_missing" in violations:
                break
    return len(violations) == 0, sorted(set(violations))


def _first_feasible_memory_idx(
    cands: List[List[str]],
    memory_candidates: List[List[str]],
    constraints: Dict,
    executed_prefix: List[str] | None = None,
) -> int:
    key_to_idx = {tuple(c): i for i, c in enumerate(cands)}
    prefix = executed_prefix or []
    for mem_path in memory_candidates:
        idx = key_to_idx.get(tuple(mem_path), -1)
        if idx < 0:
            continue
        if prefix and mem_path[: len(prefix)] != prefix:
            continue
        if prefix and len(mem_path) <= len(prefix):
            continue
        path_to_check = mem_path if not prefix else prefix + [mem_path[len(prefix)]]
        feasible, _ = _is_feasible(path_to_check, constraints)
        if feasible:
            return idx
    return -1


def _maybe_apply_retrieval_fusion(
    idx_sorted: List[int],
    cands: List[List[str]],
    memory_candidates: List[List[str]],
    constraints: Dict,
    stage_scores: Dict[int, float],
    enabled: bool,
    margin: float,
    executed_prefix: List[str] | None = None,
) -> Tuple[List[int], bool]:
    if not enabled or not idx_sorted:
        return idx_sorted, False
    anchor_idx = _first_feasible_memory_idx(
        cands=cands,
        memory_candidates=memory_candidates,
        constraints=constraints,
        executed_prefix=executed_prefix,
    )
    if anchor_idx < 0:
        return idx_sorted, False
    best_idx = idx_sorted[0]
    if best_idx == anchor_idx:
        return idx_sorted, False
    best_score = float(stage_scores.get(best_idx, 0.0))
    anchor_score = float(stage_scores.get(anchor_idx, 0.0))
    if best_score >= anchor_score + margin:
        return idx_sorted, False
    fused = [anchor_idx] + [i for i in idx_sorted if i != anchor_idx]
    return fused, True


def _save_summary_csv(path: str, summary: Dict[str, Dict]) -> None:
    ensure_dir(str(Path(path).parent))
    base_fields = [
        "method",
        "success_rate",
        "first_error_step",
        "recovery_at_error",
        "trap_at_1",
        "plan_feasibility",
        "constraint_violation_rate",
        "plan_execution_consistency",
        "token_cost",
        "latency_cost",
        "diversity_coverage",
    ]
    extra_fields = sorted(
        {
            k
            for met in summary.values()
            for k in met.keys()
            if k not in set(base_fields) and k != "method"
        }
    )
    fields = base_fields + extra_fields
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m, met in summary.items():
            row = {"method": m}
            row.update(met)
            w.writerow(row)


def _build_expected_len_prior(
    train_rows: List[Dict],
    bucket_size: int,
) -> Tuple[float, Dict[int, float]]:
    bucket_lens: Dict[int, List[int]] = defaultdict(list)
    all_lens: List[int] = []
    for row in train_rows:
        q_len = len(row.get("query_tokens", []))
        p_len = len(row.get("oracle_path", []))
        if p_len <= 0:
            continue
        b = q_len // max(1, bucket_size)
        bucket_lens[b].append(p_len)
        all_lens.append(p_len)
    global_mean = sum(all_lens) / max(1, len(all_lens))
    bucket_mean = {b: (sum(v) / max(1, len(v))) for b, v in bucket_lens.items()}
    return float(global_mean), bucket_mean


def _expected_len_from_prior(
    query_tokens: List[str],
    bucket_size: int,
    global_mean: float,
    bucket_mean: Dict[int, float],
) -> float:
    b = len(query_tokens) // max(1, bucket_size)
    return float(bucket_mean.get(b, global_mean))


def _resolve_prefix_step_alpha(
    row: Dict,
    cfg: Dict,
    default_alpha: float,
) -> Tuple[float, str]:
    if not bool(cfg.get("adaptive_prefix_alpha_enabled", False)):
        return float(default_alpha), "fixed"

    mode = str(cfg.get("adaptive_prefix_alpha_mode", "dataset")).lower()
    ds = str(row.get("dataset", "")).lower()

    ds_map_raw = cfg.get("adaptive_prefix_alpha_by_dataset", {}) or {}
    ds_map = {str(k).lower(): float(v) for k, v in ds_map_raw.items()} if isinstance(ds_map_raw, dict) else {}

    # Priority 1: dataset-specific alpha.
    if mode in {"dataset", "hybrid", "dataset_then_len"} and ds in ds_map:
        return float(ds_map[ds]), f"dataset:{ds}"

    # Priority 2: query-length heuristic fallback.
    if mode in {"query_len", "hybrid", "dataset_then_len"}:
        q_len = len(row.get("query_tokens", []))
        short_thr = int(cfg.get("adaptive_prefix_alpha_query_len_short_thr", 10))
        long_thr = int(cfg.get("adaptive_prefix_alpha_query_len_long_thr", 18))
        short_alpha = float(cfg.get("adaptive_prefix_alpha_short", max(0.0, default_alpha * 0.25)))
        long_alpha = float(cfg.get("adaptive_prefix_alpha_long", default_alpha))
        mid_alpha = float(cfg.get("adaptive_prefix_alpha_mid", default_alpha))
        if q_len <= short_thr:
            return short_alpha, f"query_len:short<=${short_thr}".replace("$", "")
        if q_len >= long_thr:
            return long_alpha, f"query_len:long>=${long_thr}".replace("$", "")
        return mid_alpha, f"query_len:mid({short_thr},{long_thr})"

    return float(default_alpha), "fixed_fallback"


def _dedupe_with_count(candidates: List[List[str]]) -> Tuple[List[List[str]], Dict[Tuple[str, ...], int]]:
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    out: List[List[str]] = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        counts[key] += 1
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out, counts


@torch.no_grad()
def _decode_latents_to_paths(
    autoencoder: PathAutoencoder,
    path_vocab,
    z: torch.Tensor,
    max_path_len: int,
) -> List[List[str]]:
    seq_ids, pred_lens = autoencoder.decode_greedy(
        z,
        bos_id=path_vocab.stoi["<bos>"],
        eos_id=path_vocab.stoi[EOS],
        max_len=max_path_len,
    )
    cands = []
    for ids, ln in zip(seq_ids, pred_lens):
        ids = ids[: max(1, min(ln, max_path_len))]
        cands.append(path_vocab.decode(ids, skip_special=True))
    return cands


@torch.no_grad()
def _sample_latent_value_guided(
    planner: DiffusionPlanner,
    autoencoder: PathAutoencoder,
    value_model: ValueRanker | None,
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    q_ids: torch.Tensor,
    latent_dim: int,
    diffusion_steps: int,
    max_path_len: int,
    device: torch.device,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
    prediction_target: str,
    guidance_scale: float,
    guidance_interval: int,
    guidance_temperature: float,
    expected_len: float | None,
    length_penalty_alpha: float,
) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    z = torch.randn((q_ids.size(0), latent_dim), device=device)
    target = prediction_target.lower()
    interval = max(1, int(guidance_interval))
    temp = max(1e-6, float(guidance_temperature))

    for t in reversed(range(diffusion_steps)):
        t_idx = torch.full((q_ids.size(0),), t, dtype=torch.long, device=device)
        t_norm = t_idx.float() / max(1, diffusion_steps - 1)
        a_t = alpha_bars[t]
        model_out = planner(z, q_ids, t_norm)
        if target == "z0":
            z0 = model_out
            eps = (z - a_t.sqrt() * z0) / (1.0 - a_t).sqrt().clamp(min=1e-6)
        else:
            eps = model_out
            z0 = (z - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-6)

        should_guide = value_model is not None and guidance_scale > 0.0 and (t % interval == 0 or t == diffusion_steps - 1)
        if should_guide:
            z_decode = z0
            if latent_mean is not None and latent_std is not None:
                z_decode = z_decode * latent_std + latent_mean
            cands = _decode_latents_to_paths(autoencoder, path_vocab, z_decode, max_path_len)
            scores = _score_candidates(
                value_model=value_model,
                path_vocab=path_vocab,
                query_vocab=query_vocab,
                query_tokens=query_tokens,
                candidates=cands,
                device=device,
                expected_len=expected_len,
                length_penalty_alpha=length_penalty_alpha,
            )
            weights = torch.softmax(torch.tensor(scores, dtype=torch.float32, device=device) / temp, dim=0).view(-1, 1)
            center = (weights * z0).sum(dim=0, keepdim=True)
            z0 = z0 + float(guidance_scale) * (center - z0)

        if t == 0:
            z = z0
            continue
        a_prev = alpha_bars[t - 1]
        z = a_prev.sqrt() * z0 + (1.0 - a_prev).sqrt() * eps
    return z


@torch.no_grad()
def _generate_candidates(
    planner: DiffusionPlanner | MLPPlanner,
    autoencoder: PathAutoencoder,
    value_model: ValueRanker | None,
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    num_candidates: int,
    diffusion_steps: int,
    max_path_len: int,
    device: torch.device,
    latent_mean: torch.Tensor | None = None,
    latent_std: torch.Tensor | None = None,
    prediction_target: str = "z0",
    planner_type: str = "diffusion",
    candidate_latent_jitter_std: float = 0.0,
    value_guided_sampling: bool = False,
    value_guidance_scale: float = 0.0,
    value_guidance_interval: int = 4,
    value_guidance_temperature: float = 0.5,
    expected_len: float | None = None,
    length_penalty_alpha: float = 0.0,
) -> Tuple[List[List[str]], float]:
    q_ids = query_vocab.encode(query_tokens, add_bos_eos=False, max_len=24)
    if not q_ids:
        q_ids = [query_vocab.stoi[PAD]]
    q_batch = torch.tensor([q_ids for _ in range(num_candidates)], dtype=torch.long, device=device)
    if planner_type == "mlp":
        z = planner(q_batch)
    elif value_guided_sampling and value_model is not None:
        z = _sample_latent_value_guided(
            planner=planner,
            autoencoder=autoencoder,
            value_model=value_model,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            query_tokens=query_tokens,
            q_ids=q_batch,
            latent_dim=planner.latent_dim,
            diffusion_steps=diffusion_steps,
            max_path_len=max_path_len,
            device=device,
            latent_mean=latent_mean,
            latent_std=latent_std,
            prediction_target=prediction_target,
            guidance_scale=value_guidance_scale,
            guidance_interval=value_guidance_interval,
            guidance_temperature=value_guidance_temperature,
            expected_len=expected_len,
            length_penalty_alpha=length_penalty_alpha,
        )
    else:
        z = sample_latent(
            planner=planner,
            q_ids=q_batch,
            latent_dim=planner.latent_dim,
            diffusion_steps=diffusion_steps,
            device=device,
            prediction_target=prediction_target,
        )
    if candidate_latent_jitter_std > 0.0:
        z = z + torch.randn_like(z) * candidate_latent_jitter_std
    if latent_mean is not None and latent_std is not None:
        z = z * latent_std + latent_mean
    cands = _decode_latents_to_paths(autoencoder, path_vocab, z, max_path_len)
    uniq = len({tuple(x) for x in cands}) / max(1, len(cands))
    return cands, float(uniq)


@torch.no_grad()
def _score_candidates(
    value_model: ValueRanker | None,
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    candidates: List[List[str]],
    device: torch.device,
    expected_len: float | None = None,
    length_penalty_alpha: float = 0.0,
) -> List[float]:
    expected = float(expected_len) if expected_len is not None else float(max(1, len(query_tokens)))
    if value_model is None:
        return [-(abs(len(c) - expected)) for c in candidates]
    q = query_vocab.encode(query_tokens, add_bos_eos=False, max_len=24)
    if not q:
        q = [query_vocab.stoi[PAD]]
    rows = []
    for c in candidates:
        p = path_vocab.encode(c, add_bos_eos=False, max_len=8)
        if not p:
            p = [path_vocab.stoi[PAD]]
        rows.append((q, p, 0.0))
    q_ids, p_ids, _ = collate_value(rows, query_vocab.stoi[PAD], path_vocab.stoi[PAD])
    logits = value_model(q_ids.to(device), p_ids.to(device))
    scores = logits.detach().cpu().tolist()
    if length_penalty_alpha > 0.0:
        scores = [s - length_penalty_alpha * abs(len(c) - expected) for s, c in zip(scores, candidates)]
    return scores


@torch.no_grad()
def _prefix_step_penalties(
    value_model: ValueRanker | None,
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    candidates: List[List[str]],
    device: torch.device,
    gamma: float = 0.85,
) -> List[float]:
    if value_model is None or not candidates:
        return [0.0 for _ in candidates]
    q = query_vocab.encode(query_tokens, add_bos_eos=False, max_len=24)
    if not q:
        q = [query_vocab.stoi[PAD]]
    penalties = [0.0 for _ in candidates]
    max_len = max(len(c) for c in candidates) if candidates else 0
    for step in range(1, max_len + 1):
        rows = []
        idxs = []
        for i, c in enumerate(candidates):
            if len(c) < step:
                continue
            pref = c[:step]
            p = path_vocab.encode(pref, add_bos_eos=False, max_len=8)
            if not p:
                p = [path_vocab.stoi[PAD]]
            rows.append((q, p, 0.0))
            idxs.append(i)
        if not rows:
            continue
        q_ids, p_ids, _ = collate_value(rows, query_vocab.stoi[PAD], path_vocab.stoi[PAD])
        s = value_model(q_ids.to(device), p_ids.to(device)).detach().cpu().tolist()
        s_max = max(s)
        decay = gamma ** (step - 1)
        for ii, sv in zip(idxs, s):
            deficit = max(0.0, s_max - float(sv))
            penalties[ii] += decay * deficit
    return penalties


@torch.no_grad()
def _predict_diplan(
    row: Dict,
    planner: DiffusionPlanner | MLPPlanner,
    autoencoder: PathAutoencoder,
    value_model: ValueRanker | None,
    path_vocab,
    query_vocab,
    device: torch.device,
    num_candidates: int,
    diffusion_steps: int,
    receding_horizon: bool,
    memory_candidates: List[List[str]],
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
    prediction_target: str,
    planner_type: str,
    candidate_latent_jitter_std: float,
    candidate_multi_jitter_stds: List[float],
    expected_total_len: float,
    length_penalty_alpha: float,
    value_score_sign: float,
    rerank_stage1_topk: int,
    rerank_consensus_weight: float,
    rerank_memory_bonus: float,
    rerank_memory_rank_bonus: float,
    rerank_stage2_length_penalty_alpha: float,
    rerank_prefix_consensus_weight: float,
    prefix_step_penalty_alpha: float,
    prefix_step_penalty_gamma: float,
    value_guided_sampling: bool,
    value_guidance_scale: float,
    value_guidance_interval: int,
    value_guidance_temperature: float,
    retrieval_fusion_enabled: bool,
    retrieval_fusion_margin: float,
    disable_generated_candidates: bool,
    save_candidate_pool_topk: int,
    save_episode_trace: bool,
) -> Dict:
    max_steps = int(row["constraints"].get("max_steps", 8))
    query_tokens = row["query_tokens"]
    all_candidates = []
    pool_seen = set()
    all_violations: List[str] = []
    candidate_debug: List[Dict] = []
    episode_trace: List[Dict] = []
    retrieval_fusion_used = False
    executed: List[str] = []
    last_plan: List[str] = []
    if not receding_horizon:
        generated_all: List[List[str]] = []
        if not disable_generated_candidates:
            jitter_list = [candidate_latent_jitter_std] + [x for x in candidate_multi_jitter_stds if x != candidate_latent_jitter_std]
            for jit in jitter_list:
                c_gen, _ = _generate_candidates(
                    planner,
                    autoencoder,
                    value_model,
                    path_vocab,
                    query_vocab,
                    query_tokens,
                    num_candidates,
                    diffusion_steps,
                    max_steps,
                    device,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    prediction_target=prediction_target,
                    planner_type=planner_type,
                    candidate_latent_jitter_std=jit,
                    value_guided_sampling=value_guided_sampling,
                    value_guidance_scale=value_guidance_scale,
                    value_guidance_interval=value_guidance_interval,
                    value_guidance_temperature=value_guidance_temperature,
                    expected_len=expected_total_len,
                    length_penalty_alpha=length_penalty_alpha,
                )
                generated_all.extend(c_gen)
        generated_unique, generated_counts = _dedupe_with_count(generated_all)
        cands = _merge_candidates(generated_unique, memory_candidates)
        if not cands:
            return {
                "planned_path": [],
                "executed_path": [],
                "candidate_count": 0,
                "candidate_unique_ratio": 0.0,
                "violations": [],
                "replanning_steps": 1,
                "candidate_pool_hit": False,
                "candidate_pool_size": 0,
                "candidate_pool_top": [],
                "retrieval_fusion_used": False,
            }
        mem_set = {tuple(x) for x in memory_candidates}
        mem_rank_scores = _memory_rank_scores(memory_candidates)
        first_hop_counts: Dict[str, int] = defaultdict(int)
        for c in cands:
            if c:
                first_hop_counts[c[0]] += 1
        total_first = sum(first_hop_counts.values())
        for c in cands:
            pool_seen.add(tuple(c))
        uniq = len({tuple(x) for x in generated_all}) / max(1, len(generated_all))
        scores = _score_candidates(
            value_model,
            path_vocab,
            query_vocab,
            query_tokens,
            cands,
            device,
            expected_len=expected_total_len,
            length_penalty_alpha=length_penalty_alpha,
        )
        prefix_penalties = _prefix_step_penalties(
            value_model=value_model,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            query_tokens=query_tokens,
            candidates=cands,
            device=device,
            gamma=prefix_step_penalty_gamma,
        )
        if prefix_step_penalty_alpha > 0.0:
            scores = [s - prefix_step_penalty_alpha * p for s, p in zip(scores, prefix_penalties)]
        if value_score_sign < 0.0:
            scores = [-s for s in scores]
        failure_weights = row.get("_failure_bad_first_action_weights", {})
        failure_penalty = float(row.get("_failure_action_penalty", 0.0))
        if failure_penalty > 0.0 and isinstance(failure_weights, dict):
            scores = [
                s - failure_penalty * float(failure_weights.get(c[0], 0.0) if c else 0.0)
                for s, c in zip(scores, cands)
            ]
        all_candidates.extend(cands)
        idx_sorted = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        best = idx_sorted[0]
        stage2_scores: Dict[int, float] = {i: scores[i] for i in range(len(cands))}
        if rerank_stage1_topk > 0:
            topk = idx_sorted[: min(rerank_stage1_topk, len(idx_sorted))]
            for i in topk:
                key = tuple(cands[i])
                consensus = generated_counts.get(key, 0) / max(1, len(generated_all))
                in_memory = 1.0 if key in mem_set else 0.0
                first_hop_consensus = (
                    first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                )
                len_pen = rerank_stage2_length_penalty_alpha * abs(len(cands[i]) - expected_total_len)
                memory_rank = mem_rank_scores.get(key, 0.0)
                stage2_scores[i] = (
                    scores[i]
                    + rerank_consensus_weight * consensus
                    + rerank_prefix_consensus_weight * first_hop_consensus
                    + rerank_memory_bonus * in_memory
                    + rerank_memory_rank_bonus * memory_rank
                    - len_pen
                )
            best = max(topk, key=lambda i: stage2_scores.get(i, -1e18))
        selection_order = sorted(range(len(cands)), key=lambda i: stage2_scores.get(i, scores[i]), reverse=True)
        selection_order, fusion_used = _maybe_apply_retrieval_fusion(
            idx_sorted=selection_order,
            cands=cands,
            memory_candidates=memory_candidates,
            constraints=row["constraints"],
            stage_scores=stage2_scores,
            enabled=retrieval_fusion_enabled,
            margin=retrieval_fusion_margin,
        )
        retrieval_fusion_used = retrieval_fusion_used or fusion_used
        best = selection_order[0]
        if save_candidate_pool_topk > 0:
            for i in idx_sorted[: min(save_candidate_pool_topk, len(idx_sorted))]:
                candidate_debug.append(
                    {
                        "path": cands[i],
                        "stage1_score": float(scores[i]),
                        "in_memory": bool(tuple(cands[i]) in mem_set),
                        "gen_count": int(generated_counts.get(tuple(cands[i]), 0)),
                        "memory_rank_score": float(mem_rank_scores.get(tuple(cands[i]), 0.0)),
                        "first_hop_consensus": (
                            first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                        ),
                        "prefix_penalty": float(prefix_penalties[i]) if i < len(prefix_penalties) else 0.0,
                    }
                )
        last_plan = cands[best]
        feasible, violations = _is_feasible(last_plan, row["constraints"])
        all_violations.extend(violations)
        executed = last_plan if feasible else last_plan[:max_steps]
        if save_episode_trace:
            episode_trace.append(
                {
                    "step": 1,
                    "remaining_query_tokens": list(query_tokens),
                    "candidate_count": len(cands),
                    "candidate_unique_ratio": uniq,
                    "selected_path": list(last_plan),
                    "selected_score": float(stage2_scores.get(best, scores[best])) if cands else 0.0,
                    "selected_stage1_score": float(scores[best]) if cands else 0.0,
                    "top_stage1_path": list(cands[idx_sorted[0]]) if idx_sorted else [],
                    "top_stage1_score": float(scores[idx_sorted[0]]) if idx_sorted else 0.0,
                    "feasible_prefix": bool(feasible),
                    "feedback_rejections_before_acceptance": 0,
                    "violations": list(violations),
                    "replanned": False,
                }
            )
        out = {
            "planned_path": last_plan,
            "executed_path": executed,
            "candidate_count": len(cands),
            "candidate_unique_ratio": uniq,
            "violations": sorted(set(all_violations)),
            "replanning_steps": 1,
            "candidate_pool_hit": tuple(row["oracle_path"]) in pool_seen,
            "candidate_pool_size": len(pool_seen),
            "candidate_pool_top": candidate_debug,
            "retrieval_fusion_used": retrieval_fusion_used,
        }
        if save_episode_trace:
            out["episode_trace"] = episode_trace
            out["episode_trace_len"] = len(episode_trace)
        return out

    uniq_vals = []
    for _step in range(max_steps):
        # Receding-horizon planning keeps the original task context and commits
        # only to the next action of a full candidate plan.
        rem_q = query_tokens
        expected_remaining = expected_total_len
        generated_all: List[List[str]] = []
        if not disable_generated_candidates:
            jitter_list = [candidate_latent_jitter_std] + [x for x in candidate_multi_jitter_stds if x != candidate_latent_jitter_std]
            for jit in jitter_list:
                c_gen, _ = _generate_candidates(
                    planner,
                    autoencoder,
                    value_model,
                    path_vocab,
                    query_vocab,
                    rem_q,
                    num_candidates,
                    diffusion_steps,
                    max_steps,
                    device,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    prediction_target=prediction_target,
                    planner_type=planner_type,
                    candidate_latent_jitter_std=jit,
                    value_guided_sampling=value_guided_sampling,
                    value_guidance_scale=value_guidance_scale,
                    value_guidance_interval=value_guidance_interval,
                    value_guidance_temperature=value_guidance_temperature,
                    expected_len=expected_remaining,
                    length_penalty_alpha=length_penalty_alpha,
                )
                generated_all.extend(c_gen)
        generated_unique, generated_counts = _dedupe_with_count(generated_all)
        cands = _merge_candidates(generated_unique, memory_candidates)
        mem_set = {tuple(x) for x in memory_candidates}
        mem_rank_scores = _memory_rank_scores(memory_candidates)
        first_hop_counts: Dict[str, int] = defaultdict(int)
        for c in cands:
            if c:
                first_hop_counts[c[0]] += 1
        total_first = sum(first_hop_counts.values())
        for c in cands:
            pool_seen.add(tuple(c))
        uniq = len({tuple(x) for x in generated_all}) / max(1, len(generated_all))
        scores = _score_candidates(
            value_model,
            path_vocab,
            query_vocab,
            rem_q,
            cands,
            device,
            expected_len=expected_remaining,
            length_penalty_alpha=length_penalty_alpha,
        )
        prefix_penalties = _prefix_step_penalties(
            value_model=value_model,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            query_tokens=rem_q,
            candidates=cands,
            device=device,
            gamma=prefix_step_penalty_gamma,
        )
        if prefix_step_penalty_alpha > 0.0:
            scores = [s - prefix_step_penalty_alpha * p for s, p in zip(scores, prefix_penalties)]
        if value_score_sign < 0.0:
            scores = [-s for s in scores]
        failure_weights = row.get("_failure_bad_first_action_weights", {})
        failure_penalty = float(row.get("_failure_action_penalty", 0.0))
        if failure_penalty > 0.0 and isinstance(failure_weights, dict):
            scores = [
                s - failure_penalty * float(failure_weights.get(c[0], 0.0) if c else 0.0)
                for s, c in zip(scores, cands)
            ]
        all_candidates.extend(cands)
        uniq_vals.append(uniq)
        idx_sorted = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        stage2_scores: Dict[int, float] = {i: scores[i] for i in range(len(cands))}
        if rerank_stage1_topk > 0:
            topk = idx_sorted[: min(rerank_stage1_topk, len(idx_sorted))]
            for i in topk:
                key = tuple(cands[i])
                consensus = generated_counts.get(key, 0) / max(1, len(generated_all))
                in_memory = 1.0 if key in mem_set else 0.0
                first_hop_consensus = (
                    first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                )
                len_pen = rerank_stage2_length_penalty_alpha * abs(len(cands[i]) - expected_remaining)
                memory_rank = mem_rank_scores.get(key, 0.0)
                stage2_scores[i] = (
                    scores[i]
                    + rerank_consensus_weight * consensus
                    + rerank_prefix_consensus_weight * first_hop_consensus
                    + rerank_memory_bonus * in_memory
                    + rerank_memory_rank_bonus * memory_rank
                    - len_pen
                )
            idx_sorted = sorted(topk, key=lambda i: stage2_scores.get(i, -1e18), reverse=True)
        idx_sorted, fusion_used = _maybe_apply_retrieval_fusion(
            idx_sorted=idx_sorted,
            cands=cands,
            memory_candidates=memory_candidates,
            constraints=row["constraints"],
            stage_scores=stage2_scores,
            enabled=retrieval_fusion_enabled,
            margin=retrieval_fusion_margin,
            executed_prefix=executed,
        )
        retrieval_fusion_used = retrieval_fusion_used or fusion_used
        if save_candidate_pool_topk > 0:
            for i in idx_sorted[: min(save_candidate_pool_topk, len(idx_sorted))]:
                candidate_debug.append(
                    {
                        "path": cands[i],
                        "stage1_score": float(scores[i]),
                        "in_memory": bool(tuple(cands[i]) in mem_set),
                        "gen_count": int(generated_counts.get(tuple(cands[i]), 0)),
                        "memory_rank_score": float(mem_rank_scores.get(tuple(cands[i]), 0.0)),
                        "first_hop_consensus": (
                            first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                        ),
                        "prefix_penalty": float(prefix_penalties[i]) if i < len(prefix_penalties) else 0.0,
                    }
                )
        picked = None
        picked_next_action = ""
        selected_idx = -1
        selected_violations: List[str] = []
        rejected_before_accept = 0
        prefix_pos = len(executed)
        selection_passes = (True, False)
        for require_prefix_match in selection_passes:
            for i in idx_sorted:
                cand = cands[i]
                if not cand:
                    continue
                if require_prefix_match and cand[:prefix_pos] != executed:
                    rejected_before_accept += 1
                    continue
                if len(cand) <= prefix_pos:
                    rejected_before_accept += 1
                    continue
                next_action = cand[prefix_pos] if cand[:prefix_pos] == executed else cand[0]
                feasible, violations = _is_feasible(executed + [next_action], row["constraints"])
                if feasible:
                    selected_idx = i
                    selected_violations = list(violations)
                    picked = cand
                    picked_next_action = next_action
                    break
                all_violations.extend(violations)
                rejected_before_accept += 1
            if picked is not None:
                break
        if picked is None:
            break
        last_plan = list(executed) + [picked_next_action]
        executed.append(picked_next_action)
        if save_episode_trace:
            episode_trace.append(
                {
                    "step": len(executed),
                    "remaining_query_tokens": list(rem_q),
                    "candidate_count": len(cands),
                    "candidate_unique_ratio": uniq,
                    "selected_path": list(picked),
                    "selected_next_action": picked_next_action,
                    "selected_score": float(stage2_scores.get(selected_idx, scores[selected_idx])) if selected_idx >= 0 else 0.0,
                    "selected_stage1_score": float(scores[selected_idx]) if selected_idx >= 0 else 0.0,
                    "top_stage1_path": list(cands[idx_sorted[0]]) if idx_sorted else [],
                    "top_stage1_score": float(scores[idx_sorted[0]]) if idx_sorted else 0.0,
                    "feasible_prefix": True,
                    "prefix_violations": selected_violations,
                    "feedback_rejections_before_acceptance": rejected_before_accept,
                    "replanned": True,
                }
            )
        if len(executed) >= len(row["oracle_path"]):
            break
    out = {
        "planned_path": executed if executed else (last_plan if last_plan else executed),
        "executed_path": executed,
        "candidate_count": len(all_candidates),
        "candidate_unique_ratio": sum(uniq_vals) / max(1, len(uniq_vals)),
        "violations": sorted(set(all_violations)),
        "replanning_steps": max(1, len(executed)),
        "candidate_pool_hit": tuple(row["oracle_path"]) in pool_seen,
        "candidate_pool_size": len(pool_seen),
        "candidate_pool_top": candidate_debug,
        "retrieval_fusion_used": retrieval_fusion_used,
    }
    if save_episode_trace:
        out["episode_trace"] = episode_trace
        out["episode_trace_len"] = len(episode_trace)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_torch_kgqa.yaml")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, default="")
    parser.add_argument("--out", type=str, default="results/main_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["test_path"])
    include_datasets = [str(x).lower() for x in cfg.get("include_datasets", [])]
    if include_datasets:
        rows = [r for r in rows if str(r.get("dataset", "")).lower() in set(include_datasets)]
        print(f"[eval] include_datasets={include_datasets} kept={len(rows)}")
    include_task_names = [str(x).lower() for x in cfg.get("include_task_names", [])]
    if include_task_names:
        rows = [
            r
            for r in rows
            if str(r.get("meta", {}).get("task_name", "")).lower() in set(include_task_names)
            or str(r.get("task_id", "")).lower() in set(include_task_names)
        ]
        print(f"[eval] include_task_names={include_task_names} kept={len(rows)}")
    max_tasks = int(cfg.get("max_tasks", 0))
    if max_tasks > 0:
        rows = rows[:max_tasks]
        print(f"[eval] max_tasks={max_tasks} kept={len(rows)}")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")

    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    use_value_model = bool(cfg.get("use_value_model", True))
    value_ckpt = None
    if use_value_model:
        if not args.value_ckpt:
            raise ValueError("use_value_model=true requires --value_ckpt")
        value_ckpt = torch.load(args.value_ckpt, map_location="cpu")

    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])

    ae_cfg = ae_ckpt["model_config"]
    autoencoder = PathAutoencoder(
        vocab_size=ae_cfg["vocab_size"],
        emb_dim=ae_cfg["emb_dim"],
        hid_dim=ae_cfg["hid_dim"],
        latent_dim=ae_cfg["latent_dim"],
        max_path_len=ae_cfg["max_path_len"],
        pad_id=ae_cfg["pad_id"],
        latent_noise_std=float(ae_cfg.get("latent_noise_std", 0.0)),
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["model_state"])
    autoencoder.eval()

    pl_cfg = planner_ckpt["model_config"]
    planner_type = str(pl_cfg.get("planner_type", "diffusion")).lower()
    if planner_type == "mlp":
        planner = MLPPlanner(
            latent_dim=pl_cfg["latent_dim"],
            q_vocab_size=pl_cfg["q_vocab_size"],
            q_emb_dim=pl_cfg["q_emb_dim"],
            q_pad_id=pl_cfg["q_pad_id"],
            hidden_dim=int(pl_cfg.get("hidden_dim", 256)),
        ).to(device)
    else:
        planner = DiffusionPlanner(
            latent_dim=pl_cfg["latent_dim"],
            q_vocab_size=pl_cfg["q_vocab_size"],
            q_emb_dim=pl_cfg["q_emb_dim"],
            q_pad_id=pl_cfg["q_pad_id"],
            time_dim=pl_cfg["time_dim"],
        ).to(device)
    planner.load_state_dict(planner_ckpt["model_state"])
    planner.eval()
    prediction_target = str(pl_cfg.get("prediction_target", "z0")).lower()
    if planner_type == "diffusion" and prediction_target not in {"z0", "eps"}:
        raise ValueError(f"Unsupported planner prediction_target in ckpt: {prediction_target}")
    print(f"[eval] planner_type={planner_type} prediction_target={prediction_target}")

    latent_mean = None
    latent_std = None
    latent_norm = planner_ckpt.get("latent_norm")
    if isinstance(latent_norm, dict) and bool(latent_norm.get("enabled", False)):
        latent_mean = torch.tensor(latent_norm["mean"], dtype=torch.float32, device=device).view(1, -1)
        latent_std = torch.tensor(latent_norm["std"], dtype=torch.float32, device=device).view(1, -1)
        print(
            f"[eval] latent denorm enabled, mean={latent_mean.mean().item():.4f}, "
            f"std={latent_std.mean().item():.4f}"
        )

    value_model = None
    if use_value_model and value_ckpt is not None:
        v_cfg = value_ckpt["model_config"]
        value_model = ValueRanker(
            q_vocab_size=v_cfg["q_vocab_size"],
            p_vocab_size=v_cfg["p_vocab_size"],
            emb_dim=v_cfg["emb_dim"],
            q_pad_id=v_cfg["q_pad_id"],
            p_pad_id=v_cfg["p_pad_id"],
            architecture=str(v_cfg.get("architecture", "legacy")),
            hidden_dim=int(v_cfg.get("hidden_dim", 256)),
            dropout=float(v_cfg.get("dropout", 0.1)),
        ).to(device)
        value_model.load_state_dict(value_ckpt["model_state"])
        value_model.eval()

    use_memory_retrieval = bool(cfg.get("use_memory_retrieval", False))
    memory_prefilter_feasible = bool(cfg.get("memory_prefilter_feasible", True))
    memory_top_k = int(cfg.get("memory_top_k", 8))
    length_penalty_alpha = float(cfg.get("length_penalty_alpha", 0.0))
    expected_length_bucket_size = int(cfg.get("expected_length_bucket_size", 4))
    use_expected_length_prior = bool(cfg.get("use_expected_length_prior", length_penalty_alpha > 0.0))
    global_expected_len = 2.0
    bucket_expected_len: Dict[int, float] = {}
    token_to_ids: Dict[str, List[int]] = {}
    path_bank: List[List[str]] = []
    train_rows = None
    if use_memory_retrieval or use_expected_length_prior:
        train_path = cfg.get("memory_train_path") or cfg.get("train_path")
        if not train_path:
            raise ValueError("use_memory_retrieval=true requires memory_train_path or train_path in config.")
        train_rows = load_jsonl(train_path)
    if use_memory_retrieval:
        assert train_rows is not None
        token_to_ids, path_bank = _build_memory_index(
            train_rows=train_rows,
            max_postings_per_token=int(cfg.get("memory_max_postings_per_token", 1200)),
        )
        print(f"[memory] enabled, train_rows={len(train_rows)} indexed_paths={len(path_bank)}")
    if use_expected_length_prior:
        assert train_rows is not None
        global_expected_len, bucket_expected_len = _build_expected_len_prior(train_rows, expected_length_bucket_size)
        print(
            f"[eval] expected_len_prior enabled, global={global_expected_len:.3f}, "
            f"buckets={len(bucket_expected_len)}, alpha={length_penalty_alpha:.3f}"
        )

    method = "diplan_torch_mem" if use_memory_retrieval else "diplan_torch"
    candidate_latent_jitter_std = float(cfg.get("candidate_latent_jitter_std", 0.0))
    candidate_multi_jitter_stds = [float(x) for x in cfg.get("candidate_multi_jitter_stds", [])]
    value_score_sign = float(cfg.get("value_score_sign", 1.0))
    if candidate_latent_jitter_std > 0:
        print(f"[eval] candidate_latent_jitter_std={candidate_latent_jitter_std}")
    if candidate_multi_jitter_stds:
        print(f"[eval] candidate_multi_jitter_stds={candidate_multi_jitter_stds}")
    rerank_stage1_topk = int(cfg.get("rerank_stage1_topk", 0))
    rerank_consensus_weight = float(cfg.get("rerank_consensus_weight", 0.0))
    rerank_memory_bonus = float(cfg.get("rerank_memory_bonus", 0.0))
    rerank_memory_rank_bonus = float(cfg.get("rerank_memory_rank_bonus", 0.0))
    rerank_stage2_length_penalty_alpha = float(cfg.get("rerank_stage2_length_penalty_alpha", 0.0))
    rerank_prefix_consensus_weight = float(cfg.get("rerank_prefix_consensus_weight", 0.0))
    prefix_step_penalty_alpha = float(cfg.get("prefix_step_penalty_alpha", 0.0))
    prefix_step_penalty_gamma = float(cfg.get("prefix_step_penalty_gamma", 0.85))
    value_guided_sampling = bool(cfg.get("value_guided_sampling", False))
    value_guidance_scale = float(cfg.get("value_guidance_scale", 0.0))
    value_guidance_interval = int(cfg.get("value_guidance_interval", 4))
    value_guidance_temperature = float(cfg.get("value_guidance_temperature", 0.5))
    retrieval_fusion_enabled = bool(cfg.get("retrieval_fusion_enabled", False))
    retrieval_fusion_margin = float(cfg.get("retrieval_fusion_margin", 0.0))
    disable_generated_candidates = bool(cfg.get("disable_generated_candidates", False))
    save_candidate_pool_topk = int(cfg.get("save_candidate_pool_topk", 0))
    save_episode_trace = bool(cfg.get("save_episode_trace", False))
    emit_structured_plan = bool(cfg.get("emit_structured_plan", False))
    emit_tool_calls = bool(cfg.get("emit_tool_calls", False))
    failure_memory = None
    failure_memory_path = str(cfg.get("failure_memory_path", "")).strip()
    failure_action_penalty = float(cfg.get("failure_action_penalty", 0.0))
    failure_memory_top_k = int(cfg.get("failure_memory_top_k", 8))
    if failure_memory_path:
        failure_rows = load_jsonl(failure_memory_path)
        failure_memory = build_failure_memory(
            failure_rows,
            max_postings_per_token=int(cfg.get("failure_memory_max_postings_per_token", 1200)),
        )
        print(f"[failure-memory] enabled, failures={len(failure_memory.rows)}, penalty={failure_action_penalty:.3f}")
    adaptive_prefix_alpha_enabled = bool(cfg.get("adaptive_prefix_alpha_enabled", False))
    if adaptive_prefix_alpha_enabled:
        print(
            "[eval] adaptive_prefix_alpha enabled, "
            f"mode={str(cfg.get('adaptive_prefix_alpha_mode', 'dataset')).lower()}, "
            f"default_alpha={prefix_step_penalty_alpha:.3f}"
        )
    if value_guided_sampling:
        print(
            "[eval] value_guided_sampling enabled, "
            f"scale={value_guidance_scale:.3f}, interval={value_guidance_interval}, "
            f"temperature={value_guidance_temperature:.3f}"
        )
        if planner_type != "diffusion":
            print("[eval] value_guided_sampling is only active for diffusion planners; current planner is not diffusion.")
        if value_model is None:
            print("[eval] value_guided_sampling requested but value_model is disabled or missing.")
    if retrieval_fusion_enabled:
        print(f"[eval] retrieval_fusion enabled, margin={retrieval_fusion_margin:.3f}")
    if disable_generated_candidates:
        print("[eval] generated candidates disabled; ranking memory/retrieval candidates only.")
    if value_score_sign < 0.0:
        print("[eval] value_score_sign=-1 enabled; lower raw value logits are treated as better.")
    records = []
    memory_prefilter_dropped = 0
    alpha_bucket_counter: Dict[str, int] = defaultdict(int)
    for row in rows:
        row_for_pred = dict(row)
        failure_hits = []
        failure_bad_first_action_weights = {}
        if failure_memory is not None:
            failure_hits = failure_memory.retrieve(row.get("query_tokens", []), top_k=failure_memory_top_k)
            failure_bad_first_action_weights = failure_memory.bad_first_action_weights(failure_hits)
            row_for_pred["_failure_bad_first_action_weights"] = failure_bad_first_action_weights
            row_for_pred["_failure_action_penalty"] = failure_action_penalty
        mem_cands = (
            _retrieve_memory_candidates(row["query_tokens"], token_to_ids, path_bank, top_k=memory_top_k)
            if use_memory_retrieval
            else []
        )
        if use_memory_retrieval and memory_prefilter_feasible:
            mem_cands, dropped = _filter_feasible_candidates(mem_cands, row["constraints"])
            memory_prefilter_dropped += dropped
        row_prefix_alpha, alpha_source = _resolve_prefix_step_alpha(
            row=row,
            cfg=cfg,
            default_alpha=prefix_step_penalty_alpha,
        )
        alpha_bucket_counter[f"{alpha_source}:{row_prefix_alpha:.3f}"] += 1
        pred = _predict_diplan(
            row=row_for_pred,
            planner=planner,
            autoencoder=autoencoder,
            value_model=value_model,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            device=device,
            num_candidates=int(cfg.get("num_candidates", 12)),
            diffusion_steps=int(pl_cfg.get("diffusion_steps", 20)),
            receding_horizon=bool(cfg.get("receding_horizon", True)),
            memory_candidates=mem_cands,
            latent_mean=latent_mean,
            latent_std=latent_std,
            prediction_target=prediction_target,
            planner_type=planner_type,
            candidate_latent_jitter_std=candidate_latent_jitter_std,
            candidate_multi_jitter_stds=candidate_multi_jitter_stds,
            expected_total_len=_expected_len_from_prior(
                row.get("query_tokens", []),
                expected_length_bucket_size,
                global_expected_len,
                bucket_expected_len,
            ),
            length_penalty_alpha=length_penalty_alpha,
            value_score_sign=value_score_sign,
            rerank_stage1_topk=rerank_stage1_topk,
            rerank_consensus_weight=rerank_consensus_weight,
            rerank_memory_bonus=rerank_memory_bonus,
            rerank_memory_rank_bonus=rerank_memory_rank_bonus,
            rerank_stage2_length_penalty_alpha=rerank_stage2_length_penalty_alpha,
            rerank_prefix_consensus_weight=rerank_prefix_consensus_weight,
            prefix_step_penalty_alpha=row_prefix_alpha,
            prefix_step_penalty_gamma=prefix_step_penalty_gamma,
            value_guided_sampling=value_guided_sampling,
            value_guidance_scale=value_guidance_scale,
            value_guidance_interval=value_guidance_interval,
            value_guidance_temperature=value_guidance_temperature,
            retrieval_fusion_enabled=retrieval_fusion_enabled,
            retrieval_fusion_margin=retrieval_fusion_margin,
            disable_generated_candidates=disable_generated_candidates,
            save_candidate_pool_topk=save_candidate_pool_topk,
            save_episode_trace=save_episode_trace,
        )
        executed = pred["executed_path"]
        feasible, violations = _is_feasible(executed, row["constraints"])
        oracle_in_candidate_pool = bool(pred.get("candidate_pool_hit", False))
        success = executed == row["oracle_path"]
        candidate_count = int(pred["candidate_count"])
        token_cost = (
            len(executed) + candidate_count
            if bool(cfg.get("receding_horizon", True))
            else len(executed) * (1 + candidate_count)
        )
        rec = {
            "task_id": row["task_id"],
            "dataset": row["dataset"],
            "query": row.get("question", " ".join(row.get("query_tokens", []))),
            "query_tokens": row.get("query_tokens", []),
            "method": method,
            "oracle_path": row["oracle_path"],
            "planned_path": pred["planned_path"],
            "executed_path": executed,
            "success": success,
            "first_error_step": first_error_step(executed, row["oracle_path"]),
            "recovery_at_error": recovery_at_error(executed, row["oracle_path"]),
            "trap_at_1": trap_at_1(executed, row["trap_path"]),
            "feasible": feasible,
            "violations": sorted(set(violations + pred["violations"])),
            "plan_execution_consistency": plan_execution_consistency(pred["planned_path"], executed),
            "token_cost": token_cost,
            "latency_cost": candidate_count * 0.02,
            "diversity_coverage": pred["candidate_unique_ratio"],
            "replanning_steps": pred["replanning_steps"],
            "episode_trace_len": int(pred.get("episode_trace_len", 0)),
            "oracle_in_candidate_pool": oracle_in_candidate_pool,
            "candidate_pool_size": int(pred.get("candidate_pool_size", 0)),
            "ranking_error": bool((not success) and oracle_in_candidate_pool),
            "retrieval_fusion_used": bool(pred.get("retrieval_fusion_used", False)),
            "prefix_step_penalty_alpha": float(row_prefix_alpha),
            "prefix_step_penalty_alpha_source": alpha_source,
        }
        if failure_memory is not None:
            rec["failure_memory_hits"] = len(failure_hits)
            rec["failure_bad_first_action_weights"] = failure_bad_first_action_weights
        if emit_structured_plan:
            rec["structured_planned_path"] = path_to_structured_plan(pred["planned_path"])
            rec["structured_executed_path"] = path_to_structured_plan(executed)
        if emit_tool_calls:
            rec["tool_calls"] = path_to_tool_calls(executed)
        if save_candidate_pool_topk > 0:
            rec["candidate_pool_top"] = pred.get("candidate_pool_top", [])
        if save_episode_trace:
            rec["episode_trace"] = pred.get("episode_trace", [])
        records.append(rec)

    summary = {method: aggregate_method_metrics(records)}
    summary[method].update(
        {
            "candidate_pool_hit_rate": sum(1.0 if r["oracle_in_candidate_pool"] else 0.0 for r in records)
            / max(1, len(records)),
            "ranking_error_rate": sum(1.0 if r["ranking_error"] else 0.0 for r in records) / max(1, len(records)),
            "candidate_pool_avg_size": sum(float(r["candidate_pool_size"]) for r in records) / max(1, len(records)),
            "retrieval_fusion_rate": sum(1.0 if r.get("retrieval_fusion_used") else 0.0 for r in records)
            / max(1, len(records)),
            "conditional_success_given_pool_hit": (
                sum(1.0 if r["success"] else 0.0 for r in records if r["oracle_in_candidate_pool"])
                / max(1, sum(1 for r in records if r["oracle_in_candidate_pool"]))
            ),
        }
    )
    ensure_dir(args.out)
    dump_jsonl(str(Path(args.out) / "predictions.jsonl"), records)
    dump_json(str(Path(args.out) / "summary_metrics.json"), summary)
    _save_summary_csv(str(Path(args.out) / "summary_table.csv"), summary)
    by_dataset = {}
    for ds in sorted({r.get("dataset", "unknown") for r in records}):
        ds_recs = [r for r in records if r.get("dataset", "unknown") == ds]
        by_dataset[ds] = aggregate_method_metrics(ds_recs)
        by_dataset[ds].update(
            {
                "candidate_pool_hit_rate": sum(1.0 if r["oracle_in_candidate_pool"] else 0.0 for r in ds_recs)
                / max(1, len(ds_recs)),
                "ranking_error_rate": sum(1.0 if r["ranking_error"] else 0.0 for r in ds_recs) / max(1, len(ds_recs)),
                "candidate_pool_avg_size": sum(float(r["candidate_pool_size"]) for r in ds_recs) / max(1, len(ds_recs)),
                "retrieval_fusion_rate": sum(1.0 if r.get("retrieval_fusion_used") else 0.0 for r in ds_recs)
                / max(1, len(ds_recs)),
            }
        )
    dump_json(str(Path(args.out) / "summary_by_dataset.json"), by_dataset)
    print(f"Evaluated {len(rows)} tasks with {method}.")
    if use_memory_retrieval and memory_prefilter_feasible:
        print(f"[memory] prefilter dropped={memory_prefilter_dropped}")
    if adaptive_prefix_alpha_enabled:
        print(f"[eval] adaptive alpha bucket counts={dict(alpha_bucket_counter)}")
    print(summary[method])


if __name__ == "__main__":
    main()
