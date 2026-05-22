import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_config, load_jsonl
from src.diplan.metrics import (
    aggregate_method_metrics,
    first_error_step,
    plan_execution_consistency,
    recovery_at_error,
    trap_at_1,
)
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
    return len(violations) == 0, sorted(set(violations))


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
def _generate_candidates(
    planner: DiffusionPlanner | MLPPlanner,
    autoencoder: PathAutoencoder,
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
) -> Tuple[List[List[str]], float]:
    q_ids = query_vocab.encode(query_tokens, add_bos_eos=False, max_len=24)
    if not q_ids:
        q_ids = [query_vocab.stoi[PAD]]
    q_batch = torch.tensor([q_ids for _ in range(num_candidates)], dtype=torch.long, device=device)
    if planner_type == "mlp":
        z = planner(q_batch)
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
    rerank_stage1_topk: int,
    rerank_consensus_weight: float,
    rerank_memory_bonus: float,
    rerank_stage2_length_penalty_alpha: float,
    rerank_prefix_consensus_weight: float,
    prefix_step_penalty_alpha: float,
    prefix_step_penalty_gamma: float,
    save_candidate_pool_topk: int,
) -> Dict:
    max_steps = int(row["constraints"].get("max_steps", 8))
    query_tokens = row["query_tokens"]
    all_candidates = []
    pool_seen = set()
    all_violations: List[str] = []
    candidate_debug: List[Dict] = []
    executed: List[str] = []
    last_plan: List[str] = []
    if not receding_horizon:
        jitter_list = [candidate_latent_jitter_std] + [x for x in candidate_multi_jitter_stds if x != candidate_latent_jitter_std]
        generated_all: List[List[str]] = []
        for jit in jitter_list:
            c_gen, _ = _generate_candidates(
                planner,
                autoencoder,
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
            )
            generated_all.extend(c_gen)
        generated_unique, generated_counts = _dedupe_with_count(generated_all)
        cands = _merge_candidates(generated_unique, memory_candidates)
        mem_set = {tuple(x) for x in memory_candidates}
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
        all_candidates.extend(cands)
        idx_sorted = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        best = idx_sorted[0]
        if rerank_stage1_topk > 0:
            topk = idx_sorted[: min(rerank_stage1_topk, len(idx_sorted))]
            stage2_scores: Dict[int, float] = {}
            for i in topk:
                key = tuple(cands[i])
                consensus = generated_counts.get(key, 0) / max(1, len(generated_all))
                in_memory = 1.0 if key in mem_set else 0.0
                first_hop_consensus = (
                    first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                )
                len_pen = rerank_stage2_length_penalty_alpha * abs(len(cands[i]) - expected_total_len)
                stage2_scores[i] = (
                    scores[i]
                    + rerank_consensus_weight * consensus
                    + rerank_prefix_consensus_weight * first_hop_consensus
                    + rerank_memory_bonus * in_memory
                    - len_pen
                )
            best = max(topk, key=lambda i: stage2_scores.get(i, -1e18))
        if save_candidate_pool_topk > 0:
            for i in idx_sorted[: min(save_candidate_pool_topk, len(idx_sorted))]:
                candidate_debug.append(
                    {
                        "path": cands[i],
                        "stage1_score": float(scores[i]),
                        "in_memory": bool(tuple(cands[i]) in mem_set),
                        "gen_count": int(generated_counts.get(tuple(cands[i]), 0)),
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
        return {
            "planned_path": last_plan,
            "executed_path": executed,
            "candidate_count": len(cands),
            "candidate_unique_ratio": uniq,
            "violations": sorted(set(all_violations)),
            "replanning_steps": 1,
            "candidate_pool_hit": tuple(row["oracle_path"]) in pool_seen,
            "candidate_pool_size": len(pool_seen),
            "candidate_pool_top": candidate_debug,
        }

    uniq_vals = []
    for _step in range(max_steps):
        rem_q = query_tokens[len(executed) :] if len(executed) < len(query_tokens) else query_tokens
        jitter_list = [candidate_latent_jitter_std] + [x for x in candidate_multi_jitter_stds if x != candidate_latent_jitter_std]
        generated_all: List[List[str]] = []
        for jit in jitter_list:
            c_gen, _ = _generate_candidates(
                planner,
                autoencoder,
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
            )
            generated_all.extend(c_gen)
        generated_unique, generated_counts = _dedupe_with_count(generated_all)
        cands = _merge_candidates(generated_unique, memory_candidates)
        mem_set = {tuple(x) for x in memory_candidates}
        first_hop_counts: Dict[str, int] = defaultdict(int)
        for c in cands:
            if c:
                first_hop_counts[c[0]] += 1
        total_first = sum(first_hop_counts.values())
        for c in cands:
            pool_seen.add(tuple(c))
        uniq = len({tuple(x) for x in generated_all}) / max(1, len(generated_all))
        expected_remaining = max(1.0, expected_total_len - len(executed))
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
        all_candidates.extend(cands)
        uniq_vals.append(uniq)
        idx_sorted = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        if rerank_stage1_topk > 0:
            topk = idx_sorted[: min(rerank_stage1_topk, len(idx_sorted))]
            stage2_scores: Dict[int, float] = {}
            for i in topk:
                key = tuple(cands[i])
                consensus = generated_counts.get(key, 0) / max(1, len(generated_all))
                in_memory = 1.0 if key in mem_set else 0.0
                first_hop_consensus = (
                    first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                )
                len_pen = rerank_stage2_length_penalty_alpha * abs(len(cands[i]) - expected_remaining)
                stage2_scores[i] = (
                    scores[i]
                    + rerank_consensus_weight * consensus
                    + rerank_prefix_consensus_weight * first_hop_consensus
                    + rerank_memory_bonus * in_memory
                    - len_pen
                )
            idx_sorted = sorted(topk, key=lambda i: stage2_scores.get(i, -1e18), reverse=True)
        if save_candidate_pool_topk > 0:
            for i in idx_sorted[: min(save_candidate_pool_topk, len(idx_sorted))]:
                candidate_debug.append(
                    {
                        "path": cands[i],
                        "stage1_score": float(scores[i]),
                        "in_memory": bool(tuple(cands[i]) in mem_set),
                        "gen_count": int(generated_counts.get(tuple(cands[i]), 0)),
                        "first_hop_consensus": (
                            first_hop_counts.get(cands[i][0], 0) / max(1, total_first) if cands[i] else 0.0
                        ),
                        "prefix_penalty": float(prefix_penalties[i]) if i < len(prefix_penalties) else 0.0,
                    }
                )
        picked = None
        for i in idx_sorted:
            feasible, violations = _is_feasible(executed + cands[i][:1], row["constraints"])
            if feasible and cands[i]:
                picked = cands[i]
                break
            all_violations.extend(violations)
        if picked is None:
            break
        last_plan = picked
        executed.append(picked[0])
        if len(executed) >= len(row["oracle_path"]):
            break
    return {
        "planned_path": last_plan if last_plan else executed,
        "executed_path": executed,
        "candidate_count": len(all_candidates),
        "candidate_unique_ratio": sum(uniq_vals) / max(1, len(uniq_vals)),
        "violations": sorted(set(all_violations)),
        "replanning_steps": max(1, len(executed)),
        "candidate_pool_hit": tuple(row["oracle_path"]) in pool_seen,
        "candidate_pool_size": len(pool_seen),
        "candidate_pool_top": candidate_debug,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_torch_kgqa.yaml")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/main_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["test_path"])
    include_datasets = [str(x).lower() for x in cfg.get("include_datasets", [])]
    if include_datasets:
        rows = [r for r in rows if str(r.get("dataset", "")).lower() in set(include_datasets)]
        print(f"[eval] include_datasets={include_datasets} kept={len(rows)}")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")

    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    use_value_model = bool(cfg.get("use_value_model", True))
    value_ckpt = torch.load(args.value_ckpt, map_location="cpu") if use_value_model else None

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
    if candidate_latent_jitter_std > 0:
        print(f"[eval] candidate_latent_jitter_std={candidate_latent_jitter_std}")
    if candidate_multi_jitter_stds:
        print(f"[eval] candidate_multi_jitter_stds={candidate_multi_jitter_stds}")
    rerank_stage1_topk = int(cfg.get("rerank_stage1_topk", 0))
    rerank_consensus_weight = float(cfg.get("rerank_consensus_weight", 0.0))
    rerank_memory_bonus = float(cfg.get("rerank_memory_bonus", 0.0))
    rerank_stage2_length_penalty_alpha = float(cfg.get("rerank_stage2_length_penalty_alpha", 0.0))
    rerank_prefix_consensus_weight = float(cfg.get("rerank_prefix_consensus_weight", 0.0))
    prefix_step_penalty_alpha = float(cfg.get("prefix_step_penalty_alpha", 0.0))
    prefix_step_penalty_gamma = float(cfg.get("prefix_step_penalty_gamma", 0.85))
    save_candidate_pool_topk = int(cfg.get("save_candidate_pool_topk", 0))
    adaptive_prefix_alpha_enabled = bool(cfg.get("adaptive_prefix_alpha_enabled", False))
    if adaptive_prefix_alpha_enabled:
        print(
            "[eval] adaptive_prefix_alpha enabled, "
            f"mode={str(cfg.get('adaptive_prefix_alpha_mode', 'dataset')).lower()}, "
            f"default_alpha={prefix_step_penalty_alpha:.3f}"
        )
    records = []
    memory_prefilter_dropped = 0
    alpha_bucket_counter: Dict[str, int] = defaultdict(int)
    for row in rows:
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
            row=row,
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
            rerank_stage1_topk=rerank_stage1_topk,
            rerank_consensus_weight=rerank_consensus_weight,
            rerank_memory_bonus=rerank_memory_bonus,
            rerank_stage2_length_penalty_alpha=rerank_stage2_length_penalty_alpha,
            rerank_prefix_consensus_weight=rerank_prefix_consensus_weight,
            prefix_step_penalty_alpha=row_prefix_alpha,
            prefix_step_penalty_gamma=prefix_step_penalty_gamma,
            save_candidate_pool_topk=save_candidate_pool_topk,
        )
        executed = pred["executed_path"]
        feasible, violations = _is_feasible(executed, row["constraints"])
        oracle_in_candidate_pool = bool(pred.get("candidate_pool_hit", False))
        success = executed == row["oracle_path"]
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
            "token_cost": len(executed) * (1 + pred["candidate_count"]),
            "latency_cost": pred["candidate_count"] * 0.02,
            "diversity_coverage": pred["candidate_unique_ratio"],
            "replanning_steps": pred["replanning_steps"],
            "oracle_in_candidate_pool": oracle_in_candidate_pool,
            "candidate_pool_size": int(pred.get("candidate_pool_size", 0)),
            "ranking_error": bool((not success) and oracle_in_candidate_pool),
            "prefix_step_penalty_alpha": float(row_prefix_alpha),
            "prefix_step_penalty_alpha_source": alpha_source,
        }
        if save_candidate_pool_topk > 0:
            rec["candidate_pool_top"] = pred.get("candidate_pool_top", [])
        records.append(rec)

    summary = {method: aggregate_method_metrics(records)}
    summary[method].update(
        {
            "candidate_pool_hit_rate": sum(1.0 if r["oracle_in_candidate_pool"] else 0.0 for r in records)
            / max(1, len(records)),
            "ranking_error_rate": sum(1.0 if r["ranking_error"] else 0.0 for r in records) / max(1, len(records)),
            "candidate_pool_avg_size": sum(float(r["candidate_pool_size"]) for r in records) / max(1, len(records)),
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
