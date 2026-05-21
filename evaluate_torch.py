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
    fields = [
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
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m, met in summary.items():
            row = {"method": m}
            row.update(met)
            w.writerow(row)


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
) -> List[float]:
    if value_model is None:
        q_len = max(1, len(query_tokens))
        return [-(abs(len(c) - q_len)) for c in candidates]
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
    return logits.detach().cpu().tolist()


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
) -> Dict:
    max_steps = int(row["constraints"].get("max_steps", 8))
    query_tokens = row["query_tokens"]
    all_candidates = []
    all_violations: List[str] = []
    executed: List[str] = []
    last_plan: List[str] = []
    if not receding_horizon:
        cands, uniq = _generate_candidates(
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
            candidate_latent_jitter_std=candidate_latent_jitter_std,
        )
        cands = _merge_candidates(cands, memory_candidates)
        uniq = len({tuple(x) for x in cands}) / max(1, len(cands))
        scores = _score_candidates(value_model, path_vocab, query_vocab, query_tokens, cands, device)
        all_candidates.extend(cands)
        idx_sorted = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        best = idx_sorted[0]
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
        }

    uniq_vals = []
    for _step in range(max_steps):
        rem_q = query_tokens[len(executed) :] if len(executed) < len(query_tokens) else query_tokens
        cands, uniq = _generate_candidates(
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
            candidate_latent_jitter_std=candidate_latent_jitter_std,
        )
        cands = _merge_candidates(cands, memory_candidates)
        uniq = len({tuple(x) for x in cands}) / max(1, len(cands))
        scores = _score_candidates(value_model, path_vocab, query_vocab, rem_q, cands, device)
        all_candidates.extend(cands)
        uniq_vals.append(uniq)
        idx_sorted = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
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
        ).to(device)
        value_model.load_state_dict(value_ckpt["model_state"])
        value_model.eval()

    use_memory_retrieval = bool(cfg.get("use_memory_retrieval", False))
    memory_prefilter_feasible = bool(cfg.get("memory_prefilter_feasible", True))
    memory_top_k = int(cfg.get("memory_top_k", 8))
    token_to_ids: Dict[str, List[int]] = {}
    path_bank: List[List[str]] = []
    if use_memory_retrieval:
        memory_train_path = cfg.get("memory_train_path") or cfg.get("train_path")
        if not memory_train_path:
            raise ValueError("use_memory_retrieval=true requires memory_train_path or train_path in config.")
        train_rows = load_jsonl(memory_train_path)
        token_to_ids, path_bank = _build_memory_index(
            train_rows=train_rows,
            max_postings_per_token=int(cfg.get("memory_max_postings_per_token", 1200)),
        )
        print(f"[memory] enabled, train_rows={len(train_rows)} indexed_paths={len(path_bank)}")

    method = "diplan_torch_mem" if use_memory_retrieval else "diplan_torch"
    candidate_latent_jitter_std = float(cfg.get("candidate_latent_jitter_std", 0.0))
    if candidate_latent_jitter_std > 0:
        print(f"[eval] candidate_latent_jitter_std={candidate_latent_jitter_std}")
    records = []
    memory_prefilter_dropped = 0
    for row in rows:
        mem_cands = (
            _retrieve_memory_candidates(row["query_tokens"], token_to_ids, path_bank, top_k=memory_top_k)
            if use_memory_retrieval
            else []
        )
        if use_memory_retrieval and memory_prefilter_feasible:
            mem_cands, dropped = _filter_feasible_candidates(mem_cands, row["constraints"])
            memory_prefilter_dropped += dropped
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
        )
        executed = pred["executed_path"]
        feasible, violations = _is_feasible(executed, row["constraints"])
        rec = {
            "task_id": row["task_id"],
            "dataset": row["dataset"],
            "query": row.get("question", " ".join(row.get("query_tokens", []))),
            "query_tokens": row.get("query_tokens", []),
            "method": method,
            "oracle_path": row["oracle_path"],
            "planned_path": pred["planned_path"],
            "executed_path": executed,
            "success": executed == row["oracle_path"],
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
        }
        records.append(rec)

    summary = {method: aggregate_method_metrics(records)}
    ensure_dir(args.out)
    dump_jsonl(str(Path(args.out) / "predictions.jsonl"), records)
    dump_json(str(Path(args.out) / "summary_metrics.json"), summary)
    _save_summary_csv(str(Path(args.out) / "summary_table.csv"), summary)
    print(f"Evaluated {len(rows)} tasks with {method}.")
    if use_memory_retrieval and memory_prefilter_feasible:
        print(f"[memory] prefilter dropped={memory_prefilter_dropped}")
    print(summary[method])


if __name__ == "__main__":
    main()
