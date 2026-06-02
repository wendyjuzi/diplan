import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_config, load_jsonl
from src.diplan.metrics import (
    aggregate_method_metrics,
    first_error_step,
    plan_execution_consistency,
    recovery_at_error,
    trap_at_1,
)


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
        "candidate_pool_hit_rate",
        "ranking_error_rate",
        "candidate_pool_avg_size",
        "conditional_success_given_pool_hit",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m, met in summary.items():
            row = {"method": m}
            row.update(met)
            w.writerow(row)


def _is_feasible(path: List[str], constraints: Dict) -> Tuple[bool, List[str]]:
    violations = []
    if len(path) > int(constraints.get("max_steps", 8)):
        violations.append("max_steps_exceeded")
    banned = set(constraints.get("banned_relations", []))
    for rel in path:
        if rel in banned:
            violations.append("banned_relation")
    return len(violations) == 0, sorted(set(violations))


def _build_index(train_rows: List[Dict], max_postings_per_token: int) -> Tuple[Dict[str, List[int]], List[List[str]]]:
    token_to_ids: Dict[str, List[int]] = defaultdict(list)
    path_bank: List[List[str]] = []
    for row in train_rows:
        path = row.get("oracle_path", [])
        if not isinstance(path, list) or not path:
            continue
        path_id = len(path_bank)
        path_bank.append(path)
        seen = set()
        for token in row.get("query_tokens", []):
            if not isinstance(token, str):
                continue
            t = token.strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            posting = token_to_ids[t]
            if len(posting) < max_postings_per_token:
                posting.append(path_id)
    return token_to_ids, path_bank


def _retrieve(query_tokens: List[str], token_to_ids: Dict[str, List[int]], path_bank: List[List[str]], top_k: int) -> List[List[str]]:
    score = defaultdict(int)
    for token in set(t.strip().lower() for t in query_tokens if isinstance(t, str) and t.strip()):
        for path_id in token_to_ids.get(token, []):
            score[path_id] += 1
    if not score:
        return []
    ranked = sorted(
        score.items(),
        key=lambda x: (-x[1], abs(len(path_bank[x[0]]) - max(1, len(query_tokens)))),
    )
    out = []
    seen = set()
    for path_id, _ in ranked:
        path = path_bank[path_id]
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
        if len(out) >= top_k:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_retrieval_baseline.yaml")
    parser.add_argument("--out", type=str, default="results/retrieval_baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_rows = load_jsonl(cfg["train_path"])
    test_rows = load_jsonl(cfg["test_path"])
    include_datasets = [str(x).lower() for x in cfg.get("include_datasets", [])]
    if include_datasets:
        test_rows = [r for r in test_rows if str(r.get("dataset", "")).lower() in set(include_datasets)]
        print(f"[retrieval] include_datasets={include_datasets} kept={len(test_rows)}")
    max_tasks = int(cfg.get("max_tasks", 0))
    if max_tasks > 0:
        test_rows = test_rows[:max_tasks]
        print(f"[retrieval] max_tasks={max_tasks} kept={len(test_rows)}")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    token_to_ids, path_bank = _build_index(
        train_rows=train_rows,
        max_postings_per_token=int(cfg.get("max_postings_per_token", 1200)),
    )
    top_k = int(cfg.get("top_k", 8))
    fallback_path = random.choice(path_bank) if path_bank else []

    method = "retrieval_only"
    records = []
    for row in test_rows:
        candidates = _retrieve(row.get("query_tokens", []), token_to_ids, path_bank, top_k=top_k)
        predicted = candidates[0] if candidates else fallback_path
        feasible, violations = _is_feasible(predicted, row["constraints"])
        oracle_in_candidate_pool = tuple(row["oracle_path"]) in {tuple(c) for c in candidates}
        success = predicted == row["oracle_path"]
        rec = {
            "task_id": row["task_id"],
            "dataset": row["dataset"],
            "query": row.get("question", " ".join(row.get("query_tokens", []))),
            "query_tokens": row.get("query_tokens", []),
            "method": method,
            "oracle_path": row["oracle_path"],
            "planned_path": predicted,
            "executed_path": predicted,
            "success": success,
            "first_error_step": first_error_step(predicted, row["oracle_path"]),
            "recovery_at_error": recovery_at_error(predicted, row["oracle_path"]),
            "trap_at_1": trap_at_1(predicted, row["trap_path"]),
            "feasible": feasible,
            "violations": violations,
            "plan_execution_consistency": plan_execution_consistency(predicted, predicted),
            "token_cost": len(predicted),
            "latency_cost": 0.001 * max(1, len(candidates)),
            "diversity_coverage": float(len({tuple(x) for x in candidates}) / max(1, len(candidates) if candidates else 1)),
            "oracle_in_candidate_pool": oracle_in_candidate_pool,
            "candidate_pool_size": len({tuple(x) for x in candidates}),
            "ranking_error": bool((not success) and oracle_in_candidate_pool),
        }
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
    print(f"[retrieval] train={len(train_rows)} test={len(test_rows)} indexed_paths={len(path_bank)}")
    print(summary[method])


if __name__ == "__main__":
    main()
