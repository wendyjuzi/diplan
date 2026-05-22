import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

from src.diplan.io_utils import load_jsonl


def _summarize(records: List[Dict]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    hit = [1.0 if r.get("oracle_in_candidate_pool", False) else 0.0 for r in records]
    succ = [1.0 if r.get("success", False) else 0.0 for r in records]
    feasible = [1.0 if r.get("feasible", False) else 0.0 for r in records]
    ranking_loss = [1.0 if (r.get("oracle_in_candidate_pool", False) and not r.get("success", False)) else 0.0 for r in records]
    miss_pool = [1.0 if not r.get("oracle_in_candidate_pool", False) else 0.0 for r in records]
    oracle_ranking_upper = mean(hit)  # if ranker always picks gold once present
    return {
        "n": n,
        "final_success_rate": mean(succ),
        "candidate_pool_hit_rate": mean(hit),
        "oracle_ranking_upper_bound": oracle_ranking_upper,
        "ranking_loss_rate": mean(ranking_loss),
        "pool_miss_rate": mean(miss_pool),
        "plan_feasibility": mean(feasible),
        "selection_efficiency_given_hit": (mean(succ) / max(1e-9, mean(hit))),
        "recoverable_gain_to_upper": max(0.0, oracle_ranking_upper - mean(succ)),
    }


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="results/diagnostics/oracle_decomposition.csv")
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    rows = []
    overall = _summarize(preds)
    overall["dataset"] = "overall"
    rows.append(overall)

    by_ds = defaultdict(list)
    for r in preds:
        by_ds[str(r.get("dataset", "unknown")).lower()].append(r)
    for ds, recs in sorted(by_ds.items()):
        row = _summarize(recs)
        row["dataset"] = ds
        rows.append(row)

    fields = [
        "dataset",
        "n",
        "final_success_rate",
        "candidate_pool_hit_rate",
        "oracle_ranking_upper_bound",
        "ranking_loss_rate",
        "pool_miss_rate",
        "plan_feasibility",
        "selection_efficiency_given_hit",
        "recoverable_gain_to_upper",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    print(f"[ok] wrote {args.out_csv}")
    for r in rows:
        print(
            f"{r['dataset']:>8} n={int(r['n'])} "
            f"success={r['final_success_rate']:.4f} hit={r['candidate_pool_hit_rate']:.4f} "
            f"upper={r['oracle_ranking_upper_bound']:.4f} ranking_loss={r['ranking_loss_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
