"""Summarize MCTS/FLARE vs DiPLaN efficiency from ToG-subgraph runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


EFF_KEYS = [
    "hits@1",
    "avg_steps",
    "wall_time_s_mean",
    "wall_time_s_total",
    "wall_time_per_success",
    "candidate_expansions_mean",
    "candidate_expansions_total",
    "trajectory_evals_mean",
    "trajectory_evals_total",
    "mcts_simulations_mean",
    "mcts_simulations_total",
    "mcts_expanded_nodes_mean",
    "mcts_rollout_steps_mean",
    "diffusion_calls_mean",
    "diffusion_calls_total",
    "diffusion_samples_mean",
    "diffusion_samples_total",
    "diplan_candidate_rerank_calls_mean",
    "value_rerank_items_mean",
]


def _load(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Directory containing summary_metrics.json")
    ap.add_argument("--mcts_label", default="flare")
    ap.add_argument("--diplan_label", default="diplan_diffusion")
    ap.add_argument("--out_prefix", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary = _load(run_dir / "summary_metrics.json")
    rows: List[Dict] = []
    for method, metrics in summary.items():
        row = {"method": method}
        for key in EFF_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)

    mcts = summary.get(args.mcts_label, {})
    diplan = summary.get(args.diplan_label, {})
    comparison = {
        "mcts_label": args.mcts_label,
        "diplan_label": args.diplan_label,
        "mcts_hits@1": mcts.get("hits@1"),
        "diplan_hits@1": diplan.get("hits@1"),
        "mcts_wall_time_total": mcts.get("wall_time_s_total"),
        "diplan_wall_time_total": diplan.get("wall_time_s_total"),
        "wall_time_speedup_mcts_over_diplan": _safe_div(
            float(mcts.get("wall_time_s_total", 0.0)),
            float(diplan.get("wall_time_s_total", 0.0)),
        ),
        "mcts_candidate_expansions_total": mcts.get("candidate_expansions_total"),
        "diplan_candidate_expansions_total": diplan.get("candidate_expansions_total"),
        "candidate_expansion_reduction": 1.0
        - _safe_div(
            float(diplan.get("candidate_expansions_total", 0.0)),
            float(mcts.get("candidate_expansions_total", 0.0)),
        ),
        "mcts_trajectory_evals_total": mcts.get("trajectory_evals_total"),
        "diplan_diffusion_samples_total": diplan.get("diffusion_samples_total"),
    }

    out_prefix = args.out_prefix or str(run_dir / "mcts_efficiency")
    out_csv = Path(out_prefix + ".csv")
    out_json = Path(out_prefix + ".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + EFF_KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    out_json.write_text(json.dumps({"rows": rows, "comparison": comparison}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"csv": str(out_csv), "json": str(out_json), "comparison": comparison}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
