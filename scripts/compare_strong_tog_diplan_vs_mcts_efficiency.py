"""Compare strong official-ToG DiPLaN efficiency against MCTS/FLARE diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _method_summary(summary, method: str):
    if method in summary:
        return summary[method]
    return summary


def _div(a, b):
    a = float(a or 0.0)
    b = float(b or 0.0)
    return a / b if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcts_summary", required=True, help="summary_metrics.json from run_tog_subgraph_planning_eval.py")
    ap.add_argument("--diplan_summary", required=True, help="summary_metrics.json from official main_subgraph_diplan.py")
    ap.add_argument("--mcts_method", default="flare")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mcts_all = _load(Path(args.mcts_summary))
    mcts = _method_summary(mcts_all, args.mcts_method)
    diplan = _load(Path(args.diplan_summary))
    out = {
        "mcts_method": args.mcts_method,
        "mcts_n": mcts.get("n"),
        "diplan_n": diplan.get("n"),
        "mcts_hits@1": mcts.get("hits@1"),
        "diplan_hits@1": diplan.get("hits@1"),
        "mcts_wall_time_s_total": mcts.get("wall_time_s_total"),
        "diplan_wall_time_s_total": diplan.get("wall_time_s_total"),
        "wall_time_speedup_mcts_over_diplan": _div(mcts.get("wall_time_s_total"), diplan.get("wall_time_s_total")),
        "mcts_wall_time_per_success": mcts.get("wall_time_per_success"),
        "diplan_wall_time_per_success": diplan.get("wall_time_s_per_success"),
        "wall_time_per_success_speedup": _div(mcts.get("wall_time_per_success"), diplan.get("wall_time_s_per_success")),
        "mcts_candidate_expansions_total": mcts.get("candidate_expansions_total"),
        "diplan_rerank_calls": diplan.get("diplan_rerank_calls"),
        "diplan_value_items": diplan.get("diplan_value_items"),
        "diplan_prior_samples": diplan.get("diplan_prior_samples"),
        "diplan_guided_rollout_paths": diplan.get("diplan_guided_rollout_paths"),
        "diplan_fusion_items": diplan.get("diplan_fusion_items"),
        "mcts_trajectory_evals_total": mcts.get("trajectory_evals_total"),
        "mcts_simulations_total": mcts.get("mcts_simulations_total"),
        "mcts_llm_calls_note": "Use diagnostics.json from the same MCTS run if needed; strong DiPLaN summary tracks llm_calls directly.",
        "diplan_llm_calls": diplan.get("llm_calls"),
        "diplan_llm_total_tokens_est": diplan.get("llm_total_tokens_est"),
        "diplan_rerank_wall_time_s": diplan.get("diplan_rerank_wall_time_s"),
        "diplan_rerank_wall_time_per_call": diplan.get("diplan_rerank_wall_time_per_call"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
