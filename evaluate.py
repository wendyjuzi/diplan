import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src.diplan.evaluate_core import evaluate_suite
from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_config, load_json, load_jsonl


def _flatten_records(records_by_method: Dict[str, List[Dict]]) -> List[Dict]:
    rows: List[Dict] = []
    for method, recs in records_by_method.items():
        rows.extend(recs)
    return rows


def _save_summary_table(path: str, summary: Dict[str, Dict]) -> None:
    ensure_dir(str(Path(path).parent))
    fieldnames = [
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for method, met in summary.items():
            row = {"method": method}
            row.update(met)
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_kgqa.yaml")
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--constraint_ckpt", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["cwq", "webqsp", "grailqa"])
    parser.add_argument("--out", type=str, default="results/main")
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_rows_all = load_jsonl(cfg["test_path"])
    test_rows = [r for r in test_rows_all if r["dataset"] in set(args.benchmarks)]

    planner_model = load_json(args.planner_ckpt)
    value_model = load_json(args.value_ckpt)
    constraint_model = load_json(args.constraint_ckpt)

    result = evaluate_suite(
        test_rows=test_rows,
        planner_model=planner_model,
        value_model=value_model,
        constraint_model=constraint_model,
        methods=cfg["methods"],
        seed=int(cfg.get("seed", 42)),
        options=cfg.get("diplan_options", {}),
    )
    records = _flatten_records(result["records"])
    summary = result["summary"]

    ensure_dir(args.out)
    dump_jsonl(str(Path(args.out) / "predictions.jsonl"), records)
    dump_json(str(Path(args.out) / "summary_metrics.json"), summary)
    _save_summary_table(str(Path(args.out) / "summary_table.csv"), summary)

    print(f"Evaluated {len(test_rows)} tasks across {len(cfg['methods'])} methods.")
    print(f"Results written to {args.out}")
    for m, met in summary.items():
        print(f"{m:>12} | success={met['success_rate']:.3f} trap@1={met['trap_at_1']:.3f} first_err={met['first_error_step']:.2f}")


if __name__ == "__main__":
    main()

