import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.diagnostics import aggregate_by_dataset, aggregate_diagnostics
from src.diplan.io_utils import dump_json, ensure_dir, load_jsonl


def _flat_row(label: str, metrics: dict) -> dict:
    keys = [
        "n",
        "success_rate",
        "first_error_step",
        "recovery_at_error",
        "trap_at_1",
        "plan_feasibility",
        "constraint_violation_rate",
        "plan_execution_consistency",
        "candidate_pool_hit_rate",
        "ranking_error_rate",
        "candidate_pool_avg_size",
    ]
    row = {"label": label}
    for key in keys:
        row[key] = metrics.get(key, "")
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/diagnostics_report")
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    overall = aggregate_diagnostics(rows)
    by_dataset = aggregate_by_dataset(rows)

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    dump_json(str(out_dir / "diagnostics_overall.json"), overall)
    dump_json(str(out_dir / "diagnostics_by_dataset.json"), by_dataset)

    csv_rows = [_flat_row("overall", overall)]
    for dataset, metrics in by_dataset.items():
        csv_rows.append(_flat_row(dataset, metrics))
    _write_csv(out_dir / "diagnostics_table.csv", csv_rows)
    print(f"[ok] wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
