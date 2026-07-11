from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_json


def _pick_summary(path: str) -> tuple[str, Dict[str, Any]]:
    raw = load_json(path)
    if "success_rate" in raw:
        return Path(path).stem, raw
    if len(raw) != 1:
        raise ValueError(f"Expected single-method summary JSON: {path}")
    key = next(iter(raw))
    return key, raw[key]


def _row(env: str, label: str, path: str) -> Dict[str, Any]:
    method_key, summary = _pick_summary(path)
    success = float(summary.get("success_rate", 0.0))
    feas = float(summary.get("plan_feasibility", 0.0))
    return {
        "environment": env,
        "method": label,
        "summary_key": method_key,
        "success_rate": success,
        "plan_feasibility": feas,
        "gap_proxy": max(0.0, feas - success),
        "plan_execution_consistency": float(summary.get("plan_execution_consistency", 0.0)),
        "first_error_step": float(summary.get("first_error_step", 0.0)),
        "llm_calls": float(summary.get("llm_calls", 0.0)),
        "token_cost": float(summary.get("token_cost", 0.0)),
        "latency_cost": float(summary.get("latency_cost", 0.0)),
        "source": path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comparison table across ALFWorld/ScienceWorld for ReAct, CoT, and DiPLaN.")
    parser.add_argument("--alfworld-react", type=str, required=True)
    parser.add_argument("--alfworld-cot", type=str, required=True)
    parser.add_argument("--alfworld-diplan", type=str, required=True)
    parser.add_argument("--scienceworld-react", type=str, required=True)
    parser.add_argument("--scienceworld-cot", type=str, required=True)
    parser.add_argument("--scienceworld-diplan", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = [
        _row("ALFWorld", "ReAct", args.alfworld_react),
        _row("ALFWorld", "CoT", args.alfworld_cot),
        _row("ALFWorld", "DiPLaN", args.alfworld_diplan),
        _row("ScienceWorld", "ReAct", args.scienceworld_react),
        _row("ScienceWorld", "CoT", args.scienceworld_cot),
        _row("ScienceWorld", "DiPLaN", args.scienceworld_diplan),
    ]

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    dump_json(str(out_dir / "comparison_table.json"), {"rows": rows})

    csv_path = out_dir / "comparison_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "environment",
                "method",
                "success_rate",
                "plan_feasibility",
                "gap_proxy",
                "plan_execution_consistency",
                "first_error_step",
                "llm_calls",
                "token_cost",
                "latency_cost",
                "summary_key",
                "source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "| Environment | Method | Success | Feasibility | GapProxy | Consistency | FirstErr | Calls | Tokens | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {environment} | {method} | {success_rate:.3f} | {plan_feasibility:.3f} | {gap_proxy:.3f} | "
            "{plan_execution_consistency:.3f} | {first_error_step:.2f} | {llm_calls:.2f} | {token_cost:.2f} | {latency_cost:.2f} |".format(**row)
        )
    (out_dir / "comparison_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[ok] comparison table written to {out_dir}")


if __name__ == "__main__":
    main()
