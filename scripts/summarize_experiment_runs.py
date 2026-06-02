import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl
from src.diplan.stats_utils import bootstrap_mean_diff, mcnemar_test_paired


def _parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec: {spec}. Expected label=path.")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --run spec: {spec}. Expected label=path.")
    return label, Path(path)


def _load_run(label: str, run_dir: Path) -> Dict:
    summary_path = run_dir / "summary_metrics.json"
    preds_path = run_dir / "predictions.jsonl"
    summary = load_json(str(summary_path))
    preds = load_jsonl(str(preds_path))
    if not summary:
        raise ValueError(f"Empty summary_metrics.json in {run_dir}")
    method = next(iter(summary.keys()))
    return {
        "label": label,
        "dir": str(run_dir),
        "method": method,
        "summary": summary[method],
        "preds": preds,
    }


def _write_csv(path: Path, rows: List[Dict], header: List[str]) -> None:
    all_fields = list(header)
    seen = set(all_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({key: r.get(key, "") for key in all_fields})


def _dataset_breakdown(run: Dict) -> List[Dict]:
    grouped = defaultdict(list)
    for r in run["preds"]:
        grouped[r.get("dataset", "unknown")].append(r)
    rows = []
    for ds, recs in sorted(grouped.items()):
        n = len(recs)
        success_rate = sum(1 for r in recs if r.get("success")) / max(1, n)
        plan_feasibility = sum(1 for r in recs if r.get("feasible", False)) / max(1, n)
        diversity = sum(float(r.get("diversity_coverage", 0.0)) for r in recs) / max(1, n)
        rows.append(
            {
                "setting": run["label"],
                "dataset": ds,
                "n": n,
                "success_rate": success_rate,
                "plan_feasibility": plan_feasibility,
                "avg_diversity_coverage": diversity,
            }
        )
    return rows


def _aligned_success(preds: List[Dict]) -> Dict[str, int]:
    return {str(r["task_id"]): 1 if r.get("success") else 0 for r in preds}


def _significance(ref: Dict, other: Dict, bootstrap_n: int, seed: int) -> Dict:
    ref_map = _aligned_success(ref["preds"])
    oth_map = _aligned_success(other["preds"])
    common = sorted(set(ref_map.keys()) & set(oth_map.keys()))
    if not common:
        raise ValueError(f"No overlapping task_id between {ref['label']} and {other['label']}")
    a = [ref_map[k] for k in common]
    b = [oth_map[k] for k in common]
    mcnemar = mcnemar_test_paired(a, b)
    boot = bootstrap_mean_diff(a, b, n_resamples=bootstrap_n, seed=seed)
    return {
        "reference": ref["label"],
        "other": other["label"],
        "n_common": len(common),
        "ref_success_rate": sum(a) / max(1, len(a)),
        "other_success_rate": sum(b) / max(1, len(b)),
        "success_delta_other_minus_ref": (sum(b) - sum(a)) / max(1, len(a)),
        "mcnemar_p_approx": mcnemar.get("p_approx"),
        "mcnemar_b": mcnemar.get("b"),
        "mcnemar_c": mcnemar.get("c"),
        "bootstrap_ci_low": boot.get("ci_low"),
        "bootstrap_ci_high": boot.get("ci_high"),
        "bootstrap_mean_diff": boot.get("mean_diff"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec label=path_to_run_dir (repeatable).",
    )
    parser.add_argument("--out_dir", type=str, default="results/diagnostics")
    parser.add_argument("--prefix", type=str, default="paper_ready")
    parser.add_argument("--ref_label", type=str, default="")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        label, run_dir = _parse_run(spec)
        runs.append(_load_run(label, run_dir))

    if len(runs) < 2:
        raise ValueError("Please provide at least two --run items.")

    ensure_dir(args.out_dir)
    out_dir = Path(args.out_dir)

    aggregate_rows = []
    for run in runs:
        row = {"setting": run["label"], "method": run["method"]}
        row.update(run["summary"])
        aggregate_rows.append(row)
    agg_header = [
        "setting",
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
    agg_header += sorted({k for r in aggregate_rows for k in r.keys() if k not in set(agg_header)})
    _write_csv(out_dir / f"{args.prefix}_aggregate.csv", aggregate_rows, agg_header)

    ds_rows = []
    for run in runs:
        ds_rows.extend(_dataset_breakdown(run))
    ds_header = ["setting", "dataset", "n", "success_rate", "plan_feasibility", "avg_diversity_coverage"]
    _write_csv(out_dir / f"{args.prefix}_dataset_breakdown.csv", ds_rows, ds_header)

    label_map = {r["label"]: r for r in runs}
    ref_label = args.ref_label.strip() if args.ref_label else runs[0]["label"]
    if ref_label not in label_map:
        raise ValueError(f"ref_label={ref_label} not found in run labels.")
    ref_run = label_map[ref_label]
    sig_rows = []
    for run in runs:
        if run["label"] == ref_label:
            continue
        sig_rows.append(_significance(ref_run, run, bootstrap_n=args.bootstrap, seed=args.seed))
    sig_header = [
        "reference",
        "other",
        "n_common",
        "ref_success_rate",
        "other_success_rate",
        "success_delta_other_minus_ref",
        "mcnemar_p_approx",
        "mcnemar_b",
        "mcnemar_c",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_mean_diff",
    ]
    _write_csv(out_dir / f"{args.prefix}_significance_vs_{ref_label}.csv", sig_rows, sig_header)

    dump_json(
        str(out_dir / f"{args.prefix}_manifest.json"),
        {
            "runs": [{"label": r["label"], "dir": r["dir"], "method": r["method"]} for r in runs],
            "reference": ref_label,
            "outputs": {
                "aggregate_csv": str(out_dir / f"{args.prefix}_aggregate.csv"),
                "dataset_breakdown_csv": str(out_dir / f"{args.prefix}_dataset_breakdown.csv"),
                "significance_csv": str(out_dir / f"{args.prefix}_significance_vs_{ref_label}.csv"),
            },
        },
    )
    print(f"[ok] wrote aggregate/dataset/significance tables under {out_dir}")


if __name__ == "__main__":
    main()
