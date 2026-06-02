import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl


def _parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec: {spec}. Expected label=path.")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --run spec: {spec}. Expected label=path.")
    return label, Path(path)


def _load_summary(run_dir: Path) -> Dict:
    summary = load_json(str(run_dir / "summary_metrics.json"))
    if not summary:
        raise ValueError(f"Empty summary_metrics.json in {run_dir}")
    method = next(iter(summary.keys()))
    return summary[method]


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _metrics_from_predictions(run_dir: Path) -> Dict:
    preds = load_jsonl(str(run_dir / "predictions.jsonl"))
    n = len(preds)
    if n == 0:
        return {"n": 0}
    hits = [r for r in preds if r.get("oracle_in_candidate_pool")]
    success = sum(1.0 if r.get("success") else 0.0 for r in preds) / n
    coverage = len(hits) / n
    conditional = sum(1.0 if r.get("success") else 0.0 for r in hits) / max(1, len(hits))
    out = {
        "n": n,
        "success_rate": success,
        "candidate_pool_hit_rate": coverage,
        "conditional_success_given_pool_hit": conditional,
        "decomposed_success": coverage * conditional,
        "unexplained_gap": success - (coverage * conditional),
    }
    for key in ("oracle_mrr", "oracle_hit_at_1", "oracle_hit_at_3", "oracle_hit_at_5", "oracle_rank_mean"):
        vals = [_safe_float(r.get(key, 0.0)) for r in preds]
        out[key] = sum(vals) / max(1, len(vals))
    return out


def _write_csv(path: Path, rows: List[Dict]) -> None:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _write_markdown(path: Path, rows: List[Dict]) -> None:
    headers = [
        "method",
        "success_rate",
        "candidate_pool_hit_rate",
        "conditional_success_given_pool_hit",
        "oracle_mrr",
        "oracle_hit_at_3",
        "oracle_hit_at_5",
        "oracle_rank_mean",
    ]
    lines = [
        "| Method | Success | Candidate Hit | Conditional Success | MRR | Hit@3 | Hit@5 | Mean Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("setting", "")),
                    _fmt_pct(_safe_float(row.get("success_rate"))),
                    _fmt_pct(_safe_float(row.get("candidate_pool_hit_rate"))),
                    _fmt_pct(_safe_float(row.get("conditional_success_given_pool_hit"))),
                    f"{_safe_float(row.get('oracle_mrr')):.3f}",
                    _fmt_pct(_safe_float(row.get("oracle_hit_at_3"))),
                    _fmt_pct(_safe_float(row.get("oracle_hit_at_5"))),
                    f"{_safe_float(row.get('oracle_rank_mean')):.2f}",
                ]
            )
            + " |"
        )
    ensure_dir(str(path.parent))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose KGQA success into coverage and conditional ranking success.")
    parser.add_argument("--run", action="append", required=True, help="Run spec label=path_to_run_dir.")
    parser.add_argument("--out_dir", type=str, default="results/ranking_bottleneck_decomposition")
    parser.add_argument("--prefix", type=str, default="ranking_bottleneck")
    args = parser.parse_args()

    rows: List[Dict] = []
    for spec in args.run:
        label, run_dir = _parse_run(spec)
        summary = _load_summary(run_dir)
        pred_metrics = _metrics_from_predictions(run_dir)
        row = {"setting": label, "run_dir": str(run_dir)}
        row.update(summary)
        row.update(pred_metrics)
        rows.append(row)

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    _write_csv(out_dir / f"{args.prefix}_decomposition.csv", rows)
    _write_markdown(out_dir / f"{args.prefix}_table.md", rows)
    dump_json(
        str(out_dir / f"{args.prefix}_manifest.json"),
        {
            "runs": [{"setting": r["setting"], "run_dir": r["run_dir"]} for r in rows],
            "outputs": {
                "csv": str(out_dir / f"{args.prefix}_decomposition.csv"),
                "markdown": str(out_dir / f"{args.prefix}_table.md"),
            },
            "interpretation_note": (
                "If candidate_pool_hit_rate is stable while conditional_success_given_pool_hit "
                "and ranking metrics improve, the evidence supports a ranking-quality explanation "
                "rather than a candidate-coverage explanation."
            ),
        },
    )
    print(f"[ok] wrote decomposition tables under {out_dir}")


if __name__ == "__main__":
    main()
