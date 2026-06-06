"""Summarize DiPLaN experiment folders into paper-ready tables.

The script recursively scans result directories for ``summary_metrics.json`` and
exports compact CSV + Markdown tables for KGQA, ALFWorld, and all runs. It is
designed for messy research workspaces where run names encode most ablations.

Examples:
    python scripts/summarize_paper_results.py \
        --results_root results \
        --out_dir result_paper/tables

    python scripts/summarize_paper_results.py \
        --run "ALFWorld exec-v4=results/final_alfworld_seed42/full_ood134_exec_v4" \
        --run "KGQA seed43=results/final_kgqa_pool48_strong_multiseed/seed_43/mlp_memory_prefilter_cross_fullpool" \
        --out_dir result_paper/tables_selected
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ALL_COLUMNS = [
    "label",
    "task",
    "method",
    "n",
    "success_rate",
    "first_error_step",
    "trap_at_1",
    "recovery_at_error",
    "plan_feasibility",
    "constraint_violation_rate",
    "plan_execution_consistency",
    "grounded_fallback_rate",
    "candidate_pool_avg_size",
    "avg_steps",
    "token_cost",
    "latency_cost",
    "candidate_pool_hit_rate",
    "conditional_success_given_pool_hit",
    "ranking_error_rate",
    "oracle_mrr",
    "oracle_hit_at_1",
    "oracle_hit_at_3",
    "oracle_hit_at_5",
    "selected_rank_by_projection",
    "selected_rank_by_value",
    "projection_score_selected",
    "projection_density_selected",
    "path",
]

ALFWORLD_COLUMNS = [
    "label",
    "n",
    "success_rate",
    "first_error_step",
    "trap_at_1",
    "plan_feasibility",
    "grounded_fallback_rate",
    "candidate_pool_avg_size",
    "avg_steps",
    "selected_rank_by_projection",
    "path",
]

KGQA_COLUMNS = [
    "label",
    "n",
    "success_rate",
    "candidate_pool_hit_rate",
    "conditional_success_given_pool_hit",
    "ranking_error_rate",
    "oracle_mrr",
    "oracle_hit_at_1",
    "oracle_hit_at_3",
    "oracle_hit_at_5",
    "first_error_step",
    "path",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Any:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return x
    return x


def _fmt(x: Any) -> str:
    x = _safe_float(x)
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x) >= 100:
            return f"{x:.1f}"
        if abs(x) >= 10:
            return f"{x:.2f}"
        return f"{x:.4f}"
    return str(x)


def _parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run {spec!r}; expected label=path")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --run {spec!r}; expected label=path")
    return label, Path(path)


def _default_label(run_dir: Path, root: Path) -> str:
    try:
        rel = run_dir.relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(run_dir).replace("\\", "/")


def _infer_task(path: str, method: str, metrics: Dict[str, Any]) -> str:
    text = f"{path} {method}".lower()
    if "alfworld" in text or "alf_" in text:
        return "alfworld"
    if "kgqa" in text or "cwq" in text or "webqsp" in text:
        return "kgqa"
    if "grounded_fallback_rate" in metrics:
        return "alfworld"
    if "oracle_mrr" in metrics or "candidate_pool_hit_rate" in metrics:
        return "kgqa"
    return "other"


def _read_summary(label: str, run_dir: Path, root: Path) -> Dict[str, Any] | None:
    summary_path = run_dir / "summary_metrics.json"
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[warn] failed reading {summary_path}: {exc}")
        return None
    if not isinstance(summary, dict) or not summary:
        return None
    method = next(iter(summary.keys()))
    metrics = summary.get(method, {})
    if not isinstance(metrics, dict):
        return None
    path_s = str(run_dir).replace("\\", "/")
    row: Dict[str, Any] = {
        "label": label or _default_label(run_dir, root),
        "method": method,
        "path": path_s,
    }
    row.update(metrics)
    row["task"] = _infer_task(path_s, method, metrics)
    return row


def _scan_results(root: Path, max_depth: int) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    root = root.resolve()
    for summary_path in root.rglob("summary_metrics.json"):
        run_dir = summary_path.parent
        try:
            depth = len(run_dir.relative_to(root).parts)
        except ValueError:
            depth = 999
        if depth <= max_depth:
            out.append((_default_label(run_dir, root), run_dir))
    out.sort(key=lambda x: x[0])
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({k for r in rows for k in r.keys() if k not in columns})
    fields = columns + extras
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_float(row.get(k, "")) for k in fields})


def _md_table(rows: List[Dict[str, Any]], columns: List[str], title: str) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        lines += ["_No runs found._", ""]
        return "\n".join(lines)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = [_fmt(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> Tuple[str, float, str]:
        sr = row.get("success_rate", -1)
        try:
            sr_f = float(sr)
        except Exception:
            sr_f = -1.0
        return (str(row.get("task", "")), -sr_f, str(row.get("label", "")))

    return sorted(rows, key=key)


def _write_markdown(path: Path, all_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alf = _sort_rows([r for r in all_rows if r.get("task") == "alfworld"])
    kgqa = _sort_rows([r for r in all_rows if r.get("task") == "kgqa"])
    other = _sort_rows([r for r in all_rows if r.get("task") == "other"])
    text = [
        "# DiPLaN Paper Result Tables",
        "",
        "_Generated by `scripts/summarize_paper_results.py`._",
        "",
        _md_table(alf, ALFWORLD_COLUMNS, "ALFWorld Runs"),
        _md_table(kgqa, KGQA_COLUMNS, "KGQA Runs"),
        _md_table(other, ALL_COLUMNS[:12] + ["path"], "Other Runs"),
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _print_highlights(rows: List[Dict[str, Any]]) -> None:
    for task in ("alfworld", "kgqa"):
        subset = [r for r in rows if r.get("task") == task and "success_rate" in r]
        if not subset:
            continue
        best = max(subset, key=lambda r: float(r.get("success_rate", -1)))
        print(
            f"[best:{task}] {best.get('label')} "
            f"success={_fmt(best.get('success_rate'))} path={best.get('path')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DiPLaN result folders for paper tables.")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="result_paper/tables")
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional explicit label=run_dir. If provided, these are added in addition to scanned runs.",
    )
    parser.add_argument("--no_scan", action="store_true", help="Only use --run entries; do not scan results_root.")
    args = parser.parse_args()

    root = Path(args.results_root)
    specs: List[Tuple[str, Path]] = []
    if not args.no_scan and root.exists():
        specs.extend(_scan_results(root, max_depth=int(args.max_depth)))
    for spec in args.run:
        specs.append(_parse_run_spec(spec))

    # De-duplicate by resolved path, while keeping explicit labels if repeated.
    dedup: Dict[str, Tuple[str, Path]] = {}
    for label, run_dir in specs:
        key = str(run_dir.resolve())
        dedup[key] = (label, run_dir)

    rows = []
    for label, run_dir in dedup.values():
        row = _read_summary(label, run_dir, root)
        if row is not None:
            rows.append(row)
    rows = _sort_rows(rows)

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "all_runs.csv", rows, ALL_COLUMNS)
    _write_csv(out_dir / "alfworld_runs.csv", [r for r in rows if r.get("task") == "alfworld"], ALFWORLD_COLUMNS)
    _write_csv(out_dir / "kgqa_runs.csv", [r for r in rows if r.get("task") == "kgqa"], KGQA_COLUMNS)
    _write_markdown(out_dir / "paper_tables.md", rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "results_root": str(root),
                "num_runs": len(rows),
                "outputs": [
                    "all_runs.csv",
                    "alfworld_runs.csv",
                    "kgqa_runs.csv",
                    "paper_tables.md",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[ok] summarized {len(rows)} runs -> {out_dir}")
    _print_highlights(rows)


if __name__ == "__main__":
    main()
