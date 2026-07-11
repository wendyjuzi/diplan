"""Build paper-ready efficiency tables from DiPLaN/MCTS summary files.

The script accepts both official ToG summaries (a flat JSON dict) and the
controlled subgraph diagnostic summaries (a dict keyed by method name).

Example:
    python scripts/build_efficiency_evidence_table.py \
      --run "DiPLaN-strong|results/diplan_strong/summary_metrics.json" \
      --run "MCTS-strong|results/mcts_strong/summary_metrics.json" \
      --run "Core-DiPLaN|results/core_diag/summary_metrics.json|diplan_diffusion" \
      --run "Core-FLARE|results/core_diag/summary_metrics.json|flare" \
      --out_prefix results/efficiency/evidence
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FIELDS = [
    "label",
    "source_method",
    "n",
    "hits@1",
    "trap@1",
    "wall_time_s_total",
    "wall_time_s_per_task",
    "wall_time_s_per_success",
    "llm_calls",
    "llm_calls_per_task",
    "llm_total_tokens_est",
    "llm_wall_time_s",
    "core_time_s_total",
    "core_time_s_per_task",
    "core_share_of_wall",
    "candidate_expansions_total",
    "trajectory_evals_total",
    "mcts_simulations_total",
    "diffusion_calls_total",
    "diffusion_samples_total",
    "answer_reaching_executed_top1_rate",
    "trap_reduction_note",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_summary(raw: Dict[str, Any], method: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if method:
        if method not in raw:
            raise KeyError(f"method '{method}' not found; available keys: {list(raw)[:20]}")
        return method, raw[method]
    if "hits@1" in raw or "success_rate" in raw:
        return str(raw.get("planning_strategy") or raw.get("method") or "official"), raw
    if len(raw) == 1:
        key = next(iter(raw))
        return str(key), raw[key]
    raise ValueError(
        "Nested summary has multiple methods. Pass a method as LABEL|PATH|METHOD. "
        f"Available keys: {list(raw)[:20]}"
    )


def _num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _first(summary: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key in summary and summary[key] not in (None, ""):
            return summary[key]
    return default


def _core_time(summary: Dict[str, Any]) -> float:
    return _num(
        _first(
            summary,
            [
                "diplan_rerank_wall_time_s",
                "mcts_rerank_wall_time_s",
                "wall_time_s_total",  # controlled diagnostics report method-level wall time.
            ],
            0.0,
        )
    )


def _normalize(label: str, source_method: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    n = _num(summary.get("n"), 0.0)
    wall_total = _num(_first(summary, ["wall_time_s_total"], 0.0))
    wall_task = _num(_first(summary, ["wall_time_s_per_task", "wall_time_s_mean"], 0.0))
    if not wall_task and n:
        wall_task = wall_total / n
    wall_success = _num(_first(summary, ["wall_time_s_per_success", "wall_time_per_success"], 0.0))
    core_total = _core_time(summary)
    core_task = core_total / n if n else 0.0
    row = {
        "label": label,
        "source_method": source_method,
        "n": int(n) if n.is_integer() else n,
        "hits@1": _first(summary, ["hits@1", "success_rate"], ""),
        "trap@1": _first(summary, ["trap@1"], ""),
        "wall_time_s_total": wall_total,
        "wall_time_s_per_task": wall_task,
        "wall_time_s_per_success": wall_success,
        "llm_calls": _first(summary, ["llm_calls"], ""),
        "llm_calls_per_task": _first(summary, ["llm_calls_per_task"], ""),
        "llm_total_tokens_est": _first(summary, ["llm_total_tokens_est"], ""),
        "llm_wall_time_s": _first(summary, ["llm_wall_time_s"], ""),
        "core_time_s_total": core_total,
        "core_time_s_per_task": core_task,
        "core_share_of_wall": core_total / wall_total if wall_total else "",
        "candidate_expansions_total": _first(summary, ["candidate_expansions_total"], ""),
        "trajectory_evals_total": _first(summary, ["trajectory_evals_total", "mcts_trajectory_evals"], ""),
        "mcts_simulations_total": _first(summary, ["mcts_simulations_total", "mcts_simulations"], ""),
        "diffusion_calls_total": _first(summary, ["diffusion_calls_total"], ""),
        "diffusion_samples_total": _first(summary, ["diffusion_samples_total"], ""),
        "answer_reaching_executed_top1_rate": _first(summary, ["answer_reaching_executed_top1_rate"], ""),
        "trap_reduction_note": "",
    }
    return row


def _parse_run(spec: str) -> Tuple[str, Path, Optional[str]]:
    parts = spec.split("|")
    if len(parts) not in {2, 3}:
        raise ValueError("Use --run 'LABEL|PATH' or --run 'LABEL|PATH|METHOD'")
    label = parts[0].strip()
    path = Path(parts[1].strip())
    method = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
    return label, path, method


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 0.01:
            return f"{x:.4g}"
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    cols = [
        "label",
        "hits@1",
        "trap@1",
        "wall_time_s_per_task",
        "llm_calls_per_task",
        "llm_total_tokens_est",
        "core_time_s_total",
        "core_time_s_per_task",
        "core_share_of_wall",
        "answer_reaching_executed_top1_rate",
        "trajectory_evals_total",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(cols) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison(rows: List[Dict[str, Any]], path: Path) -> None:
    comparisons: List[Dict[str, Any]] = []
    by_label = {str(r["label"]): r for r in rows}
    labels = list(by_label)
    for i, a_label in enumerate(labels):
        for b_label in labels[i + 1 :]:
            a = by_label[a_label]
            b = by_label[b_label]
            a_time = _num(a.get("wall_time_s_per_task"))
            b_time = _num(b.get("wall_time_s_per_task"))
            a_hits = _num(a.get("hits@1"))
            b_hits = _num(b.get("hits@1"))
            comparisons.append(
                {
                    "a": a_label,
                    "b": b_label,
                    "time_speedup_a_over_b": a_time / b_time if b_time else "",
                    "time_speedup_b_over_a": b_time / a_time if a_time else "",
                    "hits_delta_a_minus_b": a_hits - b_hits,
                    "hits_retention_a_over_b": a_hits / b_hits if b_hits else "",
                    "hits_retention_b_over_a": b_hits / a_hits if a_hits else "",
                }
            )
    path.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL|PATH or LABEL|PATH|METHOD")
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for spec in args.run:
        label, path, method = _parse_run(spec)
        source_method, summary = _pick_summary(_load_json(path), method)
        rows.append(_normalize(label, source_method, summary))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    cmp_path = out_prefix.with_name(out_prefix.name + "_comparisons.json")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(rows, md_path)
    _write_comparison(rows, cmp_path)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "md": str(md_path), "comparisons": str(cmp_path)}, indent=2))


if __name__ == "__main__":
    main()
