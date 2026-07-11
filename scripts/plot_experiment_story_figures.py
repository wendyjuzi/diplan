"""Generate the paper-story figures for DiPLaN experiments.

This script focuses on the three figures that best support the
"diagnosis -> repair -> mechanism" narrative:

1. Execution funnel (pool -> filtered -> selected -> executed)
2. Efficiency scatter (accuracy vs latency / LLM calls)
3. Selection-to-Execution gap vs step

It supports two usage modes:

- Paper-default mode: no extra inputs required; uses the headline numbers from
  the current paper draft plus an illustrative step-gap curve.
- Real-data mode: pass summary JSONs and trace JSONLs to render figures from
  actual experiment outputs.

Examples
--------
Paper-default figures:
    python scripts/plot_experiment_story_figures.py \
      --out_dir result_paper/latex/figures/story

Use real summary metrics:
    python scripts/plot_experiment_story_figures.py \
      --funnel_summary results/webqsp/diplan/summary_metrics.json \
      --efficiency_summary "DiPLaN|results/webqsp/diplan/summary_metrics.json" \
      --efficiency_summary "FLARE-MCTS|results/webqsp/flare/summary_metrics.json" \
      --out_dir result_paper/latex/figures/story

Use real per-step traces:
    python scripts/plot_experiment_story_figures.py \
      --gap_trace "DiPLaN|results/official_diplan/trace_predictions.jsonl" \
      --gap_trace "ToG|results/official_tog/trace_predictions.jsonl" \
      --gap_trace "PoG|results/official_pog/trace_predictions.jsonl" \
      --out_dir result_paper/latex/figures/story
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "result_paper" / "latex" / "figures" / "story"


COLORS = {
    "DiPLaN": "#1F6FEB",
    "DiPLaN-lite": "#2AA198",
    "FLARE-MCTS": "#C44536",
    "FLARE": "#C44536",
    "ToG": "#F39C12",
    "PoG": "#6AB04C",
    "RoG": "#8E44AD",
    "Single Step": "#F39C12",
    "Lookahead": "#8E44AD",
}

MARKERS = {
    "DiPLaN": "o",
    "DiPLaN-lite": "D",
    "FLARE-MCTS": "X",
    "FLARE": "X",
    "ToG": "s",
    "PoG": "^",
    "RoG": "P",
}


DEFAULT_FUNNEL = {
    "title": "WebQSP Execution Funnel",
    "subtitle": "Full diagnostic run, relation_first_k=16",
    "stages": ["In Pool", "After Filter", "Selected", "Executed"],
    "values": [99.07, 97.04, 93.85, 59.62],
}


DEFAULT_EFFICIENCY_ROWS = [
    {
        "method": "FLARE-MCTS",
        "accuracy": 0.90,
        "latency": 193.71,
        "llm_calls": 152.55,
        "tokens": 684097,
        "note": "explicit future search",
    },
    {
        "method": "DiPLaN",
        "accuracy": 0.95,
        "latency": 8.62,
        "llm_calls": 5.05,
        "tokens": 39671,
        "note": "amortized future decoding",
    },
    {
        "method": "DiPLaN-lite",
        "accuracy": 0.83,
        "latency": 12.24,
        "llm_calls": 7.17,
        "tokens": None,
        "note": "lighter lexical-pruning variant",
    },
]


DEFAULT_GAP_SERIES = {
    "title": "Selection-to-Execution Gap vs Step (Illustrative)",
    "subtitle": "Replace with trace-derived values for the final paper figure",
    "steps": [1, 2, 3, 4, 5],
    "series": {
        "ToG": [11.0, 18.0, 25.0, 31.0, 36.0],
        "PoG": [9.0, 15.0, 21.0, 26.0, 30.0],
        "DiPLaN": [5.0, 8.0, 10.0, 12.0, 13.0],
    },
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
            "grid.alpha": 0.28,
            "grid.color": "#B8C2CC",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pick_summary(raw: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    if method:
        if method not in raw:
            raise KeyError(f"method '{method}' not found in summary; available keys: {list(raw)[:10]}")
        return raw[method]
    if "hits@1" in raw or "success_rate" in raw:
        return raw
    if len(raw) == 1:
        return raw[next(iter(raw))]
    raise ValueError(
        "Summary JSON contains multiple methods. Use LABEL|PATH|METHOD for --efficiency_summary."
    )


def _parse_spec(spec: str, allow_method: bool = False) -> tuple[str, Path, str | None]:
    parts = [p.strip() for p in spec.split("|")]
    if allow_method:
        if len(parts) not in {2, 3}:
            raise ValueError("Expected LABEL|PATH or LABEL|PATH|METHOD")
        label, path = parts[0], Path(parts[1])
        method = parts[2] if len(parts) == 3 and parts[2] else None
        return label, path, method
    if len(parts) != 2:
        raise ValueError("Expected LABEL|PATH")
    return parts[0], Path(parts[1]), None


def _num(x: Any, default: float | None = None) -> float | None:
    try:
        if x in (None, ""):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _first(summary: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in summary and summary[key] not in (None, ""):
            return summary[key]
    return default


def _safe_color(name: str) -> str:
    return COLORS.get(name, "#4C78A8")


def _safe_marker(name: str) -> str:
    return MARKERS.get(name, "o")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _load_funnel(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None:
        return dict(DEFAULT_FUNNEL)
    summary = _pick_summary(_load_json(summary_path))
    pool = 100.0 * float(_first(summary, ["answer_reaching_in_pool_rate"], DEFAULT_FUNNEL["values"][0] / 100.0))
    keep = 100.0 * float(_first(summary, ["answer_reaching_in_keep_rate"], DEFAULT_FUNNEL["values"][1] / 100.0))
    sel = 100.0 * float(_first(summary, ["answer_reaching_selected_rate"], DEFAULT_FUNNEL["values"][2] / 100.0))
    exe = 100.0 * float(
        _first(summary, ["answer_reaching_executed_top1_rate"], DEFAULT_FUNNEL["values"][3] / 100.0)
    )
    return {
        "title": "Execution Funnel",
        "subtitle": summary_path.name,
        "stages": list(DEFAULT_FUNNEL["stages"]),
        "values": [pool, keep, sel, exe],
    }


def _load_efficiency(specs: list[str]) -> list[dict[str, Any]]:
    if not specs:
        return [dict(row) for row in DEFAULT_EFFICIENCY_ROWS]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        label, path, method = _parse_spec(spec, allow_method=True)
        summary = _pick_summary(_load_json(path), method)
        rows.append(
            {
                "method": label,
                "accuracy": _num(_first(summary, ["hits@1", "success_rate"]), 0.0),
                "latency": _num(_first(summary, ["wall_time_s_per_task", "wall_time_s_mean"]), 0.0),
                "llm_calls": _num(_first(summary, ["llm_calls_per_task"]), 0.0),
                "tokens": _num(_first(summary, ["llm_total_tokens_est"]), None),
                "note": str(path.name),
            }
        )
    return rows


def _load_gap_series(specs: list[str]) -> dict[str, Any]:
    if not specs:
        return dict(DEFAULT_GAP_SERIES)
    series: dict[str, list[float]] = {}
    max_len = 0
    for spec in specs:
        label, path, _ = _parse_spec(spec, allow_method=False)
        rows = _load_jsonl(path)
        per_step_selected: list[list[float]] = []
        per_step_executed: list[list[float]] = []
        for row in rows:
            steps = row.get("steps") or row.get("episode_trace") or []
            if not isinstance(steps, list):
                continue
            for idx, step in enumerate(steps):
                if len(per_step_selected) <= idx:
                    per_step_selected.append([])
                    per_step_executed.append([])
                selected = step.get("dynamic_selected")
                executed = step.get("dynamic_executed_top1")
                if selected is None or executed is None:
                    continue
                per_step_selected[idx].append(1.0 if selected else 0.0)
                per_step_executed[idx].append(1.0 if executed else 0.0)
        gap_values: list[float] = []
        for sel_bucket, exe_bucket in zip(per_step_selected, per_step_executed):
            if not sel_bucket or not exe_bucket:
                gap_values.append(float("nan"))
                continue
            gap_values.append(100.0 * (mean(sel_bucket) - mean(exe_bucket)))
        max_len = max(max_len, len(gap_values))
        series[label] = gap_values
    steps = list(range(1, max_len + 1))
    for label, values in list(series.items()):
        if len(values) < max_len:
            series[label] = values + [float("nan")] * (max_len - len(values))
    return {
        "title": "Selection-to-Execution Gap vs Step",
        "subtitle": "Trace-derived from per-step dynamic reachability diagnostics",
        "steps": steps,
        "series": series,
    }


def plot_execution_funnel(funnel: dict[str, Any], out_dir: Path) -> None:
    stages = list(funnel["stages"])
    values = [float(v) for v in funnel["values"]]
    max_width = 110.0
    y = np.arange(len(stages))
    lefts = [(max_width - v) / 2.0 for v in values]
    colors = ["#D8E7F7", "#AFD0F3", "#6EA8FE", "#D64550"]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for idx, (stage, value, left) in enumerate(zip(stages, values, lefts)):
        ax.barh(idx, value, left=left, height=0.68, color=colors[idx], edgecolor="white", linewidth=1.5)
        ax.text(max_width / 2.0, idx, f"{stage}\n{value:.2f}%", ha="center", va="center", fontsize=11, color="#102A43")

    sel = values[-2]
    exe = values[-1]
    gap = sel - exe
    ax.annotate(
        f"Gap = {gap:.2f} pts",
        xy=(max_width / 2.0, len(stages) - 1),
        xytext=(max_width - 6, len(stages) - 1.55),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#C44536"},
        ha="right",
        va="center",
        fontsize=11,
        color="#C44536",
        fontweight="bold",
    )

    ax.set_title(str(funnel.get("title", "Execution Funnel")), pad=12)
    subtitle = str(funnel.get("subtitle", "")).strip()
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#52606D")
    ax.set_xlim(0, max_width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save_figure(fig, out_dir, "execution_funnel")


def _annotate_point(ax: plt.Axes, label: str, x: float, y: float, note: str = "") -> None:
    text = label if not note else f"{label}\n{note}"
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=9.5,
        color="#243B53",
    )


def plot_efficiency_scatter(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    metrics = [
        ("latency", "Latency per Task (s, log scale)", "efficiency_accuracy_vs_latency"),
        ("llm_calls", "LLM Calls per Task (log scale)", "efficiency_accuracy_vs_calls"),
    ]

    for ax, (metric, ylabel, _) in zip(axes, metrics):
        xs = []
        ys = []
        for row in rows:
            method = str(row["method"])
            x = 100.0 * float(row["accuracy"])
            y = max(1e-3, float(row[metric]))
            xs.append(x)
            ys.append(y)
            ax.scatter(
                x,
                y,
                s=140,
                marker=_safe_marker(method),
                color=_safe_color(method),
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            _annotate_point(ax, method, x, y, str(row.get("note", "")))

        ax.set_yscale("log")
        ax.grid(True, which="major", axis="both")
        ax.set_axisbelow(True)
        ax.set_xlabel("WebQSP Accuracy (Hits@1, %)")
        ax.set_ylabel(ylabel)
        ax.set_title("Higher quality at lower cost")

        if len(rows) >= 2:
            diplan = next((r for r in rows if str(r["method"]).lower() == "diplan"), None)
            flare = next((r for r in rows if "flare" in str(r["method"]).lower()), None)
            if diplan and flare:
                dx, dy = 100.0 * float(diplan["accuracy"]), max(1e-3, float(diplan[metric]))
                fx, fy = 100.0 * float(flare["accuracy"]), max(1e-3, float(flare[metric]))
                ax.annotate(
                    "",
                    xy=(dx, dy),
                    xytext=(fx, fy),
                    arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#7B8794", "linestyle": "--"},
                )

    fig.suptitle("Amortized Planning vs Explicit Search", y=1.02, fontsize=16)
    _save_figure(fig, out_dir, "efficiency_scatter")


def plot_gap_vs_step(gap_spec: dict[str, Any], out_dir: Path) -> None:
    steps = list(gap_spec["steps"])
    series = dict(gap_spec["series"])

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    for label, values in series.items():
        arr = np.array(values, dtype=float)
        ax.plot(
            steps,
            arr,
            label=label,
            color=_safe_color(label),
            marker=_safe_marker(label),
            linewidth=2.0,
            markersize=6,
        )

    ax.set_title(str(gap_spec.get("title", "Selection-to-Execution Gap vs Step")), pad=10)
    subtitle = str(gap_spec.get("subtitle", "")).strip()
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#52606D")
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Gap (Selection - Execution, pts)")
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.set_xticks(steps)
    ax.legend(frameon=False, loc="upper left")
    _save_figure(fig, out_dir, "gap_vs_step")


def _write_plot_data(
    out_dir: Path,
    funnel: dict[str, Any],
    efficiency_rows: list[dict[str, Any]],
    gap_spec: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "funnel": funnel,
        "efficiency": efficiency_rows,
        "gap_vs_step": gap_spec,
    }
    (out_dir / "plot_data_snapshot.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--funnel_summary",
        default="",
        help="Optional summary_metrics.json path used to fill the execution funnel.",
    )
    parser.add_argument(
        "--efficiency_summary",
        action="append",
        default=[],
        help="LABEL|PATH or LABEL|PATH|METHOD summary spec for efficiency scatter.",
    )
    parser.add_argument(
        "--gap_trace",
        action="append",
        default=[],
        help="LABEL|TRACE_JSONL spec. Each trace JSONL line should contain a `steps` list with dynamic_selected/dynamic_executed_top1.",
    )
    args = parser.parse_args()

    _apply_style()
    out_dir = Path(args.out_dir)
    funnel = _load_funnel(Path(args.funnel_summary)) if args.funnel_summary else _load_funnel(None)
    efficiency_rows = _load_efficiency(list(args.efficiency_summary))
    gap_spec = _load_gap_series(list(args.gap_trace))

    plot_execution_funnel(funnel, out_dir)
    plot_efficiency_scatter(efficiency_rows, out_dir)
    plot_gap_vs_step(gap_spec, out_dir)
    _write_plot_data(out_dir, funnel, efficiency_rows, gap_spec)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "files": [
                    str(out_dir / "execution_funnel.png"),
                    str(out_dir / "execution_funnel.pdf"),
                    str(out_dir / "efficiency_scatter.png"),
                    str(out_dir / "efficiency_scatter.pdf"),
                    str(out_dir / "gap_vs_step.png"),
                    str(out_dir / "gap_vs_step.pdf"),
                    str(out_dir / "plot_data_snapshot.json"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
