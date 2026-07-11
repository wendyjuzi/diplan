"""Generate paper figures from real planning predictions.

This script replaces the old demo-only plotting template. It reads real
`predictions.jsonl` outputs and computes the panel figures directly from
experiment records, so the exported figures can be used in the paper without
hand-written placeholder values.

Important: pass an explicit paper-grade `predictions.jsonl` via `--predictions`.
Do not rely on toy or smoke-test runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "result_paper" / "latex" / "figures"
COLORS = {
    "Single Step": "#F09A5A",
    "Beam Search": "#E31A1C",
    "Lookahead": "#7A3C8C",
    "DiPLaN": "#2A6F97",
}

METHOD_NAME_MAP = {
    "single_step": "Single Step",
    "beam": "Beam Search",
    "beam_search": "Beam Search",
    "lookahead": "Lookahead",
    "diplan": "DiPLaN",
    "diplan_diffusion": "DiPLaN",
}

PLOT_METHODS = ["Single Step", "Beam Search", "Lookahead"]
DATASETS = ["cwq", "webqsp"]
DATASET_TITLES = {"cwq": "CWQ", "webqsp": "WebQSP"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.6,
            "grid.alpha": 0.35,
            "grid.color": "#B9C2D0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            method = METHOD_NAME_MAP.get(str(row.get("method", "")).lower())
            dataset = str(row.get("dataset", "")).lower()
            if method not in PLOT_METHODS or dataset not in DATASETS:
                continue
            row["_plot_method"] = method
            row["_dataset"] = dataset
            row["_oracle_len"] = len(row.get("oracle_path") or [])
            row["_first_error"] = int(row.get("first_error_step", 0) or 0)
            row["_success"] = bool(row.get("success"))
            row["_trap_at_1"] = bool(row.get("trap_at_1"))
            executed = row.get("executed_path") or []
            oracle = row.get("oracle_path") or []
            row["_prefix_match_len"] = _prefix_match_len(executed, oracle)
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return rows


def _prefix_match_len(executed: list[str], oracle: list[str]) -> int:
    n = min(len(executed), len(oracle))
    for i in range(n):
        if executed[i] != oracle[i]:
            return i
    return n


def _group(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["_dataset"], row["_plot_method"])].append(row)
    return grouped


def _horizon_curve(rows: list[dict], horizon_bins: list[int]) -> dict[str, list[float]]:
    grouped = _group(rows)
    curves: dict[str, list[float]] = {}
    for dataset in DATASETS:
        for method in PLOT_METHODS:
            subset = grouped.get((dataset, method), [])
            vals = []
            for h in horizon_bins:
                bucket = [r for r in subset if r["_oracle_len"] == h]
                vals.append(100.0 * mean(1.0 if r["_success"] else 0.0 for r in bucket) if bucket else np.nan)
            curves[f"{dataset}::{method}"] = vals
    return curves


def _bar_metrics(rows: list[dict]) -> dict[str, list[float]]:
    grouped = _group(rows)
    out: dict[str, list[float]] = {}
    for dataset in DATASETS:
        for method in PLOT_METHODS:
            subset = grouped.get((dataset, method), [])
            if not subset:
                out[f"{dataset}::{method}"] = [np.nan, np.nan, np.nan]
                continue
            trap = 100.0 * mean(1.0 if r["_trap_at_1"] else 0.0 for r in subset)
            first_error_step1 = 100.0 * mean(1.0 if r["_first_error"] == 1 else 0.0 for r in subset)
            recovery = 100.0 * mean(1.0 if bool(r.get("recovery_at_error")) else 0.0 for r in subset)
            out[f"{dataset}::{method}"] = [trap, first_error_step1, recovery]
    return out


def _prefix_curve(rows: list[dict], max_step: int) -> dict[str, list[float]]:
    grouped = _group(rows)
    curves: dict[str, list[float]] = {}
    for dataset in DATASETS:
        for method in PLOT_METHODS:
            subset = grouped.get((dataset, method), [])
            vals = []
            for step in range(1, max_step + 1):
                rate = mean(1.0 if r["_prefix_match_len"] >= step else 0.0 for r in subset) if subset else np.nan
                vals.append(100.0 * rate)
            curves[f"{dataset}::{method}"] = vals
    return curves


def _recovery_curve(rows: list[dict], max_error_step: int) -> dict[str, list[float]]:
    grouped = _group(rows)
    curves: dict[str, list[float]] = {}
    for dataset in DATASETS:
        for method in PLOT_METHODS:
            subset = grouped.get((dataset, method), [])
            vals = []
            for pos in range(1, max_error_step + 1):
                bucket = [r for r in subset if r["_first_error"] == pos]
                vals.append(100.0 * mean(1.0 if r["_success"] else 0.0 for r in bucket) if bucket else np.nan)
            curves[f"{dataset}::{method}"] = vals
    return curves


def plot_line_panel(
    ax: plt.Axes,
    x,
    series: dict[str, list[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str] | None = None,
    legend_loc: str = "best",
) -> None:
    markers = {
        "Single Step": "o",
        "Beam Search": "s",
        "Lookahead": "D",
    }
    for name, values in series.items():
        ax.plot(
            x,
            values,
            label=name,
            color=COLORS[name],
            marker=markers.get(name, "o"),
            linewidth=1.6,
            markersize=5,
        )
    ax.set_title(title, pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
    ax.legend(frameon=False, loc=legend_loc)


def plot_grouped_bar_panel(
    ax: plt.Axes,
    categories: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    legend_loc: str = "best",
) -> None:
    x = np.arange(len(categories))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(series))
    for offset, (name, values) in zip(offsets, series.items()):
        ax.bar(x + offset, values, width=width, label=name, color=COLORS[name])
    ax.set_title(title, pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc=legend_loc)


def make_figure1(rows: list[dict], out_dir: Path) -> None:
    horizon_bins = [2, 3, 4, 5]
    horizon_labels = [str(x) for x in horizon_bins]
    curves = _horizon_curve(rows, horizon_bins)
    bars = _bar_metrics(rows)
    categories = ["Trap select\n@step1", "First error\nat step1", "Recovery @\nfirst error"]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))

    for col, dataset in enumerate(DATASETS):
        series = {m: curves[f"{dataset}::{m}"] for m in PLOT_METHODS}
        plot_line_panel(
            axes[0, col],
            np.arange(len(horizon_bins)),
            series,
            title=DATASET_TITLES[dataset],
            xlabel="Required Planning Horizon",
            ylabel="Performance (Hits@1, %)",
            xticklabels=horizon_labels,
            legend_loc="lower left",
        )
        bar_series = {m: bars[f"{dataset}::{m}"] for m in PLOT_METHODS}
        plot_grouped_bar_panel(
            axes[1, col],
            categories,
            bar_series,
            title=DATASET_TITLES[dataset],
            ylabel="Rate (%)",
            legend_loc="upper right",
        )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "panel_figure1_real.png")
    fig.savefig(out_dir / "panel_figure1_real.pdf")
    plt.close(fig)


def make_figure2(rows: list[dict], out_dir: Path) -> None:
    max_step_by_dataset = {
        dataset: max((r["_oracle_len"] for r in rows if r["_dataset"] == dataset), default=1)
        for dataset in DATASETS
    }
    prefix = _prefix_curve(rows, max(max_step_by_dataset.values()))
    recovery = _recovery_curve(rows, max(max_step_by_dataset.values()))

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))

    for col, dataset in enumerate(DATASETS):
        steps = list(range(1, max_step_by_dataset[dataset] + 1))
        step_labels = [str(s) for s in steps]
        prefix_series = {m: prefix[f"{dataset}::{m}"][: len(steps)] for m in PLOT_METHODS}
        recovery_series = {m: recovery[f"{dataset}::{m}"][: len(steps)] for m in PLOT_METHODS}
        plot_line_panel(
            axes[0, col],
            np.arange(len(steps)),
            prefix_series,
            title=DATASET_TITLES[dataset],
            xlabel="Step $t$",
            ylabel="Still-Correct Prefix (%)",
            xticklabels=step_labels,
            legend_loc="upper right",
        )
        plot_line_panel(
            axes[1, col],
            np.arange(len(steps)),
            recovery_series,
            title=DATASET_TITLES[dataset],
            xlabel="First-Error Position",
            ylabel="Success (%) After First-Error",
            xticklabels=step_labels,
            legend_loc="upper left",
        )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "panel_figure2_real.png")
    fig.savefig(out_dir / "panel_figure2_real.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper panels from real predictions.")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to a paper-grade predictions.jsonl file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(OUT_DIR),
        help="Directory for exported figures.",
    )
    args = parser.parse_args()

    _apply_style()
    rows = _load_rows(Path(args.predictions))
    out_dir = Path(args.out_dir)
    make_figure1(rows, out_dir)
    make_figure2(rows, out_dir)
    print(f"[ok] wrote real figures to {out_dir}")


if __name__ == "__main__":
    main()
