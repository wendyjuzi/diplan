"""Plot BEACON-style stacked proportion panels.

This script generates the multi-panel stacked bar chart style shown in the
example figure: one panel per method, one stacked bar per training step, and
percentage labels inside each colored segment.

Usage
-----
Use the built-in demo data:
    python scripts/plot_beacon_style_stacked_panels.py \
      --out_dir result_paper/latex/figures/beacon_style

Use a custom JSON spec:
    python scripts/plot_beacon_style_stacked_panels.py \
      --spec path/to/stacked_panels.json \
      --out_dir result_paper/latex/figures/beacon_style

JSON format
-----------
{
  "ylabel": "Proportion (%)",
  "xlabel": "Training Steps",
  "panels": [
    {
      "title": "Method A",
      "x_labels": ["50", "100", "150"],
      "stacks": [
        {"name": "Solved", "color": "#43b56b", "values": [21, 46, 77]},
        {"name": "Partial", "color": "#faad2f", "values": [46, 36, 13]},
        {"name": "Failed", "color": "#c4cbd1", "values": [33, 18, 10]}
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "result_paper" / "latex" / "figures" / "beacon_style"


DEFAULT_SPEC = {
    "ylabel": "Proportion (%)",
    "xlabel": "Training Steps",
    "figure_title": "",
    "panels": [
        {
            "title": "GRPO",
            "x_labels": ["50", "100", "150"],
            "stacks": [
                {"name": "Good", "color": "#45b36b", "values": [4, 27, 23]},
                {"name": "Medium", "color": "#f6ab2f", "values": [47, 39, 39]},
                {"name": "Bad", "color": "#c8cfd4", "values": [49, 34, 38]},
            ],
        },
        {
            "title": "GiGPO",
            "x_labels": ["50", "100", "150"],
            "stacks": [
                {"name": "Good", "color": "#45b36b", "values": [8, 46, 50]},
                {"name": "Medium", "color": "#f6ab2f", "values": [61, 38, 28]},
                {"name": "Bad", "color": "#c8cfd4", "values": [31, 16, 22]},
            ],
        },
        {
            "title": "BEACON (Ours)",
            "x_labels": ["50", "100", "150"],
            "stacks": [
                {"name": "Good", "color": "#45b36b", "values": [21, 46, 77]},
                {"name": "Medium", "color": "#f6ab2f", "values": [46, 36, 13]},
                {"name": "Bad", "color": "#c8cfd4", "values": [33, 18, 10]},
            ],
        },
    ],
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def _load_spec(path: str) -> dict[str, Any]:
    if not path:
        return json.loads(json.dumps(DEFAULT_SPEC))
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_spec(spec: dict[str, Any]) -> None:
    panels = spec.get("panels") or []
    if not panels:
        raise ValueError("Spec must contain a non-empty 'panels' list.")
    for panel in panels:
        x_labels = panel.get("x_labels") or []
        stacks = panel.get("stacks") or []
        if not x_labels:
            raise ValueError(f"Panel '{panel.get('title', '')}' is missing x_labels.")
        if not stacks:
            raise ValueError(f"Panel '{panel.get('title', '')}' is missing stacks.")
        n = len(x_labels)
        for stack in stacks:
            vals = stack.get("values") or []
            if len(vals) != n:
                raise ValueError(
                    f"Panel '{panel.get('title', '')}', stack '{stack.get('name', '')}' "
                    f"has {len(vals)} values but expected {n}."
                )


def _label_color(fill_color: str) -> str:
    dark_fills = {"#45b36b", "#43b56b", "#1f6feb", "#2aa198", "#6ab04c", "#3cb371"}
    return "white" if fill_color.lower() in dark_fills or fill_color.lower().startswith("#4") else "white"


def _plot_panel(
    ax: plt.Axes,
    panel: dict[str, Any],
    *,
    show_ylabel: bool,
    ylabel: str,
    xlabel: str,
) -> None:
    x_labels = list(panel["x_labels"])
    stacks = list(panel["stacks"])
    x = np.arange(len(x_labels))
    width = 0.64
    bottom = np.zeros(len(x_labels), dtype=float)

    for stack in stacks:
        vals = np.array(stack["values"], dtype=float)
        color = str(stack["color"])
        bars = ax.bar(
            x,
            vals,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor="#3E4C59",
            linewidth=1.1,
        )
        for bar, v, b in zip(bars, vals, bottom):
            if v < 9:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                b + v / 2.0,
                f"{int(round(v))}%",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=_label_color(color),
            )
        bottom += vals

    ax.set_title(str(panel.get("title", "")), pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(False)


def plot_stacked_panels(spec: dict[str, Any], out_dir: Path) -> None:
    _validate_spec(spec)
    panels = list(spec["panels"])
    ylabel = str(spec.get("ylabel", "Proportion (%)"))
    xlabel = str(spec.get("xlabel", "Training Steps"))
    figure_title = str(spec.get("figure_title", "")).strip()

    fig_width = max(10.5, 3.8 * len(panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 3.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        _plot_panel(
            ax,
            panel,
            show_ylabel=(idx == 0),
            ylabel=ylabel,
            xlabel=xlabel,
        )

    if figure_title:
        fig.suptitle(figure_title, y=1.03, fontsize=16, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "beacon_style_stacked_panels.png")
    fig.savefig(out_dir / "beacon_style_stacked_panels.pdf")
    plt.close(fig)

    (out_dir / "beacon_style_stacked_panels.spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="", help="Optional JSON spec for the stacked panels.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    _apply_style()
    spec = _load_spec(args.spec)
    out_dir = Path(args.out_dir)
    plot_stacked_panels(spec, out_dir)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "files": [
                    str(out_dir / "beacon_style_stacked_panels.png"),
                    str(out_dir / "beacon_style_stacked_panels.pdf"),
                    str(out_dir / "beacon_style_stacked_panels.spec.json"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
