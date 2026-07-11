"""Draw a clean background/frame layout for the DiPLaN overview figure.

This script only renders the panel backgrounds, rounded boxes, section titles,
and major layout guides. It is meant to be used as a visual scaffold that you
can later fill with icons, equations, funnels, and other content.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = Path("result_paper/latex/figures")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str,
    face: str = "#ffffff",
    lw: float = 1.6,
    radius: float = 0.012,
    alpha: float = 1.0,
    dashed: bool = False,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    return patch


def label(ax, x: float, y: float, text: str, *, size: int = 11, weight: str = "normal", color: str = "#111111", ha: str = "center", va: str = "center") -> None:
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def draw_column(
    ax,
    x: float,
    w: float,
    title: str,
    title_color: str,
    border: str,
    blocks: list[tuple[float, str]],
    *,
    y_top: float = 0.90,
    y_bottom: float = 0.25,
) -> None:
    rounded_box(ax, x, y_bottom, w, y_top - y_bottom, edge=border, face="#fcfdff", lw=1.4, radius=0.01)
    label(ax, x + w / 2, y_top - 0.012, title, size=12, weight="bold", color=title_color)

    gap = 0.012
    inner_x = x + 0.018
    inner_w = w - 0.036
    current_y = y_top - 0.07

    for block_h, block_name in blocks:
        rounded_box(ax, inner_x, current_y - block_h, inner_w, block_h, edge=border, face="#ffffff", lw=1.0, radius=0.01)
        label(ax, inner_x + inner_w / 2, current_y - 0.018, block_name, size=10.5, weight="bold", color="#222222")
        current_y -= block_h + gap


def main() -> None:
    apply_style()

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    label(
        ax,
        0.5,
        0.965,
        "DiPLaN: Diagnosing and Closing the Selection-to-Execution Gap",
        size=28,
        weight="bold",
    )

    # Main four columns
    col_y_bottom = 0.25
    col_y_top = 0.90
    col_w = 0.225
    gap = 0.015
    xs = [0.01, 0.01 + col_w + gap, 0.01 + 2 * (col_w + gap), 0.01 + 3 * (col_w + gap)]

    draw_column(
        ax,
        xs[0],
        col_w,
        "(1) Executor-Confinement Setting",
        "#214b9b",
        "#8fb0ea",
        [
            (0.14, "Current state  $s_t$"),
            (0.23, "Legal action set  $A(s_t)$"),
            (0.16, "Environment / Executor"),
        ],
        y_top=col_y_top,
        y_bottom=col_y_bottom,
    )

    draw_column(
        ax,
        xs[1],
        col_w,
        "(2) Selection-to-Execution Diagnostic",
        "#214b9b",
        "#9eb5df",
        [
            (0.52, "Diagnostic Funnel"),
            (0.12, "Key diagnostic notes"),
        ],
        y_top=col_y_top,
        y_bottom=col_y_bottom,
    )

    draw_column(
        ax,
        xs[2],
        col_w,
        "(3) Offline Amortization (Training Once)",
        "#2a6e37",
        "#a6d3af",
        [
            (0.16, "Training Sources"),
            (0.30, "Train Modules"),
            (0.05, "Compact reusable modules"),
        ],
        y_top=col_y_top,
        y_bottom=col_y_bottom,
    )

    draw_column(
        ax,
        xs[3],
        col_w,
        "(4) Inference-Time Decoding",
        "#5a3caa",
        "#c7b2ea",
        [
            (0.20, "Score & Rank"),
            (0.05, "Select top-1 action"),
            (0.14, "Execute"),
            (0.09, "New state / Replan"),
            (0.05, "Repeat until terminal condition"),
        ],
        y_top=col_y_top,
        y_bottom=col_y_bottom,
    )

    # Inter-column arrows
    arrow_y = 0.58
    for i in range(3):
        x0 = xs[i] + col_w
        x1 = xs[i + 1]
        ax.annotate(
            "",
            xy=(x1 - 0.004, arrow_y),
            xytext=(x0 + 0.004, arrow_y),
            arrowprops=dict(arrowstyle="simple", color="#111111", lw=0.8, shrinkA=0, shrinkB=0),
        )

    # Bottom band
    rounded_box(ax, 0.01, 0.04, 0.98, 0.16, edge="#efc98c", face="#fffdfa", lw=1.4, radius=0.012)
    label(ax, 0.5, 0.185, "(5) Efficiency & Amortization Benefits (Heavy-Search Budget)", size=14, weight="bold", color="#946200")

    # Bottom band sections
    bottom_y = 0.06
    bottom_h = 0.11
    left_w = 0.27
    center_w = 0.20
    right_w = 0.27
    note_w = 0.13

    rounded_box(ax, 0.025, bottom_y, left_w, bottom_h, edge="#dcc9f2", face="#ffffff", lw=1.0, radius=0.01)
    label(ax, 0.025 + left_w / 2, bottom_y + bottom_h - 0.02, "Online search baseline: MCTS", size=11, weight="bold", color="#6d43b1")

    rounded_box(ax, 0.325, bottom_y, center_w, bottom_h, edge="#f2d28a", face="#fffef7", lw=1.0, radius=0.01)
    label(ax, 0.325 + center_w / 2, bottom_y + bottom_h / 2, "Heavy-search budget:\n≈30× fewer LLM calls", size=13, weight="bold", color="#a57600")

    rounded_box(ax, 0.555, bottom_y, right_w, bottom_h, edge="#b7d0ef", face="#ffffff", lw=1.0, radius=0.01)
    label(ax, 0.555 + right_w / 2, bottom_y + bottom_h - 0.02, "DiPLaN (ours)", size=11, weight="bold", color="#295fb5")

    rounded_box(ax, 0.85, bottom_y, note_w, bottom_h, edge="#cfcfcf", face="#ffffff", lw=1.0, radius=0.01)
    label(ax, 0.85 + note_w / 2, bottom_y + bottom_h / 2, "Takeaway box", size=11, weight="bold", color="#444444")

    # Footer hint
    label(
        ax,
        0.5,
        0.012,
        "Template background only: fill with funnels, icons, equations, and curves as needed.",
        size=10,
        color="#444444",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "diplan_framework_background.png")
    fig.savefig(OUT_DIR / "diplan_framework_background.pdf")
    plt.close(fig)
    print(f"[ok] wrote background layout to {OUT_DIR}")


if __name__ == "__main__":
    main()
