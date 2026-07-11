"""Plot a BEACON-style small gap-dynamics figure for DiPLaN.

This script draws the left-panel style figure the user asked for:

- x-axis: epoch
- left y-axis: selection / execution / success
- right y-axis: loss
- annotations: peak gap and time lag

Input CSV schema
----------------
epoch,selection_rate,execution_rate,success_rate,val_loss

All rate columns should be in [0, 1].

Examples
--------
Use demo data to preview the figure style:
    python scripts/plot_gap_dynamics_small_figure.py \
      --out_dir results/gap_small_demo

Use your real checkpoint table:
    python scripts/plot_gap_dynamics_small_figure.py \
      --csv results/checkpoint_gap_curve/gap_over_epochs.csv \
      --title "WebQSP Gap Dynamics" \
      --out_dir results/checkpoint_gap_curve
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.linewidth": 1.4,
            "grid.alpha": 0.25,
            "grid.color": "#B8C2CC",
            "savefig.bbox": "tight",
            "savefig.dpi": 240,
        }
    )


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x in (None, ""):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _load_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": _safe_float(row.get("epoch")),
                    "selection_rate": _safe_float(row.get("selection_rate")),
                    "execution_rate": _safe_float(row.get("execution_rate")),
                    "success_rate": _safe_float(row.get("success_rate")),
                    "val_loss": _safe_float(row.get("val_loss")),
                }
            )
    return rows


def _demo_rows() -> list[dict[str, float]]:
    return [
        {"epoch": 1, "selection_rate": 0.06, "execution_rate": 0.02, "success_rate": 0.02, "val_loss": 3.30},
        {"epoch": 2, "selection_rate": 0.11, "execution_rate": 0.03, "success_rate": 0.03, "val_loss": 3.05},
        {"epoch": 3, "selection_rate": 0.17, "execution_rate": 0.05, "success_rate": 0.05, "val_loss": 2.72},
        {"epoch": 4, "selection_rate": 0.28, "execution_rate": 0.08, "success_rate": 0.08, "val_loss": 2.34},
        {"epoch": 5, "selection_rate": 0.41, "execution_rate": 0.12, "success_rate": 0.12, "val_loss": 1.93},
        {"epoch": 6, "selection_rate": 0.57, "execution_rate": 0.17, "success_rate": 0.17, "val_loss": 1.58},
        {"epoch": 7, "selection_rate": 0.71, "execution_rate": 0.24, "success_rate": 0.24, "val_loss": 1.24},
        {"epoch": 8, "selection_rate": 0.83, "execution_rate": 0.31, "success_rate": 0.30, "val_loss": 0.98},
        {"epoch": 9, "selection_rate": 0.90, "execution_rate": 0.39, "success_rate": 0.38, "val_loss": 0.77},
        {"epoch": 10, "selection_rate": 0.94, "execution_rate": 0.46, "success_rate": 0.45, "val_loss": 0.62},
        {"epoch": 11, "selection_rate": 0.96, "execution_rate": 0.54, "success_rate": 0.53, "val_loss": 0.51},
        {"epoch": 12, "selection_rate": 0.97, "execution_rate": 0.61, "success_rate": 0.60, "val_loss": 0.43},
        {"epoch": 13, "selection_rate": 0.98, "execution_rate": 0.68, "success_rate": 0.67, "val_loss": 0.37},
        {"epoch": 14, "selection_rate": 0.98, "execution_rate": 0.74, "success_rate": 0.73, "val_loss": 0.32},
        {"epoch": 15, "selection_rate": 0.99, "execution_rate": 0.79, "success_rate": 0.78, "val_loss": 0.29},
        {"epoch": 16, "selection_rate": 0.99, "execution_rate": 0.83, "success_rate": 0.82, "val_loss": 0.26},
        {"epoch": 17, "selection_rate": 0.99, "execution_rate": 0.86, "success_rate": 0.85, "val_loss": 0.24},
        {"epoch": 18, "selection_rate": 0.99, "execution_rate": 0.88, "success_rate": 0.87, "val_loss": 0.22},
    ]


def _find_saturation_epoch(values: np.ndarray, threshold: float) -> int | None:
    idx = np.where(values >= threshold)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def _plot(rows: list[dict[str, float]], title: str, subtitle: str, out_dir: Path, stem: str) -> None:
    epochs = np.array([r["epoch"] for r in rows], dtype=float)
    selection = np.array([r["selection_rate"] for r in rows], dtype=float)
    execution = np.array([r["execution_rate"] for r in rows], dtype=float)
    success = np.array([r["success_rate"] for r in rows], dtype=float)
    loss = np.array([r["val_loss"] for r in rows], dtype=float)
    gap = selection - execution

    peak_gap_idx = int(np.argmax(gap))
    peak_epoch = epochs[peak_gap_idx]
    peak_sel = selection[peak_gap_idx]
    peak_exe = execution[peak_gap_idx]
    peak_gap = gap[peak_gap_idx]

    sel_sat_idx = _find_saturation_epoch(selection, threshold=0.95)
    exe_sat_idx = _find_saturation_epoch(execution, threshold=0.75)
    time_lag = None
    if sel_sat_idx is not None and exe_sat_idx is not None:
        time_lag = epochs[exe_sat_idx] - epochs[sel_sat_idx]

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax2 = ax1.twinx()

    line_sel = ax1.plot(
        epochs,
        selection,
        color="#2B8CBE",
        marker="o",
        linewidth=2.3,
        markersize=7,
        label="Selection",
    )[0]
    line_exe = ax1.plot(
        epochs,
        execution,
        color="#A23B72",
        marker="s",
        linewidth=2.3,
        markersize=6.5,
        label="Execution",
    )[0]
    line_suc = ax1.plot(
        epochs,
        success,
        color="#7A4EAB",
        marker="^",
        linewidth=1.9,
        markersize=6.0,
        linestyle=":",
        label="Success",
    )[0]
    line_loss = ax2.plot(
        epochs,
        loss,
        color="#9AA08F",
        linewidth=2.0,
        linestyle="--",
        label="Loss",
    )[0]

    ax1.set_xlim(float(np.min(epochs)) - 0.5, float(np.max(epochs)) + 0.8)
    ax1.set_ylim(0.0, 1.08)
    ax2.set_ylim(0.0, max(0.1, float(np.max(loss)) * 1.08))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rate")
    ax2.set_ylabel("Loss", color="#556B2F", fontweight="bold")
    ax1.grid(True, axis="both")

    ax1.annotate(
        "",
        xy=(peak_epoch, peak_sel),
        xytext=(peak_epoch, peak_exe),
        arrowprops={"arrowstyle": "<->", "lw": 2.0, "color": "black"},
    )
    ax1.text(
        peak_epoch + 0.35,
        0.5 * (peak_sel + peak_exe),
        f"Gap = {peak_gap:.2f}",
        fontsize=11,
        fontweight="bold",
        va="center",
    )

    if sel_sat_idx is not None:
        ax1.scatter([epochs[sel_sat_idx]], [selection[sel_sat_idx]], color="#2B8CBE", s=70, zorder=5)
        ax1.text(
            epochs[sel_sat_idx] - 0.2,
            min(1.04, selection[sel_sat_idx] + 0.05),
            "Selection saturates",
            color="#2B8CBE",
            fontsize=10,
            ha="left",
        )
    if exe_sat_idx is not None:
        ax1.scatter([epochs[exe_sat_idx]], [execution[exe_sat_idx]], color="#A23B72", s=70, zorder=5)
        ax1.text(
            epochs[exe_sat_idx] - 0.2,
            execution[exe_sat_idx] - 0.11,
            "Execution catches up",
            color="#A23B72",
            fontsize=10,
            ha="left",
        )
    if time_lag is not None and sel_sat_idx is not None and exe_sat_idx is not None:
        y_arrow = 0.18
        ax1.annotate(
            "",
            xy=(epochs[sel_sat_idx], y_arrow),
            xytext=(epochs[exe_sat_idx], y_arrow),
            arrowprops={"arrowstyle": "<->", "lw": 1.8, "color": "black"},
        )
        ax1.text(
            0.5 * (epochs[sel_sat_idx] + epochs[exe_sat_idx]),
            y_arrow + 0.03,
            f"Time Lag = {time_lag:.0f} epochs",
            ha="center",
            fontsize=10.5,
        )

    ax1.set_title(title, pad=10, fontweight="bold")
    if subtitle.strip():
        ax1.text(
            0.5,
            1.01,
            subtitle,
            transform=ax1.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#52606D",
        )

    ax1.text(
        epochs[-1] - 0.5,
        min(1.03, selection[-1] + 0.03),
        f"{selection[-1]:.2f}",
        color="#2B8CBE",
        fontsize=12,
        fontweight="bold",
    )
    ax1.text(
        epochs[-1] - 0.5,
        execution[-1] + 0.03,
        f"{execution[-1]:.2f}",
        color="#A23B72",
        fontsize=12,
        fontweight="bold",
    )

    lines = [line_sel, line_exe, line_suc, line_loss]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper center", ncol=4, frameon=True, fancybox=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)

    snapshot = {
        "title": title,
        "subtitle": subtitle,
        "peak_gap_epoch": float(peak_epoch),
        "peak_gap": float(peak_gap),
        "time_lag_epochs": None if time_lag is None else float(time_lag),
        "rows": rows,
    }
    (out_dir / f"{stem}_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--title", type=str, default="Selection-to-Execution Gap Dynamics")
    parser.add_argument("--subtitle", type=str, default="Selection rises earlier than execution under training")
    parser.add_argument("--out_dir", type=str, default="results/gap_small_demo")
    parser.add_argument("--stem", type=str, default="gap_dynamics_small")
    args = parser.parse_args()

    _apply_style()
    rows = _load_csv(Path(args.csv)) if args.csv else _demo_rows()
    _plot(rows, title=str(args.title), subtitle=str(args.subtitle), out_dir=Path(args.out_dir), stem=str(args.stem))
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).resolve()),
                "files": [
                    str((Path(args.out_dir) / f"{args.stem}.png").resolve()),
                    str((Path(args.out_dir) / f"{args.stem}.pdf").resolve()),
                    str((Path(args.out_dir) / f"{args.stem}_snapshot.json").resolve()),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
