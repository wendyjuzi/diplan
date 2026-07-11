"""Run WebQSP ablations and summarize strict-oracle gap reduction.

This script wraps ``evaluate_torch.py`` in a paper-oriented ablation pipeline:

1. Run each ablation config on WebQSP.
2. Read ``predictions.jsonl`` and compute strict-oracle selection/execution metrics.
3. Export a gap-reduction table in CSV/Markdown/JSON.
4. Plot a compact figure for the paper.

Important terminology
---------------------
The metrics here are **strict-oracle** diagnostics derived from ``evaluate_torch.py``:

- oracle_pool_recall: oracle path appears in candidate pool
- oracle_selection_rate: oracle path ranked at top-1
- oracle_execution_rate: executed path exactly matches oracle path
- oracle_gap: selection_rate - execution_rate

This is stricter than the answer-reaching funnel used in the patched official ToG
pipeline. The naming deliberately keeps the ``oracle_`` prefix to avoid mixing the
two diagnostic families.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_json, load_jsonl


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.3,
            "grid.alpha": 0.25,
            "grid.color": "#B8C2CC",
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def _pick_summary(raw: dict[str, Any]) -> dict[str, Any]:
    if "success_rate" in raw:
        return raw
    if len(raw) == 1:
        return raw[next(iter(raw))]
    raise ValueError(f"Unexpected summary_metrics.json format: keys={list(raw)[:10]}")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x in (None, ""):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _fmt_pct(x: Any) -> str:
    return f"{100.0 * _safe_float(x):.2f}%"


def _fmt_float(x: Any, digits: int = 4) -> str:
    return f"{_safe_float(x):.{digits}f}"


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Setting | Success | Oracle Pool | Oracle Selection | Oracle Execution | Oracle Gap | Gap vs Full | Plan Feas. | trap@1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gap_delta = row.get("oracle_gap_delta_vs_anchor")
        gap_delta_str = ""
        if gap_delta is not None:
            gap_delta_str = f"{100.0 * _safe_float(gap_delta):+.2f} pts"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("label", row.get("name", ""))),
                    _fmt_pct(row.get("success_rate")),
                    _fmt_pct(row.get("oracle_pool_recall")),
                    _fmt_pct(row.get("oracle_selection_rate")),
                    _fmt_pct(row.get("oracle_execution_rate")),
                    f"{100.0 * _safe_float(row.get('oracle_gap')):.2f} pts",
                    gap_delta_str,
                    _fmt_pct(row.get("plan_feasibility")),
                    _fmt_pct(row.get("trap_at_1")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_eval(
    *,
    eval_cfg_path: Path,
    ae_ckpt: str,
    planner_ckpt: str,
    value_ckpt: str,
    out_dir: Path,
    constraint_ckpt: str = "",
) -> None:
    cmd = [
        sys.executable,
        "evaluate_torch.py",
        "--config",
        str(eval_cfg_path),
        "--ae_ckpt",
        ae_ckpt,
        "--planner_ckpt",
        planner_ckpt,
        "--value_ckpt",
        value_ckpt,
        "--out",
        str(out_dir),
    ]
    if constraint_ckpt:
        cmd.extend(["--constraint_ckpt", constraint_ckpt])
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _strict_gap_from_predictions(pred_path: Path) -> dict[str, Any]:
    rows = load_jsonl(str(pred_path))
    if not rows:
        return {
            "n": 0,
            "oracle_pool_recall": 0.0,
            "oracle_selection_rate": 0.0,
            "oracle_execution_rate": 0.0,
            "oracle_gap": 0.0,
            "conditional_execution_given_selected": 0.0,
        }
    pool = mean(1.0 if r.get("oracle_in_candidate_pool", False) else 0.0 for r in rows)
    selection = mean(_safe_float(r.get("oracle_hit_at_1", 0.0)) for r in rows)
    execution = mean(1.0 if r.get("success", False) else 0.0 for r in rows)
    return {
        "n": len(rows),
        "oracle_pool_recall": pool,
        "oracle_selection_rate": selection,
        "oracle_execution_rate": execution,
        "oracle_gap": selection - execution,
        "conditional_execution_given_selected": execution / max(selection, 1e-9),
    }


def _load_dataset_summary(run_dir: Path, dataset: str) -> dict[str, Any]:
    ds_path = run_dir / "summary_by_dataset.json"
    if ds_path.exists():
        raw = load_json(str(ds_path))
        if dataset in raw:
            return raw[dataset]
    raw = load_json(str(run_dir / "summary_metrics.json"))
    return _pick_summary(raw)


def _plot_gap_figure(rows: list[dict[str, Any]], anchor_name: str, out_dir: Path) -> None:
    _apply_style()
    labels = [str(r["label"]) for r in rows]
    selection = np.array([100.0 * _safe_float(r["oracle_selection_rate"]) for r in rows], dtype=float)
    execution = np.array([100.0 * _safe_float(r["oracle_execution_rate"]) for r in rows], dtype=float)
    gaps = np.array([100.0 * _safe_float(r["oracle_gap"]) for r in rows], dtype=float)

    x = np.arange(len(rows))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    axes[0].bar(x - width / 2.0, selection, width=width, color="#6EA8FE", label="Oracle Selection")
    axes[0].bar(x + width / 2.0, execution, width=width, color="#45B36B", label="Oracle Execution")
    for xi, s, e in zip(x, selection, execution):
        axes[0].text(xi - width / 2.0, s + 1.0, f"{s:.1f}", ha="center", va="bottom", fontsize=9)
        axes[0].text(xi + width / 2.0, e + 1.0, f"{e:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Strict-Oracle Selection vs Execution")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].grid(True, axis="y")
    axes[0].legend(frameon=False, loc="upper right")

    colors = ["#1F6FEB" if str(r["name"]) == anchor_name else "#C44536" for r in rows]
    axes[1].bar(x, gaps, color=colors)
    for xi, g, row in zip(x, gaps, rows):
        delta = row.get("oracle_gap_delta_vs_anchor")
        txt = f"{g:.1f}"
        if delta is not None:
            txt += f"\n({100.0 * _safe_float(delta):+.1f})"
        axes[1].text(xi, g + 0.8, txt, ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Gap Reduction Relative to Full DiPLaN")
    axes[1].set_ylabel("Oracle Gap (pts)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(True, axis="y")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "webqsp_ablation_gap_reduction.png")
    fig.savefig(out_dir / "webqsp_ablation_gap_reduction.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation_torch_webqsp_gap.yaml")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--constraint_ckpt", type=str, default="")
    parser.add_argument("--out", type=str, default="results/ablation_webqsp_gap")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_eval = load_config(cfg["base_eval_config"])
    dataset = str(cfg.get("dataset", "webqsp")).lower()
    anchor_name = str(cfg.get("anchor_experiment", ""))

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    tmp_dir = out_dir / "_tmp_cfgs"
    ensure_dir(str(tmp_dir))

    rows: list[dict[str, Any]] = []
    for exp in cfg["experiments"]:
        name = str(exp["name"])
        label = str(exp.get("label", name))
        eval_cfg = dict(base_eval)
        eval_cfg["include_datasets"] = [dataset]
        eval_cfg.update(exp.get("overrides", {}))
        exp_cfg_path = tmp_dir / f"{name}.json"
        exp_cfg_path.write_text(json.dumps(eval_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        exp_out = out_dir / name
        pred_path = exp_out / "predictions.jsonl"
        if not (args.skip_existing and pred_path.exists() and (exp_out / "summary_metrics.json").exists()):
            ensure_dir(str(exp_out))
            _run_eval(
                eval_cfg_path=exp_cfg_path,
                ae_ckpt=args.ae_ckpt,
                planner_ckpt=args.planner_ckpt,
                value_ckpt=args.value_ckpt,
                constraint_ckpt=args.constraint_ckpt,
                out_dir=exp_out,
            )

        summary = _load_dataset_summary(exp_out, dataset)
        strict_gap = _strict_gap_from_predictions(pred_path)
        row = {
            "name": name,
            "label": label,
            "dataset": dataset,
            "run_dir": str(exp_out),
            "success_rate": _safe_float(summary.get("success_rate", strict_gap["oracle_execution_rate"])),
            "first_error_step": _safe_float(summary.get("first_error_step")),
            "recovery_at_error": _safe_float(summary.get("recovery_at_error")),
            "trap_at_1": _safe_float(summary.get("trap_at_1")),
            "plan_feasibility": _safe_float(summary.get("plan_feasibility")),
            "constraint_violation_rate": _safe_float(summary.get("constraint_violation_rate")),
            "plan_execution_consistency": _safe_float(summary.get("plan_execution_consistency")),
            "token_cost": _safe_float(summary.get("token_cost")),
            "latency_cost": _safe_float(summary.get("latency_cost")),
            "diversity_coverage": _safe_float(summary.get("diversity_coverage")),
            "oracle_mrr": _safe_float(summary.get("oracle_mrr")),
            "oracle_hit_at_1": _safe_float(summary.get("oracle_hit_at_1")),
            "oracle_hit_at_3": _safe_float(summary.get("oracle_hit_at_3")),
            "oracle_hit_at_5": _safe_float(summary.get("oracle_hit_at_5")),
        }
        row.update(strict_gap)
        rows.append(row)
        print(
            f"{label:>16} | success={row['success_rate']:.4f} "
            f"select={row['oracle_selection_rate']:.4f} "
            f"exec={row['oracle_execution_rate']:.4f} "
            f"gap={row['oracle_gap']:.4f}"
        )

    by_name = {r["name"]: r for r in rows}
    anchor = by_name.get(anchor_name) if anchor_name else rows[0]
    if anchor is not None:
        anchor_gap = _safe_float(anchor.get("oracle_gap"))
        anchor_success = _safe_float(anchor.get("success_rate"))
        for row in rows:
            row["oracle_gap_delta_vs_anchor"] = _safe_float(row.get("oracle_gap")) - anchor_gap
            row["success_delta_vs_anchor"] = _safe_float(row.get("success_rate")) - anchor_success

    fields = [
        "name",
        "label",
        "dataset",
        "n",
        "success_rate",
        "oracle_pool_recall",
        "oracle_selection_rate",
        "oracle_execution_rate",
        "oracle_gap",
        "oracle_gap_delta_vs_anchor",
        "conditional_execution_given_selected",
        "plan_feasibility",
        "trap_at_1",
        "first_error_step",
        "recovery_at_error",
        "constraint_violation_rate",
        "plan_execution_consistency",
        "oracle_mrr",
        "oracle_hit_at_1",
        "oracle_hit_at_3",
        "oracle_hit_at_5",
        "token_cost",
        "latency_cost",
        "diversity_coverage",
        "run_dir",
    ]
    _write_csv(out_dir / "webqsp_ablation_gap_table.csv", rows, fields)
    _write_markdown(out_dir / "webqsp_ablation_gap_table.md", rows)
    dump_json(str(out_dir / "webqsp_ablation_gap_table.json"), rows)
    dump_json(
        str(out_dir / "webqsp_ablation_gap_manifest.json"),
        {
            "config": args.config,
            "dataset": dataset,
            "anchor_experiment": anchor_name,
            "outputs": {
                "csv": str(out_dir / "webqsp_ablation_gap_table.csv"),
                "md": str(out_dir / "webqsp_ablation_gap_table.md"),
                "json": str(out_dir / "webqsp_ablation_gap_table.json"),
                "figure_png": str(out_dir / "webqsp_ablation_gap_reduction.png"),
                "figure_pdf": str(out_dir / "webqsp_ablation_gap_reduction.pdf"),
            },
            "note": (
                "The gap here is a strict-oracle gap derived from evaluate_torch.py: "
                "oracle_hit_at_1 minus exact execution success. Do not mix it with the "
                "answer-reaching funnel from the patched official ToG pipeline."
            ),
        },
    )
    _plot_gap_figure(rows, anchor_name, out_dir)
    print(f"[ok] wrote WebQSP ablation gap package to {out_dir}")


if __name__ == "__main__":
    main()
