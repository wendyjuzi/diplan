"""Build a paper-oriented causal-evidence package for DiPLaN.

This script turns raw run outputs into the figures/tables needed by the paper's
"mechanism first" story:

RQ1  Does a measurable Selection-to-Execution Gap exist?
RQ2  Is the gap associated with downstream failure?
RQ3  Does DiPLaN reduce the gap?
RQ4  Which execution behaviors change after introducing future-aware priors?
RQ5  Do these behaviors transfer beyond KGQA?
RQ6  Is gap reduction a plausible mediator of the success improvement?

Important metric note
---------------------
The script supports two metric families and keeps them separate:

1. KGQA strict-oracle diagnostics
   - selection_rate  := oracle_hit_at_1
   - execution_rate  := plan_feasibility if present, else success_rate
   - gap_rate        := selection_rate - execution_rate

2. Transfer-probe execution diagnostics
   - exact strict-oracle selection is usually unavailable
   - the script therefore focuses on first_error_step, executable_horizon,
     plan_feasibility, execution consistency, fallback rate, and success_rate

This separation is intentional: KGQA is treated as a controlled diagnostic
instrument, while ALFWorld / ScienceWorld act as transfer probes.

Examples
--------
python scripts/build_causal_evidence_suite.py \
  --run "WebQSP-ToG|kgqa|results/webqsp_tog/predictions.jsonl|results/webqsp_tog/summary_metrics.json" \
  --run "WebQSP-DiPLaN|kgqa|results/webqsp_diplan/predictions.jsonl|results/webqsp_diplan/summary_metrics.json" \
  --run "ALF-DiPLaN|alfworld|results/alfworld_diplan/predictions.jsonl|results/alfworld_diplan/summary_metrics.json" \
  --run "ALF-ReAct|alfworld|results/alfworld_react/predictions.jsonl|results/alfworld_react/summary_metrics.json" \
  --run "SW-DiPLaN|scienceworld|results/scienceworld/test_with_value_eval/predictions.jsonl|results/scienceworld/test_with_value_eval/summary_metrics.json" \
  --mediation_pair "kgqa|ToG|results/webqsp_tog/predictions.jsonl|DiPLaN|results/webqsp_diplan/predictions.jsonl" \
  --out results/causal_evidence_suite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.3,
            "grid.alpha": 0.24,
            "grid.color": "#B8C2CC",
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x in (None, ""):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    text = str(x).strip().lower()
    return text in {"1", "true", "yes"}


def _pick_summary(raw: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    if method:
        if method not in raw:
            raise KeyError(f"method '{method}' not found in summary JSON")
        return raw[method]
    if "success_rate" in raw or "hits@1" in raw:
        return raw
    if len(raw) == 1:
        return raw[next(iter(raw))]
    raise ValueError(
        "Summary JSON contains multiple method keys. Use LABEL|ENV|PRED|SUMMARY|METHOD."
    )


def _parse_run_spec(spec: str) -> dict[str, str | None]:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) not in {3, 4, 5}:
        raise ValueError(
            "Run spec must be LABEL|ENV|PRED_JSONL or LABEL|ENV|PRED_JSONL|SUMMARY_JSON or "
            "LABEL|ENV|PRED_JSONL|SUMMARY_JSON|METHOD"
        )
    label = parts[0]
    env = parts[1].lower()
    pred_path = parts[2]
    summary_path = parts[3] if len(parts) >= 4 and parts[3] else None
    method = parts[4] if len(parts) == 5 and parts[4] else None
    return {
        "label": label,
        "env": env,
        "pred_path": pred_path,
        "summary_path": summary_path,
        "method": method,
    }


def _parse_mediation_spec(spec: str) -> dict[str, str]:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 5:
        raise ValueError(
            "Mediation spec must be ENV|CONTROL_LABEL|CONTROL_PRED_JSONL|TREAT_LABEL|TREAT_PRED_JSONL"
        )
    return {
        "env": parts[0].lower(),
        "control_label": parts[1],
        "control_pred": parts[2],
        "treat_label": parts[3],
        "treat_pred": parts[4],
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(xs) / len(xs) if xs else None


def _fmt_pct(x: Any) -> str:
    val = _safe_float(x)
    return "" if val is None else f"{100.0 * val:.2f}%"


def _fmt_num(x: Any, digits: int = 3) -> str:
    val = _safe_float(x)
    return "" if val is None else f"{val:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_markdown(path: Path, rows: list[dict[str, Any]], fields: list[str], formatters: dict[str, Any]) -> None:
    headers = [f.replace("_", " ").title() for f in fields]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            fmt = formatters.get(field)
            vals.append(fmt(value) if fmt else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_steps(row: dict[str, Any]) -> int:
    for key in ("num_steps", "episode_trace_len"):
        val = _safe_float(row.get(key))
        if val is not None and val > 0:
            return int(round(val))
    for key in ("actions", "executed_path", "planned_path", "episode_trace"):
        val = row.get(key)
        if isinstance(val, list) and val:
            return len(val)
    return 0


def _normalize_row(row: dict[str, Any], env: str, label: str) -> dict[str, Any]:
    success = 1.0 if _safe_bool(row.get("success")) else 0.0
    steps = _infer_steps(row)
    first_error = _safe_float(row.get("first_error_step"))
    if first_error is None:
        first_error = float(steps + 1 if success > 0.0 else max(1, steps))
    plan_feasibility = _safe_float(row.get("plan_feasibility"))
    consistency = _safe_float(row.get("plan_execution_consistency"))
    fallback = _safe_float(row.get("grounded_fallback_rate"))
    if fallback is None:
        fallback = _safe_float(row.get("parse_failure_rate"))
    execution_rate = plan_feasibility
    if execution_rate is None:
        feasible = row.get("feasible")
        if feasible is not None:
            execution_rate = 1.0 if _safe_bool(feasible) else 0.0
    if execution_rate is None:
        execution_rate = success

    selection_rate = _safe_float(row.get("oracle_hit_at_1"))
    oracle_pool = _safe_float(row.get("oracle_in_candidate_pool"))
    exact_gap = None
    if selection_rate is not None:
        exact_gap = selection_rate - execution_rate
    success_gap = None
    if execution_rate is not None:
        success_gap = execution_rate - success
    executable_horizon = None
    if steps > 0:
        executable_horizon = max(0.0, min(float(steps), float(first_error) - 1.0)) / float(steps)

    return {
        "label": label,
        "env": env,
        "success": success,
        "num_steps": steps,
        "first_error_step": first_error,
        "plan_feasibility": plan_feasibility,
        "execution_rate": execution_rate,
        "selection_rate": selection_rate,
        "oracle_pool_rate": oracle_pool,
        "selection_execution_gap": exact_gap,
        "execution_success_gap": success_gap,
        "executable_horizon": executable_horizon,
        "plan_execution_consistency": consistency,
        "fallback_rate": fallback,
        "token_cost": _safe_float(row.get("token_cost")),
        "latency_cost": _safe_float(row.get("latency_cost")),
        "grounded_fallback_rate": _safe_float(row.get("grounded_fallback_rate")),
        "constraint_violation": 1.0 if _safe_bool(row.get("constraint_violation")) else 0.0,
    }


def _load_run(spec: str) -> dict[str, Any]:
    parsed = _parse_run_spec(spec)
    pred_path = Path(str(parsed["pred_path"]))
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path
    summary_path = None
    if parsed["summary_path"]:
        summary_path = Path(str(parsed["summary_path"]))
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path

    predictions = load_jsonl(str(pred_path)) if pred_path.exists() else []
    summary = None
    if summary_path and summary_path.exists():
        summary = _pick_summary(load_json(str(summary_path)), parsed["method"])

    normalized_rows = [_normalize_row(row, str(parsed["env"]), str(parsed["label"])) for row in predictions]
    metric_family = "success_only"
    if any(r["selection_rate"] is not None for r in normalized_rows):
        metric_family = "strict_oracle_gap"
    elif any(r["plan_feasibility"] is not None for r in normalized_rows):
        metric_family = "execution_probe"

    summary_row = _summarize_run(
        label=str(parsed["label"]),
        env=str(parsed["env"]),
        rows=normalized_rows,
        summary_json=summary or {},
        pred_path=pred_path,
        summary_path=summary_path,
        metric_family=metric_family,
    )
    return {
        "label": parsed["label"],
        "env": parsed["env"],
        "pred_path": str(pred_path),
        "summary_path": str(summary_path) if summary_path else "",
        "summary": summary or {},
        "rows": normalized_rows,
        "summary_row": summary_row,
    }


def _summarize_run(
    *,
    label: str,
    env: str,
    rows: list[dict[str, Any]],
    summary_json: dict[str, Any],
    pred_path: Path,
    summary_path: Path | None,
    metric_family: str,
) -> dict[str, Any]:
    def summary_first(keys: list[str]) -> float | None:
        for key in keys:
            val = _safe_float(summary_json.get(key))
            if val is not None:
                return val
        return None

    n = len(rows)
    success_rate = _mean_or_none([r["success"] for r in rows])
    first_error_step = _mean_or_none([r["first_error_step"] for r in rows])
    selection_rate = _mean_or_none([r["selection_rate"] for r in rows])
    execution_rate = _mean_or_none([r["execution_rate"] for r in rows])
    pool_rate = _mean_or_none([r["oracle_pool_rate"] for r in rows])
    gap_rate = _mean_or_none([r["selection_execution_gap"] for r in rows])
    exec_success_gap = _mean_or_none([r["execution_success_gap"] for r in rows])
    executable_horizon = _mean_or_none([r["executable_horizon"] for r in rows])
    consistency = _mean_or_none([r["plan_execution_consistency"] for r in rows])
    fallback = _mean_or_none([r["fallback_rate"] for r in rows])
    token_cost = _mean_or_none([r["token_cost"] for r in rows])
    latency_cost = _mean_or_none([r["latency_cost"] for r in rows])

    success_rate = success_rate if success_rate is not None else summary_first(["success_rate", "hits@1"])
    first_error_step = first_error_step if first_error_step is not None else summary_first(["first_error_step"])
    selection_rate = selection_rate if selection_rate is not None else summary_first(["oracle_hit_at_1"])
    execution_rate = execution_rate if execution_rate is not None else summary_first(["plan_feasibility", "success_rate"])
    pool_rate = pool_rate if pool_rate is not None else summary_first(["oracle_pool_recall", "candidate_pool_hit_rate"])
    gap_rate = gap_rate if gap_rate is not None else (
        None if selection_rate is None or execution_rate is None else selection_rate - execution_rate
    )
    exec_success_gap = exec_success_gap if exec_success_gap is not None else (
        None if execution_rate is None or success_rate is None else execution_rate - success_rate
    )
    executable_horizon = executable_horizon if executable_horizon is not None else execution_rate
    consistency = consistency if consistency is not None else summary_first(["plan_execution_consistency"])
    fallback = fallback if fallback is not None else summary_first(["grounded_fallback_rate", "parse_failure_rate"])
    token_cost = token_cost if token_cost is not None else summary_first(["token_cost", "prompt_tokens"])
    latency_cost = latency_cost if latency_cost is not None else summary_first(["latency_cost", "wall_time_s_per_task"])

    return {
        "label": label,
        "env": env,
        "n": n,
        "metric_family": metric_family,
        "success_rate": success_rate,
        "oracle_pool_rate": pool_rate,
        "selection_rate": selection_rate,
        "execution_rate": execution_rate,
        "gap_rate": gap_rate,
        "execution_success_gap": exec_success_gap,
        "first_error_step": first_error_step,
        "executable_horizon": executable_horizon,
        "plan_execution_consistency": consistency,
        "fallback_rate": fallback,
        "constraint_violation_rate": summary_first(["constraint_violation_rate"]),
        "recovery_at_error": summary_first(["recovery_at_error"]),
        "avg_steps": summary_first(["avg_steps"]) or _mean_or_none([float(r["num_steps"]) for r in rows]),
        "token_cost": token_cost,
        "latency_cost": latency_cost,
        "predictions_path": str(pred_path),
        "summary_metrics_path": str(summary_path) if summary_path else "",
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx.tolist(), ry.tolist())


def _ols(y: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    return {
        "coef": coef,
        "pred": pred,
        "resid": resid,
        "mse": float(np.mean(resid ** 2)),
    }


def _bootstrap_mediation(treatment: np.ndarray, mediator: np.ndarray, outcome: np.ndarray, seed: int, num_boot: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(treatment)
    a_vals: list[float] = []
    b_vals: list[float] = []
    indirect_vals: list[float] = []
    direct_vals: list[float] = []
    total_vals: list[float] = []
    for _ in range(num_boot):
        idx = rng.integers(0, n, size=n)
        t = treatment[idx]
        m = mediator[idx]
        y = outcome[idx]
        X_a = np.column_stack([np.ones(n), t])
        a_fit = _ols(m, X_a)
        X_total = np.column_stack([np.ones(n), t])
        total_fit = _ols(y, X_total)
        X_med = np.column_stack([np.ones(n), t, m])
        med_fit = _ols(y, X_med)
        a = float(a_fit["coef"][1])
        total = float(total_fit["coef"][1])
        direct = float(med_fit["coef"][1])
        b = float(med_fit["coef"][2])
        a_vals.append(a)
        b_vals.append(b)
        total_vals.append(total)
        direct_vals.append(direct)
        indirect_vals.append(a * b)

    def interval(vals: list[float]) -> list[float]:
        arr = np.asarray(vals, dtype=float)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "a_ci95": interval(a_vals),
        "b_ci95": interval(b_vals),
        "indirect_ci95": interval(indirect_vals),
        "direct_ci95": interval(direct_vals),
        "total_ci95": interval(total_vals),
    }


def _run_mediation(spec: str, seed: int, num_boot: int) -> dict[str, Any]:
    parsed = _parse_mediation_spec(spec)
    control_rows = load_jsonl(str((ROOT / parsed["control_pred"]).resolve()))
    treat_rows = load_jsonl(str((ROOT / parsed["treat_pred"]).resolve()))
    control = [_normalize_row(row, parsed["env"], parsed["control_label"]) for row in control_rows]
    treat = [_normalize_row(row, parsed["env"], parsed["treat_label"]) for row in treat_rows]

    pooled = []
    for row in control:
        gap = row["selection_execution_gap"]
        if gap is None:
            gap = row["execution_success_gap"]
        if gap is None:
            continue
        pooled.append((0.0, float(gap), float(row["success"])))
    for row in treat:
        gap = row["selection_execution_gap"]
        if gap is None:
            gap = row["execution_success_gap"]
        if gap is None:
            continue
        pooled.append((1.0, float(gap), float(row["success"])))
    if len(pooled) < 8:
        return {
            "env": parsed["env"],
            "control_label": parsed["control_label"],
            "treat_label": parsed["treat_label"],
            "supported": False,
            "reason": "Not enough per-episode rows with mediator values.",
        }

    arr = np.asarray(pooled, dtype=float)
    treatment = arr[:, 0]
    mediator = arr[:, 1]
    outcome = arr[:, 2]

    X_a = np.column_stack([np.ones(len(arr)), treatment])
    a_fit = _ols(mediator, X_a)
    X_total = np.column_stack([np.ones(len(arr)), treatment])
    total_fit = _ols(outcome, X_total)
    X_med = np.column_stack([np.ones(len(arr)), treatment, mediator])
    med_fit = _ols(outcome, X_med)

    a = float(a_fit["coef"][1])
    total = float(total_fit["coef"][1])
    direct = float(med_fit["coef"][1])
    b = float(med_fit["coef"][2])
    indirect = a * b
    boot = _bootstrap_mediation(treatment, mediator, outcome, seed=seed, num_boot=num_boot)

    return {
        "env": parsed["env"],
        "control_label": parsed["control_label"],
        "treat_label": parsed["treat_label"],
        "supported": True,
        "n_rows": len(arr),
        "mediator_name": "selection_execution_gap" if any(r["selection_execution_gap"] is not None for r in control + treat) else "execution_success_gap",
        "model_note": "Linear-probability proxy used for mechanistic evidence; not a causal identification proof.",
        "a_treatment_to_gap": a,
        "b_gap_to_success_controlling_treatment": b,
        "c_total_treatment_to_success": total,
        "c_prime_direct_treatment_to_success": direct,
        "indirect_effect_ab": indirect,
        **boot,
    }


def _build_run_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(run["summary_row"]) for run in runs]


def _compute_gap_correlations(run_table: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    run_level_rows = [r for r in run_table if r.get("gap_rate") is not None and r.get("success_rate") is not None]
    run_level = {
        "n": len(run_level_rows),
        "pearson_gap_vs_success": _pearson(
            [float(r["gap_rate"]) for r in run_level_rows],
            [float(r["success_rate"]) for r in run_level_rows],
        ),
        "spearman_gap_vs_success": _spearman(
            [float(r["gap_rate"]) for r in run_level_rows],
            [float(r["success_rate"]) for r in run_level_rows],
        ),
    }

    pooled_rows = []
    for run in runs:
        for row in run["rows"]:
            gap = row["selection_execution_gap"]
            if gap is None:
                gap = row["execution_success_gap"]
            if gap is None:
                continue
            pooled_rows.append((gap, row["success"], run["label"], run["env"]))
    pooled = {
        "n": len(pooled_rows),
        "pearson_gap_vs_success": _pearson(
            [float(x[0]) for x in pooled_rows],
            [float(x[1]) for x in pooled_rows],
        ),
        "spearman_gap_vs_success": _spearman(
            [float(x[0]) for x in pooled_rows],
            [float(x[1]) for x in pooled_rows],
        ),
    }
    return {
        "run_level": run_level,
        "episode_level": pooled,
    }


def _plot_kgqa_funnel(run_table: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [r for r in run_table if str(r["env"]).lower() == "kgqa"]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(max(5.0, 4.6 * len(rows)), 4.6), sharey=True)
    if len(rows) == 1:
        axes = [axes]
    stages = ["Pool", "Selected", "Executable", "Success"]
    colors = ["#D6E4F0", "#8BB8F8", "#59B17A", "#C44536"]
    for ax, row in zip(axes, rows):
        values = [
            100.0 * float(row["oracle_pool_rate"] if row["oracle_pool_rate"] is not None else 1.0),
            100.0 * float(row["selection_rate"] if row["selection_rate"] is not None else 0.0),
            100.0 * float(row["execution_rate"] if row["execution_rate"] is not None else row["success_rate"] or 0.0),
            100.0 * float(row["success_rate"] if row["success_rate"] is not None else 0.0),
        ]
        x = np.arange(len(stages))
        ax.bar(x, values, color=colors, width=0.7)
        for xi, yi in zip(x, values):
            ax.text(xi, yi + 1.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=9)
        gap = values[1] - values[2]
        ax.set_title(str(row["label"]))
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=15)
        ax.grid(True, axis="y")
        ax.set_ylim(0, 105)
        ax.annotate(
            f"Gap {gap:.1f} pts",
            xy=(1.5, (values[1] + values[2]) / 2.0),
            xytext=(2.7, min(100.0, values[1] + 8.0)),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#C44536"},
            fontsize=9,
            color="#C44536",
        )
    axes[0].set_ylabel("Rate (%)")
    fig.suptitle("KGQA as Controlled Diagnostic Instrument", y=1.02, fontsize=15)
    fig.savefig(out_dir / "rq1_kgqa_execution_funnel.png")
    fig.savefig(out_dir / "rq1_kgqa_execution_funnel.pdf")
    plt.close(fig)


def _plot_gap_success_scatter(run_table: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [r for r in run_table if r.get("gap_rate") is not None and r.get("success_rate") is not None]
    if len(rows) < 2:
        return
    env_colors = {
        "kgqa": "#1F6FEB",
        "alfworld": "#C44536",
        "scienceworld": "#2AA198",
    }
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    xs = []
    ys = []
    for row in rows:
        env = str(row["env"]).lower()
        x = 100.0 * float(row["gap_rate"])
        y = 100.0 * float(row["success_rate"])
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=110, color=env_colors.get(env, "#6B7280"), edgecolor="white", linewidth=1.0)
        ax.annotate(str(row["label"]), xy=(x, y), xytext=(6, 5), textcoords="offset points", fontsize=8.8)
    corr = _pearson(xs, ys)
    ax.set_xlabel("Gap (Selection - Execution, pts)")
    ax.set_ylabel("Task Success (%)")
    ax.set_title("RQ2: Gap Is Associated with Failure")
    ax.grid(True)
    if corr is not None:
        ax.text(
            0.03,
            0.96,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#243B53",
        )
    fig.savefig(out_dir / "rq2_gap_success_correlation.png")
    fig.savefig(out_dir / "rq2_gap_success_correlation.pdf")
    plt.close(fig)


def _plot_first_error_bars(run_table: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [r for r in run_table if r.get("first_error_step") is not None]
    if not rows:
        return
    labels = [str(r["label"]) for r in rows]
    values = [float(r["first_error_step"]) for r in rows]
    colors = ["#1F6FEB" if "diplan" in label.lower() else "#C44536" for label in labels]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.35 * len(rows)), 4.8))
    x = np.arange(len(rows))
    ax.bar(x, values, color=colors, width=0.72)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + 0.12, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average First Error Step")
    ax.set_title("RQ2/RQ5: Failure Timeline")
    ax.grid(True, axis="y")
    fig.savefig(out_dir / "rq2_first_error_timeline.png")
    fig.savefig(out_dir / "rq2_first_error_timeline.pdf")
    plt.close(fig)


def _survival_curve(rows: list[dict[str, Any]], max_t: int) -> list[float]:
    out: list[float] = []
    if not rows:
        return out
    for t in range(1, max_t + 1):
        alive = 0
        for row in rows:
            fe = float(row["first_error_step"])
            alive += 1 if fe > t else 0
        out.append(alive / len(rows))
    return out


def _plot_survival_curves(runs: list[dict[str, Any]], out_dir: Path) -> None:
    usable = [run for run in runs if run["rows"] and any(r.get("first_error_step") is not None for r in run["rows"])]
    if not usable:
        return
    max_t = 1
    for run in usable:
        max_t = max(max_t, max(int(float(r["first_error_step"])) for r in run["rows"]))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for run in usable:
        curve = _survival_curve(run["rows"], max_t)
        steps = np.arange(1, len(curve) + 1)
        color = "#1F6FEB" if "diplan" in str(run["label"]).lower() else "#C44536"
        linestyle = "-" if str(run["env"]).lower() == "kgqa" else "--"
        ax.step(steps, curve, where="post", label=str(run["label"]), linewidth=2.0, color=color, linestyle=linestyle)
    ax.set_xlabel("Step")
    ax.set_ylabel("P(still executable before first failure)")
    ax.set_title("RQ2: Execution Survival Curve")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(out_dir / "rq2_execution_survival_curve.png")
    fig.savefig(out_dir / "rq2_execution_survival_curve.pdf")
    plt.close(fig)


def _plot_transfer_probe_mechanisms(run_table: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [r for r in run_table if str(r["env"]).lower() in {"alfworld", "scienceworld"}]
    if not rows:
        return
    metrics = [
        ("success_rate", "Success"),
        ("first_error_step", "First Error"),
        ("execution_rate", "Executable"),
        ("plan_execution_consistency", "Consistency"),
        ("fallback_rate", "Fallback"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.5, max(4.8, 0.52 * len(rows) + 2.0)))
    for ax, (key, title) in zip(axes, metrics):
        vals = []
        labels = []
        colors = []
        for row in rows:
            val = _safe_float(row.get(key))
            if val is None:
                continue
            vals.append(float(val))
            labels.append(f"{row['env']} | {row['label']}")
            colors.append("#1F6FEB" if "diplan" in str(row["label"]).lower() else "#C44536")
        if not vals:
            ax.axis("off")
            continue
        y = np.arange(len(vals))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if key == "success_rate" else [])
        ax.set_title(title)
        ax.grid(True, axis="x")
        if key in {"success_rate", "execution_rate", "plan_execution_consistency", "fallback_rate"}:
            ax.set_xlim(0.0, 1.05)
        for yi, vi in zip(y, vals):
            shown = f"{100.0 * vi:.1f}%" if key != "first_error_step" else f"{vi:.2f}"
            ax.text(vi + (0.01 if key != "first_error_step" else 0.05), yi, shown, va="center", fontsize=8)
    fig.suptitle("RQ5: Transfer-Probe Mechanism Changes", y=1.02, fontsize=15)
    fig.savefig(out_dir / "rq5_transfer_probe_mechanisms.png")
    fig.savefig(out_dir / "rq5_transfer_probe_mechanisms.pdf")
    plt.close(fig)


def _write_outputs(
    out_dir: Path,
    runs: list[dict[str, Any]],
    run_table: list[dict[str, Any]],
    correlations: dict[str, Any],
    mediation: list[dict[str, Any]],
) -> None:
    fields = [
        "label",
        "env",
        "n",
        "metric_family",
        "success_rate",
        "oracle_pool_rate",
        "selection_rate",
        "execution_rate",
        "gap_rate",
        "execution_success_gap",
        "first_error_step",
        "executable_horizon",
        "plan_execution_consistency",
        "fallback_rate",
        "constraint_violation_rate",
        "recovery_at_error",
        "avg_steps",
        "token_cost",
        "latency_cost",
        "predictions_path",
        "summary_metrics_path",
    ]
    _write_csv(out_dir / "claim_suite_run_table.csv", run_table, fields)
    _write_markdown(
        out_dir / "claim_suite_run_table.md",
        run_table,
        fields=[
            "label",
            "env",
            "metric_family",
            "success_rate",
            "gap_rate",
            "execution_rate",
            "first_error_step",
            "executable_horizon",
            "plan_execution_consistency",
            "fallback_rate",
            "token_cost",
        ],
        formatters={
            "success_rate": _fmt_pct,
            "gap_rate": _fmt_pct,
            "execution_rate": _fmt_pct,
            "executable_horizon": _fmt_pct,
            "plan_execution_consistency": _fmt_pct,
            "fallback_rate": _fmt_pct,
            "first_error_step": _fmt_num,
            "token_cost": lambda x: _fmt_num(x, digits=1),
        },
    )

    kgqa_rows = [r for r in run_table if str(r["env"]).lower() == "kgqa"]
    transfer_rows = [r for r in run_table if str(r["env"]).lower() in {"alfworld", "scienceworld"}]
    if kgqa_rows:
        _write_csv(out_dir / "kgqa_diagnostic_table.csv", kgqa_rows, fields)
    if transfer_rows:
        _write_csv(out_dir / "transfer_probe_table.csv", transfer_rows, fields)
        _write_markdown(
            out_dir / "transfer_probe_table.md",
            transfer_rows,
            fields=[
                "label",
                "env",
                "success_rate",
                "first_error_step",
                "execution_rate",
                "executable_horizon",
                "plan_execution_consistency",
                "fallback_rate",
                "token_cost",
            ],
            formatters={
                "success_rate": _fmt_pct,
                "execution_rate": _fmt_pct,
                "executable_horizon": _fmt_pct,
                "plan_execution_consistency": _fmt_pct,
                "fallback_rate": _fmt_pct,
                "first_error_step": _fmt_num,
                "token_cost": lambda x: _fmt_num(x, digits=1),
            },
        )

    dump_json(
        str(out_dir / "claim_suite_summary.json"),
        {
            "run_table": run_table,
            "correlations": correlations,
            "mediation": mediation,
            "n_runs": len(runs),
            "note": (
                "KGQA metrics are treated as controlled diagnostics; ALFWorld and ScienceWorld are "
                "treated as transfer probes. Do not over-interpret proxy execution metrics as exact "
                "strict-oracle gaps when the environment does not expose oracle futures."
            ),
        },
    )
    dump_json(str(out_dir / "gap_success_correlation.json"), correlations)
    dump_json(str(out_dir / "mediation_analysis.json"), {"pairs": mediation})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="LABEL|ENV|PRED_JSONL or LABEL|ENV|PRED_JSONL|SUMMARY_JSON or LABEL|ENV|PRED_JSONL|SUMMARY_JSON|METHOD",
    )
    parser.add_argument(
        "--mediation_pair",
        action="append",
        default=[],
        help="ENV|CONTROL_LABEL|CONTROL_PRED_JSONL|TREAT_LABEL|TREAT_PRED_JSONL",
    )
    parser.add_argument("--out", type=str, default="results/causal_evidence_suite")
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    args = parser.parse_args()

    _apply_style()
    out_dir = (ROOT / args.out).resolve()
    ensure_dir(str(out_dir))

    runs = [_load_run(spec) for spec in args.run]
    run_table = _build_run_table(runs)
    correlations = _compute_gap_correlations(run_table, runs)
    mediation = [
        _run_mediation(spec, seed=int(args.bootstrap_seed), num_boot=int(args.bootstrap_samples))
        for spec in args.mediation_pair
    ]

    _plot_kgqa_funnel(run_table, out_dir)
    _plot_gap_success_scatter(run_table, out_dir)
    _plot_first_error_bars(run_table, out_dir)
    _plot_survival_curves(runs, out_dir)
    _plot_transfer_probe_mechanisms(run_table, out_dir)
    _write_outputs(out_dir, runs, run_table, correlations, mediation)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "files": [
                    str(out_dir / "claim_suite_run_table.csv"),
                    str(out_dir / "claim_suite_run_table.md"),
                    str(out_dir / "kgqa_diagnostic_table.csv"),
                    str(out_dir / "transfer_probe_table.csv"),
                    str(out_dir / "gap_success_correlation.json"),
                    str(out_dir / "mediation_analysis.json"),
                    str(out_dir / "rq1_kgqa_execution_funnel.png"),
                    str(out_dir / "rq2_gap_success_correlation.png"),
                    str(out_dir / "rq2_first_error_timeline.png"),
                    str(out_dir / "rq2_execution_survival_curve.png"),
                    str(out_dir / "rq5_transfer_probe_mechanisms.png"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
