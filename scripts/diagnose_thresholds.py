import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_method_metrics(summary_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(summary_metrics, dict) or not summary_metrics:
        return {}
    # summary_metrics.json usually has one top-level method key.
    if "success_rate" in summary_metrics:
        return summary_metrics
    k = next(iter(summary_metrics.keys()))
    v = summary_metrics.get(k, {})
    return v if isinstance(v, dict) else {}


def _fmt_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def _status_line(
    value: Optional[float],
    target: float,
    higher_is_better: bool = True,
    na_hint: str = "无法评估",
) -> Tuple[str, str]:
    if value is None:
        return "⚪", na_hint
    if higher_is_better:
        if value >= target:
            return "✅", "达标"
        return "❌", f"未达标 (距门槛还差 {target - value:.4f})"
    if value <= target:
        return "✅", "达标"
    return "❌", f"未达标 (超出门槛 {value - target:.4f})"


def _try_get_rows_total(manifest: Optional[Dict[str, Any]]) -> Optional[int]:
    if not manifest:
        return None
    # Compatible with processed data manifest.
    if isinstance(manifest.get("rows_total"), int):
        return int(manifest["rows_total"])
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        vals = []
        for k in ("train", "val", "test"):
            v = counts.get(k)
            if isinstance(v, int):
                vals.append(v)
        if vals:
            return int(sum(vals))
    return None


def _try_get_split(manifest: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not manifest:
        return None, None, None
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        tr = counts.get("train") if isinstance(counts.get("train"), int) else None
        va = counts.get("val") if isinstance(counts.get("val"), int) else None
        te = counts.get("test") if isinstance(counts.get("test"), int) else None
        return tr, va, te
    return None, None, None


def _extract_multiseed_std(multiseed: Optional[Dict[str, Any]]) -> Optional[float]:
    if not multiseed:
        return None
    agg = multiseed.get("aggregate")
    if not isinstance(agg, dict):
        return None
    sr = agg.get("success_rate")
    if isinstance(sr, dict) and isinstance(sr.get("std"), (int, float)):
        return float(sr["std"])
    return None


def _extract_pvalue(significance: Optional[Dict[str, Any]]) -> Optional[float]:
    if not significance:
        return None
    for k in (
        "mcnemar_p_approx",
        "mcnemar_p",
        "p_value",
        "p",
    ):
        v = significance.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _auto_find_neighbor(path: Path, candidate_rel: str) -> Optional[str]:
    cand = (path.parent.parent / candidate_rel).resolve()
    if cand.exists():
        return str(cand)
    return None


def diagnose(
    summary_with_value: Dict[str, Any],
    summary_no_value: Optional[Dict[str, Any]],
    data_manifest: Optional[Dict[str, Any]],
    multiseed_summary: Optional[Dict[str, Any]],
    significance: Optional[Dict[str, Any]],
) -> str:
    m = _pick_method_metrics(summary_with_value)
    m0 = _pick_method_metrics(summary_no_value) if summary_no_value else {}

    # Core metrics.
    success_rate = float(m.get("success_rate")) if isinstance(m.get("success_rate"), (int, float)) else None
    hit_rate = (
        float(m.get("candidate_pool_hit_rate"))
        if isinstance(m.get("candidate_pool_hit_rate"), (int, float))
        else None
    )
    pool_avg = (
        float(m.get("candidate_pool_avg_size"))
        if isinstance(m.get("candidate_pool_avg_size"), (int, float))
        else None
    )
    cond_sr = (
        float(m.get("conditional_success_given_pool_hit"))
        if isinstance(m.get("conditional_success_given_pool_hit"), (int, float))
        else None
    )
    rank_err = (
        float(m.get("ranking_error_rate"))
        if isinstance(m.get("ranking_error_rate"), (int, float))
        else None
    )
    rank_err_base = (
        float(m0.get("ranking_error_rate"))
        if isinstance(m0.get("ranking_error_rate"), (int, float))
        else None
    )
    rank_err_delta = None
    if rank_err is not None and rank_err_base is not None:
        rank_err_delta = rank_err - rank_err_base

    rows_total = _try_get_rows_total(data_manifest)
    tr, va, te = _try_get_split(data_manifest)
    sr_std = _extract_multiseed_std(multiseed_summary)
    p_value = _extract_pvalue(significance)

    # A. Data scale checks.
    a_rows_icon, a_rows_msg = _status_line(
        float(rows_total) if rows_total is not None else None,
        300.0,
        higher_is_better=True,
        na_hint="无法评估 (未提供数据 manifest)",
    )
    test_ok = (te is not None and te >= 80)
    if te is None or tr is None:
        a_split_icon, a_split_msg = "⚪", "无法评估 (未提供 train/test 计数)"
    else:
        if test_ok:
            a_split_icon, a_split_msg = "✅", "达标"
        else:
            a_split_icon, a_split_msg = "❌", f"未达标 (测试集 {te} < 80)"

    # B. Candidate pool checks (no-value stage ideally).
    b_hit_icon, b_hit_msg = _status_line(hit_rate, 0.25, higher_is_better=True)
    if hit_rate is not None and hit_rate <= 1e-12:
        b_hit_icon, b_hit_msg = "🚨", "严重红灯! (命中率为 0)"
    b_pool_icon, b_pool_msg = _status_line(pool_avg, 8.0, higher_is_better=True)

    # C. Ranker checks (with value).
    if hit_rate is None or hit_rate <= 1e-12:
        c_cond_icon, c_cond_msg = "⚪", "无法评估 (前置候选池命中率为 0)"
    else:
        c_cond_icon, c_cond_msg = _status_line(cond_sr, 0.20, higher_is_better=True)

    if rank_err_delta is None:
        c_rank_icon, c_rank_msg = "⚪", "无法评估 (未提供 no-value 对照)"
    else:
        if rank_err_delta <= -0.05:
            c_rank_icon, c_rank_msg = "✅", "达标 (ranking_error 至少下降 5pp)"
        else:
            c_rank_icon, c_rank_msg = "❌", f"未达标 (当前变化 {rank_err_delta:+.4f})"

    # D. End-to-end.
    d_sr_icon, d_sr_msg = _status_line(success_rate, 0.10, higher_is_better=True)

    # E. Stability.
    if sr_std is None:
        e_std_icon, e_std_msg = "❌", "未达标 (当前未满足多种子或未提供 multiseed 汇总)"
    else:
        if sr_std <= 0.015:
            e_std_icon, e_std_msg = "✅", "达标"
        else:
            e_std_icon, e_std_msg = "❌", f"未达标 (std={sr_std:.4f} > 0.0150)"

    if p_value is None:
        e_p_icon, e_p_msg = "❌", "未达标 (未提供显著性检验文件)"
    else:
        if p_value < 0.05:
            e_p_icon, e_p_msg = "✅", "达标"
        else:
            e_p_icon, e_p_msg = "❌", f"未达标 (p={p_value:.4g} >= 0.05)"

    # Root cause summary.
    root = []
    advice = []
    if hit_rate is None or hit_rate <= 0.10:
        root.append("当前实验卡在【B. 候选池门槛】。Planner 未稳定探索到包含正确解的候选轨迹。")
        advice.append("提高候选采样预算（如 num_candidates=50/100）并检查 memory 检索是否退化到单一路径。")
        advice.append("核查数据预处理 Token 映射与 oracle_path 保真度，避免动作标准化后语义塌缩。")
    elif cond_sr is not None and cond_sr < 0.20:
        root.append("候选池已有信号，但【C. 排序器门槛】未达标。Value 模型区分能力不足。")
        advice.append("优先上 full-pool listwise + hard negatives，降低 ranking_error。")
    elif success_rate is not None and success_rate < 0.10:
        root.append("前两关有信号，但端到端策略仍在【D. 端到端门槛】受限。")
        advice.append("加强重规划触发规则与约束校验，减少 early drift。")
    else:
        root.append("主链路指标整体健康，可进入多种子与显著性收敛阶段。")
        advice.append("固定配置后跑 3-seed，补齐 std 与显著性报告。")

    # Console encoding fallback (Windows/GBK may fail on emoji).
    use_ascii = False
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "gbk" in enc or "cp936" in enc:
        use_ascii = True

    if use_ascii:
        sym_map = {
            "✅": "[OK]",
            "❌": "[NO]",
            "🚨": "[ALERT]",
            "⚪": "[N/A]",
        }
    else:
        sym_map = {}

    def sym(x: str) -> str:
        return sym_map.get(x, x)

    lines = []
    lines.append("======================================================================")
    lines.append("         DiPLaN 实验达标阈值看板 (Paper-Ready Diagnostic)")
    lines.append("======================================================================")
    lines.append("[A. 数据规模门槛]")
    lines.append(
        f"  - 总样本数 (rows_total): {rows_total if rows_total is not None else 'N/A':>6} "
        f"--> {sym(a_rows_icon)} {a_rows_msg}"
    )
    split_str = "N/A" if tr is None or te is None else f"{tr}/{te} (train/test)"
    lines.append(f"  - 划分比例 (train/test): {split_str:<24} --> {sym(a_split_icon)} {a_split_msg}")
    lines.append("")
    lines.append("[B. 候选池门槛 (无 Value)]")
    lines.append(
        f"  - 命中率 (candidate_pool_hit_rate): {_fmt_float(hit_rate, 4):<8} "
        f"--> {sym(b_hit_icon)} {b_hit_msg}"
    )
    lines.append(
        f"  - 平均池大小 (pool_avg_size): {_fmt_float(pool_avg, 2):<8} "
        f"--> {sym(b_pool_icon)} {b_pool_msg}"
    )
    lines.append("")
    lines.append("[C. 排序器门槛 (有 Value)]")
    lines.append(
        f"  - 命中后成功率 (conditional_SR): {_fmt_float(cond_sr, 4):<8} "
        f"--> {sym(c_cond_icon)} {c_cond_msg}"
    )
    lines.append(
        f"  - 排序错误率变化 (ranking_error_delta): "
        f"{(_fmt_float(rank_err_delta, 4) if rank_err_delta is not None else 'N/A'):<8} "
        f"--> {sym(c_rank_icon)} {c_rank_msg}"
    )
    lines.append("")
    lines.append("[D. 端到端门槛]")
    lines.append(
        f"  - 最终成功率 (success_rate): {_fmt_float(success_rate, 4):<8} "
        f"--> {sym(d_sr_icon)} {d_sr_msg}"
    )
    lines.append("")
    lines.append("[E. 稳定性门槛]")
    lines.append(
        f"  - 标准差 (std, success_rate): {_fmt_float(sr_std, 4):<8} "
        f"--> {sym(e_std_icon)} {e_std_msg}"
    )
    pv_str = "N/A" if p_value is None else f"{p_value:.4g}"
    lines.append(f"  - 统计检验 p-value: {pv_str:<14} --> {sym(e_p_icon)} {e_p_msg}")
    lines.append("")
    lines.append("----------------------------------------------------------------------")
    lines.append(f"{sym('🚨')} 核心根因诊断 (Root Cause):")
    for r in root:
        lines.append(r)
    lines.append("Actionable Suggestions:")
    for i, a in enumerate(advice, 1):
        lines.append(f"{i}. {a}")
    lines.append("======================================================================")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_with_value", type=str, required=True, help="Path to summary_metrics.json (with value).")
    parser.add_argument("--summary_no_value", type=str, default="", help="Optional no-value summary_metrics.json for delta.")
    parser.add_argument("--data_manifest", type=str, default="", help="Optional data manifest.json for rows/split checks.")
    parser.add_argument("--multiseed_summary", type=str, default="", help="Optional multiseed summary json (mean/std).")
    parser.add_argument("--significance", type=str, default="", help="Optional significance json with p-value.")
    parser.add_argument("--out", type=str, default="", help="Optional output text file path.")
    args = parser.parse_args()

    with_value = _load_json(args.summary_with_value)
    if with_value is None:
        raise FileNotFoundError(f"Cannot read summary_with_value: {args.summary_with_value}")

    no_value = _load_json(args.summary_no_value)
    data_manifest = _load_json(args.data_manifest)
    multiseed = _load_json(args.multiseed_summary)
    significance = _load_json(args.significance)

    # Lightweight auto-discovery if optional inputs are missing.
    swv = Path(args.summary_with_value).resolve()
    if no_value is None:
        auto_no = _auto_find_neighbor(swv, "train_eval_no_value/summary_metrics.json")
        no_value = _load_json(auto_no)
    if data_manifest is None:
        # Try sibling processed manifest by simple name heuristic.
        parent = swv
        for _ in range(6):
            parent = parent.parent
            cand = parent / "data"
            if cand.exists():
                break
        # no hard failure here.

    board = diagnose(
        summary_with_value=with_value,
        summary_no_value=no_value,
        data_manifest=data_manifest,
        multiseed_summary=multiseed,
        significance=significance,
    )
    print(board)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(board, encoding="utf-8")
        print(f"[ok] wrote report: {outp}")


if __name__ == "__main__":
    main()
