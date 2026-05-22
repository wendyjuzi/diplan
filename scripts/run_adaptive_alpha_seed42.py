import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl
from src.diplan.stats_utils import bootstrap_mean_diff, mcnemar_test_paired


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _metric_row(label: str, metrics: Dict) -> Dict:
    row = {"setting": label}
    row.update(metrics)
    return row


def _dataset_breakdown(pred_rows: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in pred_rows:
        grouped[str(r.get("dataset", "unknown"))].append(r)
    out = {}
    for ds, recs in grouped.items():
        n = len(recs)
        out[ds] = {
            "n": n,
            "success_rate": sum(1.0 if x.get("success") else 0.0 for x in recs) / max(1, n),
            "candidate_pool_hit_rate": sum(1.0 if x.get("oracle_in_candidate_pool") else 0.0 for x in recs)
            / max(1, n),
            "ranking_error_rate": sum(1.0 if x.get("ranking_error") else 0.0 for x in recs) / max(1, n),
        }
    return out


def _paired_significance(ref_preds: List[Dict], oth_preds: List[Dict], bootstrap_n: int) -> Dict:
    ref = {str(r["task_id"]): 1 if r.get("success") else 0 for r in ref_preds}
    oth = {str(r["task_id"]): 1 if r.get("success") else 0 for r in oth_preds}
    keys = sorted(set(ref.keys()) & set(oth.keys()))
    a = [ref[k] for k in keys]
    b = [oth[k] for k in keys]
    mcn = mcnemar_test_paired(a, b)
    boot = bootstrap_mean_diff(a, b, n_resamples=bootstrap_n, seed=42)
    ci = boot.get("ci95", [None, None])
    return {
        "n": len(keys),
        "ref_success_rate": sum(a) / max(1, len(a)),
        "other_success_rate": sum(b) / max(1, len(b)),
        "delta_other_minus_ref": (sum(b) - sum(a)) / max(1, len(a)),
        "mcnemar_p_approx": mcn.get("p_approx"),
        "mcnemar_b": mcn.get("b"),
        "mcnemar_c": mcn.get("c"),
        "bootstrap_mean_ref_minus_other": boot.get("mean_diff"),
        "bootstrap_ci95_ref_minus_other": ci,
        "bootstrap_mean_other_minus_ref": -float(boot.get("mean_diff", 0.0)),
        "bootstrap_ci95_other_minus_ref": [
            -float(ci[1]) if ci[1] is not None else None,
            -float(ci[0]) if ci[0] is not None else None,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/eval_torch_kgqa.tune4.high_recall_multistage.cwq_webqsp.json",
    )
    parser.add_argument("--out_root", type=str, default="results/diagnostics/adaptive_alpha_seed42")
    parser.add_argument("--ae_ckpt", type=str, default="runs/ae_kgqa_torch_real_tune3_noise003/best.pt")
    parser.add_argument(
        "--planner_ckpt",
        type=str,
        default="runs/multiseed_cross_infonce_cwq_webqsp/seed_42/mlp_planner/best.pt",
    )
    parser.add_argument(
        "--value_ckpt",
        type=str,
        default="runs/multiseed_cross_infonce_cwq_webqsp/seed_42/value_cross_infonce/best.pt",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(cfg_root))

    base_cfg = load_json(str(repo / args.base_config))
    base_cfg["seed"] = int(args.seed)

    variants = [
        (
            "fixed_alpha020",
            {
                "prefix_step_penalty_alpha": 0.20,
                "prefix_step_penalty_gamma": 0.85,
                "adaptive_prefix_alpha_enabled": False,
            },
        ),
        (
            "adaptive_dataset",
            {
                "prefix_step_penalty_alpha": 0.20,
                "prefix_step_penalty_gamma": 0.85,
                "adaptive_prefix_alpha_enabled": True,
                "adaptive_prefix_alpha_mode": "dataset",
                "adaptive_prefix_alpha_by_dataset": {"webqsp": 0.05, "cwq": 0.20},
            },
        ),
        (
            "adaptive_dataset_then_len",
            {
                "prefix_step_penalty_alpha": 0.20,
                "prefix_step_penalty_gamma": 0.85,
                "adaptive_prefix_alpha_enabled": True,
                "adaptive_prefix_alpha_mode": "dataset_then_len",
                "adaptive_prefix_alpha_by_dataset": {"webqsp": 0.05, "cwq": 0.20},
                "adaptive_prefix_alpha_query_len_short_thr": 10,
                "adaptive_prefix_alpha_query_len_long_thr": 18,
                "adaptive_prefix_alpha_short": 0.05,
                "adaptive_prefix_alpha_mid": 0.12,
                "adaptive_prefix_alpha_long": 0.20,
            },
        ),
    ]

    py_exec = sys.executable
    metrics_rows = []
    preds_by_variant: Dict[str, List[Dict]] = {}
    by_dataset = {}

    for name, patch in variants:
        cfg = dict(base_cfg)
        cfg.update(patch)
        cfg_path = cfg_root / f"{name}.json"
        _write_json(cfg_path, cfg)

        run_out = out_root / name
        ensure_dir(str(run_out))
        _run(
            [
                py_exec,
                "evaluate_torch.py",
                "--config",
                str(cfg_path),
                "--ae_ckpt",
                str(repo / args.ae_ckpt),
                "--planner_ckpt",
                str(repo / args.planner_ckpt),
                "--value_ckpt",
                str(repo / args.value_ckpt),
                "--out",
                str(run_out),
            ],
            cwd=repo,
        )

        sm = load_json(str(run_out / "summary_metrics.json"))
        m = sm[next(iter(sm.keys()))]
        metrics_rows.append(_metric_row(name, m))
        preds = load_jsonl(str(run_out / "predictions.jsonl"))
        preds_by_variant[name] = preds
        by_dataset[name] = _dataset_breakdown(preds)

    # Save metrics csv/json.
    metrics_csv = out_root / "adaptive_alpha_metrics.csv"
    if metrics_rows:
        header = sorted(metrics_rows[0].keys())
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in metrics_rows:
                w.writerow(r)

    # Significance against fixed alpha.
    ref_label = "fixed_alpha020"
    sig = {}
    for name, preds in preds_by_variant.items():
        if name == ref_label:
            continue
        sig[name] = _paired_significance(preds_by_variant[ref_label], preds, bootstrap_n=int(args.bootstrap))

    dump_json(
        str(out_root / "adaptive_alpha_summary.json"),
        {
            "seed": int(args.seed),
            "reference": ref_label,
            "metrics_rows": metrics_rows,
            "dataset_breakdown": by_dataset,
            "significance_vs_reference": sig,
        },
    )
    print(f"[ok] outputs written to {out_root}")


if __name__ == "__main__":
    main()
