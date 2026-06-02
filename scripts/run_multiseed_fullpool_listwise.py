import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
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


def _metric_mean_std(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(mean(vals)), "std": float(stdev(vals))}


def _dataset_breakdown(pred_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in pred_rows:
        grouped[r.get("dataset", "unknown")].append(r)
    out = {}
    for ds, recs in grouped.items():
        n = len(recs)
        out[ds] = {
            "n": n,
            "success_rate": sum(1.0 if x.get("success") else 0.0 for x in recs) / max(1, n),
            "candidate_pool_hit_rate": sum(1.0 if x.get("oracle_in_candidate_pool", False) else 0.0 for x in recs)
            / max(1, n),
            "ranking_error_rate": sum(1.0 if x.get("ranking_error", False) else 0.0 for x in recs) / max(1, n),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--train_path", type=str, default="data/real_processed/kgqa_train.jsonl")
    parser.add_argument("--test_path", type=str, default="data/real_processed/kgqa_test.jsonl")
    parser.add_argument("--ae_ckpt", type=str, default="runs/ae_kgqa_torch_real_tune3_noise003/best.pt")
    parser.add_argument(
        "--planner_root",
        type=str,
        default="runs/multiseed_cross_infonce_cwq_webqsp",
        help="Root containing seed_x/mlp_planner/best.pt",
    )
    parser.add_argument(
        "--baseline_root",
        type=str,
        default="results/multiseed_cross_infonce_alpha020_cwq_webqsp",
        help="Stage-4 baseline root for paired significance.",
    )
    parser.add_argument(
        "--baseline_value_root",
        type=str,
        default="runs/multiseed_cross_infonce_cwq_webqsp",
        help="Root containing seed_x/value_cross_infonce/best.pt (only used as required arg in no-value eval).",
    )
    parser.add_argument(
        "--baseline_run_name",
        type=str,
        default="mlp_memory_prefilter_cross_infonce",
        help="Baseline run directory name under baseline_root/seed_x.",
    )
    parser.add_argument(
        "--baseline_run_template",
        type=str,
        default="",
        help=(
            "Optional explicit baseline run path template. May contain {seed}. "
            "Overrides baseline_root/seed_x/baseline_run_name when set."
        ),
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default="stage4_cross_infonce",
        help="Label used for the baseline in summary and significance outputs.",
    )
    parser.add_argument(
        "--train_eval_cfg_base",
        type=str,
        default="configs/eval_torch_kgqa.tune5.train_pool_export.seed42.json",
    )
    parser.add_argument(
        "--test_eval_cfg_base",
        type=str,
        default="configs/eval_torch_kgqa.tune4.high_recall_multistage.cwq_webqsp.prefixpen.json",
    )
    parser.add_argument(
        "--value_cfg_base",
        type=str,
        default="configs/value_torch_kgqa.tune5.full_pool_listwise.json",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="results/multiseed_fullpool_listwise_cwq_webqsp",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="runs/multiseed_fullpool_listwise_cwq_webqsp",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--retrieval_pool_aware",
        action="store_true",
        help="Train listwise value model only on rows where gold appears in retrieved candidate pools.",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_root))

    py_exec = sys.executable
    train_eval_cfg_base = load_json(str(repo / args.train_eval_cfg_base))
    test_eval_cfg_base = load_json(str(repo / args.test_eval_cfg_base))
    value_cfg_base = load_json(str(repo / args.value_cfg_base))

    by_seed_summary: Dict[str, List[Dict]] = defaultdict(list)
    by_seed_preds: Dict[str, Dict[int, List[Dict]]] = defaultdict(dict)

    for seed in args.seeds:
        print(f"\n===== seed={seed} =====")
        seed_cfg_dir = cfg_root / f"seed_{seed}"
        seed_out_dir = out_root / f"seed_{seed}"
        seed_run_dir = runs_root / f"seed_{seed}"
        ensure_dir(str(seed_cfg_dir))
        ensure_dir(str(seed_out_dir))
        ensure_dir(str(seed_run_dir))

        planner_ckpt = repo / args.planner_root / f"seed_{seed}" / "mlp_planner" / "best.pt"
        baseline_value_ckpt = repo / args.baseline_value_root / f"seed_{seed}" / "value_cross_infonce" / "best.pt"
        if not planner_ckpt.exists():
            raise FileNotFoundError(f"Missing planner_ckpt: {planner_ckpt}")
        if not baseline_value_ckpt.exists():
            raise FileNotFoundError(f"Missing baseline value_ckpt: {baseline_value_ckpt}")

        # 1) Train-split candidate pool export using no-value evaluation.
        train_eval_cfg = dict(train_eval_cfg_base)
        train_eval_cfg.update(
            {
                "seed": int(seed),
                "train_path": args.train_path,
                "test_path": args.train_path,
                "use_cuda": bool(args.use_cuda),
                "use_value_model": False,
                "use_memory_retrieval": True,
                "save_candidate_pool_topk": 32,
            }
        )
        train_eval_cfg_path = seed_cfg_dir / "eval_train_pool_export.json"
        _write_json(train_eval_cfg_path, train_eval_cfg)
        train_eval_out = seed_out_dir / "train_eval_no_value_pool32"
        _run(
            [
                py_exec,
                "evaluate_torch.py",
                "--config",
                str(train_eval_cfg_path),
                "--ae_ckpt",
                str(repo / args.ae_ckpt),
                "--planner_ckpt",
                str(planner_ckpt),
                "--value_ckpt",
                str(baseline_value_ckpt),
                "--out",
                str(train_eval_out),
            ],
            cwd=repo,
        )

        # 2) Build full-pool listwise training file.
        full_pool_path = seed_out_dir / "train_full_pool_candidates.jsonl"
        _run(
            [
                py_exec,
                "scripts/build_full_pool_listwise_data.py",
                "--predictions",
                str(train_eval_out / "predictions.jsonl"),
                "--train_path",
                args.train_path,
                "--out",
                str(full_pool_path),
                "--pool_size",
                "32",
                "--seed",
                str(seed),
                "--use_planned_and_executed",
            ]
            + (["--require_gold_in_pool", "--no_synthetic_negatives"] if args.retrieval_pool_aware else []),
            cwd=repo,
        )

        # 3) Train value model with full-pool listwise CE.
        value_cfg = dict(value_cfg_base)
        value_cfg.update(
            {
                "seed": int(seed),
                "train_path": args.train_path,
                "use_cuda": bool(args.use_cuda),
                "full_pool_candidates_path": str(full_pool_path),
            }
        )
        value_cfg_path = seed_cfg_dir / "value_full_pool_listwise.json"
        _write_json(value_cfg_path, value_cfg)
        value_ckpt_dir = seed_run_dir / "value_full_pool_listwise"
        _run(
            [
                py_exec,
                "train_value_model_torch.py",
                "--config",
                str(value_cfg_path),
                "--planner_ckpt",
                str(planner_ckpt),
                "--out",
                str(value_ckpt_dir),
            ],
            cwd=repo,
        )

        # 4) Evaluate stage-5 model on test split.
        test_eval_cfg = dict(test_eval_cfg_base)
        test_eval_cfg.update(
            {
                "seed": int(seed),
                "train_path": args.train_path,
                "test_path": args.test_path,
                "use_cuda": bool(args.use_cuda),
            }
        )
        test_eval_cfg_path = seed_cfg_dir / "eval_test_stage5.json"
        _write_json(test_eval_cfg_path, test_eval_cfg)
        stage5_out = seed_out_dir / "mlp_memory_prefilter_cross_fullpool"
        _run(
            [
                py_exec,
                "evaluate_torch.py",
                "--config",
                str(test_eval_cfg_path),
                "--ae_ckpt",
                str(repo / args.ae_ckpt),
                "--planner_ckpt",
                str(planner_ckpt),
                "--value_ckpt",
                str(value_ckpt_dir / "best.pt"),
                "--out",
                str(stage5_out),
            ],
            cwd=repo,
        )

        # 5) Load stage-4 baseline and stage-5 metrics/predictions.
        if args.baseline_run_template:
            stage4_out = Path(args.baseline_run_template.format(seed=seed))
            if not stage4_out.is_absolute():
                stage4_out = repo / stage4_out
        else:
            stage4_out = repo / args.baseline_root / f"seed_{seed}" / args.baseline_run_name
        if not stage4_out.exists():
            raise FileNotFoundError(f"Missing stage-4 baseline dir: {stage4_out}")

        run_map = {
            args.baseline_label: stage4_out,
            "stage5_cross_fullpool_listwise": stage5_out,
        }
        for label, out_dir in run_map.items():
            sm = load_json(str(out_dir / "summary_metrics.json"))
            method = next(iter(sm.keys()))
            metrics = sm[method]
            by_seed_summary[label].append(metrics)
            by_seed_preds[label][seed] = load_jsonl(str(out_dir / "predictions.jsonl"))

    # Aggregate mean/std table.
    rows = []
    for setting, metrics_by_seed in by_seed_summary.items():
        keys = sorted({k for m in metrics_by_seed for k in m.keys()})
        row = {"setting": setting, "n_seeds": len(metrics_by_seed)}
        for k in keys:
            vals = [float(m.get(k, 0.0)) for m in metrics_by_seed]
            ms = _metric_mean_std(vals)
            row[f"{k}_mean"] = ms["mean"]
            row[f"{k}_std"] = ms["std"]
        rows.append(row)
    csv_path = out_root / "multiseed_summary_mean_std.csv"
    if rows:
        header = sorted(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    dump_json(str(out_root / "multiseed_summary_mean_std.json"), rows)

    # Dataset breakdown by seed.
    ds_breakdown = {}
    for setting, seed_map in by_seed_preds.items():
        ds_breakdown[setting] = {}
        for seed, preds in seed_map.items():
            ds_breakdown[setting][str(seed)] = _dataset_breakdown(preds)
    dump_json(str(out_root / "multiseed_dataset_breakdown.json"), ds_breakdown)

    # Paired significance stage5 vs stage4 (across seeds with task alignment per seed).
    ref_setting = args.baseline_label
    oth_setting = "stage5_cross_fullpool_listwise"
    a = []
    b = []
    for seed in args.seeds:
        ref_map = {str(r["task_id"]): 1 if r.get("success") else 0 for r in by_seed_preds[ref_setting][seed]}
        oth_map = {str(r["task_id"]): 1 if r.get("success") else 0 for r in by_seed_preds[oth_setting][seed]}
        keys = sorted(set(ref_map.keys()) & set(oth_map.keys()))
        a.extend([ref_map[k] for k in keys])
        b.extend([oth_map[k] for k in keys])
    mcn = mcnemar_test_paired(a, b)
    boot = bootstrap_mean_diff(a, b, n_resamples=int(args.bootstrap), seed=42)
    ci = boot.get("ci95", [None, None])
    sig = {
        "reference": ref_setting,
        "other": oth_setting,
        "n": len(a),
        "ref_success_rate": sum(a) / max(1, len(a)),
        "other_success_rate": sum(b) / max(1, len(b)),
        "delta_other_minus_ref": (sum(b) - sum(a)) / max(1, len(a)),
        "mcnemar_p_approx": mcn.get("p_approx"),
        "mcnemar_b": mcn.get("b"),
        "mcnemar_c": mcn.get("c"),
        "bootstrap_mean_diff_ref_minus_other": boot.get("mean_diff"),
        "bootstrap_ci95_ref_minus_other": ci,
        "bootstrap_mean_diff_other_minus_ref": -float(boot.get("mean_diff", 0.0)),
        "bootstrap_ci95_other_minus_ref": [
            -float(ci[1]) if ci[1] is not None else None,
            -float(ci[0]) if ci[0] is not None else None,
        ],
    }
    dump_json(str(out_root / "multiseed_significance_stage5_vs_stage4.json"), sig)
    print(f"[ok] outputs written to {out_root}")


if __name__ == "__main__":
    main()
