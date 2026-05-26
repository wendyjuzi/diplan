import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl
from src.diplan.stats_utils import bootstrap_mean_diff, mcnemar_test_paired


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


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
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--source_results_root", type=str, default="results/multiseed_scienceworld_easy")
    parser.add_argument("--source_runs_root", type=str, default="runs/multiseed_scienceworld_easy")
    parser.add_argument("--data_root", type=str, default="data/scienceworld_multiseed_easy")
    parser.add_argument("--out_root", type=str, default="results/multiseed_scienceworld_easy_fullpool_listwise")
    parser.add_argument("--runs_root", type=str, default="runs/multiseed_scienceworld_easy_fullpool_listwise")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    source_results_root = (repo / args.source_results_root).resolve()
    source_runs_root = (repo / args.source_runs_root).resolve()
    data_root = (repo / args.data_root).resolve()
    out_root = (repo / args.out_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_root))

    py_exec = args.python_exec
    by_setting_rows: Dict[str, List[Dict]] = defaultdict(list)
    by_setting_preds: Dict[str, Dict[int, List[Dict]]] = defaultdict(dict)

    for seed in args.seeds:
        print(f"\n===== seed={seed} =====")
        seed_src = source_results_root / f"seed_{seed}"
        seed_data = data_root / f"seed_{seed}"
        seed_out = out_root / f"seed_{seed}"
        seed_run = runs_root / f"seed_{seed}"
        ensure_dir(str(seed_out))
        ensure_dir(str(seed_run))
        ensure_dir(str(seed_out / "generated_configs"))

        base_train_path = seed_data / "scienceworld_train.jsonl"
        base_test_path = seed_data / "scienceworld_test.jsonl"
        train_eval_preds = seed_src / "train_eval_no_value" / "predictions.jsonl"
        if not train_eval_preds.exists():
            raise FileNotFoundError(f"Missing train eval predictions: {train_eval_preds}")
        if not base_train_path.exists() or not base_test_path.exists():
            raise FileNotFoundError(f"Missing ScienceWorld data for seed {seed}: {base_train_path} / {base_test_path}")

        planner_ckpt = source_runs_root / f"seed_{seed}" / "mlp_planner" / "best.pt"
        ae_ckpt = source_runs_root / f"seed_{seed}" / "ae" / "best.pt"
        if not planner_ckpt.exists():
            raise FileNotFoundError(f"Missing planner_ckpt: {planner_ckpt}")
        if not ae_ckpt.exists():
            raise FileNotFoundError(f"Missing ae_ckpt: {ae_ckpt}")

        # 1) Build full-pool training candidates.
        full_pool_path = seed_out / "train_full_pool_candidates.jsonl"
        _run(
            [
                py_exec,
                "scripts/build_full_pool_listwise_data.py",
                "--predictions",
                str(train_eval_preds),
                "--train_path",
                str(base_train_path),
                "--out",
                str(full_pool_path),
                "--pool_size",
                "32",
                "--seed",
                str(seed),
                "--use_planned_and_executed",
            ],
            cwd=repo,
        )

        # 2) Train full-pool value model.
        base_value_cfg_path = seed_src / "generated_configs" / "value.json"
        base_value_cfg = load_json(str(base_value_cfg_path))
        fullpool_value_cfg = dict(base_value_cfg)
        fullpool_value_cfg.update(
            {
                "seed": int(seed),
                "train_path": str(base_train_path),
                "use_cuda": bool(args.use_cuda),
                "training_mode": "full_pool_listwise",
                "full_pool_candidates_path": str(full_pool_path),
                "full_pool_num_negatives": 31,
            }
        )
        value_cfg_path = seed_out / "generated_configs" / "value_full_pool_listwise.json"
        dump_json(str(value_cfg_path), fullpool_value_cfg)
        value_out = seed_run / "value_full_pool_listwise"
        _run(
            [
                py_exec,
                "train_value_model_torch.py",
                "--config",
                str(value_cfg_path),
                "--planner_ckpt",
                str(planner_ckpt),
                "--out",
                str(value_out),
            ],
            cwd=repo,
        )

        # 3) Evaluate full-pool model on test split.
        base_eval_cfg_path = seed_src / "generated_configs" / "eval_test_with_value.json"
        base_eval_cfg = load_json(str(base_eval_cfg_path))
        fullpool_eval_cfg = dict(base_eval_cfg)
        fullpool_eval_cfg.update(
            {
                "seed": int(seed),
                "train_path": str(base_train_path),
                "test_path": str(base_test_path),
                "use_cuda": bool(args.use_cuda),
                "use_value_model": True,
                "save_candidate_pool_topk": 12,
            }
        )
        eval_cfg_path = seed_out / "generated_configs" / "eval_test_fullpool.json"
        dump_json(str(eval_cfg_path), fullpool_eval_cfg)
        stage5_out = seed_out / "test_with_value_eval_fullpool"
        _run(
            [
                py_exec,
                "evaluate_torch.py",
                "--config",
                str(eval_cfg_path),
                "--ae_ckpt",
                str(ae_ckpt),
                "--planner_ckpt",
                str(planner_ckpt),
                "--value_ckpt",
                str(value_out / "best.pt"),
                "--out",
                str(stage5_out),
            ],
            cwd=repo,
        )

        baseline_out = seed_src / "test_with_value_eval"
        if not baseline_out.exists():
            raise FileNotFoundError(f"Missing baseline eval dir: {baseline_out}")

        run_map = {
            "baseline_cross_infonce": baseline_out,
            "stage5_fullpool_listwise": stage5_out,
        }
        for label, out_dir in run_map.items():
            sm = load_json(str(out_dir / "summary_metrics.json"))
            method = next(iter(sm.keys()))
            metrics = sm[method]
            by_setting_rows[label].append(metrics)
            by_setting_preds[label][seed] = load_jsonl(str(out_dir / "predictions.jsonl"))

    rows = []
    for setting, metrics_by_seed in by_setting_rows.items():
        keys = sorted({k for m in metrics_by_seed for k in m.keys()})
        row = {"setting": setting, "n_seeds": len(metrics_by_seed)}
        for k in keys:
            vals = [float(m.get(k, 0.0)) for m in metrics_by_seed]
            ms = _metric_mean_std(vals)
            row[f"{k}_mean"] = ms["mean"]
            row[f"{k}_std"] = ms["std"]
        rows.append(row)
    if rows:
        header = sorted(rows[0].keys())
        with (out_root / "multiseed_summary_mean_std.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    dump_json(str(out_root / "multiseed_summary_mean_std.json"), rows)

    ds_breakdown = {}
    for setting, seed_map in by_setting_preds.items():
        ds_breakdown[setting] = {}
        for seed, preds in seed_map.items():
            ds_breakdown[setting][str(seed)] = _dataset_breakdown(preds)
    dump_json(str(out_root / "multiseed_dataset_breakdown.json"), ds_breakdown)

    ref_setting = "baseline_cross_infonce"
    oth_setting = "stage5_fullpool_listwise"
    a = []
    b = []
    for seed in args.seeds:
        ref_map = {str(r["task_id"]): 1 if r.get("success") else 0 for r in by_setting_preds[ref_setting][seed]}
        oth_map = {str(r["task_id"]): 1 if r.get("success") else 0 for r in by_setting_preds[oth_setting][seed]}
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
    dump_json(str(out_root / "multiseed_significance_stage5_vs_baseline.json"), sig)
    print(f"[ok] outputs written to {out_root}")


if __name__ == "__main__":
    main()
