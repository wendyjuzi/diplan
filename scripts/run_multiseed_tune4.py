import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl
from src.diplan.stats_utils import bootstrap_mean_diff, mcnemar_test_paired


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _setting_eval_cfg(setting: str, seed: int, train_path: str, test_path: str, use_cuda: bool) -> Dict:
    common = {
        "train_path": train_path,
        "test_path": test_path,
        "seed": seed,
        "receding_horizon": False,
        "use_cuda": use_cuda,
    }
    if setting == "mlp_direct":
        common.update(
            {
                "num_candidates": 1,
                "use_value_model": False,
                "use_memory_retrieval": False,
            }
        )
        return common
    if setting == "mlp_memory_prefilter_no_value":
        common.update(
            {
                "num_candidates": 32,
                "use_value_model": False,
                "use_memory_retrieval": True,
                "memory_prefilter_feasible": True,
                "memory_top_k": 32,
                "memory_max_postings_per_token": 1800,
                "candidate_latent_jitter_std": 0.05,
                "candidate_multi_jitter_stds": [0.02, 0.05, 0.08],
                "use_expected_length_prior": True,
                "expected_length_bucket_size": 4,
                "length_penalty_alpha": 0.05,
                "save_candidate_pool_topk": 12,
            }
        )
        return common
    if setting in {"mlp_memory_prefilter_bce_value", "mlp_memory_prefilter_pairwise_value"}:
        common.update(
            {
                "num_candidates": 32,
                "use_value_model": True,
                "use_memory_retrieval": True,
                "memory_prefilter_feasible": True,
                "memory_top_k": 32,
                "memory_max_postings_per_token": 1800,
                "candidate_latent_jitter_std": 0.05,
                "candidate_multi_jitter_stds": [0.02, 0.05, 0.08],
                "use_expected_length_prior": True,
                "expected_length_bucket_size": 4,
                "length_penalty_alpha": 0.05,
                "rerank_stage1_topk": 8,
                "rerank_consensus_weight": 0.75,
                "rerank_prefix_consensus_weight": 0.9,
                "rerank_memory_bonus": 0.15,
                "rerank_memory_rank_bonus": 0.6,
                "rerank_stage2_length_penalty_alpha": 0.08,
                "save_candidate_pool_topk": 12,
            }
        )
        return common
    raise ValueError(f"Unknown setting: {setting}")


def _metric_mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(stdev(vals))


def _aggregate_multiseed_summary(
    by_seed_summary: Dict[str, List[Dict]],
    out_path: Path,
) -> None:
    metric_names = sorted(
        {
            k
            for _setting, seed_metrics in by_seed_summary.items()
            for m in seed_metrics
            for k in m.keys()
        }
    )
    table = {}
    for setting, seed_metrics in by_seed_summary.items():
        row = {}
        for mk in metric_names:
            vals = [float(m.get(mk, 0.0)) for m in seed_metrics]
            mu, sd = _metric_mean_std(vals)
            row[mk] = {"mean": mu, "std": sd}
        table[setting] = row
    dump_json(str(out_path), table)


def _build_dataset_breakdown(pred_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for r in pred_rows:
        grouped[r.get("dataset", "unknown")].append(r)
    out = {}
    for ds, recs in grouped.items():
        n = len(recs)
        out[ds] = {
            "n": n,
            "success_rate": sum(1.0 if x.get("success") else 0.0 for x in recs) / max(1, n),
            "plan_feasibility": sum(1.0 if x.get("feasible", False) else 0.0 for x in recs) / max(1, n),
            "candidate_pool_hit_rate": sum(
                1.0 if x.get("oracle_in_candidate_pool", False) else 0.0 for x in recs
            )
            / max(1, n),
        }
    return out


def _write_dataset_breakdown(
    by_seed_preds: Dict[str, Dict[int, List[Dict]]],
    out_path: Path,
) -> None:
    # Output as JSON for easier paper-table extraction.
    nested = {}
    for setting, seed_map in by_seed_preds.items():
        nested[setting] = {}
        for seed, preds in seed_map.items():
            nested[setting][str(seed)] = _build_dataset_breakdown(preds)
    dump_json(str(out_path), nested)


def _significance_vs_ref(
    ref_setting: str,
    by_seed_preds: Dict[str, Dict[int, List[Dict]]],
    out_path: Path,
    bootstrap_n: int,
    seed: int,
) -> None:
    ref_seed_map = by_seed_preds[ref_setting]
    ref = {}
    for s, rows in ref_seed_map.items():
        for r in rows:
            ref[(s, str(r["task_id"]))] = 1 if r.get("success") else 0

    results = []
    for setting, seed_map in by_seed_preds.items():
        if setting == ref_setting:
            continue
        oth = {}
        for s, rows in seed_map.items():
            for r in rows:
                oth[(s, str(r["task_id"]))] = 1 if r.get("success") else 0
        keys = sorted(set(ref.keys()) & set(oth.keys()))
        a = [ref[k] for k in keys]
        b = [oth[k] for k in keys]
        mcnemar = mcnemar_test_paired(a, b)
        boot = bootstrap_mean_diff(a, b, n_resamples=bootstrap_n, seed=seed)
        ci = boot.get("ci95", [None, None])
        results.append(
            {
                "reference": ref_setting,
                "other": setting,
                "n": len(keys),
                "ref_success_rate": sum(a) / max(1, len(a)),
                "other_success_rate": sum(b) / max(1, len(b)),
                "delta_other_minus_ref": (sum(b) - sum(a)) / max(1, len(a)),
                "mcnemar_p_approx": mcnemar.get("p_approx"),
                "bootstrap_mean_diff_ref_minus_other": boot.get("mean_diff"),
                "bootstrap_ci95_ref_minus_other": ci,
                "bootstrap_mean_diff_other_minus_ref": -float(boot.get("mean_diff", 0.0)),
                "bootstrap_ci95_other_minus_ref": [
                    -float(ci[1]) if ci[1] is not None else None,
                    -float(ci[0]) if ci[0] is not None else None,
                ],
            }
        )
    dump_json(str(out_path), {"reference": ref_setting, "comparisons": results})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--train_path", type=str, default="data/real_processed/kgqa_train.jsonl")
    parser.add_argument("--test_path", type=str, default="data/real_processed/kgqa_test.jsonl")
    parser.add_argument("--ae_ckpt", type=str, default="runs/ae_kgqa_torch_real_tune3_noise003/best.pt")
    parser.add_argument("--diffusion_cfg_base", type=str, default="configs/diffusion_torch_kgqa.tune3.json")
    parser.add_argument("--value_bce_cfg_base", type=str, default="configs/value_torch_kgqa.tune3.json")
    parser.add_argument("--value_pair_cfg_base", type=str, default="configs/value_torch_kgqa.tune3.pairwise.json")
    parser.add_argument("--out_root", type=str, default="results/multiseed_tune4")
    parser.add_argument("--runs_root", type=str, default="runs/multiseed_tune4")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--include_datasets", type=str, nargs="*", default=[])
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_root))

    base_diff = _read_json(repo / args.diffusion_cfg_base)
    base_bce = _read_json(repo / args.value_bce_cfg_base)
    base_pair = _read_json(repo / args.value_pair_cfg_base)
    py_exec = sys.executable

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

        # 1) Train MLP planner.
        diff_cfg = dict(base_diff)
        diff_cfg.update({"seed": seed, "train_path": args.train_path, "use_cuda": bool(args.use_cuda)})
        diff_cfg_path = seed_cfg_dir / "diffusion_mlp.json"
        _write_json(diff_cfg_path, diff_cfg)
        mlp_ckpt_dir = seed_run_dir / "mlp_planner"
        _run(
            [
                py_exec,
                "train_mlp_planner.py",
                "--config",
                str(diff_cfg_path),
                "--ae_ckpt",
                str(repo / args.ae_ckpt),
                "--out",
                str(mlp_ckpt_dir),
            ],
            cwd=repo,
        )

        # 2) Train BCE value model.
        bce_cfg = dict(base_bce)
        bce_cfg.update({"seed": seed, "train_path": args.train_path, "use_cuda": bool(args.use_cuda)})
        bce_cfg_path = seed_cfg_dir / "value_bce.json"
        _write_json(bce_cfg_path, bce_cfg)
        bce_ckpt_dir = seed_run_dir / "value_bce"
        _run(
            [
                py_exec,
                "train_value_model_torch.py",
                "--config",
                str(bce_cfg_path),
                "--planner_ckpt",
                str(mlp_ckpt_dir / "best.pt"),
                "--out",
                str(bce_ckpt_dir),
            ],
            cwd=repo,
        )

        # 3) Eval direct + memory no-value.
        eval_plan = [
            ("mlp_direct", bce_ckpt_dir / "best.pt"),
            ("mlp_memory_prefilter_no_value", bce_ckpt_dir / "best.pt"),
        ]
        eval_outputs = {}
        for setting, value_ckpt_for_call in eval_plan:
            eval_cfg = _setting_eval_cfg(
                setting=setting,
                seed=seed,
                train_path=args.train_path,
                test_path=args.test_path,
                use_cuda=bool(args.use_cuda),
            )
            if args.include_datasets:
                eval_cfg["include_datasets"] = [x.lower() for x in args.include_datasets]
            eval_cfg_path = seed_cfg_dir / f"eval_{setting}.json"
            _write_json(eval_cfg_path, eval_cfg)
            out_dir = seed_out_dir / setting
            _run(
                [
                    py_exec,
                    "evaluate_torch.py",
                    "--config",
                    str(eval_cfg_path),
                    "--ae_ckpt",
                    str(repo / args.ae_ckpt),
                    "--planner_ckpt",
                    str(mlp_ckpt_dir / "best.pt"),
                    "--value_ckpt",
                    str(value_ckpt_for_call),
                    "--out",
                    str(out_dir),
                ],
                cwd=repo,
            )
            eval_outputs[setting] = out_dir

        # 4) Train pairwise value model with hard negatives from no-value predictions.
        pair_cfg = dict(base_pair)
        pair_cfg.update(
            {
                "seed": seed,
                "train_path": args.train_path,
                "use_cuda": bool(args.use_cuda),
                "hard_negative_predictions_path": str(
                    eval_outputs["mlp_memory_prefilter_no_value"] / "predictions.jsonl"
                ),
                "hard_neg_per_query": 2,
                "hard_neg_from_candidate_pool": 2,
                "hard_neg_ranking_error_only": True,
                "hard_neg_prefer_nearmiss_firsthop": True,
                "hard_neg_nearmiss_repeat": 3,
            }
        )
        pair_cfg_path = seed_cfg_dir / "value_pairwise_hardneg.json"
        _write_json(pair_cfg_path, pair_cfg)
        pair_ckpt_dir = seed_run_dir / "value_pairwise_hardneg"
        _run(
            [
                py_exec,
                "train_value_model_torch.py",
                "--config",
                str(pair_cfg_path),
                "--planner_ckpt",
                str(mlp_ckpt_dir / "best.pt"),
                "--out",
                str(pair_ckpt_dir),
            ],
            cwd=repo,
        )

        # 5) Eval memory+BCE and memory+pairwise.
        eval_plan_2 = [
            ("mlp_memory_prefilter_bce_value", bce_ckpt_dir / "best.pt"),
            ("mlp_memory_prefilter_pairwise_value", pair_ckpt_dir / "best.pt"),
        ]
        for setting, value_ckpt in eval_plan_2:
            eval_cfg = _setting_eval_cfg(
                setting=setting,
                seed=seed,
                train_path=args.train_path,
                test_path=args.test_path,
                use_cuda=bool(args.use_cuda),
            )
            if args.include_datasets:
                eval_cfg["include_datasets"] = [x.lower() for x in args.include_datasets]
            eval_cfg_path = seed_cfg_dir / f"eval_{setting}.json"
            _write_json(eval_cfg_path, eval_cfg)
            out_dir = seed_out_dir / setting
            _run(
                [
                    py_exec,
                    "evaluate_torch.py",
                    "--config",
                    str(eval_cfg_path),
                    "--ae_ckpt",
                    str(repo / args.ae_ckpt),
                    "--planner_ckpt",
                    str(mlp_ckpt_dir / "best.pt"),
                    "--value_ckpt",
                    str(value_ckpt),
                    "--out",
                    str(out_dir),
                ],
                cwd=repo,
            )
            eval_outputs[setting] = out_dir

        # 6) Collect summaries and predictions.
        for setting, out_dir in eval_outputs.items():
            sm = load_json(str(out_dir / "summary_metrics.json"))
            method = next(iter(sm.keys()))
            metrics = sm[method]
            by_seed_summary[setting].append(metrics)
            by_seed_preds[setting][seed] = load_jsonl(str(out_dir / "predictions.jsonl"))

    # Output multiseed summary, dataset breakdown, and significance tests.
    _aggregate_multiseed_summary(by_seed_summary, out_root / "multiseed_summary_mean_std.json")
    _write_dataset_breakdown(by_seed_preds, out_root / "multiseed_dataset_breakdown.json")
    _significance_vs_ref(
        ref_setting="mlp_direct",
        by_seed_preds=by_seed_preds,
        out_path=out_root / "multiseed_significance_vs_direct.json",
        bootstrap_n=int(args.bootstrap),
        seed=42,
    )

    # Also output paper-friendly CSV.
    rows = []
    for setting, metrics_by_seed in by_seed_summary.items():
        keys = sorted({k for m in metrics_by_seed for k in m.keys()})
        row = {"setting": setting, "n_seeds": len(metrics_by_seed)}
        for k in keys:
            vals = [float(m.get(k, 0.0)) for m in metrics_by_seed]
            mu, sd = _metric_mean_std(vals)
            row[f"{k}_mean"] = mu
            row[f"{k}_std"] = sd
        rows.append(row)
    csv_path = out_root / "multiseed_summary_mean_std.csv"
    ensure_dir(str(csv_path.parent))
    if rows:
        header = sorted(rows[0].keys())
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(h, "")) for h in header) + "\n")

    print(f"[ok] outputs written to {out_root}")


if __name__ == "__main__":
    main()
