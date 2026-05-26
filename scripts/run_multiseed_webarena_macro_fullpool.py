import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


def _binom_two_sided_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * prob)


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0}
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": m, "std": 0.0}
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": m, "std": math.sqrt(max(0.0, v))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to run all scripts.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="data/webarena_raw/webarena_from_trace_aug.jsonl",
        help="WebArena converted trajectories file.",
    )
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--summary_out", type=str, default="results/multiseed_webarena_macro_fullpool")
    args = parser.parse_args()

    py = args.python_exec
    summary_root = (REPO / args.summary_out).resolve()
    summary_root.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    pooled_b = 0
    pooled_c = 0

    for seed in args.seeds:
        processed_dir = f"data/webarena_processed_macro_seed{seed}"
        out_root = f"results/webarena_seed{seed}_macro"
        runs_root = f"runs/webarena_seed{seed}_macro"

        # 1) Prepare macro_dedup data for this seed.
        _run(
            [
                py,
                "scripts/prepare_webarena_data.py",
                "--inputs",
                args.inputs,
                "--out",
                processed_dir,
                "--seed",
                str(seed),
                "--only_success",
                "--path_mode",
                "macro_dedup",
                "--max_rows",
                str(args.max_rows),
            ]
        )

        # 2) Base pipeline (AE -> planner -> train-no-value eval -> infonce value -> test-with-value eval).
        _run(
            [
                py,
                "scripts/run_webarena_pipeline.py",
                "--python_exec",
                py,
                "--seed",
                str(seed),
                "--processed_dir",
                processed_dir,
                "--out_root",
                out_root,
                "--runs_root",
                runs_root,
            ]
        )

        # 3) Same test split, no-value evaluation for strict same-pool baseline.
        eval_cfg_path = REPO / out_root / "generated_configs" / "eval_test_with_value.json"
        eval_no_value_cfg_path = REPO / out_root / "generated_configs" / "eval_test_no_value.json"
        eval_cfg = json.loads(eval_cfg_path.read_text(encoding="utf-8"))
        eval_cfg["use_value_model"] = False
        eval_no_value_cfg_path.write_text(json.dumps(eval_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        _run(
            [
                py,
                "evaluate_torch.py",
                "--config",
                str(eval_no_value_cfg_path),
                "--ae_ckpt",
                str(REPO / runs_root / "ae" / "best.pt"),
                "--planner_ckpt",
                str(REPO / runs_root / "mlp_planner" / "best.pt"),
                "--out",
                str(REPO / out_root / "test_no_value_eval"),
            ]
        )

        # 4) Build full-pool listwise candidates.
        full_pool_path = REPO / out_root / "train_full_pool_candidates.jsonl"
        _run(
            [
                py,
                "scripts/build_full_pool_listwise_data.py",
                "--predictions",
                str(REPO / out_root / "train_eval_no_value" / "predictions.jsonl"),
                "--train_path",
                str(REPO / processed_dir / "webarena_train.jsonl"),
                "--out",
                str(full_pool_path),
                "--pool_size",
                "32",
                "--seed",
                str(seed),
                "--use_planned_and_executed",
            ]
        )

        # 5) Train full-pool listwise value model.
        value_fullpool_cfg = {
            "train_path": str((REPO / processed_dir / "webarena_train.jsonl").resolve()),
            "seed": int(seed),
            "max_path_len": 96,
            "max_query_len": 96,
            "batch_size": 64,
            "emb_dim": 128,
            "lr": 2.5e-4,
            "epochs": 8,
            "use_cuda": False,
            "value_architecture": "cross_terminal",
            "value_hidden_dim": 256,
            "value_dropout": 0.1,
            "training_mode": "full_pool_listwise",
            "infonce_temperature": 0.18,
            "full_pool_candidates_path": str(full_pool_path.resolve()),
            "full_pool_num_negatives": 31,
            "add_terminal_truncation_negatives": True,
            "terminal_neg_repeat": 2,
            "terminal_stage_prefix": "SUBMIT_OR_CONFIRM",
            "terminal_margin_gamma": 0.3,
            "terminal_margin_weight": 1.0,
        }
        value_fullpool_cfg_path = REPO / out_root / "generated_configs" / "value_fullpool_macro.json"
        value_fullpool_cfg_path.write_text(
            json.dumps(value_fullpool_cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fullpool_value_out = REPO / runs_root / "value_fullpool_listwise_macro"
        _run(
            [
                py,
                "train_value_model_torch.py",
                "--config",
                str(value_fullpool_cfg_path),
                "--planner_ckpt",
                str(REPO / runs_root / "mlp_planner" / "best.pt"),
                "--out",
                str(fullpool_value_out),
            ]
        )

        # 6) Evaluate with full-pool value model.
        eval_fullpool_out = REPO / out_root / "test_with_value_eval_fullpool_macro"
        _run(
            [
                py,
                "evaluate_torch.py",
                "--config",
                str(REPO / out_root / "generated_configs" / "eval_test_with_value.json"),
                "--ae_ckpt",
                str(REPO / runs_root / "ae" / "best.pt"),
                "--planner_ckpt",
                str(REPO / runs_root / "mlp_planner" / "best.pt"),
                "--value_ckpt",
                str(fullpool_value_out / "best.pt"),
                "--out",
                str(eval_fullpool_out),
            ]
        )

        # 7) Strict same-pool comparison: use NO-VALUE test candidate pools, apply value ranking.
        same_pool_out = REPO / out_root / "same_pool_compare_fullpool_macro"
        _run(
            [
                py,
                "scripts/eval_samepool_no_value_vs_value.py",
                "--predictions_no_value",
                str(REPO / out_root / "test_no_value_eval" / "predictions.jsonl"),
                "--value_ckpt",
                str(fullpool_value_out / "best.pt"),
                "--planner_ckpt",
                str(REPO / runs_root / "mlp_planner" / "best.pt"),
                "--ae_ckpt",
                str(REPO / runs_root / "ae" / "best.pt"),
                "--out",
                str(same_pool_out),
            ]
        )

        smry_path = same_pool_out / "same_pool_comparison_summary.json"
        smry = json.loads(smry_path.read_text(encoding="utf-8"))
        smry["seed"] = seed
        seed_rows.append(smry)
        pooled_b += int(smry.get("mcnemar_b_no_yes", 0))
        pooled_c += int(smry.get("mcnemar_c_yes_no", 0))

    deltas = [float(x.get("delta_with_minus_no_value", 0.0)) for x in seed_rows]
    hit_rates = [float(x.get("candidate_pool_hit_rate_same_pool", 0.0)) for x in seed_rows]
    no_sr = [float(x.get("no_value_top1_success_rate_same_pool", 0.0)) for x in seed_rows]
    with_sr = [float(x.get("with_value_top1_success_rate_same_pool", 0.0)) for x in seed_rows]

    final = {
        "seeds": args.seeds,
        "per_seed": seed_rows,
        "aggregate": {
            "delta_with_minus_no_value": _mean_std(deltas),
            "candidate_pool_hit_rate_same_pool": _mean_std(hit_rates),
            "no_value_top1_success_rate_same_pool": _mean_std(no_sr),
            "with_value_top1_success_rate_same_pool": _mean_std(with_sr),
            "positive_delta_seed_count": sum(1 for d in deltas if d > 0),
            "pooled_mcnemar_b_no_yes": pooled_b,
            "pooled_mcnemar_c_yes_no": pooled_c,
            "pooled_mcnemar_p_two_sided": _binom_two_sided_p_value(pooled_b, pooled_c),
        },
    }

    out_json = summary_root / "multiseed_same_pool_summary.json"
    out_json.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote multiseed summary -> {out_json}")
    print(json.dumps(final["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
