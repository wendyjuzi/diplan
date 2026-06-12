import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_config


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.replace(",", " ").split() if x.strip()]


def _metric_mean_std(vals: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in vals]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": float(mean(vals)), "std": float(stdev(vals))}


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run(cmd: List[str], cwd: Path, log_path: Path) -> None:
    ensure_dir(str(log_path.parent))
    print("[run]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _memory_top_k(pool: int, multiplier: float, minimum: int) -> int:
    return max(int(minimum), int(round(float(pool) * float(multiplier))))


def _load_summary(path: Path) -> Dict:
    raw = json.load(path.open("r", encoding="utf-8"))
    if len(raw) != 1:
        raise ValueError(f"Expected one method in {path}, got keys={list(raw)}")
    return next(iter(raw.values()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run KGQA candidate-pool oracle recall curves across pool sizes and seeds. "
            "This is intentionally a no-value evaluation: it measures whether the "
            "gold/oracle path enters the candidate pool, not final reranking."
        )
    )
    parser.add_argument("--seeds", type=str, default="42 43 44")
    parser.add_argument("--pools", type=str, default="16 32 48 64 96")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/eval_torch_kgqa.tune4.high_recall_multistage.cwq_webqsp.prefixpen.json",
    )
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
        "--value_root",
        type=str,
        default="runs/multiseed_cross_infonce_cwq_webqsp",
        help="Root containing seed_x/value_cross_infonce/best.pt; passed only to satisfy evaluate_torch args.",
    )
    parser.add_argument("--out_root", type=str, default="results/kgqa_pool_recall_curve")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--memory_top_k_multiplier", type=float, default=1.5)
    parser.add_argument("--memory_top_k_min", type=int, default=32)
    parser.add_argument("--memory_max_postings_per_token", type=int, default=3200)
    parser.add_argument("--candidate_multi_jitter_stds", type=str, default="0.0 0.03 0.06")
    args = parser.parse_args()

    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if not os.environ.get(env_name, "").isdigit():
            os.environ[env_name] = "1"

    seeds = _parse_ints(args.seeds)
    pools = _parse_ints(args.pools)
    jitter = _parse_floats(args.candidate_multi_jitter_stds)
    out_root = ROOT / args.out_root
    cfg_root = out_root / "generated_configs"
    runs_root = out_root / "runs"
    log_root = out_root / "logs"
    ensure_dir(str(out_root))

    manifest = {
        "seeds": seeds,
        "pools": pools,
        "base_config": args.base_config,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "ae_ckpt": args.ae_ckpt,
        "planner_root": args.planner_root,
        "value_root": args.value_root,
        "candidate_multi_jitter_stds": jitter,
        "runs": [],
    }

    base_cfg = load_config(str(ROOT / args.base_config))
    py_exec = sys.executable
    per_seed_rows: List[Dict] = []

    for seed in seeds:
        planner_ckpt = ROOT / args.planner_root / f"seed_{seed}" / "mlp_planner" / "best.pt"
        value_ckpt = ROOT / args.value_root / f"seed_{seed}" / "value_cross_infonce" / "best.pt"
        if not planner_ckpt.exists():
            raise FileNotFoundError(f"Missing planner checkpoint: {planner_ckpt}")
        if not value_ckpt.exists():
            raise FileNotFoundError(f"Missing value checkpoint: {value_ckpt}")

        for pool in pools:
            label = f"seed_{seed}_pool_{pool}"
            run_dir = runs_root / f"seed_{seed}" / f"pool_{pool}"
            cfg_path = cfg_root / f"seed_{seed}" / f"pool_{pool}.json"
            log_path = log_root / f"seed_{seed}" / f"pool_{pool}.log"

            cfg = dict(base_cfg)
            cfg.update(
                {
                    "seed": int(seed),
                    "train_path": args.train_path,
                    "test_path": args.test_path,
                    "use_cuda": bool(args.use_cuda),
                    "use_value_model": False,
                    "use_memory_retrieval": True,
                    "memory_prefilter_feasible": True,
                    "num_candidates": int(pool),
                    "rerank_stage1_topk": int(min(pool, max(12, pool // 2))),
                    "save_candidate_pool_topk": int(pool),
                    "memory_top_k": _memory_top_k(pool, args.memory_top_k_multiplier, args.memory_top_k_min),
                    "memory_max_postings_per_token": int(args.memory_max_postings_per_token),
                    "candidate_multi_jitter_stds": list(jitter),
                    "candidate_latent_jitter_std": float(max(jitter)) if jitter else 0.0,
                }
            )
            _write_json(cfg_path, cfg)
            manifest["runs"].append({"label": label, "run_dir": str(run_dir), "config": str(cfg_path)})

            if args.skip_existing and (run_dir / "summary_metrics.json").exists():
                print(f"[skip] {label}: {run_dir}")
            else:
                cmd = [
                    py_exec,
                    "evaluate_torch.py",
                    "--config",
                    str(cfg_path),
                    "--ae_ckpt",
                    str(ROOT / args.ae_ckpt),
                    "--planner_ckpt",
                    str(planner_ckpt),
                    "--value_ckpt",
                    str(value_ckpt),
                    "--out",
                    str(run_dir),
                ]
                _run(cmd, ROOT, log_path)

            metrics = _load_summary(run_dir / "summary_metrics.json")
            n_path = run_dir / "predictions.jsonl"
            n = sum(1 for _ in n_path.open("r", encoding="utf-8")) if n_path.exists() else metrics.get("n", 0)
            per_seed_rows.append(
                {
                    "seed": int(seed),
                    "pool": int(pool),
                    "n": int(n),
                    "success_rate": float(metrics.get("success_rate", 0.0)),
                    "candidate_pool_hit_rate": float(metrics.get("candidate_pool_hit_rate", 0.0)),
                    "oracle_mrr": float(metrics.get("oracle_mrr", 0.0)),
                    "oracle_hit_at_1": float(metrics.get("oracle_hit_at_1", 0.0)),
                    "oracle_hit_at_3": float(metrics.get("oracle_hit_at_3", 0.0)),
                    "oracle_hit_at_5": float(metrics.get("oracle_hit_at_5", 0.0)),
                    "oracle_rank_mean": float(metrics.get("oracle_rank_mean", 0.0)),
                    "candidate_pool_avg_size": float(metrics.get("candidate_pool_avg_size", 0.0)),
                    "conditional_success_given_pool_hit": float(
                        metrics.get("conditional_success_given_pool_hit", 0.0)
                    ),
                    "ranking_error_rate": float(metrics.get("ranking_error_rate", 0.0)),
                    "path": str(run_dir),
                }
            )

    aggregate_rows: List[Dict] = []
    metric_keys = [
        "success_rate",
        "candidate_pool_hit_rate",
        "oracle_mrr",
        "oracle_hit_at_1",
        "oracle_hit_at_3",
        "oracle_hit_at_5",
        "oracle_rank_mean",
        "candidate_pool_avg_size",
        "conditional_success_given_pool_hit",
        "ranking_error_rate",
    ]
    for pool in pools:
        subset = [r for r in per_seed_rows if r["pool"] == pool]
        row = {"pool": int(pool), "n_seeds": len(subset), "n": subset[0]["n"] if subset else 0}
        for key in metric_keys:
            ms = _metric_mean_std(r[key] for r in subset)
            row[f"{key}_mean"] = ms["mean"]
            row[f"{key}_std"] = ms["std"]
        aggregate_rows.append(row)

    tables_dir = out_root / "tables"
    ensure_dir(str(tables_dir))
    per_seed_csv = tables_dir / "pool_recall_per_seed.csv"
    agg_csv = tables_dir / "pool_recall_mean_std.csv"
    if per_seed_rows:
        with per_seed_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_seed_rows)
    if aggregate_rows:
        with agg_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    dump_json(str(tables_dir / "pool_recall_per_seed.json"), {"rows": per_seed_rows})
    dump_json(str(tables_dir / "pool_recall_mean_std.json"), {"rows": aggregate_rows})
    dump_json(str(out_root / "pool_recall_curve_manifest.json"), manifest)
    print(f"[ok] wrote {per_seed_csv}")
    print(f"[ok] wrote {agg_csv}")
    print("[summary]")
    for row in aggregate_rows:
        print(
            f"Pool@{row['pool']}: recall={row['candidate_pool_hit_rate_mean']:.4f} "
            f"+/- {row['candidate_pool_hit_rate_std']:.4f}, "
            f"MRR={row['oracle_mrr_mean']:.4f} +/- {row['oracle_mrr_std']:.4f}"
        )


if __name__ == "__main__":
    main()
