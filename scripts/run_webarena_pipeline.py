import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir


def _run(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--python_exec", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--processed_dir", type=str, default="data/webarena_processed")
    parser.add_argument("--out_root", type=str, default="results/webarena_seed42")
    parser.add_argument("--runs_root", type=str, default="runs/webarena_seed42")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_root))

    train_path = (repo / args.processed_dir / "webarena_train.jsonl").resolve()
    test_path = (repo / args.processed_dir / "webarena_test.jsonl").resolve()
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing WebArena processed data. Expected: {train_path} and {test_path}. "
            "Run scripts/prepare_webarena_data.py first."
        )

    # 1) Train AE
    ae_cfg = {
        "train_path": str(train_path),
        "seed": int(args.seed),
        "min_freq": 1,
        "max_path_len": 96,
        "batch_size": 32,
        "emb_dim": 128,
        "hid_dim": 192,
        "latent_dim": 128,
        "lr": 3e-4,
        "epochs": 8,
        "length_loss_weight": 0.25,
        "latent_noise_std": 0.1,
        "use_cuda": bool(args.use_cuda),
    }
    ae_cfg_path = cfg_root / "ae.json"
    _write_json(ae_cfg_path, ae_cfg)
    ae_out = runs_root / "ae"
    _run([args.python_exec, "train_autoencoder_torch.py", "--config", str(ae_cfg_path), "--out", str(ae_out)], repo)
    ae_ckpt = ae_out / "best.pt"

    # 2) Train MLP planner
    planner_cfg = {
        "train_path": str(train_path),
        "seed": int(args.seed),
        "max_path_len": 96,
        "max_query_len": 96,
        "batch_size": 32,
        "q_emb_dim": 128,
        "hidden_dim": 256,
        "lr": 3e-4,
        "epochs": 8,
        "use_cuda": bool(args.use_cuda),
    }
    planner_cfg_path = cfg_root / "planner.json"
    _write_json(planner_cfg_path, planner_cfg)
    planner_out = runs_root / "mlp_planner"
    _run(
        [args.python_exec, "train_mlp_planner.py", "--config", str(planner_cfg_path), "--ae_ckpt", str(ae_ckpt), "--out", str(planner_out)],
        repo,
    )
    planner_ckpt = planner_out / "best.pt"

    # 3) Train-split eval without value (mine candidate pool / hard negatives)
    eval_train_cfg = {
        "test_path": str(train_path),
        "train_path": str(train_path),
        "seed": int(args.seed),
        "num_candidates": 32,
        "receding_horizon": False,
        "use_value_model": False,
        "use_cuda": bool(args.use_cuda),
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
        "rerank_stage2_length_penalty_alpha": 0.08,
        "save_candidate_pool_topk": 32,
    }
    eval_train_cfg_path = cfg_root / "eval_train_no_value.json"
    _write_json(eval_train_cfg_path, eval_train_cfg)
    train_eval_out = out_root / "train_eval_no_value"
    _run(
        [args.python_exec, "evaluate_torch.py", "--config", str(eval_train_cfg_path), "--ae_ckpt", str(ae_ckpt), "--planner_ckpt", str(planner_ckpt), "--out", str(train_eval_out)],
        repo,
    )

    # 4) Train value model (cross InfoNCE + hardneg)
    value_cfg = {
        "train_path": str(train_path),
        "seed": int(args.seed),
        "max_path_len": 96,
        "max_query_len": 96,
        "neg_per_pos": 2,
        "batch_size": 64,
        "emb_dim": 128,
        "lr": 2.5e-4,
        "epochs": 6,
        "use_cuda": bool(args.use_cuda),
        "value_architecture": "cross",
        "value_hidden_dim": 256,
        "value_dropout": 0.1,
        "training_mode": "infonce",
        "infonce_temperature": 0.2,
        "infonce_num_negatives": 10,
        "hard_negative_predictions_path": str(train_eval_out / "predictions.jsonl"),
        "hard_neg_max_per_query": 2,
        "hard_neg_nearmiss_repeat": 2,
        "prefix_neg_per_pos": 1,
        "prefix_neg_repeat": 1,
    }
    value_cfg_path = cfg_root / "value.json"
    _write_json(value_cfg_path, value_cfg)
    value_out = runs_root / "value_cross_infonce"
    _run(
        [args.python_exec, "train_value_model_torch.py", "--config", str(value_cfg_path), "--planner_ckpt", str(planner_ckpt), "--out", str(value_out)],
        repo,
    )
    value_ckpt = value_out / "best.pt"

    # 5) Test eval with value
    eval_test_cfg = dict(eval_train_cfg)
    eval_test_cfg.update(
        {
            "test_path": str(test_path),
            "train_path": str(train_path),
            "use_value_model": True,
            "receding_horizon": True,
            "save_episode_trace": True,
            "save_candidate_pool_topk": 12,
        }
    )
    eval_test_cfg_path = cfg_root / "eval_test_with_value.json"
    _write_json(eval_test_cfg_path, eval_test_cfg)
    test_eval_out = out_root / "test_with_value_eval"
    _run(
        [
            args.python_exec,
            "evaluate_torch.py",
            "--config",
            str(eval_test_cfg_path),
            "--ae_ckpt",
            str(ae_ckpt),
            "--planner_ckpt",
            str(planner_ckpt),
            "--value_ckpt",
            str(value_ckpt),
            "--out",
            str(test_eval_out),
        ],
        repo,
    )

    manifest = {
        "seed": int(args.seed),
        "processed_dir": str(Path(args.processed_dir)),
        "checkpoints": {"ae_ckpt": str(ae_ckpt), "planner_ckpt": str(planner_ckpt), "value_ckpt": str(value_ckpt)},
        "outputs": {
            "train_eval_no_value": str(train_eval_out),
            "test_with_value_eval": str(test_eval_out),
            "summary_metrics": str(test_eval_out / "summary_metrics.json"),
            "summary_by_dataset": str(test_eval_out / "summary_by_dataset.json"),
        },
    }
    dump_json(str(out_root / "pipeline_manifest.json"), manifest)
    print(f"[ok] webarena pipeline completed: {out_root / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    main()

