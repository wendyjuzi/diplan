import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(cmd, cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/real_processed/kgqa_train.jsonl")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="experiments/sanity_overfit10")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--prediction_target", type=str, default="z0")
    parser.add_argument("--planner_type", type=str, default="diffusion", choices=["diffusion", "mlp"])
    parser.add_argument("--ae_latent_noise_std", type=float, default=0.1)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--receding_horizon", action="store_true")
    parser.add_argument("--use_value_model", action="store_true")
    parser.add_argument("--use_memory_retrieval", action="store_true")
    parser.add_argument("--memory_top_k", type=int, default=10)
    parser.add_argument("--memory_max_postings_per_token", type=int, default=1000)
    args = parser.parse_args()

    root = Path(".").resolve()
    out_dir = root / args.out_dir
    data_dir = out_dir / "data"
    cfg_dir = out_dir / "configs"
    runs_dir = out_dir / "runs"
    results_dir = out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(root / args.train_path)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    tiny = rows[: args.n]
    if len(tiny) < args.n:
        raise ValueError(f"Not enough rows in {args.train_path}, expected >= {args.n}, got {len(tiny)}")

    # Overfit setup: train/val/test all point to the same 10 samples.
    train_jsonl = data_dir / "kgqa_train_10.jsonl"
    val_jsonl = data_dir / "kgqa_val_10.jsonl"
    test_jsonl = data_dir / "kgqa_test_10.jsonl"
    _dump_jsonl(train_jsonl, tiny)
    _dump_jsonl(val_jsonl, tiny)
    _dump_jsonl(test_jsonl, tiny)
    print(f"[data] prepared tiny overfit split: n={len(tiny)}")

    ae_cfg = {
        "train_path": str(train_jsonl.relative_to(root)).replace("\\", "/"),
        "seed": args.seed,
        "min_freq": 1,
        "max_path_len": 8,
        "batch_size": 10,
        "emb_dim": 128,
        "hid_dim": 192,
        "latent_dim": 128,
        "lr": 0.001,
        "epochs": args.epochs,
        "length_loss_weight": 0.25,
        "latent_noise_std": float(args.ae_latent_noise_std),
        "use_cuda": bool(args.use_cuda),
    }
    planner_cfg = {
        "train_path": str(train_jsonl.relative_to(root)).replace("\\", "/"),
        "seed": args.seed,
        "max_path_len": 8,
        "max_query_len": 24,
        "batch_size": 10,
        "q_emb_dim": 128,
        "time_dim": 32,
        "diffusion_steps": 20,
        "lr": 0.001,
        "epochs": args.epochs,
        "use_cuda": bool(args.use_cuda),
        "prediction_target": str(args.prediction_target).lower(),
        "hidden_dim": 256,
        "planner_type": str(args.planner_type).lower(),
    }
    value_cfg = {
        "train_path": str(train_jsonl.relative_to(root)).replace("\\", "/"),
        "seed": args.seed,
        "max_path_len": 8,
        "max_query_len": 24,
        "neg_per_pos": 2,
        "batch_size": 20,
        "emb_dim": 128,
        "lr": 0.001,
        "epochs": args.epochs,
        "use_cuda": bool(args.use_cuda),
    }
    eval_cfg = {
        "test_path": str(test_jsonl.relative_to(root)).replace("\\", "/"),
        "train_path": str(train_jsonl.relative_to(root)).replace("\\", "/"),
        "seed": args.seed,
        "num_candidates": int(args.num_candidates),
        "receding_horizon": bool(args.receding_horizon),
        "use_value_model": bool(args.use_value_model),
        "use_cuda": bool(args.use_cuda),
        "use_memory_retrieval": bool(args.use_memory_retrieval),
        "memory_top_k": int(args.memory_top_k),
        "memory_max_postings_per_token": int(args.memory_max_postings_per_token),
    }

    ae_cfg_p = cfg_dir / "autoencoder_10.json"
    planner_cfg_p = cfg_dir / "planner_10.json"
    value_cfg_p = cfg_dir / "value_10.json"
    eval_cfg_p = cfg_dir / "eval_10.json"
    _write_json(ae_cfg_p, ae_cfg)
    _write_json(planner_cfg_p, planner_cfg)
    _write_json(value_cfg_p, value_cfg)
    _write_json(eval_cfg_p, eval_cfg)

    ae_out = runs_dir / "ae"
    planner_out = runs_dir / "planner"
    value_out = runs_dir / "value"
    eval_out = results_dir / "eval"

    _run([sys.executable, "train_autoencoder_torch.py", "--config", str(ae_cfg_p), "--out", str(ae_out)], root)
    planner_entry = "train_diffusion_planner_torch.py" if args.planner_type == "diffusion" else "train_mlp_planner.py"
    _run(
        [
            sys.executable,
            planner_entry,
            "--config",
            str(planner_cfg_p),
            "--ae_ckpt",
            str(ae_out / "best.pt"),
            "--out",
            str(planner_out),
        ],
        root,
    )
    _run(
        [
            sys.executable,
            "train_value_model_torch.py",
            "--config",
            str(value_cfg_p),
            "--planner_ckpt",
            str(planner_out / "best.pt"),
            "--out",
            str(value_out),
        ],
        root,
    )
    _run(
        [
            sys.executable,
            "evaluate_torch.py",
            "--config",
            str(eval_cfg_p),
            "--ae_ckpt",
            str(ae_out / "best.pt"),
            "--planner_ckpt",
            str(planner_out / "best.pt"),
            "--value_ckpt",
            str(value_out / "best.pt"),
            "--out",
            str(eval_out),
        ],
        root,
    )

    summary_path = eval_out / "summary_metrics.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    method = list(summary.keys())[0]
    print("[sanity] summary:", summary[method])
    print(f"[sanity] done. outputs in: {out_dir}")


if __name__ == "__main__":
    main()
