import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--out_dir", type=str, default="experiments/ae_overfit_isolation")
    args = parser.parse_args()

    root = Path(".").resolve()
    out_dir = root / args.out_dir
    data_dir = out_dir / "data"
    cfg_dir = out_dir / "configs"
    run_dir = out_dir / "runs" / "ae"
    diag_dir = out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(root / args.train_path)
    rows = [r for r in rows if isinstance(r.get("oracle_path"), list) and r["oracle_path"]]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    tiny = rows[: args.n]
    if len(tiny) < args.n:
        raise ValueError(f"Not enough rows in {args.train_path}, expected >= {args.n}, got {len(tiny)}")

    tiny_path = data_dir / "kgqa_train_10.jsonl"
    _dump_jsonl(tiny_path, tiny)

    ae_cfg = {
        "train_path": str(tiny_path.relative_to(root)).replace("\\", "/"),
        "seed": args.seed,
        "min_freq": 1,
        "max_path_len": 8,
        "batch_size": max(2, args.n),
        "emb_dim": 128,
        "hid_dim": 192,
        "latent_dim": 128,
        "lr": 0.001,
        "epochs": args.epochs,
        "length_loss_weight": 0.25,
        "use_cuda": bool(args.use_cuda),
    }
    cfg_path = cfg_dir / "autoencoder_overfit.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(ae_cfg, f, ensure_ascii=False, indent=2)

    _run(
        [
            sys.executable,
            "train_autoencoder_torch.py",
            "--config",
            str(cfg_path),
            "--out",
            str(run_dir),
        ],
        root,
    )
    _run(
        [
            sys.executable,
            "scripts/ae_reconstruction_isolation_test.py",
            "--ae_ckpt",
            str(run_dir / "best.pt"),
            "--data",
            str(tiny_path),
            "--n",
            str(args.n),
            "--seed",
            str(args.seed),
            "--show_cases",
            str(args.n),
            "--out",
            str(diag_dir / "ae_reconstruction_cases.jsonl"),
        ],
        root,
    )
    print(f"[done] outputs in: {out_dir}")


if __name__ == "__main__":
    main()

