import argparse
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_jsonl
from src.diplan.modeling import train_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder_kgqa.yaml")
    parser.add_argument("--out", type=str, default="runs/ae_kgqa")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_path = cfg["train_path"]
    rows = load_jsonl(train_path)
    model = train_autoencoder(rows)

    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "best.pt"), model)
    dump_json(
        str(Path(args.out) / "train_metrics.json"),
        {"num_samples": len(rows), "vocab_size": model["vocab_size"]},
    )
    print(f"Autoencoder trained: {len(rows)} samples, vocab={model['vocab_size']}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

