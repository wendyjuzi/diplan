import argparse
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_jsonl
from src.diplan.modeling import train_value_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/value_kgqa.yaml")
    parser.add_argument("--out", type=str, default="runs/value_kgqa")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["train_path"])
    model = train_value_model(rows)

    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "best.pt"), model)
    dump_json(
        str(Path(args.out) / "train_metrics.json"),
        {"num_samples": len(rows), "query_vocab": len(model["assoc_prob"])},
    )
    print(f"Value model trained: {len(rows)} samples")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

