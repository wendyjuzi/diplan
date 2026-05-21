import argparse
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_jsonl
from src.diplan.modeling import train_constraint_checker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/constraint_kgqa.yaml")
    parser.add_argument("--out", type=str, default="runs/constraint_kgqa")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["train_path"])
    model = train_constraint_checker(rows)

    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "best.pt"), model)
    dump_json(
        str(Path(args.out) / "train_metrics.json"),
        {"num_samples": len(rows), "allowed_relations": len(model["allowed_relations"])},
    )
    print(f"Constraint checker trained: {len(rows)} samples")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

