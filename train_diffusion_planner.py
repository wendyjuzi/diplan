import argparse
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_json, load_jsonl
from src.diplan.modeling import train_diffusion_planner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_kgqa.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/diplan_kgqa")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["train_path"])
    ae_model = load_json(args.ckpt)
    planner = train_diffusion_planner(rows, ae_model)
    planner["sampling_steps"] = cfg.get("sampling_steps", 20)
    planner["guidance_scale"] = cfg.get("guidance_scale", 1.2)

    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "best.pt"), planner)
    dump_json(
        str(Path(args.out) / "train_metrics.json"),
        {"num_samples": len(rows), "sampling_steps": planner["sampling_steps"]},
    )
    print(f"Diffusion planner trained: {len(rows)} samples")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

