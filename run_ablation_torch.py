import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation_torch_kgqa.yaml")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/ablation_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_eval = load_config(cfg["base_eval_config"])
    ensure_dir(args.out)
    tmp_dir = Path(args.out) / "_tmp"
    ensure_dir(str(tmp_dir))

    report = {}
    for exp in cfg["experiments"]:
        name = exp["name"]
        exp_cfg = dict(base_eval)
        exp_cfg.update(exp.get("overrides", {}))
        exp_cfg_path = tmp_dir / f"{name}.json"
        with open(exp_cfg_path, "w", encoding="utf-8") as f:
            json.dump(exp_cfg, f, ensure_ascii=False, indent=2)

        exp_out = Path(args.out) / name
        ensure_dir(str(exp_out))
        cmd = [
            sys.executable,
            "evaluate_torch.py",
            "--config",
            str(exp_cfg_path),
            "--ae_ckpt",
            args.ae_ckpt,
            "--planner_ckpt",
            args.planner_ckpt,
            "--value_ckpt",
            args.value_ckpt,
            "--out",
            str(exp_out),
        ]
        subprocess.run(cmd, check=True)

        summary_path = exp_out / "summary_metrics.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if not summary:
            raise ValueError(f"Empty summary_metrics.json for ablation experiment: {name}")
        method = "diplan_torch" if "diplan_torch" in summary else next(iter(summary.keys()))
        report[name] = dict(summary[method])
        report[name]["method"] = method
        met = report[name]
        print(f"{name:>28} | success={met['success_rate']:.3f} first_err={met['first_error_step']:.2f} trap@1={met['trap_at_1']:.3f}")

    dump_json(str(Path(args.out) / "ablation_summary.json"), report)
    print(f"Ablation report written to {args.out}")


if __name__ == "__main__":
    main()
