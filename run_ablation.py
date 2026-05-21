import argparse
from pathlib import Path

from src.diplan.evaluate_core import evaluate_suite
from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_json, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation_kgqa.yaml")
    parser.add_argument("--planner_ckpt", type=str, default="runs/diplan_kgqa/best.pt")
    parser.add_argument("--value_ckpt", type=str, default="runs/value_kgqa/best.pt")
    parser.add_argument("--constraint_ckpt", type=str, default="runs/constraint_kgqa/best.pt")
    parser.add_argument("--out", type=str, default="results/ablation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_rows = load_jsonl(cfg["test_path"])
    planner_model = load_json(args.planner_ckpt)
    value_model = load_json(args.value_ckpt)
    constraint_model = load_json(args.constraint_ckpt)

    ensure_dir(args.out)
    report = {}
    for exp in cfg["experiments"]:
        name = exp["name"]
        options = exp["options"]
        result = evaluate_suite(
            test_rows=test_rows,
            planner_model=planner_model,
            value_model=value_model,
            constraint_model=constraint_model,
            methods=["diplan"],
            seed=int(cfg.get("seed", 42)),
            options=options,
        )
        summary = result["summary"]["diplan"]
        report[name] = summary
        print(f"{name:>24} | success={summary['success_rate']:.3f} first_err={summary['first_error_step']:.2f} trap@1={summary['trap_at_1']:.3f}")

    dump_json(str(Path(args.out) / "ablation_summary.json"), report)
    print(f"Ablation report written to {args.out}")


if __name__ == "__main__":
    main()

