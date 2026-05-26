import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_json


def _run(cmd, cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_json(path: Path, obj) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--processed_dir", type=str, default="data/clinical_ai_hospital_processed")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--eval_cfg_base", type=str, default="configs/eval_torch_clinical_ai_hospital.json")
    parser.add_argument("--out_root", type=str, default="results/clinical_ai_hospital_closed_loop_seed42")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(cfg_root))

    eval_base = load_json(str(repo / args.eval_cfg_base))
    eval_cfg = dict(eval_base)
    eval_cfg.update(
        {
            "seed": int(args.seed),
            "train_path": str((repo / args.processed_dir / "clinical_train.jsonl").resolve()),
            "test_path": str((repo / args.processed_dir / "clinical_test.jsonl").resolve()),
            "use_cuda": bool(args.use_cuda),
            "use_value_model": True,
            "use_memory_retrieval": True,
            "receding_horizon": True,
            "save_episode_trace": True,
            "save_candidate_pool_topk": 12,
        }
    )
    eval_cfg_path = cfg_root / "eval_closed_loop.json"
    _write_json(eval_cfg_path, eval_cfg)

    eval_out = out_root / "closed_loop_eval"
    _run(
        [
            sys.executable,
            "evaluate_torch.py",
            "--config",
            str(eval_cfg_path),
            "--ae_ckpt",
            str(args.ae_ckpt),
            "--planner_ckpt",
            str(args.planner_ckpt),
            "--value_ckpt",
            str(args.value_ckpt),
            "--out",
            str(eval_out),
        ],
        cwd=repo,
    )

    manifest = {
        "seed": int(args.seed),
        "eval_config": str(eval_cfg_path),
        "outputs": {
            "predictions": str(eval_out / "predictions.jsonl"),
            "summary_metrics": str(eval_out / "summary_metrics.json"),
            "summary_by_dataset": str(eval_out / "summary_by_dataset.json"),
            "summary_table": str(eval_out / "summary_table.csv"),
        },
    }
    dump_json(str(out_root / "closed_loop_manifest.json"), manifest)
    print(f"[ok] closed-loop benchmark done: {out_root / 'closed_loop_manifest.json'}")


if __name__ == "__main__":
    main()
