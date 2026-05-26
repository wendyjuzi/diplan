import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.diplan.io_utils import dump_json, ensure_dir, load_json


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--agenthospital_raw", type=str, default="")
    parser.add_argument("--clinicalbench_raw", type=str, default="")
    parser.add_argument("--processed_dir", type=str, default="data/clinical_processed")
    parser.add_argument("--ae_cfg_base", type=str, default="configs/autoencoder_torch_clinical.tune1.json")
    parser.add_argument("--planner_cfg_base", type=str, default="configs/diffusion_torch_clinical.mlp_tune1.json")
    parser.add_argument(
        "--value_cfg_base",
        type=str,
        default="configs/value_torch_clinical.cross_infonce_hardneg.tune1.json",
    )
    parser.add_argument(
        "--eval_cfg_base",
        type=str,
        default="configs/eval_torch_clinical_agenthospital.json",
    )
    parser.add_argument("--out_root", type=str, default="results/clinical_agenthospital_seed42")
    parser.add_argument("--runs_root", type=str, default="runs/clinical_agenthospital_seed42")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_root))

    py_exec = sys.executable

    processed_dir = (repo / args.processed_dir).resolve()
    train_path = processed_dir / "clinical_train.jsonl"
    test_path = processed_dir / "clinical_test.jsonl"

    # Optional preparation from raw files.
    if args.agenthospital_raw or args.clinicalbench_raw:
        cmd = [
            py_exec,
            "scripts/prepare_agenthospital_data.py",
            "--out",
            str(processed_dir),
            "--seed",
            str(int(args.seed)),
        ]
        if args.agenthospital_raw:
            cmd.extend(["--agenthospital", args.agenthospital_raw])
        if args.clinicalbench_raw:
            cmd.extend(["--clinicalbench", args.clinicalbench_raw])
        _run(cmd, cwd=repo)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed clinical data. Expected: {train_path} and {test_path}. "
            "Provide --agenthospital_raw/--clinicalbench_raw or prepare data first."
        )

    # 1) Train autoencoder.
    ae_cfg = load_json(str(repo / args.ae_cfg_base))
    ae_cfg.update({"seed": int(args.seed), "train_path": str(train_path), "use_cuda": bool(args.use_cuda)})
    ae_cfg_path = cfg_root / "ae.json"
    _write_json(ae_cfg_path, ae_cfg)
    ae_out = runs_root / "ae"
    _run([py_exec, "train_autoencoder_torch.py", "--config", str(ae_cfg_path), "--out", str(ae_out)], cwd=repo)
    ae_ckpt = ae_out / "best.pt"

    # 2) Train MLP planner.
    planner_cfg = load_json(str(repo / args.planner_cfg_base))
    planner_cfg.update({"seed": int(args.seed), "train_path": str(train_path), "use_cuda": bool(args.use_cuda)})
    planner_cfg_path = cfg_root / "planner.json"
    _write_json(planner_cfg_path, planner_cfg)
    planner_out = runs_root / "mlp_planner"
    _run(
        [
            py_exec,
            "train_mlp_planner.py",
            "--config",
            str(planner_cfg_path),
            "--ae_ckpt",
            str(ae_ckpt),
            "--out",
            str(planner_out),
        ],
        cwd=repo,
    )
    planner_ckpt = planner_out / "best.pt"

    # 3) Eval on train split without value model, to mine hard negatives.
    eval_base = load_json(str(repo / args.eval_cfg_base))
    eval_train_no_value = dict(eval_base)
    eval_train_no_value.update(
        {
            "seed": int(args.seed),
            "train_path": str(train_path),
            "test_path": str(train_path),
            "use_cuda": bool(args.use_cuda),
            "use_value_model": False,
            "use_memory_retrieval": True,
            "save_candidate_pool_topk": 12,
        }
    )
    eval_train_no_value_path = cfg_root / "eval_train_no_value.json"
    _write_json(eval_train_no_value_path, eval_train_no_value)
    train_no_value_out = out_root / "train_no_value_eval"
    _run(
        [
            py_exec,
            "evaluate_torch.py",
            "--config",
            str(eval_train_no_value_path),
            "--ae_ckpt",
            str(ae_ckpt),
            "--planner_ckpt",
            str(planner_ckpt),
            "--out",
            str(train_no_value_out),
        ],
        cwd=repo,
    )

    # 4) Train value model.
    value_cfg = load_json(str(repo / args.value_cfg_base))
    value_cfg.update(
        {
            "seed": int(args.seed),
            "train_path": str(train_path),
            "use_cuda": bool(args.use_cuda),
            "hard_negative_predictions_path": str(train_no_value_out / "predictions.jsonl"),
        }
    )
    value_cfg_path = cfg_root / "value.json"
    _write_json(value_cfg_path, value_cfg)
    value_out = runs_root / "value_cross_infonce"
    _run(
        [
            py_exec,
            "train_value_model_torch.py",
            "--config",
            str(value_cfg_path),
            "--planner_ckpt",
            str(planner_ckpt),
            "--out",
            str(value_out),
        ],
        cwd=repo,
    )
    value_ckpt = value_out / "best.pt"

    # 5) Evaluate on test split with value model.
    eval_test = dict(eval_base)
    eval_test.update(
        {
            "seed": int(args.seed),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "use_cuda": bool(args.use_cuda),
            "use_value_model": True,
            "use_memory_retrieval": True,
        }
    )
    eval_test_path = cfg_root / "eval_test_with_value.json"
    _write_json(eval_test_path, eval_test)
    test_eval_out = out_root / "test_with_value_eval"
    _run(
        [
            py_exec,
            "evaluate_torch.py",
            "--config",
            str(eval_test_path),
            "--ae_ckpt",
            str(ae_ckpt),
            "--planner_ckpt",
            str(planner_ckpt),
            "--value_ckpt",
            str(value_ckpt),
            "--out",
            str(test_eval_out),
        ],
        cwd=repo,
    )

    manifest = {
        "seed": int(args.seed),
        "processed_dir": str(processed_dir),
        "checkpoints": {
            "ae_ckpt": str(ae_ckpt),
            "planner_ckpt": str(planner_ckpt),
            "value_ckpt": str(value_ckpt),
        },
        "outputs": {
            "train_no_value_eval": str(train_no_value_out),
            "test_with_value_eval": str(test_eval_out),
            "test_summary_metrics": str(test_eval_out / "summary_metrics.json"),
            "test_summary_by_dataset": str(test_eval_out / "summary_by_dataset.json"),
        },
    }
    dump_json(str(out_root / "pipeline_manifest.json"), manifest)
    print(f"[ok] clinical pipeline completed. manifest={out_root / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    main()

