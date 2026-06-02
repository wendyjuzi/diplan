import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_config


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
    parser.add_argument("--train_path", type=str, default="data/real_processed/kgqa_train.jsonl")
    parser.add_argument("--test_path", type=str, default="data/real_processed/kgqa_test.jsonl")
    parser.add_argument("--ae_ckpt", type=str, default="runs/ae_kgqa_torch_real_tune3_noise003/best.pt")
    parser.add_argument("--ae_config", type=str, default="configs/autoencoder_torch_kgqa.tune3.noise003.json")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_torch_kgqa.tune3.eps.json")
    parser.add_argument("--value_ckpt", type=str, default="")
    parser.add_argument("--value_config", type=str, default="configs/value_torch_kgqa.yaml")
    parser.add_argument("--eval_config", type=str, default="configs/eval_torch_kgqa.diffusion_value_guided.json")
    parser.add_argument("--out_root", type=str, default="results/diplan_core_diffusion_seed42")
    parser.add_argument("--runs_root", type=str, default="runs/diplan_core_diffusion_seed42")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    cfg_root = out_root / "generated_configs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_root))

    py_exec = sys.executable
    ae_ckpt = (repo / args.ae_ckpt).resolve()

    if not ae_ckpt.exists():
        ae_cfg = load_config(str(repo / args.ae_config))
        ae_cfg.update({"seed": int(args.seed), "train_path": args.train_path, "use_cuda": bool(args.use_cuda)})
        ae_cfg_path = cfg_root / "autoencoder.json"
        _write_json(ae_cfg_path, ae_cfg)
        ae_out = runs_root / "autoencoder"
        _run([py_exec, "train_autoencoder_torch.py", "--config", str(ae_cfg_path), "--out", str(ae_out)], repo)
        ae_ckpt = ae_out / "best.pt"

    diffusion_cfg = load_config(str(repo / args.diffusion_config))
    diffusion_cfg.update({"seed": int(args.seed), "train_path": args.train_path, "use_cuda": bool(args.use_cuda)})
    diffusion_cfg_path = cfg_root / "diffusion_planner.json"
    _write_json(diffusion_cfg_path, diffusion_cfg)
    diffusion_out = runs_root / "diffusion_planner"
    _run(
        [
            py_exec,
            "train_diffusion_planner_torch.py",
            "--config",
            str(diffusion_cfg_path),
            "--ae_ckpt",
            str(ae_ckpt),
            "--out",
            str(diffusion_out),
        ],
        repo,
    )
    diffusion_ckpt = diffusion_out / "best.pt"

    if args.value_ckpt:
        value_ckpt = (repo / args.value_ckpt).resolve()
    else:
        value_cfg = load_config(str(repo / args.value_config))
        value_cfg.update({"seed": int(args.seed), "train_path": args.train_path, "use_cuda": bool(args.use_cuda)})
        value_cfg_path = cfg_root / "value_ranker.json"
        _write_json(value_cfg_path, value_cfg)
        value_out = runs_root / "value_ranker"
        _run(
            [
                py_exec,
                "train_value_model_torch.py",
                "--config",
                str(value_cfg_path),
                "--planner_ckpt",
                str(diffusion_ckpt),
                "--out",
                str(value_out),
            ],
            repo,
        )
        value_ckpt = value_out / "best.pt"

    eval_cfg = load_config(str(repo / args.eval_config))
    eval_cfg.update(
        {
            "seed": int(args.seed),
            "train_path": args.train_path,
            "test_path": args.test_path,
            "use_cuda": bool(args.use_cuda),
            "use_value_model": True,
            "value_guided_sampling": True,
            "emit_structured_plan": True,
            "emit_tool_calls": True,
        }
    )
    eval_cfg_path = cfg_root / "eval_diffusion_value_guided.json"
    _write_json(eval_cfg_path, eval_cfg)
    eval_out = out_root / "diffusion_value_guided_eval"
    _run(
        [
            py_exec,
            "evaluate_torch.py",
            "--config",
            str(eval_cfg_path),
            "--ae_ckpt",
            str(ae_ckpt),
            "--planner_ckpt",
            str(diffusion_ckpt),
            "--value_ckpt",
            str(value_ckpt),
            "--out",
            str(eval_out),
        ],
        repo,
    )

    diagnostics_out = out_root / "diagnostics"
    _run(
        [
            py_exec,
            "scripts/analyze_diagnostics.py",
            "--predictions",
            str(eval_out / "predictions.jsonl"),
            "--out",
            str(diagnostics_out),
        ],
        repo,
    )

    manifest = {
        "seed": int(args.seed),
        "checkpoints": {
            "ae_ckpt": str(ae_ckpt),
            "diffusion_planner_ckpt": str(diffusion_ckpt),
            "value_ckpt": str(value_ckpt),
        },
        "outputs": {
            "eval": str(eval_out),
            "summary_metrics": str(eval_out / "summary_metrics.json"),
            "predictions": str(eval_out / "predictions.jsonl"),
            "diagnostics": str(diagnostics_out),
        },
    }
    dump_json(str(out_root / "diplan_core_manifest.json"), manifest)
    print(f"[ok] DiPLaN-core pipeline complete: {out_root / 'diplan_core_manifest.json'}")


if __name__ == "__main__":
    main()
