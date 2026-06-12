import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.diplan.io_utils import dump_json, ensure_dir, load_config
from run_kgqa_baselines import _load_run, _summarize


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run(cmd: List[str], cwd: Path, log_path: Path) -> None:
    ensure_dir(str(log_path.parent))
    print("[run]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _normalize_include_datasets(value: str) -> List[str]:
    return [x.strip().lower() for x in value.replace(",", " ").split() if x.strip()]


def _base_config(args: argparse.Namespace) -> Dict:
    cfg = load_config(args.base_config)
    if args.train_path:
        cfg["train_path"] = args.train_path
    if args.test_path:
        cfg["test_path"] = args.test_path
    if args.include_datasets:
        cfg["include_datasets"] = _normalize_include_datasets(args.include_datasets)
    if args.max_tasks > 0:
        cfg["max_tasks"] = int(args.max_tasks)
    cfg["seed"] = int(args.seed)
    cfg["use_cuda"] = bool(args.use_cuda)
    cfg["use_memory_retrieval"] = True
    cfg["memory_prefilter_feasible"] = True
    cfg["use_value_model"] = True
    cfg.setdefault("emit_structured_plan", True)
    cfg.setdefault("emit_tool_calls", True)
    cfg.setdefault("retrieval_fusion_enabled", False)
    return cfg


def _preset_specs(profile: str) -> List[Dict]:
    # Keep the sweep intentionally small: each row answers whether coverage gains
    # come from a larger pool, broader latent exploration, or both.
    compact = [
        {
            "label": "pool24_mem32_base",
            "num_candidates": 24,
            "memory_top_k": 32,
            "postings": 1800,
            "jitter": [0.0, 0.03, 0.06],
        },
        {
            "label": "pool32_mem48",
            "num_candidates": 32,
            "memory_top_k": 48,
            "postings": 2400,
            "jitter": [0.0, 0.03, 0.06],
        },
        {
            "label": "pool48_mem64",
            "num_candidates": 48,
            "memory_top_k": 64,
            "postings": 3200,
            "jitter": [0.0, 0.03, 0.06],
        },
        {
            "label": "pool64_mem96",
            "num_candidates": 64,
            "memory_top_k": 96,
            "postings": 4800,
            "jitter": [0.0, 0.03, 0.06],
        },
    ]
    full_extra = [
        {
            "label": "pool64_mem96_broad",
            "num_candidates": 64,
            "memory_top_k": 96,
            "postings": 4800,
            "jitter": [0.0, 0.02, 0.05, 0.08, 0.12],
        },
        {
            "label": "pool96_mem128",
            "num_candidates": 96,
            "memory_top_k": 128,
            "postings": 6400,
            "jitter": [0.0, 0.03, 0.06],
        },
        {
            "label": "pool96_mem160_broad",
            "num_candidates": 96,
            "memory_top_k": 160,
            "postings": 8000,
            "jitter": [0.0, 0.02, 0.05, 0.08, 0.12],
        },
        {
            "label": "pool128_mem192_broad",
            "num_candidates": 128,
            "memory_top_k": 192,
            "postings": 9600,
            "jitter": [0.0, 0.02, 0.05, 0.08, 0.12],
        },
    ]
    if profile == "compact":
        return compact
    if profile == "full":
        return compact + full_extra
    raise ValueError(f"Unknown profile: {profile}")


def _config_for_spec(base_cfg: Dict, spec: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(base_cfg)
    num_candidates = int(spec["num_candidates"])
    cfg.update(
        {
            "num_candidates": num_candidates,
            "memory_top_k": int(spec["memory_top_k"]),
            "memory_max_postings_per_token": int(spec["postings"]),
            "candidate_multi_jitter_stds": list(spec["jitter"]),
            "candidate_latent_jitter_std": float(max(spec["jitter"])) if spec["jitter"] else 0.0,
            "rerank_stage1_topk": int(min(num_candidates, max(12, num_candidates // 2))),
            "save_candidate_pool_topk": int(max(args.save_candidate_pool_topk, min(num_candidates, 32))),
            "receding_horizon": bool(args.receding_horizon),
            "value_guided_sampling": bool(args.value_guided_sampling),
            "retrieval_fusion_enabled": bool(args.retrieval_fusion),
        }
    )
    if args.retrieval_fusion:
        cfg["retrieval_fusion_margin"] = float(args.retrieval_fusion_margin)
    if args.value_guided_sampling:
        cfg.setdefault("value_guidance_scale", 0.18)
        cfg.setdefault("value_guidance_interval", 4)
        cfg.setdefault("value_guidance_temperature", 0.5)
    return cfg


def _run_eval(args: argparse.Namespace, label: str, cfg: Dict, run_dir: Path, cfg_dir: Path, log_dir: Path) -> None:
    cfg_path = cfg_dir / f"{label}.json"
    _write_json(cfg_path, cfg)
    cmd = [
        sys.executable,
        "evaluate_torch.py",
        "--config",
        str(cfg_path),
        "--ae_ckpt",
        args.ae_ckpt,
        "--planner_ckpt",
        args.planner_ckpt,
        "--out",
        str(run_dir),
    ]
    if args.value_ckpt:
        cmd.extend(["--value_ckpt", args.value_ckpt])
    _run(cmd, ROOT, log_dir / f"{label}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep high-coverage candidate-pool settings for DiPLaN KGQA.")
    parser.add_argument("--base-config", type=str, default="configs/eval_torch_kgqa.diffusion_value_guided.json")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="results/candidate_coverage_sweep_seed42")
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--include-datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--profile", choices=["compact", "full"], default="compact")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ref-label", type=str, default="pool24_mem32_base")
    parser.add_argument("--save-candidate-pool-topk", type=int, default=32)
    parser.add_argument("--receding-horizon", action="store_true")
    parser.add_argument("--value-guided-sampling", action="store_true")
    parser.add_argument("--retrieval-fusion", action="store_true")
    parser.add_argument("--retrieval-fusion-margin", type=float, default=16.0)
    args = parser.parse_args()

    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if not os.environ.get(env_name, "").isdigit():
            os.environ[env_name] = "1"

    out_root = Path(args.out_root)
    runs_dir = out_root / "runs"
    cfg_dir = out_root / "generated_configs"
    log_dir = out_root / "logs"
    tables_dir = out_root / "tables"
    ensure_dir(str(out_root))

    base_cfg = _base_config(args)
    specs = _preset_specs(args.profile)
    completed = []
    manifest = {
        "base_config": args.base_config,
        "ae_ckpt": args.ae_ckpt,
        "planner_ckpt": args.planner_ckpt,
        "value_ckpt": args.value_ckpt,
        "profile": args.profile,
        "runs": [],
    }

    for spec in specs:
        label = spec["label"]
        run_dir = runs_dir / label
        cfg = _config_for_spec(base_cfg, spec, args)
        manifest["runs"].append({"label": label, "run_dir": str(run_dir), "overrides": spec})
        if args.skip_existing and (run_dir / "summary_metrics.json").exists() and (run_dir / "predictions.jsonl").exists():
            print(f"[skip] {label} already exists at {run_dir}")
        else:
            _run_eval(args, label, cfg, run_dir, cfg_dir, log_dir)
        completed.append(_load_run(label, run_dir))

    _summarize(
        completed,
        out_dir=tables_dir,
        ref_label=args.ref_label,
        bootstrap_n=args.bootstrap,
        seed=args.seed,
    )
    dump_json(str(out_root / "candidate_coverage_sweep_manifest.json"), manifest)
    print(f"[ok] candidate coverage sweep finished: {out_root}")
    print(f"[ok] tables: {tables_dir}")


if __name__ == "__main__":
    main()
