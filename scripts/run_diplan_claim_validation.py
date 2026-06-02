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


def _write_json(path: Path, obj: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _parse_runs(items: List[str]) -> List[str]:
    return [x for x in items if x and "=" in x]


def _eval_config(args: argparse.Namespace, base_cfg: Dict, overrides: Dict) -> Dict:
    cfg = dict(base_cfg)
    cfg.update(
        {
            "train_path": args.train_path,
            "test_path": args.test_path,
            "seed": int(args.seed),
            "use_cuda": bool(args.use_cuda),
            "include_datasets": [x.strip().lower() for x in args.include_datasets.replace(",", " ").split() if x.strip()],
            "emit_structured_plan": True,
            "emit_tool_calls": True,
            "save_candidate_pool_topk": int(args.save_candidate_pool_topk),
        }
    )
    if args.max_tasks > 0:
        cfg["max_tasks"] = int(args.max_tasks)
    cfg.update(overrides)
    return cfg


def _run_eval(
    args: argparse.Namespace,
    label: str,
    cfg: Dict,
    planner_ckpt: str,
    value_ckpt: str,
    out_dir: Path,
    cfg_dir: Path,
    log_dir: Path,
) -> None:
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
        planner_ckpt,
        "--out",
        str(out_dir),
    ]
    if value_ckpt:
        cmd.extend(["--value_ckpt", value_ckpt])
    _run(cmd, ROOT, log_dir / f"{label}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiPLaN claim-validation experiments.")
    parser.add_argument("--train_path", type=str, default="data/real_processed/kgqa_train.jsonl")
    parser.add_argument("--test_path", type=str, default="data/real_processed/kgqa_test.jsonl")
    parser.add_argument("--ae_ckpt", type=str, default="runs/ae_kgqa_torch_real_tune3_noise003/best.pt")
    parser.add_argument("--mlp_planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, default="")
    parser.add_argument("--diffusion_config", type=str, default="configs/diffusion_torch_kgqa.tune3.json")
    parser.add_argument("--out_root", type=str, default="results/diplan_claim_validation_seed42")
    parser.add_argument("--runs_root", type=str, default="runs/diplan_claim_validation_seed42")
    parser.add_argument("--base_eval_config", type=str, default="configs/eval_torch_kgqa.diffusion_value_guided.json")
    parser.add_argument("--include_datasets", type=str, default="cwq webqsp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--skip_train_diffusion", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_candidate_pool_topk", type=int, default=12)
    parser.add_argument(
        "--external_run",
        action="append",
        default=[],
        help="Additional summary run spec label=path_to_run_dir for final comparison table.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    runs_root = Path(args.runs_root)
    cfg_dir = out_root / "generated_configs"
    log_dir = out_root / "logs"
    ensure_dir(str(out_root))
    ensure_dir(str(runs_root))
    ensure_dir(str(cfg_dir))

    diffusion_ckpt = args.diffusion_ckpt
    if not diffusion_ckpt:
        diffusion_ckpt = str(runs_root / "diffusion_planner" / "best.pt")

    if not args.skip_train_diffusion and not Path(diffusion_ckpt).exists():
        diff_cfg = load_config(args.diffusion_config)
        diff_cfg.update(
            {
                "train_path": args.train_path,
                "seed": int(args.seed),
                "use_cuda": bool(args.use_cuda),
            }
        )
        diff_cfg_path = cfg_dir / "diffusion_train.json"
        _write_json(diff_cfg_path, diff_cfg)
        _run(
            [
                sys.executable,
                "train_diffusion_planner_torch.py",
                "--config",
                str(diff_cfg_path),
                "--ae_ckpt",
                args.ae_ckpt,
                "--out",
                str(runs_root / "diffusion_planner"),
            ],
            ROOT,
            log_dir / "train_diffusion_planner.log",
        )
    if not Path(diffusion_ckpt).exists():
        raise FileNotFoundError(f"Missing diffusion checkpoint: {diffusion_ckpt}")

    base_cfg = load_config(args.base_eval_config)
    experiments = [
        {
            "label": "mlp_listwise_full_plan",
            "planner": args.mlp_planner_ckpt,
            "overrides": {
                "receding_horizon": False,
                "value_guided_sampling": False,
                "retrieval_fusion_enabled": False,
            },
        },
        {
            "label": "mlp_listwise_receding",
            "planner": args.mlp_planner_ckpt,
            "overrides": {
                "receding_horizon": True,
                "value_guided_sampling": False,
                "retrieval_fusion_enabled": False,
            },
        },
        {
            "label": "diffusion_no_guidance_full_plan",
            "planner": diffusion_ckpt,
            "overrides": {
                "receding_horizon": False,
                "value_guided_sampling": False,
                "retrieval_fusion_enabled": False,
            },
        },
        {
            "label": "diffusion_value_guided_full_plan",
            "planner": diffusion_ckpt,
            "overrides": {
                "receding_horizon": False,
                "value_guided_sampling": True,
                "value_guidance_scale": 0.18,
                "value_guidance_interval": 4,
                "value_guidance_temperature": 0.5,
                "retrieval_fusion_enabled": False,
            },
        },
        {
            "label": "diffusion_value_guided_receding",
            "planner": diffusion_ckpt,
            "overrides": {
                "receding_horizon": True,
                "value_guided_sampling": True,
                "value_guidance_scale": 0.18,
                "value_guidance_interval": 4,
                "value_guidance_temperature": 0.5,
                "retrieval_fusion_enabled": False,
            },
        },
    ]

    run_specs: List[str] = []
    for exp in experiments:
        label = exp["label"]
        run_dir = out_root / "runs" / label
        if args.skip_existing and (run_dir / "summary_metrics.json").exists():
            print(f"[skip] {label} exists at {run_dir}")
        else:
            cfg = _eval_config(args, base_cfg, exp["overrides"])
            _run_eval(
                args=args,
                label=label,
                cfg=cfg,
                planner_ckpt=str(exp["planner"]),
                value_ckpt=args.value_ckpt,
                out_dir=run_dir,
                cfg_dir=cfg_dir,
                log_dir=log_dir,
            )
        run_specs.append(f"{label}={run_dir}")

    run_specs.extend(_parse_runs(args.external_run))
    if len(run_specs) >= 2:
        cmd = [
            sys.executable,
            "scripts/summarize_experiment_runs.py",
            "--out_dir",
            str(out_root / "summary"),
            "--prefix",
            "diplan_claim_validation",
            "--ref_label",
            "mlp_listwise_full_plan",
        ]
        for spec in run_specs:
            cmd.extend(["--run", spec])
        _run(cmd, ROOT, log_dir / "summarize_claim_validation.log")

    dump_json(
        str(out_root / "claim_validation_manifest.json"),
        {
            "diffusion_ckpt": diffusion_ckpt,
            "mlp_planner_ckpt": args.mlp_planner_ckpt,
            "value_ckpt": args.value_ckpt,
            "experiments": experiments,
            "external_runs": args.external_run,
        },
    )
    print(f"[ok] claim validation outputs written to {out_root}")


if __name__ == "__main__":
    main()
