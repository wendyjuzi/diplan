import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def _parse_margins(text: str) -> List[float]:
    out = []
    for part in text.replace(",", " ").split():
        p = part.strip().lower()
        if not p:
            continue
        if p in {"inf", "infty", "infinite"}:
            out.append(1_000_000.0)
        else:
            out.append(float(p))
    return out


def _fmt_margin(margin: float) -> str:
    if margin >= 999_999:
        return "inf"
    return str(margin).replace(".", "p").replace("-", "neg")


def _parse_external_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --external_run spec: {spec}. Expected label=path.")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --external_run spec: {spec}. Expected label=path.")
    return label, Path(path)


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


def _run_eval(args: argparse.Namespace, label: str, cfg: Dict, out_dir: Path, cfg_dir: Path, log_dir: Path) -> None:
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
        str(out_dir),
    ]
    if args.value_ckpt:
        cmd.extend(["--value_ckpt", args.value_ckpt])
    _run(cmd, ROOT, log_dir / f"{label}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrieval-fusion margins for one fixed eval config.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="results/single_config_fusion_sweep")
    parser.add_argument("--label-prefix", type=str, default="fusion")
    parser.add_argument("--margins", type=str, default="0 1 2 4 8 16 inf")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ref-label", type=str, default="")
    parser.add_argument(
        "--external_run",
        action="append",
        default=[],
        help="Optional extra run spec label=path_to_run_dir included in the summary table.",
    )
    args = parser.parse_args()

    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if not os.environ.get(env_name, "").isdigit():
            os.environ[env_name] = "1"

    margins = _parse_margins(args.margins)
    if not margins:
        raise ValueError("No margins provided.")

    out_root = Path(args.out_root)
    runs_dir = out_root / "runs"
    cfg_dir = out_root / "generated_configs"
    log_dir = out_root / "logs"
    tables_dir = out_root / "tables"
    ensure_dir(str(out_root))

    base_cfg = load_config(args.config)
    completed = []
    for spec in args.external_run:
        label, run_dir = _parse_external_run(spec)
        completed.append(_load_run(label, run_dir))

    for margin in margins:
        label = f"{args.label_prefix}_m{_fmt_margin(margin)}"
        run_dir = runs_dir / label
        cfg = dict(base_cfg)
        cfg.update(
            {
                "retrieval_fusion_enabled": True,
                "retrieval_fusion_margin": float(margin),
                "use_memory_retrieval": True,
                "memory_prefilter_feasible": True,
            }
        )
        if args.skip_existing and (run_dir / "summary_metrics.json").exists() and (run_dir / "predictions.jsonl").exists():
            print(f"[skip] {label} already exists at {run_dir}")
        else:
            _run_eval(args, label, cfg, run_dir, cfg_dir, log_dir)
        completed.append(_load_run(label, run_dir))

    ref_label = args.ref_label.strip() or completed[0]["label"]
    _summarize(
        completed,
        out_dir=tables_dir,
        ref_label=ref_label,
        bootstrap_n=args.bootstrap,
        seed=int(base_cfg.get("seed", 42)),
    )
    dump_json(
        str(out_root / "single_config_fusion_sweep_manifest.json"),
        {
            "config": args.config,
            "margins": margins,
            "runs": [{"label": r["label"], "dir": r["dir"], "method": r["method"]} for r in completed],
            "tables": str(tables_dir),
        },
    )
    print(f"[ok] single-config fusion sweep finished: {out_root}")
    print(f"[ok] tables: {tables_dir}")


if __name__ == "__main__":
    main()
