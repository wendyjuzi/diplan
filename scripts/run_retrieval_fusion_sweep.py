import argparse
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
from run_kgqa_baselines import _load_run, _run, _summarize, _write_json


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


def _base_config(args: argparse.Namespace) -> Dict:
    cfg = load_config(args.base_config)
    if args.train_path:
        cfg["train_path"] = args.train_path
    if args.test_path:
        cfg["test_path"] = args.test_path
    if args.include_datasets:
        cfg["include_datasets"] = [
            x.strip().lower() for x in args.include_datasets.replace(",", " ").split() if x.strip()
        ]
    if args.max_tasks > 0:
        cfg["max_tasks"] = int(args.max_tasks)
    cfg["seed"] = int(args.seed)
    cfg["use_cuda"] = bool(args.use_cuda)
    cfg.setdefault("use_memory_retrieval", True)
    cfg.setdefault("memory_prefilter_feasible", True)
    cfg.setdefault("save_candidate_pool_topk", 12)
    return cfg


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


def _run_retrieval_feasible(args: argparse.Namespace, cfg: Dict, out_dir: Path, cfg_dir: Path, log_dir: Path) -> None:
    cfg_path = cfg_dir / "retrieval_feasible.json"
    retrieval_cfg = {
        "train_path": cfg["train_path"],
        "test_path": cfg["test_path"],
        "seed": int(cfg.get("seed", 42)),
        "top_k": int(cfg.get("memory_top_k", 32)),
        "max_postings_per_token": int(cfg.get("memory_max_postings_per_token", 1200)),
        "include_datasets": cfg.get("include_datasets", []),
        "max_tasks": int(cfg.get("max_tasks", 0)),
        "filter_feasible": True,
    }
    _write_json(cfg_path, retrieval_cfg)
    cmd = [
        sys.executable,
        "evaluate_retrieval_baseline.py",
        "--config",
        str(cfg_path),
        "--out",
        str(out_dir),
    ]
    _run(cmd, ROOT, log_dir / "retrieval_feasible.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep retrieval-fusion margins for KGQA ranking diagnostics.")
    parser.add_argument("--base-config", type=str, default="configs/eval_torch_kgqa.tune4.high_recall_multistage.cwq_webqsp.prefixpen.json")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="results/retrieval_fusion_sweep")
    parser.add_argument("--margins", type=str, default="0 1 2 4 6 8 16 inf")
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--include-datasets", type=str, default="")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    margins = _parse_margins(args.margins)
    if not margins:
        raise ValueError("No margins provided.")

    out_root = Path(args.out_root)
    runs_dir = out_root / "runs"
    cfg_dir = out_root / "generated_configs"
    log_dir = out_root / "logs"
    tables_dir = out_root / "tables"
    ensure_dir(str(out_root))

    base_cfg = _base_config(args)
    completed = []

    retrieval_dir = runs_dir / "retrieval_feasible"
    if args.skip_existing and (retrieval_dir / "summary_metrics.json").exists():
        print(f"[skip] retrieval_feasible already exists at {retrieval_dir}")
    else:
        _run_retrieval_feasible(args, base_cfg, retrieval_dir, cfg_dir, log_dir)
    completed.append(_load_run("retrieval_feasible", retrieval_dir))

    for margin in margins:
        label = f"fusion_m{_fmt_margin(margin)}"
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
        if args.skip_existing and (run_dir / "summary_metrics.json").exists():
            print(f"[skip] {label} already exists at {run_dir}")
        else:
            _run_eval(args, label, cfg, run_dir, cfg_dir, log_dir)
        completed.append(_load_run(label, run_dir))

    _summarize(
        completed,
        out_dir=tables_dir,
        ref_label="retrieval_feasible",
        bootstrap_n=args.bootstrap,
        seed=args.seed,
    )
    dump_json(
        str(out_root / "retrieval_fusion_sweep_manifest.json"),
        {
            "margins": margins,
            "runs": [{"label": r["label"], "dir": r["dir"], "method": r["method"]} for r in completed],
            "tables": str(tables_dir),
        },
    )
    print(f"[ok] retrieval-fusion sweep finished: {out_root}")
    print(f"[ok] tables: {tables_dir}")


if __name__ == "__main__":
    main()
