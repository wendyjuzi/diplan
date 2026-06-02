import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_json, load_jsonl
from src.diplan.metrics import aggregate_method_metrics
from src.diplan.stats_utils import bootstrap_mean_diff, mcnemar_test_paired


METRIC_KEYS = [
    "n",
    "success_rate",
    "first_error_step",
    "recovery_at_error",
    "trap_at_1",
    "plan_feasibility",
    "constraint_violation_rate",
    "plan_execution_consistency",
    "token_cost",
    "latency_cost",
    "diversity_coverage",
    "candidate_pool_hit_rate",
    "ranking_error_rate",
    "candidate_pool_avg_size",
    "conditional_success_given_pool_hit",
    "llm_error_rate",
    "llm_fallback_rate",
]


def _write_json(path: Path, data: Dict) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _run(cmd: List[str], cwd: Path, log_path: Path) -> None:
    ensure_dir(str(log_path.parent))
    print(f"[run] {' '.join(cmd)}")
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
            try:
                print(line, end="")
            except UnicodeEncodeError:
                sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
                sys.stdout.flush()
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _normalize_include_datasets(value: str) -> List[str]:
    if not value.strip():
        return []
    return [x.strip().lower() for x in value.replace(",", " ").split() if x.strip()]


def _base_eval_config(args: argparse.Namespace) -> Dict:
    cfg = load_config(args.base_config)
    if args.train_path:
        cfg["train_path"] = args.train_path
    if args.test_path:
        cfg["test_path"] = args.test_path
    if args.include_datasets:
        cfg["include_datasets"] = _normalize_include_datasets(args.include_datasets)
    if args.max_tasks > 0:
        cfg["max_tasks"] = args.max_tasks
    cfg["seed"] = int(args.seed)
    cfg["use_cuda"] = bool(args.use_cuda)
    cfg.setdefault("save_candidate_pool_topk", 12)
    return cfg


def _baseline_specs(args: argparse.Namespace, base_cfg: Dict) -> List[Dict]:
    # These baselines are deliberately cheap and reproducible on the KGQA setup.
    # LLM-based baselines are optional because they require a running OpenAI-compatible server.
    specs = [
        {
            "label": "diplan_full",
            "runner": "evaluate_torch",
            "overrides": {},
            "description": "Full DiPLaN setting from the base config.",
        },
        {
            "label": "no_memory",
            "runner": "evaluate_torch",
            "overrides": {
                "use_memory_retrieval": False,
                "rerank_memory_bonus": 0.0,
                "rerank_memory_rank_bonus": 0.0,
            },
            "description": "Ablation without retrieval memory.",
        },
        {
            "label": "no_value",
            "runner": "evaluate_torch",
            "overrides": {
                "use_value_model": False,
                "value_guided_sampling": False,
            },
            "description": "Generate/select without the learned value ranker.",
        },
        {
            "label": "no_rerank",
            "runner": "evaluate_torch",
            "overrides": {
                "rerank_stage1_topk": 0,
                "rerank_consensus_weight": 0.0,
                "rerank_prefix_consensus_weight": 0.0,
                "rerank_memory_bonus": 0.0,
                "rerank_memory_rank_bonus": 0.0,
                "rerank_stage2_length_penalty_alpha": 0.0,
            },
            "description": "Stage-1 value ranking only; no second-stage memory/consensus reranker.",
        },
        {
            "label": "no_value_guidance",
            "runner": "evaluate_torch",
            "overrides": {
                "value_guided_sampling": False,
                "value_guidance_scale": 0.0,
            },
            "description": "Diffusion sampling without internal value-guided nudging.",
        },
        {
            "label": "receding_horizon",
            "runner": "evaluate_torch",
            "overrides": {
                "receding_horizon": True,
            },
            "description": "Limited-commitment execution; replans after each selected action.",
        },
        {
            "label": "full_plan_once",
            "runner": "evaluate_torch",
            "overrides": {
                "receding_horizon": False,
            },
            "description": "One-shot full-plan execution.",
        },
        {
            "label": "retrieval_only",
            "runner": "retrieval_baseline",
            "overrides": {},
            "description": "Tool/search baseline: token retrieval over train-set paths.",
        },
    ]
    if args.run_llm_agent:
        specs.append(
            {
                "label": "llm_agent",
                "runner": "llm_agent",
                "overrides": {
                    "llm_api_base": args.llm_api_base,
                    "llm_api_key": args.llm_api_key,
                    "llm_model": args.llm_model,
                    "llm_temperature": args.llm_temperature,
                    "llm_max_tokens": args.llm_max_tokens,
                    "llm_timeout_s": args.llm_timeout_s,
                    "llm_retries": args.llm_retries,
                    "llm_top_k": args.llm_top_k,
                    "save_episode_trace": True,
                },
                "description": "LLM-in-the-loop candidate selector over DiPLaN candidate pools.",
            }
        )

    requested = {x.strip() for x in args.baselines.split(",") if x.strip()}
    if requested:
        specs = [spec for spec in specs if spec["label"] in requested]
        missing = sorted(requested - {spec["label"] for spec in specs})
        if missing:
            raise ValueError(f"Unknown baseline label(s): {missing}")
    return specs


def _config_for_spec(base_cfg: Dict, spec: Dict) -> Dict:
    cfg = dict(base_cfg)
    cfg.update(spec.get("overrides", {}))
    return cfg


def _run_eval_baseline(args: argparse.Namespace, spec: Dict, cfg: Dict, out_dir: Path, cfg_dir: Path, log_dir: Path) -> None:
    cfg_path = cfg_dir / f"{spec['label']}.json"
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
    if cfg.get("use_value_model", True) and args.value_ckpt:
        cmd.extend(["--value_ckpt", args.value_ckpt])
    _run(cmd, ROOT, log_dir / f"{spec['label']}.log")


def _run_retrieval_baseline(spec: Dict, cfg: Dict, out_dir: Path, cfg_dir: Path, log_dir: Path) -> None:
    retrieval_cfg = {
        "train_path": cfg["train_path"],
        "test_path": cfg["test_path"],
        "seed": int(cfg.get("seed", 42)),
        "top_k": int(cfg.get("memory_top_k", 32)),
        "max_postings_per_token": int(cfg.get("memory_max_postings_per_token", 1200)),
        "include_datasets": cfg.get("include_datasets", []),
        "max_tasks": int(cfg.get("max_tasks", 0)),
    }
    cfg_path = cfg_dir / f"{spec['label']}.json"
    _write_json(cfg_path, retrieval_cfg)
    cmd = [
        sys.executable,
        "evaluate_retrieval_baseline.py",
        "--config",
        str(cfg_path),
        "--out",
        str(out_dir),
    ]
    _run(cmd, ROOT, log_dir / f"{spec['label']}.log")


def _run_llm_baseline(args: argparse.Namespace, spec: Dict, cfg: Dict, out_dir: Path, cfg_dir: Path, log_dir: Path) -> None:
    cfg_path = cfg_dir / f"{spec['label']}.json"
    _write_json(cfg_path, cfg)
    cmd = [
        sys.executable,
        "scripts/run_diplan_llm_agent.py",
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
    _run(cmd, ROOT, log_dir / f"{spec['label']}.log")


def _load_run(label: str, run_dir: Path) -> Dict:
    summary = load_json(str(run_dir / "summary_metrics.json"))
    preds = load_jsonl(str(run_dir / "predictions.jsonl"))
    method = next(iter(summary.keys())) if summary else label
    return {
        "label": label,
        "method": method,
        "dir": str(run_dir),
        "summary": summary.get(method, {}),
        "preds": preds,
    }


def _candidate_metrics(records: List[Dict]) -> Dict:
    if not records:
        return {}
    n = len(records)
    has_pool_flag = any("oracle_in_candidate_pool" in r for r in records)
    has_pool_size = any("candidate_pool_size" in r for r in records)
    has_ranking = any("ranking_error" in r for r in records)
    metrics: Dict[str, float | int | str] = {"n": n}
    if has_pool_flag:
        pool_hits = [r for r in records if r.get("oracle_in_candidate_pool")]
        metrics["candidate_pool_hit_rate"] = len(pool_hits) / max(1, n)
        metrics["conditional_success_given_pool_hit"] = (
            sum(1 for r in pool_hits if r.get("success")) / max(1, len(pool_hits))
        )
    if has_ranking:
        metrics["ranking_error_rate"] = sum(1 for r in records if r.get("ranking_error")) / max(1, n)
    if has_pool_size:
        metrics["candidate_pool_avg_size"] = sum(float(r.get("candidate_pool_size", 0.0)) for r in records) / max(1, n)
    return metrics


def _uniform_metrics(records: List[Dict], summary: Dict | None = None) -> Dict:
    metrics = {}
    if records:
        metrics.update(aggregate_method_metrics(records))
        metrics.update(_candidate_metrics(records))
    if summary:
        metrics.update(summary)
        metrics.update(_candidate_metrics(records))
        metrics.setdefault("n", len(records))
    return metrics


def _dataset_rows(run: Dict) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in run["preds"]:
        grouped[str(row.get("dataset", "unknown"))].append(row)
    out = []
    for dataset, records in sorted(grouped.items()):
        row = {
            "setting": run["label"],
            "method": run["method"],
            "dataset": dataset,
        }
        row.update(_uniform_metrics(records))
        out.append(row)
    return out


def _significance_rows(runs: List[Dict], ref_label: str, bootstrap_n: int, seed: int) -> List[Dict]:
    label_to_run = {run["label"]: run for run in runs}
    if ref_label not in label_to_run:
        raise ValueError(f"ref_label={ref_label} not found. Available: {sorted(label_to_run)}")
    ref = label_to_run[ref_label]
    ref_success = {str(r["task_id"]): 1 if r.get("success") else 0 for r in ref["preds"]}
    rows = []
    for run in runs:
        if run["label"] == ref_label:
            continue
        other_success = {str(r["task_id"]): 1 if r.get("success") else 0 for r in run["preds"]}
        common = sorted(set(ref_success) & set(other_success))
        if not common:
            continue
        a = [ref_success[k] for k in common]
        b = [other_success[k] for k in common]
        mcnemar = mcnemar_test_paired(a, b)
        boot = bootstrap_mean_diff(b, a, n_resamples=bootstrap_n, seed=seed)
        rows.append(
            {
                "reference": ref_label,
                "other": run["label"],
                "n_common": len(common),
                "reference_success_rate": sum(a) / max(1, len(a)),
                "other_success_rate": sum(b) / max(1, len(b)),
                "delta_other_minus_reference": (sum(b) - sum(a)) / max(1, len(a)),
                "mcnemar_p_approx": mcnemar.get("p_approx"),
                "mcnemar_b_ref_win": mcnemar.get("b"),
                "mcnemar_c_other_win": mcnemar.get("c"),
                "bootstrap_mean_diff_other_minus_reference": boot.get("mean_diff"),
                "bootstrap_ci95_low": (boot.get("ci95") or ["", ""])[0],
                "bootstrap_ci95_high": (boot.get("ci95") or ["", ""])[1],
            }
        )
    return rows


def _write_latex_table(path: Path, aggregate_rows: List[Dict]) -> None:
    fields = [
        "setting",
        "success_rate",
        "candidate_pool_hit_rate",
        "ranking_error_rate",
        "first_error_step",
        "recovery_at_error",
        "plan_feasibility",
        "constraint_violation_rate",
    ]
    lines = [
        "| Setting | Success | Pool Hit | Ranking Err. | First Err. | Recovery | Feasible | Violation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        vals = []
        for key in fields[1:]:
            value = row.get(key, "")
            vals.append("" if value == "" else f"{float(value):.4f}")
        lines.append(f"| {row.get('setting', '')} | " + " | ".join(vals) + " |")
    ensure_dir(str(path.parent))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(runs: List[Dict], out_dir: Path, ref_label: str, bootstrap_n: int, seed: int) -> None:
    aggregate_rows = []
    for run in runs:
        row = {
            "setting": run["label"],
            "method": run["method"],
            "run_dir": run["dir"],
        }
        row.update(_uniform_metrics(run["preds"], run["summary"]))
        aggregate_rows.append(row)

    aggregate_fields = ["setting", "method", "run_dir"] + METRIC_KEYS
    dataset_fields = ["setting", "method", "dataset"] + METRIC_KEYS
    significance_fields = [
        "reference",
        "other",
        "n_common",
        "reference_success_rate",
        "other_success_rate",
        "delta_other_minus_reference",
        "mcnemar_p_approx",
        "mcnemar_b_ref_win",
        "mcnemar_c_other_win",
        "bootstrap_mean_diff_other_minus_reference",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
    ]

    dataset_rows = []
    for run in runs:
        dataset_rows.extend(_dataset_rows(run))
    sig_rows = _significance_rows(runs, ref_label=ref_label, bootstrap_n=bootstrap_n, seed=seed)

    _write_csv(out_dir / "kgqa_baselines_aggregate.csv", aggregate_rows, aggregate_fields)
    _write_csv(out_dir / "kgqa_baselines_by_dataset.csv", dataset_rows, dataset_fields)
    _write_csv(out_dir / f"kgqa_baselines_significance_vs_{ref_label}.csv", sig_rows, significance_fields)
    _write_latex_table(out_dir / "kgqa_baselines_latex_table.md", aggregate_rows)
    dump_json(
        str(out_dir / "kgqa_baselines_manifest.json"),
        {
            "reference": ref_label,
            "runs": [{"label": r["label"], "method": r["method"], "dir": r["dir"]} for r in runs],
            "outputs": {
                "aggregate": str(out_dir / "kgqa_baselines_aggregate.csv"),
                "by_dataset": str(out_dir / "kgqa_baselines_by_dataset.csv"),
                "significance": str(out_dir / f"kgqa_baselines_significance_vs_{ref_label}.csv"),
                "latex_table": str(out_dir / "kgqa_baselines_latex_table.md"),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready KGQA baselines and mechanistic diagnostics.")
    parser.add_argument("--base-config", type=str, default="configs/eval_torch_kgqa.tune4.high_recall_multistage.cwq_webqsp.prefixpen.json")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, default="")
    parser.add_argument("--out_root", type=str, default="results/kgqa_baselines")
    parser.add_argument("--train-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--include-datasets", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--baselines", type=str, default="", help="Comma-separated subset, e.g. diplan_full,no_value,retrieval_only")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ref-label", type=str, default="diplan_full")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--run-llm-agent", action="store_true")
    parser.add_argument("--llm-api-base", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY")
    parser.add_argument("--llm-model", type=str, default="Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-tokens", type=int, default=128)
    parser.add_argument("--llm-timeout-s", type=int, default=30)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-top-k", type=int, default=8)
    args = parser.parse_args()

    # Keep BLAS/OpenMP from inheriting invalid shell values on shared servers.
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if not os.environ.get(env_name, "").isdigit():
            os.environ[env_name] = "1"

    out_root = Path(args.out_root)
    runs_dir = out_root / "runs"
    cfg_dir = out_root / "generated_configs"
    log_dir = out_root / "logs"
    tables_dir = out_root / "tables"
    ensure_dir(str(out_root))

    base_cfg = _base_eval_config(args)
    specs = _baseline_specs(args, base_cfg)
    if not specs:
        raise ValueError("No baselines selected.")

    manifest = {
        "base_config": args.base_config,
        "ae_ckpt": args.ae_ckpt,
        "planner_ckpt": args.planner_ckpt,
        "value_ckpt": args.value_ckpt,
        "baselines": [],
    }

    completed_runs = []
    for spec in specs:
        cfg = _config_for_spec(base_cfg, spec)
        run_dir = runs_dir / spec["label"]
        manifest["baselines"].append(
            {
                "label": spec["label"],
                "runner": spec["runner"],
                "description": spec.get("description", ""),
                "run_dir": str(run_dir),
                "overrides": spec.get("overrides", {}),
            }
        )
        if args.skip_existing and (run_dir / "summary_metrics.json").exists() and (run_dir / "predictions.jsonl").exists():
            print(f"[skip] {spec['label']} already exists at {run_dir}")
        else:
            if spec["runner"] == "evaluate_torch":
                _run_eval_baseline(args, spec, cfg, run_dir, cfg_dir, log_dir)
            elif spec["runner"] == "retrieval_baseline":
                _run_retrieval_baseline(spec, cfg, run_dir, cfg_dir, log_dir)
            elif spec["runner"] == "llm_agent":
                _run_llm_baseline(args, spec, cfg, run_dir, cfg_dir, log_dir)
            else:
                raise ValueError(f"Unsupported runner: {spec['runner']}")
        completed_runs.append(_load_run(spec["label"], run_dir))

    _summarize(
        completed_runs,
        out_dir=tables_dir,
        ref_label=args.ref_label,
        bootstrap_n=args.bootstrap,
        seed=args.seed,
    )
    dump_json(str(out_root / "kgqa_baselines_run_manifest.json"), manifest)
    print(f"[ok] KGQA baselines finished: {out_root}")
    print(f"[ok] Tables: {tables_dir}")


if __name__ == "__main__":
    main()
