import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_json


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _metric_mean_std(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(mean(vals)), "std": float(stdev(vals))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--python_exec", type=str, required=True)
    parser.add_argument("--java_home", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--task_names", type=str, nargs="*", default=["boil", "chemistry-mix"])
    parser.add_argument("--per_task_limit", type=int, default=6)
    parser.add_argument("--simplification", type=str, default="easy")
    parser.add_argument("--path_mode", type=str, default="action", choices=["action", "stage", "stage_dedup"])
    parser.add_argument("--task_profile", type=str, default="all", choices=["all", "easy"])
    parser.add_argument("--env_step_limit", type=int, default=80)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--data_root", type=str, default="data/scienceworld_multiseed")
    parser.add_argument("--results_root", type=str, default="results/multiseed_scienceworld")
    parser.add_argument("--runs_root", type=str, default="runs/multiseed_scienceworld")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    data_root = (repo / args.data_root).resolve()
    results_root = (repo / args.results_root).resolve()
    runs_root = (repo / args.runs_root).resolve()
    ensure_dir(str(data_root))
    ensure_dir(str(results_root))
    ensure_dir(str(runs_root))

    run_env = dict(**subprocess.os.environ)
    run_env["JAVA_HOME"] = args.java_home
    run_env["PATH"] = str(Path(args.java_home) / "bin") + ";" + run_env.get("PATH", "")

    by_seed_summary = {}
    metric_bucket = defaultdict(list)

    for seed in args.seeds:
        seed_data = data_root / f"seed_{seed}"
        seed_result = results_root / f"seed_{seed}"
        seed_run = runs_root / f"seed_{seed}"
        ensure_dir(str(seed_data))
        ensure_dir(str(seed_result))
        ensure_dir(str(seed_run))

        # 1) Export real ScienceWorld trajectories for this seed.
        cmd_export = [
            args.python_exec,
            "scripts/prepare_scienceworld_data.py",
            "--out",
            str(seed_data),
            "--seed",
            str(seed),
            "--per_task_limit",
            str(int(args.per_task_limit)),
            "--env_step_limit",
            str(int(args.env_step_limit)),
            "--simplification",
            str(args.simplification),
            "--path_mode",
            str(args.path_mode),
        ]
        if args.task_names:
            cmd_export += ["--task_names"] + list(args.task_names)
        _run(cmd_export, repo, run_env)

        # 2) Run end-to-end pipeline.
        cmd_pipe = [
            args.python_exec,
            "scripts/run_scienceworld_pipeline.py",
            "--python_exec",
            args.python_exec,
            "--seed",
            str(seed),
            "--processed_dir",
            str(seed_data.relative_to(repo)),
            "--out_root",
            str(seed_result.relative_to(repo)),
            "--runs_root",
            str(seed_run.relative_to(repo)),
            "--java_home",
            args.java_home,
            "--task_profile",
            str(args.task_profile),
        ]
        if args.use_cuda:
            cmd_pipe.append("--use_cuda")
        _run(cmd_pipe, repo, run_env)

        sm_path = seed_result / "test_with_value_eval" / "summary_metrics.json"
        sm = load_json(str(sm_path))
        row = sm.get("diplan_torch_mem", {})
        by_seed_summary[str(seed)] = row
        for k, v in row.items():
            if isinstance(v, (int, float)):
                metric_bucket[k].append(float(v))

    agg = {k: _metric_mean_std(vs) for k, vs in metric_bucket.items()}
    dump_json(
        str(results_root / "multiseed_summary_mean_std.json"),
        {
            "seeds": [int(x) for x in args.seeds],
            "task_names": list(args.task_names),
            "per_task_limit": int(args.per_task_limit),
            "simplification": str(args.simplification),
            "path_mode": str(args.path_mode),
            "task_profile": str(args.task_profile),
            "by_seed": by_seed_summary,
            "aggregate": agg,
        },
    )
    print(f"[ok] multiseed summary saved: {results_root / 'multiseed_summary_mean_std.json'}")


if __name__ == "__main__":
    main()
