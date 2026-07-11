"""Install ScienceWorld, run smoke tests, and optionally export processed data.

This script is intended for Linux GPU servers where you want a single entry for:

1. checking Java availability
2. installing the ``scienceworld`` package
3. verifying the environment can be imported and queried
4. optionally exporting DiPLaN-ready processed JSONL files

Examples
--------
Install + smoke test only:
    python scripts/fetch_scienceworld.py

Install + export a small easy split:
    python scripts/fetch_scienceworld.py \
      --prepare_processed \
      --processed_out data/scienceworld_processed \
      --per_task_limit 4 \
      --simplification easy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _check_java() -> None:
    java = shutil.which("java")
    if not java:
        raise RuntimeError("Java not found. Install Java 8+ before using ScienceWorld.")
    _run([java, "-version"])


def _install_scienceworld(python_exec: str, upgrade: bool) -> None:
    cmd = [python_exec, "-m", "pip", "install"]
    if upgrade:
        cmd.append("-U")
    cmd.append("scienceworld")
    _run(cmd)


def _smoke_test(server_path: str, env_step_limit: int) -> dict[str, object]:
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("", server_path or None, envStepLimit=int(env_step_limit))
    try:
        task_names = list(env.getTaskNames())
        max_variations = None
        first_task = task_names[0] if task_names else None
        if first_task:
            try:
                max_variations = int(env.getMaxVariations(first_task))
            except Exception:
                max_variations = None
        return {
            "num_tasks": len(task_names),
            "first_tasks": task_names[:5],
            "first_task": first_task,
            "first_task_variations": max_variations,
        }
    finally:
        env.close()


def _prepare_processed(
    *,
    python_exec: str,
    processed_out: str,
    seed: int,
    server_path: str,
    env_step_limit: int,
    task_names: list[str],
    per_task_limit: int,
    simplification: str,
    path_mode: str,
) -> None:
    cmd = [
        python_exec,
        "scripts/prepare_scienceworld_data.py",
        "--out",
        processed_out,
        "--seed",
        str(seed),
        "--env_step_limit",
        str(env_step_limit),
        "--per_task_limit",
        str(per_task_limit),
        "--simplification",
        simplification,
        "--path_mode",
        path_mode,
    ]
    if server_path:
        cmd.extend(["--server_path", server_path])
    if task_names:
        cmd.append("--task_names")
        cmd.extend(task_names)
    _run(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--server_path", type=str, default="")
    parser.add_argument("--env_step_limit", type=int, default=120)
    parser.add_argument("--prepare_processed", action="store_true")
    parser.add_argument("--processed_out", type=str, default="data/scienceworld_processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_names", nargs="*", default=[])
    parser.add_argument("--per_task_limit", type=int, default=4)
    parser.add_argument("--simplification", type=str, default="easy")
    parser.add_argument("--path_mode", type=str, default="action", choices=["action", "stage", "stage_dedup"])
    args = parser.parse_args()

    os.chdir(ROOT)
    _check_java()
    _install_scienceworld(args.python_exec, upgrade=bool(args.upgrade))

    info = _smoke_test(args.server_path, int(args.env_step_limit))
    print("[scienceworld] import ok", flush=True)
    print(f"[scienceworld] num_tasks={info['num_tasks']}", flush=True)
    print(f"[scienceworld] first_tasks={info['first_tasks']}", flush=True)
    print(f"[scienceworld] first_task_variations={info['first_task_variations']}", flush=True)

    if args.prepare_processed:
        _prepare_processed(
            python_exec=args.python_exec,
            processed_out=args.processed_out,
            seed=int(args.seed),
            server_path=str(args.server_path),
            env_step_limit=int(args.env_step_limit),
            task_names=list(args.task_names),
            per_task_limit=int(args.per_task_limit),
            simplification=str(args.simplification),
            path_mode=str(args.path_mode),
        )
        print(f"[ok] processed ScienceWorld exported to {args.processed_out}", flush=True)
    else:
        print("[ok] ScienceWorld install + smoke test complete", flush=True)


if __name__ == "__main__":
    main()
