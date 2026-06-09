"""ToG-style KGQA planning evaluation: compare planning strategies on the same
environment + evaluative signal (the paper's "differ only in action selection" axis).

Methods: single_step | beam | lookahead | flare | diplan_diffusion.
Metrics : Hits@1 (answer reached), Trap@1, First-Error Step, Recovery@First-Error.

Scorer backends:
  * stub -- offline lexical proxy (no network), for mechanics validation.
  * llm  -- served OpenAI-compatible endpoint (faithful signal); added in Phase C.

Usage:
  python scripts/run_kgqa_planning_eval.py --config configs/eval_kgqa_planning.json \
      --out results/kgqa_planning [--ae_ckpt ... --planner_ckpt ... --value_ckpt ...]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_config
from src.diplan.kg_env import KGEnv
from src.diplan.metrics import first_error_step, recovery_at_error, trap_at_1
from src.diplan.planners import (
    PlanContext,
    StubScorer,
    TrajectoryMemory,
    load_diffusion_bundle,
    make_planner,
)


def stream_rows(path: Path, max_tasks: int, include_datasets=None):
    include = set(include_datasets or [])
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if include and row.get("dataset") not in include:
                continue
            if max_tasks > 0 and n >= max_tasks:
                break
            n += 1
            yield row


def count_rows(path: Path, max_tasks: int, include_datasets=None) -> int:
    return sum(1 for _ in stream_rows(path, max_tasks, include_datasets))


def make_scorer(cfg):
    kind = str(cfg.get("scorer", "stub")).lower()
    if kind == "stub":
        return StubScorer(seed=int(cfg.get("seed", 42)))
    if kind == "llm":
        from src.diplan.kgqa_prompts import LLMScorer  # imported lazily (Phase C)
        return LLMScorer.from_config(cfg)
    raise ValueError(f"Unknown scorer: {kind}")


def run_episode(env: KGEnv, planner, ctx: PlanContext):
    state = env.reset()
    ctx.executed = []
    if ctx.traj_memory is not None:
        ctx.traj_memory.__init__(ctx.traj_memory.cap, ctx.traj_memory.sim)
    while not env.is_terminal(state):
        a = planner.select_action(env, state, ctx)
        if a is None:
            break
        ctx.executed.append(a)
        state = env.step(state, a)
    return {"executed_path": list(ctx.executed), "answer_reached": env.answer_reached(state)}


def aggregate(records):
    if not records:
        return {}
    trap_rows = [r for r in records if r["has_trap"]]
    return {
        "n": len(records),
        "hits@1": round(mean(1.0 if r["success"] else 0.0 for r in records), 4),
        "trap@1": round(mean(1.0 if r["trap_at_1"] else 0.0 for r in trap_rows), 4) if trap_rows else None,
        "first_error_step": round(mean(r["first_error_step"] for r in records), 3),
        "recovery@first_error": round(mean(1.0 if r["recovery_at_error"] else 0.0 for r in records), 4),
        "avg_steps": round(mean(r["steps"] for r in records), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/kgqa_planning")
    ap.add_argument("--ae_ckpt", default="")
    ap.add_argument("--planner_ckpt", default="")
    ap.add_argument("--value_ckpt", default="")
    ap.add_argument(
        "--progress_every",
        type=int,
        default=None,
        help="Print progress every N tasks. Defaults to config progress_every, then 1 for LLM runs.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    methods = cfg.get("methods", ["single_step", "beam", "lookahead", "flare"])
    import random

    rng = random.Random(int(cfg.get("seed", 42)))
    scorer = make_scorer(cfg)

    bundle = None
    if "diplan_diffusion" in methods:
        if not (args.ae_ckpt and args.planner_ckpt):
            raise SystemExit("diplan_diffusion needs --ae_ckpt and --planner_ckpt")
        bundle = load_diffusion_bundle(args.ae_ckpt, args.planner_ckpt, args.value_ckpt, cfg)

    planners = {m: make_planner(m, cfg, bundle) for m in methods}

    test_path = Path(cfg["test_path"])
    max_tasks = int(cfg.get("max_tasks", 0))
    include_datasets = cfg.get("include_datasets", [])
    progress_every = args.progress_every
    if progress_every is None:
        progress_every = int(cfg.get("progress_every", 1 if str(cfg.get("scorer", "stub")).lower() == "llm" else 0))
    total_tasks = count_rows(test_path, max_tasks, include_datasets)
    print(
        json.dumps(
            {
                "event": "start",
                "test_path": str(test_path),
                "selected_tasks": total_tasks,
                "max_tasks": max_tasks,
                "include_datasets": include_datasets,
                "methods": methods,
                "scorer": cfg.get("scorer", "stub"),
                "llm_api_base": cfg.get("llm_api_base"),
                "llm_model": cfg.get("llm_model"),
                "progress_every": progress_every,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    by_method = defaultdict(list)
    predictions = []

    run_t0 = time.time()
    for task_idx, row in enumerate(stream_rows(test_path, max_tasks, include_datasets), 1):
        env = KGEnv.from_rog_row(row, int(row["constraints"]["max_steps"]))
        oracle = row["oracle_path"]
        trap = row["trap_path"]
        has_trap = bool(trap) and bool(oracle) and trap[0] != oracle[0]
        for m in methods:
            method_t0 = time.time()
            ctx = PlanContext(
                question=row["question"],
                query_tokens=row["query_tokens"],
                scorer=scorer,
                rng=rng,
                bundle=bundle,
                traj_memory=TrajectoryMemory(
                    int(cfg.get("flare", {}).get("mem_cap", 200)),
                    float(cfg.get("flare", {}).get("mem_sim", 0.9)),
                ) if m == "flare" else None,
            )
            out = run_episode(env, planners[m], ctx)
            executed = out["executed_path"]
            rec = {
                "task_id": row["task_id"],
                "dataset": row["dataset"],
                "method": m,
                "success": out["answer_reached"],
                "first_error_step": first_error_step(executed, oracle),
                "recovery_at_error": recovery_at_error(executed, oracle),
                "trap_at_1": trap_at_1(executed, trap),
                "has_trap": has_trap,
                "steps": len(executed),
                "executed_path": executed,
                "oracle_path": oracle,
            }
            by_method[m].append(rec)
            predictions.append(rec)
            if progress_every and (task_idx == 1 or task_idx % progress_every == 0):
                client = getattr(scorer, "client", None)
                print(
                    json.dumps(
                        {
                            "event": "method_done",
                            "task": f"{task_idx}/{total_tasks}",
                            "task_id": row["task_id"],
                            "method": m,
                            "success": rec["success"],
                            "steps": rec["steps"],
                            "first_error_step": rec["first_error_step"],
                            "elapsed_s": round(time.time() - method_t0, 2),
                            "llm_calls": getattr(client, "calls", None),
                            "llm_errors": getattr(client, "errors", None),
                            "llm_fallbacks": getattr(scorer, "fallbacks", None),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if progress_every and (task_idx == 1 or task_idx % progress_every == 0):
            print(
                json.dumps(
                    {
                        "event": "task_done",
                        "task": f"{task_idx}/{total_tasks}",
                        "elapsed_total_s": round(time.time() - run_t0, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {m: aggregate(recs) for m, recs in by_method.items()}
    by_dataset = {}
    for m, recs in by_method.items():
        ds_groups = defaultdict(list)
        for r in recs:
            ds_groups[r["dataset"]].append(r)
        by_dataset[m] = {ds: aggregate(rs) for ds, rs in ds_groups.items()}

    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "summary_metrics.json"), summary)
    dump_json(str(Path(args.out) / "summary_by_dataset.json"), by_dataset)
    with open(Path(args.out) / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # diagnostics
    diag = {}
    if bundle is not None and "diplan_diffusion" in planners:
        dp = planners["diplan_diffusion"]
        tot = dp.grounded_hits + dp.grounded_misses
        diag["diffusion_grounding_hit_rate"] = round(dp.grounded_hits / max(1, tot), 4)
    client = getattr(scorer, "client", None)
    if client is not None:  # LLM scorer was used
        diag["llm_calls"] = getattr(client, "calls", 0)
        diag["llm_errors"] = getattr(client, "errors", 0)
        diag["llm_fallbacks"] = getattr(scorer, "fallbacks", 0)
    dump_json(str(Path(args.out) / "diagnostics.json"), diag)
    print(json.dumps({"summary": summary, "diag": diag}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
