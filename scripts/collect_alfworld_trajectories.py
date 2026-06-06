"""Collect ALFWorld expert (handcoded) trajectories as DiPLaN plan data (paper §7.1).

Runs the built-in handcoded expert on the ALFWorld TextWorld ``train`` split (the
only split that exposes ``infos["extra.expert_plan"]``), keeps successful episodes,
and writes them in the SAME schema as the KGQA processed data so the existing torch
trainers (autoencoder / diffusion / value / constraint) work unchanged.

Output (in data/long_horizon/alfworld_processed/):
    train.jsonl, val.jsonl, test.jsonl   with rows:
      {task_id, dataset, question, query_tokens, oracle_path, raw_oracle_path,
       trap_path, candidate_paths, constraints}

MUST run on the ALFWorld server (needs the ``alfworld`` package + game data).

Example:
    export ALFWORLD_DATA=/root/autodl-tmp/DiPLaN/data/long_horizon/alfworld
    python scripts/collect_alfworld_trajectories.py \
        --data_root "$ALFWORLD_DATA" --config "$ALFWORLD_DATA/base_config.tw.yaml" \
        --episodes 3000 --max_steps 60 --out data/long_horizon/alfworld_processed
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Reuse the env-bootstrap + goal helpers from the heuristic runner.
sys.path.insert(0, str(ROOT / "scripts"))
from run_alfworld_diplan_agent import (  # noqa: E402
    _extract_goal,
    _first,
    _goal_keywords,
    _make_env,
    _parse_goal,
    _text,
    _update_memory,
)
from run_alfworld_diplan_diffusion import _episode_query_tokens, _new_memory  # noqa: E402

from src.diplan.alfworld_plan import goal_query_tokens, normalize_action  # noqa: E402
from src.diplan.io_utils import dump_json, dump_jsonl  # noqa: E402


TRANSFORM_STAGE = {"heat": "HEAT", "cool": "COOL", "clean": "CLEAN"}


def _expert_next_action(infos: Dict[str, Any]) -> str:
    plan = infos.get("extra.expert_plan")
    if plan and len(plan) > 0 and plan[0]:
        return plan[0][0]
    return ""


def _build_constraints(oracle: List[str], spec: Dict[str, Any]) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {
        "max_steps": len(oracle) + 6,
        "banned_relations": [],
        "forbidden_actions": [],
    }
    transform = spec.get("transform", "")
    if transform in TRANSFORM_STAGE and spec.get("requires_put", False):
        # The object must be transformed before it is placed (paper-style ordering).
        constraints["required_before"] = {"PUT": [TRANSFORM_STAGE[transform]]}
    return constraints


def _candidate_paths(oracle: List[str]) -> List[List[str]]:
    return [m["path"] for m in _candidate_metadata(oracle) if not m.get("is_oracle", False)]


def _candidate_metadata(oracle: List[str]) -> List[Dict[str, Any]]:
    """Executable-supervision pool for diffusion/value/constraint training.

    ``is_executable`` here means "complete successful executable plan", not just
    syntactically legal prefix. Diffusion should clone high-scoring executable
    futures, while value/constraint models can use the corruptions as negatives.
    """
    cands: List[Dict[str, Any]] = []
    if oracle:
        cands.append(
            {
                "path": list(oracle),
                "is_oracle": True,
                "is_executable": True,
                "executable_score": 1.0,
                "corruption_type": "oracle",
            }
        )
    if len(oracle) >= 2:
        cands.append(
            {
                "path": oracle[:-1],
                "is_oracle": False,
                "is_executable": False,
                "executable_score": 0.0,
                "corruption_type": "truncated_final_step",
            }
        )
    if len(oracle) >= 3:
        swapped = list(oracle)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        if swapped != oracle:
            cands.append(
                {
                    "path": swapped,
                    "is_oracle": False,
                    "is_executable": False,
                    "executable_score": 0.0,
                    "corruption_type": "local_order_swap",
                }
            )
    if len(oracle) >= 2:
        reversed_path = list(reversed(oracle))
        if reversed_path != oracle:
            cands.append(
                {
                    "path": reversed_path,
                    "is_oracle": False,
                    "is_executable": False,
                    "executable_score": 0.0,
                    "corruption_type": "reversed_order",
                }
            )
    if oracle:
        utility_loop = ["OTHER::look"] + list(oracle[: max(1, len(oracle) - 1)])
        cands.append(
            {
                "path": utility_loop,
                "is_oracle": False,
                "is_executable": False,
                "executable_score": 0.0,
                "corruption_type": "utility_loop_prefix",
            }
        )

    out: List[Dict[str, Any]] = []
    seen = set()
    for item in cands:
        key = tuple(item["path"])
        if item["path"] and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _run_episode(env: Any, episode_id: int, max_steps: int) -> Dict[str, Any] | None:
    obs, infos = env.reset()
    obs_text = _text(obs)
    goal = _extract_goal(obs_text)
    if not goal:
        return None
    spec = _parse_goal(goal)
    first_admissible = list((infos.get("admissible_commands") or [[]])[0])
    query_tokens = goal_query_tokens(_goal_keywords(goal), first_admissible)
    state_args = argparse.Namespace(state_conditioning=True)
    memory = _new_memory()

    raw_actions: List[str] = []
    plan_tokens: List[str] = []
    state_query_tokens_by_prefix: List[List[str]] = []
    score = 0.0
    done = False
    for _ in range(max_steps):
        admissible = list((infos.get("admissible_commands") or [[]])[0])
        _update_memory(memory, obs_text, raw_actions[-1] if raw_actions else "", admissible)
        state_query_tokens_by_prefix.append(
            _episode_query_tokens(goal, spec, admissible, memory, state_args)
        )
        next_action = _expert_next_action(infos)
        if not next_action:
            break
        raw_actions.append(next_action)
        plan_tokens.append(normalize_action(next_action))
        obs, scores, dones, infos = env.step([next_action])
        score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        if done:
            break

    success = bool(done and score > 0)
    if not success or not plan_tokens:
        return None
    constraints = _build_constraints(plan_tokens, spec)
    return {
        "task_id": f"alfworld::{episode_id}",
        "dataset": "alfworld",
        "question": goal,
        "query_tokens": query_tokens,
        "oracle_path": plan_tokens,
        "raw_oracle_path": raw_actions,
        "state_query_tokens_by_prefix": state_query_tokens_by_prefix,
        "trap_path": [],
        "candidate_paths": _candidate_paths(plan_tokens),
        "candidate_metadata": _candidate_metadata(plan_tokens),
        "constraints": constraints,
        "meta": {"transform": spec.get("transform", ""), "num_steps": len(plan_tokens)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ALFWorld handcoded-expert trajectories for DiPLaN.")
    parser.add_argument("--data_root", type=str, default="data/long_horizon/alfworld")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--out", type=str, default="data/long_horizon/alfworld_processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    os.environ["ALFWORLD_DATA"] = str(data_root)
    # 'train' is the only split that exposes the handcoded expert plan.
    env = _make_env(Path(args.config).resolve(), data_root, "train", batch_size=1)

    rows: List[Dict[str, Any]] = []
    attempted = 0
    for episode_id in range(int(args.episodes)):
        attempted += 1
        row = _run_episode(env, episode_id, int(args.max_steps))
        if row is not None:
            rows.append(row)
        if (episode_id + 1) % 50 == 0:
            print(f"[collect] {episode_id + 1}/{args.episodes} kept={len(rows)}")

    if not rows:
        raise RuntimeError("No successful expert trajectories collected; check expert_type/handcoded + split=train.")

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    n = len(rows)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]

    out_dir = Path(args.out)
    dump_jsonl(str(out_dir / "train.jsonl"), train)
    dump_jsonl(str(out_dir / "val.jsonl"), val)
    dump_jsonl(str(out_dir / "test.jsonl"), test)
    vocab_tokens = sorted({t for r in rows for t in r["oracle_path"]})
    dump_json(
        str(out_dir / "manifest.json"),
        {
            "attempted": attempted,
            "collected": n,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "abstract_plan_vocab_size": len(vocab_tokens),
            "abstract_plan_vocab": vocab_tokens,
            "avg_plan_len": sum(len(r["oracle_path"]) for r in rows) / n,
        },
    )
    print(f"[ok] collected {n} successful trajectories -> {out_dir} (train={len(train)} val={len(val)} test={len(test)})")
    print(f"[ok] abstract plan vocab size = {len(vocab_tokens)}")


if __name__ == "__main__":
    main()
