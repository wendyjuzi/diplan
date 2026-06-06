"""Export successful ALFWorld executions as executable DiPLaN training rows.

This is the distillation bridge for Executable-guided Diffusion Planning:

    exec-v4 successful rollout
      -> abstract executed plan tokens
      -> per-prefix symbolic state conditions
      -> train/val/test JSONL rows for AE + diffusion + value + constraint

Use it after running ``scripts/run_alfworld_diplan_diffusion.py`` with the
state-conditioning patch. Rows without successful completion are skipped.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_alfworld_diplan_agent import _goal_keywords, _parse_goal  # noqa: E402
from run_alfworld_diplan_diffusion import _exec_constraints  # noqa: E402
from src.diplan.alfworld_plan import goal_query_tokens  # noqa: E402
from src.diplan.io_utils import dump_json, dump_jsonl, load_jsonl  # noqa: E402


def _candidate_paths(oracle: List[str]) -> List[List[str]]:
    return [m["path"] for m in _candidate_metadata(oracle) if not m.get("is_oracle", False)]


def _candidate_metadata(oracle: List[str]) -> List[Dict[str, Any]]:
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
    if len(oracle) >= 4:
        cands.append(
            {
                "path": oracle[1:],
                "is_oracle": False,
                "is_executable": False,
                "executable_score": 0.0,
                "corruption_type": "missing_first_step",
            }
        )
    if len(oracle) >= 3:
        swapped = list(oracle)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        cands.append(
            {
                "path": swapped,
                "is_oracle": False,
                "is_executable": False,
                "executable_score": 0.0,
                "corruption_type": "local_order_swap",
            }
        )
    if oracle:
        cands.append(
            {
                "path": ["OTHER::look"] + list(oracle[: max(1, len(oracle) - 1)]),
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


def _row_from_prediction(pred: Dict[str, Any], idx: int, max_extra_steps: int) -> Dict[str, Any] | None:
    if not bool(pred.get("success", False)):
        return None
    oracle = list(pred.get("executed_plan_tokens") or [])
    if not oracle:
        return None
    goal = str(pred.get("goal", ""))
    spec = pred.get("structured_goal") or _parse_goal(goal)
    conds = pred.get("condition_tokens_by_prefix") or pred.get("state_query_tokens_by_prefix") or []
    query_tokens = list(conds[0]) if conds else goal_query_tokens(_goal_keywords(goal), [])
    constraints = _exec_constraints(spec, max_steps=len(oracle) + int(max_extra_steps))
    return {
        "task_id": f"alfworld_exec::{idx}",
        "dataset": "alfworld_exec",
        "question": goal,
        "query_tokens": query_tokens,
        "oracle_path": oracle,
        "raw_oracle_path": list(pred.get("actions") or []),
        "state_query_tokens_by_prefix": [list(x) for x in conds],
        "trap_path": [],
        "candidate_paths": _candidate_paths(oracle),
        "candidate_metadata": _candidate_metadata(oracle),
        "constraints": constraints,
        "meta": {
            "source_episode_id": pred.get("episode_id", idx),
            "source_method": pred.get("method", ""),
            "num_steps": len(oracle),
            "success": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export successful ALFWorld executions as DiPLaN training data.")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--max_extra_steps", type=int, default=6)
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    rows = []
    for i, pred in enumerate(preds):
        row = _row_from_prediction(pred, i, int(args.max_extra_steps))
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("No successful executable plans found in predictions.")

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    n = len(rows)
    n_test = int(n * float(args.test_frac))
    n_val = int(n * float(args.val_frac))
    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]

    out_dir = Path(args.out)
    dump_jsonl(str(out_dir / "train.jsonl"), train)
    dump_jsonl(str(out_dir / "val.jsonl"), val)
    dump_jsonl(str(out_dir / "test.jsonl"), test)
    dump_json(
        str(out_dir / "manifest.json"),
        {
            "source_predictions": str(Path(args.predictions).resolve()),
            "loaded_predictions": len(preds),
            "successful_rows": n,
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "avg_plan_len": sum(len(r["oracle_path"]) for r in rows) / n,
            "with_state_prefixes": sum(1 for r in rows if r.get("state_query_tokens_by_prefix")),
        },
    )
    print(f"[ok] exported {n} executable rows -> {out_dir}")


if __name__ == "__main__":
    main()
