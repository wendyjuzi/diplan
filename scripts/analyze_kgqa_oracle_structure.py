"""Audit oracle-structure KGQA subgraph sandboxes.

This checks whether a RoG/FLARE-style local subgraph file is suitable for
controlled long-horizon planning experiments:

  * oracle_path is executable in the local graph;
  * oracle_path reaches an answer;
  * each oracle next action is present in A(s);
  * trap_path, when available, differs at step 1 and is globally dead;
  * root branching factor / oracle length distributions are reasonable.

The script does not call an LLM. It is a cheap environment sanity check before
running SingleStep/Beam/Lookahead/FLARE/DiPLaN comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.kg_env import KGEnv


def percentile(values, q: float):
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def load_rows(path: Path, max_rows: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def replay(env: KGEnv, path):
    state = env.reset()
    admissible_hits = []
    branching = []
    for rel in path:
        adm = env.admissible_relations(state)
        branching.append(len(adm))
        ok = rel in set(adm)
        admissible_hits.append(ok)
        if not ok:
            return state, admissible_hits, branching, False
        state = env.step(state, rel)
    return state, admissible_hits, branching, True


def analyze_row(row):
    max_steps = int(row.get("constraints", {}).get("max_steps", max(1, len(row.get("oracle_path", [])))))
    env = KGEnv.from_rog_row(row, max_steps=max_steps)
    oracle = list(row.get("oracle_path") or [])
    trap = list(row.get("trap_path") or [])

    root = env.reset()
    root_adm = env.admissible_relations(root)
    final_state, admissible_hits, branching, executable = replay(env, oracle)
    answer_reached = executable and env.answer_reached(final_state)

    trap_valid = False
    trap_dead = False
    trap_first_diff = False
    if trap and oracle:
        trap_first_diff = trap[0] != oracle[0]
        trap_next = env.neighbors(root, trap[0]) if trap[0] in set(root_adm) else frozenset()
        trap_dead = not env.answer_reachable_within(trap_next, max_steps - 1)
        trap_valid = trap_first_diff and trap[0] in set(root_adm) and trap_dead

    return {
        "task_id": row.get("task_id"),
        "dataset": row.get("dataset"),
        "oracle_len": len(oracle),
        "max_steps": max_steps,
        "root_branching": len(root_adm),
        "avg_branching_on_oracle": mean(branching) if branching else 0.0,
        "oracle_executable": executable,
        "oracle_answer_reached": answer_reached,
        "oracle_action_hit_rate": sum(admissible_hits) / max(1, len(admissible_hits)),
        "has_trap": bool(trap and oracle and trap[0] != oracle[0]),
        "trap_first_diff": trap_first_diff,
        "trap_dead": trap_dead,
        "trap_valid": trap_valid,
        "question": row.get("question", ""),
        "oracle_path": oracle,
        "trap_path": trap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--show_bad", type=int, default=5)
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    rows = load_rows(Path(args.path), args.max_rows)
    recs = [analyze_row(r) for r in rows]

    lens = [r["oracle_len"] for r in recs]
    root_b = [r["root_branching"] for r in recs]
    oracle_b = [r["avg_branching_on_oracle"] for r in recs]
    summary = {
        "n": len(recs),
        "oracle_executable_rate": mean(1.0 if r["oracle_executable"] else 0.0 for r in recs) if recs else 0.0,
        "oracle_answer_reached_rate": mean(1.0 if r["oracle_answer_reached"] else 0.0 for r in recs) if recs else 0.0,
        "oracle_action_hit_rate_mean": mean(r["oracle_action_hit_rate"] for r in recs) if recs else 0.0,
        "trap_available_rate": mean(1.0 if r["has_trap"] else 0.0 for r in recs) if recs else 0.0,
        "trap_valid_rate": mean(1.0 if r["trap_valid"] else 0.0 for r in recs) if recs else 0.0,
        "oracle_len_mean": mean(lens) if lens else 0.0,
        "oracle_len_p50": percentile(lens, 0.5),
        "oracle_len_p90": percentile(lens, 0.9),
        "root_branching_mean": mean(root_b) if root_b else 0.0,
        "root_branching_p90": percentile(root_b, 0.9),
        "oracle_branching_mean": mean(oracle_b) if oracle_b else 0.0,
        "oracle_source_counts": dict(Counter(str(r.get("oracle_source", "")) for r in rows)),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    bad = [
        r for r in recs
        if not r["oracle_executable"] or not r["oracle_answer_reached"] or (r["has_trap"] and not r["trap_valid"])
    ]
    for i, r in enumerate(bad[: args.show_bad]):
        print(
            f"\n[bad {i}] task_id={r['task_id']} dataset={r['dataset']} "
            f"exec={r['oracle_executable']} answer={r['oracle_answer_reached']} trap_valid={r['trap_valid']}"
        )
        print("question=", r["question"])
        print("oracle_path=", r["oracle_path"])
        print("trap_path=", r["trap_path"])

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "bad_examples": bad[: args.show_bad]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
