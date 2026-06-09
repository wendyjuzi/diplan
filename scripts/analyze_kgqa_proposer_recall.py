"""Measure oracle-action recall of the KGQA action proposer.

This script audits the first bottleneck in ToG/FLARE-style planning: whether the
LLM proposer/pruner includes the oracle next relation in its top-k candidates.
If oracle recall@k is low, SingleStep/Beam/Lookahead/FLARE cannot recover.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, load_config
from src.diplan.kg_env import KGEnv
from src.diplan.planners import StubScorer


def make_scorer(cfg):
    kind = str(cfg.get("scorer", "stub")).lower()
    if kind == "stub":
        return StubScorer(seed=int(cfg.get("seed", 42)))
    if kind == "llm":
        from src.diplan.kgqa_prompts import LLMScorer

        return LLMScorer.from_config(cfg)
    raise ValueError(f"Unknown scorer: {kind}")


def stream_rows(path: Path, max_tasks: int):
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_tasks > 0 and n >= max_tasks:
                break
            n += 1
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--path", default="", help="Overrides config test_path.")
    parser.add_argument("--out", default="results/kgqa_proposer_recall")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 8, 16])
    parser.add_argument("--progress_every", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = Path(args.path or cfg["test_path"])
    max_tasks = int(cfg.get("max_tasks", 0) if args.max_tasks is None else args.max_tasks)
    ks = sorted(set(args.ks))
    max_k = max(ks)
    scorer = make_scorer(cfg)

    records = []
    t0 = time.time()
    for row_idx, row in enumerate(stream_rows(path, max_tasks), 1):
        env = KGEnv.from_rog_row(row, int(row["constraints"]["max_steps"]))
        state = env.reset()
        oracle = list(row.get("oracle_path") or [])
        for step_idx, oracle_rel in enumerate(oracle):
            adm = env.admissible_relations(state)
            if oracle_rel not in set(adm):
                records.append(
                    {
                        "task_id": row.get("task_id"),
                        "step": step_idx,
                        "oracle_rel": oracle_rel,
                        "admissible_size": len(adm),
                        "oracle_admissible": False,
                        "picked": [],
                        **{f"hit@{k}": False for k in ks},
                    }
                )
                break
            picked = scorer.propose(row["question"], oracle[:step_idx], adm, max_k)
            rec = {
                "task_id": row.get("task_id"),
                "step": step_idx,
                "oracle_rel": oracle_rel,
                "admissible_size": len(adm),
                "oracle_admissible": True,
                "picked": picked,
            }
            for k in ks:
                rec[f"hit@{k}"] = oracle_rel in picked[:k]
            records.append(rec)
            state = env.step(state, oracle_rel)
        if args.progress_every and (row_idx == 1 or row_idx % args.progress_every == 0):
            client = getattr(scorer, "client", None)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "rows": row_idx,
                        "steps": len(records),
                        "elapsed_s": round(time.time() - t0, 2),
                        "llm_calls": getattr(client, "calls", None),
                        "llm_errors": getattr(client, "errors", None),
                        "fallbacks": getattr(scorer, "fallbacks", None),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "n_steps": len(records),
        "n_tasks": len({r["task_id"] for r in records}),
        "oracle_admissible_rate": mean(1.0 if r["oracle_admissible"] else 0.0 for r in records) if records else 0.0,
        "admissible_size_mean": mean(r["admissible_size"] for r in records) if records else 0.0,
        "llm_calls": getattr(getattr(scorer, "client", None), "calls", 0),
        "llm_errors": getattr(getattr(scorer, "client", None), "errors", 0),
        "fallbacks": getattr(scorer, "fallbacks", 0),
    }
    for k in ks:
        summary[f"recall@{k}"] = mean(1.0 if r[f"hit@{k}"] else 0.0 for r in records) if records else 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(str(out / "summary.json"), summary)
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
