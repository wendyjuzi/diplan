"""Evaluate candidate-conditioned denoising planner recall before ToG execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.diplan.candidate_diffusion import load_candidate_diffusion, score_candidate_relations
from src.diplan.io_utils import load_jsonl
from src.diplan.kg_env import KGEnv
from src.diplan.relation_scorer import question_text_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tasks", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8, 16])
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    args = ap.parse_args()

    rows = load_jsonl(args.path)
    if args.max_tasks > 0:
        rows = rows[: args.max_tasks]
    bundle = load_candidate_diffusion(args.ckpt)
    ranks = []
    records = []
    for row in rows:
        oracle = list(row.get("oracle_path", []) or [])
        if not oracle:
            continue
        max_steps = int((row.get("constraints") or {}).get("max_steps", max(1, len(oracle))))
        env = KGEnv.from_rog_row(row, max_steps=max_steps)
        state = env.reset()
        prefix = []
        query_tokens = list(row.get("query_tokens") or question_text_tokens(str(row.get("question", ""))))
        for step, gold in enumerate(oracle):
            candidates = env.admissible_relations(state)
            if gold not in candidates:
                break
            scores = score_candidate_relations(
                bundle,
                str(row.get("question", "")),
                query_tokens,
                candidates,
                executed_prefix=prefix,
                guidance_scale=args.guidance_scale,
            )
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            rank = order.index(candidates.index(gold)) + 1
            ranks.append(rank)
            records.append(
                {
                    "task_id": row.get("task_id"),
                    "step": step,
                    "oracle": gold,
                    "rank": rank,
                    "num_candidates": len(candidates),
                    "top": [
                        {"relation": candidates[i], "score": float(scores[i])}
                        for i in order[: max(args.ks)]
                    ],
                }
            )
            state = env.step(state, gold)
            prefix.append(gold)
            if env.is_terminal(state):
                break

    summary = {
        "n_tasks": len(rows),
        "n_steps": len(ranks),
        "mrr": mean([1.0 / r for r in ranks]) if ranks else 0.0,
        "mean_rank": mean(ranks) if ranks else None,
        "guidance_scale": args.guidance_scale,
    }
    for k in args.ks:
        summary[f"recall@{k}"] = mean([1.0 if r <= k else 0.0 for r in ranks]) if ranks else 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
