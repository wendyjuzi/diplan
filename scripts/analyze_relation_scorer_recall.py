"""Evaluate question-conditioned relation retrieval before agent execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.diplan.io_utils import load_jsonl
from src.diplan.relation_scorer import load_relation_scorer, score_relations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tasks", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8, 16])
    args = ap.parse_args()

    rows = load_jsonl(args.path)
    if args.max_tasks > 0:
        rows = rows[: args.max_tasks]
    bundle = load_relation_scorer(args.ckpt)
    ranks = []
    records = []
    for row in rows:
        relations = sorted({
            str(t[1])
            for t in row.get("graph", []) or []
            if isinstance(t, (list, tuple)) and len(t) >= 2
        })
        prefix = []
        for step, oracle in enumerate(row.get("oracle_path", []) or []):
            if oracle not in relations:
                continue
            scores = score_relations(
                bundle,
                str(row.get("question", "")),
                list(row.get("query_tokens", [])),
                relations,
                executed_prefix=prefix,
            )
            order = sorted(range(len(relations)), key=lambda i: scores[i], reverse=True)
            rank = order.index(relations.index(oracle)) + 1
            ranks.append(rank)
            records.append({
                "task_id": row.get("task_id"),
                "step": step,
                "oracle": oracle,
                "rank": rank,
                "top": [
                    {"relation": relations[i], "score": float(scores[i])}
                    for i in order[: max(args.ks)]
                ],
            })
            prefix.append(oracle)

    summary = {
        "n_tasks": len(rows),
        "n_steps": len(ranks),
        "mrr": mean([1.0 / r for r in ranks]) if ranks else 0.0,
        "mean_rank": mean(ranks) if ranks else None,
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
