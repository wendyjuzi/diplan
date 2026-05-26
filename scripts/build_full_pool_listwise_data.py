import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, dump_jsonl, load_jsonl


def _query_key(tokens: List[str]) -> Tuple[str, ...]:
    if not isinstance(tokens, list):
        return tuple()
    return tuple(str(x) for x in tokens)


def _mutate_path(pos: List[str], rel_bank: List[str], rng: random.Random) -> List[str]:
    out = list(pos)
    if out and rel_bank:
        j = rng.randrange(len(out))
        old = out[j]
        repl = rng.choice(rel_bank)
        if len(rel_bank) > 1 and repl == old:
            repl = rel_bank[(rel_bank.index(repl) + 1) % len(rel_bank)]
        out[j] = repl
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--pool_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_planned_and_executed", action="store_true")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    pred_rows = load_jsonl(args.predictions)
    rel_bank: List[str] = []
    if args.train_path:
        train_rows = load_jsonl(args.train_path)
        rel_bank = sorted(
            list(
                {
                    rel
                    for r in train_rows
                    for rel in r.get("oracle_path", [])
                    if isinstance(rel, str)
                }
            )
        )

    out_rows = []
    stats = defaultdict(int)

    for r in pred_rows:
        q_tokens = r.get("query_tokens", [])
        gold = r.get("oracle_path", [])
        if not isinstance(q_tokens, list) or not isinstance(gold, list) or not gold:
            continue

        negs: List[List[str]] = []
        seen = set()

        pool = r.get("candidate_pool_top", [])
        if isinstance(pool, list):
            for item in pool:
                cand = item.get("path") if isinstance(item, dict) else item
                if not isinstance(cand, list) or not cand:
                    continue
                if cand == gold:
                    continue
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                negs.append(cand)
                if len(negs) >= max(1, int(args.pool_size) - 1):
                    break

        if args.use_planned_and_executed:
            for key_name in ("planned_path", "executed_path"):
                cand = r.get(key_name, [])
                if not isinstance(cand, list) or not cand:
                    continue
                if cand == gold:
                    continue
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                negs.append(cand)
                if len(negs) >= max(1, int(args.pool_size) - 1):
                    break

        target_negs = max(1, int(args.pool_size) - 1)
        tries = 0
        max_tries = max(200, target_negs * 50)
        while len(negs) < target_negs and tries < max_tries:
            tries += 1
            if not rel_bank:
                break
            cand = _mutate_path(gold, rel_bank, rng)
            if cand == gold:
                continue
            key = tuple(cand)
            if key in seen:
                continue
            seen.add(key)
            negs.append(cand)

        negs = negs[:target_negs]
        all_cands = [gold] + negs
        out_rows.append(
            {
                "task_id": r.get("task_id"),
                "dataset": r.get("dataset", "unknown"),
                "query_tokens": q_tokens,
                "gold_path": gold,
                "candidates": all_cands,
                "neg_candidates": negs,
                "source_pool_size": len(pool) if isinstance(pool, list) else 0,
            }
        )
        stats["rows"] += 1
        stats["avg_neg_total"] += len(negs)
        stats["filled_to_target"] += 1 if len(negs) >= max(1, int(args.pool_size) - 1) else 0

    if stats["rows"] > 0:
        stats["avg_neg_total"] = stats["avg_neg_total"] / stats["rows"]
        stats["filled_to_target_rate"] = stats["filled_to_target"] / stats["rows"]
    dump_jsonl(args.out, out_rows)
    dump_json(
        str(args.out).replace(".jsonl", ".summary.json"),
        {
            "input_predictions": args.predictions,
            "output_rows": len(out_rows),
            "pool_size_target": int(args.pool_size),
            "stats": dict(stats),
        },
    )
    print(f"[ok] wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
