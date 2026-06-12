"""Re-process downloaded RoG KGQA rows into adjacency-preserving JSONL.

Unlike ``prepare_real_kgqa_data.py`` (which keeps only the oracle relation chain), this
script retains the full subgraph so the KG environment can do genuine T(s,a)/A(s) tree
search at eval time. The oracle relation path is recovered by shortest-path BFS inside
each question's subgraph (RoG rows carry no explicit relation chain), and a faithful
myopic trap is constructed per appendix E.5.

Input : data/rog/<dataset>/<split>.jsonl   (from scripts/download_rog_kgqa_data.py)
Output: data/rog_processed/<dataset>_<split>.jsonl

Each output row:
  {task_id, dataset, question, query_tokens, graph, q_entity, a_entity, answer,
   oracle_path, oracle_source, trap_path, constraints}
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import ensure_dir


def write_jsonl(path: str, rows) -> None:
    """Write JSONL with plain '\\n' endings (avoids os.linesep double-CR on Windows)."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
from src.diplan.kg_env import KGEnv, align_oracle_path, construct_myopic_trap
from src.diplan.real_data import extract_question, tokenize_question

EXTRACT_BUDGET = 6  # generous hop budget used only to recover an oracle path


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def stream_rows(path: Path, max_tasks: int):
    """Yield parsed JSONL rows, stopping early when ``max_tasks`` reached (0 = all)."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_tasks > 0 and n >= max_tasks:
                break
            n += 1
            yield json.loads(line)


def process_split(rows, dataset: str, rng: random.Random, max_budget: int):
    out = []
    sources = Counter()
    n_trap = 0
    n_in = 0
    for idx, row in enumerate(rows):
        n_in += 1
        question = extract_question(row) or str(row.get("question", "")).strip()
        graph = row.get("graph") or []
        q_entity = _as_list(row.get("q_entity"))
        a_entity = _as_list(row.get("a_entity"))
        if not question or not graph or not q_entity or not a_entity:
            sources["skip_incomplete"] += 1
            continue

        # 1) recover an executable oracle path via BFS over the real subgraph.
        ext_env = KGEnv.from_rog_row(row, max_steps=max_budget)
        oracle, source = align_oracle_path(ext_env, row.get("relation_path") or [])
        if oracle is None:
            sources["none"] += 1
            continue
        sources[source] += 1

        # 2) final per-row hop budget = oracle length + 1 step of slack (bounded).
        max_steps = max(2, min(max_budget, len(oracle) + 1))
        env = KGEnv.from_rog_row(row, max_steps=max_steps)
        qtokens = tokenize_question(question)

        # 3) faithful myopic trap on the same graph (may be None).
        trap = construct_myopic_trap(env, oracle, qtokens, rng)
        if trap is not None:
            n_trap += 1
        else:
            trap = list(oracle)  # no qualifying decoy -> Trap@1 effectively disabled

        out.append(
            {
                "task_id": f"{dataset}_{idx:07d}",
                "dataset": dataset,
                "question": question,
                "query_tokens": qtokens,
                "graph": [list(t) for t in graph],
                "q_entity": q_entity,
                "a_entity": a_entity,
                "answer": _as_list(row.get("answer")),
                "oracle_path": oracle,
                "oracle_source": source,
                "trap_path": trap,
                "constraints": {"max_steps": max_steps},
            }
        )
    stats = {
        "dataset": dataset,
        "input": n_in,
        "kept": len(out),
        "sources": dict(sources),
        "trap_constructed": n_trap,
        "align_rate": round(len(out) / max(1, n_in), 3),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["webqsp"])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--in_root", default="data/rog")
    ap.add_argument("--out_root", default="data/rog_processed")
    ap.add_argument("--max_tasks", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max_budget", type=int, default=EXTRACT_BUDGET)
    ap.add_argument(
        "--dev_fraction",
        type=float,
        default=0.0,
        help="Deterministically hold out this fraction of a processed train split as dev.",
    )
    ap.add_argument("--dev_name", default="dev")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0.0 <= args.dev_fraction < 1.0:
        ap.error("--dev_fraction must be in [0, 1)")

    ensure_dir(args.out_root)
    rng = random.Random(args.seed)
    for ds in args.datasets:
        for split in args.splits:
            in_path = Path(args.in_root) / ds / f"{split}.jsonl"
            if not in_path.exists():
                print(f"[skip] missing {in_path}")
                continue
            rows = stream_rows(in_path, args.max_tasks)
            out, stats = process_split(rows, ds, rng, args.max_budget)
            if split == "train" and args.dev_fraction > 0.0:
                if len(out) < 2:
                    raise ValueError("Need at least two aligned train rows to create a dev split")
                order = list(range(len(out)))
                random.Random(args.seed + 1009).shuffle(order)
                n_dev = max(1, round(len(out) * args.dev_fraction))
                n_dev = min(n_dev, len(out) - 1)
                dev_indices = set(order[:n_dev])
                train_out = [row for i, row in enumerate(out) if i not in dev_indices]
                dev_out = [row for i, row in enumerate(out) if i in dev_indices]
                train_path = Path(args.out_root) / f"{ds}_train.jsonl"
                dev_path = Path(args.out_root) / f"{ds}_{args.dev_name}.jsonl"
                write_jsonl(str(train_path), train_out)
                write_jsonl(str(dev_path), dev_out)
                print(
                    f"[ok] {train_path} n={len(train_out)}; "
                    f"{dev_path} n={len(dev_out)}; source_stats={stats}"
                )
                continue
            out_path = Path(args.out_root) / f"{ds}_{split}.jsonl"
            write_jsonl(str(out_path), out)
            print(f"[ok] {out_path}  {stats}")


if __name__ == "__main__":
    main()
