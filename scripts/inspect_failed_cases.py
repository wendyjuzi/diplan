import argparse
import random
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_jsonl, load_jsonl


def _to_path_str(path_list: List[str]) -> str:
    if not path_list:
        return "<empty>"
    return " -> ".join(path_list)


def _trim(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--test", type=str, default="")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    test_map: Dict[str, Dict] = {}
    if args.test:
        for row in load_jsonl(args.test):
            test_map[row["task_id"]] = row

    failed = [p for p in preds if not bool(p.get("success", False))]
    random.Random(args.seed).shuffle(failed)
    sample = failed[: args.num]

    if not sample:
        print("No failed cases found.")
        return

    rows_for_save = []
    q_w, e_w, o_w = 56, 56, 56
    print("=" * (q_w + e_w + o_w + 12))
    print(f"{'QUERY':<{q_w}} | {'EXECUTED_PATH':<{e_w}} | {'ORACLE_PATH':<{o_w}}")
    print("=" * (q_w + e_w + o_w + 12))
    for p in sample:
        task_id = p["task_id"]
        test_row = test_map.get(task_id, {})
        query = p.get("query") or test_row.get("question") or " ".join(test_row.get("query_tokens", []))
        executed = p.get("executed_path", [])
        oracle = p.get("oracle_path", [])
        executed_s = _to_path_str(executed)
        oracle_s = _to_path_str(oracle)
        print(f"{_trim(query, q_w):<{q_w}} | {_trim(executed_s, e_w):<{e_w}} | {_trim(oracle_s, o_w):<{o_w}}")
        rows_for_save.append(
            {
                "task_id": task_id,
                "query": query,
                "executed_path": executed,
                "oracle_path": oracle,
                "first_error_step": p.get("first_error_step"),
            }
        )
    print("=" * (q_w + e_w + o_w + 12))
    print(f"Failed total={len(failed)}, sampled={len(sample)}")

    if args.out:
        dump_jsonl(args.out, rows_for_save)
        print(f"Saved sampled cases to {args.out}")


if __name__ == "__main__":
    main()

