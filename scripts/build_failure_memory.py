import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/failure_memory")
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    failures = []
    for row in rows:
        if row.get("success", False):
            continue
        executed = row.get("executed_path", [])
        if not isinstance(executed, list) or not executed:
            continue
        failures.append(
            {
                "task_id": row.get("task_id", ""),
                "dataset": row.get("dataset", ""),
                "query": row.get("query", ""),
                "query_tokens": row.get("query_tokens", []),
                "oracle_path": row.get("oracle_path", []),
                "executed_path": executed,
                "first_error_step": row.get("first_error_step", 1),
                "violations": row.get("violations", []),
            }
        )
        if args.max_rows > 0 and len(failures) >= args.max_rows:
            break

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    dump_jsonl(str(out_dir / "failure_memory.jsonl"), failures)
    dump_json(
        str(out_dir / "manifest.json"),
        {
            "source_predictions": args.predictions,
            "total_predictions": len(rows),
            "failures": len(failures),
            "failure_memory": str(out_dir / "failure_memory.jsonl"),
        },
    )
    print(f"[ok] wrote {len(failures)} failures to {out_dir / 'failure_memory.jsonl'}")


if __name__ == "__main__":
    main()
