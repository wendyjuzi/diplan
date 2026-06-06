"""Inspect whether ALFWorld executable supervision is present before training."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--show", type=int, default=1)
    args = parser.parse_args()

    rows = load_jsonl(args.path)
    n = len(rows)
    with_state = sum(1 for r in rows if r.get("state_query_tokens_by_prefix"))
    meta_rows = sum(1 for r in rows if r.get("candidate_metadata"))
    exec_pos = 0
    nonexec = 0
    corrupt = {}
    for r in rows:
        for m in r.get("candidate_metadata", []) or []:
            if m.get("is_executable", False):
                exec_pos += 1
            else:
                nonexec += 1
            ct = str(m.get("corruption_type", "unknown"))
            corrupt[ct] = corrupt.get(ct, 0) + 1

    print(f"rows={n}")
    print(f"with_state_prefixes={with_state}")
    print(f"with_candidate_metadata={meta_rows}")
    print(f"candidate_exec_pos={exec_pos}")
    print(f"candidate_nonexec={nonexec}")
    print("corruption_types=" + ", ".join(f"{k}:{v}" for k, v in sorted(corrupt.items())))
    for i, r in enumerate(rows[: max(0, int(args.show))]):
        print(f"\n[example {i}] question={r.get('question')}")
        print("query_tokens=", r.get("query_tokens", [])[:20])
        states = r.get("state_query_tokens_by_prefix") or []
        print("state_prefix_0=", states[0][:30] if states else [])
        print("oracle_path=", r.get("oracle_path", [])[:20])
        print("candidate_metadata=", (r.get("candidate_metadata") or [])[:3])


if __name__ == "__main__":
    main()
