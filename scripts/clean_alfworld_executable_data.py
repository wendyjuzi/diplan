"""Minimal-clean ALFWorld executable trajectory data for DiPLaN.

The handcoded ALFWorld expert often searches multiple concrete instances of the
same receptacle, e.g. shelf 1, shelf 2, ... . After instance-free normalization
these become repeated abstract tokens such as ``GOTO::shelf``. This script keeps
the embodied search signal while preventing the diffusion target from becoming a
long run of indistinguishable repeated tokens.

It deliberately does NOT shortest-path compress trajectories and does NOT remove
OPEN/CLOSE exploration. It only caps consecutive identical abstract actions.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stage(tok: str) -> str:
    tok = str(tok)
    if "::" in tok:
        return tok.split("::", 1)[0]
    return tok.split("_", 1)[0].upper() if tok else "OTHER"


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
    if len(oracle) >= 3:
        swapped = list(oracle)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        if swapped != oracle:
            cands.append(
                {
                    "path": swapped,
                    "is_oracle": False,
                    "is_executable": False,
                    "executable_score": 0.0,
                    "corruption_type": "local_order_swap",
                }
            )
    if len(oracle) >= 2:
        reversed_path = list(reversed(oracle))
        if reversed_path != oracle:
            cands.append(
                {
                    "path": reversed_path,
                    "is_oracle": False,
                    "is_executable": False,
                    "executable_score": 0.0,
                    "corruption_type": "reversed_order",
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


def _clean_path(path: List[str], max_same_goto: int, max_same_other: int, max_same_action: int) -> Tuple[List[str], List[int]]:
    cleaned: List[str] = []
    kept_indices: List[int] = []
    last = None
    run = 0
    for i, tok in enumerate(path):
        if tok == last:
            run += 1
        else:
            last = tok
            run = 1
        st = _stage(tok)
        cap = max_same_action
        if st == "GOTO":
            cap = max_same_goto
        elif st == "OTHER":
            cap = max_same_other
        if run <= cap:
            cleaned.append(tok)
            kept_indices.append(i)
    return cleaned, kept_indices


def _clean_row(row: Dict[str, Any], args) -> Tuple[Dict[str, Any], Dict[str, float]]:
    old_path = list(row.get("oracle_path", []))
    new_path, kept = _clean_path(
        old_path,
        max_same_goto=int(args.max_same_goto),
        max_same_other=int(args.max_same_other),
        max_same_action=int(args.max_same_action),
    )
    new_row = dict(row)
    new_row["oracle_path"] = new_path
    states = list(row.get("state_query_tokens_by_prefix") or [])
    if states:
        # State before compressed step k should be the state before the original
        # action kept at compressed index k.
        new_row["state_query_tokens_by_prefix"] = [
            states[i] for i in kept if i < len(states)
        ]
    new_meta = _candidate_metadata(new_path)
    new_row["candidate_metadata"] = new_meta
    new_row["candidate_paths"] = [m["path"] for m in new_meta if not m.get("is_oracle", False)]
    old_len = len(old_path)
    new_len = len(new_path)
    return new_row, {
        "old_len": float(old_len),
        "new_len": float(new_len),
        "removed": float(old_len - new_len),
    }


def _process_file(src: Path, dst: Path, args) -> Dict[str, float]:
    rows = _load_jsonl(src)
    out = []
    stats = []
    for row in rows:
        cleaned, st = _clean_row(row, args)
        if cleaned.get("oracle_path"):
            out.append(cleaned)
            stats.append(st)
    _dump_jsonl(dst, out)
    old_total = sum(s["old_len"] for s in stats)
    new_total = sum(s["new_len"] for s in stats)
    return {
        "rows": float(len(out)),
        "old_avg_len": old_total / max(1, len(stats)),
        "new_avg_len": new_total / max(1, len(stats)),
        "removed_tokens": old_total - new_total,
        "removed_rate": (old_total - new_total) / max(1.0, old_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_same_goto", type=int, default=3)
    parser.add_argument("--max_same_other", type=int, default=1)
    parser.add_argument("--max_same_action", type=int, default=2)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    summary: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        summary[split] = _process_file(src, out_dir / f"{split}.jsonl", args)
    (out_dir / "clean_manifest.json").write_text(
        json.dumps(
            {
                "source": str(in_dir),
                "max_same_goto": int(args.max_same_goto),
                "max_same_other": int(args.max_same_other),
                "max_same_action": int(args.max_same_action),
                "splits": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] wrote cleaned data to {out_dir}")


if __name__ == "__main__":
    main()
