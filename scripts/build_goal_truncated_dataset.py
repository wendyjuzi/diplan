"""Build a goal-truncated ALFWorld executable dataset.

For each successful expert trajectory, keep actions up to and including the
goal-completing action. This removes post-completion wandering without replacing
embodied search with shortest paths.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_alfworld_goal_completion import completion_index  # noqa: E402
from collect_alfworld_trajectories import _build_constraints, _candidate_metadata  # noqa: E402


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


def _truncate_row(row: Dict[str, Any], drop_missing: bool) -> Dict[str, Any] | None:
    path = list(row.get("oracle_path", []))
    if not path:
        return None
    idx, reason = completion_index(row)
    if idx < 0:
        if drop_missing:
            return None
        idx = len(path) - 1
        reason = "fallback_full"

    new_path = path[: idx + 1]
    new_row = dict(row)
    new_row["oracle_path"] = new_path
    if isinstance(row.get("raw_oracle_path"), list):
        new_row["raw_oracle_path"] = list(row.get("raw_oracle_path", []))[: idx + 1]
    if isinstance(row.get("state_query_tokens_by_prefix"), list):
        new_row["state_query_tokens_by_prefix"] = list(row.get("state_query_tokens_by_prefix", []))[: idx + 1]

    spec = row.get("structured_goal", {}) or {}
    constraints = _build_constraints(new_path, spec)
    new_row["constraints"] = constraints
    meta = dict(row.get("meta", {}) or {})
    meta.update(
        {
            "original_num_steps": len(path),
            "num_steps": len(new_path),
            "goal_completion_index": idx,
            "goal_completion_reason": reason,
            "goal_truncated": True,
        }
    )
    new_row["meta"] = meta
    cand_meta = _candidate_metadata(new_path)
    new_row["candidate_metadata"] = cand_meta
    new_row["candidate_paths"] = [m["path"] for m in cand_meta if not m.get("is_oracle", False)]
    return new_row


def _process_split(src: Path, dst: Path, args) -> Dict[str, float]:
    rows = _load_jsonl(src)
    out = []
    old_len = 0
    new_len = 0
    missing = 0
    for row in rows:
        path = list(row.get("oracle_path", []))
        old_len += len(path)
        new_row = _truncate_row(row, bool(args.drop_missing_completion))
        if new_row is None:
            missing += 1
            continue
        new_len += len(new_row.get("oracle_path", []))
        out.append(new_row)
    _dump_jsonl(dst, out)
    return {
        "input_rows": float(len(rows)),
        "output_rows": float(len(out)),
        "missing_dropped": float(missing),
        "old_avg_len": old_len / max(1, len(rows)),
        "new_avg_len": new_len / max(1, len(out)),
        "removed_rate": (old_len - new_len) / max(1.0, float(old_len)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--drop_missing_completion", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    summary: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        summary[split] = _process_split(src, out_dir / f"{split}.jsonl", args)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "goal_truncation_manifest.json").write_text(
        json.dumps(
            {
                "source": str(in_dir),
                "drop_missing_completion": bool(args.drop_missing_completion),
                "splits": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] wrote goal-truncated data to {out_dir}")


if __name__ == "__main__":
    main()
