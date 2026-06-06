"""Analyze ALFWorld executable trajectory quality before retraining.

This script is intentionally diagnostic-only. It helps decide whether a failed
state-aware diffusion run is caused by conditioning/gating distribution shift or
by noisy expert demonstrations.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stage(tok: str) -> str:
    tok = str(tok)
    if "::" in tok:
        return tok.split("::", 1)[0]
    return tok.split("_", 1)[0].upper() if tok else "OTHER"


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return float(xs[idx])


def _row_stats(path: List[str]) -> Dict[str, float]:
    n = len(path)
    stages = [_stage(t) for t in path]
    consecutive_dup = sum(1 for a, b in zip(path, path[1:]) if a == b)
    consecutive_goto_dup = sum(
        1 for a, b in zip(path, path[1:])
        if a == b and _stage(a) == "GOTO"
    )
    consecutive_look_dup = sum(
        1 for a, b in zip(path, path[1:])
        if a == b and _stage(a) == "OTHER"
    )
    adjacent_open_close = sum(
        1 for a, b in zip(path, path[1:])
        if _stage(a) == "OPEN" and _stage(b) == "CLOSE"
    )
    return {
        "len": float(n),
        "other_rate": stages.count("OTHER") / max(1, n),
        "goto_rate": stages.count("GOTO") / max(1, n),
        "open_close_adjacent_rate": adjacent_open_close / max(1, n - 1),
        "consecutive_dup_rate": consecutive_dup / max(1, n - 1),
        "consecutive_goto_dup_rate": consecutive_goto_dup / max(1, n - 1),
        "consecutive_look_dup_rate": consecutive_look_dup / max(1, n - 1),
        "unique_token_rate": len(set(path)) / max(1, n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--show_examples", type=int, default=3)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.path))
    per = [_row_stats(list(r.get("oracle_path", []))) for r in rows]
    lens = [x["len"] for x in per]
    stage_counts = Counter()
    goal_type_counts = Counter()
    by_goal_len = defaultdict(list)
    for r in rows:
        path = list(r.get("oracle_path", []))
        for tok in path:
            stage_counts[_stage(tok)] += 1
        meta = r.get("meta", {}) or {}
        goal_type = str(meta.get("transform", "")) or "put/look/other"
        goal_type_counts[goal_type] += 1
        by_goal_len[goal_type].append(len(path))

    summary = {
        "n": len(rows),
        "len_mean": statistics.mean(lens) if lens else 0.0,
        "len_p50": _quantile(lens, 0.50),
        "len_p90": _quantile(lens, 0.90),
        "len_p95": _quantile(lens, 0.95),
        "other_rate_mean": statistics.mean(x["other_rate"] for x in per) if per else 0.0,
        "goto_rate_mean": statistics.mean(x["goto_rate"] for x in per) if per else 0.0,
        "open_close_adjacent_rate_mean": statistics.mean(x["open_close_adjacent_rate"] for x in per) if per else 0.0,
        "consecutive_dup_rate_mean": statistics.mean(x["consecutive_dup_rate"] for x in per) if per else 0.0,
        "consecutive_goto_dup_rate_mean": statistics.mean(x["consecutive_goto_dup_rate"] for x in per) if per else 0.0,
        "consecutive_look_dup_rate_mean": statistics.mean(x["consecutive_look_dup_rate"] for x in per) if per else 0.0,
        "unique_token_rate_mean": statistics.mean(x["unique_token_rate"] for x in per) if per else 0.0,
        "stage_counts": dict(stage_counts),
        "goal_type_counts": dict(goal_type_counts),
        "goal_type_len_mean": {
            k: statistics.mean(v) for k, v in sorted(by_goal_len.items()) if v
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.show_examples > 0:
        ranked = sorted(
            rows,
            key=lambda r: (
                _row_stats(list(r.get("oracle_path", [])))["consecutive_goto_dup_rate"],
                len(r.get("oracle_path", [])),
            ),
            reverse=True,
        )
        for i, row in enumerate(ranked[: args.show_examples]):
            path = list(row.get("oracle_path", []))
            print(f"\n[example {i}] len={len(path)} question={row.get('question')}")
            print("path[:80]=", path[:80])

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
