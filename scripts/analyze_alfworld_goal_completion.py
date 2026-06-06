"""Analyze where ALFWorld expert trajectories reach the goal-critical action.

This answers whether successful expert trajectories contain substantial action
mass after the goal-completing step. It does not modify data.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


TRANSFORM_STAGES = {"heat": "HEAT", "cool": "COOL", "clean": "CLEAN"}


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
        return tok.split("::", 1)[0].upper()
    return tok.split("_", 1)[0].upper() if tok else "OTHER"


def _payload(tok: str) -> str:
    tok = str(tok)
    return tok.split("::", 1)[1].lower() if "::" in tok else ""


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return float(xs[idx])


def _goal_terms(row: Dict[str, Any]) -> Tuple[str, str, str, bool]:
    meta = row.get("meta", {}) or {}
    transform = str(meta.get("transform", "") or "").lower()
    question = str(row.get("question", "")).lower()
    # Prefer structured fields if rows came from the executor; collector rows
    # usually only have question/meta.
    spec = row.get("structured_goal", {}) or {}
    obj = str(spec.get("target_object", "") or "").lower().replace(" ", "")
    recep = str(spec.get("target_receptacle", "") or "").lower().replace(" ", "")
    requires_put = bool(spec.get("requires_put", False))
    if not requires_put:
        requires_put = any(x in question for x in (" put ", " in ", " on "))
    return transform, obj, recep, requires_put


def completion_index(row: Dict[str, Any]) -> Tuple[int, str]:
    path = list(row.get("oracle_path", []))
    if not path:
        return -1, "empty"
    transform, obj, recep, requires_put = _goal_terms(row)

    # Put tasks, including heat/cool/clean-and-put, complete at the final PUT.
    put_positions = [i for i, tok in enumerate(path) if _stage(tok) == "PUT"]
    if requires_put and put_positions:
        return put_positions[-1], "put"

    # Pure transform tasks complete at the corresponding transform action.
    if transform in TRANSFORM_STAGES:
        target_stage = TRANSFORM_STAGES[transform]
        pos = [i for i, tok in enumerate(path) if _stage(tok) == target_stage]
        if pos:
            return pos[-1], transform

    # Look/examine tasks complete at USE/EXAMINE. USE desklamp is how ALFWorld
    # handcoded plans often finish "look/examine under lamp" tasks.
    for stage in ("USE", "EXAMINE"):
        pos = [i for i, tok in enumerate(path) if _stage(tok) == stage]
        if pos:
            return pos[-1], stage.lower()

    # Fallback: if there is a final goal-like stage, use it; otherwise say
    # unknown rather than silently truncating to the full path.
    for stage in ("PUT", "HEAT", "COOL", "CLEAN", "TAKE"):
        pos = [i for i, tok in enumerate(path) if _stage(tok) == stage]
        if pos:
            return pos[-1], f"fallback_{stage.lower()}"
    return -1, "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--show_examples", type=int, default=5)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.path))
    ratios = []
    after_counts = []
    lengths = []
    reason_counts = Counter()
    by_reason = defaultdict(list)
    examples = []
    missing = 0
    for row in rows:
        path = list(row.get("oracle_path", []))
        n = len(path)
        if not n:
            continue
        idx, reason = completion_index(row)
        reason_counts[reason] += 1
        lengths.append(float(n))
        if idx < 0:
            missing += 1
            continue
        ratio = (idx + 1) / max(1, n)
        after = n - idx - 1
        ratios.append(float(ratio))
        after_counts.append(float(after))
        by_reason[reason].append(float(ratio))
        if after > 0:
            examples.append((after, row.get("question", ""), idx, path))

    summary = {
        "n": len(rows),
        "missing_completion": missing,
        "trajectory_length_mean": statistics.mean(lengths) if lengths else 0.0,
        "completion_ratio_mean": statistics.mean(ratios) if ratios else 0.0,
        "completion_ratio_p50": _quantile(ratios, 0.50),
        "completion_ratio_p90": _quantile(ratios, 0.90),
        "post_completion_steps_mean": statistics.mean(after_counts) if after_counts else 0.0,
        "post_completion_steps_p90": _quantile(after_counts, 0.90),
        "has_post_completion_rate": sum(1 for x in after_counts if x > 0) / max(1, len(after_counts)),
        "reason_counts": dict(reason_counts),
        "reason_completion_ratio_mean": {
            k: statistics.mean(v) for k, v in sorted(by_reason.items()) if v
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    examples.sort(reverse=True, key=lambda x: x[0])
    for i, (after, question, idx, path) in enumerate(examples[: max(0, int(args.show_examples))]):
        print(f"\n[example {i}] after={after} completion_step={idx + 1}/{len(path)} question={question}")
        print("before+completion=", path[: idx + 1])
        print("after=", path[idx + 1 :])

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
