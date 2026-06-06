"""Build goal-focused ALFWorld plan targets.

The collected handcoded expert traces are successful but dominated by search.
After instance-free normalization, searching many concrete instances of the same
receptacle becomes repeated abstract tokens such as ``GOTO::shelf``. This script
distills each trace into the actions that directly advance task completion plus
the minimal local navigation/open actions observed before them.

This is not shortest-path planning: it never invents object locations beyond the
abstract receptacle already encoded in the successful action token, and it keeps
OPEN actions when the original trace opened the relevant receptacle before the
core action.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_alfworld_trajectories import _build_constraints, _candidate_metadata  # noqa: E402
from run_alfworld_diplan_agent import _parse_goal  # noqa: E402


CORE_STAGES = {"TAKE", "PUT", "HEAT", "COOL", "CLEAN", "USE", "EXAMINE"}
OBJECT_RECEP_STAGES = {"TAKE", "PUT", "HEAT", "COOL", "CLEAN"}


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


def _split(tok: str) -> Tuple[str, str]:
    tok = str(tok)
    if "::" not in tok:
        return tok.upper(), ""
    st, payload = tok.split("::", 1)
    return st.upper(), payload.lower()


def _stage(tok: str) -> str:
    return _split(tok)[0]


def _recep(tok: str) -> str:
    st, payload = _split(tok)
    parts = [p for p in payload.split("_") if p]
    if st in OBJECT_RECEP_STAGES and len(parts) >= 2:
        return parts[1]
    if st in {"GOTO", "OPEN", "CLOSE", "USE", "EXAMINE"} and parts:
        return parts[0]
    return ""


def _obj(tok: str) -> str:
    st, payload = _split(tok)
    parts = [p for p in payload.split("_") if p]
    if st in OBJECT_RECEP_STAGES and parts:
        return parts[0]
    return ""


def _append(out: List[str], tok: str) -> None:
    if tok and (not out or out[-1] != tok):
        out.append(tok)


def _nearest_previous_goto(path: List[str], idx: int) -> str:
    for j in range(idx - 1, -1, -1):
        if _stage(path[j]) == "GOTO":
            return path[j]
    return ""


def _opened_relevant_recep(path: List[str], start: int, idx: int, recep: str) -> bool:
    if not recep:
        return False
    for tok in path[max(0, start) : idx]:
        if _stage(tok) == "OPEN" and _recep(tok) == recep:
            return True
    return False


def _is_look_task(row: Dict[str, Any]) -> bool:
    meta = row.get("meta", {}) or {}
    transform = str(meta.get("transform", "") or "").lower()
    question = str(row.get("question", "") or "").lower()
    return transform == "look" or "look at" in question or "examine" in question


def _core_stages_for_row(row: Dict[str, Any]) -> set:
    stages = {"TAKE", "PUT", "HEAT", "COOL", "CLEAN"}
    if _is_look_task(row):
        stages.update({"USE", "EXAMINE"})
    return stages


def _goal_spec(row: Dict[str, Any]) -> Dict[str, Any]:
    spec = row.get("structured_goal", {}) or {}
    if spec:
        return spec
    return _parse_goal(str(row.get("question", "")))


def _is_target_core(tok: str, row: Dict[str, Any]) -> bool:
    st = _stage(tok)
    spec = _goal_spec(row)
    target_obj = str(spec.get("target_object", "") or "").lower().replace(" ", "")
    target_recep = str(spec.get("target_receptacle", "") or "").lower().replace(" ", "")
    is_look = _is_look_task(row)

    if st in {"TAKE", "PUT", "HEAT", "COOL", "CLEAN"}:
        obj = _obj(tok)
        recep = _recep(tok)
        if target_obj and obj != target_obj:
            return False
        if st == "PUT" and target_recep and recep != target_recep and not is_look:
            return False
        return True

    if st in {"USE", "EXAMINE"}:
        return is_look
    return False


def goal_focused_path(row: Dict[str, Any], keep_initial_look: bool = False) -> List[str]:
    path = list(row.get("oracle_path", []))
    core_stages = _core_stages_for_row(row)
    core_indices = [
        i for i, tok in enumerate(path)
        if _stage(tok) in core_stages and _is_target_core(tok, row)
    ]
    if not core_indices:
        return list(path)

    out: List[str] = []
    if keep_initial_look and path and _stage(path[0]) == "OTHER":
        _append(out, path[0])

    last_core = 0
    for idx in core_indices:
        tok = path[idx]
        st = _stage(tok)
        recep = _recep(tok)

        if st in OBJECT_RECEP_STAGES and recep:
            _append(out, f"GOTO::{recep}")
            if _opened_relevant_recep(path, last_core, idx, recep):
                _append(out, f"OPEN::{recep}")
        elif st in {"USE", "EXAMINE"}:
            # USE::desklamp does not encode the surface holding the lamp, so
            # preserve the nearest previous navigation from the original trace.
            prev_goto = _nearest_previous_goto(path, idx)
            if prev_goto:
                _append(out, prev_goto)

        _append(out, tok)
        last_core = idx + 1

    return out


def _focus_row(row: Dict[str, Any], keep_initial_look: bool) -> Tuple[Dict[str, Any], Dict[str, float]]:
    old_path = list(row.get("oracle_path", []))
    new_path = goal_focused_path(row, keep_initial_look=keep_initial_look)
    new_row = dict(row)
    new_row["oracle_path"] = new_path
    # Raw commands/state prefixes are no longer position-aligned after
    # goal-focused distillation, so keep them only as provenance.
    new_row["raw_oracle_path_full"] = row.get("raw_oracle_path", [])
    new_row["state_query_tokens_by_prefix_full"] = row.get("state_query_tokens_by_prefix", [])
    if row.get("state_query_tokens_by_prefix"):
        # Use the initial state for the distilled plan; receding-horizon state
        # prefixes are regenerated online at evaluation time.
        new_row["state_query_tokens_by_prefix"] = [row["state_query_tokens_by_prefix"][0] for _ in new_path]
    meta = dict(row.get("meta", {}) or {})
    meta.update(
        {
            "original_num_steps": len(old_path),
            "num_steps": len(new_path),
            "goal_focused": True,
            "keep_initial_look": bool(keep_initial_look),
        }
    )
    new_row["meta"] = meta
    spec = row.get("structured_goal", {}) or {}
    new_row["constraints"] = _build_constraints(new_path, spec)
    cand_meta = _candidate_metadata(new_path)
    new_row["candidate_metadata"] = cand_meta
    new_row["candidate_paths"] = [m["path"] for m in cand_meta if not m.get("is_oracle", False)]
    return new_row, {
        "old_len": float(len(old_path)),
        "new_len": float(len(new_path)),
        "removed": float(len(old_path) - len(new_path)),
    }


def _process_split(src: Path, dst: Path, args) -> Dict[str, float]:
    rows = _load_jsonl(src)
    out = []
    stats = []
    for row in rows:
        new_row, st = _focus_row(row, keep_initial_look=bool(args.keep_initial_look))
        if new_row.get("oracle_path"):
            out.append(new_row)
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
    parser.add_argument("--keep_initial_look", action="store_true")
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
    (out_dir / "goal_focused_manifest.json").write_text(
        json.dumps(
            {
                "source": str(in_dir),
                "keep_initial_look": bool(args.keep_initial_look),
                "splits": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] wrote goal-focused data to {out_dir}")


if __name__ == "__main__":
    main()
