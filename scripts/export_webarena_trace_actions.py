import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _shorten(s: str, n: int = 96) -> str:
    s = _safe_text(s)
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _action_from_before(obj: Dict[str, Any]) -> str:
    api = _safe_text(obj.get("apiName", "")).lower()
    params = obj.get("params", {}) or {}
    if not isinstance(params, dict):
        params = {}

    selector = _shorten(params.get("selector", ""))
    key = _shorten(params.get("key", ""))
    url = _shorten(params.get("url", ""))
    text = _shorten(params.get("text", ""))

    if api.endswith("click"):
        return f"CLICK::{selector or 'unknown'}"
    if api.endswith("dblclick"):
        return f"DBLCLICK::{selector or 'unknown'}"
    if api.endswith("check"):
        return f"CHECK::{selector or 'unknown'}"
    if api.endswith("uncheck"):
        return f"UNCHECK::{selector or 'unknown'}"
    if api.endswith("selectoption"):
        val = params.get("values") or params.get("value") or ""
        return f"SELECT::{_shorten(val) or selector or 'unknown'}"
    if api.endswith("fill"):
        return f"FILL::{selector or 'unknown'}::{_shorten(text) or '<text>'}"
    if api.endswith("type"):
        return f"TYPE::{selector or 'unknown'}::{_shorten(text) or '<text>'}"
    if api.endswith("press"):
        return f"PRESS::{selector or 'page'}::{key or 'unknown'}"
    if api.endswith("goto"):
        return f"GOTO::{url or 'unknown'}"
    if api.endswith("goback"):
        return "BACK"
    if api.endswith("reload"):
        return "RELOAD"
    return f"ACT::{api or 'unknown'}"


def _find_trace_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for x in inputs:
        p = Path(x)
        if not p.exists():
            continue
        if p.is_file() and p.name == "trace.trace":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob("trace.trace")))
    # de-dup
    seen = set()
    out: List[Path] = []
    for p in files:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _parse_trace(trace_file: Path) -> Dict[str, Any]:
    actions: List[str] = []
    current_url = ""
    first_url = ""
    with trace_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type")
            if t == "before":
                act = _action_from_before(obj)
                if act:
                    if not actions or actions[-1] != act:
                        actions.append(act)
                continue
            if obj.get("class") == "Frame" and obj.get("method") == "navigated":
                u = _safe_text((obj.get("params") or {}).get("url", ""))
                if u:
                    current_url = u
                    if not first_url:
                        first_url = u
    tid = trace_file.parent.name
    question = f"Complete WebArena task from trace {tid}"
    if first_url:
        question += f" starting at {first_url}"
    row = {
        "task_id": f"webarena::{tid}",
        "instruction": question,
        "actions": actions,
        "success": True,
        "meta": {
            "trace_dir": str(trace_file.parent),
            "first_url": first_url,
            "final_url": current_url,
            "raw_action_len": len(actions),
        },
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Directories or trace.trace files.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--min_actions", type=int, default=2)
    parser.add_argument("--max_traces", type=int, default=0, help="0 means all.")
    parser.add_argument(
        "--augment_prefixes",
        action="store_true",
        help="Create additional samples by slicing each real trajectory into prefixes.",
    )
    parser.add_argument("--prefix_min_len", type=int, default=4)
    parser.add_argument("--prefix_max_per_trace", type=int, default=12)
    args = parser.parse_args()

    trace_files = _find_trace_files(args.inputs)
    if args.max_traces and args.max_traces > 0:
        trace_files = trace_files[: args.max_traces]

    rows: List[Dict[str, Any]] = []
    for tf in trace_files:
        row = _parse_trace(tf)
        acts = row.get("actions", [])
        if len(acts) < args.min_actions:
            continue
        rows.append(row)
        if args.augment_prefixes:
            lo = max(args.prefix_min_len, args.min_actions)
            if len(acts) >= lo:
                made = 0
                for end in range(lo, len(acts) + 1):
                    if made >= max(0, args.prefix_max_per_trace):
                        break
                    sub = dict(row)
                    sub_actions = acts[:end]
                    sub["task_id"] = f"{row['task_id']}::p{end}"
                    sub["instruction"] = f"{row['instruction']} [prefix_len={end}]"
                    sub["actions"] = sub_actions
                    m = dict(row.get("meta", {}))
                    m["is_prefix_augmented"] = True
                    m["prefix_len"] = end
                    sub["meta"] = m
                    rows.append(sub)
                    made += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] exported rows={len(rows)} from trace_files={len(trace_files)} -> {out_path}")


if __name__ == "__main__":
    main()
