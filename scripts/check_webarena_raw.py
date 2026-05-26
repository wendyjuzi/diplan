import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in ["rows", "data", "items", "episodes", "results", "trajectories"]:
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [obj]
    return []


def _extract_actions_count(row: Dict[str, Any]) -> int:
    for k in ["gold_actions", "oracle_path", "action_path", "actions", "action_history", "parsed_actions", "trajectory", "steps"]:
        v = row.get(k)
        if isinstance(v, list):
            return len(v)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    args = parser.parse_args()

    files: List[Path] = []
    for x in args.inputs:
        p = Path(x)
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
            files.extend(sorted(p.rglob("*.json")))

    if not files:
        print("[err] no json/jsonl files found")
        return

    total_rows = 0
    valid_rows = 0
    for fp in files:
        try:
            rows = _load_json_or_jsonl(fp)
        except Exception:
            continue
        total_rows += len(rows)
        for r in rows:
            q = r.get("instruction") or r.get("question") or r.get("task") or r.get("prompt") or r.get("goal")
            n_act = _extract_actions_count(r)
            if isinstance(q, str) and q.strip() and n_act >= 2:
                valid_rows += 1

    print(f"[ok] files={len(files)} total_rows={total_rows} valid_rows={valid_rows}")
    print("tip: valid_rows should be > 0 before running prepare_webarena_data.py")


if __name__ == "__main__":
    main()

