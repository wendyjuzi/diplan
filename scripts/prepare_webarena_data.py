import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir


WORD_RE = re.compile(r"[A-Za-z0-9_\.]+")


def _tokenize(text: str, max_len: int = 96) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall((text or "").lower()) if t]
    return toks[:max_len]


def _normalize_action(raw: str) -> str:
    txt = str(raw or "").strip().lower()
    txt = re.sub(r"\s+", "_", txt)
    txt = txt.replace("(", "_").replace(")", "")
    txt = txt.replace(",", "_").replace(":", "_")
    txt = txt.replace("[", "_").replace("]", "")
    txt = re.sub(r"[^a-z0-9_]+", "_", txt).strip("_")
    return txt or "noop"


def _stage_for_action(norm_action: str) -> str:
    a = norm_action.upper()
    if any(k in a for k in ["OBSERVE", "READ", "VIEW", "FIND", "CHECK", "EXTRACT", "INSPECT"]):
        return "OBSERVE"
    if any(k in a for k in ["NAV", "CLICK", "OPEN", "GOTO", "GO_TO", "BACK", "SCROLL", "VISIT"]):
        return "NAV"
    if any(k in a for k in ["TYPE", "INPUT", "FILL", "SELECT", "CHOOSE", "SEARCH", "ENTER"]):
        return "FORM"
    if any(k in a for k in ["SUBMIT", "CONFIRM", "STOP", "DONE", "FINISH", "ANSWER"]):
        return "VERIFY"
    return "ACT"


def _to_oracle_path(actions: List[str]) -> List[str]:
    out: List[str] = []
    for raw in actions:
        norm = _normalize_action(raw)
        st = _stage_for_action(norm)
        out.append(f"{st}::{norm}")
    return out


def _to_stage_path(actions: List[str], dedupe: bool) -> List[str]:
    out: List[str] = []
    last_stage = ""
    for raw in actions:
        norm = _normalize_action(raw)
        st = _stage_for_action(norm)
        if dedupe and st == last_stage:
            continue
        out.append(st)
        last_stage = st
    return out


def _macro_for_action(norm_action: str) -> str:
    a = norm_action.upper()
    if any(k in a for k in ["PASSWORD", "PASSWD", "USERNAME", "USER_NAME", "LOGIN", "SIGNIN", "SIGN_IN", "EMAIL"]):
        return "INPUT_CREDENTIALS"
    if any(k in a for k in ["SEARCH", "QUERY", "FILTER", "KEYWORD"]):
        return "QUERY_OR_FILTER"
    if any(k in a for k in ["SUBMIT", "CONFIRM", "CHECKOUT", "PLACE_ORDER", "SAVE", "SEND", "APPLY", "DONE", "FINISH"]):
        return "SUBMIT_OR_CONFIRM"
    if any(k in a for k in ["CLICK", "OPEN", "GOTO", "GO_TO", "VISIT", "BACK", "SCROLL", "NAV"]):
        return "NAVIGATE_TO_TARGET"
    if any(k in a for k in ["TYPE", "INPUT", "FILL", "SELECT", "CHOOSE", "CHECK", "UNCHECK", "PRESS", "ENTER"]):
        return "FILL_OR_SELECT_FIELD"
    if any(k in a for k in ["READ", "VIEW", "OBSERVE", "EXTRACT", "INSPECT"]):
        return "OBSERVE_PAGE"
    if any(k in a for k in ["UPLOAD", "DOWNLOAD", "ATTACH", "IMPORT", "EXPORT"]):
        return "FILE_OPERATION"
    return "GENERIC_ACTION"


def _to_macro_path(actions: List[str], dedupe: bool) -> List[str]:
    out: List[str] = []
    last = ""
    for raw in actions:
        norm = _normalize_action(raw)
        m = _macro_for_action(norm)
        if dedupe and m == last:
            continue
        out.append(m)
        last = m
    return out


def _build_trap_path(path: List[str], rng: random.Random) -> List[str]:
    trap = list(path)
    if not trap:
        return trap
    idx = rng.randrange(len(trap))
    tok = trap[idx]
    if "::" in tok:
        st, act = tok.split("::", 1)
        trap[idx] = f"{st}::{act}_wrong"
    else:
        trap[idx] = f"{tok}_wrong"
    return trap


def _build_constraints(path: List[str]) -> Dict[str, Any]:
    stages: List[str] = []
    seen = set()
    for tok in path:
        st = str(tok).split("::", 1)[0].strip().upper()
        if st and st not in seen:
            seen.add(st)
            stages.append(st)
    if len(stages) < 2:
        stages = ["OBSERVE", "NAV", "VERIFY"]
    return {
        "max_steps": min(96, max(6, len(path) + 2)),
        "required_stage_order": stages,
        "must_precede": [{"first": stages[i], "second": stages[i + 1]} for i in range(len(stages) - 1)],
        "required_before": {stages[i + 1]: [stages[i]] for i in range(len(stages) - 1)},
        "forbidden_actions": [],
    }


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


def _iter_input_files(inputs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for x in inputs:
        p = Path(x)
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
            out.append(p)
            continue
        if p.is_dir():
            out.extend(sorted(p.rglob("*.jsonl")))
            out.extend(sorted(p.rglob("*.json")))
    # de-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _extract_question(row: Dict[str, Any]) -> str:
    for k in ["instruction", "question", "task", "prompt", "goal", "intent", "query"]:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Some traces put intent under nested config/meta.
    for parent in ["meta", "config", "info"]:
        obj = row.get(parent)
        if not isinstance(obj, dict):
            continue
        for k in ["instruction", "task", "goal", "intent"]:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _extract_actions_from_list(xs: List[Any]) -> List[str]:
    out: List[str] = []
    for it in xs:
        if isinstance(it, str):
            if it.strip():
                out.append(it.strip())
            continue
        if not isinstance(it, dict):
            continue
        for k in ["parsed_action", "action", "raw_action", "action_text", "name", "op", "intent", "text"]:
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
                break
    return out


def _extract_actions(row: Dict[str, Any]) -> List[str]:
    # Common direct keys
    for k in [
        "gold_actions",
        "oracle_path",
        "action_path",
        "actions",
        "action_history",
        "parsed_actions",
        "trajectory",
        "steps",
    ]:
        v = row.get(k)
        if isinstance(v, list):
            xs = _extract_actions_from_list(v)
            if xs:
                return xs

    # Nested logs
    for parent in ["meta", "trace", "episode", "result", "data"]:
        obj = row.get(parent)
        if not isinstance(obj, dict):
            continue
        for k in ["actions", "action_history", "trajectory", "steps"]:
            v = obj.get(k)
            if isinstance(v, list):
                xs = _extract_actions_from_list(v)
                if xs:
                    return xs
    return []


def _is_success_row(row: Dict[str, Any]) -> bool:
    for k in ["success", "passed", "pass", "done", "is_success"]:
        v = row.get(k)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v) > 0
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in {"true", "pass", "passed", "success", "succeeded", "yes", "1"}:
                return True
            if vv in {"false", "fail", "failed", "no", "0"}:
                return False
    return True


def _split_rows(rows: List[Dict[str, Any]], seed: int, ratios: List[float]) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    data = list(rows)
    rng.shuffle(data)
    n = len(data)
    r_train, r_val, r_test = ratios
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    n_test = n - n_train - n_val
    if n > 0 and n_test <= 0:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
    return {
        "train": data[:n_train],
        "val": data[n_train : n_train + n_val],
        "test": data[n_train + n_val : n_train + n_val + n_test],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="WebArena trajectory files or directories.")
    parser.add_argument("--out", type=str, default="data/webarena_processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_success", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--per_file_limit", type=int, default=0)
    parser.add_argument(
        "--path_mode",
        type=str,
        default="action",
        choices=["action", "stage", "stage_dedup", "macro", "macro_dedup"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    files = _iter_input_files(args.inputs)
    if not files:
        raise FileNotFoundError("No json/jsonl files found under --inputs.")

    rng = random.Random(int(args.seed))
    mode = str(args.path_mode).lower().strip()
    all_rows: List[Dict[str, Any]] = []

    for fp in files:
        raw_rows = _load_json_or_jsonl(fp)
        if args.per_file_limit and args.per_file_limit > 0:
            raw_rows = raw_rows[: int(args.per_file_limit)]
        for i, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                continue
            if args.only_success and (not _is_success_row(row)):
                continue
            q = _extract_question(row)
            actions = _extract_actions(row)
            if not q or len(actions) < 2:
                continue
            if mode == "action":
                oracle_path = _to_oracle_path(actions)
            elif mode == "stage":
                oracle_path = _to_stage_path(actions, dedupe=False)
            elif mode == "stage_dedup":
                oracle_path = _to_stage_path(actions, dedupe=True)
            elif mode == "macro":
                oracle_path = _to_macro_path(actions, dedupe=False)
            else:
                oracle_path = _to_macro_path(actions, dedupe=True)
            if len(oracle_path) < 2:
                continue
            task_id = str(row.get("task_id") or row.get("id") or f"webarena::{fp.stem}::{i}")
            out_row = {
                "task_id": task_id,
                "dataset": "webarena",
                "question": q,
                "query_tokens": _tokenize(q, max_len=96),
                "oracle_path": oracle_path,
                "trap_path": _build_trap_path(oracle_path, rng),
                "constraints": _build_constraints(oracle_path),
                "meta": {
                    "source_file": str(fp),
                    "path_mode": mode,
                    "raw_action_len": len(actions),
                    "oracle_path_len": len(oracle_path),
                },
            }
            all_rows.append(out_row)
            if args.max_rows and args.max_rows > 0 and len(all_rows) >= int(args.max_rows):
                break
        if args.max_rows and args.max_rows > 0 and len(all_rows) >= int(args.max_rows):
            break

    if not all_rows:
        raise RuntimeError("No valid WebArena rows extracted. Check inputs and field formats.")

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    split = _split_rows(
        all_rows,
        seed=int(args.seed),
        ratios=[float(args.train_ratio), float(args.val_ratio), float(args.test_ratio)],
    )
    dump_jsonl(str(out_dir / "webarena_train.jsonl"), split["train"])
    dump_jsonl(str(out_dir / "webarena_val.jsonl"), split["val"])
    dump_jsonl(str(out_dir / "webarena_test.jsonl"), split["test"])
    dump_json(
        str(out_dir / "manifest.json"),
        {
            "dataset": "webarena",
            "seed": int(args.seed),
            "path_mode": mode,
            "inputs": [str(x) for x in files],
            "rows_total": len(all_rows),
            "counts": {
                "train": len(split["train"]),
                "val": len(split["val"]),
                "test": len(split["test"]),
            },
        },
    )
    print(
        f"[ok] webarena exported: total={len(all_rows)} "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )


if __name__ == "__main__":
    main()
