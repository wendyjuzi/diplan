#!/usr/bin/env python3
"""Summarize patched-official-ToG runs into the main KGQA result table.

Expected layout under results_root:
  results_root/
    webqsp/{tog,pog,flare,diplan}/summary_metrics.json
    cwq/{tog,pog,flare,diplan}/summary_metrics.json

Optional RoG summaries can be provided manually because RoG is not run by this
repo's official-ToG scaffold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


METHOD_ORDER = ["RoG", "ToG", "PoG", "FLARE", "DiPLaN"]
METHOD_TO_DIR = {
    "ToG": "tog",
    "PoG": "pog",
    "FLARE": "flare",
    "DiPLaN": "diplan",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_load(path_str: str) -> Optional[Dict[str, Any]]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_json(path)


def _pick_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "hits@1" in raw:
        return raw
    if len(raw) == 1:
        return raw[next(iter(raw))]
    return raw


def _num(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    num = _num(value)
    if num is None:
        return ""
    return f"{num:.{digits}f}".rstrip("0").rstrip(".")


def _row(method: str, webqsp: Optional[Dict[str, Any]], cwq: Optional[Dict[str, Any]], diag_dataset: str = "webqsp") -> Dict[str, Any]:
    diag = webqsp if diag_dataset == "webqsp" else cwq
    sel = _num((diag or {}).get("answer_reaching_selected_rate"))
    exe = _num((diag or {}).get("answer_reaching_executed_top1_rate"))
    gap = (sel - exe) if sel is not None and exe is not None else None
    return {
        "Method": method,
        "WebQSP": _num((webqsp or {}).get("hits@1")),
        "CWQ": _num((cwq or {}).get("hits@1")),
        "Selection": sel,
        "Execution": exe,
        "Gap": gap,
        "Calls": _num((diag or {}).get("llm_calls_per_task")),
        "Time": _num((diag or {}).get("wall_time_s_per_task")),
    }


def _load_method_summary(results_root: Path, dataset: str, method: str) -> Optional[Dict[str, Any]]:
    subdir = METHOD_TO_DIR.get(method)
    if not subdir:
        return None
    path = results_root / dataset / subdir / "summary_metrics.json"
    if not path.exists():
        return None
    return _pick_summary(_load_json(path))


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    fields = ["Method", "WebQSP", "CWQ", "Selection", "Execution", "Gap", "Calls", "Time"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: list[Dict[str, Any]]) -> None:
    lines = [
        "| Method | WebQSP | CWQ | Selection↑ | Execution↑ | Gap↓ | Calls↓ | Time↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {Method} | {WebQSP} | {CWQ} | {Selection} | {Execution} | {Gap} | {Calls} | {Time} |".format(
                Method=row["Method"],
                WebQSP=_fmt(row["WebQSP"]),
                CWQ=_fmt(row["CWQ"]),
                Selection=_fmt(row["Selection"]),
                Execution=_fmt(row["Execution"]),
                Gap=_fmt(row["Gap"]),
                Calls=_fmt(row["Calls"], digits=2),
                Time=_fmt(row["Time"], digits=2),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--rog_webqsp_summary", default="")
    ap.add_argument("--rog_cwq_summary", default="")
    ap.add_argument("--diag_dataset", default="webqsp", choices=["webqsp", "cwq"])
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    rog_webqsp = _maybe_load(args.rog_webqsp_summary)
    rog_cwq = _maybe_load(args.rog_cwq_summary)

    rows: list[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method == "RoG":
            webqsp = _pick_summary(rog_webqsp) if rog_webqsp else None
            cwq = _pick_summary(rog_cwq) if rog_cwq else None
        else:
            webqsp = _load_method_summary(results_root, "webqsp", method)
            cwq = _load_method_summary(results_root, "cwq", method)
        rows.append(_row(method, webqsp, cwq, diag_dataset=args.diag_dataset))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")

    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "md": str(md_path),
                "json": str(json_path),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

