"""Post-process a DiPLaN predictions.jsonl into standard KGQA answer metrics.

DiPLaN's eval (evaluate_torch.py) reports path-level metrics only. This script
turns predicted *relation paths* into *answer entities* by executing them over
per-question subgraphs, then reports answer-entity Hits@1 / F1 -- the standard
KGQA metric reviewers expect. Fully offline; no Freebase server needed.

The oracle_path is executed alongside as a correctness gate. If oracle F1 is
low, the graph source is not suitable for answer-level claims. Use
``--graph_source rog`` for RoG's answer-containing graphs when EPR simple
subgraphs are too retrieval-pruned.

Usage:
    python scripts/evaluate_kgqa_answers.py \
        --predictions results/<run>/predictions.jsonl
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_jsonl
from src.diplan.kgqa_answer_eval import (
    answer_metrics,
    build_adjacency,
    execute_path,
    graph_sanity_stats,
    load_rog_subgraphs,
    load_webqsp_subgraphs,
    normalize_question,
)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _aggregate(rows: List[Dict], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_hits1": _mean([r[f"{prefix}_hits1"] for r in rows]),
        f"{prefix}_f1": _mean([r[f"{prefix}_f1"] for r in rows]),
        f"{prefix}_precision": _mean([r[f"{prefix}_precision"] for r in rows]),
        f"{prefix}_recall": _mean([r[f"{prefix}_recall"] for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument(
        "--graph_source",
        type=str,
        default="epr",
        choices=["epr", "rog"],
        help="Subgraph source used for path execution.",
    )
    parser.add_argument(
        "--webqsp_dir", type=str, default="data/raw/EPR-KGQA/data/dataset/WebQSP"
    )
    parser.add_argument(
        "--rog_path",
        type=str,
        default="data/rog/webqsp/test.jsonl",
        help="RoG JSONL file or directory with test/dev/val/train.jsonl.",
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default="webqsp",
        help="Comma-separated prediction dataset names to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--path_field",
        type=str,
        default="executed_path",
        choices=["executed_path", "planned_path"],
    )
    parser.add_argument("--out", type=str, default="")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Also follow reverse edges when executing paths (fallback for "
        "one-directional subgraphs). Decide via oracle-upper-bound F1.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_path = Path(args.out) if args.out else pred_path.parent / "answer_metrics.json"
    out_rows_path = pred_path.parent / "predictions_with_answers.jsonl"

    if args.graph_source == "rog":
        subgraphs = load_rog_subgraphs(args.rog_path)
        print(f"[answer-eval] indexed {len(subgraphs)} RoG subgraphs from {args.rog_path}")
    else:
        subgraphs = load_webqsp_subgraphs(args.webqsp_dir)
        print(f"[answer-eval] indexed {len(subgraphs)} EPR WebQSP subgraphs from {args.webqsp_dir}")
    sanity = graph_sanity_stats(subgraphs)
    print(
        "[answer-eval] graph sanity: "
        f"answer_in_graph={sanity['answer_entity_in_graph_rate']:.1%} "
        f"topic_in_graph={sanity['topic_entity_in_graph_rate']:.1%} "
        f"reachable<=3={sanity['q_to_answer_reachable_rate']:.1%}"
    )

    all_rows = load_jsonl(str(pred_path))
    if args.dataset_filter.strip().lower() == "all":
        eval_rows = all_rows
        dataset_names = ["all"]
    else:
        dataset_names = [x.strip().lower() for x in args.dataset_filter.split(",") if x.strip()]
        eval_rows = [r for r in all_rows if str(r.get("dataset", "")).lower() in dataset_names]
    print(f"[answer-eval] predictions={len(all_rows)} eval_rows={len(eval_rows)} datasets={dataset_names}")

    matched: List[Dict] = []
    unmatched = 0
    out_rows: List[Dict] = []
    for row in eval_rows:
        key = normalize_question(row.get("query", "") or row.get("question", ""))
        sg = subgraphs.get(key)
        if sg is None:
            unmatched += 1
            continue
        adj = build_adjacency(sg["tuples"], bidirectional=args.bidirectional)
        gold = sg["gold_answers"]
        topic = sg["topic_entities"]

        pred_ans = execute_path(topic, row.get(args.path_field, []) or [], adj)
        oracle_ans = execute_path(topic, row.get("oracle_path", []) or [], adj)
        m_pred = answer_metrics(pred_ans, gold)
        m_oracle = answer_metrics(oracle_ans, gold)

        rec = dict(row)
        rec["pred_answers"] = sorted(pred_ans)
        rec["gold_answers"] = sorted(gold)
        rec["ans_hits1"] = m_pred["hits1"]
        rec["ans_f1"] = m_pred["f1"]
        rec["ans_precision"] = m_pred["precision"]
        rec["ans_recall"] = m_pred["recall"]
        rec["oracle_ans_hits1"] = m_oracle["hits1"]
        rec["oracle_ans_f1"] = m_oracle["f1"]
        rec["oracle_ans_precision"] = m_oracle["precision"]
        rec["oracle_ans_recall"] = m_oracle["recall"]
        out_rows.append(rec)

        matched.append(
            {
                "pred_hits1": m_pred["hits1"],
                "pred_f1": m_pred["f1"],
                "pred_precision": m_pred["precision"],
                "pred_recall": m_pred["recall"],
                "oracle_hits1": m_oracle["hits1"],
                "oracle_f1": m_oracle["f1"],
                "oracle_precision": m_oracle["precision"],
                "oracle_recall": m_oracle["recall"],
            }
        )

    n_web = len(eval_rows)
    n_matched = len(matched)
    summary = {
        "predictions_path": str(pred_path),
        "graph_source": args.graph_source,
        "graph_path": args.rog_path if args.graph_source == "rog" else args.webqsp_dir,
        "dataset_filter": dataset_names,
        "path_field": args.path_field,
        "bidirectional": bool(args.bidirectional),
        "eval_total": n_web,
        "matched": n_matched,
        "unmatched": unmatched,
        "join_rate": (n_matched / n_web) if n_web else 0.0,
        "graph_sanity": sanity,
        "diplan": _aggregate(matched, "pred"),
        "oracle_upper_bound": _aggregate(matched, "oracle"),
    }

    ensure_dir(str(out_path.parent))
    dump_json(str(out_path), summary)
    dump_jsonl(str(out_rows_path), out_rows)

    d = summary["diplan"]
    o = summary["oracle_upper_bound"]
    print(
        f"[answer-eval] join_rate={summary['join_rate']:.1%} "
        f"(matched={n_matched}/{n_web}, unmatched={unmatched})"
    )
    print("[answer-eval] KGQA answer metrics (mean over matched questions):")
    print(f"  {'method':<22}{'Hits@1':>10}{'F1':>10}{'P':>10}{'R':>10}")
    print(
        f"  {'DiPLaN (' + args.path_field + ')':<22}"
        f"{d['pred_hits1']:>10.4f}{d['pred_f1']:>10.4f}"
        f"{d['pred_precision']:>10.4f}{d['pred_recall']:>10.4f}"
    )
    print(
        f"  {'oracle path (ceiling)':<22}"
        f"{o['oracle_hits1']:>10.4f}{o['oracle_f1']:>10.4f}"
        f"{o['oracle_precision']:>10.4f}{o['oracle_recall']:>10.4f}"
    )
    print(f"[answer-eval] wrote {out_path}")
    print(f"[answer-eval] wrote {out_rows_path}")


if __name__ == "__main__":
    main()
