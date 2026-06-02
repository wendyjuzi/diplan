import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_jsonl
from src.diplan.torch_pipeline import PAD, ValueRanker, collate_value, load_vocab


def _dedupe_paths(paths: List[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    seen = set()
    for p in paths:
        key = tuple(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _score_with_value_model(
    value_model: ValueRanker,
    query_tokens: List[str],
    candidates: List[List[str]],
    query_vocab,
    path_vocab,
    device: torch.device,
    max_query_len: int = 96,
    max_path_len: int = 96,
) -> List[float]:
    rows = []
    q_ids = query_vocab.encode(query_tokens, add_bos_eos=False, max_len=max_query_len)
    for c in candidates:
        p_ids = path_vocab.encode(c, add_bos_eos=False, max_len=max_path_len)
        rows.append((q_ids, p_ids, 0.0))
    q_pad = query_vocab.stoi[PAD]
    p_pad = path_vocab.stoi[PAD]
    q_t, p_t, _ = collate_value(rows, q_pad, p_pad)
    with torch.no_grad():
        logits = value_model(q_t.to(device), p_t.to(device)).detach().cpu().tolist()
    return [float(x) for x in logits]


def _binom_two_sided_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2.0 * prob)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_no_value", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True, help="Needed for query vocab.")
    parser.add_argument("--ae_ckpt", type=str, required=True, help="Needed for path vocab.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    preds = load_jsonl(args.predictions_no_value)
    if not preds:
        raise RuntimeError("No rows found in predictions file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    value_ckpt = torch.load(args.value_ckpt, map_location="cpu")
    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])

    v_cfg = value_ckpt["model_config"]
    value_model = ValueRanker(
        q_vocab_size=v_cfg["q_vocab_size"],
        p_vocab_size=v_cfg["p_vocab_size"],
        emb_dim=v_cfg["emb_dim"],
        q_pad_id=v_cfg["q_pad_id"],
        p_pad_id=v_cfg["p_pad_id"],
        architecture=str(v_cfg.get("architecture", "legacy")),
        hidden_dim=int(v_cfg.get("hidden_dim", 256)),
        dropout=float(v_cfg.get("dropout", 0.1)),
    ).to(device)
    value_model.load_state_dict(value_ckpt["model_state"])
    value_model.eval()

    rows_out: List[Dict] = []
    n = 0
    pool_hit = 0
    no_value_success = 0
    with_value_success = 0
    b_no_yes = 0  # no-value wrong, with-value correct
    c_yes_no = 0  # no-value correct, with-value wrong

    for r in preds:
        cands_raw = [x.get("path", []) for x in r.get("candidate_pool_top", []) if isinstance(x, dict)]
        cands = _dedupe_paths([c for c in cands_raw if isinstance(c, list) and c])
        if not cands:
            continue
        n += 1
        oracle = r.get("oracle_path", [])
        oracle_key = tuple(oracle)
        pool_has_oracle = oracle_key in {tuple(c) for c in cands}
        if pool_has_oracle:
            pool_hit += 1

        # no-value baseline uses stage1_score order if available, otherwise first candidate order.
        stage_scored = []
        for x in r.get("candidate_pool_top", []):
            if not isinstance(x, dict):
                continue
            p = x.get("path", [])
            if not isinstance(p, list) or not p:
                continue
            s = float(x.get("stage1_score", 0.0))
            stage_scored.append((p, s))
        if stage_scored:
            stage_scored.sort(key=lambda t: t[1], reverse=True)
            no_value_top1 = stage_scored[0][0]
        else:
            no_value_top1 = cands[0]

        value_scores = _score_with_value_model(
            value_model=value_model,
            query_tokens=r.get("query_tokens", []),
            candidates=cands,
            query_vocab=query_vocab,
            path_vocab=path_vocab,
            device=device,
        )
        best_idx = max(range(len(cands)), key=lambda i: value_scores[i])
        with_value_top1 = cands[best_idx]

        no_ok = tuple(no_value_top1) == oracle_key
        with_ok = tuple(with_value_top1) == oracle_key
        no_value_success += 1 if no_ok else 0
        with_value_success += 1 if with_ok else 0
        if (not no_ok) and with_ok:
            b_no_yes += 1
        if no_ok and (not with_ok):
            c_yes_no += 1

        rows_out.append(
            {
                "task_id": r.get("task_id"),
                "dataset": r.get("dataset", "unknown"),
                "oracle_in_pool": pool_has_oracle,
                "oracle_path": oracle,
                "no_value_top1": no_value_top1,
                "with_value_top1": with_value_top1,
                "no_value_success": no_ok,
                "with_value_success": with_ok,
                "candidate_pool_size_used": len(cands),
            }
        )

    if n == 0:
        raise RuntimeError("No usable candidate pools found. Ensure predictions include candidate_pool_top.")

    summary = {
        "num_tasks_used": n,
        "candidate_pool_hit_rate_same_pool": pool_hit / max(1, n),
        "no_value_top1_success_rate_same_pool": no_value_success / max(1, n),
        "with_value_top1_success_rate_same_pool": with_value_success / max(1, n),
        "delta_with_minus_no_value": (with_value_success - no_value_success) / max(1, n),
        "conditional_no_value_success_given_pool_hit": (
            sum(1 for x in rows_out if x["oracle_in_pool"] and x["no_value_success"]) / max(1, pool_hit)
        ),
        "conditional_with_value_success_given_pool_hit": (
            sum(1 for x in rows_out if x["oracle_in_pool"] and x["with_value_success"]) / max(1, pool_hit)
        ),
        "mcnemar_b_no_yes": b_no_yes,
        "mcnemar_c_yes_no": c_yes_no,
        "mcnemar_p_two_sided": _binom_two_sided_p_value(b_no_yes, c_yes_no),
    }

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    dump_jsonl = Path(args.out) / "same_pool_comparison_rows.jsonl"
    with dump_jsonl.open("w", encoding="utf-8") as f:
        for x in rows_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    dump_json(str(Path(args.out) / "same_pool_comparison_summary.json"), summary)
    print(summary)
    print(f"[ok] wrote {dump_jsonl} and {Path(args.out) / 'same_pool_comparison_summary.json'}")


if __name__ == "__main__":
    main()
