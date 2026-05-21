import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_jsonl, load_jsonl
from src.diplan.torch_pipeline import BOS, EOS, PAD, PathAutoencoder, load_vocab


def _lcp_len(a: List[str], b: List[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _token_f1(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    from collections import Counter

    ca = Counter(a)
    cb = Counter(b)
    inter = sum((ca & cb).values())
    p = inter / max(1, len(a))
    r = inter / max(1, len(b))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="jsonl with oracle_path field")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--show_cases", type=int, default=10)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    cfg = ckpt["model_config"]
    path_vocab = load_vocab(ckpt["path_vocab"])
    rows = load_jsonl(args.data)
    rows = [r for r in rows if isinstance(r.get("oracle_path"), list) and len(r["oracle_path"]) > 0]
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    rows = rows[: min(args.n, len(rows))]

    model = PathAutoencoder(
        vocab_size=cfg["vocab_size"],
        emb_dim=cfg["emb_dim"],
        hid_dim=cfg["hid_dim"],
        latent_dim=cfg["latent_dim"],
        max_path_len=cfg["max_path_len"],
        pad_id=cfg["pad_id"],
        latent_noise_std=float(cfg.get("latent_noise_std", 0.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bos_id = path_vocab.stoi[BOS]
    eos_id = path_vocab.stoi[EOS]
    pad_id = path_vocab.stoi[PAD]
    max_len = int(cfg["max_path_len"])

    results = []
    strict_match = 0
    repeat_all = 0
    lcp_ratio_sum = 0.0
    f1_sum = 0.0

    for row in rows:
        oracle = row["oracle_path"]
        ids = path_vocab.encode(oracle, add_bos_eos=True, max_len=max_len)
        x = torch.tensor([ids], dtype=torch.long)
        lens = torch.tensor([len(ids)], dtype=torch.long)

        z = model.encode(x, lens)
        seq_ids, pred_lens = model.decode_greedy(z, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
        pred_ids = seq_ids[0][: max(1, min(pred_lens[0], max_len))]
        pred = path_vocab.decode(pred_ids, skip_special=True)

        is_match = pred == oracle
        strict_match += int(is_match)
        if len(pred) > 1 and len(set(pred)) == 1:
            repeat_all += 1
        lcp = _lcp_len(pred, oracle)
        lcp_ratio = lcp / max(1, len(oracle))
        lcp_ratio_sum += lcp_ratio
        tf1 = _token_f1(pred, oracle)
        f1_sum += tf1

        results.append(
            {
                "task_id": row.get("task_id", ""),
                "query": row.get("question", " ".join(row.get("query_tokens", []))),
                "oracle_path": oracle,
                "reconstructed_path": pred,
                "strict_match": is_match,
                "lcp_len": lcp,
                "lcp_ratio": lcp_ratio,
                "token_f1": tf1,
            }
        )

    n = max(1, len(results))
    summary = {
        "n_eval": len(results),
        "strict_reconstruction_acc": strict_match / n,
        "repeat_all_same_relation_rate": repeat_all / n,
        "avg_lcp_ratio": lcp_ratio_sum / n,
        "avg_token_f1": f1_sum / n,
    }

    print("[ae-isolation] summary:", json.dumps(summary, ensure_ascii=False))
    print("-" * 80)
    show = min(args.show_cases, len(results))
    for i in range(show):
        r = results[i]
        print(f"CASE {i+1}")
        print("query:", r["query"])
        print("oracle:", r["oracle_path"])
        print("recon :", r["reconstructed_path"])
        print("strict_match:", r["strict_match"], "lcp_ratio:", round(r["lcp_ratio"], 3), "token_f1:", round(r["token_f1"], 3))
        print("-" * 80)

    if args.out:
        out_path = Path(args.out)
        dump_jsonl(str(out_path), results)
        print(f"[ae-isolation] saved cases -> {out_path}")


if __name__ == "__main__":
    main()
