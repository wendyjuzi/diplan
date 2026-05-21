import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.io_utils import ensure_dir, load_config, load_jsonl
from src.diplan.torch_pipeline import (
    PAD,
    ValueDataset,
    ValueRanker,
    collate_value,
    load_vocab,
    pad_2d,
    set_seed,
)


def _build_pairwise_samples(
    rows,
    path_vocab,
    query_vocab,
    max_path_len: int,
    max_query_len: int,
    neg_per_pos: int,
    seed: int,
):
    rng = random.Random(seed)
    all_rel = sorted(list({r for row in rows for r in row.get("oracle_path", []) if isinstance(r, str)}))
    samples = []
    for row in rows:
        q = query_vocab.encode(row.get("query_tokens", []), add_bos_eos=False, max_len=max_query_len)
        pos_tokens = row.get("oracle_path", [])
        pos = path_vocab.encode(pos_tokens, add_bos_eos=False, max_len=max_path_len)
        if not q or not pos:
            continue
        for _ in range(max(1, neg_per_pos)):
            neg_tokens = list(pos_tokens)
            if neg_tokens and all_rel:
                j = rng.randrange(len(neg_tokens))
                old = neg_tokens[j]
                repl = rng.choice(all_rel)
                if len(all_rel) > 1 and repl == old:
                    repl = all_rel[(all_rel.index(repl) + 1) % len(all_rel)]
                neg_tokens[j] = repl
            neg = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
            if not neg:
                neg = [path_vocab.stoi[PAD]]
            samples.append((q, pos, neg))
    return samples


def _collate_pairwise(batch, q_pad: int, p_pad: int):
    q = [x[0] for x in batch]
    pos = [x[1] for x in batch]
    neg = [x[2] for x in batch]
    q_ids, _ = pad_2d(q, q_pad)
    pos_ids, _ = pad_2d(pos, p_pad)
    neg_ids, _ = pad_2d(neg, p_pad)
    return q_ids, pos_ids, neg_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/value_torch_kgqa.yaml")
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/value_kgqa_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rows = load_jsonl(cfg["train_path"])
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    path_vocab = load_vocab(planner_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])
    training_mode = str(cfg.get("training_mode", "bce")).lower()
    if training_mode not in {"bce", "pairwise"}:
        raise ValueError(f"Unsupported training_mode={training_mode}. Expected bce|pairwise.")

    if training_mode == "bce":
        dataset = ValueDataset(
            rows=rows,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            max_path_len=int(cfg.get("max_path_len", 8)),
            max_query_len=int(cfg.get("max_query_len", 24)),
            neg_per_pos=int(cfg.get("neg_per_pos", 2)),
            seed=int(cfg.get("seed", 42)),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.get("batch_size", 128)),
            shuffle=True,
            collate_fn=lambda b: collate_value(b, query_vocab.stoi[PAD], path_vocab.stoi[PAD]),
        )
        n_samples = len(dataset)
    else:
        dataset = _build_pairwise_samples(
            rows=rows,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            max_path_len=int(cfg.get("max_path_len", 8)),
            max_query_len=int(cfg.get("max_query_len", 24)),
            neg_per_pos=int(cfg.get("neg_per_pos", 2)),
            seed=int(cfg.get("seed", 42)),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.get("batch_size", 128)),
            shuffle=True,
            collate_fn=lambda b: _collate_pairwise(b, query_vocab.stoi[PAD], path_vocab.stoi[PAD]),
        )
        n_samples = len(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")
    model = ValueRanker(
        q_vocab_size=len(query_vocab.itos),
        p_vocab_size=len(path_vocab.itos),
        emb_dim=int(cfg.get("emb_dim", 128)),
        q_pad_id=query_vocab.stoi[PAD],
        p_pad_id=path_vocab.stoi[PAD],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))

    best_loss = float("inf")
    best_state = None
    epochs = int(cfg.get("epochs", 8))
    ranking_margin = float(cfg.get("ranking_margin", 0.2))
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        total_gap = 0.0
        count = 0
        for batch in loader:
            if training_mode == "bce":
                q_ids, p_ids, y = batch
                q_ids = q_ids.to(device)
                p_ids = p_ids.to(device)
                y = y.to(device)
                logits = model(q_ids, p_ids)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                gap = torch.tensor(0.0, device=device)
            else:
                q_ids, pos_ids, neg_ids = batch
                q_ids = q_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)
                s_pos = model(q_ids, pos_ids)
                s_neg = model(q_ids, neg_ids)
                target = torch.ones_like(s_pos)
                loss = F.margin_ranking_loss(s_pos, s_neg, target, margin=ranking_margin)
                gap = (s_pos - s_neg).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            total_gap += float(gap.item())
            count += 1
        avg = total / max(1, count)
        avg_gap = total_gap / max(1, count)
        if training_mode == "pairwise":
            print(f"[value] epoch {ep}/{epochs} loss={avg:.4f} pair_gap={avg_gap:.4f}")
        else:
            print(f"[value] epoch {ep}/{epochs} loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ensure_dir(args.out)
    ckpt = {
        "model_state": best_state,
        "model_config": {
            "q_vocab_size": len(query_vocab.itos),
            "p_vocab_size": len(path_vocab.itos),
            "emb_dim": int(cfg.get("emb_dim", 128)),
            "q_pad_id": query_vocab.stoi[PAD],
            "p_pad_id": path_vocab.stoi[PAD],
            "training_mode": training_mode,
            "ranking_margin": ranking_margin,
        },
        "query_vocab": planner_ckpt["query_vocab"],
        "path_vocab": planner_ckpt["path_vocab"],
        "train_info": {"samples": n_samples, "best_loss": best_loss, "training_mode": training_mode},
    }
    out_path = Path(args.out) / "best.pt"
    torch.save(ckpt, out_path)
    print(f"Saved torch value model to {out_path}")


if __name__ == "__main__":
    main()
