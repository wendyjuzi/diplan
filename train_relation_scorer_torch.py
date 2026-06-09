"""Train a question-conditioned KG relation retriever with InfoNCE."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.diplan.io_utils import load_jsonl
from src.diplan.relation_scorer import (
    QueryRelationScorer,
    build_relation_vocabs,
    conditioned_query_tokens,
    question_text_tokens,
    relation_text_tokens,
)
from src.diplan.torch_pipeline import PAD, UNK, pad_2d, save_vocab, set_seed


def _relation_token_set(rel: str):
    return set(relation_text_tokens(rel))


def _hard_negative(positive: str, graph_relations):
    pos_tokens = _relation_token_set(positive)
    pos_domain = positive.split(".", 1)[0]

    def key(rel):
        toks = _relation_token_set(rel)
        union = len(pos_tokens | toks)
        sim = len(pos_tokens & toks) / union if union else 0.0
        same_domain = 1 if rel.split(".", 1)[0] == pos_domain else 0
        return same_domain, sim, rel

    negatives = [r for r in graph_relations if r != positive]
    return max(negatives, key=key) if negatives else ""


def build_samples(rows, q_vocab, r_vocab, max_query_len, max_relation_len):
    samples = []
    for row in rows:
        q_tokens = list(row.get("query_tokens") or question_text_tokens(str(row.get("question", ""))))
        graph_relations = sorted({
            str(t[1])
            for t in row.get("graph", []) or []
            if isinstance(t, (list, tuple)) and len(t) >= 2
        })
        oracle = list(row.get("oracle_path", []) or [])
        for step, rel in enumerate(oracle):
            conditioned = conditioned_query_tokens(q_tokens, oracle[:step])
            q = q_vocab.encode(conditioned, max_len=max_query_len) or [q_vocab.stoi[UNK]]
            rel = str(rel)
            r = r_vocab.encode(relation_text_tokens(rel), max_len=max_relation_len)
            neg_rel = _hard_negative(rel, graph_relations)
            neg = r_vocab.encode(relation_text_tokens(neg_rel), max_len=max_relation_len) if neg_rel else []
            if r:
                samples.append((q, r, neg or [r_vocab.stoi[UNK]], rel))
    return samples


def collate(batch, q_pad, r_pad):
    q, r, neg, relation_keys = zip(*batch)
    q_ids, _ = pad_2d(list(q), q_pad)
    r_ids, _ = pad_2d(list(r), r_pad)
    neg_ids, _ = pad_2d(list(neg), r_pad)
    return q_ids, r_ids, neg_ids, list(relation_keys)


def multi_positive_nce(logits: torch.Tensor, relation_keys, symmetric: bool = True) -> torch.Tensor:
    """InfoNCE where examples sharing the same gold relation are positives."""
    device = logits.device
    positive = torch.tensor(
        [[a == b for b in relation_keys] for a in relation_keys],
        dtype=torch.bool,
        device=device,
    )

    def directional(x, mask):
        log_den = torch.logsumexp(x, dim=1)
        log_num = torch.logsumexp(x.masked_fill(~mask, -1e9), dim=1)
        return -(log_num - log_den).mean()

    loss = directional(logits, positive)
    if symmetric:
        loss = 0.5 * (loss + directional(logits.t(), positive.t()))
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--valid_path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--margin_weight", type=float, default=0.5)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_query_len", type=int, default=48)
    ap.add_argument("--max_relation_len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    train_rows = load_jsonl(args.train_path)
    valid_rows = load_jsonl(args.valid_path) if args.valid_path else []
    q_vocab, r_vocab = build_relation_vocabs(train_rows)
    train_samples = build_samples(
        train_rows, q_vocab, r_vocab, args.max_query_len, args.max_relation_len
    )
    valid_samples = build_samples(
        valid_rows, q_vocab, r_vocab, args.max_query_len, args.max_relation_len
    )
    if not train_samples:
        raise ValueError("No relation training samples were found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QueryRelationScorer(
        len(q_vocab.itos),
        len(r_vocab.itos),
        q_vocab.stoi[PAD],
        r_vocab.stoi[PAD],
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, q_vocab.stoi[PAD], r_vocab.stoi[PAD]),
    )
    valid_loader = DataLoader(
        valid_samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, q_vocab.stoi[PAD], r_vocab.stoi[PAD]),
    ) if valid_samples else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for q_ids, r_ids, neg_ids, relation_keys in train_loader:
            q_vec = model.encode_query(q_ids.to(device))
            r_vec = model.encode_relation(r_ids.to(device))
            logits = q_vec @ r_vec.t() / args.temperature
            contrastive = multi_positive_nce(logits, relation_keys)
            neg_vec = model.encode_relation(neg_ids.to(device))
            pos_score = (q_vec * r_vec).sum(-1)
            neg_score = (q_vec * neg_vec).sum(-1)
            margin_loss = torch.relu(args.margin - pos_score + neg_score).mean()
            loss = contrastive + args.margin_weight * margin_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        correct = total = 0
        if valid_loader:
            with torch.no_grad():
                for q_ids, r_ids, _neg_ids, relation_keys in valid_loader:
                    q_vec = model.encode_query(q_ids.to(device))
                    r_vec = model.encode_relation(r_ids.to(device))
                    pred = (q_vec @ r_vec.t()).argmax(-1).tolist()
                    correct += sum(
                        1 for i, j in enumerate(pred) if relation_keys[i] == relation_keys[j]
                    )
                    total += len(pred)
        recall1 = correct / max(1, total)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "valid_inbatch_recall1": recall1,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        metric = recall1 if valid_loader else -row["train_loss"]
        if metric > best:
            best = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "emb_dim": args.emb_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "max_query_len": args.max_query_len,
                        "max_relation_len": args.max_relation_len,
                    },
                    "query_vocab": save_vocab(q_vocab),
                    "relation_vocab": save_vocab(r_vocab),
                },
                out / "best.pt",
            )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
