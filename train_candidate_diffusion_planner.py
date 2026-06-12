"""Train a candidate-conditioned discrete denoising planner for KGQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.candidate_diffusion import (
    CandidateDenoisingPlanner,
    build_candidate_diffusion_vocabs,
    build_step_samples,
    collate_candidate_diffusion,
)
from src.diplan.io_utils import load_jsonl
from src.diplan.torch_pipeline import PAD, save_vocab, set_seed


def evaluate(model, loader, device):
    model.eval()
    total = correct1 = correct3 = 0
    losses = []
    with torch.no_grad():
        for q_ids, noisy_ids, t_ids, cand, mask, gold in loader:
            logits = model(
                q_ids.to(device),
                noisy_ids.to(device),
                t_ids.to(device),
                cand.to(device),
                mask.to(device),
            )
            gold = gold.to(device)
            loss = F.cross_entropy(logits, gold)
            losses.append(float(loss.item()))
            order = logits.argsort(dim=1, descending=True)
            correct1 += int((order[:, 0] == gold).sum().item())
            correct3 += int((order[:, : min(3, order.shape[1])] == gold.unsqueeze(1)).any(dim=1).sum().item())
            total += int(gold.numel())
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "recall@1": correct1 / max(1, total),
        "recall@3": correct3 / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--valid_path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--condition_dropout", type=float, default=0.1)
    ap.add_argument("--noise_strategy", choices=["random", "hard"], default="hard")
    ap.add_argument("--max_query_len", type=int, default=48)
    ap.add_argument("--max_relation_len", type=int, default=16)
    ap.add_argument("--max_candidates", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    train_rows = load_jsonl(args.train_path)
    valid_rows = load_jsonl(args.valid_path) if args.valid_path else []
    q_vocab, r_vocab = build_candidate_diffusion_vocabs(train_rows)
    train_samples = build_step_samples(
        train_rows,
        q_vocab,
        r_vocab,
        args.max_query_len,
        args.max_relation_len,
        args.max_candidates,
        args.num_steps,
        seed=args.seed,
        condition_dropout=args.condition_dropout,
        noise_strategy=args.noise_strategy,
    )
    valid_samples = build_step_samples(
        valid_rows,
        q_vocab,
        r_vocab,
        args.max_query_len,
        args.max_relation_len,
        args.max_candidates,
        args.num_steps,
        seed=args.seed + 1,
        condition_dropout=0.0,
        noise_strategy=args.noise_strategy,
    ) if valid_rows else []
    if not train_samples:
        raise ValueError("No candidate diffusion training samples were found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateDenoisingPlanner(
        len(q_vocab.itos),
        len(r_vocab.itos),
        q_vocab.stoi[PAD],
        r_vocab.stoi[PAD],
        num_steps=args.num_steps,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    collate = lambda b: collate_candidate_diffusion(b, q_vocab.stoi[PAD], r_vocab.stoi[PAD])
    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if valid_samples else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for q_ids, noisy_ids, t_ids, cand, mask, gold in train_loader:
            logits = model(
                q_ids.to(device),
                noisy_ids.to(device),
                t_ids.to(device),
                cand.to(device),
                mask.to(device),
            )
            loss = F.cross_entropy(logits, gold.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        valid = evaluate(model, valid_loader, device) if valid_loader else {"loss": 0.0, "recall@1": 0.0, "recall@3": 0.0}
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "valid_loss": valid["loss"],
            "valid_recall@1": valid["recall@1"],
            "valid_recall@3": valid["recall@3"],
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        metric = valid["recall@1"] if valid_loader else -row["train_loss"]
        if metric > best:
            best = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "num_steps": args.num_steps,
                        "emb_dim": args.emb_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "condition_dropout": args.condition_dropout,
                        "noise_strategy": args.noise_strategy,
                        "max_query_len": args.max_query_len,
                        "max_relation_len": args.max_relation_len,
                        "max_candidates": args.max_candidates,
                    },
                    "query_vocab": save_vocab(q_vocab),
                    "relation_vocab": save_vocab(r_vocab),
                },
                out / "best.pt",
            )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
