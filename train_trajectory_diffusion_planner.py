"""Train trajectory-level discrete denoising over relation sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.io_utils import load_jsonl
from src.diplan.torch_pipeline import PAD, save_vocab, set_seed
from src.diplan.trajectory_diffusion import (
    TrajectoryDenoiser,
    build_trajectory_samples,
    build_trajectory_vocabs,
    collate_trajectory,
)


@torch.no_grad()
def evaluate(model, loader, pad_id, device):
    model.eval()
    losses = []
    correct = total = 0
    for q_ids, noisy, target, t_ids in loader:
        logits = model(q_ids.to(device), noisy.to(device), t_ids.to(device))
        target = target.to(device)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad_id)
        losses.append(float(loss.item()))
        mask = target != pad_id
        pred = logits.argmax(-1)
        correct += int(((pred == target) & mask).sum().item())
        total += int(mask.sum().item())
    return {"loss": sum(losses) / max(1, len(losses)), "token_acc": correct / max(1, total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--valid_path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--condition_dropout", type=float, default=0.1)
    ap.add_argument("--max_query_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    train_rows = load_jsonl(args.train_path)
    valid_rows = load_jsonl(args.valid_path) if args.valid_path else []
    q_vocab, r_vocab = build_trajectory_vocabs(train_rows)
    train_samples = build_trajectory_samples(
        train_rows, q_vocab, r_vocab, args.horizon, args.num_steps, args.max_query_len,
        seed=args.seed, condition_dropout=args.condition_dropout,
    )
    valid_samples = build_trajectory_samples(
        valid_rows, q_vocab, r_vocab, args.horizon, args.num_steps, args.max_query_len,
        seed=args.seed + 1, condition_dropout=0.0,
    ) if valid_rows else []
    if not train_samples:
        raise ValueError("No trajectory diffusion samples were found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryDenoiser(
        len(q_vocab.itos),
        len(r_vocab.itos),
        q_vocab.stoi[PAD],
        r_vocab.stoi[PAD],
        horizon=args.horizon,
        num_steps=args.num_steps,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    collate = lambda b: collate_trajectory(b, q_vocab.stoi[PAD])
    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if valid_samples else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for q_ids, noisy, target, t_ids in train_loader:
            logits = model(q_ids.to(device), noisy.to(device), t_ids.to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.to(device).reshape(-1),
                ignore_index=r_vocab.stoi[PAD],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        valid = evaluate(model, valid_loader, r_vocab.stoi[PAD], device) if valid_loader else {"loss": 0.0, "token_acc": 0.0}
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "valid_loss": valid["loss"],
            "valid_token_acc": valid["token_acc"],
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
        }
        print(json.dumps(row), flush=True)
        history.append(row)
        metric = valid["token_acc"] if valid_loader else -row["train_loss"]
        if metric > best:
            best = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "horizon": args.horizon,
                        "num_steps": args.num_steps,
                        "emb_dim": args.emb_dim,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout,
                        "condition_dropout": args.condition_dropout,
                        "max_query_len": args.max_query_len,
                    },
                    "query_vocab": save_vocab(q_vocab),
                    "relation_vocab": save_vocab(r_vocab),
                },
                out / "best.pt",
            )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
