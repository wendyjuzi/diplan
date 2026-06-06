import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.io_utils import ensure_dir, load_config, load_jsonl
from src.diplan.torch_pipeline import (
    AutoencoderDataset,
    BOS,
    EOS,
    PAD,
    PathAutoencoder,
    build_vocabs,
    collate_autoencoder,
    save_vocab,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder_torch_kgqa.yaml")
    parser.add_argument("--out", type=str, default="runs/ae_kgqa_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rows = load_jsonl(cfg["train_path"])

    # When prefix conditioning is enabled (paper §5.1/§5.5) the query/condition vocab
    # is unioned with plan tokens + <sep> so an executed prefix can be packed into the
    # planner condition. AE, planner and value models all share this vocab.
    path_vocab, query_vocab = build_vocabs(
        rows,
        min_freq=int(cfg.get("min_freq", 1)),
        prefix_conditioning=bool(cfg.get("prefix_conditioning", False)),
    )
    dataset = AutoencoderDataset(
        rows=rows,
        path_vocab=path_vocab,
        max_path_len=int(cfg.get("max_path_len", 8)),
    )
    pad_id = path_vocab.stoi[PAD]
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
        collate_fn=lambda b: collate_autoencoder(b, pad_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")
    model = PathAutoencoder(
        vocab_size=len(path_vocab.itos),
        emb_dim=int(cfg.get("emb_dim", 128)),
        hid_dim=int(cfg.get("hid_dim", 192)),
        latent_dim=int(cfg.get("latent_dim", 128)),
        max_path_len=int(cfg.get("max_path_len", 8)),
        pad_id=pad_id,
        latent_noise_std=float(cfg.get("latent_noise_std", 0.1)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))

    best_loss = float("inf")
    best_state = None
    epochs = int(cfg.get("epochs", 8))
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, lens in loader:
            x = x.to(device)
            lens = lens.to(device)
            logits, target, len_logits = model(x, lens)
            loss_seq = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=pad_id,
            )
            target_len = (lens - 2).clamp(min=1, max=int(cfg.get("max_path_len", 8))).long()
            loss_len = F.cross_entropy(len_logits, target_len)
            loss = loss_seq + float(cfg.get("length_loss_weight", 0.25)) * loss_len
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            count += 1
        avg = total / max(1, count)
        print(f"[ae] epoch {ep}/{epochs} loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ensure_dir(args.out)
    ckpt = {
        "model_state": best_state,
        "model_config": {
            "vocab_size": len(path_vocab.itos),
            "emb_dim": int(cfg.get("emb_dim", 128)),
            "hid_dim": int(cfg.get("hid_dim", 192)),
            "latent_dim": int(cfg.get("latent_dim", 128)),
            "max_path_len": int(cfg.get("max_path_len", 8)),
            "pad_id": pad_id,
            "bos_id": path_vocab.stoi[BOS],
            "eos_id": path_vocab.stoi[EOS],
            "latent_noise_std": float(cfg.get("latent_noise_std", 0.1)),
        },
        "path_vocab": save_vocab(path_vocab),
        "query_vocab": save_vocab(query_vocab),
        "train_info": {"samples": len(dataset), "best_loss": best_loss},
    }
    out_path = Path(args.out) / "best.pt"
    torch.save(ckpt, out_path)
    print(f"Saved torch autoencoder to {out_path}")


if __name__ == "__main__":
    main()
