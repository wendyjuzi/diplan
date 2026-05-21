import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_jsonl
from src.diplan.torch_pipeline import (
    DiffusionDataset,
    MLPPlanner,
    PAD,
    PathAutoencoder,
    collate_diffusion,
    load_vocab,
    set_seed,
)


@torch.no_grad()
def _compute_latent_stats(
    loader: DataLoader,
    ae: PathAutoencoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_z = None
    sum_sq = None
    n = 0
    for p_ids, p_lens, _q_ids, _q_lens in loader:
        p_ids = p_ids.to(device)
        p_lens = p_lens.to(device)
        z = ae.encode(p_ids, p_lens).detach().cpu()
        if sum_z is None:
            sum_z = torch.zeros_like(z[0])
            sum_sq = torch.zeros_like(z[0])
        sum_z += z.sum(dim=0)
        sum_sq += (z * z).sum(dim=0)
        n += z.size(0)
    if n == 0:
        raise ValueError("No samples to compute latent stats.")
    mean = sum_z / n
    var = sum_sq / n - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_torch_kgqa.yaml")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/mlp_kgqa_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rows = load_jsonl(cfg["train_path"])
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")

    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(ae_ckpt["query_vocab"])
    pad_p = path_vocab.stoi[PAD]
    pad_q = query_vocab.stoi[PAD]

    dataset = DiffusionDataset(
        rows=rows,
        path_vocab=path_vocab,
        query_vocab=query_vocab,
        max_path_len=int(cfg.get("max_path_len", 8)),
        max_query_len=int(cfg.get("max_query_len", 24)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
        collate_fn=lambda b: collate_diffusion(b, pad_p, pad_q),
    )
    stats_loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        collate_fn=lambda b: collate_diffusion(b, pad_p, pad_q),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")

    ae_cfg = ae_ckpt["model_config"]
    ae = PathAutoencoder(
        vocab_size=ae_cfg["vocab_size"],
        emb_dim=ae_cfg["emb_dim"],
        hid_dim=ae_cfg["hid_dim"],
        latent_dim=ae_cfg["latent_dim"],
        max_path_len=ae_cfg["max_path_len"],
        pad_id=ae_cfg["pad_id"],
        latent_noise_std=float(ae_cfg.get("latent_noise_std", 0.0)),
    ).to(device)
    ae.load_state_dict(ae_ckpt["model_state"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    latent_mean_cpu, latent_std_cpu = _compute_latent_stats(stats_loader, ae, device)
    latent_mean = latent_mean_cpu.to(device)
    latent_std = latent_std_cpu.to(device)

    model = MLPPlanner(
        latent_dim=ae_cfg["latent_dim"],
        q_vocab_size=len(query_vocab.itos),
        q_emb_dim=int(cfg.get("q_emb_dim", 128)),
        q_pad_id=pad_q,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))

    best_loss = float("inf")
    best_state = None
    train_metrics = []
    epochs = int(cfg.get("epochs", 8))
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0
        count = 0
        for p_ids, p_lens, q_ids, _q_lens in loader:
            p_ids = p_ids.to(device)
            p_lens = p_lens.to(device)
            q_ids = q_ids.to(device)
            with torch.no_grad():
                z0_raw = ae.encode(p_ids, p_lens)
                z0 = (z0_raw - latent_mean) / latent_std
            z_hat = model(q_ids)
            loss = F.mse_loss(z_hat, z0)
            mse = F.mse_loss(z_hat, z0)
            cos = F.cosine_similarity(z_hat, z0, dim=1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            total_mse += float(mse.item())
            total_cos += float(cos.item())
            count += 1
        avg_loss = total_loss / max(1, count)
        avg_mse = total_mse / max(1, count)
        avg_cos = total_cos / max(1, count)
        train_metrics.append({"epoch": ep, "loss": avg_loss, "z0_mse": avg_mse, "z0_cosine": avg_cos})
        print(f"[mlp] epoch {ep}/{epochs} loss={avg_loss:.4f} z0_mse={avg_mse:.4f} z0_cos={avg_cos:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ensure_dir(args.out)
    ckpt = {
        "model_state": best_state,
        "model_config": {
            "planner_type": "mlp",
            "latent_dim": ae_cfg["latent_dim"],
            "q_vocab_size": len(query_vocab.itos),
            "q_emb_dim": int(cfg.get("q_emb_dim", 128)),
            "q_pad_id": pad_q,
            "hidden_dim": int(cfg.get("hidden_dim", 256)),
            "prediction_target": "z0",
            "diffusion_steps": 1,
        },
        "query_vocab": ae_ckpt["query_vocab"],
        "path_vocab": ae_ckpt["path_vocab"],
        "latent_norm": {
            "enabled": True,
            "mean": latent_mean_cpu.tolist(),
            "std": latent_std_cpu.tolist(),
        },
        "ae_ref": args.ae_ckpt,
        "train_info": {"samples": len(dataset), "best_loss": best_loss, "planner_type": "mlp"},
    }
    out_path = Path(args.out) / "best.pt"
    torch.save(ckpt, out_path)
    dump_json(
        str(Path(args.out) / "train_metrics.json"),
        {
            "samples": len(dataset),
            "planner_type": "mlp",
            "epochs": train_metrics,
        },
    )
    print(f"Saved mlp planner to {out_path}")


if __name__ == "__main__":
    main()
