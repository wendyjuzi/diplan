import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.io_utils import ensure_dir, load_config, load_jsonl
from src.diplan.io_utils import dump_json
from src.diplan.torch_pipeline import (
    DiffusionDataset,
    DiffusionPlanner,
    PAD,
    PathAutoencoder,
    WeightedDiffusionDataset,
    collate_diffusion,
    collate_weighted_diffusion,
    linear_beta_schedule,
    load_vocab,
    q_sample,
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
    for batch in loader:
        # Works for both collate_diffusion (4-tuple) and collate_weighted_diffusion (5-tuple).
        p_ids, p_lens = batch[0], batch[1]
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
    parser.add_argument("--out", type=str, default="runs/diplan_kgqa_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rows = load_jsonl(cfg["train_path"])
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")

    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(ae_ckpt["query_vocab"])
    pad_p = path_vocab.stoi[PAD]
    pad_q = query_vocab.stoi[PAD]

    # Decision-weighted diffusion (paper §7.3.2) and/or prefix-conditioned planning
    # state (paper §5.1/§5.5) are opt-in via config; when both are off this falls
    # back to the original plain DiffusionDataset so existing runs are unchanged.
    decision_weighting = bool(cfg.get("decision_weighting", False))
    prefix_conditioning = bool(cfg.get("prefix_conditioning", False))
    weighted = decision_weighting or prefix_conditioning
    if weighted:
        dataset = WeightedDiffusionDataset(
            rows=rows,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            max_path_len=int(cfg.get("max_path_len", 8)),
            max_query_len=int(cfg.get("max_query_len", 24)),
            alpha=float(cfg.get("decision_weight_alpha", 1.0)) if decision_weighting else 0.0,
            include_oracle=bool(cfg.get("include_oracle", True)),
            use_feasibility=bool(cfg.get("decision_use_feasibility", True)),
            prefix_conditioning=prefix_conditioning,
            prefix_sample_prob=float(cfg.get("prefix_sample_prob", 0.5)),
            seed=int(cfg.get("seed", 42)),
        )
        collate = lambda b: collate_weighted_diffusion(b, pad_p, pad_q)
        print(
            f"[planner] decision_weighting={decision_weighting} "
            f"alpha={cfg.get('decision_weight_alpha', 1.0)} "
            f"prefix_conditioning={prefix_conditioning} samples={len(dataset)}"
        )
        if hasattr(dataset, "n_pool"):
            avg_w = float(getattr(dataset, "weight_sum", 0.0)) / max(1, int(getattr(dataset, "n_kept", 0)))
            print(
                "[planner] executable supervision "
                f"pool={getattr(dataset, 'n_pool', 0)} "
                f"exec_pos={getattr(dataset, 'n_exec_positive', 0)} "
                f"nonexec={getattr(dataset, 'n_nonexec', 0)} "
                f"kept_for_diffusion={getattr(dataset, 'n_kept', 0)} "
                f"avg_weight={avg_w:.4f}"
            )
    else:
        dataset = DiffusionDataset(
            rows=rows,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            max_path_len=int(cfg.get("max_path_len", 8)),
            max_query_len=int(cfg.get("max_query_len", 24)),
        )
        collate = lambda b: collate_diffusion(b, pad_p, pad_q)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
        collate_fn=collate,
    )
    stats_loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate,
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
    print(
        f"[planner] latent stats mean={latent_mean.mean().item():.4f} "
        f"std={latent_std.mean().item():.4f} "
        f"min={latent_mean.min().item():.4f}/{latent_std.min().item():.4f} "
        f"max={latent_mean.max().item():.4f}/{latent_std.max().item():.4f}"
    )

    planner = DiffusionPlanner(
        latent_dim=ae_cfg["latent_dim"],
        q_vocab_size=len(query_vocab.itos),
        q_emb_dim=int(cfg.get("q_emb_dim", 128)),
        q_pad_id=pad_q,
        time_dim=int(cfg.get("time_dim", 32)),
    ).to(device)
    opt = torch.optim.AdamW(planner.parameters(), lr=float(cfg.get("lr", 3e-4)))
    prediction_target = str(cfg.get("prediction_target", "z0")).lower()
    if prediction_target not in {"z0", "eps"}:
        raise ValueError(f"Unsupported prediction_target={prediction_target}, expected 'z0' or 'eps'.")

    steps = int(cfg.get("diffusion_steps", 20))
    betas = linear_beta_schedule(steps).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    best_loss = float("inf")
    best_state = None
    train_metrics = []
    epochs = int(cfg.get("epochs", 8))
    for ep in range(1, epochs + 1):
        planner.train()
        total_loss = 0.0
        total_mse_z0 = 0.0
        total_cos_z0 = 0.0
        count = 0
        for batch in loader:
            p_ids, p_lens, q_ids = batch[0], batch[1], batch[2]
            w = batch[4] if (weighted and len(batch) >= 5) else None
            p_ids = p_ids.to(device)
            p_lens = p_lens.to(device)
            q_ids = q_ids.to(device)
            if w is not None:
                w = w.to(device)
            with torch.no_grad():
                z0_raw = ae.encode(p_ids, p_lens)
                z0 = (z0_raw - latent_mean) / latent_std
            t_idx = torch.randint(0, steps, (z0.size(0),), device=device)
            t_norm = t_idx.float() / max(1, steps - 1)
            noise = torch.randn_like(z0)
            z_t = q_sample(z0, t_idx, alpha_bars, noise)
            model_out = planner(z_t, q_ids, t_norm)
            a_t = alpha_bars[t_idx].unsqueeze(1)
            if prediction_target == "z0":
                z0_hat = model_out
            else:
                eps_hat = model_out
                z0_hat = (z_t - (1.0 - a_t).sqrt() * eps_hat) / a_t.sqrt().clamp(min=1e-6)
            # Per-sample MSE, decision-weighted per paper §7.3.2 when weights present.
            pred_for_loss = z0_hat if prediction_target == "z0" else eps_hat
            target_for_loss = z0 if prediction_target == "z0" else noise
            if w is not None:
                per = ((pred_for_loss - target_for_loss) ** 2).mean(dim=1)
                loss = (w * per).sum() / w.sum().clamp(min=1e-6)
            else:
                loss = F.mse_loss(pred_for_loss, target_for_loss)
            mse_z0 = F.mse_loss(z0_hat, z0)
            cos_z0 = F.cosine_similarity(z0_hat, z0, dim=1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            total_mse_z0 += float(mse_z0.item())
            total_cos_z0 += float(cos_z0.item())
            count += 1
        avg_loss = total_loss / max(1, count)
        avg_mse_z0 = total_mse_z0 / max(1, count)
        avg_cos_z0 = total_cos_z0 / max(1, count)
        train_metrics.append(
            {
                "epoch": ep,
                "loss": avg_loss,
                "z0_mse": avg_mse_z0,
                "z0_cosine": avg_cos_z0,
            }
        )
        print(
            f"[planner] epoch {ep}/{epochs} loss={avg_loss:.4f} "
            f"z0_mse={avg_mse_z0:.4f} z0_cos={avg_cos_z0:.4f}"
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu() for k, v in planner.state_dict().items()}

    ensure_dir(args.out)
    ckpt = {
        "model_state": best_state,
        "model_config": {
            "latent_dim": ae_cfg["latent_dim"],
            "q_vocab_size": len(query_vocab.itos),
            "q_emb_dim": int(cfg.get("q_emb_dim", 128)),
            "q_pad_id": pad_q,
            "time_dim": int(cfg.get("time_dim", 32)),
            "diffusion_steps": steps,
            "prediction_target": prediction_target,
            "prefix_conditioning": prefix_conditioning,
            "decision_weighting": decision_weighting,
            "max_query_len": int(cfg.get("max_query_len", 24)),
        },
        "query_vocab": ae_ckpt["query_vocab"],
        "path_vocab": ae_ckpt["path_vocab"],
        "latent_norm": {
            "enabled": True,
            "mean": latent_mean_cpu.tolist(),
            "std": latent_std_cpu.tolist(),
        },
        "ae_ref": args.ae_ckpt,
        "train_info": {"samples": len(dataset), "best_loss": best_loss, "prediction_target": prediction_target},
    }
    out_path = Path(args.out) / "best.pt"
    torch.save(ckpt, out_path)
    dump_json(
        str(Path(args.out) / "train_metrics.json"),
        {
            "samples": len(dataset),
            "prediction_target": prediction_target,
            "epochs": train_metrics,
        },
    )
    print(f"Saved torch planner to {out_path}")


if __name__ == "__main__":
    main()
