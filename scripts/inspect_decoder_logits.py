import argparse
from pathlib import Path
from typing import Dict, List

import torch

from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_jsonl
from src.diplan.torch_pipeline import (
    BOS,
    EOS,
    PAD,
    DiffusionPlanner,
    PathAutoencoder,
    load_vocab,
    pad_2d,
    sample_latent,
)


@torch.no_grad()
def _trace_decoder(
    autoencoder: PathAutoencoder,
    path_vocab,
    z: torch.Tensor,
    max_len: int,
    top_k: int,
) -> Dict:
    bos_id = path_vocab.stoi[BOS]
    eos_id = path_vocab.stoi[EOS]
    hidden = autoencoder.from_latent(z).unsqueeze(0)
    cur = torch.full((z.size(0), 1), bos_id, dtype=torch.long, device=z.device)
    steps: List[Dict] = []
    emitted: List[int] = []
    for step in range(max_len + 1):
        d, hidden = autoencoder.dec(autoencoder.emb(cur), hidden)
        logits = autoencoder.out(d[:, -1, :])
        k = min(top_k, logits.size(-1))
        vals, idxs = torch.topk(logits[0], k=k, dim=-1)
        greedy_id = int(torch.argmax(logits[0]).item())
        token = path_vocab.itos[greedy_id] if 0 <= greedy_id < len(path_vocab.itos) else "<oov>"
        steps.append(
            {
                "step": step,
                "greedy_id": greedy_id,
                "greedy_token": token,
                "topk": [
                    {
                        "id": int(tid.item()),
                        "token": path_vocab.itos[int(tid.item())]
                        if 0 <= int(tid.item()) < len(path_vocab.itos)
                        else "<oov>",
                        "logit": float(tv.item()),
                    }
                    for tv, tid in zip(vals, idxs)
                ],
            }
        )
        cur = torch.tensor([[greedy_id]], dtype=torch.long, device=z.device)
        if greedy_id == eos_id:
            break
        emitted.append(greedy_id)

    seq_ids, pred_lens = autoencoder.decode_greedy(
        z,
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=max_len,
    )
    decoded = path_vocab.decode(seq_ids[0][: max(1, min(pred_lens[0], max_len))], skip_special=True)
    return {
        "decoded_ids": emitted,
        "decoded_tokens": path_vocab.decode(emitted, skip_special=True),
        "decode_greedy_tokens": decoded,
        "pred_len": int(pred_lens[0]),
        "steps": steps,
    }


def _select_row(rows: List[Dict], task_id: str | None, index: int) -> Dict:
    if task_id:
        for row in rows:
            if str(row.get("task_id")) == str(task_id):
                return row
        raise ValueError(f"task_id={task_id} not found in dataset")
    if index < 0 or index >= len(rows):
        raise ValueError(f"index out of range: {index}, dataset size={len(rows)}")
    return rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_torch_kgqa.tune3.mem_opt.json")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/diagnostics/decoder_logits_trace.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["test_path"])
    row = _select_row(rows, args.task_id, args.index)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")

    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])

    ae_cfg = ae_ckpt["model_config"]
    autoencoder = PathAutoencoder(
        vocab_size=ae_cfg["vocab_size"],
        emb_dim=ae_cfg["emb_dim"],
        hid_dim=ae_cfg["hid_dim"],
        latent_dim=ae_cfg["latent_dim"],
        max_path_len=ae_cfg["max_path_len"],
        pad_id=ae_cfg["pad_id"],
        latent_noise_std=float(ae_cfg.get("latent_noise_std", 0.0)),
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["model_state"])
    autoencoder.eval()

    pl_cfg = planner_ckpt["model_config"]
    prediction_target = str(pl_cfg.get("prediction_target", "z0")).lower()
    planner = DiffusionPlanner(
        latent_dim=pl_cfg["latent_dim"],
        q_vocab_size=pl_cfg["q_vocab_size"],
        q_emb_dim=pl_cfg["q_emb_dim"],
        q_pad_id=pl_cfg["q_pad_id"],
        time_dim=pl_cfg["time_dim"],
    ).to(device)
    planner.load_state_dict(planner_ckpt["model_state"])
    planner.eval()

    latent_mean = None
    latent_std = None
    latent_norm = planner_ckpt.get("latent_norm")
    if isinstance(latent_norm, dict) and bool(latent_norm.get("enabled", False)):
        latent_mean = torch.tensor(latent_norm["mean"], dtype=torch.float32, device=device).view(1, -1)
        latent_std = torch.tensor(latent_norm["std"], dtype=torch.float32, device=device).view(1, -1)

    q_ids = query_vocab.encode(row.get("query_tokens", []), add_bos_eos=False, max_len=24)
    if not q_ids:
        q_ids = [query_vocab.stoi[PAD]]
    q_batch = torch.tensor([q_ids], dtype=torch.long, device=device)

    z_norm = sample_latent(
        planner=planner,
        q_ids=q_batch,
        latent_dim=planner.latent_dim,
        diffusion_steps=int(pl_cfg.get("diffusion_steps", 20)),
        device=device,
        prediction_target=prediction_target,
    )
    z_denorm = z_norm
    if latent_mean is not None and latent_std is not None:
        z_denorm = z_norm * latent_std + latent_mean

    oracle_ids = path_vocab.encode(
        row.get("oracle_path", []),
        add_bos_eos=True,
        max_len=int(cfg.get("max_path_len", 8)),
    )
    p_ids, p_lens = pad_2d([oracle_ids], path_vocab.stoi[PAD])
    z_oracle = autoencoder.encode(p_ids.to(device), p_lens.to(device))

    max_len = int(cfg.get("max_path_len", 8))
    trace_norm = _trace_decoder(autoencoder, path_vocab, z_norm, max_len=max_len, top_k=max(1, args.top_k))
    trace_denorm = _trace_decoder(autoencoder, path_vocab, z_denorm, max_len=max_len, top_k=max(1, args.top_k))
    trace_oracle = _trace_decoder(autoencoder, path_vocab, z_oracle, max_len=max_len, top_k=max(1, args.top_k))

    out = {
        "task_id": row.get("task_id"),
        "dataset": row.get("dataset"),
        "query": row.get("question", " ".join(row.get("query_tokens", []))),
        "query_tokens": row.get("query_tokens", []),
        "oracle_path": row.get("oracle_path", []),
        "planner_prediction_target": prediction_target,
        "latent_norm_enabled": bool(latent_mean is not None and latent_std is not None),
        "z_norm_stats": {
            "mean": float(z_norm.mean().item()),
            "std": float(z_norm.std().item()),
            "min": float(z_norm.min().item()),
            "max": float(z_norm.max().item()),
        },
        "z_denorm_stats": {
            "mean": float(z_denorm.mean().item()),
            "std": float(z_denorm.std().item()),
            "min": float(z_denorm.min().item()),
            "max": float(z_denorm.max().item()),
        },
        "z_oracle_stats": {
            "mean": float(z_oracle.mean().item()),
            "std": float(z_oracle.std().item()),
            "min": float(z_oracle.min().item()),
            "max": float(z_oracle.max().item()),
        },
        "trace_pre_denorm": trace_norm,
        "trace_post_denorm": trace_denorm,
        "trace_oracle_z": trace_oracle,
    }

    ensure_dir(str(Path(args.out).parent))
    dump_json(args.out, out)
    print(f"Saved logits trace to {args.out}")
    print(
        f"task_id={out['task_id']} pred_target={prediction_target} "
        f"pre={trace_norm['decode_greedy_tokens']} post={trace_denorm['decode_greedy_tokens']}"
    )


if __name__ == "__main__":
    main()
