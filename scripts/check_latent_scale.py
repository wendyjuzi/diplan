import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import load_jsonl
from src.diplan.torch_pipeline import PathAutoencoder, load_vocab


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--max_path_len", type=int, default=8)
    args = parser.parse_args()

    ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    cfg = ckpt["model_config"]
    path_vocab = load_vocab(ckpt["path_vocab"])

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

    rows = load_jsonl(args.data)
    rows = [r for r in rows if isinstance(r.get("oracle_path"), list) and r["oracle_path"]]

    z_list = []
    for r in rows:
        ids = path_vocab.encode(r["oracle_path"], add_bos_eos=True, max_len=args.max_path_len)
        x = torch.tensor([ids], dtype=torch.long)
        lens = torch.tensor([len(ids)], dtype=torch.long)
        z = model.encode(x, lens)  # [1, latent_dim]
        z_list.append(z.squeeze(0))

    if not z_list:
        raise ValueError("No valid rows with oracle_path found.")

    z_all = torch.stack(z_list, dim=0)  # [N, D]
    print(f"N={z_all.shape[0]}, D={z_all.shape[1]}")
    print(f"global_mean={z_all.mean().item():.6f}")
    print(f"global_std={z_all.std(unbiased=False).item():.6f}")
    print(f"global_min={z_all.min().item():.6f}")
    print(f"global_max={z_all.max().item():.6f}")

    per_dim_mean = z_all.mean(dim=0)
    per_dim_std = z_all.std(dim=0, unbiased=False)
    print(f"per_dim_mean_abs_avg={per_dim_mean.abs().mean().item():.6f}")
    print(f"per_dim_std_avg={per_dim_std.mean().item():.6f}")
    print(f"per_dim_std_min={per_dim_std.min().item():.6f}")
    print(f"per_dim_std_max={per_dim_std.max().item():.6f}")


if __name__ == "__main__":
    main()
