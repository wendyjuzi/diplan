"""Reusable DiPLaN inference helpers (planner sampling / decoding / value scoring).

These are task-agnostic and are shared by the ALFWorld diffusion executor
(``scripts/run_alfworld_diplan_diffusion.py``). The KGQA evaluator keeps its own
heavily-tuned variants of the same primitives; this module is the clean,
self-contained path for new agents and is intentionally dependency-light.
"""

from typing import List, Optional

import torch

from src.diplan.torch_pipeline import (
    EOS,
    PAD,
    SEP,
    DiffusionPlanner,
    MLPPlanner,
    PathAutoencoder,
    ValueRanker,
    collate_value,
    sample_latent,
)


def encode_condition(
    query_vocab,
    query_tokens: List[str],
    executed_prefix: Optional[List[str]] = None,
    max_len: int = 32,
    use_prefix: bool = False,
) -> List[int]:
    """Build the planner condition token-id stream (paper §5.1 / §5.5).

    When ``use_prefix`` is set the executed prefix is appended after a ``SEP``
    separator so re-planning is state-aware: ``query_tokens [SEP] executed_prefix``.
    """
    tokens = list(query_tokens)
    if use_prefix and executed_prefix:
        tokens = tokens + [SEP] + list(executed_prefix)
    ids = query_vocab.encode(tokens, add_bos_eos=False, max_len=max_len)
    if not ids:
        ids = [query_vocab.stoi[PAD]]
    return ids


@torch.no_grad()
def decode_latents_to_paths(
    autoencoder: PathAutoencoder,
    path_vocab,
    z: torch.Tensor,
    max_path_len: int,
) -> List[List[str]]:
    seq_ids, pred_lens = autoencoder.decode_greedy(
        z,
        bos_id=path_vocab.stoi["<bos>"],
        eos_id=path_vocab.stoi[EOS],
        max_len=max_path_len,
    )
    cands: List[List[str]] = []
    for ids, ln in zip(seq_ids, pred_lens):
        ids = ids[: max(1, min(ln, max_path_len))]
        cands.append(path_vocab.decode(ids, skip_special=True))
    return cands


@torch.no_grad()
def sample_plan_candidates(
    planner,
    autoencoder: PathAutoencoder,
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    num_candidates: int,
    diffusion_steps: int,
    max_path_len: int,
    device: torch.device,
    executed_prefix: Optional[List[str]] = None,
    use_prefix: bool = False,
    cond_max_len: int = 32,
    latent_mean: Optional[torch.Tensor] = None,
    latent_std: Optional[torch.Tensor] = None,
    prediction_target: str = "z0",
    planner_type: str = "diffusion",
    jitter_std: float = 0.0,
) -> List[List[str]]:
    """Sample ``num_candidates`` plans and decode them to abstract token lists."""
    q_ids = encode_condition(
        query_vocab, query_tokens, executed_prefix, max_len=cond_max_len, use_prefix=use_prefix
    )
    q_batch = torch.tensor([q_ids for _ in range(num_candidates)], dtype=torch.long, device=device)
    if planner_type == "mlp":
        z = planner(q_batch)
    else:
        z = sample_latent(
            planner=planner,
            q_ids=q_batch,
            latent_dim=planner.latent_dim,
            diffusion_steps=diffusion_steps,
            device=device,
            prediction_target=prediction_target,
        )
    if jitter_std > 0.0:
        z = z + torch.randn_like(z) * jitter_std
    if latent_mean is not None and latent_std is not None:
        z = z * latent_std + latent_mean
    return decode_latents_to_paths(autoencoder, path_vocab, z, max_path_len)


@torch.no_grad()
def score_candidates_with_value(
    value_model: Optional[ValueRanker],
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    candidates: List[List[str]],
    device: torch.device,
    max_query_len: int = 32,
    max_path_len: int = 30,
) -> List[float]:
    """Return a value score per candidate (length-based fallback if no model)."""
    if not candidates:
        return []
    if value_model is None:
        expected = float(max(1, len(query_tokens)))
        return [-(abs(len(c) - expected)) for c in candidates]
    q = query_vocab.encode(query_tokens, add_bos_eos=False, max_len=max_query_len)
    if not q:
        q = [query_vocab.stoi[PAD]]
    rows = []
    for c in candidates:
        p = path_vocab.encode(c, add_bos_eos=False, max_len=max_path_len)
        if not p:
            p = [path_vocab.stoi[PAD]]
        rows.append((q, p, 0.0))
    q_ids, p_ids, _ = collate_value(rows, query_vocab.stoi[PAD], path_vocab.stoi[PAD])
    logits = value_model(q_ids.to(device), p_ids.to(device))
    return logits.detach().cpu().tolist()


@torch.no_grad()
def constraint_violation_scores(
    constraint_model: Optional[ValueRanker],
    path_vocab,
    query_vocab,
    query_tokens: List[str],
    candidates: List[List[str]],
    device: torch.device,
    max_query_len: int = 32,
    max_path_len: int = 30,
) -> List[float]:
    """Return P(violation) in [0,1] per candidate from the learned constraint model."""
    if constraint_model is None or not candidates:
        return [0.0 for _ in candidates]
    raw = score_candidates_with_value(
        constraint_model,
        path_vocab,
        query_vocab,
        query_tokens,
        candidates,
        device,
        max_query_len=max_query_len,
        max_path_len=max_path_len,
    )
    return torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).tolist()
