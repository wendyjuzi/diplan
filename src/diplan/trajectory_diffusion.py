"""Trajectory-level discrete denoising planner for KGQA relation sequences."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from .relation_scorer import conditioned_query_tokens, question_text_tokens
from .torch_pipeline import EOS, PAD, UNK, SimpleVocab, pad_2d


MASK_REL = "<mask_rel>"


def build_trajectory_vocabs(rows: Iterable[Dict], min_freq: int = 1) -> tuple[SimpleVocab, SimpleVocab]:
    rows = list(rows)

    def query_streams():
        for row in rows:
            base = list(row.get("query_tokens") or question_text_tokens(str(row.get("question", ""))))
            oracle = list(row.get("oracle_path", []) or [])
            for step in range(max(1, len(oracle))):
                yield conditioned_query_tokens(base, oracle[:step])

    def relation_streams():
        yield [MASK_REL]
        for row in rows:
            for rel in row.get("oracle_path", []) or []:
                yield [str(rel)]
            for triple in row.get("graph", []) or []:
                if isinstance(triple, (list, tuple)) and len(triple) >= 2:
                    yield [str(triple[1])]

    return SimpleVocab.build(query_streams(), min_freq=min_freq), SimpleVocab.build(relation_streams(), min_freq=min_freq)


class TrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        q_vocab_size: int,
        r_vocab_size: int,
        q_pad_id: int,
        r_pad_id: int,
        horizon: int = 3,
        num_steps: int = 20,
        emb_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q_pad_id = int(q_pad_id)
        self.r_pad_id = int(r_pad_id)
        self.horizon = int(horizon)
        self.num_steps = int(num_steps)
        self.q_emb = nn.Embedding(q_vocab_size, emb_dim, padding_idx=q_pad_id)
        self.r_emb = nn.Embedding(r_vocab_size, emb_dim, padding_idx=r_pad_id)
        self.pos_emb = nn.Embedding(self.horizon, emb_dim)
        self.t_emb = nn.Embedding(self.num_steps + 1, emb_dim)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(emb_dim, r_vocab_size)

    @staticmethod
    def _mean(ids: torch.Tensor, emb: nn.Embedding, pad_id: int) -> torch.Tensor:
        x = emb(ids)
        mask = (ids != pad_id).float().unsqueeze(-1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, q_ids: torch.Tensor, noisy_rel_ids: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        bsz, horizon = noisy_rel_ids.shape
        pos = torch.arange(horizon, device=noisy_rel_ids.device).unsqueeze(0).expand(bsz, -1)
        q_vec = self.q_proj(self._mean(q_ids, self.q_emb, self.q_pad_id)).unsqueeze(1)
        x = self.r_emb(noisy_rel_ids) + self.pos_emb(pos) + self.t_emb(t_ids.clamp(0, self.num_steps)).unsqueeze(1) + q_vec
        return self.out(self.encoder(x))


@dataclass
class TrajectoryDiffusionBundle:
    model: TrajectoryDenoiser
    query_vocab: SimpleVocab
    relation_vocab: SimpleVocab
    device: torch.device
    horizon: int = 3
    num_steps: int = 20
    max_query_len: int = 64


def build_trajectory_samples(
    rows: Sequence[Dict],
    q_vocab: SimpleVocab,
    r_vocab: SimpleVocab,
    horizon: int,
    num_steps: int,
    max_query_len: int,
    seed: int = 42,
    condition_dropout: float = 0.0,
) -> List[Dict]:
    rng = random.Random(seed)
    samples = []
    mask_id = r_vocab.stoi.get(MASK_REL, r_vocab.stoi[UNK])
    pad_id = r_vocab.stoi[PAD]
    for row in rows:
        oracle = [str(x) for x in (row.get("oracle_path", []) or [])]
        if not oracle:
            continue
        base_q = list(row.get("query_tokens") or question_text_tokens(str(row.get("question", ""))))
        all_rels = sorted({
            str(t[1])
            for t in row.get("graph", []) or []
            if isinstance(t, (list, tuple)) and len(t) >= 2
        })
        for step in range(len(oracle)):
            target_rels = oracle[step : step + horizon]
            target = [r_vocab.stoi.get(rel, r_vocab.stoi[UNK]) for rel in target_rels]
            target = target + [pad_id] * max(0, horizon - len(target))
            t = rng.randint(1, num_steps)
            p = t / max(1, num_steps)
            noisy = list(target)
            for i in range(len(target_rels)):
                if rng.random() < p:
                    if rng.random() < 0.5 or not all_rels:
                        noisy[i] = mask_id
                    else:
                        noisy[i] = r_vocab.stoi.get(rng.choice(all_rels), r_vocab.stoi[UNK])
            q_tokens = conditioned_query_tokens(base_q, oracle[:step])
            if condition_dropout > 0.0 and rng.random() < condition_dropout:
                q_tokens = []
            q = q_vocab.encode(q_tokens, max_len=max_query_len) or [q_vocab.stoi[UNK]]
            samples.append({"q": q, "noisy": noisy, "target": target, "t": t})
    return samples


def collate_trajectory(batch, q_pad: int):
    q_ids, _ = pad_2d([x["q"] for x in batch], q_pad)
    noisy = torch.tensor([x["noisy"] for x in batch], dtype=torch.long)
    target = torch.tensor([x["target"] for x in batch], dtype=torch.long)
    t_ids = torch.tensor([x["t"] for x in batch], dtype=torch.long)
    return q_ids, noisy, target, t_ids


@torch.no_grad()
def score_first_relations(
    bundle: TrajectoryDiffusionBundle,
    question: str,
    query_tokens: Sequence[str],
    relations: Sequence[str],
    executed_prefix: Sequence[str] = (),
    guidance_scale: float = 1.0,
) -> List[float]:
    if not relations:
        return []
    mask_id = bundle.relation_vocab.stoi.get(MASK_REL, bundle.relation_vocab.stoi[UNK])
    q_tokens = conditioned_query_tokens(list(query_tokens) or question_text_tokens(question), executed_prefix)
    q = bundle.query_vocab.encode(q_tokens, max_len=bundle.max_query_len) or [bundle.query_vocab.stoi[UNK]]
    q_ids, _ = pad_2d([q], bundle.query_vocab.stoi[PAD])
    noisy = torch.full((1, bundle.horizon), mask_id, dtype=torch.long)
    t_ids = torch.tensor([bundle.num_steps], dtype=torch.long)
    bundle.model.eval()
    cond = bundle.model(q_ids.to(bundle.device), noisy.to(bundle.device), t_ids.to(bundle.device))[0, 0]
    if abs(float(guidance_scale) - 1.0) > 1e-8:
        uq, _ = pad_2d([[bundle.query_vocab.stoi[UNK]]], bundle.query_vocab.stoi[PAD])
        uncond = bundle.model(uq.to(bundle.device), noisy.to(bundle.device), t_ids.to(bundle.device))[0, 0]
        logits = uncond + float(guidance_scale) * (cond - uncond)
    else:
        logits = cond
    return [float(logits[bundle.relation_vocab.stoi.get(str(rel), bundle.relation_vocab.stoi[UNK])].detach().cpu()) for rel in relations]


def load_trajectory_diffusion(path: str, device: str = "cpu") -> TrajectoryDiffusionBundle:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    q_vocab = SimpleVocab(stoi=ckpt["query_vocab"]["stoi"], itos=ckpt["query_vocab"]["itos"])
    r_vocab = SimpleVocab(stoi=ckpt["relation_vocab"]["stoi"], itos=ckpt["relation_vocab"]["itos"])
    cfg = ckpt["model_config"]
    model = TrajectoryDenoiser(
        len(q_vocab.itos),
        len(r_vocab.itos),
        q_vocab.stoi[PAD],
        r_vocab.stoi[PAD],
        horizon=int(cfg.get("horizon", 3)),
        num_steps=int(cfg.get("num_steps", 20)),
        emb_dim=int(cfg.get("emb_dim", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    dev = torch.device(device)
    model.to(dev).eval()
    return TrajectoryDiffusionBundle(
        model=model,
        query_vocab=q_vocab,
        relation_vocab=r_vocab,
        device=dev,
        horizon=int(cfg.get("horizon", 3)),
        num_steps=int(cfg.get("num_steps", 20)),
        max_query_len=int(cfg.get("max_query_len", 64)),
    )
