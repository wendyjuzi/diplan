"""Candidate-conditioned discrete denoising planner for KGQA.

This is a lightweight D3PM-style planner over ToG candidate relations.  The
model does not generate arbitrary KG relations; it denoises a corrupted current
relation back to one relation inside the legal candidate set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kg_env import KGEnv
from .relation_scorer import conditioned_query_tokens, question_text_tokens, relation_text_tokens
from .torch_pipeline import PAD, UNK, SimpleVocab, pad_2d


MASK_REL = "<mask_rel>"


def _relation_tokens(relation: str) -> List[str]:
    if relation == MASK_REL:
        return [MASK_REL]
    return relation_text_tokens(relation)


def _relation_jaccard(a: str, b: str) -> float:
    aa = set(_relation_tokens(a))
    bb = set(_relation_tokens(b))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _hard_negative_relation(gold: str, candidates: Sequence[str]) -> str:
    pool = [rel for rel in candidates if rel != gold]
    if not pool:
        return MASK_REL
    return max(pool, key=lambda rel: (_relation_jaccard(gold, rel), rel))


def build_candidate_diffusion_vocabs(rows: Iterable[Dict], min_freq: int = 1) -> tuple[SimpleVocab, SimpleVocab]:
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
                yield _relation_tokens(str(rel))
            for triple in row.get("graph", []) or []:
                if isinstance(triple, (list, tuple)) and len(triple) >= 2:
                    yield _relation_tokens(str(triple[1]))

    return SimpleVocab.build(query_streams(), min_freq=min_freq), SimpleVocab.build(relation_streams(), min_freq=min_freq)


class CandidateDenoisingPlanner(nn.Module):
    def __init__(
        self,
        q_vocab_size: int,
        r_vocab_size: int,
        q_pad_id: int,
        r_pad_id: int,
        num_steps: int = 20,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q_pad_id = int(q_pad_id)
        self.r_pad_id = int(r_pad_id)
        self.num_steps = int(num_steps)
        self.q_emb = nn.Embedding(q_vocab_size, emb_dim, padding_idx=q_pad_id)
        self.r_emb = nn.Embedding(r_vocab_size, emb_dim, padding_idx=r_pad_id)
        self.t_emb = nn.Embedding(self.num_steps + 1, emb_dim)
        self.ctx_proj = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rel_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    @staticmethod
    def _mean(ids: torch.Tensor, emb: nn.Embedding, pad_id: int) -> torch.Tensor:
        x = emb(ids)
        mask = (ids != pad_id).float().unsqueeze(-1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def _mean_rel_3d(self, ids: torch.Tensor) -> torch.Tensor:
        bsz, n_cand, rel_len = ids.shape
        flat = ids.reshape(bsz * n_cand, rel_len)
        vec = self._mean(flat, self.r_emb, self.r_pad_id)
        return vec.reshape(bsz, n_cand, -1)

    def forward(
        self,
        q_ids: torch.Tensor,
        noisy_ids: torch.Tensor,
        t_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_vec = self._mean(q_ids, self.q_emb, self.q_pad_id)
        noisy_vec = self._mean(noisy_ids, self.r_emb, self.r_pad_id)
        t_vec = self.t_emb(t_ids.clamp(0, self.num_steps))
        ctx = self.ctx_proj(torch.cat([q_vec, noisy_vec, t_vec], dim=-1))
        cand_vec = self.rel_proj(self._mean_rel_3d(candidate_ids))
        logits = (cand_vec * ctx.unsqueeze(1)).sum(-1)
        return logits.masked_fill(~candidate_mask, -1e9)


@dataclass
class CandidateDiffusionBundle:
    model: CandidateDenoisingPlanner
    query_vocab: SimpleVocab
    relation_vocab: SimpleVocab
    device: torch.device
    num_steps: int = 20
    max_query_len: int = 48
    max_relation_len: int = 16
    max_candidates: int = 64


def build_step_samples(
    rows: Sequence[Dict],
    q_vocab: SimpleVocab,
    r_vocab: SimpleVocab,
    max_query_len: int,
    max_relation_len: int,
    max_candidates: int,
    num_steps: int,
    seed: int = 42,
    condition_dropout: float = 0.0,
    noise_strategy: str = "random",
) -> List[Dict]:
    rng = random.Random(seed)
    samples: List[Dict] = []
    for row in rows:
        oracle = list(row.get("oracle_path", []) or [])
        if not oracle:
            continue
        max_steps = int((row.get("constraints") or {}).get("max_steps", max(1, len(oracle))))
        env = KGEnv.from_rog_row(row, max_steps=max_steps)
        state = env.reset()
        base_q = list(row.get("query_tokens") or question_text_tokens(str(row.get("question", ""))))
        prefix: List[str] = []
        for step, gold in enumerate(oracle):
            candidates = env.admissible_relations(state)
            if gold not in candidates:
                break
            if len(candidates) > max_candidates:
                keep = [gold]
                others = [r for r in candidates if r != gold]
                rng.shuffle(others)
                candidates = sorted(keep + others[: max_candidates - 1])
            gold_idx = candidates.index(gold)
            t = rng.randint(1, num_steps)
            p_corrupt = t / max(1, num_steps)
            if rng.random() < p_corrupt:
                if rng.random() < 0.5 or len(candidates) == 1:
                    noisy = MASK_REL
                elif noise_strategy == "hard":
                    noisy = _hard_negative_relation(gold, candidates)
                else:
                    pool = [r for r in candidates if r != gold]
                    noisy = rng.choice(pool) if pool else MASK_REL
            else:
                noisy = gold
            q_tokens = conditioned_query_tokens(base_q, prefix)
            if condition_dropout > 0.0 and rng.random() < condition_dropout:
                q_tokens = []
            samples.append(
                {
                    "q": q_vocab.encode(q_tokens, max_len=max_query_len) or [q_vocab.stoi[UNK]],
                    "noisy": r_vocab.encode(_relation_tokens(noisy), max_len=max_relation_len) or [r_vocab.stoi[UNK]],
                    "t": t,
                    "candidates": [
                        r_vocab.encode(_relation_tokens(rel), max_len=max_relation_len) or [r_vocab.stoi[UNK]]
                        for rel in candidates
                    ],
                    "gold_idx": gold_idx,
                    "gold_relation": gold,
                    "candidate_relations": candidates,
                }
            )
            state = env.step(state, gold)
            prefix.append(gold)
            if env.is_terminal(state):
                break
    return samples


def collate_candidate_diffusion(batch, q_pad: int, r_pad: int):
    q_ids, _ = pad_2d([x["q"] for x in batch], q_pad)
    noisy_ids, _ = pad_2d([x["noisy"] for x in batch], r_pad)
    max_c = max(len(x["candidates"]) for x in batch)
    max_l = max(max(len(c) for c in x["candidates"]) for x in batch)
    cand = torch.full((len(batch), max_c, max_l), r_pad, dtype=torch.long)
    mask = torch.zeros((len(batch), max_c), dtype=torch.bool)
    for i, row in enumerate(batch):
        for j, ids in enumerate(row["candidates"]):
            cand[i, j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, j] = True
    t_ids = torch.tensor([x["t"] for x in batch], dtype=torch.long)
    gold = torch.tensor([x["gold_idx"] for x in batch], dtype=torch.long)
    return q_ids, noisy_ids, t_ids, cand, mask, gold


@torch.no_grad()
def score_candidate_relations(
    bundle: CandidateDiffusionBundle,
    question: str,
    query_tokens: Sequence[str],
    relations: Sequence[str],
    executed_prefix: Sequence[str] = (),
    timestep: int | None = None,
    guidance_scale: float = 1.0,
) -> List[float]:
    if not relations:
        return []
    q_tokens = conditioned_query_tokens(list(query_tokens) or question_text_tokens(question), executed_prefix)
    q = bundle.query_vocab.encode(q_tokens, max_len=bundle.max_query_len) or [bundle.query_vocab.stoi[UNK]]
    noisy = bundle.relation_vocab.encode([MASK_REL], max_len=bundle.max_relation_len) or [bundle.relation_vocab.stoi[UNK]]
    candidate_rows = [
        bundle.relation_vocab.encode(_relation_tokens(rel), max_len=bundle.max_relation_len)
        or [bundle.relation_vocab.stoi[UNK]]
        for rel in relations
    ]
    q_ids, _ = pad_2d([q], bundle.query_vocab.stoi[PAD])
    noisy_ids, _ = pad_2d([noisy], bundle.relation_vocab.stoi[PAD])
    max_l = max(len(x) for x in candidate_rows)
    cand = torch.full((1, len(candidate_rows), max_l), bundle.relation_vocab.stoi[PAD], dtype=torch.long)
    mask = torch.ones((1, len(candidate_rows)), dtype=torch.bool)
    for j, ids in enumerate(candidate_rows):
        cand[0, j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    t = bundle.num_steps if timestep is None else int(timestep)
    t_ids = torch.tensor([t], dtype=torch.long)
    bundle.model.eval()
    cond_logits = bundle.model(
        q_ids.to(bundle.device),
        noisy_ids.to(bundle.device),
        t_ids.to(bundle.device),
        cand.to(bundle.device),
        mask.to(bundle.device),
    )[0]
    if abs(float(guidance_scale) - 1.0) > 1e-8:
        uncond = [bundle.query_vocab.stoi[UNK]]
        uncond_ids, _ = pad_2d([uncond], bundle.query_vocab.stoi[PAD])
        uncond_logits = bundle.model(
            uncond_ids.to(bundle.device),
            noisy_ids.to(bundle.device),
            t_ids.to(bundle.device),
            cand.to(bundle.device),
            mask.to(bundle.device),
        )[0]
        logits = uncond_logits + float(guidance_scale) * (cond_logits - uncond_logits)
    else:
        logits = cond_logits
    return logits.detach().cpu().tolist()


def load_candidate_diffusion(path: str, device: str = "cpu") -> CandidateDiffusionBundle:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    q_vocab = SimpleVocab(stoi=ckpt["query_vocab"]["stoi"], itos=ckpt["query_vocab"]["itos"])
    r_vocab = SimpleVocab(stoi=ckpt["relation_vocab"]["stoi"], itos=ckpt["relation_vocab"]["itos"])
    cfg = ckpt["model_config"]
    model = CandidateDenoisingPlanner(
        q_vocab_size=len(q_vocab.itos),
        r_vocab_size=len(r_vocab.itos),
        q_pad_id=q_vocab.stoi[PAD],
        r_pad_id=r_vocab.stoi[PAD],
        num_steps=int(cfg.get("num_steps", 20)),
        emb_dim=int(cfg.get("emb_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    dev = torch.device(device)
    model.to(dev).eval()
    return CandidateDiffusionBundle(
        model=model,
        query_vocab=q_vocab,
        relation_vocab=r_vocab,
        device=dev,
        num_steps=int(cfg.get("num_steps", 20)),
        max_query_len=int(cfg.get("max_query_len", 48)),
        max_relation_len=int(cfg.get("max_relation_len", 16)),
        max_candidates=int(cfg.get("max_candidates", 64)),
    )
