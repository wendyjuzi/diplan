"""Question-conditioned relation retrieval for KGQA.

The scorer is intentionally lightweight: it learns separate query and relation
token encoders with in-batch contrastive training. It complements the trajectory
prior by estimating P(relation | question) directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_pipeline import PAD, UNK, SimpleVocab, pad_2d


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def relation_text_tokens(relation: str) -> List[str]:
    return [x.lower() for x in TOKEN_RE.findall(relation.replace(".", " ").replace("_", " "))]


def question_text_tokens(question: str) -> List[str]:
    return [x.lower() for x in TOKEN_RE.findall(question)]


def conditioned_query_tokens(query_tokens: Sequence[str], executed_prefix: Sequence[str]) -> List[str]:
    out = list(query_tokens)
    if executed_prefix:
        out.append("<sep>")
        for rel in executed_prefix:
            out.extend(relation_text_tokens(str(rel)))
    return out


def build_relation_vocabs(rows: Iterable[Dict], min_freq: int = 1) -> tuple[SimpleVocab, SimpleVocab]:
    rows = list(rows)

    def query_streams():
        for row in rows:
            base = list(row.get("query_tokens") or question_text_tokens(str(row.get("question", ""))))
            oracle = list(row.get("oracle_path", []) or [])
            for step in range(max(1, len(oracle))):
                yield conditioned_query_tokens(base, oracle[:step])

    q_vocab = SimpleVocab.build(
        query_streams(),
        min_freq=min_freq,
    )

    def relation_streams():
        for row in rows:
            for rel in row.get("oracle_path", []) or []:
                yield relation_text_tokens(str(rel))
            for triple in row.get("graph", []) or []:
                if isinstance(triple, (list, tuple)) and len(triple) >= 2:
                    yield relation_text_tokens(str(triple[1]))

    r_vocab = SimpleVocab.build(relation_streams(), min_freq=min_freq)
    return q_vocab, r_vocab


class QueryRelationScorer(nn.Module):
    def __init__(
        self,
        q_vocab_size: int,
        r_vocab_size: int,
        q_pad_id: int,
        r_pad_id: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q_pad_id = int(q_pad_id)
        self.r_pad_id = int(r_pad_id)
        self.q_emb = nn.Embedding(q_vocab_size, emb_dim, padding_idx=q_pad_id)
        self.r_emb = nn.Embedding(r_vocab_size, emb_dim, padding_idx=r_pad_id)
        self.q_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.r_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    @staticmethod
    def _mean(ids: torch.Tensor, emb: nn.Embedding, pad_id: int) -> torch.Tensor:
        x = emb(ids)
        mask = (ids != pad_id).float().unsqueeze(-1)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def encode_query(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_proj(self._mean(ids, self.q_emb, self.q_pad_id)), dim=-1)

    def encode_relation(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.r_proj(self._mean(ids, self.r_emb, self.r_pad_id)), dim=-1)

    def forward(self, q_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        return (self.encode_query(q_ids) * self.encode_relation(r_ids)).sum(-1)


@dataclass
class RelationScorerBundle:
    model: QueryRelationScorer
    query_vocab: SimpleVocab
    relation_vocab: SimpleVocab
    device: torch.device
    max_query_len: int = 48
    max_relation_len: int = 16


def score_relations(
    bundle: RelationScorerBundle,
    question: str,
    query_tokens: Sequence[str],
    relations: Sequence[str],
    executed_prefix: Sequence[str] = (),
) -> List[float]:
    if not relations:
        return []
    q_tokens = conditioned_query_tokens(
        list(query_tokens) or question_text_tokens(question),
        executed_prefix,
    )
    q = bundle.query_vocab.encode(q_tokens, max_len=bundle.max_query_len)
    if not q:
        q = [bundle.query_vocab.stoi[UNK]]
    relation_ids = []
    for rel in relations:
        ids = bundle.relation_vocab.encode(relation_text_tokens(rel), max_len=bundle.max_relation_len)
        relation_ids.append(ids or [bundle.relation_vocab.stoi[UNK]])
    q_ids, _ = pad_2d([q] * len(relations), bundle.query_vocab.stoi[PAD])
    r_ids, _ = pad_2d(relation_ids, bundle.relation_vocab.stoi[PAD])
    bundle.model.eval()
    with torch.no_grad():
        scores = bundle.model(q_ids.to(bundle.device), r_ids.to(bundle.device))
    return scores.detach().cpu().tolist()


def load_relation_scorer(path: str, device: str = "cpu") -> RelationScorerBundle:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    q_vocab = SimpleVocab(stoi=ckpt["query_vocab"]["stoi"], itos=ckpt["query_vocab"]["itos"])
    r_vocab = SimpleVocab(stoi=ckpt["relation_vocab"]["stoi"], itos=ckpt["relation_vocab"]["itos"])
    cfg = ckpt["model_config"]
    model = QueryRelationScorer(
        q_vocab_size=len(q_vocab.itos),
        r_vocab_size=len(r_vocab.itos),
        q_pad_id=q_vocab.stoi[PAD],
        r_pad_id=r_vocab.stoi[PAD],
        emb_dim=int(cfg["emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    dev = torch.device(device)
    model.to(dev).eval()
    return RelationScorerBundle(
        model=model,
        query_vocab=q_vocab,
        relation_vocab=r_vocab,
        device=dev,
        max_query_len=int(cfg.get("max_query_len", 48)),
        max_relation_len=int(cfg.get("max_relation_len", 16)),
    )
