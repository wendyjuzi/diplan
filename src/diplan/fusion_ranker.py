
"""Learned fusion head for parallel KGQA planning signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


FEATURE_NAMES = [
    "base_score",
    "value_z",
    "question_z",
    "candidate_diffusion_z",
    "trajectory_diffusion_z",
    "prior_z",
    "guided_z",
    "entity_count_log",
    "candidate_rank_frac",
    "depth_frac",
    "has_entities",
]


class FusionRanker(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class FusionRankerBundle:
    model: FusionRanker
    feature_names: List[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    device: torch.device


def z_norm(xs: Sequence[float]) -> List[float]:
    vals = [float(x) for x in xs]
    if not vals:
        return []
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    std = var ** 0.5
    if std < 1e-8:
        return [0.0 for _ in vals]
    return [(x - mu) / std for x in vals]


def normalize_features(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.to(x.device)) / std.to(x.device).clamp(min=1e-6)


@torch.no_grad()
def score_fusion(bundle: FusionRankerBundle, features: Sequence[Sequence[float]]) -> List[float]:
    if not features:
        return []
    x = torch.tensor(features, dtype=torch.float32, device=bundle.device)
    x = normalize_features(x, bundle.feature_mean, bundle.feature_std)
    bundle.model.eval()
    return bundle.model(x).detach().cpu().tolist()


def load_fusion_ranker(path: str, device: str = "cpu") -> FusionRankerBundle:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    feature_names = list(ckpt["feature_names"])
    model = FusionRanker(
        n_features=len(feature_names),
        hidden_dim=int(ckpt["model_config"].get("hidden_dim", 64)),
        dropout=float(ckpt["model_config"].get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    dev = torch.device(device)
    model.to(dev).eval()
    return FusionRankerBundle(
        model=model,
        feature_names=feature_names,
        feature_mean=torch.tensor(ckpt["feature_mean"], dtype=torch.float32),
        feature_std=torch.tensor(ckpt["feature_std"], dtype=torch.float32),
        device=dev,
    )
