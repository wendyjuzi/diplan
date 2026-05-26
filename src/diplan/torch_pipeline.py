import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SimpleVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, token_lists: Iterable[List[str]], min_freq: int = 1) -> "SimpleVocab":
        freq: Dict[str, int] = {}
        for toks in token_lists:
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
        itos = [PAD, BOS, EOS, UNK]
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and tok not in itos:
                itos.append(tok)
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, tokens: List[str], add_bos_eos: bool = False, max_len: int | None = None) -> List[int]:
        out = [self.stoi.get(t, self.stoi[UNK]) for t in tokens]
        if max_len is not None:
            out = out[:max_len]
        if add_bos_eos:
            out = [self.stoi[BOS]] + out + [self.stoi[EOS]]
        return out

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        tokens = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            t = self.itos[i]
            if skip_special and t in (PAD, BOS, EOS):
                continue
            tokens.append(t)
        return tokens


def pad_2d(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs) if seqs else 1
    out = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long)
    lens = torch.zeros((len(seqs),), dtype=torch.long)
    for i, s in enumerate(seqs):
        lens[i] = len(s)
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lens


class AutoencoderDataset(Dataset):
    def __init__(self, rows: List[Dict], path_vocab: SimpleVocab, max_path_len: int) -> None:
        self.samples = []
        for row in rows:
            ids = path_vocab.encode(row["oracle_path"], add_bos_eos=True, max_len=max_path_len)
            if len(ids) >= 3:
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.samples[idx]


class DiffusionDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        path_vocab: SimpleVocab,
        query_vocab: SimpleVocab,
        max_path_len: int,
        max_query_len: int,
    ) -> None:
        self.samples = []
        for row in rows:
            path_ids = path_vocab.encode(row["oracle_path"], add_bos_eos=True, max_len=max_path_len)
            query_ids = query_vocab.encode(row["query_tokens"], add_bos_eos=False, max_len=max_query_len)
            if len(path_ids) >= 3 and len(query_ids) >= 1:
                self.samples.append((path_ids, query_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


class ValueDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        path_vocab: SimpleVocab,
        query_vocab: SimpleVocab,
        max_path_len: int,
        max_query_len: int,
        neg_per_pos: int = 2,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        all_rel = sorted(list({r for row in rows for r in row["oracle_path"]}))
        self.samples = []
        for row in rows:
            q = query_vocab.encode(row["query_tokens"], add_bos_eos=False, max_len=max_query_len)
            pos = path_vocab.encode(row["oracle_path"], add_bos_eos=False, max_len=max_path_len)
            if not q or not pos:
                continue
            self.samples.append((q, pos, 1.0))
            for _ in range(neg_per_pos):
                neg_tokens = list(row["oracle_path"])
                if neg_tokens:
                    j = rng.randrange(len(neg_tokens))
                    neg_tokens[j] = rng.choice(all_rel)
                neg = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
                self.samples.append((q, neg, 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], float]:
        return self.samples[idx]


class PathAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hid_dim: int,
        latent_dim: int,
        max_path_len: int,
        pad_id: int,
        latent_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_path_len = max_path_len
        self.pad_id = pad_id
        self.latent_noise_std = float(latent_noise_std)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.enc = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.to_latent = nn.Linear(hid_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hid_dim)
        self.dec = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)
        self.len_head = nn.Linear(latent_dim, max_path_len + 1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        h, _ = self.enc(e)
        idx = (lengths - 1).clamp(min=0)
        last = h[torch.arange(h.size(0), device=h.device), idx]
        z = self.to_latent(last)
        return z

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x, lengths)
        if self.training and self.latent_noise_std > 0.0:
            # Force decoder robustness to small latent perturbations.
            z = z + torch.randn_like(z) * self.latent_noise_std
        inp = x[:, :-1]
        target = x[:, 1:]
        h0 = self.from_latent(z).unsqueeze(0)
        d, _ = self.dec(self.emb(inp), h0)
        logits = self.out(d)
        len_logits = self.len_head(z)
        return logits, target, len_logits

    def decode_greedy(self, z: torch.Tensor, bos_id: int, eos_id: int, max_len: int) -> Tuple[List[List[int]], List[int]]:
        bsz = z.size(0)
        hidden = self.from_latent(z).unsqueeze(0)
        cur = torch.full((bsz, 1), bos_id, dtype=torch.long, device=z.device)
        done = torch.zeros((bsz,), dtype=torch.bool, device=z.device)
        outputs: List[List[int]] = [[] for _ in range(bsz)]
        for _ in range(max_len + 1):
            d, hidden = self.dec(self.emb(cur), hidden)
            logits = self.out(d[:, -1, :])
            nxt = torch.argmax(logits, dim=-1)
            cur = nxt.unsqueeze(1)
            for i in range(bsz):
                tok = int(nxt[i].item())
                if not done[i]:
                    if tok == eos_id:
                        done[i] = True
                    else:
                        outputs[i].append(tok)
            if torch.all(done):
                break
        len_logits = self.len_head(z)
        pred_len = torch.argmax(len_logits, dim=-1).tolist()
        pred_len = [max(1, int(x)) for x in pred_len]
        return outputs, pred_len


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, pad_id: int) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

    def forward(self, q_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(q_ids)
        mask = (q_ids != self.pad_id).float().unsqueeze(-1)
        summed = (e * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1)
    )
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.size(0), 1), device=t.device)], dim=1)
    return emb


class DiffusionPlanner(nn.Module):
    def __init__(self, latent_dim: int, q_vocab_size: int, q_emb_dim: int, q_pad_id: int, time_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.q_enc = QuestionEncoder(q_vocab_size, q_emb_dim, q_pad_id)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + q_emb_dim + time_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, q_ids: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        q = self.q_enc(q_ids)
        t_emb = timestep_embedding(t_norm, self.time_dim)
        x = torch.cat([z_t, q, t_emb], dim=1)
        return self.net(x)


class MLPPlanner(nn.Module):
    def __init__(self, latent_dim: int, q_vocab_size: int, q_emb_dim: int, q_pad_id: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.q_enc = QuestionEncoder(q_vocab_size, q_emb_dim, q_pad_id)
        self.net = nn.Sequential(
            nn.Linear(q_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, q_ids: torch.Tensor) -> torch.Tensor:
        q = self.q_enc(q_ids)
        return self.net(q)


class ValueRanker(nn.Module):
    def __init__(
        self,
        q_vocab_size: int,
        p_vocab_size: int,
        emb_dim: int,
        q_pad_id: int,
        p_pad_id: int,
        architecture: str = "legacy",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.architecture = str(architecture).lower()
        if self.architecture not in {"legacy", "cross", "cross_terminal"}:
            raise ValueError(f"Unsupported ValueRanker architecture: {architecture}")
        self.q_pad_id = q_pad_id
        self.p_pad_id = p_pad_id
        self.q_emb = nn.Embedding(q_vocab_size, emb_dim, padding_idx=q_pad_id)
        self.p_emb = nn.Embedding(p_vocab_size, emb_dim, padding_idx=p_pad_id)
        if self.architecture == "legacy":
            # Keep the original layout for checkpoint compatibility.
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        elif self.architecture == "cross":
            feat_dim = emb_dim * 6
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            feat_dim = emb_dim * 9 + 2
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _masked_mean(self, ids: torch.Tensor, emb: nn.Embedding, pad_id: int) -> torch.Tensor:
        e = emb(ids)
        m = (ids != pad_id).float().unsqueeze(-1)
        s = (e * m).sum(dim=1)
        d = m.sum(dim=1).clamp(min=1.0)
        return s / d

    def _masked_last(self, ids: torch.Tensor, emb: nn.Embedding, pad_id: int) -> torch.Tensor:
        e = emb(ids)
        lens = (ids != pad_id).long().sum(dim=1).clamp(min=1) - 1
        idx = torch.arange(ids.size(0), device=ids.device)
        return e[idx, lens]

    def _masked_weighted_mean(self, ids: torch.Tensor, emb: nn.Embedding, pad_id: int) -> torch.Tensor:
        e = emb(ids)
        mask = (ids != pad_id).float()
        pos = torch.arange(ids.size(1), device=ids.device, dtype=torch.float32) + 1.0
        w = mask * pos.unsqueeze(0)
        s = (e * w.unsqueeze(-1)).sum(dim=1)
        d = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        return s / d

    def _cross_attention_mean(
        self,
        q_e: torch.Tensor,
        p_e: torch.Tensor,
        q_ids: torch.Tensor,
        p_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q_e: [B, Q, D], p_e: [B, P, D]
        d = q_e.size(-1)
        scale = math.sqrt(max(1, d))
        scores = torch.matmul(q_e, p_e.transpose(1, 2)) / scale  # [B, Q, P]
        p_mask = (p_ids != self.p_pad_id).unsqueeze(1)  # [B,1,P]
        scores = scores.masked_fill(~p_mask, -1e9)
        attn_q2p = torch.softmax(scores, dim=-1)
        q_ctx = torch.matmul(attn_q2p, p_e)  # [B,Q,D]
        q_mask = (q_ids != self.q_pad_id).float().unsqueeze(-1)
        q_ctx_mean = (q_ctx * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1.0)

        scores_t = scores.transpose(1, 2)  # [B,P,Q]
        q_mask_t = (q_ids != self.q_pad_id).unsqueeze(1)  # [B,1,Q]
        scores_t = scores_t.masked_fill(~q_mask_t, -1e9)
        attn_p2q = torch.softmax(scores_t, dim=-1)
        p_ctx = torch.matmul(attn_p2q, q_e)  # [B,P,D]
        p_mask_f = (p_ids != self.p_pad_id).float().unsqueeze(-1)
        p_ctx_mean = (p_ctx * p_mask_f).sum(dim=1) / p_mask_f.sum(dim=1).clamp(min=1.0)
        return q_ctx_mean, p_ctx_mean

    def forward(self, q_ids: torch.Tensor, p_ids: torch.Tensor) -> torch.Tensor:
        q = self._masked_mean(q_ids, self.q_emb, self.q_pad_id)
        p = self._masked_mean(p_ids, self.p_emb, self.p_pad_id)
        if self.architecture == "legacy":
            x = torch.cat([q, p], dim=1)
            return self.mlp(x).squeeze(1)

        q_e = self.q_emb(q_ids)
        p_e = self.p_emb(p_ids)
        q_ctx_mean, p_ctx_mean = self._cross_attention_mean(q_e, p_e, q_ids, p_ids)
        if self.architecture == "cross":
            x = torch.cat(
                [
                    q,
                    p,
                    torch.abs(q - p),
                    q * p,
                    q_ctx_mean,
                    p_ctx_mean,
                ],
                dim=1,
            )
        else:
            q_last = self._masked_last(q_ids, self.q_emb, self.q_pad_id)
            p_last = self._masked_last(p_ids, self.p_emb, self.p_pad_id)
            p_weighted = self._masked_weighted_mean(p_ids, self.p_emb, self.p_pad_id)
            q_len = (q_ids != self.q_pad_id).float().sum(dim=1, keepdim=True) / max(1, q_ids.size(1))
            p_len = (p_ids != self.p_pad_id).float().sum(dim=1, keepdim=True) / max(1, p_ids.size(1))
            x = torch.cat(
                [
                    q,
                    p,
                    q_last,
                    p_last,
                    torch.abs(q - p),
                    q * p,
                    q_ctx_mean,
                    p_ctx_mean,
                    p_weighted,
                    q_len,
                    p_len,
                ],
                dim=1,
            )
        return self.mlp(x).squeeze(1)


def collate_autoencoder(batch: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, lens = pad_2d(batch, pad_id)
    return x, lens


def collate_diffusion(batch: List[Tuple[List[int], List[int]]], p_pad: int, q_pad: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = [x[0] for x in batch]
    q = [x[1] for x in batch]
    p_ids, p_lens = pad_2d(p, p_pad)
    q_ids, q_lens = pad_2d(q, q_pad)
    return p_ids, p_lens, q_ids, q_lens


def collate_value(batch: List[Tuple[List[int], List[int], float]], q_pad: int, p_pad: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = [x[0] for x in batch]
    p = [x[1] for x in batch]
    y = torch.tensor([x[2] for x in batch], dtype=torch.float32)
    q_ids, _ = pad_2d(q, q_pad)
    p_ids, _ = pad_2d(p, p_pad)
    return q_ids, p_ids, y


def build_vocabs(rows: List[Dict], min_freq: int = 1) -> Tuple[SimpleVocab, SimpleVocab]:
    path_vocab = SimpleVocab.build((r["oracle_path"] for r in rows), min_freq=min_freq)
    query_vocab = SimpleVocab.build((r["query_tokens"] for r in rows), min_freq=min_freq)
    return path_vocab, query_vocab


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps)


def q_sample(z0: torch.Tensor, t_idx: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    a = alpha_bars[t_idx].unsqueeze(1)
    return a.sqrt() * z0 + (1.0 - a).sqrt() * noise


@torch.no_grad()
def sample_latent(
    planner: DiffusionPlanner,
    q_ids: torch.Tensor,
    latent_dim: int,
    diffusion_steps: int,
    device: torch.device,
    prediction_target: str = "z0",
) -> torch.Tensor:
    betas = linear_beta_schedule(diffusion_steps).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    z = torch.randn((q_ids.size(0), latent_dim), device=device)
    target = prediction_target.lower()
    if target not in {"z0", "eps"}:
        raise ValueError(f"Unsupported prediction_target: {prediction_target}")
    for t in reversed(range(diffusion_steps)):
        t_idx = torch.full((q_ids.size(0),), t, dtype=torch.long, device=device)
        t_norm = t_idx.float() / max(1, diffusion_steps - 1)
        a_t = alpha_bars[t]
        model_out = planner(z, q_ids, t_norm)
        if target == "z0":
            z0 = model_out
            eps = (z - a_t.sqrt() * z0) / (1.0 - a_t).sqrt().clamp(min=1e-6)
        else:
            eps = model_out
            z0 = (z - (1.0 - a_t).sqrt() * eps) / a_t.sqrt().clamp(min=1e-6)
        if t == 0:
            z = z0
            continue
        a_prev = alpha_bars[t - 1]
        z = a_prev.sqrt() * z0 + (1.0 - a_prev).sqrt() * eps
    return z


def save_vocab(v: SimpleVocab) -> Dict:
    return {"stoi": v.stoi, "itos": v.itos}


def load_vocab(obj: Dict) -> SimpleVocab:
    return SimpleVocab(stoi=obj["stoi"], itos=obj["itos"])
