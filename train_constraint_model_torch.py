"""Train the learned constraint model C_xi (paper §5.3 / §7.3.3).

Predicts ``P(violation)`` for a (query, plan) pair. Positives are rule-feasible
plans (label 0); negatives are rule-infeasible candidate plans plus synthetic
corruptions that deliberately trigger each constraint class the checker knows
(banned relation, max-steps overflow, ordering break). Trained with BCE.

The checkpoint mirrors the value-model ``model_config`` block so evaluate_torch can
load it with the same :class:`ValueRanker`/:class:`ConstraintModel` plumbing.

Usage:
    python train_constraint_model_torch.py --config configs/constraint_torch_kgqa.diplan_full.json \
        --planner_ckpt runs/<planner>/best.pt --out runs/<constraint>
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.diplan.constraints import is_feasible
from src.diplan.io_utils import dump_json, ensure_dir, load_config, load_jsonl
from src.diplan.torch_pipeline import (
    PAD,
    ConstraintModel,
    collate_value,
    load_vocab,
    set_seed,
)


def _corrupt_to_violate(oracle: List[str], constraints: Dict, all_rel: List[str], rng: random.Random) -> List[List[str]]:
    """Synthetic infeasible plans that trigger each known constraint class."""
    negs: List[List[str]] = []
    # 1) banned relation injected (if any banned relations declared).
    banned = list(constraints.get("banned_relations", []) or [])
    if banned and oracle:
        bad = list(oracle)
        bad[rng.randrange(len(bad))] = rng.choice(banned)
        negs.append(bad)
    # 2) exceed max_steps by padding with extra (legal) relations.
    max_steps = int(constraints.get("max_steps", 8))
    if all_rel:
        over = list(oracle) + [rng.choice(all_rel) for _ in range(max(1, max_steps - len(oracle) + 1))]
        if len(over) > max_steps:
            negs.append(over)
    # 3) ordering break: reverse the plan (violates required_before / stage order if present).
    if (constraints.get("required_before") or constraints.get("required_stage_order") or constraints.get("must_precede")) and len(oracle) > 1:
        negs.append(list(reversed(oracle)))
    # Keep only the ones the checker actually rejects.
    return [n for n in negs if not is_feasible(n, constraints)[0]]


class ConstraintDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        path_vocab,
        query_vocab,
        max_path_len: int,
        max_query_len: int,
        max_pos_per_row: int = 4,
        seed: int = 42,
    ) -> None:
        rng = random.Random(seed)
        all_rel = sorted({r for row in rows for r in row.get("oracle_path", [])})
        self.samples: List[Tuple[List[int], List[int], float]] = []
        n_pos = n_neg = 0
        for row in rows:
            constraints = row.get("constraints", {}) or {}
            state_prefixes = row.get("state_query_tokens_by_prefix") or []
            q_tokens = list(state_prefixes[0]) if state_prefixes else row.get("query_tokens", [])
            q = query_vocab.encode(q_tokens, add_bos_eos=False, max_len=max_query_len)
            if not q:
                continue
            oracle = list(row.get("oracle_path", []))
            if not oracle:
                continue
            meta_by_path = {}
            for item in row.get("candidate_metadata", []) or []:
                if isinstance(item, dict) and isinstance(item.get("path"), list):
                    meta_by_path[tuple(item["path"])] = item

            # Positives: rule-feasible plans (oracle + feasible candidate paths).
            pos_paths: List[List[str]] = [oracle]
            for c in row.get("candidate_paths", []) or []:
                meta = meta_by_path.get(tuple(c), {}) if isinstance(c, list) else {}
                explicit_ok = meta.get("is_executable", None)
                ok = bool(explicit_ok) if explicit_ok is not None else is_feasible(c, constraints)[0]
                if isinstance(c, list) and c and ok:
                    pos_paths.append(c)
            seen = set()
            kept_pos = []
            for p in pos_paths:
                key = tuple(p)
                if key in seen:
                    continue
                seen.add(key)
                kept_pos.append(p)
            kept_pos = kept_pos[:max_pos_per_row]

            # Negatives: rule-infeasible candidates + synthetic corruptions.
            neg_paths: List[List[str]] = []
            for c in row.get("candidate_paths", []) or []:
                meta = meta_by_path.get(tuple(c), {}) if isinstance(c, list) else {}
                explicit_ok = meta.get("is_executable", None)
                bad = (not bool(explicit_ok)) if explicit_ok is not None else (not is_feasible(c, constraints)[0])
                if isinstance(c, list) and c and bad:
                    neg_paths.append(c)
            neg_paths.extend(_corrupt_to_violate(oracle, constraints, all_rel, rng))
            neg_seen = set()
            kept_neg = []
            for p in neg_paths:
                key = tuple(p)
                if key in neg_seen:
                    continue
                neg_seen.add(key)
                kept_neg.append(p)

            for p in kept_pos:
                pid = path_vocab.encode(p, add_bos_eos=False, max_len=max_path_len)
                if pid:
                    self.samples.append((q, pid, 0.0))
                    n_pos += 1
            for p in kept_neg:
                pid = path_vocab.encode(p, add_bos_eos=False, max_len=max_path_len)
                if pid:
                    self.samples.append((q, pid, 1.0))
                    n_neg += 1
        self.n_pos = n_pos
        self.n_neg = n_neg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], float]:
        return self.samples[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/constraint_torch_kgqa.diplan_full.json")
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/constraint_kgqa_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rows = load_jsonl(cfg["train_path"])
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    path_vocab = load_vocab(planner_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])
    pad_q = query_vocab.stoi[PAD]
    pad_p = path_vocab.stoi[PAD]

    dataset = ConstraintDataset(
        rows=rows,
        path_vocab=path_vocab,
        query_vocab=query_vocab,
        max_path_len=int(cfg.get("max_path_len", 8)),
        max_query_len=int(cfg.get("max_query_len", 24)),
        max_pos_per_row=int(cfg.get("max_pos_per_row", 4)),
        seed=int(cfg.get("seed", 42)),
    )
    print(f"[constraint] samples={len(dataset)} pos={dataset.n_pos} neg={dataset.n_neg}")
    if dataset.n_neg == 0:
        print("[constraint] WARNING: no infeasible negatives found; constraint model will be trivial.")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        collate_fn=lambda b: collate_value(b, pad_q, pad_p),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")
    model = ConstraintModel(
        q_vocab_size=len(query_vocab.itos),
        p_vocab_size=len(path_vocab.itos),
        emb_dim=int(cfg.get("emb_dim", 128)),
        q_pad_id=pad_q,
        p_pad_id=pad_p,
        architecture=str(cfg.get("value_architecture", "cross")),
        hidden_dim=int(cfg.get("value_hidden_dim", 256)),
        dropout=float(cfg.get("value_dropout", 0.1)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))

    # Class weighting to counter the feasible-heavy distribution.
    pos_weight = None
    if dataset.n_neg > 0:
        pw = float(dataset.n_pos) / float(dataset.n_neg)
        pos_weight = torch.tensor([max(0.1, min(10.0, pw))], dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = None
    metrics = []
    epochs = int(cfg.get("epochs", 4))
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        correct = 0
        n = 0
        for q_ids, p_ids, y in loader:
            q_ids = q_ids.to(device)
            p_ids = p_ids.to(device)
            y = y.to(device)
            logits = model(q_ids, p_ids)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            count += 1
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((pred == y).sum().item())
            n += int(y.numel())
        avg = total / max(1, count)
        acc = correct / max(1, n)
        metrics.append({"epoch": ep, "loss": avg, "acc": acc})
        print(f"[constraint] epoch {ep}/{epochs} loss={avg:.4f} acc={acc:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ensure_dir(args.out)
    ckpt = {
        "model_state": best_state,
        "model_config": {
            "q_vocab_size": len(query_vocab.itos),
            "p_vocab_size": len(path_vocab.itos),
            "emb_dim": int(cfg.get("emb_dim", 128)),
            "q_pad_id": pad_q,
            "p_pad_id": pad_p,
            "architecture": str(cfg.get("value_architecture", "cross")),
            "hidden_dim": int(cfg.get("value_hidden_dim", 256)),
            "dropout": float(cfg.get("value_dropout", 0.1)),
            "is_constraint_model": True,
        },
        "query_vocab": planner_ckpt["query_vocab"],
        "path_vocab": planner_ckpt["path_vocab"],
        "train_info": {"samples": len(dataset), "pos": dataset.n_pos, "neg": dataset.n_neg, "best_loss": best_loss},
    }
    out_path = Path(args.out) / "best.pt"
    torch.save(ckpt, out_path)
    dump_json(str(Path(args.out) / "train_metrics.json"), {"epochs": metrics, "pos": dataset.n_pos, "neg": dataset.n_neg})
    print(f"Saved torch constraint model to {out_path}")


if __name__ == "__main__":
    main()
