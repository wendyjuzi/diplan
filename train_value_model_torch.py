import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.diplan.io_utils import ensure_dir, load_config, load_jsonl
from src.diplan.torch_pipeline import (
    PAD,
    ValueDataset,
    ValueRanker,
    collate_value,
    load_vocab,
    pad_2d,
    set_seed,
)


def _query_key(tokens: List[str]) -> Tuple[str, ...]:
    return tuple(tokens) if isinstance(tokens, list) else tuple()


def _build_hard_negative_map(
    pred_rows: List[Dict],
    max_per_query: int,
    max_from_candidate_pool: int = 2,
    use_ranking_error_only: bool = True,
    prefer_nearmiss_firsthop: bool = True,
) -> Dict[Tuple[str, ...], List[List[str]]]:
    by_query: Dict[Tuple[str, ...], List[List[str]]] = {}
    for r in pred_rows:
        if use_ranking_error_only and (not bool(r.get("ranking_error", False))):
            continue
        if r.get("success", False):
            continue
        pred = r.get("executed_path", [])
        oracle = r.get("oracle_path", [])
        key = _query_key(r.get("query_tokens", []))
        if not key:
            continue
        oracle = r.get("oracle_path", [])
        bucket = by_query.setdefault(key, [])
        if isinstance(pred, list) and pred and pred != oracle and pred not in bucket:
            bucket.append(pred)
        pool = r.get("candidate_pool_top", [])
        if isinstance(pool, list):
            added = 0
            for item in pool:
                hp = item.get("path") if isinstance(item, dict) else None
                if not isinstance(hp, list) or not hp or hp == oracle:
                    continue
                if hp not in bucket:
                    bucket.append(hp)
                    added += 1
                if added >= max(0, max_from_candidate_pool):
                    break
    # Cap per-query hard negatives.
    for k, v in list(by_query.items()):
        if prefer_nearmiss_firsthop:
            # Promote near-miss negatives: first hop matches oracle first hop.
            oracle_first = None
            # Find any row with this query key to infer oracle first hop.
            for r in pred_rows:
                if _query_key(r.get("query_tokens", [])) == k:
                    o = r.get("oracle_path", [])
                    if isinstance(o, list) and o:
                        oracle_first = o[0]
                        break
            if oracle_first is not None:
                near = [p for p in v if p and p[0] == oracle_first]
                far = [p for p in v if not p or p[0] != oracle_first]
                v = near + far
        by_query[k] = v[: max(0, max_per_query)]
    return by_query


def _build_pairwise_samples(
    rows,
    path_vocab,
    query_vocab,
    max_path_len: int,
    max_query_len: int,
    neg_per_pos: int,
    seed: int,
    hard_negatives_by_query: Optional[Dict[Tuple[str, ...], List[List[str]]]] = None,
    near_miss_repeat: int = 2,
    prefix_neg_per_pos: int = 0,
    prefix_neg_repeat: int = 1,
    add_terminal_truncation_negatives: bool = False,
    terminal_neg_repeat: int = 1,
    terminal_stage_prefix: str = "FOLLOWUP::",
    terminal_margin_gamma: float = 0.3,
    default_margin: float = 0.2,
):
    rng = random.Random(seed)
    all_rel = sorted(list({r for row in rows for r in row.get("oracle_path", []) if isinstance(r, str)}))

    def _prefix_nearmiss(pos_tokens: List[str], count: int) -> List[List[str]]:
        out: List[List[str]] = []
        if len(pos_tokens) < 2 or not all_rel:
            return out
        for _ in range(max(0, count)):
            cut = rng.randrange(1, len(pos_tokens))  # keep at least one correct prefix token
            neg = list(pos_tokens[:cut])
            gold_next = pos_tokens[cut] if cut < len(pos_tokens) else None
            choices = [r for r in all_rel if r != gold_next] or all_rel
            wrong = rng.choice(choices)
            neg.append(wrong)
            while len(neg) < len(pos_tokens):
                neg.append(rng.choice(all_rel))
            if neg != pos_tokens:
                out.append(neg[: len(pos_tokens)])
        return out

    samples = []

    def _terminal_truncation_neg(pos_tokens: List[str]) -> List[str]:
        if len(pos_tokens) < 2:
            return []
        tail = pos_tokens[-1]
        if isinstance(tail, str) and tail.startswith(terminal_stage_prefix):
            return list(pos_tokens[:-1])
        return []
    for row in rows:
        q = query_vocab.encode(row.get("query_tokens", []), add_bos_eos=False, max_len=max_query_len)
        pos_tokens = row.get("oracle_path", [])
        pos = path_vocab.encode(pos_tokens, add_bos_eos=False, max_len=max_path_len)
        if not q or not pos:
            continue
        for _ in range(max(1, neg_per_pos)):
            neg_tokens = list(pos_tokens)
            if neg_tokens and all_rel:
                j = rng.randrange(len(neg_tokens))
                old = neg_tokens[j]
                repl = rng.choice(all_rel)
                if len(all_rel) > 1 and repl == old:
                    repl = all_rel[(all_rel.index(repl) + 1) % len(all_rel)]
                neg_tokens[j] = repl
            neg = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
            if not neg:
                neg = [path_vocab.stoi[PAD]]
            samples.append((q, pos, neg, float(default_margin)))
        # Process-supervision style negatives: correct prefix, wrong continuation.
        for neg_tokens in _prefix_nearmiss(pos_tokens, prefix_neg_per_pos):
            neg = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
            if not neg:
                continue
            for _ in range(max(1, prefix_neg_repeat)):
                samples.append((q, pos, neg, float(default_margin)))
        if hard_negatives_by_query:
            hard_paths = hard_negatives_by_query.get(_query_key(row.get("query_tokens", [])), [])
            oracle_first = pos_tokens[0] if pos_tokens else None
            for hp in hard_paths:
                neg = path_vocab.encode(hp, add_bos_eos=False, max_len=max_path_len)
                if not neg:
                    continue
                rep = 1
                if oracle_first is not None and hp and hp[0] == oracle_first:
                    rep = max(1, near_miss_repeat)
                for _ in range(rep):
                    samples.append((q, pos, neg, float(default_margin)))
        if add_terminal_truncation_negatives:
            tneg_tokens = _terminal_truncation_neg(pos_tokens)
            if tneg_tokens:
                tneg = path_vocab.encode(tneg_tokens, add_bos_eos=False, max_len=max_path_len)
                if tneg:
                    for _ in range(max(1, terminal_neg_repeat)):
                        samples.append((q, pos, tneg, float(terminal_margin_gamma)))
    return samples


def _collate_pairwise(batch, q_pad: int, p_pad: int):
    q = [x[0] for x in batch]
    pos = [x[1] for x in batch]
    neg = [x[2] for x in batch]
    margins = torch.tensor([float(x[3]) for x in batch], dtype=torch.float32)
    q_ids, _ = pad_2d(q, q_pad)
    pos_ids, _ = pad_2d(pos, p_pad)
    neg_ids, _ = pad_2d(neg, p_pad)
    return q_ids, pos_ids, neg_ids, margins


def _build_listwise_samples(
    rows,
    path_vocab,
    query_vocab,
    max_path_len: int,
    max_query_len: int,
    neg_per_pos: int,
    seed: int,
    num_negatives: int,
    hard_negatives_by_query: Optional[Dict[Tuple[str, ...], List[List[str]]]] = None,
    near_miss_repeat: int = 2,
    prefix_neg_per_pos: int = 0,
    prefix_neg_repeat: int = 1,
    add_terminal_truncation_negatives: bool = False,
    terminal_neg_repeat: int = 1,
    terminal_stage_prefix: str = "FOLLOWUP::",
):
    rng = random.Random(seed)
    all_rel = sorted(list({r for row in rows for r in row.get("oracle_path", []) if isinstance(r, str)}))
    samples = []

    def _random_neg(pos_tokens: List[str]) -> List[str]:
        neg_tokens = list(pos_tokens)
        if neg_tokens and all_rel:
            j = rng.randrange(len(neg_tokens))
            old = neg_tokens[j]
            repl = rng.choice(all_rel)
            if len(all_rel) > 1 and repl == old:
                repl = all_rel[(all_rel.index(repl) + 1) % len(all_rel)]
            neg_tokens[j] = repl
        return neg_tokens

    def _prefix_nearmiss(pos_tokens: List[str], count: int) -> List[List[str]]:
        out: List[List[str]] = []
        if len(pos_tokens) < 2 or not all_rel:
            return out
        for _ in range(max(0, count)):
            cut = rng.randrange(1, len(pos_tokens))
            neg = list(pos_tokens[:cut])
            gold_next = pos_tokens[cut] if cut < len(pos_tokens) else None
            choices = [r for r in all_rel if r != gold_next] or all_rel
            wrong = rng.choice(choices)
            neg.append(wrong)
            while len(neg) < len(pos_tokens):
                neg.append(rng.choice(all_rel))
            if neg != pos_tokens:
                out.append(neg[: len(pos_tokens)])
        return out

    def _terminal_truncation_neg(pos_tokens: List[str]) -> List[str]:
        if len(pos_tokens) < 2:
            return []
        tail = pos_tokens[-1]
        if isinstance(tail, str) and tail.startswith(terminal_stage_prefix):
            return list(pos_tokens[:-1])
        return []

    for row in rows:
        q_tokens = row.get("query_tokens", [])
        pos_tokens = row.get("oracle_path", [])
        q = query_vocab.encode(q_tokens, add_bos_eos=False, max_len=max_query_len)
        pos = path_vocab.encode(pos_tokens, add_bos_eos=False, max_len=max_path_len)
        if not q or not pos:
            continue
        neg_paths: List[List[int]] = []
        terminal_flags: List[bool] = []
        seen = set()
        if add_terminal_truncation_negatives:
            tneg_tokens = _terminal_truncation_neg(pos_tokens)
            if tneg_tokens:
                tenc = path_vocab.encode(tneg_tokens, add_bos_eos=False, max_len=max_path_len)
                if tenc:
                    tkey = tuple(tenc)
                    if tkey not in seen:
                        seen.add(tkey)
                        for _ in range(max(1, terminal_neg_repeat)):
                            neg_paths.append(tenc)
                            terminal_flags.append(True)
        oracle_first = pos_tokens[0] if pos_tokens else None
        # Add process-supervision style prefix negatives first.
        for neg_tokens in _prefix_nearmiss(pos_tokens, prefix_neg_per_pos):
            enc = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
            if not enc:
                continue
            key = tuple(enc)
            if key in seen:
                continue
            seen.add(key)
            for _ in range(max(1, prefix_neg_repeat)):
                neg_paths.append(enc)
                terminal_flags.append(False)
                if len(neg_paths) >= num_negatives:
                    break
            if len(neg_paths) >= num_negatives:
                break
        if hard_negatives_by_query:
            hard_paths = hard_negatives_by_query.get(_query_key(q_tokens), [])
            for hp in hard_paths:
                enc = path_vocab.encode(hp, add_bos_eos=False, max_len=max_path_len)
                if not enc:
                    continue
                key = tuple(enc)
                if key in seen or hp == pos_tokens:
                    continue
                seen.add(key)
                rep = 1
                if oracle_first is not None and hp and hp[0] == oracle_first:
                    rep = max(1, near_miss_repeat)
                for _ in range(rep):
                    neg_paths.append(enc)
                    terminal_flags.append(False)
                    if len(neg_paths) >= num_negatives:
                        break
                if len(neg_paths) >= num_negatives:
                    break
        tries = 0
        max_tries = max(10, num_negatives * 6)
        while len(neg_paths) < num_negatives and tries < max_tries:
            tries += 1
            neg_tokens = _random_neg(pos_tokens)
            enc = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
            if not enc:
                continue
            key = tuple(enc)
            if key in seen or neg_tokens == pos_tokens:
                continue
            seen.add(key)
            neg_paths.append(enc)
            terminal_flags.append(False)
        if not neg_paths:
            continue
        if len(neg_paths) > num_negatives:
            neg_paths = neg_paths[:num_negatives]
            terminal_flags = terminal_flags[:num_negatives]
        if len(terminal_flags) < len(neg_paths):
            terminal_flags.extend([False] * (len(neg_paths) - len(terminal_flags)))
        samples.append((q, pos, neg_paths, terminal_flags))
    return samples


def _build_full_pool_listwise_samples(
    pool_rows,
    path_vocab,
    query_vocab,
    max_path_len: int,
    max_query_len: int,
    num_negatives: int,
    seed: int,
    add_terminal_truncation_negatives: bool = False,
    terminal_neg_repeat: int = 1,
    terminal_stage_prefix: str = "FOLLOWUP::",
):
    rng = random.Random(seed)
    all_rel = sorted(
        list(
            {
                r
                for row in pool_rows
                for r in (
                    row.get("gold_path")
                    or row.get("oracle_path")
                    or []
                )
                if isinstance(r, str)
            }
        )
    )

    def _mutate_neg(pos_tokens: List[str]) -> List[str]:
        out = list(pos_tokens)
        if out and all_rel:
            j = rng.randrange(len(out))
            old = out[j]
            repl = rng.choice(all_rel)
            if len(all_rel) > 1 and repl == old:
                repl = all_rel[(all_rel.index(repl) + 1) % len(all_rel)]
            out[j] = repl
        return out

    samples = []
    for row in pool_rows:
        q_tokens = row.get("query_tokens", [])
        pos_tokens = row.get("gold_path") or row.get("oracle_path") or []
        if not isinstance(q_tokens, list) or not isinstance(pos_tokens, list) or not pos_tokens:
            continue
        q = query_vocab.encode(q_tokens, add_bos_eos=False, max_len=max_query_len)
        pos = path_vocab.encode(pos_tokens, add_bos_eos=False, max_len=max_path_len)
        if not q or not pos:
            continue

        raw_cands = row.get("candidates")
        if not isinstance(raw_cands, list):
            raw_cands = row.get("neg_candidates")
        if not isinstance(raw_cands, list):
            raw_cands = row.get("candidate_pool_top")

        neg_paths: List[List[int]] = []
        seen = set()
        if add_terminal_truncation_negatives and len(pos_tokens) >= 2:
            tail = pos_tokens[-1]
            if isinstance(tail, str) and tail.startswith(terminal_stage_prefix):
                t_tokens = list(pos_tokens[:-1])
                t_enc = path_vocab.encode(t_tokens, add_bos_eos=False, max_len=max_path_len)
                if t_enc:
                    key = tuple(t_enc)
                    if key not in seen:
                        seen.add(key)
                        for _ in range(max(1, terminal_neg_repeat)):
                            neg_paths.append(t_enc)
        for item in (raw_cands or []):
            cand = item
            if isinstance(item, dict):
                cand = item.get("path")
            if not isinstance(cand, list) or not cand:
                continue
            if cand == pos_tokens:
                continue
            enc = path_vocab.encode(cand, add_bos_eos=False, max_len=max_path_len)
            if not enc:
                continue
            key = tuple(enc)
            if key in seen:
                continue
            seen.add(key)
            neg_paths.append(enc)
            if len(neg_paths) >= num_negatives:
                break

        tries = 0
        max_tries = max(20, num_negatives * 8)
        while len(neg_paths) < num_negatives and tries < max_tries:
            tries += 1
            neg_tokens = _mutate_neg(pos_tokens)
            if neg_tokens == pos_tokens:
                continue
            enc = path_vocab.encode(neg_tokens, add_bos_eos=False, max_len=max_path_len)
            if not enc:
                continue
            key = tuple(enc)
            if key in seen:
                continue
            seen.add(key)
            neg_paths.append(enc)

        if not neg_paths:
            continue
        if len(neg_paths) > num_negatives:
            neg_paths = neg_paths[:num_negatives]
        samples.append((q, pos, neg_paths))
    return samples


def _collate_listwise(batch, q_pad: int, p_pad: int):
    q = [x[0] for x in batch]
    pos = [x[1] for x in batch]
    neg_lists = [x[2] for x in batch]
    terminal_flags = [x[3] if len(x) > 3 else [False] * len(x[2]) for x in batch]
    q_ids, _ = pad_2d(q, q_pad)
    pos_ids, _ = pad_2d(pos, p_pad)
    max_k = max(len(x) for x in neg_lists) if neg_lists else 1
    flat_neg = []
    neg_mask = torch.zeros((len(batch), max_k), dtype=torch.bool)
    terminal_mask = torch.zeros((len(batch), max_k), dtype=torch.bool)
    for i, nlist in enumerate(neg_lists):
        for j in range(max_k):
            if j < len(nlist):
                flat_neg.append(nlist[j])
                neg_mask[i, j] = True
                if j < len(terminal_flags[i]) and bool(terminal_flags[i][j]):
                    terminal_mask[i, j] = True
            else:
                flat_neg.append([p_pad])
    neg_ids_flat, _ = pad_2d(flat_neg, p_pad)
    neg_ids = neg_ids_flat.view(len(batch), max_k, -1)
    return q_ids, pos_ids, neg_ids, neg_mask, terminal_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/value_torch_kgqa.yaml")
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/value_kgqa_torch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rows = load_jsonl(cfg["train_path"])
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    path_vocab = load_vocab(planner_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])
    training_mode = str(cfg.get("training_mode", "bce")).lower()
    if training_mode not in {"bce", "pairwise", "infonce", "full_pool_listwise"}:
        raise ValueError(
            f"Unsupported training_mode={training_mode}. Expected bce|pairwise|infonce|full_pool_listwise."
        )

    if training_mode == "bce":
        dataset = ValueDataset(
            rows=rows,
            path_vocab=path_vocab,
            query_vocab=query_vocab,
            max_path_len=int(cfg.get("max_path_len", 8)),
            max_query_len=int(cfg.get("max_query_len", 24)),
            neg_per_pos=int(cfg.get("neg_per_pos", 2)),
            seed=int(cfg.get("seed", 42)),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.get("batch_size", 128)),
            shuffle=True,
            collate_fn=lambda b: collate_value(b, query_vocab.stoi[PAD], path_vocab.stoi[PAD]),
        )
        n_samples = len(dataset)
    else:
        terminal_margin_weight = float(cfg.get("terminal_margin_weight", 1.0))
        terminal_margin_gamma = float(cfg.get("terminal_margin_gamma", 0.3))
        add_terminal_truncation_negatives = bool(cfg.get("add_terminal_truncation_negatives", False))
        terminal_neg_repeat = int(cfg.get("terminal_neg_repeat", 1))
        terminal_stage_prefix = str(cfg.get("terminal_stage_prefix", "FOLLOWUP::"))
        hard_negatives_by_query = None
        hard_neg_path = str(cfg.get("hard_negative_predictions_path", "")).strip()
        if hard_neg_path:
            pred_rows = load_jsonl(hard_neg_path)
            hard_negatives_by_query = _build_hard_negative_map(
                pred_rows=pred_rows,
                max_per_query=int(cfg.get("hard_neg_per_query", 2)),
                max_from_candidate_pool=int(cfg.get("hard_neg_from_candidate_pool", 2)),
                use_ranking_error_only=bool(cfg.get("hard_neg_ranking_error_only", True)),
                prefer_nearmiss_firsthop=bool(cfg.get("hard_neg_prefer_nearmiss_firsthop", True)),
            )
            print(
                f"[value] hard negatives enabled from {hard_neg_path}, "
                f"queries={len(hard_negatives_by_query)}"
            )
        if training_mode == "pairwise":
            dataset = _build_pairwise_samples(
                rows=rows,
                path_vocab=path_vocab,
                query_vocab=query_vocab,
                max_path_len=int(cfg.get("max_path_len", 8)),
                max_query_len=int(cfg.get("max_query_len", 24)),
                neg_per_pos=int(cfg.get("neg_per_pos", 2)),
                seed=int(cfg.get("seed", 42)),
                hard_negatives_by_query=hard_negatives_by_query,
                near_miss_repeat=int(cfg.get("hard_neg_nearmiss_repeat", 2)),
                prefix_neg_per_pos=int(cfg.get("prefix_neg_per_pos", 0)),
                prefix_neg_repeat=int(cfg.get("prefix_neg_repeat", 1)),
                add_terminal_truncation_negatives=add_terminal_truncation_negatives,
                terminal_neg_repeat=terminal_neg_repeat,
                terminal_stage_prefix=terminal_stage_prefix,
                terminal_margin_gamma=terminal_margin_gamma,
                default_margin=ranking_margin,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(cfg.get("batch_size", 128)),
                shuffle=True,
                collate_fn=lambda b: _collate_pairwise(b, query_vocab.stoi[PAD], path_vocab.stoi[PAD]),
            )
        elif training_mode == "infonce":
            dataset = _build_listwise_samples(
                rows=rows,
                path_vocab=path_vocab,
                query_vocab=query_vocab,
                max_path_len=int(cfg.get("max_path_len", 8)),
                max_query_len=int(cfg.get("max_query_len", 24)),
                neg_per_pos=int(cfg.get("neg_per_pos", 2)),
                seed=int(cfg.get("seed", 42)),
                num_negatives=int(cfg.get("infonce_num_negatives", 8)),
                hard_negatives_by_query=hard_negatives_by_query,
                near_miss_repeat=int(cfg.get("hard_neg_nearmiss_repeat", 2)),
                prefix_neg_per_pos=int(cfg.get("prefix_neg_per_pos", 0)),
                prefix_neg_repeat=int(cfg.get("prefix_neg_repeat", 1)),
                add_terminal_truncation_negatives=add_terminal_truncation_negatives,
                terminal_neg_repeat=terminal_neg_repeat,
                terminal_stage_prefix=terminal_stage_prefix,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(cfg.get("batch_size", 128)),
                shuffle=True,
                collate_fn=lambda b: _collate_listwise(b, query_vocab.stoi[PAD], path_vocab.stoi[PAD]),
            )
        else:
            full_pool_path = str(cfg.get("full_pool_candidates_path", "")).strip()
            if not full_pool_path:
                raise ValueError("training_mode=full_pool_listwise requires full_pool_candidates_path in config.")
            pool_rows = load_jsonl(full_pool_path)
            dataset = _build_full_pool_listwise_samples(
                pool_rows=pool_rows,
                path_vocab=path_vocab,
                query_vocab=query_vocab,
                max_path_len=int(cfg.get("max_path_len", 8)),
                max_query_len=int(cfg.get("max_query_len", 24)),
                num_negatives=int(cfg.get("full_pool_num_negatives", 31)),
                seed=int(cfg.get("seed", 42)),
                add_terminal_truncation_negatives=add_terminal_truncation_negatives,
                terminal_neg_repeat=terminal_neg_repeat,
                terminal_stage_prefix=terminal_stage_prefix,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(cfg.get("batch_size", 128)),
                shuffle=True,
                collate_fn=lambda b: _collate_listwise(b, query_vocab.stoi[PAD], path_vocab.stoi[PAD]),
            )
            print(
                f"[value] full_pool_listwise enabled, pool_rows={len(pool_rows)}, "
                f"samples={len(dataset)}, negatives_per_sample={int(cfg.get('full_pool_num_negatives', 31))}"
            )
        n_samples = len(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")
    model = ValueRanker(
        q_vocab_size=len(query_vocab.itos),
        p_vocab_size=len(path_vocab.itos),
        emb_dim=int(cfg.get("emb_dim", 128)),
        q_pad_id=query_vocab.stoi[PAD],
        p_pad_id=path_vocab.stoi[PAD],
        architecture=str(cfg.get("value_architecture", "legacy")),
        hidden_dim=int(cfg.get("value_hidden_dim", 256)),
        dropout=float(cfg.get("value_dropout", 0.1)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)))

    best_loss = float("inf")
    best_state = None
    epochs = int(cfg.get("epochs", 8))
    ranking_margin = float(cfg.get("ranking_margin", 0.2))
    infonce_temperature = float(cfg.get("infonce_temperature", 0.2))
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        total_gap = 0.0
        count = 0
        for batch in loader:
            if training_mode == "bce":
                q_ids, p_ids, y = batch
                q_ids = q_ids.to(device)
                p_ids = p_ids.to(device)
                y = y.to(device)
                logits = model(q_ids, p_ids)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                gap = torch.tensor(0.0, device=device)
            else:
                if training_mode == "pairwise":
                    q_ids, pos_ids, neg_ids, margins = batch
                    q_ids = q_ids.to(device)
                    pos_ids = pos_ids.to(device)
                    neg_ids = neg_ids.to(device)
                    margins = margins.to(device)
                    s_pos = model(q_ids, pos_ids)
                    s_neg = model(q_ids, neg_ids)
                    loss = torch.relu(margins - (s_pos - s_neg)).mean()
                    gap = (s_pos - s_neg).mean()
                else:
                    q_ids, pos_ids, neg_ids, neg_mask, terminal_mask = batch
                    q_ids = q_ids.to(device)
                    pos_ids = pos_ids.to(device)
                    neg_ids = neg_ids.to(device)
                    neg_mask = neg_mask.to(device)
                    terminal_mask = terminal_mask.to(device)
                    s_pos = model(q_ids, pos_ids)  # [B]
                    bsz, k, plen = neg_ids.shape
                    q_rep = q_ids.unsqueeze(1).expand(-1, k, -1).reshape(bsz * k, -1)
                    neg_flat = neg_ids.reshape(bsz * k, plen)
                    s_neg = model(q_rep, neg_flat).view(bsz, k)  # [B, K]
                    s_neg = s_neg.masked_fill(~neg_mask, -1e9)
                    logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1) / max(1e-6, infonce_temperature)
                    target = torch.zeros((bsz,), dtype=torch.long, device=device)
                    loss = F.cross_entropy(logits, target)
                    if terminal_margin_weight > 0.0 and terminal_mask.any():
                        terminal_neg = s_neg.masked_fill(~terminal_mask, -1e9).max(dim=1).values
                        has_terminal = terminal_mask.any(dim=1).float()
                        terminal_hinge = torch.relu(terminal_margin_gamma - (s_pos - terminal_neg)) * has_terminal
                        loss = loss + terminal_margin_weight * terminal_hinge.mean()
                    hardest_neg = s_neg.max(dim=1).values
                    gap = (s_pos - hardest_neg).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            total_gap += float(gap.item())
            count += 1
        avg = total / max(1, count)
        avg_gap = total_gap / max(1, count)
        if training_mode in {"pairwise", "infonce"}:
            print(f"[value] epoch {ep}/{epochs} mode={training_mode} loss={avg:.4f} pair_gap={avg_gap:.4f}")
        else:
            print(f"[value] epoch {ep}/{epochs} loss={avg:.4f}")
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
            "q_pad_id": query_vocab.stoi[PAD],
            "p_pad_id": path_vocab.stoi[PAD],
            "architecture": str(cfg.get("value_architecture", "legacy")),
            "hidden_dim": int(cfg.get("value_hidden_dim", 256)),
            "dropout": float(cfg.get("value_dropout", 0.1)),
            "training_mode": training_mode,
            "ranking_margin": ranking_margin,
            "infonce_temperature": infonce_temperature,
            "terminal_margin_gamma": float(cfg.get("terminal_margin_gamma", 0.3)),
            "terminal_margin_weight": float(cfg.get("terminal_margin_weight", 1.0)),
            "add_terminal_truncation_negatives": bool(cfg.get("add_terminal_truncation_negatives", False)),
            "terminal_neg_repeat": int(cfg.get("terminal_neg_repeat", 1)),
            "terminal_stage_prefix": str(cfg.get("terminal_stage_prefix", "FOLLOWUP::")),
        },
        "query_vocab": planner_ckpt["query_vocab"],
        "path_vocab": planner_ckpt["path_vocab"],
        "train_info": {"samples": n_samples, "best_loss": best_loss, "training_mode": training_mode},
    }
    out_path = Path(args.out) / "best.pt"
    torch.save(ckpt, out_path)
    print(f"Saved torch value model to {out_path}")


if __name__ == "__main__":
    main()
