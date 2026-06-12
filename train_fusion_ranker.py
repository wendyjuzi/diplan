"""Train an entity-aware learned fusion head over parallel KGQA scorer signals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.diplan.candidate_diffusion import load_candidate_diffusion, score_candidate_relations
from src.diplan.fusion_ranker import FEATURE_NAMES, FusionRanker, normalize_features, z_norm
from src.diplan.inference import sample_plan_candidates, score_candidates_with_value
from src.diplan.io_utils import load_jsonl
from src.diplan.kg_env import KGEnv
from src.diplan.planners import load_diffusion_bundle
from src.diplan.relation_scorer import load_relation_scorer, question_text_tokens, score_relations
from src.diplan.torch_pipeline import set_seed
from src.diplan.trajectory_diffusion import load_trajectory_diffusion, score_first_relations


def _overlap_score(relation: str, question: str) -> float:
    q = {t.lower() for t in question.replace("?", " ").replace(",", " ").split() if t}
    r = {t.lower() for t in relation.replace(".", " ").replace("_", " ").split() if t}
    return len(q & r) / max(1, len(r))


def build_examples(
    rows,
    relation_bundle,
    candidate_bundle,
    trajectory_bundle,
    diffusion_bundle,
    max_candidates: int,
    guidance_scale: float,
    trajectory_guidance_scale: float,
    prior_num_samples: int,
    prior_diffusion_steps: int,
):
    features = []
    labels = []
    groups = []
    for row in rows:
        oracle = list(row.get("oracle_path", []) or [])
        if not oracle:
            continue
        question = str(row.get("question", ""))
        query_tokens = list(row.get("query_tokens") or question_text_tokens(question))
        max_steps = int((row.get("constraints") or {}).get("max_steps", max(1, len(oracle))))
        env = KGEnv.from_rog_row(row, max_steps=max_steps)
        state = env.reset()
        prefix = []
        for step, gold in enumerate(oracle):
            candidates = env.admissible_relations(state)
            if gold not in candidates:
                break
            if len(candidates) > max_candidates:
                lexical = sorted(candidates, key=lambda r: (-_overlap_score(r, question), r))
                keep = [gold] + [r for r in lexical if r != gold][: max_candidates - 1]
                candidates = sorted(set(keep))
            base = [_overlap_score(r, question) for r in candidates]
            question_scores = score_relations(
                relation_bundle,
                question,
                query_tokens,
                candidates,
                executed_prefix=prefix,
            ) if relation_bundle else [0.0 for _ in candidates]
            candiff_scores = score_candidate_relations(
                candidate_bundle,
                question,
                query_tokens,
                candidates,
                executed_prefix=prefix,
                guidance_scale=guidance_scale,
            ) if candidate_bundle else [0.0 for _ in candidates]
            trajectory_scores = score_first_relations(
                trajectory_bundle,
                question,
                query_tokens,
                candidates,
                executed_prefix=prefix,
                guidance_scale=trajectory_guidance_scale,
            ) if trajectory_bundle else [0.0 for _ in candidates]
            value_scores = score_candidates_with_value(
                diffusion_bundle.value_model,
                diffusion_bundle.path_vocab,
                diffusion_bundle.query_vocab,
                query_tokens,
                [prefix + [r] for r in candidates],
                diffusion_bundle.device,
                max_query_len=32,
                max_path_len=30,
            ) if diffusion_bundle and diffusion_bundle.value_model is not None else [0.0 for _ in candidates]
            prior_scores = [0.0 for _ in candidates]
            guided_scores = [0.0 for _ in candidates]
            if diffusion_bundle and prior_num_samples > 0:
                remaining = max(1, max_steps - step)
                sampled_paths = sample_plan_candidates(
                    planner=diffusion_bundle.planner,
                    autoencoder=diffusion_bundle.autoencoder,
                    path_vocab=diffusion_bundle.path_vocab,
                    query_vocab=diffusion_bundle.query_vocab,
                    query_tokens=query_tokens,
                    num_candidates=prior_num_samples,
                    diffusion_steps=prior_diffusion_steps,
                    max_path_len=remaining,
                    device=diffusion_bundle.device,
                    executed_prefix=prefix,
                    use_prefix=bool(getattr(diffusion_bundle, "use_prefix", True)),
                    latent_mean=getattr(diffusion_bundle, "latent_mean", None),
                    latent_std=getattr(diffusion_bundle, "latent_std", None),
                    prediction_target=str(getattr(diffusion_bundle, "prediction_target", "z0")),
                    planner_type=str(getattr(diffusion_bundle, "planner_type", "diffusion")),
                    jitter_std=float(getattr(diffusion_bundle, "jitter_std", 0.0)),
                )
                cand_to_idx = {r: i for i, r in enumerate(candidates)}
                sampled_by_first = {r: [] for r in candidates}
                sampled_full_paths = []
                sampled_firsts = []
                for path in sampled_paths:
                    if not path:
                        continue
                    first = path[0]
                    if first not in cand_to_idx:
                        continue
                    sampled_by_first[first].append(path)
                    sampled_firsts.append(first)
                    sampled_full_paths.append(prefix + list(path))
                denom_samples = max(1, len(sampled_paths))
                for rel, paths in sampled_by_first.items():
                    prior_scores[cand_to_idx[rel]] = len(paths) / denom_samples
                if sampled_full_paths and diffusion_bundle.value_model is not None:
                    sampled_values = score_candidates_with_value(
                        diffusion_bundle.value_model,
                        diffusion_bundle.path_vocab,
                        diffusion_bundle.query_vocab,
                        query_tokens,
                        sampled_full_paths,
                        diffusion_bundle.device,
                        max_query_len=32,
                        max_path_len=30,
                    )
                    best_by_first = {}
                    for first, value in zip(sampled_firsts, sampled_values):
                        best_by_first[first] = max(float(value), float(best_by_first.get(first, -1e18)))
                    for rel, value in best_by_first.items():
                        guided_scores[cand_to_idx[rel]] = float(value)

            base_z = z_norm(base)
            value_z = z_norm(value_scores)
            question_z = z_norm(question_scores)
            candiff_z = z_norm(candiff_scores)
            trajectory_z = z_norm(trajectory_scores)
            prior_z = z_norm(prior_scores)
            guided_z = z_norm(guided_scores)
            rank_order = sorted(range(len(candidates)), key=lambda i: base[i], reverse=True)
            rank_frac = [0.0 for _ in candidates]
            denom = max(1, len(candidates) - 1)
            for rank, idx in enumerate(rank_order):
                rank_frac[idx] = rank / denom
            group_start = len(labels)
            for i, rel in enumerate(candidates):
                nxt = env.neighbors(state, rel)
                feat = [
                    float(base_z[i]),
                    float(value_z[i]),
                    float(question_z[i]),
                    float(candiff_z[i]),
                    float(trajectory_z[i]),
                    float(prior_z[i]),
                    float(guided_z[i]),
                    math.log1p(len(nxt)),
                    float(rank_frac[i]),
                    step / max(1, max_steps - 1),
                    1.0 if nxt else 0.0,
                ]
                features.append(feat)
                labels.append(1.0 if rel == gold else 0.0)
            groups.append((group_start, len(labels), candidates.index(gold)))
            state = env.step(state, gold)
            prefix.append(gold)
            if env.is_terminal(state):
                break
    return features, labels, groups


@torch.no_grad()
def evaluate(model, x, y, groups, mean, std, device):
    model.eval()
    logits = model(normalize_features(x.to(device), mean.to(device), std.to(device))).detach().cpu()
    ranks = []
    for start, end, gold_idx in groups:
        scores = logits[start:end]
        order = scores.argsort(descending=True).tolist()
        ranks.append(order.index(gold_idx) + 1)
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits, y).item()),
        "mrr": sum(1.0 / r for r in ranks) / max(1, len(ranks)),
        "recall@1": sum(1 for r in ranks if r <= 1) / max(1, len(ranks)),
        "recall@3": sum(1 for r in ranks if r <= 3) / max(1, len(ranks)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--valid_path", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation_scorer_ckpt", default="")
    ap.add_argument("--candidate_diffusion_ckpt", default="")
    ap.add_argument("--candidate_guidance_scale", type=float, default=1.0)
    ap.add_argument("--trajectory_diffusion_ckpt", default="")
    ap.add_argument("--trajectory_guidance_scale", type=float, default=1.0)
    ap.add_argument("--prior_num_samples", type=int, default=32)
    ap.add_argument("--prior_diffusion_steps", type=int, default=20)
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--planner_ckpt", required=True)
    ap.add_argument("--value_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_candidates", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # Ranking objective. "listwise" optimizes the order of candidates within a
    # decision step (ListNet top-1 softmax CE with the gold candidate as target),
    # which is what the recall@1 / MRR eval measures. "pointwise" is the legacy
    # per-candidate BCE (optimizes calibration, not order). "hybrid" combines both.
    ap.add_argument("--loss", type=str, default="hybrid", choices=["listwise", "pointwise", "hybrid"])
    ap.add_argument("--listwise_weight", type=float, default=1.0)
    ap.add_argument("--pointwise_weight", type=float, default=0.2)
    ap.add_argument("--group_batch_size", type=int, default=64)
    args = ap.parse_args()

    set_seed(args.seed)
    train_rows = load_jsonl(args.train_path)
    valid_rows = load_jsonl(args.valid_path) if args.valid_path else []
    relation_bundle = load_relation_scorer(args.relation_scorer_ckpt) if args.relation_scorer_ckpt else None
    candidate_bundle = load_candidate_diffusion(args.candidate_diffusion_ckpt) if args.candidate_diffusion_ckpt else None
    trajectory_bundle = load_trajectory_diffusion(args.trajectory_diffusion_ckpt) if args.trajectory_diffusion_ckpt else None
    diffusion_bundle = load_diffusion_bundle(args.ae_ckpt, args.planner_ckpt, args.value_ckpt, {"diffusion": {}})

    train_x, train_y, train_groups = build_examples(
        train_rows, relation_bundle, candidate_bundle, trajectory_bundle, diffusion_bundle,
        args.max_candidates, args.candidate_guidance_scale, args.trajectory_guidance_scale,
        args.prior_num_samples, args.prior_diffusion_steps
    )
    valid_x, valid_y, valid_groups = build_examples(
        valid_rows, relation_bundle, candidate_bundle, trajectory_bundle, diffusion_bundle,
        args.max_candidates, args.candidate_guidance_scale, args.trajectory_guidance_scale,
        args.prior_num_samples, args.prior_diffusion_steps
    ) if valid_rows else ([], [], [])
    if not train_x:
        raise ValueError("No fusion training examples were found")

    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    mean = x.mean(0)
    std = x.std(0).clamp(min=1e-6)
    vx = torch.tensor(valid_x, dtype=torch.float32) if valid_x else x
    vy = torch.tensor(valid_y, dtype=torch.float32) if valid_y else y
    vgroups = valid_groups if valid_groups else train_groups

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionRanker(len(FEATURE_NAMES), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([(len(train_y) - sum(train_y)) / max(1.0, sum(train_y))], dtype=torch.float32, device=device)

    # Pre-normalize once with the same statistics used at eval/inference time so
    # listwise softmax operates on the deployed feature scale.
    xn = normalize_features(x.to(device), mean.to(device), std.to(device))
    yt = y.to(device)
    # Listwise needs intact decision-step groups (a gold among >=2 candidates);
    # singletons carry no ordering signal and are skipped for the listwise term.
    rank_groups = [(s, e, gi) for (s, e, gi) in train_groups if e - s >= 2]
    use_listwise = args.loss in ("listwise", "hybrid") and bool(rank_groups)
    use_pointwise = args.loss in ("pointwise", "hybrid")
    gbs = max(1, args.group_batch_size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    perm_seed = int(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        if use_listwise:
            gen = torch.Generator().manual_seed(perm_seed + epoch)
            order = torch.randperm(len(rank_groups), generator=gen).tolist()
            for bstart in range(0, len(order), gbs):
                batch = [rank_groups[i] for i in order[bstart:bstart + gbs]]
                opt.zero_grad()
                lw = xn.new_zeros(())
                pw = xn.new_zeros(())
                for (s, e, gold_idx) in batch:
                    logits = model(xn[s:e])
                    # ListNet top-1: softmax CE with the gold candidate as target.
                    lw = lw + F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([gold_idx], device=device),
                    )
                    if use_pointwise:
                        pw = pw + F.binary_cross_entropy_with_logits(
                            logits, yt[s:e], pos_weight=pos_weight
                        )
                n = float(len(batch))
                loss = (args.listwise_weight * lw + args.pointwise_weight * pw) / n
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        else:
            dataset = TensorDataset(xn, yt)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            for bx, by in loader:
                logits = model(bx)
                loss = F.binary_cross_entropy_with_logits(logits, by, pos_weight=pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        valid = evaluate(model, vx, vy, vgroups, mean, std, device)
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "valid_loss": valid["loss"],
            "valid_mrr": valid["mrr"],
            "valid_recall@1": valid["recall@1"],
            "valid_recall@3": valid["recall@3"],
            "train_examples": len(train_y),
            "valid_examples": len(valid_y),
        }
        print(json.dumps(row), flush=True)
        history.append(row)
        if valid["mrr"] > best:
            best = valid["mrr"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {"hidden_dim": args.hidden_dim, "dropout": args.dropout},
                    "feature_names": FEATURE_NAMES,
                    "feature_mean": mean.tolist(),
                    "feature_std": std.tolist(),
                },
                out / "best.pt",
            )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
