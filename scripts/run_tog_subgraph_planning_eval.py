"""ToG-style planning-strategy evaluation on oracle-structure local subgraphs.

This runner is for strict *methodological* reproduction of the FLARE setup when a
Freebase/Virtuoso endpoint is unavailable. It keeps the ToG substrate idea:

  question topic entities -> relation pruning -> entity expansion/pruning
  -> bounded-width KG traversal

but replaces ToG's SPARQL backend with the per-question oracle subgraphs in
``data/rog_processed/*.jsonl``. The only component varied across methods is the
planning/action-selection strategy:

  single_step | beam | lookahead | flare | diplan_diffusion

This is intentionally separate from ``run_kgqa_planning_eval.py``: that older
runner is a simpler KGEnv diagnostic. This file is the one to use when writing:
"we reproduce FLARE-style planning on top of a ToG-style oracle-subgraph
substrate, then replace explicit lookahead with DiPLaN diffusion priors."
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir, load_config
from src.diplan.inference import score_candidates_with_value
from src.diplan.kgqa_prompts import LLMScorer
from src.diplan.llm_client import LLMConfig
from src.diplan.metrics import first_error_step, recovery_at_error, trap_at_1
from src.diplan.planners import StubScorer, _relation_tokens, load_diffusion_bundle


@dataclass(frozen=True)
class EntityHyp:
    entity: str
    score: float
    path: Tuple[str, ...]


@dataclass(frozen=True)
class ToGState:
    hyps: Tuple[EntityHyp, ...]
    depth: int


@dataclass
class ToGCandidate:
    relation: str
    score: float
    next_state: ToGState
    source_entity: str
    target_entities: Tuple[str, ...]


class ToGSubgraphEnv:
    """Local-subgraph backend with ToG-style entity frontier hypotheses."""

    def __init__(self, row: dict, num_retain_entity: int = 5) -> None:
        self.row = row
        self.question = str(row["question"])
        self.query_tokens = list(row.get("query_tokens") or [])
        self.answer_entities = {str(x) for x in (row.get("a_entity") or [])}
        self.max_steps = int(row.get("constraints", {}).get("max_steps", 3))
        self.num_retain_entity = int(num_retain_entity)
        self.adj: Dict[str, Dict[str, List[str]]] = {}
        for triple in row.get("graph") or []:
            if len(triple) != 3:
                continue
            s, r, o = str(triple[0]), str(triple[1]), str(triple[2])
            self.adj.setdefault(s, {}).setdefault(r, [])
            if o not in self.adj[s][r]:
                self.adj[s][r].append(o)

    def initial_state(self) -> ToGState:
        q_entities = row_list(self.row.get("q_entity"))
        hyps = tuple(EntityHyp(str(e), 1.0, tuple()) for e in q_entities[: self.num_retain_entity])
        return ToGState(hyps=hyps, depth=0)

    def answer_reached(self, state: ToGState) -> bool:
        return any(h.entity in self.answer_entities for h in state.hyps)

    def is_terminal(self, state: ToGState) -> bool:
        return state.depth >= self.max_steps or not state.hyps or self.answer_reached(state)

    def admissible_relations_for_entity(self, entity: str) -> List[str]:
        return sorted(self.adj.get(entity, {}).keys())

    def all_admissible_relations(self, state: ToGState) -> List[str]:
        rels = set()
        for hyp in state.hyps:
            rels.update(self.admissible_relations_for_entity(hyp.entity))
        return sorted(rels)

    def reachable_entities(self, entity: str, relation: str) -> List[str]:
        return list(self.adj.get(entity, {}).get(relation, []))

    def candidate_actions(self, state: ToGState, scorer, k: int) -> List[ToGCandidate]:
        """ToG relation prune + entity expand/prune for one decision point."""
        out: List[ToGCandidate] = []
        for hyp in sorted(state.hyps, key=lambda h: -h.score)[: self.num_retain_entity]:
            admissible = self.admissible_relations_for_entity(hyp.entity)
            if not admissible:
                continue
            question_ctx = (
                f"{self.question}\n"
                f"Current entity: {hyp.entity}\n"
                f"Previously selected relations from this entity hypothesis: {list(hyp.path)}"
            )
            proposed = scorer.propose(question_ctx, hyp.path, admissible, k)
            for rel in proposed:
                tails = self.reachable_entities(hyp.entity, rel)
                if not tails:
                    continue
                rel_score = scorer.score_relation(question_ctx, hyp.path, rel, tails)
                # ToG entity pruning is approximated by keeping the top retained
                # expanded entities under the relation-level score. The answer
                # check happens before pruning loss can hide a reached answer.
                uniq_tails = sorted(dict.fromkeys(tails))
                answer_tails = [t for t in uniq_tails if t in self.answer_entities]
                other_tails = [t for t in uniq_tails if t not in self.answer_entities]
                ranked_tails = (answer_tails + other_tails)[: self.num_retain_entity]
                next_score = float(hyp.score) + float(rel_score)
                next_hyps = tuple(
                    EntityHyp(entity=t, score=next_score, path=hyp.path + (rel,))
                    for t in ranked_tails
                )
                out.append(
                    ToGCandidate(
                        relation=rel,
                        score=next_score,
                        next_state=ToGState(hyps=next_hyps, depth=state.depth + 1),
                        source_entity=hyp.entity,
                        target_entities=tuple(ranked_tails),
                    )
                )
        out.sort(key=lambda c: (-c.score, c.relation, c.source_entity))
        return out


def row_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def make_scorer(cfg):
    kind = str(cfg.get("scorer", "stub")).lower()
    if kind == "stub":
        return StubScorer(seed=int(cfg.get("seed", 42)))
    if kind == "llm":
        return LLMScorer.from_config(cfg)
    raise ValueError(f"Unknown scorer: {kind}")


class SingleStepToG:
    def __init__(self, k: int) -> None:
        self.k = k

    def select(self, env: ToGSubgraphEnv, state: ToGState, ctx) -> Optional[ToGCandidate]:
        cands = env.candidate_actions(state, ctx["scorer"], self.k)
        return cands[0] if cands else None


class BeamToG:
    def __init__(self, width: int, depth: int, k: int) -> None:
        self.width = width
        self.depth = depth
        self.k = k

    def select(self, env: ToGSubgraphEnv, state: ToGState, ctx) -> Optional[ToGCandidate]:
        roots = env.candidate_actions(state, ctx["scorer"], self.k)
        if not roots:
            return None
        beam = [(c.score, [c], c.next_state) for c in roots[: self.width]]
        for _ in range(1, min(self.depth, env.max_steps - state.depth)):
            expanded = []
            for score, seq, st in beam:
                if env.is_terminal(st):
                    expanded.append((score, seq, st))
                    continue
                for c in env.candidate_actions(st, ctx["scorer"], self.k):
                    expanded.append((score + c.score, seq + [c], c.next_state))
            if not expanded:
                break
            expanded.sort(key=lambda x: -x[0])
            beam = expanded[: self.width]
        return max(beam, key=lambda x: x[0])[1][0]


class LookaheadToG:
    def __init__(self, horizon: int, k: int) -> None:
        self.horizon = horizon
        self.k = k

    def select(self, env: ToGSubgraphEnv, state: ToGState, ctx) -> Optional[ToGCandidate]:
        roots = env.candidate_actions(state, ctx["scorer"], self.k)
        if not roots:
            return None
        best_root, best_score = None, -math.inf
        for root in roots:
            score = root.score
            st = root.next_state
            for _ in range(1, min(self.horizon, env.max_steps - state.depth)):
                if env.is_terminal(st):
                    break
                nxt = env.candidate_actions(st, ctx["scorer"], self.k)
                if not nxt:
                    break
                score += nxt[0].score
                st = nxt[0].next_state
            if score > best_score:
                best_root, best_score = root, score
        return best_root


@dataclass
class MCTSNode:
    state: ToGState
    depth_from_root: int = 0
    N: int = 0
    untried: Optional[List[ToGCandidate]] = None
    children: Dict[int, "MCTSNode"] = None
    Q: Dict[int, float] = None
    Na: Dict[int, int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.Q is None:
            self.Q = {}
        if self.Na is None:
            self.Na = {}


class FLAREToG:
    def __init__(self, S: int, c: float, k: int, H: int, rng: random.Random) -> None:
        self.S = S
        self.c = c
        self.k = k
        self.H = H
        self.rng = rng

    def _ucb(self, node: MCTSNode, idx: int) -> float:
        if node.Na.get(idx, 0) == 0:
            return math.inf
        return node.Q[idx] + self.c * math.sqrt(math.log(max(1, node.N)) / (node.Na[idx] + 1))

    def select(self, env: ToGSubgraphEnv, state: ToGState, ctx) -> Optional[ToGCandidate]:
        root = MCTSNode(state)
        for _ in range(self.S):
            self._simulate(env, root, ctx)
        if not root.Q or not root.untried:
            cands = env.candidate_actions(state, ctx["scorer"], self.k)
            return cands[0] if cands else None
        best_idx = max(root.Q, key=root.Q.get)
        return root.untried[best_idx]

    def _simulate(self, env: ToGSubgraphEnv, root: MCTSNode, ctx) -> None:
        node = root
        path: List[Tuple[MCTSNode, int, ToGCandidate]] = []
        traj: List[str] = []
        entity_trace: List[List[str]] = [[h.entity for h in root.state.hyps]]

        while not env.is_terminal(node.state) and node.depth_from_root < self.H:
            if node.untried is None:
                node.untried = env.candidate_actions(node.state, ctx["scorer"], self.k)
                for i in range(len(node.untried)):
                    node.Q.setdefault(i, 0.0)
                    node.Na.setdefault(i, 0)
            if not node.untried:
                break
            unexpanded = [i for i in range(len(node.untried)) if i not in node.children]
            if unexpanded:
                idx = self.rng.choice(unexpanded)
                cand = node.untried[idx]
                child = MCTSNode(cand.next_state, depth_from_root=node.depth_from_root + 1)
                node.children[idx] = child
                path.append((node, idx, cand))
                traj.append(cand.relation)
                entity_trace.append([h.entity for h in cand.next_state.hyps])
                node = child
                break
            idx = max(range(len(node.untried)), key=lambda i: self._ucb(node, i))
            cand = node.untried[idx]
            path.append((node, idx, cand))
            traj.append(cand.relation)
            entity_trace.append([h.entity for h in cand.next_state.hyps])
            node = node.children[idx]

        st = node.state
        while len(traj) < self.H and not env.is_terminal(st):
            cands = env.candidate_actions(st, ctx["scorer"], self.k)
            if not cands:
                break
            cand = self.rng.choice(cands[: max(1, min(self.k, len(cands)))])
            traj.append(cand.relation)
            st = cand.next_state
            entity_trace.append([h.entity for h in st.hyps])

        ret = ctx["scorer"].score_trajectory(env.question, traj, entity_trace)
        for n, idx, _ in path:
            n.N += 1
            n.Na[idx] += 1
            n.Q[idx] += (ret - n.Q[idx]) / n.Na[idx]


class DiPLaNToG:
    def __init__(self, bundle, fallback: SingleStepToG) -> None:
        self.bundle = bundle
        self.fallback = fallback
        self.grounded_hits = 0
        self.grounded_misses = 0
        self.fuzzy_hits = 0
        self.fallbacks = 0
        self.candidate_rerank_calls = 0

    @staticmethod
    def _relation_jaccard(a: str, b: str) -> float:
        at = _relation_tokens(a)
        bt = _relation_tokens(b)
        union = len(at | bt)
        return (len(at & bt) / union) if union else 0.0

    def _ground_first_steps(self, firsts: Sequence[str], admissible: Sequence[str], min_jaccard: float) -> Tuple[Dict[str, float], bool]:
        """Map sampled first relations to admissible ToG relations.

        Returns relation -> diffusion prior score. Exact matches get the highest
        confidence. Fuzzy matches are accepted only above ``min_jaccard`` so that
        out-of-domain diffusion samples do not force arbitrary ToG actions.
        """
        adm = list(admissible)
        scores: Dict[str, float] = {}
        exact = False
        for rank, rel in enumerate(firsts):
            prior = 1.0 / (rank + 1)
            if rel in adm:
                exact = True
                scores[rel] = max(scores.get(rel, 0.0), prior + 1.0)
                continue
            best_rel, best_j = None, 0.0
            for cand in adm:
                j = self._relation_jaccard(rel, cand)
                if j > best_j:
                    best_rel, best_j = cand, j
            if best_rel is not None and best_j >= min_jaccard:
                scores[best_rel] = max(scores.get(best_rel, 0.0), prior * best_j)
        return scores, exact

    def select(self, env: ToGSubgraphEnv, state: ToGState, ctx) -> Optional[ToGCandidate]:
        from src.diplan.inference import sample_plan_candidates

        cands = env.candidate_actions(state, ctx["scorer"], self.fallback.k)
        if not cands:
            return None
        admissible = {c.relation for c in cands}
        b = self.bundle
        sampled = sample_plan_candidates(
            planner=b.planner,
            autoencoder=b.autoencoder,
            path_vocab=b.path_vocab,
            query_vocab=b.query_vocab,
            query_tokens=env.query_tokens,
            num_candidates=b.num_candidates,
            diffusion_steps=b.diffusion_steps,
            max_path_len=max(1, env.max_steps - state.depth),
            device=b.device,
            executed_prefix=ctx["executed"],
            use_prefix=b.use_prefix,
            latent_mean=b.latent_mean,
            latent_std=b.latent_std,
            prediction_target=b.prediction_target,
            planner_type=b.planner_type,
            jitter_std=b.jitter_std,
        )
        firsts = [p[0] for p in sampled if p]
        min_jaccard = float(ctx.get("diplan_min_grounding_jaccard", 0.25))
        bonus_weight = float(ctx.get("diplan_grounding_bonus", 1.0))
        grounded_scores, exact = self._ground_first_steps(firsts, sorted(admissible), min_jaccard)
        if exact:
            self.grounded_hits += 1
        elif grounded_scores:
            self.grounded_misses += 1
            self.fuzzy_hits += 1
        else:
            self.grounded_misses += 1
            self.fallbacks += 1

        if bool(ctx.get("diplan_candidate_rerank", False)):
            self.candidate_rerank_calls += 1
            value_weight = float(ctx.get("diplan_value_weight", 1.0))
            candidate_paths = [list(ctx["executed"]) + [c.relation] for c in cands]
            value_scores = score_candidates_with_value(
                b.value_model,
                b.path_vocab,
                b.query_vocab,
                env.query_tokens,
                candidate_paths,
                b.device,
                max_query_len=int(ctx.get("diplan_value_max_query_len", 32)),
                max_path_len=int(ctx.get("diplan_value_max_path_len", 30)),
            )
            value_scores = _z_norm(value_scores)
            return max(
                zip(cands, value_scores),
                key=lambda item: item[0].score
                + bonus_weight * grounded_scores.get(item[0].relation, 0.0)
                + value_weight * item[1],
            )[0]

        if not grounded_scores:
            return cands[0]
        return max(cands, key=lambda c: c.score + bonus_weight * grounded_scores.get(c.relation, 0.0))


def _z_norm(xs: Sequence[float]) -> List[float]:
    vals = [float(x) for x in xs]
    if not vals:
        return []
    mu = mean(vals)
    var = mean((x - mu) ** 2 for x in vals)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0 for _ in vals]
    return [(x - mu) / std for x in vals]


def make_planner(name: str, cfg: dict, rng: random.Random, bundle=None):
    k = int(cfg.get("k", 8))
    if name == "single_step":
        return SingleStepToG(k)
    if name == "beam":
        bc = cfg.get("beam", {})
        return BeamToG(width=int(bc.get("B", 8)), depth=int(bc.get("depth", 3)), k=k)
    if name == "lookahead":
        lc = cfg.get("lookahead", {})
        return LookaheadToG(horizon=int(lc.get("k", 2)), k=k)
    if name == "flare":
        fc = cfg.get("flare", {})
        return FLAREToG(
            S=int(fc.get("S", 16)),
            c=float(fc.get("c", 1.4)),
            k=int(fc.get("k", k)),
            H=int(fc.get("H", 3)),
            rng=rng,
        )
    if name == "diplan_diffusion":
        if bundle is None:
            raise ValueError("diplan_diffusion requires --ae_ckpt and --planner_ckpt")
        return DiPLaNToG(bundle, fallback=SingleStepToG(k))
    raise ValueError(f"Unknown planner: {name}")


def stream_rows(path: Path, max_tasks: int, include_datasets: Sequence[str]):
    include = set(include_datasets or [])
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if include and row.get("dataset") not in include:
                continue
            if max_tasks > 0 and n >= max_tasks:
                break
            n += 1
            yield row


def run_episode(env: ToGSubgraphEnv, planner, scorer, rng: random.Random, planner_cfg: Optional[dict] = None):
    planner_cfg = planner_cfg or {}
    state = env.initial_state()
    executed: List[str] = []
    entity_trace = [[h.entity for h in state.hyps]]
    while not env.is_terminal(state):
        cand = planner.select(
            env,
            state,
            {
                "scorer": scorer,
                "rng": rng,
                "executed": executed,
                "diplan_min_grounding_jaccard": planner_cfg.get("diplan_min_grounding_jaccard", 0.25),
                "diplan_grounding_bonus": planner_cfg.get("diplan_grounding_bonus", 1.0),
                "diplan_candidate_rerank": planner_cfg.get("diplan_candidate_rerank", False),
                "diplan_value_weight": planner_cfg.get("diplan_value_weight", 1.0),
                "diplan_value_max_query_len": planner_cfg.get("diplan_value_max_query_len", 32),
                "diplan_value_max_path_len": planner_cfg.get("diplan_value_max_path_len", 30),
            },
        )
        if cand is None:
            break
        executed.append(cand.relation)
        state = cand.next_state
        entity_trace.append([h.entity for h in state.hyps])
    return {
        "executed_path": executed,
        "entity_trace": entity_trace,
        "answer_reached": env.answer_reached(state),
        "final_entities": [h.entity for h in state.hyps],
    }


def aggregate(records: List[dict]) -> dict:
    if not records:
        return {}
    trap_rows = [r for r in records if r["has_trap"]]
    return {
        "n": len(records),
        "hits@1": round(mean(1.0 if r["success"] else 0.0 for r in records), 4),
        "trap@1": round(mean(1.0 if r["trap_at_1"] else 0.0 for r in trap_rows), 4) if trap_rows else None,
        "first_error_step": round(mean(r["first_error_step"] for r in records), 3),
        "recovery@first_error": round(mean(1.0 if r["recovery_at_error"] else 0.0 for r in records), 4),
        "avg_steps": round(mean(r["steps"] for r in records), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ae_ckpt", default="")
    parser.add_argument("--planner_ckpt", default="")
    parser.add_argument("--value_ckpt", default="")
    parser.add_argument("--progress_every", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    methods = list(cfg.get("methods", ["single_step", "beam", "lookahead", "flare"]))
    rng = random.Random(int(cfg.get("seed", 42)))
    scorer = make_scorer(cfg)
    bundle = None
    if "diplan_diffusion" in methods:
        bundle = load_diffusion_bundle(args.ae_ckpt, args.planner_ckpt, args.value_ckpt, cfg)
    planners = {m: make_planner(m, cfg, rng, bundle) for m in methods}

    path = Path(cfg["test_path"])
    max_tasks = int(cfg.get("max_tasks", 0))
    include_datasets = cfg.get("include_datasets", [])
    progress_every = args.progress_every
    if progress_every is None:
        progress_every = int(cfg.get("progress_every", 1 if str(cfg.get("scorer", "stub")) == "llm" else 0))
    rows = list(stream_rows(path, max_tasks, include_datasets))
    print(
        json.dumps(
            {
                "event": "start",
                "runner": "tog_subgraph",
                "test_path": str(path),
                "n": len(rows),
                "methods": methods,
                "scorer": cfg.get("scorer", "stub"),
                "llm_model": cfg.get("llm_model"),
                "num_retain_entity": cfg.get("num_retain_entity", 5),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    by_method = defaultdict(list)
    predictions = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        oracle = list(row.get("oracle_path") or [])
        trap = list(row.get("trap_path") or [])
        has_trap = bool(trap and oracle and trap[0] != oracle[0])
        for m in methods:
            mt0 = time.time()
            env = ToGSubgraphEnv(row, num_retain_entity=int(cfg.get("num_retain_entity", 5)))
            out = run_episode(env, planners[m], scorer, rng, cfg)
            pred = out["executed_path"]
            rec = {
                "task_id": row.get("task_id"),
                "dataset": row.get("dataset"),
                "method": m,
                "success": out["answer_reached"],
                "first_error_step": first_error_step(pred, oracle),
                "recovery_at_error": recovery_at_error(pred, oracle),
                "trap_at_1": trap_at_1(pred, trap),
                "has_trap": has_trap,
                "steps": len(pred),
                "executed_path": pred,
                "oracle_path": oracle,
                "entity_trace": out["entity_trace"],
                "final_entities": out["final_entities"],
            }
            by_method[m].append(rec)
            predictions.append(rec)
            if progress_every and (i == 1 or i % progress_every == 0):
                client = getattr(scorer, "client", None)
                print(
                    json.dumps(
                        {
                            "event": "method_done",
                            "task": f"{i}/{len(rows)}",
                            "method": m,
                            "success": rec["success"],
                            "trap": rec["trap_at_1"],
                            "steps": rec["steps"],
                            "elapsed_s": round(time.time() - mt0, 2),
                            "llm_calls": getattr(client, "calls", None),
                            "llm_errors": getattr(client, "errors", None),
                            "fallbacks": getattr(scorer, "fallbacks", None),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if progress_every and (i == 1 or i % progress_every == 0):
            print(json.dumps({"event": "task_done", "task": f"{i}/{len(rows)}", "elapsed_s": round(time.time() - t0, 2)}), flush=True)

    summary = {m: aggregate(rs) for m, rs in by_method.items()}
    diag = {}
    client = getattr(scorer, "client", None)
    if client is not None:
        diag.update({"llm_calls": client.calls, "llm_errors": client.errors, "llm_fallbacks": getattr(scorer, "fallbacks", 0)})
    if "diplan_diffusion" in planners:
        dp = planners["diplan_diffusion"]
        tot = dp.grounded_hits + dp.grounded_misses
        diag["diffusion_grounding_hit_rate"] = round(dp.grounded_hits / max(1, tot), 4)
        diag["diffusion_fuzzy_grounding_rate"] = round(dp.fuzzy_hits / max(1, tot), 4)
        diag["diffusion_grounding_fallback_rate"] = round(dp.fallbacks / max(1, tot), 4)
        diag["diplan_candidate_rerank_calls"] = dp.candidate_rerank_calls

    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "summary_metrics.json"), summary)
    dump_json(str(Path(args.out) / "diagnostics.json"), diag)
    with (Path(args.out) / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": summary, "diag": diag}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
