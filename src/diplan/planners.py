"""Pluggable planning strategies over the KG environment.

All planners share one signature -- ``select_action(env, state, ctx) -> Optional[str]``
-- returning ONE relation to execute (receding horizon). They differ only in how that
action is chosen, which is exactly the paper's comparison axis:

  * SingleStepGreedy  -- local step-wise score only (reasoning-based policy).
  * BeamSearch        -- top-B prefixes by accumulated step-wise score (B=8).
  * ShallowLookahead  -- depth-k greedy rollout per candidate (k=2).
  * FLARE (MCTS)      -- Algorithm 1: UCB selection, action pruning, depth-H rollout,
                         trajectory-level evaluation with trajectory memory, backprop.
  * DiPLaNDiffusion   -- swaps the MCTS tree for the repo's diffusion planner; same env,
                         same evaluative signal -- only the action proposal changes.

The evaluative signal r-hat and the proposal function phi are provided by a ``Scorer``
(``StubScorer`` for offline validation, ``LLMScorer`` for the faithful LLM signal).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from .kg_env import KGEnv, KGState


# ----------------------------------------------------------------------------
# Scorer interface (phi + r-hat) and an offline lexical stub
# ----------------------------------------------------------------------------
class Scorer(Protocol):
    def propose(self, question: str, executed: Sequence[str], admissible: Sequence[str], k: int) -> List[str]:
        ...

    def score_relation(self, question: str, executed: Sequence[str], relation: str, resulting_entities: Sequence[str]) -> float:
        ...

    def score_trajectory(self, question: str, relations: Sequence[str], entity_trace: Sequence[Sequence[str]]) -> float:
        ...


def _relation_tokens(relation: str) -> set:
    return {t for t in relation.replace(".", " ").replace("_", " ").split() if t}


def _overlap(relation: str, question: str) -> float:
    q = {t for t in question.lower().replace("?", " ").replace(",", " ").split() if t}
    r = _relation_tokens(relation)
    if not r or not q:
        return 0.0
    return len(r & q) / len(r)


class StubScorer:
    """Lexical-overlap stand-in for the LLM signal (no network, for Phase-B validation)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def propose(self, question, executed, admissible, k):
        scored = sorted(admissible, key=lambda r: (-_overlap(r, question), r))
        return scored[: max(1, k)]

    def score_relation(self, question, executed, relation, resulting_entities):
        base = _overlap(relation, question)
        alive = 0.1 if resulting_entities else -0.2
        return base + alive

    def score_trajectory(self, question, relations, entity_trace):
        if not relations:
            return 0.0
        mean_ov = sum(_overlap(r, question) for r in relations) / len(relations)
        # mild reward for ending on a non-empty frontier (progress proxy)
        ended_alive = 0.1 if (entity_trace and entity_trace[-1]) else -0.1
        return mean_ov + ended_alive


# ----------------------------------------------------------------------------
# Trajectory memory (amortized r-hat for FLARE) -- paper sim>=delta, |M|<=M
# ----------------------------------------------------------------------------
class TrajectoryMemory:
    def __init__(self, cap: int = 200, sim_thresh: float = 0.9) -> None:
        self.cap = int(cap)
        self.sim = float(sim_thresh)
        self.items: List[tuple] = []  # (tuple(traj), set(traj), score)
        self.hits = 0
        self.misses = 0

    def lookup(self, traj: Sequence[str]) -> Optional[float]:
        key = tuple(traj)
        s = set(traj)
        best_score = None
        best_sim = 0.0
        for sig, tset, score in self.items:
            if sig == key:
                self.hits += 1
                return score
            if not s and not tset:
                j = 1.0
            else:
                union = len(s | tset)
                j = (len(s & tset) / union) if union else 0.0
            if j >= self.sim and j > best_sim:
                best_sim = j
                best_score = score
        if best_score is not None:
            self.hits += 1
        else:
            self.misses += 1
        return best_score

    def add(self, traj: Sequence[str], score: float) -> None:
        self.items.append((tuple(traj), set(traj), score))
        if len(self.items) > self.cap:
            self.items.pop(0)


# ----------------------------------------------------------------------------
# Plan context shared across a single episode
# ----------------------------------------------------------------------------
@dataclass
class PlanContext:
    question: str
    query_tokens: List[str]
    scorer: Scorer
    rng: random.Random
    executed: List[str] = field(default_factory=list)
    traj_memory: Optional[TrajectoryMemory] = None
    bundle: Optional["DiffusionBundle"] = None


# ----------------------------------------------------------------------------
# Planner 1: single-step greedy
# ----------------------------------------------------------------------------
class SingleStepGreedy:
    name = "single_step"

    def __init__(self, k: int = 8) -> None:
        self.k = k

    def select_action(self, env: KGEnv, state: KGState, ctx: PlanContext) -> Optional[str]:
        adm = env.admissible_relations(state)
        if not adm:
            return None
        cands = ctx.scorer.propose(ctx.question, ctx.executed, adm, self.k)
        return max(
            cands,
            key=lambda a: ctx.scorer.score_relation(ctx.question, ctx.executed, a, sorted(env.neighbors(state, a))),
        )


# ----------------------------------------------------------------------------
# Planner 2: beam search (B=8)
# ----------------------------------------------------------------------------
class BeamSearch:
    name = "beam"

    def __init__(self, beam_width: int = 8, beam_depth: int = 3, k: int = 8) -> None:
        self.B = beam_width
        self.depth = beam_depth
        self.k = k

    def select_action(self, env: KGEnv, state: KGState, ctx: PlanContext) -> Optional[str]:
        adm = env.admissible_relations(state)
        if not adm:
            return None
        remaining = max(1, env.max_steps - state.depth)
        depth_cap = min(remaining, self.depth)
        beam = [(0.0, [], state)]
        for _ in range(depth_cap):
            cand = []
            for sc, rels, st in beam:
                if env.is_terminal(st):
                    cand.append((sc, rels, st))
                    continue
                a_adm = env.admissible_relations(st)
                if not a_adm:
                    cand.append((sc, rels, st))
                    continue
                for a in ctx.scorer.propose(ctx.question, ctx.executed + rels, a_adm, self.k):
                    ns = env.step(st, a)
                    s2 = sc + ctx.scorer.score_relation(ctx.question, ctx.executed + rels, a, sorted(ns.frontier))
                    cand.append((s2, rels + [a], ns))
            if not cand:
                break
            cand.sort(key=lambda x: -x[0])
            beam = cand[: self.B]
        best = max(beam, key=lambda x: x[0])
        return best[1][0] if best[1] else adm[0]


# ----------------------------------------------------------------------------
# Planner 3: shallow lookahead (k=2)
# ----------------------------------------------------------------------------
class ShallowLookahead:
    name = "lookahead"

    def __init__(self, lookahead_k: int = 2, k: int = 8) -> None:
        self.lk = lookahead_k
        self.k = k

    def select_action(self, env: KGEnv, state: KGState, ctx: PlanContext) -> Optional[str]:
        adm = env.admissible_relations(state)
        if not adm:
            return None
        roots = ctx.scorer.propose(ctx.question, ctx.executed, adm, self.k)
        best_a, best_v = None, -math.inf
        for a in roots:
            st = env.step(state, a)
            val = ctx.scorer.score_relation(ctx.question, ctx.executed, a, sorted(st.frontier))
            rels = [a]
            rs = st
            for _ in range(self.lk - 1):
                if env.is_terminal(rs):
                    break
                adm2 = env.admissible_relations(rs)
                if not adm2:
                    break
                prop = ctx.scorer.propose(ctx.question, ctx.executed + rels, adm2, self.k)
                if not prop:
                    break
                a2 = max(prop, key=lambda x: ctx.scorer.score_relation(ctx.question, ctx.executed + rels, x, sorted(env.neighbors(rs, x))))
                val += ctx.scorer.score_relation(ctx.question, ctx.executed + rels, a2, sorted(env.neighbors(rs, a2)))
                rels.append(a2)
                rs = env.step(rs, a2)
            if val > best_v:
                best_v, best_a = val, a
        return best_a or adm[0]


# ----------------------------------------------------------------------------
# Planner 4: FLARE (MCTS, Algorithm 1)
# ----------------------------------------------------------------------------
@dataclass
class MCTSNode:
    state: KGState
    depth_from_root: int = 0
    N: int = 0
    untried: Optional[List[str]] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    Q: Dict[str, float] = field(default_factory=dict)
    Na: Dict[str, int] = field(default_factory=dict)


class FLAREPlanner:
    name = "flare"

    def __init__(self, S: int = 16, c: float = 1.4, k: int = 8, H: int = 3,
                 mem_cap: int = 200, mem_sim: float = 0.9) -> None:
        self.S = S
        self.c = c
        self.k = k
        self.H = H
        self.mem_cap = mem_cap
        self.mem_sim = mem_sim

    def _ucb(self, node: MCTSNode, a: str) -> float:
        if node.Na.get(a, 0) == 0:
            return math.inf
        return node.Q[a] + self.c * math.sqrt(math.log(max(1, node.N)) / (node.Na[a] + 1))

    def select_action(self, env: KGEnv, state: KGState, ctx: PlanContext) -> Optional[str]:
        adm0 = env.admissible_relations(state)
        if not adm0:
            return None
        mem = ctx.traj_memory or TrajectoryMemory(self.mem_cap, self.mem_sim)
        root = MCTSNode(state, depth_from_root=0)
        for _ in range(self.S):
            self._simulate(env, root, ctx, mem)
        if not root.Q:
            return ctx.scorer.propose(ctx.question, ctx.executed, adm0, self.k)[0]
        return max(root.Q, key=root.Q.get)

    def _simulate(self, env: KGEnv, root: MCTSNode, ctx: PlanContext, mem: TrajectoryMemory) -> None:
        path = []  # list of (node, action)
        node = root
        # --- tree policy: select / expand ---
        while True:
            if env.is_terminal(node.state) or node.depth_from_root >= self.H:
                break
            if node.untried is None:
                adm = env.admissible_relations(node.state)
                executed_here = ctx.executed + [a for _, a in path]
                node.untried = ctx.scorer.propose(ctx.question, executed_here, adm, self.k)
                for a in node.untried:
                    node.Q.setdefault(a, 0.0)
                    node.Na.setdefault(a, 0)
            if not node.untried:
                break
            unexpanded = [a for a in node.untried if a not in node.children]
            if unexpanded:
                a = ctx.rng.choice(unexpanded)
                path.append((node, a))
                child = MCTSNode(env.step(node.state, a), depth_from_root=node.depth_from_root + 1)
                node.children[a] = child
                node = child
                break  # expansion: stop descending, go to rollout
            a = max(node.untried, key=lambda x: self._ucb(node, x))
            path.append((node, a))
            node = node.children[a]
        # --- simulation (rollout) up to total depth H ---
        traj = [a for _, a in path]
        entity_trace = [sorted(root.state.frontier)]
        rs = node.state
        depth = node.depth_from_root
        while depth < self.H and not env.is_terminal(rs):
            adm = env.admissible_relations(rs)
            if not adm:
                break
            prop = ctx.scorer.propose(ctx.question, ctx.executed + traj, adm, self.k)
            if not prop:
                break
            a = ctx.rng.choice(prop)
            traj.append(a)
            rs = env.step(rs, a)
            entity_trace.append(sorted(rs.frontier))
            depth += 1
        # --- trajectory-level evaluation (amortized) ---
        ret = mem.lookup(traj)
        if ret is None:
            ret = ctx.scorer.score_trajectory(ctx.question, traj, entity_trace)
            mem.add(traj, ret)
        # --- backpropagation ---
        for n, a in path:
            n.N += 1
            n.Na[a] += 1
            n.Q[a] += (ret - n.Q[a]) / n.Na[a]


# ----------------------------------------------------------------------------
# Planner 5: DiPLaN diffusion swap
# ----------------------------------------------------------------------------
@dataclass
class DiffusionBundle:
    planner: object
    autoencoder: object
    path_vocab: object
    query_vocab: object
    device: object
    value_model: object = None
    num_candidates: int = 16
    diffusion_steps: int = 20
    use_prefix: bool = True
    prediction_target: str = "z0"
    planner_type: str = "diffusion"
    jitter_std: float = 0.05
    latent_mean: object = None
    latent_std: object = None


def _nearest_admissible(proposed: Sequence[str], admissible: set) -> List[str]:
    """Map proposed (possibly out-of-graph) relations to nearest admissible by token Jaccard."""
    out = []
    for p in proposed:
        pt = _relation_tokens(p)
        best, best_j = None, 0.0
        for a in admissible:
            at = _relation_tokens(a)
            union = len(pt | at)
            j = (len(pt & at) / union) if union else 0.0
            if j > best_j:
                best_j, best = j, a
        if best is not None and best_j > 0.0:
            out.append(best)
    return list(dict.fromkeys(out))


class DiPLaNDiffusionPlanner:
    name = "diplan_diffusion"

    def __init__(self, bundle: DiffusionBundle) -> None:
        self.b = bundle
        self.grounded_hits = 0
        self.grounded_misses = 0

    def select_action(self, env: KGEnv, state: KGState, ctx: PlanContext) -> Optional[str]:
        from .inference import sample_plan_candidates  # local import: torch-heavy

        adm = set(env.admissible_relations(state))
        if not adm:
            return None
        b = self.b
        remaining = max(1, env.max_steps - state.depth)
        cands = sample_plan_candidates(
            planner=b.planner,
            autoencoder=b.autoencoder,
            path_vocab=b.path_vocab,
            query_vocab=b.query_vocab,
            query_tokens=ctx.query_tokens,
            num_candidates=b.num_candidates,
            diffusion_steps=b.diffusion_steps,
            max_path_len=remaining,
            device=b.device,
            executed_prefix=ctx.executed,
            use_prefix=b.use_prefix,
            latent_mean=b.latent_mean,
            latent_std=b.latent_std,
            prediction_target=b.prediction_target,
            planner_type=b.planner_type,
            jitter_std=b.jitter_std,
        )
        proposed_firsts = [c[0] for c in cands if c]
        grounded = [r for r in proposed_firsts if r in adm]
        if grounded:
            self.grounded_hits += 1
        else:
            self.grounded_misses += 1
            grounded = _nearest_admissible(proposed_firsts, adm) or sorted(adm)
        uniq = list(dict.fromkeys(grounded))
        return max(
            uniq,
            key=lambda a: ctx.scorer.score_relation(ctx.question, ctx.executed, a, sorted(env.neighbors(state, a))),
        )


# ----------------------------------------------------------------------------
# Diffusion bundle loader (mirrors scripts/run_diplan_llm_agent.py)
# ----------------------------------------------------------------------------
def load_diffusion_bundle(ae_ckpt_path: str, planner_ckpt_path: str,
                          value_ckpt_path: str = "", cfg: Optional[Dict] = None) -> DiffusionBundle:
    import torch

    from .torch_pipeline import DiffusionPlanner, MLPPlanner, PathAutoencoder, ValueRanker, load_vocab

    cfg = cfg or {}
    device = torch.device("cpu")
    # weights_only=False: checkpoints store vocab objects / latent-norm metadata, not just tensors.
    ae_ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)
    planner_ckpt = torch.load(planner_ckpt_path, map_location="cpu", weights_only=False)
    value_ckpt = torch.load(value_ckpt_path, map_location="cpu", weights_only=False) if value_ckpt_path else None

    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])

    ae_cfg = ae_ckpt["model_config"]
    autoencoder = PathAutoencoder(
        vocab_size=ae_cfg["vocab_size"], emb_dim=ae_cfg["emb_dim"], hid_dim=ae_cfg["hid_dim"],
        latent_dim=ae_cfg["latent_dim"], max_path_len=ae_cfg["max_path_len"], pad_id=ae_cfg["pad_id"],
        latent_noise_std=float(ae_cfg.get("latent_noise_std", 0.0)),
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["model_state"])
    autoencoder.eval()

    pl_cfg = planner_ckpt["model_config"]
    planner_type = str(pl_cfg.get("planner_type", "diffusion")).lower()
    if planner_type == "mlp":
        planner = MLPPlanner(latent_dim=pl_cfg["latent_dim"], q_vocab_size=pl_cfg["q_vocab_size"],
                             q_emb_dim=pl_cfg["q_emb_dim"], q_pad_id=pl_cfg["q_pad_id"],
                             hidden_dim=int(pl_cfg.get("hidden_dim", 256))).to(device)
    else:
        planner = DiffusionPlanner(latent_dim=pl_cfg["latent_dim"], q_vocab_size=pl_cfg["q_vocab_size"],
                                   q_emb_dim=pl_cfg["q_emb_dim"], q_pad_id=pl_cfg["q_pad_id"],
                                   time_dim=pl_cfg["time_dim"]).to(device)
    planner.load_state_dict(planner_ckpt["model_state"])
    planner.eval()

    latent_mean = latent_std = None
    latent_norm = planner_ckpt.get("latent_norm")
    if isinstance(latent_norm, dict) and bool(latent_norm.get("enabled", False)):
        latent_mean = torch.tensor(latent_norm["mean"], dtype=torch.float32, device=device).view(1, -1)
        latent_std = torch.tensor(latent_norm["std"], dtype=torch.float32, device=device).view(1, -1)

    value_model = None
    if value_ckpt is not None:
        v_cfg = value_ckpt["model_config"]
        value_model = ValueRanker(
            q_vocab_size=v_cfg["q_vocab_size"], p_vocab_size=v_cfg["p_vocab_size"], emb_dim=v_cfg["emb_dim"],
            q_pad_id=v_cfg["q_pad_id"], p_pad_id=v_cfg["p_pad_id"],
            architecture=str(v_cfg.get("architecture", "legacy")), hidden_dim=int(v_cfg.get("hidden_dim", 256)),
            dropout=float(v_cfg.get("dropout", 0.1)),
        ).to(device)
        value_model.load_state_dict(value_ckpt["model_state"])
        value_model.eval()

    dcfg = cfg.get("diffusion", {})
    return DiffusionBundle(
        planner=planner, autoencoder=autoencoder, path_vocab=path_vocab, query_vocab=query_vocab,
        device=device, value_model=value_model,
        num_candidates=int(dcfg.get("num_candidates", 16)),
        diffusion_steps=int(dcfg.get("diffusion_steps", 20)),
        use_prefix=bool(dcfg.get("use_prefix", True)),
        prediction_target=str(pl_cfg.get("prediction_target", "z0")).lower(),
        planner_type=planner_type,
        jitter_std=float(dcfg.get("jitter_std", 0.05)),
        latent_mean=latent_mean, latent_std=latent_std,
    )


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def make_planner(name: str, cfg: Optional[Dict] = None, bundle: Optional[DiffusionBundle] = None):
    cfg = cfg or {}
    name = name.lower()
    if name == "single_step":
        return SingleStepGreedy(k=int(cfg.get("k", 8)))
    if name == "beam":
        bc = cfg.get("beam", {})
        return BeamSearch(beam_width=int(bc.get("B", 8)), beam_depth=int(bc.get("depth", 3)), k=int(cfg.get("k", 8)))
    if name == "lookahead":
        lc = cfg.get("lookahead", {})
        return ShallowLookahead(lookahead_k=int(lc.get("k", 2)), k=int(cfg.get("k", 8)))
    if name == "flare":
        fc = cfg.get("flare", {})
        return FLAREPlanner(S=int(fc.get("S", 16)), c=float(fc.get("c", 1.4)), k=int(fc.get("k", 8)),
                            H=int(fc.get("H", 3)), mem_cap=int(fc.get("mem_cap", 200)), mem_sim=float(fc.get("mem_sim", 0.9)))
    if name == "diplan_diffusion":
        if bundle is None:
            raise ValueError("diplan_diffusion requires a loaded DiffusionBundle")
        return DiPLaNDiffusionPlanner(bundle)
    raise ValueError(f"Unknown planner: {name}")
