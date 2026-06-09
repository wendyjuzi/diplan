"""Knowledge-graph environment for FLARE-style planning over RoG subgraphs.

Each RoG row carries a ``graph`` of ``[subject, relation, object]`` triples plus
``q_entity`` (topic entities) and ``a_entity`` (answer entities). This module turns
that into a deterministic state-transition system G = (S, A, T) matching the paper's
controlled oracle-structure setting:

  * State  = a frontier of entities (frozenset) plus the number of executed hops.
  * A(s)   = the outgoing relations available from any entity in the frontier.
  * T(s,r) = the union of entities reached by following relation ``r`` from the frontier.

It also provides:
  * ``align_oracle_path`` — replay the oracle relation chain on the real graph, with a
    bounded-BFS repair fallback when the chain is not directly executable.
  * ``construct_myopic_trap`` — build a step-1 myopic trap (appendix E.5): a relation
    that is locally attractive yet leads to a region from which no answer is reachable.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

Triple = Tuple[str, str, str]


@dataclass(frozen=True)
class KGState:
    frontier: FrozenSet[str]
    depth: int


class KGEnv:
    """Deterministic KG traversal environment over a single question's subgraph."""

    def __init__(
        self,
        triples: Sequence[Sequence[str]],
        q_entities: Sequence[str],
        a_entities: Sequence[str],
        max_steps: int,
    ) -> None:
        self.q_entities: List[str] = [str(e) for e in q_entities if str(e)]
        self.a_entities: Set[str] = {str(e) for e in a_entities if str(e)}
        self.max_steps = int(max_steps)
        # adjacency: entity -> relation -> set(neighbor entities)
        self.adj: Dict[str, Dict[str, Set[str]]] = {}
        self.relations: Set[str] = set()
        for t in triples:
            if not t or len(t) != 3:
                continue
            s, r, o = str(t[0]), str(t[1]), str(t[2])
            if not s or not r:
                continue
            self.adj.setdefault(s, {}).setdefault(r, set()).add(o)
            self.relations.add(r)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_rog_row(cls, row: Dict, max_steps: int) -> "KGEnv":
        graph = row.get("graph") or []
        q_entity = row.get("q_entity") or []
        a_entity = row.get("a_entity") or []
        if isinstance(q_entity, str):
            q_entity = [q_entity]
        if isinstance(a_entity, str):
            a_entity = [a_entity]
        return cls(graph, q_entity, a_entity, max_steps)

    # ---- core state machine ----------------------------------------------
    def reset(self) -> KGState:
        return KGState(frozenset(self.q_entities), 0)

    def admissible_relations(self, state: KGState) -> List[str]:
        rels: Set[str] = set()
        for e in state.frontier:
            rels.update(self.adj.get(e, {}).keys())
        return sorted(rels)

    def neighbors(self, state: KGState, relation: str) -> FrozenSet[str]:
        """T(s, relation): union of object entities reached via ``relation``."""
        out: Set[str] = set()
        for e in state.frontier:
            out.update(self.adj.get(e, {}).get(relation, ()))
        return frozenset(out)

    def step(self, state: KGState, relation: str) -> KGState:
        return KGState(self.neighbors(state, relation), state.depth + 1)

    def answer_reached(self, state: KGState) -> bool:
        return bool(state.frontier & self.a_entities)

    def is_terminal(self, state: KGState) -> bool:
        return (
            state.depth >= self.max_steps
            or not state.frontier
            or self.answer_reached(state)
        )

    # ---- reachability helpers --------------------------------------------
    def answer_reachable_within(self, frontier: FrozenSet[str], budget: int) -> bool:
        """Whether any answer entity is reachable from ``frontier`` within ``budget`` hops."""
        if frontier & self.a_entities:
            return True
        if budget <= 0:
            return False
        seen: Set[str] = set(frontier)
        cur: Set[str] = set(frontier)
        for _ in range(budget):
            nxt: Set[str] = set()
            for e in cur:
                for neigh in self.adj.get(e, {}).values():
                    nxt.update(neigh)
            nxt -= seen
            if nxt & self.a_entities:
                return True
            if not nxt:
                return False
            seen.update(nxt)
            cur = nxt
        return False

    def bfs_relation_path(self, max_len: Optional[int] = None) -> Optional[List[str]]:
        """Shortest relation chain from q_entity frontier to any answer entity.

        Returns the list of relations, or None if no path within the hop budget.
        """
        budget = self.max_steps if max_len is None else int(max_len)
        start = frozenset(self.q_entities)
        if start & self.a_entities:
            return []
        # BFS over frontiers, tracking the relation sequence that produced each.
        seen_entities: Set[str] = set(start)
        queue: deque[Tuple[FrozenSet[str], List[str]]] = deque([(start, [])])
        while queue:
            frontier, path = queue.popleft()
            if len(path) >= budget:
                continue
            state = KGState(frontier, len(path))
            for rel in self.admissible_relations(state):
                nbrs = self.neighbors(state, rel)
                if not nbrs:
                    continue
                if nbrs & self.a_entities:
                    return path + [rel]
                fresh = nbrs - seen_entities
                if not fresh:
                    continue
                seen_entities.update(fresh)
                queue.append((frozenset(nbrs), path + [rel]))
        return None


def align_oracle_path(
    env: KGEnv, oracle_relations: Sequence[str]
) -> Tuple[Optional[List[str]], str]:
    """Replay the oracle relation chain on the real graph.

    Returns ``(path, source)`` where ``source`` is one of:
      * "oracle"  — the provided chain was directly executable and reaches an answer;
      * "bfs"     — the chain failed, repaired via shortest-path BFS;
      * "none"    — no executable path to an answer exists (row should be dropped).
    """
    state = env.reset()
    ok = bool(oracle_relations)
    for rel in oracle_relations:
        admissible = set(env.admissible_relations(state))
        if rel not in admissible:
            ok = False
            break
        state = env.step(state, rel)
    if ok and env.answer_reached(state):
        return list(oracle_relations), "oracle"
    repaired = env.bfs_relation_path()
    if repaired:
        return repaired, "bfs"
    return None, "none"


def _relation_attractiveness(relation: str, question_tokens: Sequence[str]) -> float:
    """Offline lexical proxy for how locally attractive a relation looks vs the question.

    Higher = the relation's surface tokens overlap more with the question. Used so trap
    construction does not require an LLM call per candidate.
    """
    q = {t.lower() for t in question_tokens}
    if not q:
        return 0.0
    rel_tokens = {tok for tok in relation.replace(".", " ").replace("_", " ").split() if tok}
    if not rel_tokens:
        return 0.0
    overlap = len(rel_tokens & q)
    return overlap / len(rel_tokens)


def construct_myopic_trap(
    env: KGEnv,
    oracle_path: Sequence[str],
    question_tokens: Sequence[str],
    rng: Optional[random.Random] = None,
) -> Optional[List[str]]:
    """Build a step-1 myopic trap relation (paper appendix E.5).

    A trap relation must be (a) admissible at the root, (b) not the oracle's first
    relation, (c) lead to a frontier from which no answer is reachable within the
    remaining hop budget, and is chosen to be the most *locally attractive* such
    decoy (lexical overlap with the question). Returns ``[trap] + oracle_path[1:]`` to
    mirror the dataset trap shape, or None if no qualifying decoy exists.
    """
    if not oracle_path:
        return None
    root = env.reset()
    oracle_first = oracle_path[0]
    candidates: List[Tuple[float, str]] = []
    for rel in env.admissible_relations(root):
        if rel == oracle_first:
            continue
        nxt = env.neighbors(root, rel)
        if not nxt:
            # immediate dead-end is trivially globally dead but uninformative; still valid
            globally_dead = True
        else:
            globally_dead = not env.answer_reachable_within(nxt, env.max_steps - 1)
        if globally_dead:
            candidates.append((_relation_attractiveness(rel, question_tokens), rel))
    if not candidates:
        return None
    # Prefer the most attractive dead relation; break ties randomly for diversity.
    best_score = max(s for s, _ in candidates)
    top = [r for s, r in candidates if s >= best_score - 1e-9]
    trap_first = (rng or random).choice(top) if len(top) > 1 else top[0]
    return [trap_first] + list(oracle_path[1:])
