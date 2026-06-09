"""LLM-backed evaluative signal for KGQA planning (the paper's phi and r-hat).

``LLMScorer`` implements the ``Scorer`` protocol used by all planners:
  * propose(...)         -- phi: prune the admissible relations to <=k via the LLM.
  * score_relation(...)  -- local r-hat(s,a,T(s,a)) in [0,1].
  * score_trajectory(...) -- trajectory-level r-hat in [0,1] (FLARE).

All calls are cached; any LLM/parse failure falls back to a lexical proxy so a flaky
endpoint degrades gracefully instead of crashing a long run.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Sequence

from .llm_client import LLMClient, LLMConfig, LLMError
from .planners import _overlap  # lexical fallback shared with StubScorer

_MAX_ENTITIES_IN_PROMPT = 12
_MAX_RELATIONS_IN_PROMPT = 60


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _extract_indices(text: str, n_items: int, k: int) -> List[int]:
    """Parse 1-based option indices from non-strict LLM outputs.

    Qwen-style local models often return prose such as ``The best choices are
    3, 7, and 10`` despite a JSON-only instruction. Since the response does not
    include the original numbered menu, extracting integers from the response is
    safe enough and avoids falling back to the weak lexical proxy.
    """
    obj = _extract_json(text)
    raw = None
    if isinstance(obj, dict):
        raw = obj.get("indices") or obj.get("index") or obj.get("choices") or obj.get("choice")
    elif isinstance(obj, list):
        raw = obj
    if raw is None:
        raw = re.findall(r"\b\d+\b", text)
    if isinstance(raw, (int, float, str)):
        raw = [raw]
    out = []
    for x in raw or []:
        try:
            idx = int(float(str(x).strip()))
        except Exception:
            continue
        if 1 <= idx <= n_items and idx not in out:
            out.append(idx)
        if len(out) >= k:
            break
    return out


def _extract_score(text: str) -> float:
    obj = _extract_json(text)
    if isinstance(obj, dict):
        if "score" in obj:
            return _clamp01(obj.get("score"))
        # tolerate {"rating": ...} / {"value": ...}
        for key in ("rating", "value", "probability", "confidence"):
            if key in obj:
                return _clamp01(obj.get(key))
    elif obj is not None:
        return _clamp01(obj)
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
    if m:
        return _clamp01(m.group(0))
    m = re.search(r"\b(\d{1,3})\s*%", text)
    if m:
        return _clamp01(float(m.group(1)) / 100.0)
    raise LLMError("no parseable score")


def _normalize_question_for_overlap(question: str) -> str:
    # The ToG-subgraph runner appends context after newlines; lexical fallback
    # should focus on the natural-language question, not entity metadata.
    return question.split("\n", 1)[0]


def _relation_label(relation: str) -> str:
    """Human-readable hint for Freebase-style relation IDs."""
    return " ".join(t for t in re.split(r"[._/]+", relation) if t)


def _lexical_rank(question: str, relations: Sequence[str]) -> List[str]:
    q = _normalize_question_for_overlap(question)
    return sorted(
        relations,
        key=lambda r: (-max(_overlap(r, q), _overlap(_relation_label(r), q)), r),
    )


def _clamp01(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


class LLMScorer:
    def __init__(
        self,
        client: LLMClient,
        max_relations_in_prompt: int = _MAX_RELATIONS_IN_PROMPT,
        proposal_fusion: str = "llm",
    ) -> None:
        self.client = client
        self.max_relations_in_prompt = int(max_relations_in_prompt)
        self.proposal_fusion = str(proposal_fusion).lower()
        self._propose_cache: Dict[tuple, List[str]] = {}
        self._rel_cache: Dict[tuple, float] = {}
        self._traj_cache: Dict[tuple, float] = {}
        self.fallbacks = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMScorer":
        return cls(
            LLMClient(LLMConfig.from_config(cfg)),
            max_relations_in_prompt=int(cfg.get("llm_max_relations_in_prompt", _MAX_RELATIONS_IN_PROMPT)),
            proposal_fusion=str(cfg.get("llm_proposal_fusion", "llm")),
        )

    # ---- phi: action proposal / pruning ----------------------------------
    def propose(self, question, executed, admissible, k):
        admissible = list(admissible)
        if len(admissible) <= k:
            return admissible
        key = (question, tuple(executed), tuple(admissible), int(k))
        if key in self._propose_cache:
            return self._propose_cache[key]
        menu = admissible[: self.max_relations_in_prompt]
        listing = "\n".join(f"{i}. {r} | label: {_relation_label(r)}" for i, r in enumerate(menu, 1))
        system = (
            "You are a knowledge-graph reasoning agent. Choose candidate relation IDs. "
            "Freebase relations are dot-separated; use their label words to infer meaning. "
            "Return ONLY a JSON array of 1-based integers, for example: [3, 7, 10]. "
            "Do not explain."
        )
        user = (
            f"Question: {question}\n"
            f"Relations already traversed: {list(executed)}\n"
            f"Candidate next relations:\n{listing}\n"
            "Prefer relations whose label words directly match the question intent, expected answer type, "
            "and current entity context if provided.\n"
            f"Select exactly {k} indices most likely to lead toward the answer. "
            "Return only the JSON array."
        )
        try:
            raw = self.client.chat(system, user)
            picked = []
            for ii in _extract_indices(raw, len(menu), k):
                picked.append(menu[ii - 1])
            picked = list(dict.fromkeys(picked))[:k]
            if not picked:
                raise LLMError("empty proposal")
        except Exception:
            self.fallbacks += 1
            picked = _lexical_rank(question, admissible)[:k]
        if self.proposal_fusion in {"lexical_backfill", "hybrid"} and len(picked) < k:
            for rel in _lexical_rank(question, admissible):
                if rel not in picked:
                    picked.append(rel)
                if len(picked) >= k:
                    break
        elif self.proposal_fusion in {"rerank_hybrid", "hybrid_rerank"}:
            llm_rank = {rel: i for i, rel in enumerate(picked)}
            lex_ranked = _lexical_rank(question, admissible)
            lex_rank = {rel: i for i, rel in enumerate(lex_ranked)}
            pool = list(dict.fromkeys(picked + lex_ranked[: max(k, len(picked))]))
            picked = sorted(pool, key=lambda r: (llm_rank.get(r, 9999) + lex_rank.get(r, 9999), lex_rank.get(r, 9999)))[:k]
        self._propose_cache[key] = picked
        return picked

    # ---- r-hat: local relation score -------------------------------------
    def score_relation(self, question, executed, relation, resulting_entities):
        key = (question, tuple(executed), relation)
        if key in self._rel_cache:
            return self._rel_cache[key]
        ents = list(resulting_entities)[:_MAX_ENTITIES_IN_PROMPT]
        system = (
            "Rate how useful taking the given relation is for answering the question, "
            "considering the entities it leads to. Return strict JSON only: "
            '{"score": <float 0..1>}.'
        )
        user = (
            f"Question: {question}\n"
            f"Relations already traversed: {list(executed)}\n"
            f"Candidate relation: {relation}\n"
            f"Candidate relation label: {_relation_label(relation)}\n"
            f"Entities reached by it: {ents}\n"
            "Score 1.0 = directly on the path to the answer, 0.0 = irrelevant/dead-end."
        )
        try:
            raw = self.client.chat(system, user)
            score = _extract_score(raw)
        except Exception:
            self.fallbacks += 1
            score = max(
                _overlap(relation, _normalize_question_for_overlap(question)),
                _overlap(_relation_label(relation), _normalize_question_for_overlap(question)),
            ) + (0.1 if resulting_entities else -0.2)
        self._rel_cache[key] = score
        return score

    # ---- r-hat: trajectory-level score -----------------------------------
    def score_trajectory(self, question, relations, entity_trace):
        key = (question, tuple(relations))
        if key in self._traj_cache:
            return self._traj_cache[key]
        last_entities = list(entity_trace[-1])[:_MAX_ENTITIES_IN_PROMPT] if entity_trace else []
        system = (
            "Rate how likely the following relation trajectory leads from the question's "
            "topic entity to a correct answer. Judge the WHOLE trajectory, not single steps. "
            'Return strict JSON only: {"score": <float 0..1>}.'
        )
        user = (
            f"Question: {question}\n"
            f"Relation trajectory: {list(relations)}\n"
            f"Relation trajectory labels: {[_relation_label(r) for r in relations]}\n"
            f"Entities at the end of the trajectory: {last_entities}\n"
            "Score 1.0 = trajectory reaches or approaches the answer, 0.0 = off-track."
        )
        try:
            raw = self.client.chat(system, user)
            score = _extract_score(raw)
        except Exception:
            self.fallbacks += 1
            if not relations:
                score = 0.0
            else:
                mean_ov = sum(_overlap(r, question) for r in relations) / len(relations)
                score = mean_ov + (0.1 if last_entities else -0.1)
        self._traj_cache[key] = score
        return score
