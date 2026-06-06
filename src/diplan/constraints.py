"""Shared feasibility / constraint utilities for DiPLaN.

This module is the single source of truth for the rule-based feasibility checker
used by the KGQA evaluator (``evaluate_torch.py``), the decision-weighted diffusion
dataset, the learned constraint model trainer, and the ALFWorld executor.

It deliberately has no torch / project dependencies so it can be imported from any
layer without creating import cycles.
"""

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def action_stage(token: str) -> str:
    """Return the coarse 'stage'/verb of a plan token.

    Understands the ``VERB::payload``, ``VERB(payload)`` and ``verb_payload``
    conventions used across KGQA / clinical / ALFWorld plan tokens.
    """
    if not isinstance(token, str):
        return ""
    if "::" in token:
        return token.split("::", 1)[0].strip().upper()
    if "(" in token:
        return token.split("(", 1)[0].strip().upper()
    if "_" in token:
        return token.split("_", 1)[0].strip().upper()
    return token.strip().upper()


def _matches(token: str, pattern: str) -> bool:
    t = str(token).strip().upper()
    p = str(pattern).strip().upper()
    if not p:
        return False
    return t == p or t.startswith(p) or action_stage(t) == p


def is_feasible(path: List[str], constraints: Dict) -> Tuple[bool, List[str]]:
    """Rule-based feasibility check for a plan (list of plan tokens).

    Returns ``(feasible, sorted_unique_violation_codes)``. Moved verbatim from the
    former ``evaluate_torch._is_feasible`` so behaviour is unchanged.
    """
    violations: List[str] = []
    if len(path) > int(constraints.get("max_steps", 8)):
        violations.append("max_steps_exceeded")
    banned = set(constraints.get("banned_relations", []))
    for rel in path:
        if rel in banned:
            violations.append("banned_relation")

    # Generic forbidden action constraints (supports exact/prefix/stage-level patterns).
    forbidden_actions = constraints.get("forbidden_actions", [])
    if isinstance(forbidden_actions, list):
        for a in path:
            for ptn in forbidden_actions:
                if _matches(a, str(ptn)):
                    violations.append("forbidden_action")
                    break

    # Stage order constraints for long-horizon tasks (e.g., clinical workflow).
    required_stage_order = constraints.get("required_stage_order", [])
    if isinstance(required_stage_order, list) and required_stage_order:
        order_idx = {str(s).strip().upper(): i for i, s in enumerate(required_stage_order)}
        last = -1
        for a in path:
            st = action_stage(a)
            if st in order_idx:
                cur = order_idx[st]
                if cur < last:
                    violations.append("stage_order_violation")
                    break
                last = cur

    # Pairwise precedence constraints: first must happen before second (if second appears).
    must_precede = constraints.get("must_precede", [])
    if isinstance(must_precede, list):
        for rule in must_precede:
            if not isinstance(rule, dict):
                continue
            first = str(rule.get("first", "")).strip()
            second = str(rule.get("second", "")).strip()
            if not first or not second:
                continue
            first_pos = [i for i, a in enumerate(path) if _matches(a, first)]
            second_pos = [i for i, a in enumerate(path) if _matches(a, second)]
            if second_pos:
                if (not first_pos) or (min(second_pos) < min(first_pos)):
                    violations.append("precedence_violation")
                    break

    # For a target action, prerequisite actions must already exist before it.
    required_before = constraints.get("required_before", {})
    if isinstance(required_before, dict):
        for tgt, prereqs in required_before.items():
            if not isinstance(prereqs, list):
                continue
            tgt_pos = [i for i, a in enumerate(path) if _matches(a, str(tgt))]
            if not tgt_pos:
                continue
            first_tgt = min(tgt_pos)
            for pre in prereqs:
                pre_pos = [i for i, a in enumerate(path) if _matches(a, str(pre))]
                if (not pre_pos) or (min(pre_pos) > first_tgt):
                    violations.append("prerequisite_missing")
                    break
            if "prerequisite_missing" in violations:
                break
    return len(violations) == 0, sorted(set(violations))


def longest_common_prefix_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """Graded credit for a near-miss plan: shared prefix length / max length."""
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    denom = max(len(a), len(b), 1)
    return n / denom


def valid_relations_from_row(row: Dict) -> Set[str]:
    """Per-task set of legal plan tokens for feasibility projection (paper §5.4).

    Uses the oracle path plus any provided candidate paths as the available proxy
    for the underlying schema, minus banned relations.
    """
    valid: Set[str] = set()
    for tok in row.get("oracle_path", []) or []:
        if isinstance(tok, str):
            valid.add(tok)
    for cand in row.get("candidate_paths", []) or []:
        if isinstance(cand, list):
            for tok in cand:
                if isinstance(tok, str):
                    valid.add(tok)
    banned = set(row.get("constraints", {}).get("banned_relations", []) or [])
    return valid - banned


def _nearest_valid(token: str, valid_rels: Sequence[str]) -> str:
    """Cheap deterministic nearest-relation match (no embeddings)."""
    if not valid_rels:
        return token
    t = str(token)
    # 1) exact already handled by caller; here try shared-prefix / containment.
    best = None
    best_score = -1.0
    for cand in valid_rels:
        c = str(cand)
        # Character-level longest common prefix as a fast similarity proxy.
        lcp = 0
        for x, y in zip(t, c):
            if x == y:
                lcp += 1
            else:
                break
        # Token-overlap on '.'/'_'/'::'-split segments.
        seg_t = set(t.replace("::", ".").replace("_", ".").split("."))
        seg_c = set(c.replace("::", ".").replace("_", ".").split("."))
        overlap = len(seg_t & seg_c)
        score = overlap * 10.0 + lcp
        if score > best_score:
            best_score = score
            best = c
    return best if best is not None else token


def project_path_to_valid(path: List[str], valid_rels: Iterable[str]) -> List[str]:
    """Snap each out-of-set token to the nearest valid relation (paper §5.4)."""
    valid_set = set(valid_rels)
    if not valid_set:
        return list(path)
    valid_list = sorted(valid_set)
    out: List[str] = []
    for tok in path:
        if tok in valid_set:
            out.append(tok)
        else:
            out.append(_nearest_valid(tok, valid_list))
    return out
