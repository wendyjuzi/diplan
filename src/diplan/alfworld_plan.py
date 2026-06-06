"""ALFWorld plan-token normalization shared by the collector and the executor.

A raw ALFWorld command like ``"take mug 1 from desk 2"`` is mapped to a compact,
instance-free abstract plan token ``"TAKE::mug_desk"``. Using the ``VERB::payload``
convention keeps the abstract vocabulary small (verbs x object-types x receptacle-
types) and lets the shared rule checker (``src/diplan/constraints.py``) reason about
stages, ordering, and forbidden actions out of the box.

The same normalizer is used at execution time to map the planned head template to a
concrete admissible command (Tool Decoder / Feasibility Projection, paper §5.4).
"""

import re
from typing import Dict, List, Optional, Tuple

_NUM_SUFFIX = re.compile(r"\s+\d+$")


def _entity_type(text: str) -> str:
    """Strip a trailing instance id and collapse spaces: 'desk 2' -> 'desk'."""
    t = _NUM_SUFFIX.sub("", str(text).strip().lower()).strip()
    if t.startswith("the "):
        t = t[4:]
    return t.replace(" ", "")


def parse_action(raw: str) -> Dict[str, str]:
    """Parse a raw ALFWorld command into ``{verb, obj, recep}`` (types, no ids)."""
    a = str(raw).strip().lower()
    m = re.match(r"go to (.+)$", a)
    if m:
        return {"verb": "GOTO", "obj": "", "recep": _entity_type(m.group(1))}
    m = re.match(r"open (.+)$", a)
    if m:
        return {"verb": "OPEN", "obj": "", "recep": _entity_type(m.group(1))}
    m = re.match(r"close (.+)$", a)
    if m:
        return {"verb": "CLOSE", "obj": "", "recep": _entity_type(m.group(1))}
    m = re.match(r"take (.+?) from (.+)$", a)
    if m:
        return {"verb": "TAKE", "obj": _entity_type(m.group(1)), "recep": _entity_type(m.group(2))}
    m = re.match(r"(?:put|move) (.+?) (?:in/on|in|on|to) (.+)$", a)
    if m:
        return {"verb": "PUT", "obj": _entity_type(m.group(1)), "recep": _entity_type(m.group(2))}
    for verb in ("heat", "cool", "clean"):
        m = re.match(rf"{verb} (.+?) with (.+)$", a)
        if m:
            return {"verb": verb.upper(), "obj": _entity_type(m.group(1)), "recep": _entity_type(m.group(2))}
    m = re.match(r"use (.+)$", a)
    if m:
        return {"verb": "USE", "obj": "", "recep": _entity_type(m.group(1))}
    m = re.match(r"examine (.+)$", a)
    if m:
        return {"verb": "EXAMINE", "obj": "", "recep": _entity_type(m.group(1))}
    m = re.match(r"look at (.+?) under (.+)$", a)
    if m:
        return {"verb": "EXAMINE", "obj": _entity_type(m.group(1)), "recep": _entity_type(m.group(2))}
    # inventory / look / help / anything else.
    verb = a.split(" ", 1)[0].upper() if a else "OTHER"
    return {"verb": "OTHER", "obj": "", "recep": _entity_type(verb)}


def normalize_action(raw: str) -> str:
    """Raw command -> abstract ``VERB::obj_recep`` plan token."""
    p = parse_action(raw)
    payload = "_".join([x for x in (p["obj"], p["recep"]) if x]) or p["recep"] or "x"
    return f"{p['verb']}::{payload}"


def receptacle_types_from_admissible(admissible: List[str]) -> List[str]:
    """Receptacle types reachable via 'go to ...' in the current admissible set."""
    recs = []
    seen = set()
    for cmd in admissible:
        m = re.match(r"go to (.+)$", str(cmd).strip().lower())
        if m:
            r = _entity_type(m.group(1))
            if r and r not in seen:
                seen.add(r)
                recs.append(r)
    return recs


def goal_query_tokens(goal_keywords: List[str], admissible: List[str]) -> List[str]:
    """Planner condition tokens: goal keywords + scene receptacle-type tokens."""
    toks = list(goal_keywords)
    for r in receptacle_types_from_admissible(admissible):
        toks.append(f"RECEP::{r}")
    return toks


def match_plan_token_to_admissible(
    plan_token: str,
    admissible: List[str],
    prefer_receptacle: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """Tool Decoder / Feasibility Projection (paper §5.4).

    Map an abstract plan token to the best concrete admissible command. Returns
    ``(command_or_None, match_score)``. Score 1.0 = exact template match. A return of
    ``None`` means the head template is not projectable in the current state.
    """
    if not admissible:
        return None, 0.0
    want = parse_action_token(plan_token)
    best_cmd = None
    best_score = -1.0
    for cmd in admissible:
        got = parse_action(cmd)
        if got["verb"] != want["verb"]:
            continue
        score = 0.0
        # Object / receptacle type agreement.
        if want["obj"]:
            score += 2.0 if got["obj"] == want["obj"] else 0.0
        if want["recep"]:
            score += 2.0 if got["recep"] == want["recep"] else 0.0
        # Tie-break: prefer a receptacle the agent already knows holds the object.
        if prefer_receptacle and got["recep"] == _entity_type(prefer_receptacle):
            score += 0.5
        # Verb-only match still counts (lets GOTO/USE project even w/o payload).
        score += 0.1
        if score > best_score:
            best_score = score
            best_cmd = cmd
    if best_cmd is None:
        return None, 0.0
    # Normalize score to [0,1]: exact obj+recep match -> ~1.0.
    denom = 0.1 + (2.0 if want["obj"] else 0.0) + (2.0 if want["recep"] else 0.0)
    return best_cmd, min(1.0, best_score / max(0.1, denom))


def parse_action_token(plan_token: str) -> Dict[str, str]:
    """Inverse of :func:`normalize_action`: parse ``VERB::obj_recep``."""
    t = str(plan_token)
    if "::" not in t:
        return {"verb": t.strip().upper(), "obj": "", "recep": ""}
    verb, payload = t.split("::", 1)
    parts = [p for p in payload.split("_") if p]
    verb = verb.strip().upper()
    if verb in ("TAKE", "PUT", "HEAT", "COOL", "CLEAN") and len(parts) >= 2:
        return {"verb": verb, "obj": parts[0], "recep": parts[1]}
    if verb in ("TAKE", "PUT", "HEAT", "COOL", "CLEAN") and len(parts) == 1:
        return {"verb": verb, "obj": parts[0], "recep": ""}
    # GOTO/OPEN/CLOSE/USE/EXAMINE -> single receptacle payload.
    return {"verb": verb, "obj": "", "recep": parts[0] if parts else ""}
