from typing import Dict, List, Optional, Tuple


CANONICAL_STAGE_ORDER: List[str] = [
    "INTAKE",
    "EXAM",
    "DIAGNOSIS",
    "PRESCRIBE",
    "FOLLOWUP",
]


ROLE_BY_STAGE: Dict[str, str] = {
    "INTAKE": "NURSE",
    "EXAM": "DOCTOR",
    "DIAGNOSIS": "DOCTOR",
    "PRESCRIBE": "DOCTOR",
    "FOLLOWUP": "NURSE",
}


STAGE_ACTION_PRIORS: Dict[str, List[str]] = {
    "INTAKE": [
        "triage",
        "collect_vitals",
        "register",
    ],
    "EXAM": [
        "history_taking",
        "physical_exam",
        "order_test",
    ],
    "DIAGNOSIS": [
        "diagnosis",
        "differential_diagnosis",
    ],
    "PRESCRIBE": [
        "prescribe",
        "treatment_plan",
    ],
    "FOLLOWUP": [
        "followup",
        "revisit",
    ],
}


def _stage_of(token: str) -> str:
    if not isinstance(token, str):
        return ""
    tok = token.strip()
    if not tok:
        return ""
    if "::" in tok:
        return tok.split("::", 1)[0].strip().upper()
    if "(" in tok:
        return tok.split("(", 1)[0].strip().upper()
    if "_" in tok:
        return tok.split("_", 1)[0].strip().upper()
    return tok.strip().upper()


def _infer_stage_order_from_oracle(oracle_path: List[str]) -> List[str]:
    seen = []
    seen_set = set()
    for token in oracle_path:
        stage = _stage_of(token)
        if not stage or stage in seen_set:
            continue
        seen_set.add(stage)
        seen.append(stage)
    if len(seen) >= 3:
        return seen
    return list(CANONICAL_STAGE_ORDER)


def _must_precede_rules(stage_order: List[str]) -> List[Dict[str, str]]:
    rules: List[Dict[str, str]] = []
    for i in range(len(stage_order) - 1):
        rules.append({"first": stage_order[i], "second": stage_order[i + 1]})
    return rules


def _required_before_rules(stage_order: List[str]) -> Dict[str, List[str]]:
    rules: Dict[str, List[str]] = {}
    for i, stage in enumerate(stage_order):
        if i == 0:
            continue
        rules[stage] = [stage_order[i - 1]]
    return rules


def build_clinical_collab_constraints(
    oracle_path: List[str],
    max_steps_cap: int = 32,
    max_steps_pad: int = 2,
) -> Dict:
    stage_order = _infer_stage_order_from_oracle(oracle_path)
    max_steps = min(max_steps_cap, max(6, len(oracle_path) + max_steps_pad))
    return {
        "max_steps": max_steps,
        "banned_relations": [],
        "required_stage_order": stage_order,
        "must_precede": _must_precede_rules(stage_order),
        "required_before": _required_before_rules(stage_order),
        "forbidden_actions": [
            "PRESCRIBE::contraindicated",
            "PRESCRIBE::allergy_conflict",
        ],
        "collab_roles": {stage: ROLE_BY_STAGE.get(stage, "DOCTOR") for stage in stage_order},
        "stage_action_priors": {
            stage: STAGE_ACTION_PRIORS.get(stage, []) for stage in stage_order
        },
    }


def build_clinical_collab_state_machine(
    stage_order: Optional[List[str]] = None,
) -> Dict[str, object]:
    order = [s.strip().upper() for s in (stage_order or CANONICAL_STAGE_ORDER) if s]
    nodes = []
    edges: List[Tuple[str, str]] = []
    for i, stage in enumerate(order):
        nodes.append(
            {
                "stage": stage,
                "owner_role": ROLE_BY_STAGE.get(stage, "DOCTOR"),
                "action_priors": STAGE_ACTION_PRIORS.get(stage, []),
            }
        )
        if i > 0:
            edges.append((order[i - 1], stage))
    return {
        "name": "clinical_multi_agent_collaboration_fsm",
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in edges],
        "entry_stage": order[0] if order else "",
        "terminal_stage": order[-1] if order else "",
    }
