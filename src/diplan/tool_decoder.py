from typing import Dict, List

from .plan_schema import path_to_structured_plan


def step_to_tool_call(step: Dict) -> Dict:
    tool = str(step.get("tool", "plan")).strip() or "plan"
    args = dict(step.get("arguments", {}) or {})
    return {
        "id": f"call_{int(step.get('step_id', 0)):03d}",
        "tool": tool,
        "arguments": args,
        "precondition": list(step.get("precondition", []) or []),
        "expected_postcondition": list(step.get("postcondition", []) or []),
    }


def path_to_tool_calls(path: List[str]) -> List[Dict]:
    return [step_to_tool_call(step) for step in path_to_structured_plan(path)]
