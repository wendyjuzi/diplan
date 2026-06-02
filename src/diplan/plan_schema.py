from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class StructuredPlanStep:
    step_id: int
    raw_action: str
    subgoal: str
    tool: str
    arguments: Dict[str, str]
    precondition: List[str]
    postcondition: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def _split_action(action: str) -> tuple[str, str]:
    token = str(action).strip()
    if "::" in token:
        stage, op = token.split("::", 1)
        return stage.strip().lower(), op.strip()
    if "(" in token and token.endswith(")"):
        tool, arg_text = token.split("(", 1)
        return tool.strip().lower(), arg_text[:-1].strip()
    if "_" in token:
        prefix, suffix = token.split("_", 1)
        return prefix.strip().lower(), suffix.strip()
    return "plan", token


def action_to_structured_step(action: str, step_id: int) -> StructuredPlanStep:
    stage, op = _split_action(action)
    tool = stage or "plan"
    subgoal = op.replace("_", " ").strip() or str(action)
    return StructuredPlanStep(
        step_id=step_id,
        raw_action=str(action),
        subgoal=subgoal,
        tool=tool,
        arguments={"action": str(action), "operation": op},
        precondition=["previous_steps_completed"] if step_id > 1 else [],
        postcondition=[f"{tool}_completed"],
    )


def path_to_structured_plan(path: List[str]) -> List[Dict]:
    return [action_to_structured_step(action, i + 1).to_dict() for i, action in enumerate(path or [])]
