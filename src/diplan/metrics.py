from statistics import mean
from typing import Dict, List


def success_rate(records: List[Dict]) -> float:
    if not records:
        return 0.0
    return mean(1.0 if r["success"] else 0.0 for r in records)


def first_error_step(pred: List[str], oracle: List[str]) -> int:
    for i in range(min(len(pred), len(oracle))):
        if pred[i] != oracle[i]:
            return i + 1
    if len(pred) != len(oracle):
        return min(len(pred), len(oracle)) + 1
    return len(oracle) + 1


def recovery_at_error(pred: List[str], oracle: List[str]) -> bool:
    err = first_error_step(pred, oracle)
    if err > len(oracle):
        return False
    suffix_oracle = oracle[err:]
    suffix_pred = pred[err:] if err < len(pred) else []
    if not suffix_oracle:
        return False
    overlap = len(set(suffix_oracle).intersection(set(suffix_pred)))
    return overlap / len(set(suffix_oracle)) >= 0.5


def trap_at_1(pred: List[str], trap: List[str]) -> bool:
    if not pred or not trap:
        return False
    return pred[0] == trap[0]


def plan_execution_consistency(plan: List[str], executed: List[str]) -> float:
    if not plan and not executed:
        return 1.0
    if not plan or not executed:
        return 0.0
    n = min(len(plan), len(executed))
    hits = sum(1 for i in range(n) if plan[i] == executed[i])
    return hits / max(1, max(len(plan), len(executed)))


def aggregate_method_metrics(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    return {
        "success_rate": success_rate(records),
        "first_error_step": mean(r["first_error_step"] for r in records),
        "recovery_at_error": mean(1.0 if r["recovery_at_error"] else 0.0 for r in records),
        "trap_at_1": mean(1.0 if r["trap_at_1"] else 0.0 for r in records),
        "plan_feasibility": mean(1.0 if r["feasible"] else 0.0 for r in records),
        "constraint_violation_rate": mean(1.0 if r["violations"] else 0.0 for r in records),
        "plan_execution_consistency": mean(r["plan_execution_consistency"] for r in records),
        "token_cost": mean(r["token_cost"] for r in records),
        "latency_cost": mean(r["latency_cost"] for r in records),
        "diversity_coverage": mean(r["diversity_coverage"] for r in records),
    }

