from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List


def _avg(rows: List[Dict], key: str, default: float = 0.0) -> float:
    if not rows:
        return 0.0
    return mean(float(r.get(key, default)) for r in rows)


def _rate(rows: List[Dict], key: str) -> float:
    if not rows:
        return 0.0
    return mean(1.0 if r.get(key) else 0.0 for r in rows)


def aggregate_diagnostics(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    first_error_hist = Counter(str(r.get("first_error_step", "unknown")) for r in rows)
    violation_counter = Counter()
    for row in rows:
        for violation in row.get("violations", []) or []:
            violation_counter[str(violation)] += 1
    return {
        "n": len(rows),
        "success_rate": _rate(rows, "success"),
        "first_error_step": _avg(rows, "first_error_step"),
        "recovery_at_error": _rate(rows, "recovery_at_error"),
        "trap_at_1": _rate(rows, "trap_at_1"),
        "plan_feasibility": _rate(rows, "feasible"),
        "constraint_violation_rate": mean(1.0 if r.get("violations") else 0.0 for r in rows),
        "plan_execution_consistency": _avg(rows, "plan_execution_consistency"),
        "candidate_pool_hit_rate": _rate(rows, "oracle_in_candidate_pool"),
        "ranking_error_rate": _rate(rows, "ranking_error"),
        "candidate_pool_avg_size": _avg(rows, "candidate_pool_size"),
        "first_error_histogram": dict(first_error_hist),
        "violation_histogram": dict(violation_counter),
    }


def aggregate_by_dataset(rows: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset", "unknown"))].append(row)
    return {dataset: aggregate_diagnostics(items) for dataset, items in sorted(grouped.items())}
