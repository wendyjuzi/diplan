import random
from collections import defaultdict
from typing import Dict, List

from .metrics import (
    aggregate_method_metrics,
    first_error_step,
    plan_execution_consistency,
    recovery_at_error,
    trap_at_1,
)
from .planner import (
    check_constraints,
    execute_with_receding_horizon,
    path_value_score,
    sample_candidate_path,
)


def _predict_single_step(row: Dict, planner_model: Dict, value_model: Dict, rng: random.Random) -> Dict:
    query = row["query_tokens"]
    path = sample_candidate_path(query, planner_model, rng, max_len=len(query), use_value_guidance=False)
    plan = [path[0]] if path else []
    while len(plan) < min(len(query), row["constraints"].get("max_steps", 6)):
        next_token = sample_candidate_path(plan[-1:], planner_model, rng, max_len=1, use_value_guidance=False)
        plan.extend(next_token)
    return {"planned_path": plan, "executed_path": plan, "candidate_count": 1, "candidate_unique_ratio": 1.0, "violations": []}


def _predict_beam(row: Dict, planner_model: Dict, value_model: Dict, constraint_model: Dict, rng: random.Random) -> Dict:
    query = row["query_tokens"]
    beam = []
    for _ in range(6):
        c = sample_candidate_path(query, planner_model, rng, max_len=row["constraints"].get("max_steps", 6), use_value_guidance=False)
        feasible, violations = check_constraints(c, constraint_model, row["constraints"])
        if feasible:
            beam.append((path_value_score(c, query, value_model), c))
        else:
            beam.append((-999.0, c))
    best = max(beam, key=lambda x: x[0])[1]
    return {
        "planned_path": best,
        "executed_path": best,
        "candidate_count": len(beam),
        "candidate_unique_ratio": len({tuple(x[1]) for x in beam}) / max(1, len(beam)),
        "violations": [],
    }


def _predict_lookahead(row: Dict, planner_model: Dict, value_model: Dict, constraint_model: Dict, rng: random.Random) -> Dict:
    query = row["query_tokens"]
    cands = [
        sample_candidate_path(
            query_tokens=query,
            planner_model=planner_model,
            rng=rng,
            max_len=row["constraints"].get("max_steps", 6),
            use_value_guidance=True,
            value_model=value_model,
        )
        for _ in range(10)
    ]
    scored = []
    violations = []
    for c in cands:
        feasible, v = check_constraints(c, constraint_model, row["constraints"])
        if not feasible:
            violations.extend(v)
            continue
        score = path_value_score(c, query, value_model) + 0.2 * len(set(c))
        scored.append((score, c))
    best = max(scored, key=lambda x: x[0])[1] if scored else cands[0]
    return {
        "planned_path": best,
        "executed_path": best,
        "candidate_count": len(cands),
        "candidate_unique_ratio": len({tuple(x) for x in cands}) / max(1, len(cands)),
        "violations": sorted(set(violations)),
    }


def _predict_diplan(
    row: Dict,
    planner_model: Dict,
    value_model: Dict,
    constraint_model: Dict,
    rng: random.Random,
    options: Dict,
) -> Dict:
    return execute_with_receding_horizon(
        query_tokens=row["query_tokens"],
        planner_model=planner_model,
        value_model=value_model,
        constraint_model=constraint_model,
        task_constraints=row["constraints"],
        rng=rng,
        num_candidates=int(options.get("num_candidates", 16)),
        use_value_guidance=bool(options.get("use_value_guidance", True)),
        use_constraint_checker=bool(options.get("use_constraint_checker", True)),
        receding_horizon=bool(options.get("receding_horizon", True)),
    )


def evaluate_suite(
    test_rows: List[Dict],
    planner_model: Dict,
    value_model: Dict,
    constraint_model: Dict,
    methods: List[str],
    seed: int,
    options: Dict | None = None,
) -> Dict:
    options = options or {}
    rng = random.Random(seed)
    by_method_records: Dict[str, List[Dict]] = defaultdict(list)

    for row in test_rows:
        oracle = row["oracle_path"]
        trap = row["trap_path"]
        for method in methods:
            if method == "single_step":
                pred = _predict_single_step(row, planner_model, value_model, rng)
            elif method == "beam_search":
                pred = _predict_beam(row, planner_model, value_model, constraint_model, rng)
            elif method in ("lookahead", "flare", "mcts_style"):
                pred = _predict_lookahead(row, planner_model, value_model, constraint_model, rng)
            elif method == "diplan":
                pred = _predict_diplan(row, planner_model, value_model, constraint_model, rng, options)
            else:
                raise ValueError(f"Unknown method: {method}")

            planned = pred["planned_path"]
            executed = pred["executed_path"]
            feasible, violations = check_constraints(executed, constraint_model, row["constraints"])

            rec = {
                "task_id": row["task_id"],
                "dataset": row["dataset"],
                "method": method,
                "oracle_path": oracle,
                "planned_path": planned,
                "executed_path": executed,
                "success": executed == oracle,
                "first_error_step": first_error_step(executed, oracle),
                "recovery_at_error": recovery_at_error(executed, oracle),
                "trap_at_1": trap_at_1(executed, trap),
                "feasible": feasible,
                "violations": violations or pred.get("violations", []),
                "plan_execution_consistency": plan_execution_consistency(planned, executed),
                "token_cost": len(executed) * (1 + pred.get("candidate_count", 1)),
                "latency_cost": pred.get("candidate_count", 1) * 0.02,
                "diversity_coverage": float(pred.get("candidate_unique_ratio", 1.0)),
                "replanning_steps": int(pred.get("replanning_steps", 1)),
            }
            by_method_records[method].append(rec)

    summary = {m: aggregate_method_metrics(recs) for m, recs in by_method_records.items()}
    return {
        "records": by_method_records,
        "summary": summary,
    }

