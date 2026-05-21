import random
from typing import Dict, List, Tuple


def _weighted_choice(weight_map: Dict[str, float], rng: random.Random, fallback: List[str]) -> str:
    if not weight_map:
        return rng.choice(fallback)
    items = list(weight_map.items())
    total = sum(max(v, 0.0) for _, v in items)
    if total <= 0:
        return rng.choice(fallback)
    x = rng.random() * total
    s = 0.0
    for k, v in items:
        s += max(v, 0.0)
        if s >= x:
            return k
    return items[-1][0]


def _sample_length(length_prob: Dict[str, float], rng: random.Random, default: int = 4) -> int:
    if not length_prob:
        return default
    mapped = {int(k): float(v) for k, v in length_prob.items()}
    token = _weighted_choice({str(k): v for k, v in mapped.items()}, rng, [str(default)])
    return int(token)


def path_value_score(path: List[str], query_tokens: List[str], value_model: Dict) -> float:
    weights = value_model.get("weights", {})
    assoc_w = float(weights.get("assoc", 1.0))
    prefix_w = float(weights.get("prefix", 0.8))
    len_w = float(weights.get("len", 0.3))
    trap_w = float(weights.get("trap", 0.5))

    assoc_prob = value_model.get("assoc_prob", {})
    trap_penalty = value_model.get("trap_penalty", {})

    assoc_score = 0.0
    for q in query_tokens:
        rel_map = assoc_prob.get(q, {})
        for rel in path:
            assoc_score += rel_map.get(rel, 0.0)

    prefix_hits = 0
    for i, rel in enumerate(path[: len(query_tokens)]):
        if rel == query_tokens[i] or query_tokens[i].startswith(rel):
            prefix_hits += 1
    prefix_score = prefix_hits / max(1, len(path))

    len_score = 1.0 - abs(len(path) - len(query_tokens)) / max(1, len(query_tokens))
    trap_score = sum(trap_penalty.get(rel, 0.0) for rel in path) / max(1, len(path))

    return assoc_w * assoc_score + prefix_w * prefix_score + len_w * len_score - trap_w * trap_score


def check_constraints(path: List[str], constraint_model: Dict, task_constraints: Dict) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    max_steps = min(
        int(task_constraints.get("max_steps", 6)),
        int(constraint_model.get("global_max_steps", 6)),
    )
    if len(path) > max_steps:
        violations.append("max_steps_exceeded")

    banned = set(task_constraints.get("banned_relations", []))
    allowed = set(constraint_model.get("allowed_relations", []))
    for rel in path:
        if rel in banned:
            violations.append("banned_relation")
        if allowed and rel not in allowed:
            violations.append("unknown_relation")
    return len(violations) == 0, sorted(set(violations))


def sample_candidate_path(
    query_tokens: List[str],
    planner_model: Dict,
    rng: random.Random,
    max_len: int = 6,
    use_value_guidance: bool = True,
    value_model: Dict | None = None,
) -> List[str]:
    token_prob = planner_model.get("token_prob", {})
    vocab = list(token_prob.keys()) or list(set(query_tokens)) or ["rel_default"]

    length = min(_sample_length(planner_model.get("length_prob", {}), rng), max_len)
    start_prob = dict(planner_model.get("start_prob", {}))
    if query_tokens:
        start_prob[query_tokens[0]] = start_prob.get(query_tokens[0], 0.0) + 0.35
    path = [_weighted_choice(start_prob, rng, vocab)]

    bigram_prob = planner_model.get("bigram_prob", {})
    noise_eps = float(planner_model.get("noise_eps", 0.20))
    for _ in range(1, length):
        prev = path[-1]
        if rng.random() < noise_eps:
            nxt = rng.choice(vocab)
        else:
            nxt = _weighted_choice(bigram_prob.get(prev, token_prob), rng, vocab)
        path.append(nxt)

    if use_value_guidance and value_model:
        best = path
        best_score = path_value_score(path, query_tokens, value_model)
        for _ in range(2):
            alt = list(path)
            i = rng.randrange(len(alt))
            alt[i] = rng.choice(vocab)
            score = path_value_score(alt, query_tokens, value_model)
            if score > best_score:
                best = alt
                best_score = score
        return best
    return path


def execute_with_receding_horizon(
    query_tokens: List[str],
    planner_model: Dict,
    value_model: Dict,
    constraint_model: Dict,
    task_constraints: Dict,
    rng: random.Random,
    num_candidates: int = 16,
    use_value_guidance: bool = True,
    use_constraint_checker: bool = True,
    receding_horizon: bool = True,
) -> Dict:
    target_len = min(max(2, len(query_tokens)), int(task_constraints.get("max_steps", 6)))
    executed: List[str] = []
    all_candidates: List[List[str]] = []
    all_violations: List[str] = []

    if not receding_horizon:
        cands = [
            sample_candidate_path(
                query_tokens=query_tokens,
                planner_model=planner_model,
                rng=rng,
                max_len=target_len,
                use_value_guidance=use_value_guidance,
                value_model=value_model,
            )
            for _ in range(num_candidates)
        ]
        all_candidates.extend(cands)
        scored = []
        for c in cands:
            feasible, violations = check_constraints(c, constraint_model, task_constraints)
            if use_constraint_checker and not feasible:
                all_violations.extend(violations)
                continue
            scored.append((path_value_score(c, query_tokens, value_model), c))
        executed = max(scored, key=lambda x: x[0])[1] if scored else cands[0]
        return {
            "planned_path": executed,
            "executed_path": executed,
            "candidate_count": len(cands),
            "candidate_unique_ratio": len({tuple(x) for x in cands}) / max(1, len(cands)),
            "violations": sorted(set(all_violations)),
            "replanning_steps": 1,
        }

    for _step in range(target_len):
        remaining_query = query_tokens[len(executed) :] if len(executed) < len(query_tokens) else []
        query_for_step = remaining_query if remaining_query else query_tokens
        cands = [
            sample_candidate_path(
                query_tokens=query_for_step,
                planner_model=planner_model,
                rng=rng,
                max_len=target_len - len(executed),
                use_value_guidance=use_value_guidance,
                value_model=value_model,
            )
            for _ in range(num_candidates)
        ]
        all_candidates.extend(cands)
        scored = []
        for c in cands:
            candidate_path = executed + c
            feasible, violations = check_constraints(candidate_path, constraint_model, task_constraints)
            if use_constraint_checker and not feasible:
                all_violations.extend(violations)
                continue
            score = path_value_score(candidate_path, query_tokens, value_model)
            scored.append((score, c))
        if not scored:
            break
        best_next = max(scored, key=lambda x: x[0])[1]
        executed.append(best_next[0])

    return {
        "planned_path": executed,
        "executed_path": executed,
        "candidate_count": len(all_candidates),
        "candidate_unique_ratio": len({tuple(x) for x in all_candidates}) / max(1, len(all_candidates)),
        "violations": sorted(set(all_violations)),
        "replanning_steps": max(1, len(executed)),
    }

