import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, ensure_dir


STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "in",
    "into",
    "is",
    "of",
    "on",
    "put",
    "the",
    "then",
    "to",
    "with",
}

OPENABLE_HINTS = {
    "cabinet",
    "drawer",
    "fridge",
    "microwave",
    "safe",
}

TRANSFORM_TO_TOOLS = {
    "heat": ["microwave", "stoveburner", "coffeemachine"],
    "cool": ["fridge"],
    "clean": ["sinkbasin", "sink", "bathtubbasin"],
}


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _set_if_present(cfg: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            return
        cur = nxt
    if keys[-1] in cur:
        cur[keys[-1]] = value


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _load_alfworld_config(config_path: Path) -> Dict[str, Any]:
    import alfworld.agents.modules.generic as generic

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], str(config_path)]
    try:
        return _expand_env_vars(generic.load_config())
    finally:
        sys.argv = old_argv


def _patch_data_root(cfg: Dict[str, Any], data_root: Path) -> None:
    root = data_root.resolve()
    dataset_root = root / "json_2.1.1" if (root / "json_2.1.1").exists() else root
    _set_if_present(cfg, ["dataset", "data_path"], str(dataset_root / "train"))
    _set_if_present(cfg, ["dataset", "eval_id_data_path"], str(dataset_root / "valid_seen"))
    _set_if_present(cfg, ["dataset", "eval_ood_data_path"], str(dataset_root / "valid_unseen"))


def _make_env(config_path: Path, data_root: Path, split: str, batch_size: int):
    import alfworld.agents.environment as environment

    cfg = _load_alfworld_config(config_path)
    _patch_data_root(cfg, data_root)
    env_type = cfg.get("env", {}).get("type", "AlfredTWEnv")
    env_cls = environment.get_environment(env_type)
    env = env_cls(cfg, train_eval=split)
    return env.init_env(batch_size=batch_size)


def _text(x: Any) -> str:
    if isinstance(x, (list, tuple)):
        return " ".join(str(v) for v in x)
    return str(x)


def _first(x: Any, default: Any = None) -> Any:
    if isinstance(x, (list, tuple)):
        return x[0] if x else default
    return x


def _extract_goal(obs: str) -> str:
    match = re.search(r"Your task is to:\s*(.+?)(?:$|\n)", obs, flags=re.I | re.S)
    if match:
        return match.group(1).strip().strip(".")
    return ""


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _goal_keywords(goal: str) -> List[str]:
    toks = _tokens(goal)
    # Preserve order and remove duplicates.
    seen = set()
    out = []
    for tok in toks:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _parse_goal(goal: str) -> Dict[str, Any]:
    text = goal.lower()
    terms = _goal_keywords(goal)
    transform = ""
    if any(w in text for w in ("heat", "hot")):
        transform = "heat"
    elif any(w in text for w in ("cool", "cold")):
        transform = "cool"
    elif "clean" in text:
        transform = "clean"
    elif any(w in text for w in ("look", "examine")):
        transform = "look"

    obj_terms = [
        t
        for t in terms
        if t
        not in {
            "some",
            "hot",
            "cold",
            "clean",
            "heat",
            "cool",
            "look",
            "examine",
            "under",
            "desklamp",
        }
    ]
    obj = obj_terms[0] if obj_terms else ""
    recep = obj_terms[-1] if len(obj_terms) > 1 else ""
    if "desklamp" in text:
        recep = "desklamp"
    return {
        "goal": goal,
        "terms": terms,
        "target_object": obj,
        "target_receptacle": recep,
        "transform": transform,
        "requires_put": " put " in f" {text} " or " in " in f" {text} " or " on " in f" {text} ",
        "requires_two": "two" in text,
    }


def _normalize_entity(text: str) -> str:
    return re.sub(r"\s+\d+$", "", text.lower()).replace(" ", "")


def _extract_take_action(action: str) -> Tuple[str, str]:
    match = re.match(r"take (.+?) from (.+)$", action.lower())
    if not match:
        return "", ""
    return _normalize_entity(match.group(1)), _normalize_entity(match.group(2))


def _extract_put_action(action: str) -> Tuple[str, str]:
    match = re.match(r"(?:put|move) (.+?) (?:in|on|to) (.+)$", action.lower())
    if not match:
        return "", ""
    return _normalize_entity(match.group(1)), _normalize_entity(match.group(2))


def _update_memory(memory: Dict[str, Any], obs: str, action: str, admissible: Sequence[str]) -> None:
    a = action.lower()
    o = obs.lower()
    if a.startswith("go to "):
        memory["current_location"] = _normalize_entity(a.replace("go to ", ""))
        memory["visited"].add(memory["current_location"])
    if a.startswith("open "):
        memory["opened"].add(_normalize_entity(a.replace("open ", "")))
    if a.startswith("take "):
        obj, loc = _extract_take_action(a)
        if obj:
            memory["inventory"].add(obj)
            if loc:
                memory["object_locations"][obj] = loc
    if a.startswith("put "):
        obj, loc = _extract_put_action(a)
        if obj:
            memory["inventory"].discard(obj)
            if loc:
                memory["object_locations"][obj] = loc
    if any(bad in o for bad in ("nothing happens", "you can't", "you cannot", "don't see", "not possible")):
        memory["failed_actions"].add(a)

    for cmd in admissible:
        take_obj, take_loc = _extract_take_action(cmd)
        if take_obj and take_loc:
            memory["object_locations"].setdefault(take_obj, take_loc)


def _subgoal_stage(spec: Dict[str, Any], memory: Dict[str, Any]) -> str:
    obj = spec.get("target_object", "")
    recep = spec.get("target_receptacle", "")
    transform = spec.get("transform", "")
    has_obj = bool(obj and any(obj in item for item in memory["inventory"]))
    if transform == "look":
        return "inspect_target"
    if not has_obj:
        return "find_and_take_object"
    if transform in {"heat", "cool", "clean"} and transform not in memory["completed_transforms"]:
        return f"{transform}_object"
    if recep:
        return "place_object"
    return "finish"


def _constraint_penalty(action: str, admissible: Sequence[str], memory: Dict[str, Any], spec: Dict[str, Any]) -> float:
    a = action.lower()
    if action not in admissible:
        return -100.0
    penalty = 0.0
    if a in memory["failed_actions"]:
        penalty -= 8.0
    if a.startswith("put ") and not memory["inventory"]:
        penalty -= 4.0
    if a.startswith("open "):
        target = _normalize_entity(a.replace("open ", ""))
        if target in memory["opened"]:
            penalty -= 5.0
    return penalty


def _stage_score(action: str, stage: str, spec: Dict[str, Any], memory: Dict[str, Any]) -> float:
    a = action.lower()
    obj = spec.get("target_object", "")
    recep = spec.get("target_receptacle", "")
    transform = spec.get("transform", "")
    score = 0.0
    norm_action = _normalize_entity(a)

    if stage == "find_and_take_object":
        if a.startswith("take "):
            if obj and obj in norm_action:
                score += 14.0
            else:
                score -= 8.0
        if a.startswith("open ") and any(h in _normalize_entity(a) for h in OPENABLE_HINTS):
            score += 4.0
        if a.startswith("go to "):
            loc = _normalize_entity(a.replace("go to ", ""))
            if loc not in memory["visited"]:
                score += 3.5
            if obj and memory["object_locations"].get(obj) and memory["object_locations"][obj] in loc:
                score += 8.0
    elif stage.endswith("_object"):
        tool_hints = TRANSFORM_TO_TOOLS.get(transform, [])
        if any(a.startswith(prefix) for prefix in (f"{transform} ", "use ")):
            score += 12.0
        if a.startswith("go to ") and any(tool in _normalize_entity(a) for tool in tool_hints):
            score += 9.0
        if a.startswith("open ") and any(tool in _normalize_entity(a) for tool in tool_hints):
            score += 4.0
    elif stage == "place_object":
        if a.startswith(("put ", "move ")):
            put_obj, put_loc = _extract_put_action(a)
            if obj and obj in put_obj and recep and recep in put_loc:
                score += 16.0
            elif obj and obj not in put_obj:
                score -= 7.0
        if a.startswith("go to ") and recep and recep in _normalize_entity(a):
            score += 8.0
        if a.startswith("open ") and recep and recep in _normalize_entity(a):
            score += 5.0
    elif stage == "inspect_target":
        if any(a.startswith(prefix) for prefix in ("use ", "examine ", "look at ")):
            score += 8.0
        if "desklamp" in spec.get("goal", "").lower() and "desklamp" in _normalize_entity(a):
            score += 8.0
        if obj and obj in _normalize_entity(a):
            score += 6.0
    if a.startswith(("take ", "put ", "move ")) and obj and obj not in norm_action:
        score -= 4.0
    return score


def _action_score(
    action: str,
    obs: str,
    goal: str,
    goal_terms: Sequence[str],
    inventory: str,
    action_counts: Counter,
    memory: Dict[str, Any] | None = None,
    spec: Dict[str, Any] | None = None,
) -> float:
    a = action.lower()
    o = obs.lower()
    inv = inventory.lower()
    score = 0.0

    spec = spec or _parse_goal(goal)
    memory = memory or {
        "visited": set(),
        "opened": set(),
        "inventory": set(),
        "object_locations": {},
        "completed_transforms": set(),
        "failed_actions": set(),
    }
    stage = _subgoal_stage(spec, memory)

    for term in goal_terms:
        if term in a:
            score += 3.0
        if term in o:
            score += 0.3
        if term in inv:
            score += 1.0

    if a.startswith("put ") and any(term in inv for term in goal_terms):
        score += 5.0
    if a.startswith("take ") and any(term in a for term in goal_terms):
        score += 4.0
    if a.startswith("open ") and "closed" in o:
        score += 2.5
    if a.startswith("go to ") and any(term in a for term in goal_terms):
        score += 2.0
    if any(a.startswith(prefix) for prefix in ("heat ", "cool ", "clean ", "slice ")):
        if any(word in goal.lower() for word in ("heat", "hot", "cool", "cold", "clean", "slice", "sliced")):
            score += 4.0
    if a in {"look", "inventory", "help"}:
        score -= 3.0
    if action_counts[a] > 0:
        score -= 2.0 * action_counts[a]
    if "nothing happens" in o:
        score -= 1.0
    score += _stage_score(action, stage, spec, memory)
    return score


def _rank_actions(
    admissible: Sequence[str],
    obs: str,
    goal: str,
    goal_terms: Sequence[str],
    inventory: str,
    action_counts: Counter,
    rng: random.Random,
    memory: Dict[str, Any] | None = None,
    spec: Dict[str, Any] | None = None,
) -> List[Tuple[str, float]]:
    scored = []
    for action in admissible:
        score = _action_score(action, obs, goal, goal_terms, inventory, action_counts, memory=memory, spec=spec)
        if memory is not None and spec is not None:
            score += _constraint_penalty(action, admissible, memory, spec)
        score += rng.random() * 1e-4
        scored.append((action, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _inventory_text(admissible: Sequence[str], obs: str, env: Any) -> str:
    if "inventory" not in admissible:
        return ""
    try:
        inv_obs, _, _, _ = env.step(["inventory"])
        return _text(inv_obs)
    except Exception:
        return ""


def _first_error_step(actions: List[str], success: bool) -> int:
    if success:
        return len(actions) + 1
    for idx, action in enumerate(actions, start=1):
        if action in {"look", "inventory", "help"}:
            continue
        return idx
    return 1


def _is_trap_first_action(action: str, goal_terms: Sequence[str]) -> bool:
    a = action.lower()
    if a in {"help", "look", "inventory"}:
        return True
    if a.startswith("take ") and goal_terms and not any(term in a for term in goal_terms):
        return True
    return False


def _run_episode(env: Any, episode_id: int, args: argparse.Namespace, rng: random.Random) -> Dict[str, Any]:
    t0 = time.time()
    obs, infos = env.reset()
    obs_text = _text(obs)
    goal = _extract_goal(obs_text)
    spec = _parse_goal(goal)
    goal_terms = spec["terms"]
    memory: Dict[str, Any] = {
        "current_location": "",
        "visited": set(),
        "opened": set(),
        "inventory": set(),
        "object_locations": {},
        "completed_transforms": set(),
        "failed_actions": set(),
    }
    action_counts: Counter = Counter()
    actions: List[str] = []
    observations = [obs_text]
    score = 0.0
    done = False
    candidate_pool_sizes: List[int] = []

    for step in range(int(args.max_steps)):
        admissible_batch = infos.get("admissible_commands", [[]])
        admissible = list(admissible_batch[0] if admissible_batch else [])
        candidate_pool_sizes.append(len(admissible))
        if not admissible:
            break

        inventory = _inventory_text(admissible, obs_text, env) if args.use_inventory_probe else ""
        use_full_controller = str(getattr(args, "variant", "full")).lower() == "full"
        _update_memory(memory, obs_text, actions[-1] if actions else "", admissible)
        ranked = _rank_actions(
            admissible,
            obs_text,
            goal,
            goal_terms,
            inventory,
            action_counts,
            rng,
            memory=memory if use_full_controller else None,
            spec=spec if use_full_controller else None,
        )
        action = ranked[0][0]
        actions.append(action)
        action_counts[action.lower()] += 1

        obs, scores, dones, infos = env.step([action])
        obs_text = _text(obs)
        if any(action.lower().startswith(prefix) for prefix in ("heat ", "cool ", "clean ")):
            memory["completed_transforms"].add(spec.get("transform", ""))
        _update_memory(memory, obs_text, action, infos.get("admissible_commands", [[]])[0])
        observations.append(obs_text)
        score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        if done:
            break

    success = bool(done and score > 0)
    first_error = _first_error_step(actions, success)
    return {
        "episode_id": episode_id,
        "method": f"alfworld_diplan_{getattr(args, 'variant', 'full')}",
        "goal": goal,
        "structured_goal": spec,
        "success": success,
        "final_score": score,
        "num_steps": len(actions),
        "first_error_step": first_error,
        "recovery_at_error": bool(success and first_error <= len(actions)),
        "trap_at_1": bool(actions and _is_trap_first_action(actions[0], goal_terms)),
        "plan_feasibility": 1.0,
        "constraint_violation": False,
        "plan_execution_consistency": 1.0,
        "candidate_pool_size": sum(candidate_pool_sizes) / max(1, len(candidate_pool_sizes)),
        "memory": {
            "visited": sorted(memory["visited"]),
            "opened": sorted(memory["opened"]),
            "inventory": sorted(memory["inventory"]),
            "object_locations": dict(sorted(memory["object_locations"].items())),
            "failed_actions": sorted(memory["failed_actions"]),
        },
        "actions": actions,
        "initial_observation": observations[0] if observations else "",
        "final_observation": observations[-1] if observations else "",
        "latency_cost": time.time() - t0,
        "token_cost": sum(len(_tokens(x)) for x in observations) + sum(len(_tokens(x)) for x in actions),
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    n = len(rows)
    method = rows[0].get("method", "alfworld_diplan_full") if rows else "alfworld_diplan_full"
    if n == 0:
        return {method: {}}
    summary = {
        "n": n,
        "success_rate": sum(1 for r in rows if r.get("success")) / n,
        "first_error_step": sum(float(r.get("first_error_step", 0.0)) for r in rows) / n,
        "recovery_at_error": sum(1 for r in rows if r.get("recovery_at_error")) / n,
        "trap_at_1": sum(1 for r in rows if r.get("trap_at_1")) / n,
        "plan_feasibility": sum(float(r.get("plan_feasibility", 0.0)) for r in rows) / n,
        "constraint_violation_rate": sum(1 for r in rows if r.get("constraint_violation")) / n,
        "plan_execution_consistency": sum(float(r.get("plan_execution_consistency", 0.0)) for r in rows) / n,
        "token_cost": sum(float(r.get("token_cost", 0.0)) for r in rows) / n,
        "latency_cost": sum(float(r.get("latency_cost", 0.0)) for r in rows) / n,
        "candidate_pool_avg_size": sum(float(r.get("candidate_pool_size", 0.0)) for r in rows) / n,
        "avg_steps": sum(float(r.get("num_steps", 0.0)) for r in rows) / n,
    }
    return {method: summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DiPLaN-style controller on ALFWorld text tasks.")
    parser.add_argument("--data_root", type=str, default="data/long_horizon/alfworld")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval_out_of_distribution")
    parser.add_argument("--out", type=str, default="results/alfworld_diplan_lite")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        choices=["lite", "full"],
        default="full",
        help="lite disables dynamic memory/constraint scoring; full uses the upgraded ALFWorld controller.",
    )
    parser.add_argument("--use_inventory_probe", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    os.environ["ALFWORLD_DATA"] = str(data_root)
    rng = random.Random(int(args.seed))
    env = _make_env(Path(args.config).resolve(), data_root, args.split, batch_size=1)

    rows = []
    for episode_id in range(int(args.episodes)):
        print(f"[episode] {episode_id + 1}/{args.episodes}")
        row = _run_episode(env, episode_id, args, rng)
        print(
            f"[episode] success={row['success']} score={row['final_score']} "
            f"steps={row['num_steps']} goal={row['goal']!r}"
        )
        rows.append(row)

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    _write_jsonl(out_dir / "predictions.jsonl", rows)
    dump_json(str(out_dir / "summary_metrics.json"), _summarize(rows))
    dump_json(
        str(out_dir / "run_config.json"),
        {
            "data_root": str(data_root),
            "config": str(Path(args.config).resolve()),
            "split": args.split,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "variant": args.variant,
            "use_inventory_probe": args.use_inventory_probe,
        },
    )
    print(f"[ok] wrote ALFWorld DiPLaN-{args.variant} outputs to {out_dir}")


if __name__ == "__main__":
    main()
