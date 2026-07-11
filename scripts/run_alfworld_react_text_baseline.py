"""ALFWorld ReAct-style text-interface baseline for efficiency diagnostics.

This baseline intentionally uses the expensive planner-executor interface that
DiPLaN is meant to avoid: every step sends observation/history/admissible
commands to an LLM, receives free-form text, parses it back into an executable
ALFWorld command, and then steps the environment.

The goal is not to be the strongest ALFWorld agent. It is a measurement tool for
the Textual Plan Bottleneck:
  * repeated online LLM calls,
  * prompt/output token cost,
  * latency,
  * parse failures / invalid actions before grounding,
  * plan-execution consistency after grounding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_alfworld_diplan_agent import (  # noqa: E402
    _extract_goal,
    _first,
    _first_failure_step,
    _first_error_step,
    _has_failure_feedback,
    _is_trap_first_action,
    _make_env,
    _parse_goal,
    _rank_actions,
    _text,
    _tokens,
    _update_memory,
)
from src.diplan.io_utils import dump_json, ensure_dir  # noqa: E402
from src.diplan.llm_client import LLMClient, LLMConfig, LLMError  # noqa: E402


UTILITY_COMMANDS = {"look", "inventory", "help"}


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _estimate_tokens(text: str) -> int:
    # Same rough tokenizer for all methods; sufficient for relative cost.
    return len(re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE))


def _build_prompt(goal: str, obs: str, admissible: Sequence[str], history: Sequence[Tuple[str, str]], max_history: int) -> str:
    recent = list(history)[-max(0, int(max_history)) :]
    hist_lines = []
    for i, (action, ob) in enumerate(recent, start=max(1, len(history) - len(recent) + 1)):
        short_obs = " ".join(str(ob).split())[:500]
        hist_lines.append(f"{i}. Action: {action}\n   Observation: {short_obs}")
    history_block = "\n".join(hist_lines) if hist_lines else "None"
    action_block = "\n".join(f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible))
    return (
        "You are controlling an ALFWorld text environment.\n"
        "Choose exactly one executable command from the numbered admissible commands.\n"
        "Think briefly if needed, but the final line must be: ACTION: <exact command>\n\n"
        f"Task: {goal}\n\n"
        f"Current observation:\n{obs}\n\n"
        f"Recent history:\n{history_block}\n\n"
        f"Admissible commands:\n{action_block}\n\n"
        "Return only a short rationale and the final ACTION line."
    )


def _extract_action(raw: str, admissible: Sequence[str]) -> Tuple[str, bool, bool]:
    """Return (surface_action, parse_failed, invalid_before_grounding)."""
    text = str(raw).strip()
    lowered = text.lower()
    adm = list(admissible)
    adm_lower = {a.lower(): a for a in adm}

    # Preferred format: ACTION: exact command.
    matches = re.findall(r"action\s*:\s*(.+)", text, flags=re.I)
    candidates = []
    if matches:
        candidates.extend(m.strip().strip("`'\". ") for m in matches)

    # Also accept a bare exact command on any line.
    for line in text.splitlines():
        line = line.strip().strip("`'\". ")
        if line:
            candidates.append(line)

    for cand in candidates:
        key = cand.lower()
        if key in adm_lower:
            return adm_lower[key], False, False

    # Loose substring match: output contains one of the admissible commands.
    for cmd in sorted(adm, key=len, reverse=True):
        if cmd.lower() in lowered:
            return cmd, False, True

    parsed = candidates[0] if candidates else text
    return parsed, True, True


def _ground_action(
    parsed: str,
    admissible: Sequence[str],
    fallback_policy: str,
    obs: str,
    goal: str,
    spec: Dict[str, Any],
    memory: Dict[str, Any],
    action_counts: Counter,
    rng: random.Random,
) -> str:
    adm = list(admissible)
    if not adm:
        return ""
    adm_lower = {a.lower(): a for a in adm}
    if str(parsed).lower() in adm_lower:
        return adm_lower[str(parsed).lower()]

    policy = str(fallback_policy).lower()
    if policy == "look" and "look" in adm_lower:
        return adm_lower["look"]
    if policy == "first":
        useful = [a for a in adm if a.lower() not in UTILITY_COMMANDS]
        return useful[0] if useful else adm[0]

    ranked = _rank_actions(
        adm,
        obs,
        goal,
        spec.get("terms", []),
        "",
        action_counts,
        rng,
        memory=memory,
        spec=spec,
    )
    return ranked[0][0] if ranked else adm[0]


def _new_memory() -> Dict[str, Any]:
    return {
        "current_location": "",
        "visited": set(),
        "opened": set(),
        "inventory": set(),
        "object_locations": {},
        "completed_transforms": set(),
        "failed_actions": set(),
    }


def _run_episode(env: Any, episode_id: int, args: argparse.Namespace, client: LLMClient, rng: random.Random) -> Dict[str, Any]:
    t0 = time.time()
    obs, infos = env.reset()
    obs_text = _text(obs)
    goal = _extract_goal(obs_text)
    spec = _parse_goal(goal)
    memory = _new_memory()
    history: List[Tuple[str, str]] = []
    raw_actions: List[str] = []
    model_outputs: List[str] = []
    parsed_actions: List[str] = []
    post_action_observations: List[str] = []
    candidate_pool_sizes: List[int] = []
    parse_failures = 0
    invalid_before_grounding = 0
    consistency_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    action_counts: Counter = Counter()
    score = 0.0
    done = False
    llm_calls_before = client.calls
    llm_errors_before = client.errors

    for _step in range(int(args.max_steps)):
        admissible_batch = infos.get("admissible_commands", [[]])
        admissible = list(admissible_batch[0] if admissible_batch else [])
        candidate_pool_sizes.append(len(admissible))
        if not admissible:
            break
        _update_memory(memory, obs_text, raw_actions[-1] if raw_actions else "", admissible)

        prompt = _build_prompt(goal, obs_text, admissible, history, int(args.max_history))
        prompt_tokens += _estimate_tokens(prompt)
        system_prompt = (
            "You are a careful ReAct-style ALFWorld agent. "
            "You must choose one currently admissible command exactly."
        )
        try:
            raw = client.chat(system_prompt, prompt)
        except LLMError as exc:
            raw = f"ACTION: look\n# LLM_ERROR: {exc}"
            parse_failures += 1
            invalid_before_grounding += 1
        completion_tokens += _estimate_tokens(raw)
        model_outputs.append(raw)

        parsed, parse_failed, invalid = _extract_action(raw, admissible)
        parse_failures += int(parse_failed)
        invalid_before_grounding += int(invalid)
        action = _ground_action(
            parsed,
            admissible,
            str(args.fallback_policy),
            obs_text,
            goal,
            spec,
            memory,
            action_counts,
            rng,
        )
        parsed_actions.append(str(parsed))
        if str(parsed).lower() == str(action).lower():
            consistency_hits += 1

        raw_actions.append(action)
        action_counts[action.lower()] += 1
        obs, scores, dones, infos = env.step([action])
        obs_text = _text(obs)
        if any(action.lower().startswith(prefix) for prefix in ("heat ", "cool ", "clean ")):
            memory["completed_transforms"].add(spec.get("transform", ""))
        _update_memory(memory, obs_text, action, infos.get("admissible_commands", [[]])[0])
        history.append((action, obs_text))
        post_action_observations.append(obs_text)
        score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        if done:
            break

    success = bool(done and score > 0)
    steps = len(raw_actions)
    llm_calls = client.calls - llm_calls_before
    llm_errors = client.errors - llm_errors_before
    return {
        "task_id": f"alfworld_{episode_id:07d}",
        "episode_id": episode_id,
        "method": "alfworld_react_text",
        "goal": goal,
        "structured_goal": spec,
        "success": success,
        "final_score": score,
        "num_steps": steps,
        "first_error_step": _first_failure_step(post_action_observations) if not success else len(raw_actions) + 1,
        "recovery_at_error": bool(success and any(_has_failure_feedback(obs) for obs in post_action_observations)),
        "trap_at_1": bool(raw_actions and _is_trap_first_action(raw_actions[0], spec.get("terms", []))),
        "plan_feasibility": 1.0 - invalid_before_grounding / max(1, steps),
        "constraint_violation": any(_has_failure_feedback(obs) for obs in post_action_observations),
        "plan_execution_consistency": consistency_hits / max(1, steps),
        "parse_failure_rate": parse_failures / max(1, steps),
        "invalid_action_rate": invalid_before_grounding / max(1, steps),
        "llm_calls": llm_calls,
        "llm_errors": llm_errors,
        "llm_calls_per_step": llm_calls / max(1, steps),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "token_cost": prompt_tokens + completion_tokens,
        "latency_cost": time.time() - t0,
        "candidate_pool_size": sum(candidate_pool_sizes) / max(1, len(candidate_pool_sizes)),
        "actions": raw_actions,
        "parsed_actions": parsed_actions,
        "model_outputs": model_outputs if bool(args.save_model_outputs) else [],
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    n = len(rows)
    if n == 0:
        return {"alfworld_react_text": {}}
    steps = sum(float(r.get("num_steps", 0.0)) for r in rows)
    s = {
        "n": n,
        "success_rate": sum(1 for r in rows if r.get("success")) / n,
        "first_error_step": sum(float(r.get("first_error_step", 0.0)) for r in rows) / n,
        "trap_at_1": sum(1 for r in rows if r.get("trap_at_1")) / n,
        "plan_feasibility": sum(float(r.get("plan_feasibility", 0.0)) for r in rows) / n,
        "constraint_violation_rate": sum(1 for r in rows if r.get("constraint_violation")) / n,
        "plan_execution_consistency": sum(float(r.get("plan_execution_consistency", 0.0)) for r in rows) / n,
        "parse_failure_rate": sum(float(r.get("parse_failure_rate", 0.0)) for r in rows) / n,
        "invalid_action_rate": sum(float(r.get("invalid_action_rate", 0.0)) for r in rows) / n,
        "llm_calls": sum(float(r.get("llm_calls", 0.0)) for r in rows) / n,
        "llm_calls_total": sum(float(r.get("llm_calls", 0.0)) for r in rows),
        "llm_errors": sum(float(r.get("llm_errors", 0.0)) for r in rows) / n,
        "llm_calls_per_step": sum(float(r.get("llm_calls", 0.0)) for r in rows) / max(1.0, steps),
        "prompt_tokens": sum(float(r.get("prompt_tokens", 0.0)) for r in rows) / n,
        "completion_tokens": sum(float(r.get("completion_tokens", 0.0)) for r in rows) / n,
        "token_cost": sum(float(r.get("token_cost", 0.0)) for r in rows) / n,
        "latency_cost": sum(float(r.get("latency_cost", 0.0)) for r in rows) / n,
        "candidate_pool_avg_size": sum(float(r.get("candidate_pool_size", 0.0)) for r in rows) / n,
        "avg_steps": steps / n,
    }
    return {"alfworld_react_text": s}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReAct-style text-interface baseline on ALFWorld.")
    parser.add_argument("--data_root", type=str, default="data/long_horizon/alfworld")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval_out_of_distribution")
    parser.add_argument("--out", type=str, default="results/alfworld_react_text")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_history", type=int, default=6)
    parser.add_argument("--fallback_policy", choices=["heuristic", "first", "look"], default="heuristic")
    parser.add_argument("--save_model_outputs", action="store_true")
    parser.add_argument("--llm_api_base", type=str, default=os.environ.get("TOG_OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm_api_key", type=str, default=os.environ.get("TOG_OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm_model", type=str, default=os.environ.get("TOG_OPENAI_MODEL", "Llama-3.1-8B-Instruct"))
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_tokens", type=int, default=96)
    parser.add_argument("--llm_timeout_s", type=int, default=60)
    parser.add_argument("--llm_retries", type=int, default=2)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    os.environ["ALFWORLD_DATA"] = str(data_root)
    rng = random.Random(int(args.seed))
    env = _make_env(Path(args.config).resolve(), data_root, args.split, batch_size=1)
    client = LLMClient(
        LLMConfig(
            api_base=args.llm_api_base,
            api_key=args.llm_api_key,
            model=args.llm_model,
            temperature=float(args.llm_temperature),
            max_tokens=int(args.llm_max_tokens),
            timeout_s=int(args.llm_timeout_s),
            retries=int(args.llm_retries),
        )
    )

    rows = []
    for episode_id in range(int(args.episodes)):
        print(f"[episode] {episode_id + 1}/{args.episodes}", flush=True)
        row = _run_episode(env, episode_id, args, client, rng)
        print(
            f"[episode] success={row['success']} steps={row['num_steps']} "
            f"llm_calls={row['llm_calls']} parse_fail={row['parse_failure_rate']:.2f} "
            f"invalid={row['invalid_action_rate']:.2f} token={row['token_cost']} "
            f"goal={row['goal']!r}",
            flush=True,
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
            "max_history": args.max_history,
            "fallback_policy": args.fallback_policy,
            "llm_api_base": args.llm_api_base,
            "llm_model": args.llm_model,
            "llm_temperature": args.llm_temperature,
            "llm_max_tokens": args.llm_max_tokens,
            "llm_timeout_s": args.llm_timeout_s,
            "llm_retries": args.llm_retries,
        },
    )
    print(f"[ok] wrote ReAct text baseline outputs to {out_dir}")


if __name__ == "__main__":
    main()
