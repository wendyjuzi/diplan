"""ALFWorld DiPLaN executor: real diffusion plan generation + receding-horizon.

Unlike ``scripts/run_alfworld_diplan_agent.py`` (a heuristic greedy scorer with
hardcoded feasibility metrics), this executor runs the trained DiPLaN stack:

    Planning State Encoder (goal + scene + executed prefix)
      -> Diffusion Plan Generator (N latent plans)
      -> Plan decode + Value-Guided reranking + learned Constraint penalty
      -> Tool Decoder / Feasibility Projection onto admissible commands (§5.4)
      -> Instance grounding: the abstract head (e.g. GOTO::cabinet) is instance-free,
         so among concrete commands matching it we commit the least-tried instance,
         letting the neural subgoal drive systematic search instead of looping
      -> commit only the head action, observe, re-plan (receding horizon §5.5)

All mechanism metrics are MEASURED (not hardcoded):
  plan_feasibility            = projectable heads / committed steps
  constraint_violation        = executed plan violates the rule checker / repeats a failure
  plan_execution_consistency  = executed action token == the planned head token

MUST run on the ALFWorld server with ckpts trained via the alfworld configs.

Example:
    export ALFWORLD_DATA=/root/autodl-tmp/DiPLaN/data/long_horizon/alfworld
    python scripts/run_alfworld_diplan_diffusion.py \
        --data_root "$ALFWORLD_DATA" --config "$ALFWORLD_DATA/base_config.tw.yaml" \
        --split eval_out_of_distribution --episodes 20 --max_steps 50 \
        --ae_ckpt runs/ae_alfworld/best.pt --planner_ckpt runs/diff_alfworld/best.pt \
        --value_ckpt runs/value_alfworld/best.pt --constraint_ckpt runs/constraint_alfworld/best.pt \
        --out results/alfworld_diplan_diffusion_ood20
"""

import argparse
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_alfworld_diplan_agent import (  # noqa: E402
    _extract_goal,
    _first,
    _first_error_step,
    _goal_keywords,
    _is_trap_first_action,
    _make_env,
    _normalize_entity,
    _parse_goal,
    _rank_actions,
    _text,
    _update_memory,
)

from src.diplan.alfworld_plan import (  # noqa: E402
    goal_query_tokens,
    match_plan_token_to_admissible,
    normalize_action,
    parse_action,
    parse_action_token,
    receptacle_types_from_admissible,
)
from src.diplan.constraints import is_feasible  # noqa: E402
from src.diplan.inference import (  # noqa: E402
    constraint_violation_scores,
    sample_plan_candidates,
    score_candidates_with_value,
)
from src.diplan.io_utils import dump_json, ensure_dir  # noqa: E402
from src.diplan.metrics import plan_execution_consistency  # noqa: E402
from src.diplan.torch_pipeline import (  # noqa: E402
    ConstraintModel,
    DiffusionPlanner,
    MLPPlanner,
    PathAutoencoder,
    ValueRanker,
    load_vocab,
)


TRANSFORM_STAGE = {"heat": "HEAT", "cool": "COOL", "clean": "CLEAN"}
TRANSFORM_TOOLS = {
    "heat": {"microwave", "stoveburner", "coffeemachine"},
    "cool": {"fridge"},
    "clean": {"sinkbasin", "sink", "bathtubbasin"},
}
UTILITY_COMMANDS = {"look", "help", "inventory"}
FAILURE_MARKERS = (
    "nothing happens",
    "you can't",
    "you cannot",
    "don't see",
    "not possible",
    "not holding",
    "already open",
)


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


def _exec_constraints(spec: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {"max_steps": max_steps, "banned_relations": [], "forbidden_actions": []}
    transform = spec.get("transform", "")
    if transform in TRANSFORM_STAGE and spec.get("requires_put", False):
        constraints["required_before"] = {"PUT": [TRANSFORM_STAGE[transform]]}
    return constraints


def _state_condition_tokens(goal: str, spec: Dict[str, Any], admissible: List[str], memory: Dict[str, Any]) -> List[str]:
    """Compact symbolic state stream for the diffusion/value condition.

    Old ALFWorld checkpoints will map these unseen tokens to <unk>, so this is
    mainly activated by retraining on the state-aware executable trajectory rows.
    The tokens are intentionally coarse (object/receptacle types, not instance
    ids) to transfer across ALFWorld rooms.
    """
    # Put goal/state tokens before the admissible-scene tail so they survive
    # max_query_len truncation. The scene receptacles still remain available.
    toks = list(_goal_keywords(goal))
    obj = _normalize_entity(spec.get("target_object", ""))
    recep = _normalize_entity(spec.get("target_receptacle", ""))
    transform = spec.get("transform", "")
    has_obj = _holds_target(memory, spec)

    state: List[str] = []
    if obj:
        state.append(f"TARGET::{obj}")
        state.append("HAS_TARGET" if has_obj else "NEED_TARGET")
    if recep:
        state.append(f"GOAL_RECEP::{recep}")
    if transform:
        state.append(f"TRANSFORM::{transform}")
        state.append(
            f"TRANSFORM_DONE::{transform}"
            if transform in memory.get("completed_transforms", set())
            else f"TRANSFORM_PENDING::{transform}"
        )
    cur = _normalize_entity(memory.get("current_location", ""))
    if cur:
        state.append(f"AT::{cur}")

    for item in sorted(str(x).lower() for x in memory.get("inventory", set()))[:4]:
        state.append(f"INV::{_normalize_entity(item)}")
    for loc in sorted(str(x).lower() for x in memory.get("visited", set()))[:8]:
        state.append(f"VISITED::{_normalize_entity(loc)}")
    for loc in sorted(str(x).lower() for x in memory.get("opened", set()))[:6]:
        state.append(f"OPENED::{_normalize_entity(loc)}")

    # Let the condition see the currently executable action surface.
    verbs = sorted({parse_action(a).get("verb", "") for a in admissible if parse_action(a).get("verb", "")})
    for verb in verbs[:8]:
        state.append(f"ADM_VERB::{verb}")
    receps = sorted({
        _normalize_entity(parse_action(a).get("recep", ""))
        for a in admissible
        if parse_action(a).get("recep", "")
    })
    for r in receps[:8]:
        state.append(f"ADM_RECEP::{r}")

    failed = list(memory.get("failed_actions", set()))
    for cmd in sorted(str(x).lower() for x in failed)[:4]:
        p = parse_action(cmd)
        sig = "::".join(_normalize_entity(str(p.get(k, ""))) for k in ("verb", "obj", "recep"))
        state.append(f"FAILED::{sig}")

    scene = [f"RECEP::{r}" for r in receptacle_types_from_admissible(admissible)]
    return toks + state + scene


def _episode_query_tokens(goal: str, spec: Dict[str, Any], admissible: List[str], memory: Dict[str, Any], args) -> List[str]:
    if bool(getattr(args, "state_conditioning", True)):
        return _state_condition_tokens(goal, spec, admissible, memory)
    return goal_query_tokens(_goal_keywords(goal), admissible)


def _has_failure_feedback(obs_text: str) -> bool:
    text = str(obs_text).lower()
    return any(marker in text for marker in FAILURE_MARKERS)


def _action_signature(cmd: str) -> tuple:
    p = parse_action(cmd)
    return (p.get("verb", ""), p.get("obj", ""), p.get("recep", ""))


def _failed_signatures(memory: Dict[str, Any]) -> set:
    out = set()
    for a in memory.get("failed_actions", set()):
        out.add(str(a).lower())
        out.add(_action_signature(str(a)))
    return out


def _filter_failed_admissible(admissible: List[str], memory: Dict[str, Any]) -> List[str]:
    """Avoid repeating actions that the environment already rejected."""
    failed = _failed_signatures(memory)
    if not failed:
        return list(admissible)
    keep = [
        a for a in admissible
        if str(a).lower() not in failed and _action_signature(a) not in failed
    ]
    return keep or list(admissible)


def _holds_target(memory: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    obj = _normalize_entity(spec.get("target_object", ""))
    if not obj:
        return False
    return any(obj in str(item).lower() for item in memory.get("inventory", set()))


def _action_constraint_penalty(cmd: str, memory: Dict[str, Any], spec: Dict[str, Any]) -> float:
    """Hard-ish executable-plan penalty used by Pi_exec."""
    p = parse_action(cmd)
    verb = p.get("verb", "")
    obj = _normalize_entity(spec.get("target_object", ""))
    recep = _normalize_entity(spec.get("target_receptacle", ""))
    transform = spec.get("transform", "")
    cmd_l = str(cmd).lower()
    penalty = 0.0

    if cmd_l in memory.get("failed_actions", set()) or _action_signature(cmd) in _failed_signatures(memory):
        penalty -= 100.0
    if cmd_l in UTILITY_COMMANDS:
        penalty -= 25.0
    if "desklamp" in cmd_l and transform != "look":
        penalty -= 20.0

    has_obj = _holds_target(memory, spec)
    if obj and verb in {"TAKE", "PUT"} and obj not in _normalize_entity(p.get("obj", "")):
        penalty -= 15.0
    if obj and not has_obj and verb in {"PUT", "HEAT", "COOL", "CLEAN"}:
        penalty -= 20.0
    if (
        has_obj
        and transform in TRANSFORM_STAGE
        and transform not in memory.get("completed_transforms", set())
        and verb == "PUT"
    ):
        penalty -= 20.0
    if has_obj and transform in TRANSFORM_STAGE and transform not in memory.get("completed_transforms", set()):
        if verb == TRANSFORM_STAGE[transform]:
            penalty += 10.0
        if verb == "GOTO" and _normalize_entity(p.get("recep", "")) in TRANSFORM_TOOLS.get(transform, set()):
            penalty += 6.0
    if has_obj and recep and verb == "GOTO" and recep in _normalize_entity(p.get("recep", "")):
        penalty += 6.0
    if has_obj and recep and verb == "PUT" and recep in _normalize_entity(p.get("recep", "")):
        penalty += 12.0
    if obj and not has_obj and verb == "TAKE" and obj in _normalize_entity(p.get("obj", "")):
        penalty += 15.0
    return penalty


def _load_models(args, device):
    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    path_vocab = load_vocab(ae_ckpt["path_vocab"])
    query_vocab = load_vocab(planner_ckpt["query_vocab"])

    ae_cfg = ae_ckpt["model_config"]
    autoencoder = PathAutoencoder(
        vocab_size=ae_cfg["vocab_size"],
        emb_dim=ae_cfg["emb_dim"],
        hid_dim=ae_cfg["hid_dim"],
        latent_dim=ae_cfg["latent_dim"],
        max_path_len=ae_cfg["max_path_len"],
        pad_id=ae_cfg["pad_id"],
        latent_noise_std=float(ae_cfg.get("latent_noise_std", 0.0)),
    ).to(device)
    autoencoder.load_state_dict(ae_ckpt["model_state"])
    autoencoder.eval()

    pl_cfg = planner_ckpt["model_config"]
    planner_type = str(pl_cfg.get("planner_type", "diffusion")).lower()
    if planner_type == "mlp":
        planner = MLPPlanner(
            latent_dim=pl_cfg["latent_dim"],
            q_vocab_size=pl_cfg["q_vocab_size"],
            q_emb_dim=pl_cfg["q_emb_dim"],
            q_pad_id=pl_cfg["q_pad_id"],
            hidden_dim=int(pl_cfg.get("hidden_dim", 256)),
        ).to(device)
    else:
        planner = DiffusionPlanner(
            latent_dim=pl_cfg["latent_dim"],
            q_vocab_size=pl_cfg["q_vocab_size"],
            q_emb_dim=pl_cfg["q_emb_dim"],
            q_pad_id=pl_cfg["q_pad_id"],
            time_dim=pl_cfg["time_dim"],
        ).to(device)
    planner.load_state_dict(planner_ckpt["model_state"])
    planner.eval()

    latent_mean = latent_std = None
    ln = planner_ckpt.get("latent_norm")
    if isinstance(ln, dict) and bool(ln.get("enabled", False)):
        latent_mean = torch.tensor(ln["mean"], dtype=torch.float32, device=device).view(1, -1)
        latent_std = torch.tensor(ln["std"], dtype=torch.float32, device=device).view(1, -1)

    value_model = None
    if args.value_ckpt:
        v = torch.load(args.value_ckpt, map_location="cpu")
        vc = v["model_config"]
        value_model = ValueRanker(
            q_vocab_size=vc["q_vocab_size"], p_vocab_size=vc["p_vocab_size"], emb_dim=vc["emb_dim"],
            q_pad_id=vc["q_pad_id"], p_pad_id=vc["p_pad_id"], architecture=str(vc.get("architecture", "cross")),
            hidden_dim=int(vc.get("hidden_dim", 256)), dropout=float(vc.get("dropout", 0.1)),
        ).to(device)
        value_model.load_state_dict(v["model_state"])
        value_model.eval()

    constraint_model = None
    if args.constraint_ckpt:
        c = torch.load(args.constraint_ckpt, map_location="cpu")
        cc = c["model_config"]
        constraint_model = ConstraintModel(
            q_vocab_size=cc["q_vocab_size"], p_vocab_size=cc["p_vocab_size"], emb_dim=cc["emb_dim"],
            q_pad_id=cc["q_pad_id"], p_pad_id=cc["p_pad_id"], architecture=str(cc.get("architecture", "cross")),
            hidden_dim=int(cc.get("hidden_dim", 256)), dropout=float(cc.get("dropout", 0.1)),
        ).to(device)
        constraint_model.load_state_dict(c["model_state"])
        constraint_model.eval()

    return {
        "autoencoder": autoencoder, "planner": planner, "planner_type": planner_type,
        "value_model": value_model, "constraint_model": constraint_model,
        "path_vocab": path_vocab, "query_vocab": query_vocab,
        "latent_mean": latent_mean, "latent_std": latent_std,
        "max_path_len": int(ae_cfg["max_path_len"]),
        "prediction_target": str(pl_cfg.get("prediction_target", "z0")).lower(),
        "diffusion_steps": int(pl_cfg.get("diffusion_steps", 20)),
        "prefix_conditioning": bool(pl_cfg.get("prefix_conditioning", True)),
        "cond_max_len": int(pl_cfg.get("max_query_len", 32)),
    }


def _project_head_instance_aware(head_token, admissible, prefer_recep, concrete_counts):
    """Instance-grounding layer on top of the abstract-token projection (§5.4).

    The abstract plan token (e.g. ``GOTO::cabinet``) is instance-free, so every
    concrete ``go to cabinet N`` matches it equally and the bare matcher always
    returns the same instance -> the agent never searches the other cabinets.
    Here the neural plan still decides *what* to do (the abstract verb+type);
    this layer only decides *which instance*, preferring the least-tried one so
    the abstract subgoal actually drives systematic exploration.

    Projectability is unchanged: returns ``(None, 0.0)`` iff the bare matcher
    cannot project the head in the current state.
    """
    want = parse_action_token(head_token)
    # OTHER heads usually decode to utility commands such as "look". Treat them
    # as ungrounded plan heads so the grounded controller can select a useful
    # admissible action instead of looping with high false feasibility.
    if want.get("verb") == "OTHER":
        return None, 0.0
    cmd, score = match_plan_token_to_admissible(head_token, admissible, prefer_receptacle=prefer_recep)
    if cmd is None:
        return None, 0.0
    if str(cmd).lower() in UTILITY_COMMANDS and any(str(a).lower() not in UTILITY_COMMANDS for a in admissible):
        return None, 0.0
    target = parse_action(cmd)
    # Concrete commands sharing the chosen command's abstract signature.
    siblings = [c for c in admissible if parse_action(c) == target]
    if siblings:
        siblings.sort(key=lambda c: concrete_counts.get(c, 0))
        cmd = siblings[0]
    return cmd, score


def _is_useful_projected_action(cmd: str, spec: Dict[str, Any]) -> bool:
    """Reject utility/irrelevant projected actions when useful actions exist."""
    a = str(cmd).lower()
    if a in UTILITY_COMMANDS:
        return False
    # Desklamp actions are useful for examine/look tasks, but harmful for
    # ordinary put/heat/cool/clean tasks where they become a local loop.
    if "desklamp" in a and spec.get("transform") != "look":
        return False
    return True


def _project_plan_lookahead(plan, prefix_pos, admissible, constraints, executed_prefix, memory, spec, concrete_counts, args):
    """Project the first currently executable useful token in a future plan.

    The decoded diffusion sequence is a future plan, not a mandatory literal
    action pointer. If the immediate token is ungroundable or a bad utility
    action, look ahead a few steps for the first subgoal that can be grounded
    into the current ALFWorld admissible command set.
    """
    obj = spec.get("target_object", "")
    prefer_recep = memory["object_locations"].get(obj) if obj else None
    max_scan = min(len(plan), prefix_pos + int(args.plan_lookahead))
    best = (None, "", float("-inf"))
    for pos in range(prefix_pos, max_scan):
        head_token = plan[pos]
        feasible, _ = is_feasible(executed_prefix + [head_token], constraints)
        if not feasible:
            continue
        cmd, match_score = _project_head_instance_aware(head_token, admissible, prefer_recep, concrete_counts)
        if cmd is None:
            continue
        if not _is_useful_projected_action(cmd, spec):
            continue
        pos_bonus = float(args.plan_lookahead_pos_bonus) / max(1, 1 + pos - prefix_pos)
        score = float(match_score) * 10.0 + pos_bonus + _action_constraint_penalty(cmd, memory, spec)
        score -= float(args.revisit_penalty) * concrete_counts.get(cmd, 0)
        if score > best[2]:
            best = (cmd, head_token, score)
    return best


def _projection_features(plan, prefix_pos, admissible, constraints, executed_prefix, memory, spec, concrete_counts, args):
    """Non-saturated executable-projection features for candidate gating.

    The old score was mostly ``prefix + immediate + valid-token-ratio``; in
    ALFWorld that quickly saturates because many decoded plans contain syntactic
    action tokens. This feature score asks a harder question: how many future
    tokens are actually projectable *now*, how early is the first projection,
    and how strong is the best concrete projection.
    """
    max_scan = min(len(plan), prefix_pos + int(args.plan_lookahead))
    scan_len = max(1, max_scan - prefix_pos)
    projectable = 0
    valid = 0
    first_offset = None
    best_exec = 0.0
    for pos in range(prefix_pos, max_scan):
        tok = plan[pos]
        if parse_action_token(tok).get("verb", "OTHER") != "OTHER":
            valid += 1
        feasible, _ = is_feasible(executed_prefix + [tok], constraints)
        if not feasible:
            continue
        cmd, _head, exec_score = _project_plan_lookahead(
            plan[pos : pos + 1], 0, admissible, constraints, executed_prefix, memory, spec, concrete_counts, args
        )
        if cmd is None:
            continue
        projectable += 1
        best_exec = max(best_exec, float(exec_score))
        if first_offset is None:
            first_offset = pos - prefix_pos

    density = projectable / scan_len
    valid_ratio = valid / scan_len
    early = 0.0 if first_offset is None else 1.0 / (1.0 + first_offset)
    immediate = 1.0 if first_offset == 0 else 0.0
    # exec_score is roughly 0-15 in current ALFWorld runs; clip to keep value
    # ranking alive while still breaking the saturated projection ties.
    exec_quality = min(1.0, max(0.0, best_exec) / 12.0)
    score = 1.25 * immediate + 1.00 * early + 1.25 * density + 0.75 * exec_quality + 0.25 * valid_ratio
    return {
        "score": float(score),
        "exec_quality": float(exec_quality),
        "best_exec": float(best_exec),
        "density": float(density),
        "early": float(early),
        "immediate": float(immediate),
        "valid_ratio": float(valid_ratio),
        "projectable": int(projectable),
    }


@torch.no_grad()
def _plan_and_select(M, query_tokens, executed_prefix, admissible, args, constraints, memory, spec, concrete_counts):
    """Generate candidate plans, rerank, and project the head onto an admissible command.

    Returns ``(concrete_action_or_None, planned_head_token, selected_plan, n_candidates)``.
    """
    device = args._device
    cands = sample_plan_candidates(
        planner=M["planner"], autoencoder=M["autoencoder"], path_vocab=M["path_vocab"],
        query_vocab=M["query_vocab"], query_tokens=query_tokens, num_candidates=int(args.num_candidates),
        diffusion_steps=M["diffusion_steps"], max_path_len=M["max_path_len"], device=device,
        executed_prefix=executed_prefix, use_prefix=M["prefix_conditioning"], cond_max_len=M["cond_max_len"],
        latent_mean=M["latent_mean"], latent_std=M["latent_std"],
        prediction_target=M["prediction_target"], planner_type=M["planner_type"],
        jitter_std=float(args.candidate_jitter_std),
    )
    # Dedupe candidate plans.
    uniq = []
    seen = set()
    for c in cands:
        key = tuple(c)
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq:
        return None, "", [], 0, {}

    scores = score_candidates_with_value(
        M["value_model"], M["path_vocab"], M["query_vocab"], query_tokens, uniq, device,
        max_query_len=M["cond_max_len"], max_path_len=M["max_path_len"],
    )
    if M["constraint_model"] is not None and args.constraint_penalty_weight > 0.0:
        viol = constraint_violation_scores(
            M["constraint_model"], M["path_vocab"], M["query_vocab"], query_tokens, uniq, device,
            max_query_len=M["cond_max_len"], max_path_len=M["max_path_len"],
        )
        scores = [s - float(args.constraint_penalty_weight) * v for s, v in zip(scores, viol)]

    raw_value_scores = list(scores)
    prefix_pos = len(executed_prefix)
    proj_scores = []
    proj_exec_scores = []
    proj_density_scores = []
    for plan in uniq:
        has_prefix = plan[:prefix_pos] == executed_prefix
        start_pos = prefix_pos if has_prefix else 0
        prefix_bonus = 0.75 if has_prefix else 0.0
        pf = _projection_features(
            plan, start_pos, admissible, constraints, executed_prefix, memory, spec, concrete_counts, args
        )
        proj_scores.append(prefix_bonus + float(pf["score"]))
        proj_exec_scores.append(float(pf["best_exec"]))
        proj_density_scores.append(float(pf["density"]))

    projection_weight = float(args.projection_bonus_weight)
    if projection_weight > 0.0:
        scores = [s + projection_weight * p for s, p in zip(scores, proj_scores)]

    selectable = list(range(len(uniq)))
    min_proj = float(getattr(args, "projection_min_score", 0.0))
    if min_proj > 0.0:
        gated = [i for i in selectable if proj_scores[i] >= min_proj]
        if gated:
            selectable = gated
    topk = int(getattr(args, "projection_topk", 0))
    if topk > 0 and len(selectable) > topk:
        selectable = sorted(selectable, key=lambda i: (proj_scores[i], proj_exec_scores[i]), reverse=True)[:topk]

    order = sorted(selectable, key=lambda i: scores[i], reverse=True)
    obj = spec.get("target_object", "")
    prefer_recep = memory["object_locations"].get(obj) if obj else None

    # Two passes: prefer plans that extend the executed prefix, then any plan.
    projected = []
    used_signatures = Counter()
    for require_prefix in (True, False):
        for i in order:
            plan = uniq[i]
            if require_prefix and plan[:prefix_pos] != executed_prefix:
                continue
            if len(plan) <= prefix_pos:
                continue
            start_pos = prefix_pos if plan[:prefix_pos] == executed_prefix else 0
            cmd, head_token, proj_score = _project_plan_lookahead(
                plan, start_pos, admissible, constraints, executed_prefix, memory, spec, concrete_counts, args
            )
            if cmd is not None:
                sig = _action_signature(cmd)
                diversity = 0.0 if used_signatures[sig] else float(args.coverage_diversity_bonus)
                total = float(scores[i]) + proj_score + diversity
                projected.append((total, i, cmd, head_token, plan, sig, proj_score))
                used_signatures[sig] += 1
        if projected:
            projected.sort(key=lambda x: x[0], reverse=True)
            _, selected_i, cmd, head_token, plan, _sig, selected_exec_proj = projected[0]
            value_order = sorted(range(len(uniq)), key=lambda j: raw_value_scores[j], reverse=True)
            proj_order = sorted(range(len(uniq)), key=lambda j: proj_scores[j], reverse=True)
            diag = {
                "selected_projection_score": float(proj_scores[selected_i]),
                "selected_exec_projection_score": float(selected_exec_proj),
                "max_projection_score": float(max(proj_scores)) if proj_scores else 0.0,
                "mean_projection_score": float(sum(proj_scores) / max(1, len(proj_scores))),
                "selected_projection_density": float(proj_density_scores[selected_i]) if proj_density_scores else 0.0,
                "max_projection_exec_score": float(max(proj_exec_scores)) if proj_exec_scores else 0.0,
                "projection_gate_size": int(len(selectable)),
                "selected_value_rank": int(value_order.index(selected_i) + 1) if selected_i in value_order else 0,
                "selected_projection_rank": int(proj_order.index(selected_i) + 1) if selected_i in proj_order else 0,
            }
            return cmd, head_token, plan, len(uniq), diag
    # Head not projectable from any plan; signal fallback to the caller.
    best_plan = uniq[order[0]]
    best_head = best_plan[min(prefix_pos, len(best_plan) - 1)] if best_plan else ""
    diag = {
        "selected_projection_score": float(proj_scores[order[0]]) if order and proj_scores else 0.0,
        "selected_exec_projection_score": float(proj_exec_scores[order[0]]) if order and proj_exec_scores else 0.0,
        "max_projection_score": float(max(proj_scores)) if proj_scores else 0.0,
        "mean_projection_score": float(sum(proj_scores) / max(1, len(proj_scores))),
        "selected_projection_density": float(proj_density_scores[order[0]]) if order and proj_density_scores else 0.0,
        "max_projection_exec_score": float(max(proj_exec_scores)) if proj_exec_scores else 0.0,
        "projection_gate_size": int(len(selectable)),
        "selected_value_rank": 0,
        "selected_projection_rank": 0,
    }
    return None, best_head, best_plan, len(uniq), diag


def _value_rank_admissible(M, query_tokens, executed_prefix, admissible, args, concrete_counts):
    """Fallback: directly value-rank the admissible commands' abstract tokens.

    Abstract tokens collapse instances, so value scores tie across ``go to X N``;
    a per-concrete-command revisit penalty breaks the tie toward least-tried
    instances so the fallback still explores instead of looping on one instance.
    """
    device = args._device
    tokens = [normalize_action(c) for c in admissible]
    cand_paths = [executed_prefix + [t] for t in tokens]
    scores = score_candidates_with_value(
        M["value_model"], M["path_vocab"], M["query_vocab"], query_tokens, cand_paths, device,
        max_query_len=M["cond_max_len"], max_path_len=M["max_path_len"],
    )
    penalty = float(args.revisit_penalty)
    scores = [s - penalty * concrete_counts.get(admissible[i], 0) for i, s in enumerate(scores)]
    best = max(range(len(admissible)), key=lambda i: scores[i])
    return admissible[best], tokens[best]


def _diffusion_prior_score(action: str, head_token: str) -> float:
    """Soft compatibility between a decoded abstract head and a concrete command."""
    if not head_token:
        return 0.0
    want = parse_action_token(head_token)
    got = parse_action(action)
    score = 0.0
    if got["verb"] == want["verb"]:
        score += 4.0
    if want.get("obj") and got.get("obj") == want["obj"]:
        score += 2.0
    if want.get("recep") and got.get("recep") == want["recep"]:
        score += 2.0
    return score


def _hard_goal_action(admissible, memory: Dict[str, Any], spec: Dict[str, Any], concrete_counts):
    """Goal-critical admissible action shortcut.

    This is deliberately narrow: it only fires for actions that directly satisfy
    the current goal stage (take target object / place held target object). It
    prevents the grounded controller from wandering past an immediately useful
    target action.
    """
    obj = _normalize_entity(spec.get("target_object", ""))
    recep = _normalize_entity(spec.get("target_receptacle", ""))
    transform = spec.get("transform", "")
    if not obj or transform == "look":
        return None
    inventory = {str(x).lower() for x in memory.get("inventory", set())}
    has_obj = any(obj in item for item in inventory)
    parsed = [(cmd, parse_action(cmd)) for cmd in admissible]

    if not has_obj:
        takes = [
            cmd for cmd, p in parsed
            if p.get("verb") == "TAKE" and obj and obj in _normalize_entity(p.get("obj", ""))
        ]
        if takes:
            takes.sort(key=lambda c: concrete_counts.get(c, 0))
            return takes[0]

    if has_obj and recep:
        if transform in TRANSFORM_TOOLS and transform not in memory.get("completed_transforms", set()):
            transform_verb = TRANSFORM_STAGE[transform]
            transforms = [
                cmd for cmd, p in parsed
                if p.get("verb") == transform_verb and obj in _normalize_entity(p.get("obj", ""))
            ]
            if transforms:
                transforms.sort(key=lambda c: concrete_counts.get(c, 0))
                return transforms[0]
            tool_gotos = [
                cmd for cmd, p in parsed
                if p.get("verb") == "GOTO"
                and _normalize_entity(p.get("recep", "")) in TRANSFORM_TOOLS[transform]
            ]
            if tool_gotos:
                tool_gotos.sort(key=lambda c: concrete_counts.get(c, 0))
                return tool_gotos[0]
            tool_opens = [
                cmd for cmd, p in parsed
                if p.get("verb") == "OPEN"
                and _normalize_entity(p.get("recep", "")) in TRANSFORM_TOOLS[transform]
            ]
            if tool_opens:
                tool_opens.sort(key=lambda c: concrete_counts.get(c, 0))
                return tool_opens[0]

        puts = [
            cmd for cmd, p in parsed
            if p.get("verb") == "PUT"
            and obj in _normalize_entity(p.get("obj", ""))
            and recep in _normalize_entity(p.get("recep", ""))
        ]
        if puts:
            puts.sort(key=lambda c: concrete_counts.get(c, 0))
            return puts[0]

        goto_recep = [
            cmd for cmd, p in parsed
            if p.get("verb") == "GOTO" and recep in _normalize_entity(p.get("recep", ""))
        ]
        if goto_recep:
            goto_recep.sort(key=lambda c: concrete_counts.get(c, 0))
            return goto_recep[0]

        open_recep = [
            cmd for cmd, p in parsed
            if p.get("verb") == "OPEN" and recep in _normalize_entity(p.get("recep", ""))
        ]
        if open_recep:
            open_recep.sort(key=lambda c: concrete_counts.get(c, 0))
            return open_recep[0]

    return None


def _grounded_rank_admissible(
    M,
    query_tokens,
    executed_prefix,
    admissible,
    args,
    concrete_counts,
    *,
    obs_text: str,
    goal: str,
    spec: Dict[str, Any],
    memory: Dict[str, Any],
    head_token: str,
):
    """Ground fallback actions in legal ALFWorld commands.

    Pure value fallback can collapse to short utility commands (``look``). This
    combines the tested ALFWorld stage-aware controller with a soft diffusion
    prior, so the neural plan still influences the selected admissible action
    without being allowed to execute ungrounded tokens.
    """
    hard = _hard_goal_action(admissible, memory, spec, concrete_counts)
    if hard is not None:
        return hard, normalize_action(hard)

    useful = [a for a in admissible if str(a).lower() not in UTILITY_COMMANDS]
    pool = useful or list(admissible)
    action_counts = Counter({str(k).lower(): int(v) for k, v in concrete_counts.items()})
    rng = random.Random(int(args.seed) + len(executed_prefix))
    ranked = _rank_actions(
        pool,
        obs_text,
        goal,
        spec.get("terms", []),
        "",
        action_counts,
        rng,
        memory=memory,
        spec=spec,
    )
    heuristic = {a: s for a, s in ranked}

    tokens = [normalize_action(c) for c in pool]
    cand_paths = [executed_prefix + [t] for t in tokens]
    value_scores = score_candidates_with_value(
        M["value_model"], M["path_vocab"], M["query_vocab"], query_tokens, cand_paths, args._device,
        max_query_len=M["cond_max_len"], max_path_len=M["max_path_len"],
    )

    best_i = 0
    best_score = float("-inf")
    for i, cmd in enumerate(pool):
        total = 0.0
        total += float(args.grounded_heuristic_weight) * heuristic.get(cmd, 0.0)
        total += float(args.grounded_value_weight) * float(value_scores[i])
        total += float(args.grounded_diffusion_prior_weight) * _diffusion_prior_score(cmd, head_token)
        total += float(args.exec_constraint_weight) * _action_constraint_penalty(cmd, memory, spec)
        total -= float(args.revisit_penalty) * concrete_counts.get(cmd, 0)
        if str(cmd).lower() in UTILITY_COMMANDS:
            total -= 20.0
        if total > best_score:
            best_score = total
            best_i = i
    return pool[best_i], normalize_action(pool[best_i])


def _run_episode(env, M, episode_id, args) -> Dict[str, Any]:
    t0 = time.time()
    obs, infos = env.reset()
    obs_text = _text(obs)
    goal = _extract_goal(obs_text)
    spec = _parse_goal(goal)
    memory = _new_memory()

    admissible0 = list((infos.get("admissible_commands") or [[]])[0])
    query_tokens = _episode_query_tokens(goal, spec, admissible0, memory, args)
    constraints = _exec_constraints(spec, max_steps=int(args.max_steps))

    executed_prefix: List[str] = []   # abstract plan tokens actually committed
    raw_actions: List[str] = []
    concrete_counts: Counter = Counter()  # per-concrete-command tries, for instance grounding
    observations = [obs_text]
    n_candidate_list: List[int] = []
    projection_diags: List[Dict[str, Any]] = []
    condition_tokens_by_prefix: List[List[str]] = []
    projectable = 0
    grounded_fallbacks = 0
    consistency_hits = 0
    score = 0.0
    done = False
    fixed_plan: List[str] | None = None
    fixed_plan_n_candidates = 0

    for step in range(int(args.max_steps)):
        admissible_batch = infos.get("admissible_commands", [[]])
        admissible = list(admissible_batch[0] if admissible_batch else [])
        if not admissible:
            break
        _update_memory(memory, obs_text, raw_actions[-1] if raw_actions else "", admissible)
        admissible = _filter_failed_admissible(admissible, memory)
        query_tokens = _episode_query_tokens(goal, spec, admissible, memory, args)
        condition_tokens_by_prefix.append(list(query_tokens))

        if args.no_receding and fixed_plan is not None:
            prefix_pos = len(executed_prefix)
            n_cand = fixed_plan_n_candidates
            cmd = None
            head_token = ""
            if prefix_pos < len(fixed_plan):
                head_token = fixed_plan[prefix_pos]
                feasible, _ = is_feasible(executed_prefix + [head_token], constraints)
                if feasible:
                    obj = spec.get("target_object", "")
                    prefer_recep = memory["object_locations"].get(obj) if obj else None
                    cmd, _match_score = _project_head_instance_aware(
                        head_token, admissible, prefer_recep, concrete_counts
                    )
        else:
            cmd, head_token, _plan, n_cand, proj_diag = _plan_and_select(
                M, query_tokens, executed_prefix, admissible, args, constraints, memory, spec, concrete_counts
            )
            projection_diags.append(proj_diag)
            if args.no_receding and fixed_plan is None:
                fixed_plan = list(_plan)
                fixed_plan_n_candidates = n_cand
        n_candidate_list.append(n_cand)
        head_projectable = cmd is not None
        if cmd is None:
            grounded_fallbacks += 1
            if args.grounded_fallback:
                cmd, head_token = _grounded_rank_admissible(
                    M,
                    query_tokens,
                    executed_prefix,
                    admissible,
                    args,
                    concrete_counts,
                    obs_text=obs_text,
                    goal=goal,
                    spec=spec,
                    memory=memory,
                    head_token=head_token,
                )
            else:
                # Graceful degradation: value-rank admissible commands directly.
                cmd, head_token = _value_rank_admissible(
                    M, query_tokens, executed_prefix, admissible, args, concrete_counts
                )
        if head_projectable:
            projectable += 1

        # Plan-execution consistency: did we execute the token we planned?
        executed_token = normalize_action(cmd)
        if executed_token == head_token:
            consistency_hits += 1

        raw_actions.append(cmd)
        concrete_counts[cmd] += 1
        obs, scores, dones, infos = env.step([cmd])
        obs_text = _text(obs)
        if any(cmd.lower().startswith(p) for p in ("heat ", "cool ", "clean ")):
            memory["completed_transforms"].add(spec.get("transform", ""))
        _update_memory(memory, obs_text, cmd, infos.get("admissible_commands", [[]])[0])
        executed_prefix.append(executed_token)
        observations.append(obs_text)
        score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        if done:
            break

    success = bool(done and score > 0)
    steps = len(raw_actions)
    # Real mechanism metrics.
    plan_feasibility = projectable / max(1, steps)
    exec_feasible, exec_violations = is_feasible(executed_prefix, constraints)
    failed = _failed_signatures(memory)
    repeated_failure = any(str(a).lower() in failed or _action_signature(a) in failed for a in raw_actions)
    constraint_violation = bool((not exec_feasible) or repeated_failure)
    consistency = consistency_hits / max(1, steps)
    first_error = _first_error_step(raw_actions, success)
    def _diag_mean(key: str) -> float:
        vals = [float(d.get(key, 0.0)) for d in projection_diags if d]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "episode_id": episode_id,
        "method": "alfworld_diplan_diffusion",
        "goal": goal,
        "structured_goal": spec,
        "success": success,
        "final_score": score,
        "num_steps": steps,
        "first_error_step": first_error,
        "recovery_at_error": bool(success and first_error <= steps),
        "trap_at_1": bool(raw_actions and _is_trap_first_action(raw_actions[0], spec.get("terms", []))),
        "plan_feasibility": plan_feasibility,
        "constraint_violation": constraint_violation,
        "constraint_violation_codes": exec_violations,
        "plan_execution_consistency": consistency,
        "candidate_pool_size": sum(n_candidate_list) / max(1, len(n_candidate_list)),
        "projection_score_selected": _diag_mean("selected_projection_score"),
        "projection_exec_score_selected": _diag_mean("selected_exec_projection_score"),
        "projection_score_max": _diag_mean("max_projection_score"),
        "projection_score_mean": _diag_mean("mean_projection_score"),
        "projection_density_selected": _diag_mean("selected_projection_density"),
        "projection_exec_score_max": _diag_mean("max_projection_exec_score"),
        "projection_gate_size": _diag_mean("projection_gate_size"),
        "selected_rank_by_value": _diag_mean("selected_value_rank"),
        "selected_rank_by_projection": _diag_mean("selected_projection_rank"),
        "grounded_fallbacks": grounded_fallbacks,
        "grounded_fallback_rate": grounded_fallbacks / max(1, steps),
        "executed_plan_tokens": executed_prefix,
        "condition_tokens_by_prefix": condition_tokens_by_prefix,
        "actions": raw_actions,
        "initial_observation": observations[0] if observations else "",
        "final_observation": observations[-1] if observations else "",
        "latency_cost": time.time() - t0,
        "token_cost": steps + int(round(sum(n_candidate_list) / max(1, len(n_candidate_list)))),
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    n = len(rows)
    if n == 0:
        return {"alfworld_diplan_diffusion": {}}
    s = {
        "n": n,
        "success_rate": sum(1 for r in rows if r["success"]) / n,
        "first_error_step": sum(float(r["first_error_step"]) for r in rows) / n,
        "recovery_at_error": sum(1 for r in rows if r["recovery_at_error"]) / n,
        "trap_at_1": sum(1 for r in rows if r["trap_at_1"]) / n,
        "plan_feasibility": sum(float(r["plan_feasibility"]) for r in rows) / n,
        "constraint_violation_rate": sum(1 for r in rows if r["constraint_violation"]) / n,
        "plan_execution_consistency": sum(float(r["plan_execution_consistency"]) for r in rows) / n,
        "token_cost": sum(float(r["token_cost"]) for r in rows) / n,
        "latency_cost": sum(float(r["latency_cost"]) for r in rows) / n,
        "candidate_pool_avg_size": sum(float(r["candidate_pool_size"]) for r in rows) / n,
        "projection_score_selected": sum(float(r.get("projection_score_selected", 0.0)) for r in rows) / n,
        "projection_exec_score_selected": sum(float(r.get("projection_exec_score_selected", 0.0)) for r in rows) / n,
        "projection_score_max": sum(float(r.get("projection_score_max", 0.0)) for r in rows) / n,
        "projection_score_mean": sum(float(r.get("projection_score_mean", 0.0)) for r in rows) / n,
        "projection_density_selected": sum(float(r.get("projection_density_selected", 0.0)) for r in rows) / n,
        "projection_exec_score_max": sum(float(r.get("projection_exec_score_max", 0.0)) for r in rows) / n,
        "projection_gate_size": sum(float(r.get("projection_gate_size", 0.0)) for r in rows) / n,
        "selected_rank_by_value": sum(float(r.get("selected_rank_by_value", 0.0)) for r in rows) / n,
        "selected_rank_by_projection": sum(float(r.get("selected_rank_by_projection", 0.0)) for r in rows) / n,
        "grounded_fallback_rate": sum(float(r.get("grounded_fallback_rate", 0.0)) for r in rows) / n,
        "avg_steps": sum(float(r["num_steps"]) for r in rows) / n,
    }
    return {"alfworld_diplan_diffusion": s}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained diffusion DiPLaN agent on ALFWorld.")
    parser.add_argument("--data_root", type=str, default="data/long_horizon/alfworld")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval_out_of_distribution")
    parser.add_argument("--out", type=str, default="results/alfworld_diplan_diffusion")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, default="")
    parser.add_argument("--constraint_ckpt", type=str, default="")
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--candidate_jitter_std", type=float, default=0.05)
    parser.add_argument("--constraint_penalty_weight", type=float, default=0.3)
    parser.add_argument(
        "--projection_bonus_weight",
        type=float,
        default=2.0,
        help="Weight for executable projection score when reranking diffusion candidate plans.",
    )
    parser.add_argument(
        "--projection_topk",
        type=int,
        default=0,
        help="Hard executable-planning gate: keep only the top-K candidates by projection score before value rerank.",
    )
    parser.add_argument(
        "--projection_min_score",
        type=float,
        default=0.0,
        help="Hard executable-planning gate: discard candidates below this projection score when possible.",
    )
    parser.add_argument(
        "--state_conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition the diffusion/value models on symbolic environment memory as well as the goal.",
    )
    parser.add_argument("--grounded_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan_lookahead", type=int, default=6)
    parser.add_argument("--plan_lookahead_pos_bonus", type=float, default=4.0)
    parser.add_argument("--coverage_diversity_bonus", type=float, default=2.0)
    parser.add_argument("--exec_constraint_weight", type=float, default=1.0)
    parser.add_argument("--grounded_heuristic_weight", type=float, default=1.0)
    parser.add_argument("--grounded_value_weight", type=float, default=0.15)
    parser.add_argument("--grounded_diffusion_prior_weight", type=float, default=1.0)
    parser.add_argument(
        "--revisit_penalty",
        type=float,
        default=1.0,
        help="Instance-grounding: penalty per prior try of a concrete admissible command in the "
        "value-rank fallback, so collapsed-instance ties resolve toward least-visited (0 = off, "
        "recovers the original instance-blind behavior).",
    )
    parser.add_argument(
        "--no_receding",
        action="store_true",
        help="Plan once at episode start and execute the fixed decoded plan instead of re-planning after each step.",
    )
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    os.environ["ALFWORLD_DATA"] = str(data_root)
    torch.manual_seed(int(args.seed))
    args._device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[diplan] device={args._device}")

    M = _load_models(args, args._device)
    print(
        f"[diplan] planner_type={M['planner_type']} prefix_conditioning={M['prefix_conditioning']} "
        f"max_path_len={M['max_path_len']} value={M['value_model'] is not None} "
        f"constraint={M['constraint_model'] is not None}"
    )
    env = _make_env(Path(args.config).resolve(), data_root, args.split, batch_size=1)

    rows = []
    for episode_id in range(int(args.episodes)):
        print(f"[episode] {episode_id + 1}/{args.episodes}")
        row = _run_episode(env, M, episode_id, args)
        print(
            f"[episode] success={row['success']} steps={row['num_steps']} "
            f"feas={row['plan_feasibility']:.2f} cons={row['plan_execution_consistency']:.2f} "
            f"viol={row['constraint_violation']} goal={row['goal']!r}"
        )
        rows.append(row)

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        import json
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump_json(str(out_dir / "summary_metrics.json"), _summarize(rows))
    dump_json(
        str(out_dir / "run_config.json"),
        {
            "data_root": str(data_root), "config": str(Path(args.config).resolve()), "split": args.split,
            "episodes": args.episodes, "max_steps": args.max_steps, "seed": args.seed,
            "num_candidates": args.num_candidates, "candidate_jitter_std": args.candidate_jitter_std,
            "constraint_penalty_weight": args.constraint_penalty_weight,
            "projection_bonus_weight": args.projection_bonus_weight,
            "projection_topk": args.projection_topk,
            "projection_min_score": args.projection_min_score,
            "state_conditioning": args.state_conditioning,
            "revisit_penalty": args.revisit_penalty,
            "grounded_fallback": args.grounded_fallback,
            "plan_lookahead": args.plan_lookahead,
            "plan_lookahead_pos_bonus": args.plan_lookahead_pos_bonus,
            "coverage_diversity_bonus": args.coverage_diversity_bonus,
            "exec_constraint_weight": args.exec_constraint_weight,
            "grounded_heuristic_weight": args.grounded_heuristic_weight,
            "grounded_value_weight": args.grounded_value_weight,
            "grounded_diffusion_prior_weight": args.grounded_diffusion_prior_weight,
            "ae_ckpt": args.ae_ckpt, "planner_ckpt": args.planner_ckpt,
            "value_ckpt": args.value_ckpt, "constraint_ckpt": args.constraint_ckpt,
        },
    )
    print(f"[ok] wrote diffusion DiPLaN outputs to {out_dir}")


if __name__ == "__main__":
    main()
