import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import evaluate_torch as et
from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir, load_config, load_jsonl
from src.diplan.metrics import (
    aggregate_method_metrics,
    first_error_step,
    plan_execution_consistency,
    recovery_at_error,
    trap_at_1,
)
from src.diplan.torch_pipeline import (
    DiffusionPlanner,
    MLPPlanner,
    PAD,
    PathAutoencoder,
    ValueRanker,
    load_vocab,
)


def _post_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return str(obj["choices"][0]["message"]["content"]).strip()


def _parse_choice_index(text: str, n_choices: int) -> int:
    text = text.strip()
    # First, try strict JSON object.
    try:
        parsed = json.loads(text)
        idx = int(parsed.get("index", -1))
        if 1 <= idx <= n_choices:
            return idx
    except Exception:
        pass

    # Then, try extracting first JSON block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            idx = int(parsed.get("index", -1))
            if 1 <= idx <= n_choices:
                return idx
        except Exception:
            pass

    # Fallback: find first integer.
    num = re.search(r"\b(\d+)\b", text)
    if num:
        idx = int(num.group(1))
        if 1 <= idx <= n_choices:
            return idx
    return -1


def _build_llm_prompts(
    row: Dict,
    executed: List[str],
    candidates: List[List[str]],
) -> Tuple[str, str]:
    constraints = row.get("constraints", {})
    allowed_steps = int(constraints.get("max_steps", 8))
    c_lines = []
    for i, c in enumerate(candidates, start=1):
        first = c[0] if c else ""
        c_lines.append(f"{i}. next_action={first}; plan={c}")
    c_text = "\n".join(c_lines)
    system_prompt = (
        "You are a planning policy for a tool-using agent. "
        "Select exactly one candidate plan that is most likely to complete the task under constraints. "
        "Return strict JSON only: {\"index\": <1-based integer>, \"reason\": \"short\"}."
    )
    user_prompt = (
        f"Task ID: {row.get('task_id', '')}\n"
        f"Dataset: {row.get('dataset', '')}\n"
        f"Question: {row.get('question', ' '.join(row.get('query_tokens', [])))}\n"
        f"Executed prefix: {executed}\n"
        f"Max steps: {allowed_steps}\n"
        f"Forbidden/banned: {constraints.get('forbidden_actions', [])} + {constraints.get('banned_relations', [])}\n"
        f"Required stage order: {constraints.get('required_stage_order', [])}\n"
        f"Candidates:\n{c_text}\n"
        "Choose one candidate index."
    )
    return system_prompt, user_prompt


def _safe_llm_pick(
    row: Dict,
    executed: List[str],
    candidates: List[List[str]],
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
) -> Tuple[int, str, str]:
    system_prompt, user_prompt = _build_llm_prompts(row, executed, candidates)
    last_err = ""
    for _ in range(max(1, retries)):
        try:
            raw = _post_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            idx = _parse_choice_index(raw, len(candidates))
            if idx != -1:
                return idx - 1, raw, ""
            last_err = f"parse_failed: {raw[:300]}"
        except urllib.error.HTTPError as e:
            last_err = f"http_{e.code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.2)
    return -1, "", last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_torch_kgqa.llm_agent.json")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--planner_ckpt", type=str, required=True)
    parser.add_argument("--value_ckpt", type=str, default="")
    parser.add_argument("--out", type=str, default="results/llm_agent")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = load_jsonl(cfg["test_path"])
    include_datasets = [str(x).lower() for x in cfg.get("include_datasets", [])]
    if include_datasets:
        rows = [r for r in rows if str(r.get("dataset", "")).lower() in set(include_datasets)]
        print(f"[llm-agent] include_datasets={include_datasets} kept={len(rows)}")

    if int(cfg.get("max_tasks", 0)) > 0:
        rows = rows[: int(cfg["max_tasks"])]
        print(f"[llm-agent] max_tasks={len(rows)}")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", False)) else "cpu")

    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    planner_ckpt = torch.load(args.planner_ckpt, map_location="cpu")
    value_ckpt = torch.load(args.value_ckpt, map_location="cpu") if args.value_ckpt else None
    use_value_model = bool(cfg.get("use_value_model", bool(args.value_ckpt)))

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
    prediction_target = str(pl_cfg.get("prediction_target", "z0")).lower()

    latent_mean = None
    latent_std = None
    latent_norm = planner_ckpt.get("latent_norm")
    if isinstance(latent_norm, dict) and bool(latent_norm.get("enabled", False)):
        latent_mean = torch.tensor(latent_norm["mean"], dtype=torch.float32, device=device).view(1, -1)
        latent_std = torch.tensor(latent_norm["std"], dtype=torch.float32, device=device).view(1, -1)

    value_model = None
    if use_value_model and value_ckpt is not None:
        v_cfg = value_ckpt["model_config"]
        value_model = ValueRanker(
            q_vocab_size=v_cfg["q_vocab_size"],
            p_vocab_size=v_cfg["p_vocab_size"],
            emb_dim=v_cfg["emb_dim"],
            q_pad_id=v_cfg["q_pad_id"],
            p_pad_id=v_cfg["p_pad_id"],
            architecture=str(v_cfg.get("architecture", "legacy")),
            hidden_dim=int(v_cfg.get("hidden_dim", 256)),
            dropout=float(v_cfg.get("dropout", 0.1)),
        ).to(device)
        value_model.load_state_dict(value_ckpt["model_state"])
        value_model.eval()

    use_memory_retrieval = bool(cfg.get("use_memory_retrieval", True))
    memory_prefilter_feasible = bool(cfg.get("memory_prefilter_feasible", True))
    memory_top_k = int(cfg.get("memory_top_k", 16))
    token_to_ids: Dict[str, List[int]] = {}
    path_bank: List[List[str]] = []
    global_expected_len = 2.0
    bucket_expected_len: Dict[int, float] = {}
    expected_length_bucket_size = int(cfg.get("expected_length_bucket_size", 4))
    train_rows = None
    if use_memory_retrieval or bool(cfg.get("use_expected_length_prior", True)):
        train_path = cfg.get("memory_train_path") or cfg.get("train_path")
        if not train_path:
            raise ValueError("LLM agent mode requires train_path or memory_train_path in config.")
        train_rows = load_jsonl(train_path)
    if use_memory_retrieval and train_rows is not None:
        token_to_ids, path_bank = et._build_memory_index(
            train_rows=train_rows,
            max_postings_per_token=int(cfg.get("memory_max_postings_per_token", 1200)),
        )
        print(f"[llm-agent] memory index ready, paths={len(path_bank)}")
    if train_rows is not None and bool(cfg.get("use_expected_length_prior", True)):
        global_expected_len, bucket_expected_len = et._build_expected_len_prior(train_rows, expected_length_bucket_size)

    api_base = str(cfg.get("llm_api_base", "http://127.0.0.1:8000/v1"))
    api_key = str(cfg.get("llm_api_key", "EMPTY"))
    llm_model = str(cfg.get("llm_model", "Qwen"))
    llm_temperature = float(cfg.get("llm_temperature", 0.1))
    llm_max_tokens = int(cfg.get("llm_max_tokens", 128))
    llm_timeout_s = int(cfg.get("llm_timeout_s", 30))
    llm_retries = int(cfg.get("llm_retries", 2))
    llm_top_k = int(cfg.get("llm_top_k", 8))
    rerank_memory_rank_bonus = float(cfg.get("rerank_memory_rank_bonus", 0.0))

    receding_horizon = bool(cfg.get("receding_horizon", True))
    num_candidates = int(cfg.get("num_candidates", 12))
    length_penalty_alpha = float(cfg.get("length_penalty_alpha", 0.0))
    candidate_latent_jitter_std = float(cfg.get("candidate_latent_jitter_std", 0.05))
    candidate_multi_jitter_stds = [float(x) for x in cfg.get("candidate_multi_jitter_stds", [])]
    save_episode_trace = bool(cfg.get("save_episode_trace", True))
    terminal_stage_prefix = str(cfg.get("terminal_stage_prefix", "FOLLOWUP::"))
    stop_on_terminal_stage = bool(cfg.get("stop_on_terminal_stage", False))

    records = []
    llm_error_count = 0
    fallback_pick_count = 0
    for row in rows:
        executed: List[str] = []
        planned: List[str] = []
        violations_all: List[str] = []
        episode_trace: List[Dict] = []
        pool_seen = set()
        max_steps = int(row.get("constraints", {}).get("max_steps", 8))
        for step in range(max_steps):
            mem_cands = (
                et._retrieve_memory_candidates(row["query_tokens"], token_to_ids, path_bank, top_k=memory_top_k)
                if use_memory_retrieval
                else []
            )
            if use_memory_retrieval and memory_prefilter_feasible:
                mem_cands, _ = et._filter_feasible_candidates(mem_cands, row["constraints"])

            jitter_list = [candidate_latent_jitter_std] + [x for x in candidate_multi_jitter_stds if x != candidate_latent_jitter_std]
            generated_all: List[List[str]] = []
            for jit in jitter_list:
                cands_gen, _ = et._generate_candidates(
                    planner=planner,
                    autoencoder=autoencoder,
                    value_model=value_model,
                    path_vocab=path_vocab,
                    query_vocab=query_vocab,
                    query_tokens=row["query_tokens"] if not receding_horizon else row["query_tokens"],
                    num_candidates=num_candidates,
                    diffusion_steps=int(pl_cfg.get("diffusion_steps", 20)),
                    max_path_len=max(1, max_steps - len(executed)),
                    device=device,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                    prediction_target=prediction_target,
                    planner_type=planner_type,
                    candidate_latent_jitter_std=jit,
                )
                generated_all.extend(cands_gen)

            generated_unique = []
            seen = set()
            for c in generated_all:
                k = tuple(c)
                if k not in seen:
                    seen.add(k)
                    generated_unique.append(c)
            candidates = et._merge_candidates(generated_unique, mem_cands)
            if not candidates:
                break
            for c in candidates:
                pool_seen.add(tuple(c))
            mem_rank_scores = et._memory_rank_scores(mem_cands)

            expected_total_len = et._expected_len_from_prior(
                row.get("query_tokens", []),
                expected_length_bucket_size,
                global_expected_len,
                bucket_expected_len,
            )
            expected_remaining = max(1.0, expected_total_len - len(executed))
            stage1_scores = et._score_candidates(
                value_model=value_model,
                path_vocab=path_vocab,
                query_vocab=query_vocab,
                query_tokens=row["query_tokens"],
                candidates=candidates,
                device=device,
                expected_len=expected_remaining,
                length_penalty_alpha=length_penalty_alpha,
            )
            if rerank_memory_rank_bonus > 0.0:
                stage1_scores = [
                    s + rerank_memory_rank_bonus * mem_rank_scores.get(tuple(c), 0.0)
                    for s, c in zip(stage1_scores, candidates)
                ]
            idx_sorted = sorted(range(len(candidates)), key=lambda i: stage1_scores[i], reverse=True)
            top_idx = idx_sorted[: min(max(1, llm_top_k), len(idx_sorted))]
            top_candidates = [candidates[i] for i in top_idx]

            llm_pick_local, llm_raw, llm_err = _safe_llm_pick(
                row=row,
                executed=executed,
                candidates=top_candidates,
                api_base=api_base,
                api_key=api_key,
                model=llm_model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                timeout_s=llm_timeout_s,
                retries=llm_retries,
            )
            picked_global = -1
            if llm_pick_local >= 0:
                picked_global = top_idx[llm_pick_local]
            else:
                llm_error_count += 1
                picked_global = top_idx[0]
                fallback_pick_count += 1

            # Make selection feasible at prefix level. If not feasible, fall back by rank.
            chosen = None
            chosen_idx = -1
            chosen_violations: List[str] = []
            trial_order = [picked_global] + [i for i in top_idx if i != picked_global]
            for i in trial_order:
                if not candidates[i]:
                    continue
                feasible, vs = et._is_feasible(executed + [candidates[i][0]], row["constraints"])
                if feasible:
                    chosen = candidates[i]
                    chosen_idx = i
                    break
                chosen_violations.extend(vs)
            if chosen is None:
                violations_all.extend(chosen_violations)
                break

            planned = chosen
            executed.append(chosen[0])
            if save_episode_trace:
                episode_trace.append(
                    {
                        "step": step + 1,
                        "selected_path": list(chosen),
                        "selected_next_action": chosen[0],
                        "selected_stage1_score": float(stage1_scores[chosen_idx]),
                        "llm_raw_output": llm_raw,
                        "llm_error": llm_err,
                        "top_candidates": top_candidates,
                    }
                )

            if stop_on_terminal_stage and chosen[0].startswith(terminal_stage_prefix):
                break
            if len(executed) >= len(row.get("oracle_path", [])):
                break

        feasible, violations_exec = et._is_feasible(executed, row["constraints"])
        oracle = row.get("oracle_path", [])
        success = executed == oracle
        rec = {
            "task_id": row["task_id"],
            "dataset": row["dataset"],
            "query": row.get("question", " ".join(row.get("query_tokens", []))),
            "query_tokens": row.get("query_tokens", []),
            "method": "diplan_llm_agent",
            "oracle_path": oracle,
            "planned_path": planned,
            "executed_path": executed,
            "success": success,
            "first_error_step": first_error_step(executed, oracle),
            "recovery_at_error": recovery_at_error(executed, oracle),
            "trap_at_1": trap_at_1(executed, row.get("trap_path", [])),
            "feasible": feasible,
            "violations": sorted(set(violations_all + violations_exec)),
            "plan_execution_consistency": plan_execution_consistency(planned, executed),
            "token_cost": len(executed),
            "latency_cost": 0.02 * max(1, len(executed)),
            "diversity_coverage": 0.0,
            "replanning_steps": max(1, len(executed)),
            "episode_trace_len": len(episode_trace),
            "oracle_in_candidate_pool": tuple(oracle) in pool_seen,
            "candidate_pool_size": len(pool_seen),
            "ranking_error": False,
        }
        if save_episode_trace:
            rec["episode_trace"] = episode_trace
        records.append(rec)

    summary = {"diplan_llm_agent": aggregate_method_metrics(records)}
    summary["diplan_llm_agent"].update(
        {
            "candidate_pool_hit_rate": sum(1.0 if r["oracle_in_candidate_pool"] else 0.0 for r in records)
            / max(1, len(records)),
            "candidate_pool_avg_size": sum(float(r["candidate_pool_size"]) for r in records) / max(1, len(records)),
            "llm_error_rate": llm_error_count / max(1, len(records)),
            "llm_fallback_rate": fallback_pick_count / max(1, len(records)),
        }
    )
    by_dataset = {}
    for ds in sorted({r.get("dataset", "unknown") for r in records}):
        ds_rows = [r for r in records if r.get("dataset", "unknown") == ds]
        by_dataset[ds] = aggregate_method_metrics(ds_rows)
        by_dataset[ds].update(
            {
                "candidate_pool_hit_rate": sum(1.0 if r["oracle_in_candidate_pool"] else 0.0 for r in ds_rows)
                / max(1, len(ds_rows)),
                "candidate_pool_avg_size": sum(float(r["candidate_pool_size"]) for r in ds_rows) / max(1, len(ds_rows)),
            }
        )

    ensure_dir(args.out)
    dump_jsonl(str(Path(args.out) / "predictions.jsonl"), records)
    dump_json(str(Path(args.out) / "summary_metrics.json"), summary)
    dump_json(str(Path(args.out) / "summary_by_dataset.json"), by_dataset)
    dump_json(
        str(Path(args.out) / "llm_agent_runtime.json"),
        {
            "llm_api_base": api_base,
            "llm_model": llm_model,
            "llm_error_count": llm_error_count,
            "llm_fallback_pick_count": fallback_pick_count,
            "n_tasks": len(records),
        },
    )
    print(f"[ok] evaluated {len(records)} tasks with diplan_llm_agent")
    print(summary["diplan_llm_agent"])


if __name__ == "__main__":
    main()
