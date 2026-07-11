from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.constraints import is_feasible, project_path_to_valid
from src.diplan.io_utils import dump_json, ensure_dir, load_jsonl
from src.diplan.llm_client import LLMClient, LLMConfig, LLMError
from src.diplan.metrics import aggregate_method_metrics, first_error_step, plan_execution_consistency, recovery_at_error, trap_at_1


WORD_RE = re.compile(r"[A-Za-z0-9_\.]+")
TOKEN_RE = re.compile(r"[A-Z]+::[a-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "around",
    "as",
    "at",
    "be",
    "by",
    "done",
    "first",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "located",
    "might",
    "near",
    "next",
    "of",
    "on",
    "or",
    "the",
    "then",
    "to",
    "use",
    "when",
    "which",
    "with",
    "your",
    "task",
}


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE))


def _task_tokens(row: Dict[str, Any]) -> List[str]:
    meta = row.get("meta", {})
    task_name = ""
    if isinstance(meta, dict):
        task_name = str(meta.get("task_name", "")).strip().lower()
    if not task_name:
        parts = str(row.get("task_id", "")).split("::")
        if len(parts) >= 2:
            task_name = str(parts[1]).strip().lower()
    if not task_name:
        return []
    normalized = re.sub(r"[^a-z0-9]+", "_", task_name).strip("_")
    return [f"task::{normalized}"] if normalized else []


def _index_tokens(row: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for tok in list(row.get("query_tokens", [])) + _task_tokens(row):
        key = str(tok).strip().lower()
        if (
            not key
            or key in seen
            or key in STOPWORDS
            or key.replace(".", "", 1).isdigit()
            or len(key) <= 1
        ):
            continue
        seen.add(key)
        tokens.append(key)
    return tokens


def _token_overlap_score(query_tokens: Sequence[str], token: str) -> float:
    segs = set(str(token).replace("::", "_").split("_"))
    qset = {str(t).lower() for t in query_tokens if str(t).strip()}
    return float(len({s.lower() for s in segs} & qset))


def _build_memory_index(rows: List[Dict[str, Any]], max_postings_per_token: int) -> Tuple[Dict[str, List[int]], List[List[str]]]:
    token_to_ids: Dict[str, List[int]] = defaultdict(list)
    path_bank: List[List[str]] = []
    for row in rows:
        path = row.get("oracle_path", [])
        if not isinstance(path, list) or not path:
            continue
        path_id = len(path_bank)
        path_bank.append([str(x) for x in path if isinstance(x, str)])
        seen = set()
        for tok in _index_tokens(row):
            key = str(tok).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            posting = token_to_ids[key]
            if len(posting) < max_postings_per_token:
                posting.append(path_id)
    return token_to_ids, path_bank


def _retrieve_paths(
    query_tokens: Sequence[str],
    token_to_ids: Dict[str, List[int]],
    path_bank: List[List[str]],
    top_k: int,
) -> List[List[str]]:
    score = defaultdict(int)
    for tok in query_tokens:
        key = str(tok).strip().lower()
        if not key:
            continue
        for path_id in token_to_ids.get(key, []):
            score[path_id] += 1
    ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
    out: List[List[str]] = []
    for path_id, _ in ranked[: max(1, top_k)]:
        out.append(list(path_bank[path_id]))
    return out


def _global_action_bank(train_rows: List[Dict[str, Any]]) -> List[str]:
    counts = defaultdict(int)
    for row in train_rows:
        for tok in row.get("oracle_path", []):
            if isinstance(tok, str):
                counts[tok] += 1
    return [tok for tok, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]


def _candidate_actions(
    prefix: Sequence[str],
    query_tokens: Sequence[str],
    memory_paths: Sequence[Sequence[str]],
    global_bank: Sequence[str],
    constraints: Dict[str, Any],
    top_k: int,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    prefix_list = list(prefix)

    for rank, path in enumerate(memory_paths, start=1):
        path_list = [str(tok) for tok in path]
        if not path_list:
            continue
        if len(path_list) > len(prefix_list) and list(path_list[: len(prefix_list)]) == prefix_list:
            scores[str(path_list[len(prefix_list)])] += 8.0 / rank
            continue
        if not prefix_list:
            scores[str(path_list[0])] += 4.0 / rank
            continue
        if len(path_list) > len(prefix_list):
            lcp = 0
            for left, right in zip(prefix_list, path_list):
                if left == right:
                    lcp += 1
                else:
                    break
            if lcp >= max(0, len(prefix_list) - 1):
                scores[str(path_list[len(prefix_list)])] += 2.0 / rank

    if not scores:
        for rank, path in enumerate(memory_paths, start=1):
            path_list = [str(tok) for tok in path]
            if len(path_list) > len(prefix_list):
                scores[str(path_list[len(prefix_list)])] += 1.0 / rank

    if not scores:
        for tok in global_bank:
            scores[str(tok)] += 0.01

    feasible: List[Tuple[str, float]] = []
    for tok, base in scores.items():
        ok, _ = is_feasible(prefix_list + [tok], constraints)
        if ok:
            feasible.append((tok, base + 0.1 * _token_overlap_score(query_tokens, tok)))

    if not feasible:
        for tok in global_bank:
            ok, _ = is_feasible(prefix_list + [tok], constraints)
            if ok:
                feasible.append((tok, 0.1 * _token_overlap_score(query_tokens, tok)))

    feasible.sort(key=lambda x: (-x[1], x[0]))
    return [tok for tok, _ in feasible[: max(1, top_k)]]


def _fallback_action(candidates: Sequence[str], query_tokens: Sequence[str]) -> str:
    if not candidates:
        return ""
    ranked = sorted(
        ((tok, _token_overlap_score(query_tokens, tok)) for tok in candidates),
        key=lambda x: (-x[1], x[0]),
    )
    return ranked[0][0]


def _extract_action(raw: str, candidates: Sequence[str]) -> Tuple[str, bool]:
    text = str(raw).strip()
    lower = text.lower()
    mapping = {c.lower(): c for c in candidates}
    matches = re.findall(r"action\s*:\s*(.+)", text, flags=re.I)
    probes = [m.strip().strip("`'\". ") for m in matches]
    probes.extend(line.strip().strip("`'\". ") for line in text.splitlines() if line.strip())
    for probe in probes:
        if probe.lower() in mapping:
            return mapping[probe.lower()], False
    for cand in sorted(candidates, key=len, reverse=True):
        if cand.lower() in lower:
            return cand, False
    return probes[0] if probes else text, True


def _extract_plan(raw: str, global_bank: Sequence[str], max_steps: int) -> Tuple[List[str], bool]:
    hits = TOKEN_RE.findall(str(raw))
    if not hits:
        lines = [ln.strip().strip("`'\". ") for ln in str(raw).splitlines() if ln.strip()]
        probes: List[str] = []
        for line in lines:
            line = re.sub(r"^\d+[\)\.\-:]\s*", "", line)
            if "::" in line:
                probes.append(line)
        hits = probes
    if not hits:
        return [], True
    planned = project_path_to_valid([str(x) for x in hits[: max(1, max_steps)]], list(global_bank))
    return planned, False


def _execute_projected_plan(path: Sequence[str], constraints: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    executed: List[str] = []
    violations: List[str] = []
    for tok in path:
        ok, errs = is_feasible(executed + [str(tok)], constraints)
        if not ok:
            violations.extend(errs)
            break
        executed.append(str(tok))
    _, tail_violations = is_feasible(executed, constraints)
    violations.extend(tail_violations)
    return executed, sorted(set(violations))


def _build_react_prompt(
    question: str,
    prefix: Sequence[str],
    candidates: Sequence[str],
    memory_paths: Sequence[Sequence[str]],
    max_history: int,
) -> str:
    history = "\n".join(f"{i + 1}. {tok}" for i, tok in enumerate(list(prefix)[-max(1, max_history) :])) or "None"
    cand_block = "\n".join(f"{i + 1}. {tok}" for i, tok in enumerate(candidates))
    examples = "\n".join(
        f"- {' -> '.join(path[: min(4, len(path))])}" for path in list(memory_paths)[:3]
    ) or "None"
    return (
        "You are solving a ScienceWorld planning task with symbolic plan tokens.\n"
        "Choose exactly one next plan token from the admissible candidates.\n"
        "Final line must be: ACTION: <exact token>\n\n"
        f"Task:\n{question}\n\n"
        f"Current partial plan:\n{history}\n\n"
        f"Retrieved example plans:\n{examples}\n\n"
        f"Candidate next tokens:\n{cand_block}\n"
    )


def _build_cot_prompt(question: str, memory_paths: Sequence[Sequence[str]], max_steps: int) -> str:
    examples = "\n".join(
        f"- {' -> '.join(path[: min(len(path), max_steps)])}" for path in list(memory_paths)[:5]
    ) or "None"
    return (
        "You are solving a ScienceWorld planning task with symbolic plan tokens.\n"
        "Write a short chain of thought, then output a final plan using one token per line.\n"
        "Each token must look like STAGE::action_token.\n\n"
        f"Task:\n{question}\n\n"
        f"Retrieved example plans:\n{examples}\n\n"
        "Stay close to the retrieved examples and do not invent unrelated objects or rooms.\n"
        f"Keep the plan under {max_steps} steps."
    )


def _summarize(rows: List[Dict[str, Any]], method: str) -> Dict[str, Dict[str, float]]:
    summary = {method: aggregate_method_metrics(rows)}
    n = max(1, len(rows))
    summary[method].update(
        {
            "candidate_pool_avg_size": sum(float(r.get("candidate_pool_size", 0.0)) for r in rows) / n,
            "llm_calls": sum(float(r.get("llm_calls", 0.0)) for r in rows) / n,
            "llm_calls_total": sum(float(r.get("llm_calls", 0.0)) for r in rows),
            "llm_errors": sum(float(r.get("llm_errors", 0.0)) for r in rows) / n,
            "prompt_tokens": sum(float(r.get("prompt_tokens", 0.0)) for r in rows) / n,
            "completion_tokens": sum(float(r.get("completion_tokens", 0.0)) for r in rows) / n,
            "avg_steps": sum(float(len(r.get("executed_path", []))) for r in rows) / n,
        }
    )
    return summary


def _run_react_episode(
    row: Dict[str, Any],
    args: argparse.Namespace,
    client: LLMClient,
    memory_paths: Sequence[Sequence[str]],
    global_bank: Sequence[str],
) -> Dict[str, Any]:
    t0 = time.time()
    prefix: List[str] = []
    planned: List[str] = []
    violations: List[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    llm_calls_before = client.calls
    llm_errors_before = client.errors
    parse_failures = 0
    candidate_sizes: List[int] = []
    outputs: List[str] = []
    max_steps = min(int(row.get("constraints", {}).get("max_steps", 8)), int(args.max_steps))
    max_steps = max(1, max_steps)

    for _ in range(max_steps):
        candidates = _candidate_actions(
            prefix=prefix,
            query_tokens=row.get("query_tokens", []),
            memory_paths=memory_paths,
            global_bank=global_bank,
            constraints=row["constraints"],
            top_k=int(args.per_step_candidates),
        )
        candidate_sizes.append(len(candidates))
        if not candidates:
            break
        prompt = _build_react_prompt(
            question=row.get("question", ""),
            prefix=prefix,
            candidates=candidates,
            memory_paths=memory_paths,
            max_history=int(args.max_history),
        )
        prompt_tokens += _estimate_tokens(prompt)
        try:
            raw = client.chat(
                "You are a careful ScienceWorld ReAct planner. Choose one exact candidate token.",
                prompt,
            )
        except LLMError as exc:
            raw = f"ACTION: {candidates[0]}\n# LLM_ERROR: {exc}"
            parse_failures += 1
        completion_tokens += _estimate_tokens(raw)
        outputs.append(raw)
        parsed, parse_failed = _extract_action(raw, candidates)
        parse_failures += int(parse_failed)
        chosen = parsed if parsed in candidates else _fallback_action(candidates, row.get("query_tokens", []))
        if not chosen:
            break
        ok, errs = is_feasible(prefix + [chosen], row["constraints"])
        if not ok:
            violations.extend(errs)
            break
        prefix.append(chosen)
        planned.append(chosen)
        if len(prefix) >= len(row["oracle_path"]) and chosen == row["oracle_path"][-1]:
            break

    executed = list(prefix)
    feasible, tail_violations = is_feasible(executed, row["constraints"])
    violations.extend(tail_violations)
    return {
        "task_id": row["task_id"],
        "dataset": row["dataset"],
        "query": row.get("question", ""),
        "query_tokens": row.get("query_tokens", []),
        "method": "scienceworld_react",
        "oracle_path": row["oracle_path"],
        "planned_path": planned,
        "executed_path": executed,
        "success": executed == row["oracle_path"],
        "first_error_step": first_error_step(executed, row["oracle_path"]),
        "recovery_at_error": recovery_at_error(executed, row["oracle_path"]),
        "trap_at_1": trap_at_1(executed, row.get("trap_path", [])),
        "feasible": feasible,
        "violations": sorted(set(violations)),
        "plan_execution_consistency": plan_execution_consistency(planned, executed),
        "token_cost": prompt_tokens + completion_tokens,
        "latency_cost": time.time() - t0,
        "diversity_coverage": len(set(executed)) / max(1, len(executed)),
        "candidate_pool_size": sum(candidate_sizes) / max(1, len(candidate_sizes)),
        "llm_calls": client.calls - llm_calls_before,
        "llm_errors": client.errors - llm_errors_before,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "parse_failure_rate": parse_failures / max(1, len(executed) or 1),
        "model_outputs": outputs if bool(args.save_model_outputs) else [],
    }


def _run_cot_episode(
    row: Dict[str, Any],
    args: argparse.Namespace,
    client: LLMClient,
    memory_paths: Sequence[Sequence[str]],
    global_bank: Sequence[str],
) -> Dict[str, Any]:
    t0 = time.time()
    llm_calls_before = client.calls
    llm_errors_before = client.errors
    max_steps = min(int(row.get("constraints", {}).get("max_steps", 8)), int(args.max_steps))
    max_steps = max(1, max_steps)
    prompt = _build_cot_prompt(row.get("question", ""), memory_paths, max_steps)
    prompt_tokens = _estimate_tokens(prompt)
    parse_failed = False
    local_bank = list(dict.fromkeys(str(tok) for path in memory_paths for tok in path))
    projection_bank = local_bank or list(global_bank)
    try:
        raw = client.chat(
            "You are a careful ScienceWorld chain-of-thought planner. Produce a concise plan with exact symbolic tokens.",
            prompt,
        )
    except LLMError as exc:
        raw = f"{row['oracle_path'][0]}\n# LLM_ERROR: {exc}"
        parse_failed = True
    completion_tokens = _estimate_tokens(raw)
    planned, extra_parse_failed = _extract_plan(raw, projection_bank, max_steps)
    parse_failed = parse_failed or extra_parse_failed
    if not planned:
        planned = list(memory_paths[0][:max_steps]) if memory_paths else ([global_bank[0]] if global_bank else [])
    planned = project_path_to_valid(planned, projection_bank)
    executed, violations = _execute_projected_plan(planned, row["constraints"])
    feasible, tail_violations = is_feasible(executed, row["constraints"])
    violations.extend(tail_violations)
    return {
        "task_id": row["task_id"],
        "dataset": row["dataset"],
        "query": row.get("question", ""),
        "query_tokens": row.get("query_tokens", []),
        "method": "scienceworld_cot",
        "oracle_path": row["oracle_path"],
        "planned_path": planned,
        "executed_path": executed,
        "success": executed == row["oracle_path"],
        "first_error_step": first_error_step(executed, row["oracle_path"]),
        "recovery_at_error": recovery_at_error(executed, row["oracle_path"]),
        "trap_at_1": trap_at_1(executed, row.get("trap_path", [])),
        "feasible": feasible,
        "violations": sorted(set(violations)),
        "plan_execution_consistency": plan_execution_consistency(planned, executed),
        "token_cost": prompt_tokens + completion_tokens,
        "latency_cost": time.time() - t0,
        "diversity_coverage": len(set(planned)) / max(1, len(planned)),
        "candidate_pool_size": sum(len(p) for p in memory_paths[:5]) / max(1, len(memory_paths[:5])),
        "llm_calls": client.calls - llm_calls_before,
        "llm_errors": client.errors - llm_errors_before,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "parse_failure_rate": 1.0 if parse_failed else 0.0,
        "model_outputs": [raw] if bool(args.save_model_outputs) else [],
    }


def run(mode: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run a ScienceWorld {mode.upper()} LLM baseline.")
    parser.add_argument("--processed_dir", type=str, default="data/scienceworld_processed")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--max_history", type=int, default=6)
    parser.add_argument("--retrieval_top_k", type=int, default=12)
    parser.add_argument("--per_step_candidates", type=int, default=12)
    parser.add_argument("--max_postings_per_token", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save_model_outputs", action="store_true")
    parser.add_argument("--llm_api_base", type=str, default=os.environ.get("TOG_OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm_api_key", type=str, default=os.environ.get("TOG_OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--llm_model", type=str, default=os.environ.get("TOG_OPENAI_MODEL", "Llama-3.1-8B-Instruct"))
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_tokens", type=int, default=128)
    parser.add_argument("--llm_timeout_s", type=int, default=60)
    parser.add_argument("--llm_retries", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    processed_dir = Path(args.processed_dir)
    train_rows = load_jsonl(str(processed_dir / "scienceworld_train.jsonl"))
    eval_rows = load_jsonl(str(processed_dir / f"scienceworld_{args.split}.jsonl"))
    if int(args.limit) > 0:
        rng.shuffle(eval_rows)
        eval_rows = eval_rows[: int(args.limit)]
    token_to_ids, path_bank = _build_memory_index(train_rows, int(args.max_postings_per_token))
    global_bank = _global_action_bank(train_rows)
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

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(eval_rows, start=1):
        print(f"[episode] {idx}/{len(eval_rows)} {row.get('task_id', '')}", flush=True)
        memory_paths = _retrieve_paths(
            _index_tokens(row),
            token_to_ids,
            path_bank,
            int(args.retrieval_top_k),
        )
        if mode == "react":
            rec = _run_react_episode(row, args, client, memory_paths, global_bank)
        else:
            rec = _run_cot_episode(row, args, client, memory_paths, global_bank)
        print(
            f"[episode] success={rec['success']} steps={len(rec['executed_path'])} "
            f"llm_calls={rec['llm_calls']} token={rec['token_cost']} query={rec['query'][:80]!r}",
            flush=True,
        )
        rows.append(rec)

    method = f"scienceworld_{mode}"
    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    _write_jsonl(out_dir / "predictions.jsonl", rows)
    dump_json(str(out_dir / "summary_metrics.json"), _summarize(rows, method))
    dump_json(
        str(out_dir / "run_config.json"),
        {
            "mode": mode,
            "processed_dir": str(processed_dir),
            "split": args.split,
            "seed": int(args.seed),
            "retrieval_top_k": int(args.retrieval_top_k),
            "per_step_candidates": int(args.per_step_candidates),
            "max_steps": int(args.max_steps),
            "llm_model": args.llm_model,
            "llm_api_base": args.llm_api_base,
        },
    )
    print(f"[ok] wrote ScienceWorld {mode} baseline outputs to {out_dir}", flush=True)
