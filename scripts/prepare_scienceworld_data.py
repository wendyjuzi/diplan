import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir


WORD_RE = re.compile(r"[A-Za-z0-9_\.]+")


def _env_call(env, snake_name: str, camel_name: str, *args, **kwargs):
    fn = getattr(env, snake_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    fn = getattr(env, camel_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    raise AttributeError(f"ScienceWorldEnv has neither {snake_name} nor {camel_name}")


def _tokenize(text: str, max_len: int = 64) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall((text or "").lower()) if t]
    return toks[:max_len]


def _normalize_action(raw: str) -> str:
    txt = str(raw or "").strip().lower()
    txt = re.sub(r"\s+", "_", txt)
    txt = txt.replace("(", "_").replace(")", "")
    txt = txt.replace(",", "_").replace(":", "_")
    txt = re.sub(r"[^a-z0-9_]+", "_", txt).strip("_")
    return txt or "noop"


def _stage_for_action(norm_action: str) -> str:
    a = norm_action.upper()
    if "LOOK" in a or "OBSERVE" in a or "EXAMINE" in a:
        return "OBSERVE"
    if "GO_TO" in a or "MOVE" in a or "OPEN_DOOR" in a or "CLOSE_DOOR" in a:
        return "NAV"
    if "MIX" in a or "COMBINE" in a or "HEAT" in a or "COOL" in a or "BOIL" in a:
        return "OPERATE"
    if "READ" in a or "MEASURE" in a or "CHECK" in a or "VERIFY" in a:
        return "VERIFY"
    return "ACT"


def _to_oracle_path(gold_actions: List[str]) -> List[str]:
    out: List[str] = []
    for raw in gold_actions:
        norm = _normalize_action(raw)
        stage = _stage_for_action(norm)
        out.append(f"{stage}::{norm}")
    return out


def _to_stage_path(gold_actions: List[str], dedupe: bool = True) -> List[str]:
    out: List[str] = []
    last_stage = ""
    for raw in gold_actions:
        norm = _normalize_action(raw)
        stage = _stage_for_action(norm)
        if dedupe and stage == last_stage:
            continue
        out.append(stage)
        last_stage = stage
    return out


def _build_trap_path(oracle_path: List[str], rng: random.Random) -> List[str]:
    trap = list(oracle_path)
    if not trap:
        return trap
    idx = rng.randrange(len(trap))
    tok = trap[idx]
    if "::" in tok:
        stage, action = tok.split("::", 1)
        trap[idx] = f"{stage}::{action}_wrong"
    else:
        trap[idx] = f"{tok}_wrong"
    return trap


def _build_constraints(path: List[str]) -> Dict:
    stages = []
    seen = set()
    for tok in path:
        st = tok.split("::", 1)[0].strip().upper()
        if st and st not in seen:
            seen.add(st)
            stages.append(st)
    if len(stages) < 2:
        stages = ["OBSERVE", "ACT", "VERIFY"]
    return {
        "max_steps": min(96, max(6, len(path) + 2)),
        "required_stage_order": stages,
        "allow_stage_reentry": True,
        "must_precede": [{"first": stages[i], "second": stages[i + 1]} for i in range(len(stages) - 1)],
        "required_before": {stages[i + 1]: [stages[i]] for i in range(len(stages) - 1)},
        "forbidden_actions": [],
    }


def _build_rows(
    env,
    task_names: List[str],
    per_task_limit: int,
    simplification: str,
    seed: int,
    path_mode: str = "action",
) -> List[Dict]:
    return _build_rows_with_path_mode(env, task_names, per_task_limit, simplification, seed, path_mode=path_mode)


def _build_rows_with_path_mode(
    env,
    task_names: List[str],
    per_task_limit: int,
    simplification: str,
    seed: int,
    path_mode: str,
) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    mode = str(path_mode).lower().strip()
    for task in task_names:
        try:
            max_var = int(_env_call(env, "get_max_variations", "getMaxVariations", task))
        except Exception:
            continue
        if max_var <= 0:
            continue
        var_ids = list(range(max_var))
        rng.shuffle(var_ids)
        if per_task_limit > 0:
            var_ids = var_ids[:per_task_limit]
        for vid in var_ids:
            try:
                env.load(task, int(vid), simplification, generateGoldPath=True)
                obs, _ = env.reset()
                gold_actions = list(_env_call(env, "get_gold_action_sequence", "getGoldActionSequence") or [])
            except Exception:
                continue
            if not gold_actions:
                continue
            if len(gold_actions) == 1 and "ERROR: Gold path was not generated" in str(gold_actions[0]):
                continue
            if mode == "action":
                oracle_path = _to_oracle_path(gold_actions)
            elif mode in {"stage", "stage_dedup"}:
                oracle_path = _to_stage_path(gold_actions, dedupe=(mode == "stage_dedup"))
            else:
                raise ValueError(f"Unsupported path_mode: {path_mode}")
            if len(oracle_path) < 2:
                continue
            question = str(_env_call(env, "get_task_description", "getTaskDescription") or obs or task)
            q_tokens = _tokenize(question)
            rows.append(
                {
                    "task_id": f"scienceworld::{task}::{vid}",
                    "dataset": "scienceworld",
                    "question": question,
                    "query_tokens": q_tokens,
                    "oracle_path": oracle_path,
                    "trap_path": _build_trap_path(oracle_path, rng),
                    "constraints": _build_constraints(oracle_path),
                    "meta": {
                        "task_name": task,
                        "variation": int(vid),
                        "simplification": simplification,
                        "path_mode": mode,
                        "gold_path_len": len(gold_actions),
                    },
                }
            )
    return rows


def _split_rows(rows: List[Dict], seed: int, ratios: List[float]) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    data = list(rows)
    rng.shuffle(data)
    n = len(data)
    r_train, r_val, r_test = ratios
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    n_test = n - n_train - n_val
    if n > 0 and n_test <= 0:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/scienceworld_processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server_path", type=str, default="")
    parser.add_argument("--env_step_limit", type=int, default=120)
    parser.add_argument("--task_names", type=str, nargs="*", default=[])
    parser.add_argument("--per_task_limit", type=int, default=4)
    parser.add_argument("--simplification", type=str, default="easy")
    parser.add_argument("--path_mode", type=str, default="action", choices=["action", "stage", "stage_dedup"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    from scienceworld import ScienceWorldEnv

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))

    env = ScienceWorldEnv("", args.server_path or None, envStepLimit=int(args.env_step_limit))
    try:
        all_tasks = list(_env_call(env, "get_task_names", "getTaskNames"))
        task_names = [t for t in args.task_names if t in all_tasks] if args.task_names else all_tasks
        if not task_names:
            raise ValueError("No valid task names found in ScienceWorld.")
        rows = _build_rows(
            env=env,
            task_names=task_names,
            per_task_limit=int(args.per_task_limit),
            simplification=str(args.simplification),
            seed=int(args.seed),
            path_mode=str(args.path_mode),
        )
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No ScienceWorld rows exported. Try different task_names/per_task_limit/simplification.")

    split = _split_rows(
        rows,
        seed=int(args.seed),
        ratios=[float(args.train_ratio), float(args.val_ratio), float(args.test_ratio)],
    )
    dump_jsonl(str(out_dir / "scienceworld_train.jsonl"), split["train"])
    dump_jsonl(str(out_dir / "scienceworld_val.jsonl"), split["val"])
    dump_jsonl(str(out_dir / "scienceworld_test.jsonl"), split["test"])
    dump_json(
        str(out_dir / "manifest.json"),
        {
            "dataset": "scienceworld",
            "seed": int(args.seed),
            "simplification": str(args.simplification),
            "path_mode": str(args.path_mode),
            "task_count": len(task_names),
            "rows_total": len(rows),
            "counts": {"train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"])},
        },
    )
    print(
        f"[ok] scienceworld exported: total={len(rows)} "
        f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )


if __name__ == "__main__":
    main()
