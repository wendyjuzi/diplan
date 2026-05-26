import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .clinical_collab_fsm import (
    build_clinical_collab_constraints,
    build_clinical_collab_state_machine,
)
from .io_utils import dump_json, dump_jsonl, ensure_dir


WORD_RE = re.compile(r"[A-Za-z0-9_\.]+")


def _load_any(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(p, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "examples", "items", "cases"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Unsupported JSON structure: {path}")


def tokenize_question(text: str, max_len: int = 48) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall(text.lower()) if t]
    return toks[:max_len]


def _extract_question(item: Dict[str, Any], question_keys: Optional[List[str]] = None) -> str:
    keys = question_keys or [
        "instruction",
        "question",
        "prompt",
        "task",
        "chief_complaint",
        "case_summary",
        "text",
    ]
    for key in keys:
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_action_token(
    step: Any,
    step_stage_key: str = "stage",
    step_action_keys: Optional[List[str]] = None,
) -> str:
    if isinstance(step, str):
        return step.strip()
    if not isinstance(step, dict):
        return ""
    stage = str(step.get(step_stage_key, "")).strip().upper()
    action = ""
    for k in (step_action_keys or ["action", "name", "tool", "intent", "op"]):
        v = step.get(k)
        if isinstance(v, str) and v.strip():
            action = v.strip()
            break
    if not action:
        return ""
    if stage:
        return f"{stage}::{action}"
    return action


def _extract_oracle_path(
    item: Dict[str, Any],
    path_keys: Optional[List[str]] = None,
    step_stage_key: str = "stage",
    step_action_keys: Optional[List[str]] = None,
) -> List[str]:
    keys = path_keys or ["oracle_path", "gold_actions", "action_path", "plan", "trajectory", "steps"]
    for key in keys:
        v = item.get(key)
        if isinstance(v, list) and v:
            out = []
            for s in v:
                tok = _extract_action_token(
                    s,
                    step_stage_key=step_stage_key,
                    step_action_keys=step_action_keys,
                )
                if tok:
                    out.append(tok)
            if out:
                return out
    return []


def _compact_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_tag_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove inline tags like <对医生讲> / <结束>.
    text = re.sub(r"<[^>]+>", " ", text)
    return _compact_text(text)


def _contains_any(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    return any(p in text for p in patterns)


def _extract_aihospital_question(patient_row: Dict[str, Any], dialog_row: Dict[str, Any]) -> str:
    mr = patient_row.get("medical_record") if isinstance(patient_row, dict) else {}
    if not isinstance(mr, dict):
        mr = {}
    chief = _compact_text(str(mr.get("主诉", "")))
    hpi = _compact_text(str(mr.get("现病史", "")))
    title = _compact_text(str(patient_row.get("title", "")))

    first_patient = ""
    for turn in dialog_row.get("dialog_history", []):
        if str(turn.get("role", "")).strip() == "Patient":
            first_patient = _strip_tag_prefix(str(turn.get("content", "")))
            if first_patient:
                break

    parts: List[str] = []
    if title:
        parts.append(f"Case title: {title}")
    if chief:
        parts.append(f"Chief complaint: {chief}")
    if hpi:
        parts.append(f"History of present illness: {hpi}")
    if first_patient:
        parts.append(f"Patient utterance: {first_patient}")
    return " | ".join(parts).strip()


def _extract_aihospital_oracle_path(patient_row: Dict[str, Any], dialog_row: Dict[str, Any]) -> List[str]:
    stage_to_action: Dict[str, str] = {}

    def add(stage: str, action: str) -> None:
        if stage not in stage_to_action:
            stage_to_action[stage] = action

    dialog = dialog_row.get("dialog_history", [])
    if not isinstance(dialog, list):
        dialog = []

    for turn in dialog:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip()
        content = _strip_tag_prefix(str(turn.get("content", "")))
        if not content:
            continue

        if role == "Doctor":
            if _contains_any(content, ["您好", "哪里不舒服", "请问", "症状", "病史", "多久"]):
                add("INTAKE", "history_taking")
            if _contains_any(
                content,
                ["检查", "检验", "化验", "CT", "MRI", "超声", "X光", "X线", "血常规", "辅助检查"],
            ):
                add("EXAM", "order_or_interpret_tests")
            if _contains_any(content, ["诊断", "考虑", "初步"]):
                add("DIAGNOSIS", "formulate_diagnosis")
            if _contains_any(content, ["治疗", "处方", "用药", "方案", "手术", "建议"]):
                add("PRESCRIBE", "propose_treatment")
            if _contains_any(content, ["复诊", "复查", "随访", "观察病情"]):
                add("FOLLOWUP", "arrange_followup")
        elif role == "Reporter":
            add("EXAM", "review_test_results")
        elif role == "Patient":
            if _contains_any(content, ["结束", "谢谢医生", "会复诊", "按时复查"]):
                add("FOLLOWUP", "accept_followup_plan")

    # Backfill stages from structured medical record if dialogue misses them.
    mr = patient_row.get("medical_record") if isinstance(patient_row, dict) else {}
    if not isinstance(mr, dict):
        mr = {}
    if "INTAKE" not in stage_to_action and _compact_text(str(mr.get("主诉", ""))):
        add("INTAKE", "collect_chief_complaint")
    if "EXAM" not in stage_to_action and _compact_text(str(mr.get("辅助检查", ""))):
        add("EXAM", "review_auxiliary_exams")
    if "DIAGNOSIS" not in stage_to_action and (
        _compact_text(str(mr.get("诊断结果", ""))) or _compact_text(str(mr.get("初步诊断", "")))
    ):
        add("DIAGNOSIS", "confirm_diagnosis")
    if "PRESCRIBE" not in stage_to_action and _compact_text(str(mr.get("诊治经过", ""))):
        add("PRESCRIBE", "plan_treatment")
    if "FOLLOWUP" not in stage_to_action:
        add("FOLLOWUP", "discharge_and_followup")

    canonical_order = ["INTAKE", "EXAM", "DIAGNOSIS", "PRESCRIBE", "FOLLOWUP"]
    path = []
    for stage in canonical_order:
        action = stage_to_action.get(stage)
        if action:
            path.append(f"{stage}::{action}")
    return path


def _normalize_ai_hospital_repo(
    repo_root: str,
    dialog_file: str = "dialog_history_gpt4.jsonl",
    min_path_len: int = 4,
    max_path_len: int = 32,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(repo_root)
    patients_path = root / "src" / "data" / "patients.json"
    dialog_path = root / "src" / "outputs" / "dialog_history_iiyi" / dialog_file
    if not patients_path.exists():
        raise FileNotFoundError(f"AI_Hospital patients file not found: {patients_path}")
    if not dialog_path.exists():
        raise FileNotFoundError(f"AI_Hospital dialog file not found: {dialog_path}")

    patients_rows = _load_any(str(patients_path))
    dialogs_rows = _load_any(str(dialog_path))
    id_to_patient = {}
    for row in patients_rows:
        pid = row.get("id")
        if pid is not None:
            id_to_patient[str(pid)] = row

    rng = random.Random(seed)
    out = []
    skipped = {"patient_not_found": 0, "no_question": 0, "no_path": 0, "path_len_out_of_range": 0}
    total = 0

    for idx, drow in enumerate(dialogs_rows):
        total = idx + 1
        pid = drow.get("patient_id")
        prow = id_to_patient.get(str(pid))
        if prow is None:
            skipped["patient_not_found"] += 1
            continue
        question = _extract_aihospital_question(prow, drow)
        if not question:
            skipped["no_question"] += 1
            continue
        path = _extract_aihospital_oracle_path(prow, drow)
        if not path:
            skipped["no_path"] += 1
            continue
        if not (min_path_len <= len(path) <= max_path_len):
            skipped["path_len_out_of_range"] += 1
            continue

        cst = _default_clinical_constraints(path, rng)
        trap = cst.pop("_trap_path")
        out.append(
            {
                "task_id": f"ai_hospital_{int(pid):07d}" if isinstance(pid, int) else f"ai_hospital_{idx:07d}",
                "dataset": "ai_hospital",
                "question": question,
                "query_tokens": tokenize_question(question),
                "oracle_path": path,
                "trap_path": trap,
                "candidate_paths": [path],
                "constraints": cst,
                "meta": {
                    "source_repo": str(root),
                    "source_dialog_file": str(dialog_path.name),
                    "source_patient_id": pid,
                    "mapped_from_fields": {
                        "question": [
                            "patients.json: title",
                            "patients.json: medical_record.主诉",
                            "patients.json: medical_record.现病史",
                            "dialog_history: first Patient utterance",
                        ],
                        "oracle_path": [
                            "dialog_history[*].role/content -> stage-action rules",
                            "patients.json medical_record fallback: 辅助检查/诊断结果/诊治经过",
                        ],
                    },
                },
            }
        )

    stats = {
        "dataset": "ai_hospital",
        "total_input": total,
        "kept": len(out),
        "skipped": skipped,
        "source_patients_file": str(patients_path),
        "source_dialog_file": str(dialog_path),
    }
    return out, stats


def _default_clinical_constraints(oracle_path: List[str], rng: random.Random) -> Dict[str, Any]:
    cst = build_clinical_collab_constraints(oracle_path)
    stage_order = list(cst.get("required_stage_order", []))
    fsm = build_clinical_collab_state_machine(stage_order=stage_order)
    # Lightly perturbed trap action from oracle.
    trap = list(oracle_path)
    if trap:
        j = rng.randrange(len(trap))
        trap[j] = trap[j].replace("::", "_TRAP::", 1) if "::" in trap[j] else f"TRAP::{trap[j]}"
    cst["_trap_path"] = trap
    cst["collab_fsm"] = fsm
    return cst


def normalize_clinical_rows(
    rows: Iterable[Dict[str, Any]],
    dataset_name: str,
    seed: int,
    field_mapping: Optional[Dict[str, Any]] = None,
    min_path_len: int = 4,
    max_path_len: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    mapping = field_mapping or {}
    q_keys = mapping.get("question_keys")
    p_keys = mapping.get("path_keys")
    stage_key = str(mapping.get("step_stage_key", "stage"))
    action_keys = mapping.get("step_action_keys")
    out = []
    skipped = {"no_question": 0, "no_path": 0, "path_len_out_of_range": 0}
    total = 0
    for idx, item in enumerate(rows):
        total = idx + 1
        q = _extract_question(item, question_keys=q_keys)
        if not q:
            skipped["no_question"] += 1
            continue
        path = _extract_oracle_path(
            item,
            path_keys=p_keys,
            step_stage_key=stage_key,
            step_action_keys=action_keys,
        )
        if not path:
            skipped["no_path"] += 1
            continue
        if not (min_path_len <= len(path) <= max_path_len):
            skipped["path_len_out_of_range"] += 1
            continue
        cst = _default_clinical_constraints(path, rng)
        trap = cst.pop("_trap_path")
        out.append(
            {
                "task_id": f"{dataset_name}_{idx:07d}",
                "dataset": dataset_name,
                "question": q,
                "query_tokens": tokenize_question(q),
                "oracle_path": path,
                "trap_path": trap,
                "candidate_paths": [path],
                "constraints": cst,
            }
        )
    stats = {
        "dataset": dataset_name,
        "total_input": total,
        "kept": len(out),
        "skipped": skipped,
    }
    return out, stats


def split_train_val_test(rows: List[Dict[str, Any]], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    shuffled = list(rows)
    rnd.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_agenthospital_like(
    data_sources: Dict[str, str],
    out_dir: str,
    seed: int,
    field_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    manifest: Dict[str, Any] = {"seed": seed, "datasets": {}, "splits": {}}
    merged = {"train": [], "val": [], "test": []}

    for ds_name, path in data_sources.items():
        raw_rows = _load_any(path)
        normalized, stats = normalize_clinical_rows(
            raw_rows,
            ds_name,
            seed=seed,
            field_mapping=field_mapping,
        )
        split = split_train_val_test(normalized, seed=seed)
        manifest["datasets"][ds_name] = stats
        for part in ("train", "val", "test"):
            out_path = Path(out_dir) / f"{ds_name}_{part}.jsonl"
            dump_jsonl(str(out_path), split[part])
            merged[part].extend(split[part])
            manifest["splits"][f"{ds_name}_{part}"] = len(split[part])

    for part in ("train", "val", "test"):
        out_path = Path(out_dir) / f"clinical_{part}.jsonl"
        dump_jsonl(str(out_path), merged[part])
        manifest["splits"][f"clinical_{part}"] = len(merged[part])

    dump_json(str(Path(out_dir) / "manifest.json"), manifest)
    return manifest


def prepare_ai_hospital_repo(
    repo_root: str,
    out_dir: str,
    seed: int,
    dialog_file: str = "dialog_history_gpt4.jsonl",
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    manifest: Dict[str, Any] = {"seed": seed, "datasets": {}, "splits": {}}

    normalized, stats = _normalize_ai_hospital_repo(
        repo_root=repo_root,
        dialog_file=dialog_file,
        seed=seed,
    )
    split = split_train_val_test(normalized, seed=seed)
    manifest["datasets"]["ai_hospital"] = stats

    for part in ("train", "val", "test"):
        out_path = Path(out_dir) / f"ai_hospital_{part}.jsonl"
        dump_jsonl(str(out_path), split[part])
        manifest["splits"][f"ai_hospital_{part}"] = len(split[part])

    for part in ("train", "val", "test"):
        out_path = Path(out_dir) / f"clinical_{part}.jsonl"
        dump_jsonl(str(out_path), split[part])
        manifest["splits"][f"clinical_{part}"] = len(split[part])

    dump_json(str(Path(out_dir) / "manifest.json"), manifest)
    return manifest
