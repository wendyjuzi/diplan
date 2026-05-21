import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .io_utils import dump_json, dump_jsonl, ensure_dir


REL_RE = re.compile(r"ns:([A-Za-z0-9_\.]+)")
WORD_RE = re.compile(r"[A-Za-z0-9_\.]+")


def _load_any(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "examples", "items", "questions"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Unsupported JSON structure: {path}")


def _is_relation_like(token: str) -> bool:
    if not token:
        return False
    token = token.strip()
    if token.startswith("m.") or token.startswith("g."):
        return False
    if token.count(".") == 0 and "/" not in token and "_" not in token:
        return False
    return True


def _extract_from_graph_query(item: Dict[str, Any]) -> List[str]:
    gq = item.get("graph_query")
    if not isinstance(gq, dict):
        return []
    edges = gq.get("edges")
    if not isinstance(edges, list):
        return []
    rels = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        rel = e.get("relation") or e.get("rel") or e.get("predicate")
        if isinstance(rel, str) and _is_relation_like(rel):
            rels.append(rel)
    return rels


def _extract_from_relation_fields(item: Dict[str, Any]) -> List[str]:
    candidates = [
        item.get("oracle_path"),
        item.get("relation_path"),
        item.get("path"),
        item.get("relations"),
        item.get("gold_relations"),
    ]
    for c in candidates:
        if isinstance(c, list) and c and all(isinstance(x, str) for x in c):
            rels = [x.strip() for x in c if _is_relation_like(x.strip())]
            if rels:
                return rels
    return []


def _extract_from_sparql_or_sexpr(item: Dict[str, Any]) -> List[str]:
    text_parts = []
    for key in ("sparql_query", "sparql", "query", "s_expression", "sexpr"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            text_parts.append(v)
    if not text_parts:
        return []
    merged = "\n".join(text_parts)
    rels = [x for x in REL_RE.findall(merged) if _is_relation_like(x)]
    if rels:
        return rels
    fallback = [x for x in WORD_RE.findall(merged) if _is_relation_like(x)]
    return fallback


def _extract_from_webqsp_parses(item: Dict[str, Any]) -> List[str]:
    parses = item.get("Parses")
    if not isinstance(parses, list):
        return []
    best = None
    for p in parses:
        if not isinstance(p, dict):
            continue
        if p.get("InferentialChain"):
            best = p
            break
    if best is None:
        for p in parses:
            if isinstance(p, dict) and p.get("Sparql"):
                best = p
                break
    if best is None:
        return []

    chain = best.get("InferentialChain")
    if isinstance(chain, list):
        rels = [x.strip() for x in chain if isinstance(x, str) and _is_relation_like(x.strip())]
        if rels:
            return rels

    sparql = best.get("Sparql")
    if isinstance(sparql, str) and sparql.strip():
        rels = [x for x in REL_RE.findall(sparql) if _is_relation_like(x)]
        if rels:
            return rels
    return []


def extract_oracle_path(item: Dict[str, Any]) -> List[str]:
    rels = _extract_from_webqsp_parses(item)
    if rels:
        return rels
    rels = _extract_from_relation_fields(item)
    if rels:
        return rels
    rels = _extract_from_graph_query(item)
    if rels:
        return rels
    return _extract_from_sparql_or_sexpr(item)


def extract_question(item: Dict[str, Any]) -> str:
    for key in ("question", "raw_question", "RawQuestion", "ProcessedQuestion", "utterance", "query_text", "text"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def tokenize_question(question: str) -> List[str]:
    tokens = [t.lower() for t in WORD_RE.findall(question.lower()) if t]
    return tokens[:24]


def _build_constraints(oracle_path: List[str], all_relations: List[str], rng: random.Random) -> Dict[str, Any]:
    banned_candidates = [r for r in all_relations if r not in oracle_path]
    banned = [rng.choice(banned_candidates)] if banned_candidates else []
    return {
        "max_steps": min(8, max(3, len(oracle_path) + 1)),
        "banned_relations": banned,
    }


def normalize_dataset_rows(
    rows: Iterable[Dict[str, Any]],
    dataset_name: str,
    rng: random.Random,
    min_path_len: int = 2,
    max_path_len: int = 8,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    extracted = []
    all_relations = set()
    skipped = {"no_question": 0, "no_path": 0, "path_len_out_of_range": 0}
    for idx, item in enumerate(rows):
        q = extract_question(item)
        if not q:
            skipped["no_question"] += 1
            continue
        path = extract_oracle_path(item)
        if not path:
            skipped["no_path"] += 1
            continue
        if not (min_path_len <= len(path) <= max_path_len):
            skipped["path_len_out_of_range"] += 1
            continue
        all_relations.update(path)
        extracted.append(
            {
                "task_id": f"{dataset_name}_{idx:07d}",
                "dataset": dataset_name,
                "question": q,
                "query_tokens": tokenize_question(q),
                "oracle_path": path,
            }
        )

    all_rel_list = sorted(list(all_relations))
    for row in extracted:
        trap = list(row["oracle_path"])
        if trap:
            trap[0] = all_rel_list[0] if all_rel_list else trap[0]
        row["trap_path"] = trap
        row["candidate_paths"] = [row["oracle_path"]]
        row["constraints"] = _build_constraints(row["oracle_path"], all_rel_list, rng)

    stats = {
        "dataset": dataset_name,
        "total_input": idx + 1 if "idx" in locals() else 0,
        "kept": len(extracted),
        "skipped": skipped,
        "unique_relations": len(all_rel_list),
    }
    return extracted, stats


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


def prepare_real_kgqa(
    data_sources: Dict[str, str],
    out_dir: str,
    seed: int,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    rng = random.Random(seed)
    manifest: Dict[str, Any] = {"seed": seed, "datasets": {}, "splits": {}}

    merged = {"train": [], "val": [], "test": []}
    for ds_name, path in data_sources.items():
        rows = _load_any(path)
        normalized, stats = normalize_dataset_rows(rows, ds_name, rng)
        split = split_train_val_test(normalized, seed)
        manifest["datasets"][ds_name] = stats
        for part in ("train", "val", "test"):
            out = Path(out_dir) / f"{ds_name}_{part}.jsonl"
            dump_jsonl(str(out), split[part])
            merged[part].extend(split[part])
            manifest["splits"][f"{ds_name}_{part}"] = len(split[part])

    for part in ("train", "val", "test"):
        out = Path(out_dir) / f"kgqa_{part}.jsonl"
        dump_jsonl(str(out), merged[part])
        manifest["splits"][f"kgqa_{part}"] = len(merged[part])
    dump_json(str(Path(out_dir) / "manifest.json"), manifest)
    return manifest
