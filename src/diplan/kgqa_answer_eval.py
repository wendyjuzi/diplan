"""Offline answer-entity evaluation for KGQA.

DiPLaN predicts a *relation path*. Standard KGQA reviewers expect
*answer-entity* metrics (Hits@1 / F1), i.e. Question -> Answer Entity.
To turn a predicted relation path into an answer set we execute it over a
per-question subgraph, then compare to the gold answers.

WebQSP ships per-question subgraphs in the EPR-KGQA dump
(``data/raw/EPR-KGQA/data/dataset/WebQSP/{test,dev,train}_simple.jsonl``),
each row carrying ``entities`` (topic mids), ``answers`` (kb_id + text) and
``subgraph.tuples`` ([head, rel, tail]). This lets us compute real Hits@1/F1
fully offline -- no Freebase server required.

RoG (``rmanluo/RoG-webqsp`` / ``rmanluo/RoG-cwq``) additionally ships a
per-question ``graph`` with answer-containing triples. Use the RoG loader when
EPR's retrieval-pruned simple subgraphs fail the oracle sanity gate.

Hits@1 follows the RoG / NSM convention for path / set methods: a question is
a hit if the predicted answer set intersects the gold answer set.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from .io_utils import load_jsonl


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")
_REL_SEP_RE = re.compile(r"[\s/]+")


def normalize_question(q: str) -> str:
    """Normalize a question for cross-source joining.

    Lowercase, drop a trailing question mark, strip all non-alphanumeric
    characters (this also removes possessive apostrophes, e.g.
    ``griffin's`` -> ``griffins``), and collapse whitespace.
    """
    if not isinstance(q, str):
        return ""
    q = q.lower().strip()
    if q.endswith("?"):
        q = q[:-1]
    q = _NON_ALNUM_RE.sub(" ", q)
    q = _WS_RE.sub(" ", q).strip()
    return q


# WebQSP simple-format splits, indexed jointly: our test split is a seed-based
# reshuffle of the official data, so we join by question text across all splits.
WEBQSP_SIMPLE_FILES = ("test_simple.jsonl", "dev_simple.jsonl", "train_simple.jsonl")
ROG_SPLIT_FILES = ("test.jsonl", "dev.jsonl", "val.jsonl", "train.jsonl")


def normalize_relation(rel: str) -> str:
    """Normalize Freebase-style relation ids across EPR/RoG dumps."""
    if not isinstance(rel, str):
        return ""
    rel = rel.strip().lower()
    rel = rel.replace("http://rdf.freebase.com/ns/", "")
    rel = rel.replace("www.freebase.com/", "")
    rel = rel.strip("/.")
    rel = _REL_SEP_RE.sub(".", rel)
    rel = re.sub(r"\.+", ".", rel).strip(".")
    return rel


def normalize_entity_key(entity: Any) -> str:
    """Normalize an entity id/name for robust local graph execution.

    Entity ids are usually Freebase mids, but RoG exports may carry plain names
    or ``{id, name}`` dictionaries. We keep punctuation meaningful for mids and
    only lowercase/trim the textual key.
    """
    if entity is None:
        return ""
    if isinstance(entity, dict):
        keys = entity_keys(entity)
        return sorted(keys)[0] if keys else ""
    text = str(entity).strip()
    text = text.replace("http://rdf.freebase.com/ns/", "")
    text = text.replace("www.freebase.com/", "")
    text = text.strip("<>")
    return text.lower()


def entity_keys(entity: Any) -> Set[str]:
    """Return all plausible id/name keys for an entity object."""
    out: Set[str] = set()
    if entity is None:
        return out
    if isinstance(entity, dict):
        for k in (
            "id",
            "mid",
            "kb_id",
            "entity_id",
            "answer_id",
            "AnswerArgument",
            "name",
            "label",
            "text",
            "answer",
        ):
            v = entity.get(k)
            if isinstance(v, (str, int, float)):
                key = normalize_entity_key(v)
                if key:
                    out.add(key)
        return out
    if isinstance(entity, (list, tuple, set)):
        for x in entity:
            out |= entity_keys(x)
        return out
    key = normalize_entity_key(entity)
    if key:
        out.add(key)
    return out


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _gold_answer_ids(answers: List[dict]) -> Set[str]:
    out: Set[str] = set()
    for a in answers or []:
        out |= entity_keys(a)
    return out


def load_webqsp_subgraphs(webqsp_dir: str) -> Dict[str, dict]:
    """Index WebQSP simple-format rows by normalized question.

    Returns ``{normalized_question: {topic_entities, gold_answers, tuples}}``.
    Later splits do not overwrite earlier ones (test takes priority).
    """
    index: Dict[str, dict] = {}
    base = Path(webqsp_dir)
    for fname in WEBQSP_SIMPLE_FILES:
        fpath = base / fname
        if not fpath.exists():
            continue
        for row in load_jsonl(str(fpath)):
            key = normalize_question(row.get("question", ""))
            if not key or key in index:
                continue
            sg = row.get("subgraph", {}) or {}
            index[key] = {
                "topic_entities": list(row.get("entities", []) or []),
                "gold_answers": _gold_answer_ids(row.get("answers", [])),
                "tuples": sg.get("tuples", []) or [],
                "source": "epr_webqsp",
            }
    return index


def _parse_graph_string(graph: str) -> List[Any]:
    """Best-effort parser for RoG graph strings.

    Most RoG exports are structured already. This fallback handles JSON strings
    and simple line-separated triples such as ``head | relation | tail``.
    """
    text = graph.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        return _as_list(obj)
    except Exception:
        pass
    triples: List[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in ("\t", " | ", "|", " , ", ","):
            parts = [p.strip() for p in line.split(sep)]
            if len(parts) == 3:
                triples.append(parts)
                break
    return triples


def _triple_from_obj(obj: Any) -> Tuple[Any, Any, Any] | None:
    if isinstance(obj, dict):
        h = obj.get("head") or obj.get("h") or obj.get("subject") or obj.get("src")
        r = obj.get("relation") or obj.get("rel") or obj.get("r") or obj.get("predicate")
        t = obj.get("tail") or obj.get("t") or obj.get("object") or obj.get("dst")
        if h is not None and r is not None and t is not None:
            return h, r, t
        if "triple" in obj:
            return _triple_from_obj(obj["triple"])
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return obj[0], obj[1], obj[2]
    return None


def parse_rog_graph(graph: Any) -> List[List[str]]:
    """Parse RoG graph payload into normalized ``[head, relation, tail]`` triples."""
    raw = _parse_graph_string(graph) if isinstance(graph, str) else _as_list(graph)
    triples: List[List[str]] = []
    for obj in raw:
        triple = _triple_from_obj(obj)
        if triple is None:
            continue
        h, r, tail = triple
        h_keys = entity_keys(h)
        t_keys = entity_keys(tail)
        rel = normalize_relation(str(r))
        if not rel:
            continue
        for hk in h_keys:
            for tk in t_keys:
                if hk and tk:
                    triples.append([hk, rel, tk])
    return triples


def _rog_topic_entities(row: Dict[str, Any]) -> List[str]:
    vals: List[Any] = []
    for key in ("q_entity", "q_entities", "question_entities", "topic_entities", "entities"):
        vals.extend(_as_list(row.get(key)))
    out = sorted(entity_keys(vals))
    return out


def _rog_gold_answers(row: Dict[str, Any]) -> Set[str]:
    vals: List[Any] = []
    for key in ("a_entity", "a_entities", "answer_entities", "answers", "answer"):
        vals.extend(_as_list(row.get(key)))
    return entity_keys(vals)


def load_rog_subgraphs(rog_path: str) -> Dict[str, dict]:
    """Index RoG rows by normalized question.

    ``rog_path`` may point to a single JSONL file or a directory containing
    ``test/dev/val/train.jsonl``. The first occurrence of a question wins, so
    place/evaluate the desired split first when passing a custom file.
    """
    base = Path(rog_path)
    files: List[Path]
    if base.is_file():
        files = [base]
    else:
        files = [base / f for f in ROG_SPLIT_FILES if (base / f).exists()]
        if not files:
            files = sorted(base.glob("*.jsonl"))
    index: Dict[str, dict] = {}
    for fpath in files:
        for row in load_jsonl(str(fpath)):
            key = normalize_question(row.get("question", "") or row.get("query", ""))
            if not key or key in index:
                continue
            tuples = parse_rog_graph(row.get("graph", []))
            index[key] = {
                "topic_entities": _rog_topic_entities(row),
                "gold_answers": _rog_gold_answers(row),
                "tuples": tuples,
                "source": "rog",
                "rog_id": row.get("id") or row.get("qid") or row.get("question_id"),
            }
    return index


def build_adjacency(
    tuples: List[List[str]], bidirectional: bool = False
) -> Dict[Tuple[str, str], Set[str]]:
    """Build a ``(head, relation) -> {tails}`` adjacency map for one subgraph.

    With ``bidirectional=True`` also add reverse edges ``(tail, relation) ->
    {heads}`` (fallback for subgraphs that store an edge in only one direction).
    """
    adj: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for t in tuples:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        h, r, tail = t
        h = normalize_entity_key(h)
        r = normalize_relation(r)
        tail = normalize_entity_key(tail)
        if not h or not r or not tail:
            continue
        adj[(h, r)].add(tail)
        if bidirectional:
            adj[(tail, r)].add(h)
    return adj


def execute_path(
    topic_entities: List[str],
    relation_path: List[str],
    adjacency: Dict[Tuple[str, str], Set[str]],
) -> Set[str]:
    """Execute a relation path from the topic entities over a subgraph.

    Hop-by-hop frontier expansion; stops early if the frontier empties. The
    topic entities themselves are removed from the final answer set (mirrors
    the WebQSP SPARQL ``FILTER (?x != topic)``).
    """
    path = [normalize_relation(r) for r in relation_path if normalize_relation(r)]
    topic_set = {normalize_entity_key(e) for e in topic_entities if normalize_entity_key(e)}
    frontier: Set[str] = set(topic_set)
    for rel in path:
        nxt: Set[str] = set()
        for h in frontier:
            nxt |= adjacency.get((h, rel), set())
        frontier = nxt
        if not frontier:
            break
    return frontier - topic_set


def answer_metrics(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    """RoG-style any-hit Hits@1 plus set-based precision/recall/F1."""
    pred = {normalize_entity_key(x) for x in pred if normalize_entity_key(x)}
    gold = {normalize_entity_key(x) for x in gold if normalize_entity_key(x)}
    inter = pred & gold
    n_inter = len(inter)
    precision = n_inter / len(pred) if pred else 0.0
    recall = n_inter / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "hits1": 1.0 if n_inter > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pred": float(len(pred)),
        "num_gold": float(len(gold)),
    }


def graph_sanity_stats(index: Dict[str, dict], max_hops: int = 3) -> Dict[str, float]:
    """Quick source-quality gate for answer-level evaluation graphs."""
    n = len(index)
    if n == 0:
        return {
            "num_graphs": 0.0,
            "answer_entity_in_graph_rate": 0.0,
            "topic_entity_in_graph_rate": 0.0,
            "q_to_answer_reachable_rate": 0.0,
        }
    answer_in_graph = 0
    topic_in_graph = 0
    reachable = 0
    for sg in index.values():
        tuples = sg.get("tuples", []) or []
        ents = {t[0] for t in tuples if len(t) == 3} | {t[2] for t in tuples if len(t) == 3}
        gold = {normalize_entity_key(x) for x in sg.get("gold_answers", set())}
        topics = {normalize_entity_key(x) for x in sg.get("topic_entities", [])}
        if gold & ents:
            answer_in_graph += 1
        if topics & ents:
            topic_in_graph += 1
        adj = build_adjacency(tuples, bidirectional=True)
        frontier = set(topics)
        seen = set(frontier)
        hit = bool(frontier & gold)
        for _ in range(max_hops):
            nxt: Set[str] = set()
            for h in frontier:
                for (src, _rel), tails in adj.items():
                    if src == h:
                        nxt |= tails
            frontier = nxt - seen
            seen |= frontier
            if frontier & gold:
                hit = True
                break
            if not frontier:
                break
        if hit:
            reachable += 1
    return {
        "num_graphs": float(n),
        "answer_entity_in_graph_rate": answer_in_graph / n,
        "topic_entity_in_graph_rate": topic_in_graph / n,
        "q_to_answer_reachable_rate": reachable / n,
    }
