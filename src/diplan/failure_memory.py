from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def _query_key(tokens: Iterable[str]) -> List[str]:
    return [str(t).strip().lower() for t in tokens if str(t).strip()]


class FailureMemory:
    def __init__(self, rows: List[Dict], max_postings_per_token: int = 1200) -> None:
        self.rows: List[Dict] = []
        self.token_to_ids: Dict[str, List[int]] = defaultdict(list)
        for row in rows:
            if row.get("success", False):
                continue
            executed = row.get("executed_path", [])
            oracle = row.get("oracle_path", [])
            if not isinstance(executed, list) or not executed:
                continue
            item = {
                "task_id": row.get("task_id", ""),
                "query_tokens": list(row.get("query_tokens", [])),
                "executed_path": executed,
                "oracle_path": oracle if isinstance(oracle, list) else [],
                "first_error_step": int(row.get("first_error_step", 1) or 1),
            }
            item_id = len(self.rows)
            self.rows.append(item)
            seen = set()
            for tok in _query_key(item["query_tokens"]):
                if tok in seen:
                    continue
                seen.add(tok)
                posting = self.token_to_ids[tok]
                if len(posting) < max_postings_per_token:
                    posting.append(item_id)

    def retrieve(self, query_tokens: List[str], top_k: int = 8) -> List[Dict]:
        scores = Counter()
        qset = set(_query_key(query_tokens))
        for tok in qset:
            for item_id in self.token_to_ids.get(tok, []):
                scores[item_id] += 1
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [self.rows[i] for i, _score in ranked[: max(0, top_k)]]

    @staticmethod
    def bad_first_action_weights(items: List[Dict]) -> Dict[str, float]:
        counts: Counter[str] = Counter()
        for item in items:
            path = item.get("executed_path", [])
            if isinstance(path, list) and path:
                counts[str(path[0])] += 1
        if not counts:
            return {}
        total = float(max(1, sum(counts.values())))
        return {action: count / total for action, count in counts.items()}


def build_failure_memory(rows: List[Dict], max_postings_per_token: int = 1200) -> FailureMemory:
    return FailureMemory(rows, max_postings_per_token=max_postings_per_token)
