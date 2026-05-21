from collections import Counter, defaultdict
from typing import Dict, List


def train_autoencoder(rows: List[Dict]) -> Dict:
    token_freq = Counter()
    length_hist = Counter()
    for row in rows:
        path = row["oracle_path"]
        token_freq.update(path)
        length_hist[len(path)] += 1
    total_tokens = sum(token_freq.values()) or 1
    token_prob = {k: v / total_tokens for k, v in token_freq.items()}
    total_lengths = sum(length_hist.values()) or 1
    length_prob = {str(k): v / total_lengths for k, v in length_hist.items()}
    return {
        "token_prob": token_prob,
        "length_prob": length_prob,
        "vocab_size": len(token_prob),
    }


def train_diffusion_planner(rows: List[Dict], ae_model: Dict) -> Dict:
    start_counter = Counter()
    bigram_counter = defaultdict(Counter)
    for row in rows:
        path = row["oracle_path"]
        if not path:
            continue
        start_counter[path[0]] += 1
        for i in range(len(path) - 1):
            bigram_counter[path[i]][path[i + 1]] += 1

    start_total = sum(start_counter.values()) or 1
    start_prob = {k: v / start_total for k, v in start_counter.items()}
    bigram_prob = {}
    for left, cnt in bigram_counter.items():
        total = sum(cnt.values()) or 1
        bigram_prob[left] = {k: v / total for k, v in cnt.items()}

    return {
        "start_prob": start_prob,
        "bigram_prob": bigram_prob,
        "token_prob": ae_model["token_prob"],
        "length_prob": ae_model["length_prob"],
        "noise_eps": 0.20,
    }


def train_value_model(rows: List[Dict]) -> Dict:
    assoc = defaultdict(Counter)
    trap_penalty = Counter()
    for row in rows:
        query_tokens = row["query_tokens"]
        oracle = row["oracle_path"]
        trap = row["trap_path"]
        for q in query_tokens:
            for rel in oracle:
                assoc[q][rel] += 3
            for rel in trap:
                trap_penalty[rel] += 1
    assoc_prob = {}
    for q, cnt in assoc.items():
        total = sum(cnt.values()) or 1
        assoc_prob[q] = {k: v / total for k, v in cnt.items()}
    penalty_total = sum(trap_penalty.values()) or 1
    penalty_prob = {k: v / penalty_total for k, v in trap_penalty.items()}
    return {
        "assoc_prob": assoc_prob,
        "trap_penalty": penalty_prob,
        "weights": {
            "assoc": 1.0,
            "prefix": 0.8,
            "len": 0.3,
            "trap": 0.5,
        },
    }


def train_constraint_checker(rows: List[Dict]) -> Dict:
    allowed = Counter()
    max_steps = []
    banned = Counter()
    for row in rows:
        for rel in row["oracle_path"]:
            allowed[rel] += 1
        max_steps.append(row["constraints"].get("max_steps", 6))
        for rel in row["constraints"].get("banned_relations", []):
            banned[rel] += 1
    return {
        "allowed_relations": sorted(list(allowed.keys())),
        "global_max_steps": int(sum(max_steps) / len(max_steps)) if max_steps else 6,
        "frequent_banned_relations": [x for x, _ in banned.most_common(3)],
    }

