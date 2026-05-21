import random
from collections import defaultdict
from typing import Dict, List


RELATION_POOLS: Dict[str, List[str]] = {
    "cwq": [
        "located_in",
        "part_of",
        "member_of",
        "has_capital",
        "founded_by",
        "language_spoken",
        "neighbor_of",
        "has_industry",
    ],
    "webqsp": [
        "born_in",
        "educated_at",
        "award_received",
        "acted_in",
        "directed",
        "nationality",
        "genre",
        "published_in",
    ],
    "grailqa": [
        "instance_of",
        "subclass_of",
        "country_of_origin",
        "operating_system",
        "developer",
        "distribution_platform",
        "uses_language",
        "headquarters",
    ],
}


def _mutate_path(path: List[str], pool: List[str], rng: random.Random) -> List[str]:
    out = list(path)
    mutation_type = rng.choice(["replace", "swap", "drop", "insert"])
    if mutation_type == "replace" and out:
        i = rng.randrange(len(out))
        out[i] = rng.choice(pool)
    elif mutation_type == "swap" and len(out) > 1:
        i, j = sorted(rng.sample(range(len(out)), 2))
        out[i], out[j] = out[j], out[i]
    elif mutation_type == "drop" and len(out) > 2:
        out.pop(rng.randrange(len(out)))
    elif mutation_type == "insert" and len(out) < 6:
        out.insert(rng.randrange(len(out) + 1), rng.choice(pool))
    return out


def _build_query_from_oracle(path: List[str], rng: random.Random) -> List[str]:
    tokens = list(path)
    if rng.random() < 0.35 and tokens:
        idx = rng.randrange(len(tokens))
        tokens[idx] = tokens[idx] + "_hint"
    if rng.random() < 0.20 and len(tokens) > 2:
        i, j = sorted(rng.sample(range(len(tokens)), 2))
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return tokens


def _generate_sample(dataset: str, idx: int, rng: random.Random) -> Dict:
    pool = RELATION_POOLS[dataset]
    path_len = rng.randint(3, 5)
    oracle_path = [rng.choice(pool) for _ in range(path_len)]
    query_tokens = _build_query_from_oracle(oracle_path, rng)
    trap_token = pool[0]
    trap_path = [trap_token] + [rng.choice(pool) for _ in range(path_len - 1)]
    banned_relation = rng.choice([r for r in pool if r not in oracle_path] or [pool[-1]])

    decoys: List[List[str]] = []
    for _ in range(6):
        decoy = _mutate_path(oracle_path, pool, rng)
        if decoy == oracle_path:
            decoy = _mutate_path(oracle_path, pool, rng)
        decoys.append(decoy)
    decoys.append(trap_path)

    return {
        "task_id": f"{dataset}_{idx:06d}",
        "dataset": dataset,
        "question": f"find relation chain for {dataset} task {idx}",
        "query_tokens": query_tokens,
        "oracle_path": oracle_path,
        "trap_path": trap_path,
        "candidate_paths": [oracle_path] + decoys,
        "constraints": {
            "max_steps": 6,
            "banned_relations": [banned_relation],
        },
    }


def make_dataset(
    datasets: List[str],
    n_per_dataset: int,
    seed: int,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    all_rows: List[Dict] = []
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for ds in datasets:
        if ds not in RELATION_POOLS:
            raise ValueError(f"Unsupported dataset: {ds}")
        for i in range(n_per_dataset):
            row = _generate_sample(ds, i, rng)
            all_rows.append(row)
            by_dataset[ds].append(row)
    return {"all": all_rows, **by_dataset}


def split_train_val_test(rows: List[Dict], seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}

