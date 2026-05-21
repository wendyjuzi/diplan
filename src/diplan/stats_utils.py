import math
import random
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def bootstrap_mean_diff(a: List[float], b: List[float], n_resamples: int = 10000, seed: int = 42) -> Dict:
    if not a or not b:
        return {"mean_diff": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_resamples):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(mean(sa) - mean(sb))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    return {"mean_diff": mean(a) - mean(b), "ci95": [lo, hi]}


def mcnemar_test_paired(binary_a: Iterable[int], binary_b: Iterable[int]) -> Dict:
    b = 0
    c = 0
    for xa, xb in zip(binary_a, binary_b):
        if xa == 1 and xb == 0:
            b += 1
        elif xa == 0 and xb == 1:
            c += 1
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_approx": 1.0}
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p_approx": p}


def cliffs_delta(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    gt = 0
    lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def holm_bonferroni(p_values: List[Tuple[str, float]]) -> List[Dict]:
    sorted_p = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_p)
    out: List[Dict] = []
    for i, (name, p) in enumerate(sorted_p):
        threshold = 0.05 / (m - i)
        out.append(
            {
                "comparison": name,
                "p_value": p,
                "holm_threshold": threshold,
                "reject_h0": p < threshold,
            }
        )
    return out

