import argparse
from collections import defaultdict
from pathlib import Path

from src.diplan.io_utils import dump_json, ensure_dir, load_json, load_jsonl
from src.diplan.stats_utils import bootstrap_mean_diff, cliffs_delta, holm_bonferroni, mcnemar_test_paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", type=str, default="results/main")
    parser.add_argument("--out", type=str, default="results/stats")
    parser.add_argument("--ref", type=str, default="diplan")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preds = load_jsonl(str(Path(args.in_dir) / "predictions.jsonl"))
    summary = load_json(str(Path(args.in_dir) / "summary_metrics.json"))
    by_method = defaultdict(list)
    for r in preds:
        by_method[r["method"]].append(r)

    ref_method = args.ref
    if ref_method not in by_method:
        raise ValueError(f"Reference method not found: {ref_method}")

    stats = {"reference_method": ref_method, "comparisons": {}}
    pvals = []
    ref_recs = by_method[ref_method]
    ref_success = [1 if r["success"] else 0 for r in ref_recs]
    ref_fes = [r["first_error_step"] for r in ref_recs]

    for method, recs in by_method.items():
        if method == ref_method:
            continue
        m_success = [1 if r["success"] else 0 for r in recs]
        m_fes = [r["first_error_step"] for r in recs]

        mcnemar = mcnemar_test_paired(ref_success, m_success)
        boot_success = bootstrap_mean_diff(ref_success, m_success, n_resamples=args.bootstrap, seed=args.seed)
        boot_fes = bootstrap_mean_diff(ref_fes, m_fes, n_resamples=args.bootstrap, seed=args.seed + 1)
        delta_fes = cliffs_delta(ref_fes, m_fes)

        pvals.append((f"{ref_method} vs {method}", float(mcnemar["p_approx"])))
        stats["comparisons"][method] = {
            "summary_ref": summary[ref_method],
            "summary_other": summary[method],
            "success_mcnemar": mcnemar,
            "success_bootstrap": boot_success,
            "first_error_bootstrap": boot_fes,
            "first_error_cliffs_delta": delta_fes,
        }

    stats["holm_bonferroni"] = holm_bonferroni(pvals)
    ensure_dir(args.out)
    dump_json(str(Path(args.out) / "stats_report.json"), stats)
    print(f"Stats report written to {args.out}")
    for name, p in pvals:
        print(f"{name:>30} p={p:.4f}")


if __name__ == "__main__":
    main()

