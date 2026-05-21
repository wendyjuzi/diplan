import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.real_data import prepare_real_kgqa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwq", type=str, default="")
    parser.add_argument("--webqsp", type=str, default="")
    parser.add_argument("--grailqa", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/real_processed")
    args = parser.parse_args()

    sources = {}
    if args.cwq:
        sources["cwq"] = args.cwq
    if args.webqsp:
        sources["webqsp"] = args.webqsp
    if args.grailqa:
        sources["grailqa"] = args.grailqa

    if not sources:
        raise ValueError("Please provide at least one source: --cwq / --webqsp / --grailqa")

    manifest = prepare_real_kgqa(
        data_sources=sources,
        out_dir=args.out,
        seed=args.seed,
    )
    print(f"Prepared real KGQA data at {args.out}")
    print(manifest)


if __name__ == "__main__":
    main()

