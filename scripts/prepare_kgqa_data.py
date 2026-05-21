import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.data import make_dataset, split_train_val_test
from src.diplan.io_utils import dump_json, dump_jsonl, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cwq", "webqsp", "grailqa"])
    parser.add_argument("--n-per-dataset", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/processed")
    args = parser.parse_args()

    ensure_dir(args.out)
    generated = make_dataset(args.datasets, args.n_per_dataset, args.seed)

    manifest = {
        "datasets": args.datasets,
        "n_per_dataset": args.n_per_dataset,
        "seed": args.seed,
        "splits": {},
    }

    merged = {"train": [], "val": [], "test": []}
    for ds in args.datasets:
        split = split_train_val_test(generated[ds], args.seed)
        for part in ("train", "val", "test"):
            path = Path(args.out) / f"{ds}_{part}.jsonl"
            dump_jsonl(str(path), split[part])
            merged[part].extend(split[part])
            manifest["splits"][f"{ds}_{part}"] = len(split[part])

    for part in ("train", "val", "test"):
        path = Path(args.out) / f"kgqa_{part}.jsonl"
        dump_jsonl(str(path), merged[part])
        manifest["splits"][f"kgqa_{part}"] = len(merged[part])

    dump_json(str(Path(args.out) / "manifest.json"), manifest)
    print(f"Prepared data in {args.out}")
    print(manifest)


if __name__ == "__main__":
    main()
