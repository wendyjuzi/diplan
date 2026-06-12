"""Download RoG KGQA graphs from HuggingFace and export local JSONL files.

The exported files are consumed by ``scripts/evaluate_kgqa_answers.py`` with
``--graph_source rog``. Keeping this as a tiny script makes the answer-level
evaluation reproducible without relying on an interactive notebook.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diplan.io_utils import ensure_dir


DATASETS: Dict[str, str] = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq": "rmanluo/RoG-cwq",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["webqsp"],
        choices=sorted(DATASETS),
        help="RoG datasets to download/export.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to export. Use 'all' to export every split returned by HuggingFace.",
    )
    parser.add_argument("--out_root", type=str, default="data/rog")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load only from the local HuggingFace cache.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Skip a dataset/split that cannot be loaded instead of aborting the whole export.",
    )
    args = parser.parse_args()

    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise SystemExit("Please install HuggingFace datasets first: pip install -U datasets pyarrow") from exc

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    download_config = DownloadConfig(
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )

    out_root = Path(args.out_root)
    ensure_dir(str(out_root))
    manifest = {}
    for name in args.datasets:
        hf_name = DATASETS[name]
        print(f"[rog] loading {hf_name}")
        out_dir = out_root / name
        ensure_dir(str(out_dir))
        split_counts = {}
        try:
            if len(args.splits) == 1 and args.splits[0].lower() == "all":
                split_map = load_dataset(
                    hf_name,
                    cache_dir=args.cache_dir,
                    download_config=download_config,
                )
            else:
                split_map = {}
                for split in args.splits:
                    try:
                        split_map[split] = load_dataset(
                            hf_name,
                            split=split,
                            cache_dir=args.cache_dir,
                            download_config=download_config,
                        )
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        print(f"[warn] skip {hf_name} split={split}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[warn] skip dataset {hf_name}: {type(exc).__name__}: {exc}")
            manifest[name] = {"hf_dataset": hf_name, "error": repr(exc), "splits": split_counts}
            continue
        for split, split_ds in split_map.items():
            out_path = out_dir / f"{split}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for row in split_ds:
                    f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            split_counts[split] = len(split_ds)
            print(f"[rog] wrote {out_path} n={len(split_ds)}")
        manifest[name] = {"hf_dataset": hf_name, "splits": split_counts}

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
