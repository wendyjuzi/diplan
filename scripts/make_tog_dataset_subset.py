"""Create a small official-ToG dataset split for smoke reproduction runs.

The official ToG runner selects data by dataset name and reads fixed files such as
``data/WebQSP.json``. For strict reproduction smoke tests, it is safer to create a
small sibling file and temporarily swap it in than to patch ToG's algorithmic code.

Example on the server:

    python scripts/make_tog_dataset_subset.py \
      --tog_dir /root/autodl-tmp/paper_baselines/ToG \
      --dataset webqsp \
      --n 20 \
      --out_name WebQSP.smoke20.json

Then run ToG by temporarily replacing ``data/WebQSP.json`` with the subset, or by
using the printed copy commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_FILES = {
    "cwq": "cwq.json",
    "webqsp": "WebQSP.json",
    "grailqa": "grailqa.json",
    "simpleqa": "SimpleQA.json",
    "qald": "qald_10-en.json",
    "webquestions": "WebQuestions.json",
    "trex": "T-REX.json",
    "zeroshotre": "Zero_Shot_RE.json",
    "creak": "creak.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tog_dir", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out_name", default="")
    args = parser.parse_args()

    tog_dir = Path(args.tog_dir).expanduser().resolve()
    src_name = DATASET_FILES[args.dataset]
    src = tog_dir / "data" / src_name
    if not src.exists():
        raise FileNotFoundError(f"Missing official ToG dataset file: {src}")

    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {src}, got {type(rows).__name__}")
    subset = rows[args.offset : args.offset + args.n]
    out_name = args.out_name or f"{src.stem}.smoke{args.n}.json"
    out = src.with_name(out_name)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    backup = src.with_suffix(src.suffix + ".bak_full")
    print(json.dumps({
        "source": str(src),
        "output": str(out),
        "rows_total": len(rows),
        "rows_subset": len(subset),
        "backup_path": str(backup),
        "use_subset_commands": [
            f"cp {src} {backup}",
            f"cp {out} {src}",
            "# run official ToG here",
            f"mv {backup} {src}",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
