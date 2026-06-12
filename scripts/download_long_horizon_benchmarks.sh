#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:-alfworld}"
DATA_ROOT="${DATA_ROOT:-data/long_horizon}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"
ALFWORLD_FULL="${ALFWORLD_FULL:-0}"

mkdir -p "$DATA_ROOT"

download_alfworld() {
  local target_dir="$DATA_ROOT/alfworld"
  mkdir -p "$target_dir"
  export ALFWORLD_DATA="$target_dir"

  echo "[alfworld] installing package"
  if [[ "$ALFWORLD_FULL" == "1" ]]; then
    $PIP_BIN install "alfworld[full]"
  else
    $PIP_BIN install "alfworld"
  fi

  echo "[alfworld] downloading data to $ALFWORLD_DATA"
  alfworld-download

  echo "[alfworld] smoke test"
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ALFWORLD_DATA"])
print(f"ALFWORLD_DATA={root}")
print(f"exists={root.exists()}")
print("top-level:", sorted(p.name for p in root.iterdir())[:20] if root.exists() else [])

import alfworld  # noqa: F401
print("alfworld import ok")
PY
}

download_scienceworld() {
  local target_dir="$DATA_ROOT/scienceworld"
  mkdir -p "$target_dir"

  echo "[scienceworld] checking Java"
  if ! command -v java >/dev/null 2>&1; then
    echo "[scienceworld] ERROR: java is required. Install Java 1.8+ first." >&2
    exit 1
  fi
  java -version

  echo "[scienceworld] installing package"
  $PIP_BIN install "scienceworld"

  echo "[scienceworld] smoke test"
  "$PYTHON_BIN" - <<'PY'
import scienceworld  # noqa: F401
print("scienceworld import ok")

try:
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", envStepLimit=20)
    task_names = env.getTaskNames()
    print(f"num_tasks={len(task_names)}")
    print("first_tasks=", task_names[:5])
except Exception as exc:
    print(f"scienceworld environment smoke test skipped/failed: {exc}")
PY

  cat > "$target_dir/README.txt" <<'EOF'
ScienceWorld is installed as a Python package. The benchmark tasks and simulator
are loaded through the package API rather than a separate JSONL dataset.
EOF
}

case "$BENCHMARK" in
  alfworld)
    download_alfworld
    ;;
  scienceworld)
    download_scienceworld
    ;;
  both)
    download_alfworld
    download_scienceworld
    ;;
  *)
    echo "Usage: $0 {alfworld|scienceworld|both}" >&2
    exit 2
    ;;
esac

echo "[ok] finished $BENCHMARK setup under $DATA_ROOT"
