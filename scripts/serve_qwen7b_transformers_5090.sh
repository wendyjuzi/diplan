#!/usr/bin/env bash
set -Eeuo pipefail

# Robust fallback: serve Qwen2.5-7B-Instruct with plain transformers instead of vLLM.
#
# Example:
#   bash scripts/serve_qwen7b_transformers_5090.sh

CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/miniconda3}"
ENV_NAME="${ENV_NAME:-diplan}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "[fatal] conda.sh not found under $CONDA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[fatal] model dir not found: $MODEL_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install -U transformers accelerate sentencepiece

exec python scripts/serve_openai_compat_transformers.py \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-input-tokens "$MAX_INPUT_TOKENS"
