#!/usr/bin/env bash
set -Eeuo pipefail

# Unified OpenAI-compatible serving entry for DiPLaN experiments.
#
# Two backends are supported:
#   1. vllm         : fastest, preferred for KGQA / patched-ToG runs
#   2. transformers : slower fallback, but robust when vLLM is unavailable
#
# Examples:
#   BACKEND=vllm \
#   MODEL_PATH=/root/autodl-tmp/Qwen3.5-35B-A3B-FP8 \
#   SERVED_NAME=Qwen3.5-35B-A3B-FP8 \
#   PORT=8000 \
#   bash scripts/serve_longhorizon_pseudo_api.sh
#
#   BACKEND=transformers \
#   MODEL_PATH=/root/autodl-tmp/Qwen2.5-7B-Instruct \
#   SERVED_NAME=Qwen2.5-7B-Instruct \
#   PORT=8000 \
#   bash scripts/serve_longhorizon_pseudo_api.sh
#
# Then in a second shell:
#   export TOG_OPENAI_API_BASE=http://127.0.0.1:${PORT}/v1
#   export TOG_OPENAI_API_KEY=EMPTY
#   export TOG_OPENAI_MODEL=${SERVED_NAME}

BACKEND="${BACKEND:-vllm}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-8192}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== pseudo api bootstrap ==="
echo "backend=$BACKEND"
echo "model_path=$MODEL_PATH"
echo "served_name=$SERVED_NAME"
echo "host=$HOST port=$PORT"
echo "dtype=$DTYPE max_len=$MAX_LEN"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[fatal] model dir not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$BACKEND" == "vllm" ]]; then
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[fatal] vllm not found. Switch BACKEND=transformers or install vllm." >&2
    exit 1
  fi

  exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --dtype "$DTYPE"
fi

if [[ "$BACKEND" == "transformers" ]]; then
  cmd=(
    "$PYTHON_BIN" scripts/serve_openai_compat_transformers.py
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --max-input-tokens "$MAX_LEN"
  )
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi
  exec "${cmd[@]}"
fi

echo "[fatal] unsupported BACKEND: $BACKEND" >&2
echo "valid values: vllm | transformers" >&2
exit 2
