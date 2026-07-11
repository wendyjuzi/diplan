#!/usr/bin/env bash
set -Eeuo pipefail

# Serve Qwen2.5-7B-Instruct with an OpenAI-compatible vLLM server on the 5090 box.
# Run this ON the GPU machine (AutoDL / 5090), NOT on your laptop:
#   bash scripts/serve_qwen7b_5090.sh
# Then start your eval in a SECOND terminal (see bottom of this file).

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"          # 7B leaves plenty of headroom on 32GB
MAX_LEN="${MAX_LEN:-8192}"            # raise to 16384 if you need longer prompts
DTYPE="${DTYPE:-auto}"               # bf16 on Blackwell; set to bfloat16 if auto misbehaves

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || {
  echo "[fatal] nvidia-smi not found — are you on the GPU box?"; exit 1; }

if ! command -v vllm >/dev/null 2>&1; then
  echo "[fatal] vllm not installed. Install a Blackwell(sm_120)-compatible build, e.g.:"
  echo "        pip install -U 'vllm>=0.8.0'   # needs CUDA 12.8 / cu128 torch wheels for RTX 5090"
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[fatal] model dir not found: $MODEL_PATH"
  echo "        set MODEL_PATH=... or download the weights first."
  exit 1
fi

echo "=== serving $SERVED_NAME from $MODEL_PATH on :$PORT ==="
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --dtype "$DTYPE"

# ---------------------------------------------------------------------------
# In a SECOND terminal on the same box, point the eval at this server:
#
#   export TOG_OPENAI_API_BASE=http://127.0.0.1:8000/v1
#   export TOG_OPENAI_API_KEY=EMPTY
#   export TOG_OPENAI_MODEL=Qwen2.5-7B-Instruct   # must match --served-model-name
#
# then run your patched-ToG command as usual (with --relation_first_k 20).
# For scripts/run_tog_subgraph_planning_eval.py instead, set the config field
#   "llm_model": "Qwen2.5-7B-Instruct"
# ---------------------------------------------------------------------------
