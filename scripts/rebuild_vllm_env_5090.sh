#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild and diagnose a clean vLLM environment on the GPU box.
#
# Typical usage on AutoDL / Linux:
#   bash scripts/rebuild_vllm_env_5090.sh create
#   bash scripts/rebuild_vllm_env_5090.sh diagnose
#   bash scripts/rebuild_vllm_env_5090.sh serve
#
# Or do everything in one go:
#   bash scripts/rebuild_vllm_env_5090.sh all

ACTION="${1:-help}"

CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/miniconda3}"
ENV_NAME="${ENV_NAME:-vllm_compat}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch torchvision torchaudio}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Qwen2.5-7B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-4096}"
DTYPE="${DTYPE:-half}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/DiPLaN/logs}"

function _info() {
  echo "[info] $*"
}

function _warn() {
  echo "[warn] $*" >&2
}

function _die() {
  echo "[fatal] $*" >&2
  exit 1
}

function _check_gpu_box() {
  command -v nvidia-smi >/dev/null 2>&1 || _die "nvidia-smi not found. Run this on the GPU machine."
}

function _load_conda() {
  [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || _die "conda.sh not found under $CONDA_ROOT"
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
}

function _activate_env() {
  _load_conda
  conda activate "$ENV_NAME"
}

function _create_env() {
  _check_gpu_box
  _load_conda

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    _info "conda env '$ENV_NAME' already exists; keeping it as-is"
  else
    _info "creating conda env '$ENV_NAME' with python=$PYTHON_VERSION"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
  fi

  conda activate "$ENV_NAME"
  python -m pip install --upgrade pip setuptools wheel

  _info "installing torch from $TORCH_INDEX_URL"
  python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX_URL"

  _info "installing $VLLM_SPEC"
  python -m pip install -U $VLLM_SPEC

  _info "environment ready: $ENV_NAME"
}

function _diagnose_env() {
  _check_gpu_box
  _activate_env
  unset VLLM_USE_V1 || true

  _info "=== binaries ==="
  which python || true
  which pip || true
  which vllm || true

  _info "=== nvidia-smi ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv || true

  _info "=== python package sanity ==="
  python - <<'PY'
import importlib.util
import os
import shutil
import subprocess
import sys

def maybe_mod(name: str):
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin

print("python:", sys.executable)
print("pip:", shutil.which("pip"))
print("vllm:", shutil.which("vllm"))

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda:", torch.version.cuda)
    print("cuda.available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu.name:", torch.cuda.get_device_name(0))
        print("gpu.capability:", torch.cuda.get_device_capability(0))
except Exception as exc:  # noqa: BLE001
    print("torch_import_error:", repr(exc))

try:
    import vllm
    print("vllm.version:", getattr(vllm, "__version__", "unknown"))
    print("vllm.file:", getattr(vllm, "__file__", "unknown"))
except Exception as exc:  # noqa: BLE001
    print("vllm_import_error:", repr(exc))

print("flashinfer.module:", maybe_mod("flashinfer"))
print("VLLM_USE_V1:", os.environ.get("VLLM_USE_V1"))

for cmd in (["nvcc", "--version"], ["python", "-m", "pip", "show", "vllm"], ["python", "-m", "pip", "show", "flashinfer"]):
    try:
        print("\\n$ " + " ".join(cmd))
        subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001
        print("command_error:", repr(exc))
PY
}

function _serve_model() {
  _check_gpu_box
  _activate_env
  unset VLLM_USE_V1 || true

  [[ -d "$MODEL_PATH" ]] || _die "model dir not found: $MODEL_PATH"
  mkdir -p "$LOG_DIR"

  _info "starting vLLM with absolute binary path"
  _info "log file: $LOG_DIR/vllm_${ENV_NAME}_${SERVED_NAME}.log"
  _info "if this still fails, inspect the log for flashinfer / cuda capability messages"

  nohup bash -lc "
source '$CONDA_ROOT/etc/profile.d/conda.sh'
conda activate '$ENV_NAME'
unset VLLM_USE_V1 || true
'$CONDA_ROOT/envs/$ENV_NAME/bin/vllm' serve '$MODEL_PATH' \
  --served-model-name '$SERVED_NAME' \
  --host '$HOST' \
  --port '$PORT' \
  --gpu-memory-utilization '$GPU_UTIL' \
  --max-model-len '$MAX_LEN' \
  --dtype '$DTYPE' \
  --enforce-eager
" > "$LOG_DIR/vllm_${ENV_NAME}_${SERVED_NAME}.log" 2>&1 &

  _info "tail with:"
  echo "tail -f $LOG_DIR/vllm_${ENV_NAME}_${SERVED_NAME}.log"
}

case "$ACTION" in
  create)
    _create_env
    ;;
  diagnose)
    _diagnose_env
    ;;
  serve)
    _serve_model
    ;;
  all)
    _create_env
    _diagnose_env
    _serve_model
    ;;
  help|-h|--help)
    cat <<EOF
Usage: bash scripts/rebuild_vllm_env_5090.sh <action>

Actions:
  create    Create a fresh conda env and install torch + vllm
  diagnose  Print binary, CUDA, torch, vllm, and flashinfer diagnostics
  serve     Start an OpenAI-compatible vLLM server with absolute paths
  all       Run create -> diagnose -> serve

Common overrides:
  CONDA_ROOT=/root/autodl-tmp/miniconda3
  ENV_NAME=vllm_compat
  PYTHON_VERSION=3.11
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
  VLLM_SPEC='vllm'
  MODEL_PATH=/root/autodl-tmp/Qwen2.5-7B-Instruct
  SERVED_NAME=Qwen2.5-7B-Instruct
  PORT=8000
EOF
    ;;
  *)
    _die "unknown action: $ACTION"
    ;;
esac
