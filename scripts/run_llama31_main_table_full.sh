#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot runner:
# 1) optionally use HF_TOKEN from the environment
# 2) download Meta-Llama-3.1-8B-Instruct if it is not already present
# 3) start the local OpenAI-compatible transformers server
# 4) wait until the server is healthy
# 5) run the official-ToG main table (full)
# 6) summarize and print the markdown table path
#
# Usage:
#   export HF_TOKEN=...
#   bash scripts/run_llama31_main_table_full.sh

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/DiPLaN}"
CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/miniconda3}"
ENV_NAME="${ENV_NAME:-diplan}"
MODEL_REPO="${MODEL_REPO:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
SERVER_PORT="${SERVER_PORT:-8001}"
SERVER_LOG="${SERVER_LOG:-$ROOT_DIR/logs/llama31_transformers_server.log}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/main_table_official_tog_llama31_full}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/main_table_official_tog_llama31_full}"
HF_DISABLE_XET="${HF_DISABLE_XET:-1}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
DTYPE="${DTYPE:-auto}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

function _info() {
  echo "[info] $*"
}

function _die() {
  echo "[fatal] $*" >&2
  exit 1
}

function _load_conda() {
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
}

function _activate() {
  _load_conda
  conda activate "$ENV_NAME"
}

function _check_prereqs() {
  [[ -d "$ROOT_DIR" ]] || _die "ROOT_DIR not found: $ROOT_DIR"
  [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || _die "conda.sh not found under $CONDA_ROOT"
}

function _download_model() {
  if [[ "$FORCE_DOWNLOAD" != "1" && -f "$MODEL_DIR/config.json" ]]; then
    _info "model already present at $MODEL_DIR; skipping download"
    return
  fi

  mkdir -p "$(dirname "$MODEL_DIR")"
  export HF_HUB_DISABLE_XET="$HF_DISABLE_XET"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    _info "using HF_TOKEN from environment for gated model download"
  else
    _info "HF_TOKEN not set; download will rely on existing local Hugging Face credentials"
  fi
  _info "downloading model to $MODEL_DIR"
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  [[ -f "$MODEL_DIR/config.json" ]] || _die "download finished but config.json is missing under $MODEL_DIR"
}

function _start_server() {
  mkdir -p "$(dirname "$SERVER_LOG")"
  _info "starting transformers server on :$SERVER_PORT"
  nohup python "$ROOT_DIR/scripts/serve_openai_compat_transformers.py" \
    --model-path "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --dtype "$DTYPE" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    > "$SERVER_LOG" 2>&1 &
  echo $! > "$ROOT_DIR/logs/llama31_transformers_server.pid"
}

function _wait_server() {
  _info "waiting for server readiness"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
      _info "server is ready"
      return 0
    fi
    sleep 5
  done
  if [[ -f "$SERVER_LOG" ]]; then
    echo "[error] last server log lines:" >&2
    tail -n 40 "$SERVER_LOG" >&2 || true
  fi
  _die "server did not become ready; inspect $SERVER_LOG"
}

function _run_table() {
  export TOG_OPENAI_API_BASE="http://127.0.0.1:${SERVER_PORT}/v1"
  export TOG_OPENAI_API_KEY="EMPTY"
  export TOG_OPENAI_MODEL="$MODEL_NAME"

  _info "running main table full experiment"
  cd "$ROOT_DIR"
  MODEL_NAME="$MODEL_NAME" \
  OPENAI_API_BASE="$TOG_OPENAI_API_BASE" \
  OPENAI_API_KEY="$TOG_OPENAI_API_KEY" \
  OUT_ROOT="$OUT_ROOT" \
  LOG_DIR="$LOG_DIR" \
  MAX_TASKS=0 \
  bash scripts/run_main_table_official_tog.sh all
}

function _print_result() {
  local md_path="$OUT_ROOT/main_table.md"
  _info "main table markdown: $md_path"
  if [[ -f "$md_path" ]]; then
    echo
    cat "$md_path"
  fi
}

_check_prereqs
_activate
_download_model
_start_server
_wait_server
_run_table
_print_result
