#!/usr/bin/env bash
set -Eeuo pipefail

# One-command runner for:
#   ALFWorld     {ReAct, CoT, DiPLaN}
#   ScienceWorld {ReAct, CoT, DiPLaN}
# plus a unified comparison table.
#
# Typical usage:
#   export TOG_OPENAI_API_BASE=http://127.0.0.1:8000/v1
#   export TOG_OPENAI_API_KEY=EMPTY
#   export TOG_OPENAI_MODEL=Llama-3.1-8B-Instruct
#   bash scripts/run_react_cot_diplan_longhorizon.sh
#
# Optional: let this script start the local transformers server too:
#   START_SERVER=1 \
#   MODEL_DIR=/root/autodl-tmp/Meta-Llama-3.1-8B-Instruct \
#   bash scripts/run_react_cot_diplan_longhorizon.sh

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/DiPLaN}"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_SERVER="${START_SERVER:-0}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/Meta-Llama-3.1-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_DTYPE="${SERVER_DTYPE:-half}"
SERVER_MAX_INPUT_TOKENS="${SERVER_MAX_INPUT_TOKENS:-2048}"
SERVER_LOG="${SERVER_LOG:-$ROOT_DIR/logs/llama31_8b_transformers.log}"

USE_CUDA="${USE_CUDA:-1}"
FORCE="${FORCE:-0}"

ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT_DIR}/data/long_horizon/alfworld}"
ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ALFWORLD_DATA}/base_config.tw.yaml}"
ALFWORLD_SPLIT="${ALFWORLD_SPLIT:-eval_out_of_distribution}"
ALFWORLD_EPISODES="${ALFWORLD_EPISODES:-20}"
ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"
ALFWORLD_SEED="${ALFWORLD_SEED:-42}"
ALFWORLD_REACT_OUT="${ALFWORLD_REACT_OUT:-${ROOT_DIR}/results/alfworld_react_llama31}"
ALFWORLD_COT_OUT="${ALFWORLD_COT_OUT:-${ROOT_DIR}/results/alfworld_cot_llama31}"
ALFWORLD_DIPLAN_OUT="${ALFWORLD_DIPLAN_OUT:-${ROOT_DIR}/results/alfworld_diplan_diffusion}"
ALFWORLD_AE_CKPT="${ALFWORLD_AE_CKPT:-${ROOT_DIR}/runs/alfworld_3000/ae/best.pt}"
ALFWORLD_PLANNER_CKPT="${ALFWORLD_PLANNER_CKPT:-${ROOT_DIR}/runs/alfworld_3000/diff/best.pt}"
ALFWORLD_VALUE_CKPT="${ALFWORLD_VALUE_CKPT:-${ROOT_DIR}/runs/alfworld_3000/value/best.pt}"
ALFWORLD_CONSTRAINT_CKPT="${ALFWORLD_CONSTRAINT_CKPT:-${ROOT_DIR}/runs/alfworld_3000/constraint/best.pt}"

SCI_PROCESSED_DIR="${SCI_PROCESSED_DIR:-data/scienceworld_processed}"
SCI_SEED="${SCI_SEED:-42}"
SCI_REACT_OUT="${SCI_REACT_OUT:-${ROOT_DIR}/results/scienceworld_react_llama31}"
SCI_COT_OUT="${SCI_COT_OUT:-${ROOT_DIR}/results/scienceworld_cot_llama31}"
SCI_DIPLAN_OUT_ROOT="${SCI_DIPLAN_OUT_ROOT:-results/scienceworld_new_seed42}"
SCI_DIPLAN_RUNS_ROOT="${SCI_DIPLAN_RUNS_ROOT:-runs/scienceworld_new_seed42}"
SCI_TASK_PROFILE="${SCI_TASK_PROFILE:-all}"

TABLE_OUT="${TABLE_OUT:-${ROOT_DIR}/results/react_cot_diplan_table}"

cd "$ROOT_DIR"
mkdir -p logs "$(dirname "$SERVER_LOG")" "$TABLE_OUT"

export TOG_OPENAI_API_BASE="${TOG_OPENAI_API_BASE:-http://127.0.0.1:${SERVER_PORT}/v1}"
export TOG_OPENAI_API_KEY="${TOG_OPENAI_API_KEY:-EMPTY}"
export TOG_OPENAI_MODEL="${TOG_OPENAI_MODEL:-$MODEL_NAME}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export ALFWORLD_DATA

if [[ "$USE_CUDA" == "1" ]]; then
  CUDA_FLAG=(--use_cuda)
else
  CUDA_FLAG=()
fi

function _info() {
  echo "[info] $*"
}

function _die() {
  echo "[fatal] $*" >&2
  exit 1
}

function _need_file() {
  local path="$1"
  [[ -f "$path" ]] || _die "missing file: $path"
}

function _need_dir() {
  local path="$1"
  [[ -d "$path" ]] || _die "missing dir: $path"
}

function _run_logged() {
  local name="$1"
  shift
  local log_file="logs/${name}.log"
  _info "running $name"
  echo "[cmd] $*"
  "$@" > "$log_file" 2>&1
  _info "finished $name log=$log_file"
}

function _maybe_run_summary() {
  local summary_path="$1"
  shift
  if [[ "$FORCE" != "1" && -f "$summary_path" ]]; then
    _info "skip existing summary: $summary_path"
    return
  fi
  "$@"
}

function _start_server() {
  [[ -f "$MODEL_DIR/config.json" ]] || _die "model config.json not found under $MODEL_DIR"
  _info "starting transformers server on :$SERVER_PORT"
  nohup "$PYTHON_BIN" -u scripts/serve_openai_compat_transformers.py \
    --model-path "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --dtype "$SERVER_DTYPE" \
    --max-input-tokens "$SERVER_MAX_INPUT_TOKENS" \
    > "$SERVER_LOG" 2>&1 &
  echo $! > logs/react_cot_diplan_server.pid
}

function _wait_server() {
  _info "waiting for server readiness"
  for _ in $(seq 1 180); do
    if curl -fsS "${TOG_OPENAI_API_BASE}/models" >/dev/null 2>&1; then
      _info "server is ready at ${TOG_OPENAI_API_BASE}"
      return 0
    fi
    sleep 5
  done
  tail -n 80 "$SERVER_LOG" >&2 || true
  _die "server did not become ready"
}

function _run_alfworld_react() {
  _need_dir "$ALFWORLD_DATA"
  _need_file "$ALFWORLD_CONFIG"
  _maybe_run_summary "$ALFWORLD_REACT_OUT/summary_metrics.json" \
    _run_logged "alfworld_react_llama31" \
      "$PYTHON_BIN" scripts/run_alfworld_react_text_baseline.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split "$ALFWORLD_SPLIT" \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        --out "$ALFWORLD_REACT_OUT" \
        --llm_api_base "$TOG_OPENAI_API_BASE" \
        --llm_api_key "$TOG_OPENAI_API_KEY" \
        --llm_model "$TOG_OPENAI_MODEL"
}

function _run_alfworld_cot() {
  _need_dir "$ALFWORLD_DATA"
  _need_file "$ALFWORLD_CONFIG"
  _maybe_run_summary "$ALFWORLD_COT_OUT/summary_metrics.json" \
    _run_logged "alfworld_cot_llama31" \
      "$PYTHON_BIN" scripts/run_alfworld_cot_text_baseline.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split "$ALFWORLD_SPLIT" \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        --out "$ALFWORLD_COT_OUT" \
        --llm_api_base "$TOG_OPENAI_API_BASE" \
        --llm_api_key "$TOG_OPENAI_API_KEY" \
        --llm_model "$TOG_OPENAI_MODEL"
}

function _run_alfworld_diplan() {
  _need_dir "$ALFWORLD_DATA"
  _need_file "$ALFWORLD_CONFIG"
  _need_file "$ALFWORLD_AE_CKPT"
  _need_file "$ALFWORLD_PLANNER_CKPT"
  if [[ -n "$ALFWORLD_VALUE_CKPT" ]]; then
    _need_file "$ALFWORLD_VALUE_CKPT"
  fi
  if [[ -n "$ALFWORLD_CONSTRAINT_CKPT" ]]; then
    _need_file "$ALFWORLD_CONSTRAINT_CKPT"
  fi

  _maybe_run_summary "$ALFWORLD_DIPLAN_OUT/summary_metrics.json" \
    _run_logged "alfworld_diplan_diffusion" \
      "$PYTHON_BIN" scripts/run_alfworld_diplan_diffusion.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split "$ALFWORLD_SPLIT" \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        "${CUDA_FLAG[@]}" \
        --ae_ckpt "$ALFWORLD_AE_CKPT" \
        --planner_ckpt "$ALFWORLD_PLANNER_CKPT" \
        --value_ckpt "$ALFWORLD_VALUE_CKPT" \
        --constraint_ckpt "$ALFWORLD_CONSTRAINT_CKPT" \
        --out "$ALFWORLD_DIPLAN_OUT"
}

function _run_scienceworld_react() {
  _need_file "$SCI_PROCESSED_DIR/scienceworld_train.jsonl"
  _need_file "$SCI_PROCESSED_DIR/scienceworld_test.jsonl"
  _maybe_run_summary "$SCI_REACT_OUT/summary_metrics.json" \
    _run_logged "scienceworld_react_llama31" \
      "$PYTHON_BIN" scripts/run_scienceworld_react_baseline.py \
        --processed_dir "$SCI_PROCESSED_DIR" \
        --split test \
        --seed "$SCI_SEED" \
        --out "$SCI_REACT_OUT" \
        --llm_api_base "$TOG_OPENAI_API_BASE" \
        --llm_api_key "$TOG_OPENAI_API_KEY" \
        --llm_model "$TOG_OPENAI_MODEL"
}

function _run_scienceworld_cot() {
  _need_file "$SCI_PROCESSED_DIR/scienceworld_train.jsonl"
  _need_file "$SCI_PROCESSED_DIR/scienceworld_test.jsonl"
  _maybe_run_summary "$SCI_COT_OUT/summary_metrics.json" \
    _run_logged "scienceworld_cot_llama31" \
      "$PYTHON_BIN" scripts/run_scienceworld_cot_baseline.py \
        --processed_dir "$SCI_PROCESSED_DIR" \
        --split test \
        --seed "$SCI_SEED" \
        --out "$SCI_COT_OUT" \
        --llm_api_base "$TOG_OPENAI_API_BASE" \
        --llm_api_key "$TOG_OPENAI_API_KEY" \
        --llm_model "$TOG_OPENAI_MODEL"
}

function _run_scienceworld_diplan() {
  _need_file "$SCI_PROCESSED_DIR/scienceworld_train.jsonl"
  _need_file "$SCI_PROCESSED_DIR/scienceworld_test.jsonl"
  _maybe_run_summary "$SCI_DIPLAN_OUT_ROOT/test_with_value_eval/summary_metrics.json" \
    _run_logged "scienceworld_diplan_torch" \
      "$PYTHON_BIN" scripts/run_scienceworld_pipeline.py \
        --python_exec "$PYTHON_BIN" \
        --seed "$SCI_SEED" \
        --processed_dir "$SCI_PROCESSED_DIR" \
        --out_root "$SCI_DIPLAN_OUT_ROOT" \
        --runs_root "$SCI_DIPLAN_RUNS_ROOT" \
        --task_profile "$SCI_TASK_PROFILE" \
        "${CUDA_FLAG[@]}"
}

function _build_table() {
  _need_file "$ALFWORLD_REACT_OUT/summary_metrics.json"
  _need_file "$ALFWORLD_COT_OUT/summary_metrics.json"
  _need_file "$ALFWORLD_DIPLAN_OUT/summary_metrics.json"
  _need_file "$SCI_REACT_OUT/summary_metrics.json"
  _need_file "$SCI_COT_OUT/summary_metrics.json"
  _need_file "$SCI_DIPLAN_OUT_ROOT/test_with_value_eval/summary_metrics.json"

  _run_logged "react_cot_diplan_table" \
    "$PYTHON_BIN" scripts/build_react_cot_diplan_table.py \
      --alfworld-react "$ALFWORLD_REACT_OUT/summary_metrics.json" \
      --alfworld-cot "$ALFWORLD_COT_OUT/summary_metrics.json" \
      --alfworld-diplan "$ALFWORLD_DIPLAN_OUT/summary_metrics.json" \
      --scienceworld-react "$SCI_REACT_OUT/summary_metrics.json" \
      --scienceworld-cot "$SCI_COT_OUT/summary_metrics.json" \
      --scienceworld-diplan "$SCI_DIPLAN_OUT_ROOT/test_with_value_eval/summary_metrics.json" \
      --out "$TABLE_OUT"
}

if [[ "$START_SERVER" == "1" ]]; then
  _start_server
  _wait_server
else
  _info "assuming OpenAI-compatible server is already running at ${TOG_OPENAI_API_BASE}"
fi

_run_alfworld_react
_run_alfworld_cot
_run_alfworld_diplan
_run_scienceworld_react
_run_scienceworld_cot
_run_scienceworld_diplan
_build_table

_info "done"
_info "table markdown: $TABLE_OUT/comparison_table.md"
