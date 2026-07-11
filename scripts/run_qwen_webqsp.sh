#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen-based WebQSP runner for the paper mainline.
# Run from the repo root on a Linux GPU box:
#   nohup bash scripts/run_qwen_webqsp.sh > logs/qwen_webqsp.master.log 2>&1 &
#
# This script assumes you already started an OpenAI-compatible vLLM server, e.g.:
#   vllm serve /root/autodl-tmp/Qwen3.5-35B-A3B-FP8 \
#     --served-model-name Qwen3.5-35B-A3B-FP8 \
#     --port 8000 \
#     --gpu-memory-utilization 0.85 \
#     --max-model-len 8192

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/DiPLaN}"
PYTHON_BIN="${PYTHON_BIN:-python}"

AE_CKPT="${AE_CKPT:-runs/ae_kgqa_torch_real_tune3_noise003/best.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-runs/multiseed_cross_infonce_cwq_webqsp/seed_42/mlp_planner/best.pt}"
VALUE_CKPT="${VALUE_CKPT:-}"

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_DIPLAN="${RUN_DIPLAN:-1}"
FORCE="${FORCE:-0}"

SMOKE_BASELINES_CFG="configs/eval_tog_subgraph_webqsp.qwen35a3b.baselines_smoke20.json"
SMOKE_DIPLAN_CFG="configs/eval_tog_subgraph_webqsp.qwen35a3b.with_diplan_smoke20.json"
FULL_BASELINES_CFG="configs/eval_tog_subgraph_webqsp.qwen35a3b.baselines_full.json"
FULL_DIPLAN_CFG="configs/eval_tog_subgraph_webqsp.qwen35a3b.with_diplan_full.json"

OUT_ROOT="${OUT_ROOT:-results/qwen35_webqsp}"

cd "$ROOT_DIR"
mkdir -p logs "$OUT_ROOT"

log_step() {
  echo
  echo "========== $* =========="
  date "+%Y-%m-%d %H:%M:%S"
}

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[missing] $path" >&2
    return 1
  fi
}

run_logged() {
  local name="$1"
  shift
  local log_file="logs/${name}.log"
  log_step "$name"
  echo "[cmd] $*"
  "$@" > "$log_file" 2>&1
  echo "[ok] $name finished; log=$log_file"
}

maybe_run_summary() {
  local out_dir="$1"
  shift
  if [[ "$FORCE" != "1" && -f "$out_dir/summary_metrics.json" ]]; then
    echo "[skip] summary exists: $out_dir/summary_metrics.json"
    return
  fi
  "$@"
}

run_baselines() {
  local split_name="$1"
  local cfg_path="$2"
  local out_dir="$3"
  maybe_run_summary "$out_dir" \
    run_logged "qwen_webqsp_${split_name}_baselines" \
      "$PYTHON_BIN" scripts/run_tog_subgraph_planning_eval.py \
        --config "$cfg_path" \
        --out "$out_dir"
}

run_diplan() {
  local split_name="$1"
  local cfg_path="$2"
  local out_dir="$3"

  need_file "$AE_CKPT"
  need_file "$PLANNER_CKPT"

  local cmd=(
    "$PYTHON_BIN" scripts/run_tog_subgraph_planning_eval.py
    --config "$cfg_path"
    --out "$out_dir"
    --ae_ckpt "$AE_CKPT"
    --planner_ckpt "$PLANNER_CKPT"
  )
  if [[ -n "$VALUE_CKPT" ]]; then
    need_file "$VALUE_CKPT"
    cmd+=(--value_ckpt "$VALUE_CKPT")
  fi

  maybe_run_summary "$out_dir" \
    run_logged "qwen_webqsp_${split_name}_diplan" \
      "${cmd[@]}"
}

log_step "Qwen WebQSP Runner"
echo "[root] $ROOT_DIR"
echo "[ae_ckpt] $AE_CKPT"
echo "[planner_ckpt] $PLANNER_CKPT"
echo "[value_ckpt] ${VALUE_CKPT:-<none>}"
echo "[run_smoke] $RUN_SMOKE"
echo "[run_full] $RUN_FULL"
echo "[run_baselines] $RUN_BASELINES"
echo "[run_diplan] $RUN_DIPLAN"
echo "[force] $FORCE"

if [[ "$RUN_SMOKE" == "1" ]]; then
  if [[ "$RUN_BASELINES" == "1" ]]; then
    run_baselines "smoke20" "$SMOKE_BASELINES_CFG" "$OUT_ROOT/baselines_smoke20"
  fi
  if [[ "$RUN_DIPLAN" == "1" ]]; then
    run_diplan "smoke20" "$SMOKE_DIPLAN_CFG" "$OUT_ROOT/diplan_smoke20"
  fi
fi

if [[ "$RUN_FULL" == "1" ]]; then
  if [[ "$RUN_BASELINES" == "1" ]]; then
    run_baselines "full" "$FULL_BASELINES_CFG" "$OUT_ROOT/baselines_full"
  fi
  if [[ "$RUN_DIPLAN" == "1" ]]; then
    run_diplan "full" "$FULL_DIPLAN_CFG" "$OUT_ROOT/diplan_full"
  fi
fi

log_step "Qwen WebQSP Summaries"
find "$OUT_ROOT" -name summary_metrics.json -print -exec cat {} \; || true
echo "[done] qwen webqsp runs finished"
