#!/usr/bin/env bash
set -Eeuo pipefail

# One-command runner for the final DiPLaN paper-support experiments.
# Run from the repo root on AutoDL:
#   nohup bash scripts/run_all_diplan_final_experiments.sh > logs/final_experiments.master.log 2>&1 &

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/DiPLaN}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_CUDA="${USE_CUDA:-1}"

ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT_DIR}/data/long_horizon/alfworld}"
ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${ALFWORLD_DATA}/base_config.tw.yaml}"
ALFWORLD_EPISODES="${ALFWORLD_EPISODES:-134}"
ALFWORLD_MAX_STEPS="${ALFWORLD_MAX_STEPS:-50}"
ALFWORLD_COLLECT_EPISODES="${ALFWORLD_COLLECT_EPISODES:-3000}"
ALFWORLD_SEED="${ALFWORLD_SEED:-42}"

ALF_PROCESSED_DIR="${ALF_PROCESSED_DIR:-${ROOT_DIR}/data/long_horizon/alfworld_processed}"
ALF_RUNS_DIR="${ALF_RUNS_DIR:-${ROOT_DIR}/runs/alf}"
ALF_RESULTS_DIR="${ALF_RESULTS_DIR:-${ROOT_DIR}/results/final_alfworld_seed${ALFWORLD_SEED}}"

KGQA_SEEDS="${KGQA_SEEDS:-43 44}"
KGQA_OUT_ROOT="${KGQA_OUT_ROOT:-results/final_kgqa_pool48_strong_multiseed}"
KGQA_RUNS_ROOT="${KGQA_RUNS_ROOT:-runs/final_kgqa_pool48_strong_multiseed}"

RUN_ALFWORLD="${RUN_ALFWORLD:-1}"
RUN_KGQA="${RUN_KGQA:-1}"
FORCE="${FORCE:-0}"

cd "$ROOT_DIR"
mkdir -p logs "$ALF_RESULTS_DIR"
export ALFWORLD_DATA

if [[ "$USE_CUDA" == "1" ]]; then
  CUDA_FLAG=(--use_cuda)
else
  CUDA_FLAG=()
fi

log_step() {
  echo
  echo "========== $* =========="
  date "+%Y-%m-%d %H:%M:%S"
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

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[missing] $path" >&2
    return 1
  fi
}

maybe_collect_alfworld() {
  if [[ "$FORCE" != "1" && -f "$ALF_PROCESSED_DIR/train.jsonl" && -f "$ALF_PROCESSED_DIR/manifest.json" ]]; then
    echo "[skip] ALFWorld processed trajectories already exist: $ALF_PROCESSED_DIR"
    return
  fi
  run_logged "alfworld_collect_${ALFWORLD_COLLECT_EPISODES}" \
    "$PYTHON_BIN" scripts/collect_alfworld_trajectories.py \
      --data_root "$ALFWORLD_DATA" \
      --config "$ALFWORLD_CONFIG" \
      --episodes "$ALFWORLD_COLLECT_EPISODES" \
      --max_steps 60 \
      --seed "$ALFWORLD_SEED" \
      --out "$ALF_PROCESSED_DIR"
}

maybe_train_ae() {
  local ckpt="$ALF_RUNS_DIR/ae/best.pt"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] AE checkpoint exists: $ckpt"
    return
  fi
  run_logged "alfworld_train_ae" \
    "$PYTHON_BIN" train_autoencoder_torch.py \
      --config configs/autoencoder_torch_alfworld.json \
      --out "$ALF_RUNS_DIR/ae"
  need_file "$ckpt"
}

maybe_train_diffusion() {
  local ckpt="$ALF_RUNS_DIR/diff/best.pt"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] Diffusion checkpoint exists: $ckpt"
    return
  fi
  run_logged "alfworld_train_diffusion" \
    "$PYTHON_BIN" train_diffusion_planner_torch.py \
      --config configs/diffusion_torch_alfworld.json \
      --ae_ckpt "$ALF_RUNS_DIR/ae/best.pt" \
      --out "$ALF_RUNS_DIR/diff"
  need_file "$ckpt"
}

maybe_train_value() {
  local ckpt="$ALF_RUNS_DIR/value/best.pt"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] Value checkpoint exists: $ckpt"
    return
  fi
  run_logged "alfworld_train_value" \
    "$PYTHON_BIN" train_value_model_torch.py \
      --config configs/value_torch_alfworld.json \
      --planner_ckpt "$ALF_RUNS_DIR/diff/best.pt" \
      --out "$ALF_RUNS_DIR/value"
  need_file "$ckpt"
}

maybe_train_constraint() {
  local ckpt="$ALF_RUNS_DIR/constraint/best.pt"
  if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
    echo "[skip] Constraint checkpoint exists: $ckpt"
    return
  fi
  run_logged "alfworld_train_constraint" \
    "$PYTHON_BIN" train_constraint_model_torch.py \
      --config configs/constraint_torch_alfworld.json \
      --planner_ckpt "$ALF_RUNS_DIR/diff/best.pt" \
      --out "$ALF_RUNS_DIR/constraint"
  need_file "$ckpt"
}

maybe_eval_summary() {
  local out_dir="$1"
  shift
  if [[ "$FORCE" != "1" && -f "$out_dir/summary_metrics.json" ]]; then
    echo "[skip] summary exists: $out_dir/summary_metrics.json"
    return
  fi
  "$@"
}

run_alfworld_evals() {
  maybe_collect_alfworld
  maybe_train_ae
  maybe_train_diffusion
  maybe_train_value
  maybe_train_constraint

  local ae="$ALF_RUNS_DIR/ae/best.pt"
  local diff="$ALF_RUNS_DIR/diff/best.pt"
  local value="$ALF_RUNS_DIR/value/best.pt"
  local constraint="$ALF_RUNS_DIR/constraint/best.pt"

  maybe_eval_summary "$ALF_RESULTS_DIR/full_ood${ALFWORLD_EPISODES}" \
    run_logged "alfworld_full_ood${ALFWORLD_EPISODES}" \
      "$PYTHON_BIN" scripts/run_alfworld_diplan_diffusion.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split eval_out_of_distribution \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        "${CUDA_FLAG[@]}" \
        --ae_ckpt "$ae" \
        --planner_ckpt "$diff" \
        --value_ckpt "$value" \
        --constraint_ckpt "$constraint" \
        --out "$ALF_RESULTS_DIR/full_ood${ALFWORLD_EPISODES}"

  maybe_eval_summary "$ALF_RESULTS_DIR/lite_ood${ALFWORLD_EPISODES}" \
    run_logged "alfworld_lite_ood${ALFWORLD_EPISODES}" \
      "$PYTHON_BIN" scripts/run_alfworld_diplan_agent.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split eval_out_of_distribution \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        --variant lite \
        --out "$ALF_RESULTS_DIR/lite_ood${ALFWORLD_EPISODES}"

  maybe_eval_summary "$ALF_RESULTS_DIR/no_receding_ood${ALFWORLD_EPISODES}" \
    run_logged "alfworld_no_receding_ood${ALFWORLD_EPISODES}" \
      "$PYTHON_BIN" scripts/run_alfworld_diplan_diffusion.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split eval_out_of_distribution \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        "${CUDA_FLAG[@]}" \
        --ae_ckpt "$ae" \
        --planner_ckpt "$diff" \
        --value_ckpt "$value" \
        --constraint_ckpt "$constraint" \
        --no_receding \
        --out "$ALF_RESULTS_DIR/no_receding_ood${ALFWORLD_EPISODES}"

  maybe_eval_summary "$ALF_RESULTS_DIR/no_value_guidance_ood${ALFWORLD_EPISODES}" \
    run_logged "alfworld_no_value_guidance_ood${ALFWORLD_EPISODES}" \
      "$PYTHON_BIN" scripts/run_alfworld_diplan_diffusion.py \
        --data_root "$ALFWORLD_DATA" \
        --config "$ALFWORLD_CONFIG" \
        --split eval_out_of_distribution \
        --episodes "$ALFWORLD_EPISODES" \
        --max_steps "$ALFWORLD_MAX_STEPS" \
        --seed "$ALFWORLD_SEED" \
        "${CUDA_FLAG[@]}" \
        --ae_ckpt "$ae" \
        --planner_ckpt "$diff" \
        --constraint_ckpt "$constraint" \
        --out "$ALF_RESULTS_DIR/no_value_guidance_ood${ALFWORLD_EPISODES}"
}

run_kgqa() {
  run_logged "kgqa_pool48_strong_seeds_${KGQA_SEEDS// /_}" \
    "$PYTHON_BIN" scripts/run_multiseed_fullpool_listwise.py \
      --seeds $KGQA_SEEDS \
      --pool_size 48 \
      --retrieval_pool_aware \
      --train_num_candidates 48 \
      --train_memory_top_k 64 \
      --test_num_candidates 48 \
      --test_memory_top_k 64 \
      --train_candidate_multi_jitter_stds "0.0 0.03 0.06" \
      --test_candidate_multi_jitter_stds "0.0 0.03 0.06" \
      --value_epochs 30 \
      --value_lr 3e-4 \
      --value_batch_size 256 \
      --value_hidden_dim 512 \
      --value_dropout 0.1 \
      --out_root "$KGQA_OUT_ROOT" \
      --runs_root "$KGQA_RUNS_ROOT" \
      "${CUDA_FLAG[@]}"
}

log_step "DiPLaN final experiment suite"
echo "[root] $ROOT_DIR"
echo "[alfworld_data] $ALFWORLD_DATA"
echo "[alfworld_config] $ALFWORLD_CONFIG"
echo "[run_alfworld] $RUN_ALFWORLD"
echo "[run_kgqa] $RUN_KGQA"
echo "[force] $FORCE"

if [[ "$RUN_ALFWORLD" == "1" ]]; then
  run_alfworld_evals
fi

if [[ "$RUN_KGQA" == "1" ]]; then
  run_kgqa
fi

log_step "Final summaries"
find "$ALF_RESULTS_DIR" -name summary_metrics.json -maxdepth 2 -print -exec cat {} \; || true
echo "[kgqa_out_root] $KGQA_OUT_ROOT"
echo "[done] all requested DiPLaN experiments finished"
