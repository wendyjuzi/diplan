#!/usr/bin/env bash
set -euo pipefail

# One-click paper pipeline for DiPLaN.
# Includes:
# 1) (optional) real KGQA data preparation
# 2) AE training (auto if ckpt missing)
# 3) multiseed_tune4
# 4) multiseed_cross_infonce
# 5) main evaluation (seed reference)
# 6) ablation
# 7) retrieval baseline
# 8) paper-ready summary tables
# 9) (optional) DiPLaN + LLM agent evaluation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Avoid libgomp crash when OMP_NUM_THREADS is set to an invalid value.
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if [[ -z "${MKL_NUM_THREADS:-}" || ! "${MKL_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi
if [[ -z "${OPENBLAS_NUM_THREADS:-}" || ! "${OPENBLAS_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
fi

USE_CUDA=0
RUN_PREPARE_REAL=0
RUN_LLM_AGENT=0
RUN_FULLPOOL=1
RUN_KGQA_BASELINES=1

CWQ_PATH=""
WEBQSP_PATH=""
GRAILQA_PATH=""

TRAIN_PATH="data/real_processed/kgqa_train.jsonl"
TEST_PATH="data/real_processed/kgqa_test.jsonl"

AE_CONFIG="configs/autoencoder_torch_kgqa.tune3.noise003.json"
AE_OUT="runs/ae_kgqa_torch_real_tune3_noise003"
AE_CKPT=""

SEEDS_STR="42 43 44"
INCLUDE_DATASETS_STR="cwq webqsp"

LLM_API_BASE="http://127.0.0.1:8000/v1"
LLM_API_KEY="EMPTY"
LLM_MODEL="Qwen3.5-35B-A3B-FP8"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="results/paper_all_${TIMESTAMP}"
RUNS_ROOT="runs/paper_all_${TIMESTAMP}"
LOG_DIR="${RESULT_ROOT}/logs"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_all_paper.sh [options]

Options:
  --use-cuda
  --prepare-real-data
  --cwq PATH
  --webqsp PATH
  --grailqa PATH
  --train-path PATH
  --test-path PATH
  --ae-config PATH
  --ae-out PATH
  --ae-ckpt PATH
  --seeds "42 43 44"
  --include-datasets "cwq webqsp"
  --run-llm-agent
  --skip-fullpool
  --skip-kgqa-baselines
  --llm-api-base URL
  --llm-api-key KEY
  --llm-model NAME
  --result-root DIR
  --runs-root DIR
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-cuda) USE_CUDA=1; shift ;;
    --prepare-real-data) RUN_PREPARE_REAL=1; shift ;;
    --cwq) CWQ_PATH="$2"; shift 2 ;;
    --webqsp) WEBQSP_PATH="$2"; shift 2 ;;
    --grailqa) GRAILQA_PATH="$2"; shift 2 ;;
    --train-path) TRAIN_PATH="$2"; shift 2 ;;
    --test-path) TEST_PATH="$2"; shift 2 ;;
    --ae-config) AE_CONFIG="$2"; shift 2 ;;
    --ae-out) AE_OUT="$2"; shift 2 ;;
    --ae-ckpt) AE_CKPT="$2"; shift 2 ;;
    --seeds) SEEDS_STR="$2"; shift 2 ;;
    --include-datasets) INCLUDE_DATASETS_STR="$2"; shift 2 ;;
    --run-llm-agent) RUN_LLM_AGENT=1; shift ;;
    --skip-fullpool) RUN_FULLPOOL=0; shift ;;
    --skip-kgqa-baselines) RUN_KGQA_BASELINES=0; shift ;;
    --llm-api-base) LLM_API_BASE="$2"; shift 2 ;;
    --llm-api-key) LLM_API_KEY="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$RESULT_ROOT" "$RUNS_ROOT" "$LOG_DIR"
mkdir -p "$RESULT_ROOT/generated_configs"

if [[ -z "$AE_CKPT" ]]; then
  AE_CKPT="${AE_OUT}/best.pt"
fi

read -r -a SEEDS <<<"$SEEDS_STR"
read -r -a INCLUDE_DATASETS <<<"$INCLUDE_DATASETS_STR"
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "No seeds provided."
  exit 1
fi
REF_SEED="${SEEDS[0]}"

to_json_array() {
  local arr=("$@")
  if [[ ${#arr[@]} -eq 0 ]]; then
    echo "[]"
    return
  fi
  local out="["
  local i
  for i in "${!arr[@]}"; do
    out+="\"${arr[$i]}\""
    if [[ "$i" -lt $((${#arr[@]} - 1)) ]]; then
      out+=", "
    fi
  done
  out+="]"
  echo "$out"
}

INCLUDE_DATASETS_JSON="$(to_json_array "${INCLUDE_DATASETS[@]}")"

run_cmd() {
  local log_file="$1"
  shift
  echo
  echo "[run] $*"
  "$@" 2>&1 | tee "$log_file"
}

if [[ "$RUN_PREPARE_REAL" -eq 1 ]]; then
  PREP_CMD=(python scripts/prepare_real_kgqa_data.py --out data/real_processed)
  if [[ -n "$CWQ_PATH" ]]; then PREP_CMD+=(--cwq "$CWQ_PATH"); fi
  if [[ -n "$WEBQSP_PATH" ]]; then PREP_CMD+=(--webqsp "$WEBQSP_PATH"); fi
  if [[ -n "$GRAILQA_PATH" ]]; then PREP_CMD+=(--grailqa "$GRAILQA_PATH"); fi
  if [[ ${#PREP_CMD[@]} -le 4 ]]; then
    echo "You enabled --prepare-real-data but did not provide any source path."
    exit 1
  fi
  run_cmd "${LOG_DIR}/01_prepare_real_data.log" "${PREP_CMD[@]}"
fi

if [[ ! -f "$TRAIN_PATH" || ! -f "$TEST_PATH" ]]; then
  echo "Missing train/test data:"
  echo "  train: $TRAIN_PATH"
  echo "  test : $TEST_PATH"
  echo "Run with --prepare-real-data and source paths, or set --train-path/--test-path."
  exit 1
fi

if [[ ! -f "$AE_CKPT" ]]; then
  AE_CMD=(python train_autoencoder_torch.py --config "$AE_CONFIG" --out "$AE_OUT")
  run_cmd "${LOG_DIR}/02_train_ae.log" "${AE_CMD[@]}"
fi
if [[ ! -f "$AE_CKPT" ]]; then
  echo "AE checkpoint not found after training: $AE_CKPT"
  exit 1
fi

TUNE4_OUT="${RESULT_ROOT}/multiseed_tune4"
TUNE4_RUNS="${RUNS_ROOT}/multiseed_tune4"
TUNE4_CMD=(
  python scripts/run_multiseed_tune4.py
  --seeds "${SEEDS[@]}"
  --train_path "$TRAIN_PATH"
  --test_path "$TEST_PATH"
  --ae_ckpt "$AE_CKPT"
  --out_root "$TUNE4_OUT"
  --runs_root "$TUNE4_RUNS"
)
if [[ "$USE_CUDA" -eq 1 ]]; then TUNE4_CMD+=(--use_cuda); fi
if [[ ${#INCLUDE_DATASETS[@]} -gt 0 ]]; then TUNE4_CMD+=(--include_datasets "${INCLUDE_DATASETS[@]}"); fi
run_cmd "${LOG_DIR}/03_multiseed_tune4.log" "${TUNE4_CMD[@]}"

CROSS_OUT="${RESULT_ROOT}/multiseed_cross_infonce"
CROSS_RUNS="${RUNS_ROOT}/multiseed_cross_infonce"
CROSS_CMD=(
  python scripts/run_multiseed_cross_infonce.py
  --seeds "${SEEDS[@]}"
  --train_path "$TRAIN_PATH"
  --test_path "$TEST_PATH"
  --ae_ckpt "$AE_CKPT"
  --out_root "$CROSS_OUT"
  --runs_root "$CROSS_RUNS"
)
if [[ "$USE_CUDA" -eq 1 ]]; then CROSS_CMD+=(--use_cuda); fi
if [[ ${#INCLUDE_DATASETS[@]} -gt 0 ]]; then CROSS_CMD+=(--include_datasets "${INCLUDE_DATASETS[@]}"); fi
run_cmd "${LOG_DIR}/04_multiseed_cross_infonce.log" "${CROSS_CMD[@]}"

FULLPOOL_OUT="${RESULT_ROOT}/multiseed_fullpool_listwise"
FULLPOOL_RUNS="${RUNS_ROOT}/multiseed_fullpool_listwise"
if [[ "$RUN_FULLPOOL" -eq 1 ]]; then
  FULLPOOL_CMD=(
    python scripts/run_multiseed_fullpool_listwise.py
    --seeds "${SEEDS[@]}"
    --train_path "$TRAIN_PATH"
    --test_path "$TEST_PATH"
    --ae_ckpt "$AE_CKPT"
    --planner_root "$CROSS_RUNS"
    --baseline_root "$CROSS_OUT"
    --baseline_value_root "$CROSS_RUNS"
    --baseline_run_name "mlp_memory_prefilter_cross_infonce"
    --baseline_label "stage4_cross_infonce"
    --out_root "$FULLPOOL_OUT"
    --runs_root "$FULLPOOL_RUNS"
  )
  if [[ "$USE_CUDA" -eq 1 ]]; then FULLPOOL_CMD+=(--use_cuda); fi
  run_cmd "${LOG_DIR}/05_multiseed_fullpool_listwise.log" "${FULLPOOL_CMD[@]}"
fi

REF_PLANNER_CKPT="${CROSS_RUNS}/seed_${REF_SEED}/mlp_planner/best.pt"
REF_VALUE_CKPT="${CROSS_RUNS}/seed_${REF_SEED}/value_cross_infonce/best.pt"
if [[ "$RUN_FULLPOOL" -eq 1 && -f "${FULLPOOL_RUNS}/seed_${REF_SEED}/value_full_pool_listwise/best.pt" ]]; then
  REF_VALUE_CKPT="${FULLPOOL_RUNS}/seed_${REF_SEED}/value_full_pool_listwise/best.pt"
fi
if [[ ! -f "$REF_PLANNER_CKPT" || ! -f "$REF_VALUE_CKPT" ]]; then
  echo "Reference planner/value checkpoint missing for seed ${REF_SEED}."
  echo "planner: $REF_PLANNER_CKPT"
  echo "value  : $REF_VALUE_CKPT"
  exit 1
fi

EVAL_MAIN_CFG="${RESULT_ROOT}/generated_configs/eval_main_seed${REF_SEED}.json"
cat > "$EVAL_MAIN_CFG" <<EOF
{
  "test_path": "${TEST_PATH}",
  "train_path": "${TRAIN_PATH}",
  "seed": ${REF_SEED},
  "include_datasets": ${INCLUDE_DATASETS_JSON},
  "num_candidates": 32,
  "receding_horizon": false,
  "use_value_model": true,
  "use_cuda": $( [[ "$USE_CUDA" -eq 1 ]] && echo "true" || echo "false" ),
  "use_memory_retrieval": true,
  "memory_prefilter_feasible": true,
  "memory_top_k": 32,
  "memory_max_postings_per_token": 1800,
  "candidate_latent_jitter_std": 0.05,
  "candidate_multi_jitter_stds": [0.02, 0.05, 0.08],
  "use_expected_length_prior": true,
  "expected_length_bucket_size": 4,
  "length_penalty_alpha": 0.05,
  "rerank_stage1_topk": 8,
  "rerank_consensus_weight": 0.75,
  "rerank_prefix_consensus_weight": 0.9,
  "rerank_memory_bonus": 0.15,
  "rerank_memory_rank_bonus": 0.6,
  "rerank_stage2_length_penalty_alpha": 0.08,
  "prefix_step_penalty_alpha": 0.2,
  "prefix_step_penalty_gamma": 0.85,
  "save_candidate_pool_topk": 12,
  "emit_structured_plan": true,
  "emit_tool_calls": true,
  "failure_memory_path": "",
  "failure_action_penalty": 0.0,
  "failure_memory_top_k": 8
}
EOF

MAIN_OUT="${RESULT_ROOT}/main_eval_seed${REF_SEED}"
run_cmd "${LOG_DIR}/05_main_eval.log" \
  python evaluate_torch.py \
  --config "$EVAL_MAIN_CFG" \
  --ae_ckpt "$AE_CKPT" \
  --planner_ckpt "$REF_PLANNER_CKPT" \
  --value_ckpt "$REF_VALUE_CKPT" \
  --out "$MAIN_OUT"

KGQA_BASELINES_OUT="${RESULT_ROOT}/kgqa_baselines_seed${REF_SEED}"
if [[ "$RUN_KGQA_BASELINES" -eq 1 ]]; then
  KGQA_BASELINES_CMD=(
    python scripts/run_kgqa_baselines.py
    --base-config "$EVAL_MAIN_CFG"
    --ae_ckpt "$AE_CKPT"
    --planner_ckpt "$REF_PLANNER_CKPT"
    --value_ckpt "$REF_VALUE_CKPT"
    --out_root "$KGQA_BASELINES_OUT"
    --ref-label diplan_full
    --bootstrap 2000
  )
  if [[ "$USE_CUDA" -eq 1 ]]; then KGQA_BASELINES_CMD+=(--use-cuda); fi
  run_cmd "${LOG_DIR}/06_kgqa_baselines.log" "${KGQA_BASELINES_CMD[@]}"
fi

ABLATION_CFG="${RESULT_ROOT}/generated_configs/ablation_seed${REF_SEED}.json"
cat > "$ABLATION_CFG" <<EOF
{
  "base_eval_config": "${EVAL_MAIN_CFG}",
  "experiments": [
    {"name": "full_diplan_torch", "overrides": {"num_candidates": 32, "receding_horizon": false, "use_value_model": true}},
    {"name": "no_value_guidance_torch", "overrides": {"use_value_model": false}},
    {"name": "no_receding_horizon_torch", "overrides": {"receding_horizon": false}},
    {"name": "low_candidate_budget_torch", "overrides": {"num_candidates": 4}}
  ]
}
EOF

ABLATION_OUT="${RESULT_ROOT}/ablation_seed${REF_SEED}"
run_cmd "${LOG_DIR}/06_ablation.log" \
  python run_ablation_torch.py \
  --config "$ABLATION_CFG" \
  --ae_ckpt "$AE_CKPT" \
  --planner_ckpt "$REF_PLANNER_CKPT" \
  --value_ckpt "$REF_VALUE_CKPT" \
  --out "$ABLATION_OUT"

RETR_OUT="${RESULT_ROOT}/retrieval_baseline_seed${REF_SEED}"
run_cmd "${LOG_DIR}/07_retrieval_baseline.log" \
  python evaluate_retrieval_baseline.py \
  --config configs/eval_retrieval_baseline.yaml \
  --out "$RETR_OUT"

DIRECT_RUN="${TUNE4_OUT}/seed_${REF_SEED}/mlp_direct"
PAIRWISE_RUN="${TUNE4_OUT}/seed_${REF_SEED}/mlp_memory_prefilter_pairwise_value"
INFONCE_RUN="${CROSS_OUT}/seed_${REF_SEED}/mlp_memory_prefilter_cross_infonce"
FULLPOOL_RUN="${FULLPOOL_OUT}/seed_${REF_SEED}/mlp_memory_prefilter_cross_fullpool"
SUMMARY_OUT="${RESULT_ROOT}/paper_tables"

SUMMARY_CMD=(
  python scripts/summarize_experiment_runs.py
  --run direct="${DIRECT_RUN}"
  --run pairwise="${PAIRWISE_RUN}"
  --run infonce="${INFONCE_RUN}"
)
if [[ "$RUN_FULLPOOL" -eq 1 && -d "$FULLPOOL_RUN" ]]; then
  SUMMARY_CMD+=(--run fullpool="${FULLPOOL_RUN}")
fi
SUMMARY_CMD+=(
  --run retrieval="${RETR_OUT}"
  --run main="${MAIN_OUT}"
  --out_dir "${SUMMARY_OUT}"
  --prefix "paper_ready_seed${REF_SEED}"
  --ref_label direct
)
run_cmd "${LOG_DIR}/08_paper_summary.log" "${SUMMARY_CMD[@]}"

if [[ "$RUN_LLM_AGENT" -eq 1 ]]; then
  LLM_CFG="${RESULT_ROOT}/generated_configs/eval_llm_agent_seed${REF_SEED}.json"
  cat > "$LLM_CFG" <<EOF
{
  "test_path": "${TEST_PATH}",
  "train_path": "${TRAIN_PATH}",
  "seed": ${REF_SEED},
  "use_cuda": $( [[ "$USE_CUDA" -eq 1 ]] && echo "true" || echo "false" ),
  "include_datasets": ${INCLUDE_DATASETS_JSON},
  "use_memory_retrieval": true,
  "memory_prefilter_feasible": true,
  "memory_top_k": 16,
  "memory_max_postings_per_token": 1200,
  "use_expected_length_prior": true,
  "expected_length_bucket_size": 4,
  "length_penalty_alpha": 0.05,
  "num_candidates": 12,
  "receding_horizon": true,
  "candidate_latent_jitter_std": 0.05,
  "candidate_multi_jitter_stds": [0.02, 0.08],
  "use_value_model": false,
  "save_episode_trace": true,
  "llm_api_base": "${LLM_API_BASE}",
  "llm_api_key": "${LLM_API_KEY}",
  "llm_model": "${LLM_MODEL}",
  "llm_temperature": 0.1,
  "llm_max_tokens": 128,
  "llm_timeout_s": 30,
  "llm_retries": 2,
  "llm_top_k": 8,
  "rerank_memory_rank_bonus": 0.6
}
EOF

  LLM_OUT="${RESULT_ROOT}/llm_agent_seed${REF_SEED}"
  run_cmd "${LOG_DIR}/09_llm_agent.log" \
    python scripts/run_diplan_llm_agent.py \
    --config "$LLM_CFG" \
    --ae_ckpt "$AE_CKPT" \
    --planner_ckpt "$REF_PLANNER_CKPT" \
    --out "$LLM_OUT"
fi

cat <<EOF

[done] Paper pipeline finished.
Results root: ${RESULT_ROOT}
Runs root   : ${RUNS_ROOT}

Key outputs:
  - ${RESULT_ROOT}/multiseed_tune4
  - ${RESULT_ROOT}/multiseed_cross_infonce
  - ${RESULT_ROOT}/multiseed_fullpool_listwise
  - ${MAIN_OUT}/summary_metrics.json
  - ${KGQA_BASELINES_OUT}/tables/kgqa_baselines_aggregate.csv
  - ${ABLATION_OUT}
  - ${SUMMARY_OUT}
EOF
