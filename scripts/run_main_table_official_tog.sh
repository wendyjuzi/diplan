#!/usr/bin/env bash
set -Eeuo pipefail

# Run the paper main-result table under the patched official ToG scaffold.
#
# This script targets the table:
# | Method | WebQSP | CWQ | Selection | Execution | Gap | Calls | Time |
#
# It runs ToG / PoG / FLARE / DiPLaN under the same local oracle-subgraph
# scaffold. RoG is left as an external hook because this repo does not contain
# the original RoG inference runner.
#
# Example:
#   bash scripts/run_main_table_official_tog.sh run
#   bash scripts/run_main_table_official_tog.sh summarize
#   bash scripts/run_main_table_official_tog.sh all

ACTION="${1:-help}"

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/DiPLaN}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OFFICIAL_TOG_DIR="${OFFICIAL_TOG_DIR:-/root/autodl-tmp/paper_baselines/ToG/ToG}"
OFFICIAL_TOG_PARENT="${OFFICIAL_TOG_PARENT:-$(dirname "$OFFICIAL_TOG_DIR")}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/main_table_official_tog}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/main_table_official_tog}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

WEBQSP_PATH="${WEBQSP_PATH:-$ROOT_DIR/data/rog_processed/webqsp_test.jsonl}"
CWQ_PATH="${CWQ_PATH:-$ROOT_DIR/data/rog_processed/cwq_test.jsonl}"
MAX_TASKS="${MAX_TASKS:-0}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-128}"
MAX_PROMPT_RELATIONS="${MAX_PROMPT_RELATIONS:-128}"
TEMP_EXPLORATION="${TEMP_EXPLORATION:-0.1}"
TEMP_REASONING="${TEMP_REASONING:-0}"
WIDTH="${WIDTH:-8}"
DEPTH="${DEPTH:-3}"
NUM_RETAIN_ENTITY="${NUM_RETAIN_ENTITY:-20}"
RELATION_FIRST_K="${RELATION_FIRST_K:-16}"
RELATION_PRUNE_TOOLS="${RELATION_PRUNE_TOOLS:-llm}"
ENTITY_PRUNE_TOOLS="${ENTITY_PRUNE_TOOLS:-lexical}"
DIPLAN_POOL_STRATEGY="${DIPLAN_POOL_STRATEGY:-all_legal}"

RUN_TOG="${RUN_TOG:-1}"
RUN_POG="${RUN_POG:-1}"
RUN_FLARE="${RUN_FLARE:-1}"
RUN_DIPLAN="${RUN_DIPLAN:-1}"
FORCE="${FORCE:-0}"

MCTS_SIMS="${MCTS_SIMS:-16}"
MCTS_HORIZON="${MCTS_HORIZON:-3}"
MCTS_K="${MCTS_K:-8}"
MCTS_UCB_C="${MCTS_UCB_C:-1.4}"
MCTS_USE_MEMORY="${MCTS_USE_MEMORY:-True}"
MCTS_MEMORY_CAP="${MCTS_MEMORY_CAP:-200}"
MCTS_MEMORY_SIM="${MCTS_MEMORY_SIM:-0.9}"
MCTS_COMMITMENT="${MCTS_COMMITMENT:-receding}"

DIPLAN_RELATION_SCORER_CKPT="${DIPLAN_RELATION_SCORER_CKPT:-$ROOT_DIR/runs/relation_scorer_webqsp_seed42/best.pt}"
DIPLAN_CANDIDATE_DIFFUSION_CKPT="${DIPLAN_CANDIDATE_DIFFUSION_CKPT:-$ROOT_DIR/runs/candidate_diffusion_webqsp_seed42_hardcfg/best.pt}"
DIPLAN_TRAJECTORY_DIFFUSION_CKPT="${DIPLAN_TRAJECTORY_DIFFUSION_CKPT:-$ROOT_DIR/runs/trajectory_diffusion_webqsp_seed42/best.pt}"
DIPLAN_FUSION_CKPT="${DIPLAN_FUSION_CKPT:-$ROOT_DIR/runs/fusion_ranker_webqsp_seed42_prior_guided_listwise/best.pt}"
DIPLAN_AE_CKPT="${DIPLAN_AE_CKPT:-$ROOT_DIR/runs/ae_kgqa_torch_real_tune3_noise003/best.pt}"
DIPLAN_PLANNER_CKPT="${DIPLAN_PLANNER_CKPT:-$ROOT_DIR/runs/multiseed_cross_infonce_cwq_webqsp/seed_42/mlp_planner/best.pt}"
DIPLAN_VALUE_CKPT="${DIPLAN_VALUE_CKPT:-$ROOT_DIR/runs/final_kgqa_pool48_strong_multiseed/seed_42/value_full_pool_listwise/best.pt}"
DIPLAN_RERANK_POOL_SIZE="${DIPLAN_RERANK_POOL_SIZE:-256}"
DIPLAN_POST_RERANK_TOPK="${DIPLAN_POST_RERANK_TOPK:-16}"
DIPLAN_HORIZON="${DIPLAN_HORIZON:-3}"

ROG_WEBQSP_SUMMARY="${ROG_WEBQSP_SUMMARY:-}"
ROG_CWQ_SUMMARY="${ROG_CWQ_SUMMARY:-}"

function _info() {
  echo "[info] $*"
}

function _die() {
  echo "[fatal] $*" >&2
  exit 1
}

function _ensure_dirs() {
  mkdir -p "$OUT_ROOT" "$LOG_DIR"
}

function _check_paths() {
  [[ -d "$ROOT_DIR" ]] || _die "ROOT_DIR not found: $ROOT_DIR"
  [[ -d "$OFFICIAL_TOG_DIR" ]] || _die "OFFICIAL_TOG_DIR not found: $OFFICIAL_TOG_DIR"
  [[ -f "$WEBQSP_PATH" ]] || _die "WEBQSP_PATH not found: $WEBQSP_PATH"
  [[ -f "$CWQ_PATH" ]] || _die "CWQ_PATH not found: $CWQ_PATH"
}

function _patch_if_needed() {
  if [[ ! -f "$OFFICIAL_TOG_DIR/main_subgraph_diplan.py" ]]; then
    _info "patching official ToG with main_subgraph_diplan.py"
    cd "$ROOT_DIR"
    "$PYTHON_BIN" scripts/patch_official_tog_subgraph_diplan.py --tog_dir "$OFFICIAL_TOG_PARENT"
  fi
}

function _common_args() {
  cat <<EOF
--seed $SEED
--max_length $MAX_LENGTH
--max_prompt_relations $MAX_PROMPT_RELATIONS
--temperature_exploration $TEMP_EXPLORATION
--temperature_reasoning $TEMP_REASONING
--width $WIDTH
--depth $DEPTH
--remove_unnecessary_rel True
--LLM_type $MODEL_NAME
--opeani_api_keys $OPENAI_API_KEY
--num_retain_entity $NUM_RETAIN_ENTITY
--prune_tools llm
--relation_prune_tools $RELATION_PRUNE_TOOLS
--entity_prune_tools $ENTITY_PRUNE_TOOLS
--selection_mode relation_first
--relation_first_k $RELATION_FIRST_K
--diplan_pool_strategy $DIPLAN_POOL_STRATEGY
--show_trace False
EOF
}

function _run_method() {
  local dataset="$1"
  local subgraph_jsonl="$2"
  local method_label="$3"
  shift 3
  local extra_args=("$@")

  local out_dir="$OUT_ROOT/$dataset/$method_label"
  local log_file="$LOG_DIR/${dataset}_${method_label}.log"
  mkdir -p "$out_dir"

  if [[ "$FORCE" != "1" && -f "$out_dir/summary_metrics.json" ]]; then
    _info "skip existing $dataset/$method_label"
    return
  fi

  local cmd=(
    "$PYTHON_BIN" main_subgraph_diplan.py
    --subgraph_jsonl "$subgraph_jsonl"
    --out "$out_dir"
  )
  if [[ "$MAX_TASKS" != "0" ]]; then
    cmd+=(--max_tasks "$MAX_TASKS")
  fi

  while IFS= read -r line; do
    [[ -n "$line" ]] && cmd+=($line)
  done < <(_common_args)

  cmd+=("${extra_args[@]}")

  _info "running $dataset/$method_label"
  (
    cd "$OFFICIAL_TOG_DIR"
    export TOG_OPENAI_API_BASE="$OPENAI_API_BASE"
    export TOG_OPENAI_API_KEY="$OPENAI_API_KEY"
    export TOG_OPENAI_MODEL="$MODEL_NAME"
    "${cmd[@]}"
  ) >"$log_file" 2>&1

  _info "finished $dataset/$method_label log=$log_file"
}

function _run_dataset() {
  local dataset="$1"
  local path="$2"

  [[ "$RUN_TOG" == "1" ]] && _run_method "$dataset" "$path" "tog" \
    --planning_strategy tog

  [[ "$RUN_POG" == "1" ]] && _run_method "$dataset" "$path" "pog" \
    --planning_strategy pog \
    --pog_guidance_weight 0.15 \
    --pog_memory_weight 0.4 \
    --pog_reflection_weight 0.25 \
    --pog_fanout_penalty 0.12

  [[ "$RUN_FLARE" == "1" ]] && _run_method "$dataset" "$path" "flare" \
    --planning_strategy tog_mcts \
    --mcts_eval_mode flare \
    --mcts_sims "$MCTS_SIMS" \
    --mcts_ucb_c "$MCTS_UCB_C" \
    --mcts_horizon "$MCTS_HORIZON" \
    --mcts_k "$MCTS_K" \
    --mcts_use_memory "$MCTS_USE_MEMORY" \
    --mcts_memory_cap "$MCTS_MEMORY_CAP" \
    --mcts_memory_sim "$MCTS_MEMORY_SIM" \
    --mcts_commitment "$MCTS_COMMITMENT"

  [[ "$RUN_DIPLAN" == "1" ]] && _run_method "$dataset" "$path" "diplan" \
    --planning_strategy tog_diplan \
    --diplan_repo "$ROOT_DIR" \
    --diplan_relation_scorer_ckpt "$DIPLAN_RELATION_SCORER_CKPT" \
    --diplan_candidate_diffusion_ckpt "$DIPLAN_CANDIDATE_DIFFUSION_CKPT" \
    --diplan_candidate_guidance_scale 1.0 \
    --diplan_trajectory_diffusion_ckpt "$DIPLAN_TRAJECTORY_DIFFUSION_CKPT" \
    --diplan_trajectory_guidance_scale 1.0 \
    --diplan_fusion_ckpt "$DIPLAN_FUSION_CKPT" \
    --diplan_ae_ckpt "$DIPLAN_AE_CKPT" \
    --diplan_planner_ckpt "$DIPLAN_PLANNER_CKPT" \
    --diplan_value_ckpt "$DIPLAN_VALUE_CKPT" \
    --diplan_score_mode learned_fusion \
    --diplan_rerank_pool_size "$DIPLAN_RERANK_POOL_SIZE" \
    --diplan_post_rerank_topk "$DIPLAN_POST_RERANK_TOPK" \
    --diplan_horizon "$DIPLAN_HORIZON"
}

function _summarize() {
  cd "$ROOT_DIR"
  "$PYTHON_BIN" scripts/build_main_table_official_tog.py \
    --results_root "$OUT_ROOT" \
    --rog_webqsp_summary "$ROG_WEBQSP_SUMMARY" \
    --rog_cwq_summary "$ROG_CWQ_SUMMARY" \
    --out_prefix "$OUT_ROOT/main_table"
}

case "$ACTION" in
  run)
    _ensure_dirs
    _check_paths
    _patch_if_needed
    _run_dataset webqsp "$WEBQSP_PATH"
    _run_dataset cwq "$CWQ_PATH"
    ;;
  summarize)
    _ensure_dirs
    _summarize
    ;;
  all)
    _ensure_dirs
    _check_paths
    _patch_if_needed
    _run_dataset webqsp "$WEBQSP_PATH"
    _run_dataset cwq "$CWQ_PATH"
    _summarize
    ;;
  help|-h|--help)
    cat <<EOF
Usage: bash scripts/run_main_table_official_tog.sh <run|summarize|all>

Important environment variables:
  ROOT_DIR=/root/autodl-tmp/DiPLaN
  OFFICIAL_TOG_DIR=/root/autodl-tmp/paper_baselines/ToG/ToG
  MODEL_NAME=Qwen2.5-7B-Instruct
  OPENAI_API_BASE=http://127.0.0.1:8000/v1
  OPENAI_API_KEY=EMPTY
  WEBQSP_PATH=$ROOT_DIR/data/rog_processed/webqsp_test.jsonl
  CWQ_PATH=$ROOT_DIR/data/rog_processed/cwq_test.jsonl
  MAX_TASKS=0
  FORCE=0

Optional external RoG summaries for the final table:
  ROG_WEBQSP_SUMMARY=/abs/path/to/rog_webqsp_summary.json
  ROG_CWQ_SUMMARY=/abs/path/to/rog_cwq_summary.json
EOF
    ;;
  *)
    _die "unknown action: $ACTION"
    ;;
esac
