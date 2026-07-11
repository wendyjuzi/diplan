# Paper Mainline Experiment Runbook

This runbook collects the concrete commands for the next round of DiPLaN paper experiments.
It is intentionally scoped to the paper mainline:

- WebQSP mainline KGQA result
- WebQSP efficiency comparison
- CWQ transfer / retraining
- ALFWorld 3000-trajectory retraining

The commands assume a Linux GPU box and that you run them from the repo root.

## 0. Environment

```bash
cd /root/autodl-tmp/DiPLaN
conda activate diplan

export PYTHONPATH=/root/autodl-tmp/DiPLaN
export PYTHON_BIN=python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
```

If you need the LLM-backed ToG runs, also serve the backbone:

```bash
vllm serve /root/autodl-tmp/Meta-Llama-3.1-8B-Instruct \
  --served-model-name Llama-3.1-8B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

## 1. WebQSP Mainline Headline

Use this line when you want to reinforce the paper's core KGQA story:

- dataset: `WebQSP`
- scaffold: patched official ToG over RoG subgraphs
- setting: `relation_first_k=16`, `all_legal`, learned fusion

The authoritative numbers are already tracked in:

- `result_paper/DiPLaN_FINAL_v2.md`
- `result_paper/DiPLaN_AAAI.md`

If you want to rerun the lighter matched version first, use the official ToG subgraph planning config:

```bash
python scripts/run_tog_subgraph_planning_eval.py \
  --config configs/eval_tog_subgraph_webqsp.efficiency_mcts_vs_diplan_llama8b_smoke100.json \
  --out results/webqsp_efficiency_smoke100 \
  --ae_ckpt runs/ae_kgqa_torch_real_tune3_noise003/best.pt \
  --planner_ckpt runs/multiseed_cross_infonce_cwq_webqsp/seed_42/mlp_planner/best.pt
```

## 2. WebQSP Efficiency: FLARE-MCTS vs DiPLaN

### 2.1 Matched FLARE/MCTS run

```bash
python scripts/run_kgqa_planning_eval.py \
  --config configs/eval_kgqa_planning.llm.json \
  --out results/kgqa_planning_llm_webqsp \
  --ae_ckpt runs/ae_kgqa_torch_real_tune3_noise003/best.pt \
  --planner_ckpt runs/multiseed_cross_infonce_cwq_webqsp/seed_42/mlp_planner/best.pt
```

### 2.2 Summarize the efficiency comparison

```bash
python scripts/summarize_mcts_efficiency.py \
  --run_dir results/kgqa_planning_llm_webqsp \
  --mcts_label flare \
  --diplan_label diplan_diffusion
```

If you are comparing a strong official-ToG DiPLaN run against the MCTS summary:

```bash
python scripts/compare_strong_tog_diplan_vs_mcts_efficiency.py \
  --mcts_summary results/kgqa_planning_llm_webqsp/summary_metrics.json \
  --diplan_summary results/official_tog_diplan_webqsp_llama31_k16_entitylex_smoke100/summary_metrics.json \
  --mcts_method flare \
  --out results/efficiency_compare_webqsp_smoke100.json
```

## 3. WebQSP Module Training

These are the most useful retraining commands if you want to tighten the paper story around
candidate diffusion, trajectory diffusion, and learned fusion.

### 3.1 Prepare RoG KG data

```bash
python scripts/download_rog_kgqa_data.py --datasets webqsp --splits train test

python scripts/prepare_rog_kg_env_data.py \
  --datasets webqsp \
  --splits train test \
  --dev_fraction 0.1 \
  --seed 42
```

### 3.2 Train the relation scorer

```bash
python train_relation_scorer_torch.py \
  --train_path data/rog_processed/webqsp_train.jsonl \
  --valid_path data/rog_processed/webqsp_dev.jsonl \
  --out runs/relation_scorer_webqsp_seed42 \
  --seed 42
```

### 3.3 Train candidate diffusion

```bash
python train_candidate_diffusion_planner.py \
  --train_path data/rog_processed/webqsp_train.jsonl \
  --valid_path data/rog_processed/webqsp_dev.jsonl \
  --out runs/candidate_diffusion_webqsp_seed42 \
  --epochs 20 \
  --batch_size 64 \
  --condition_dropout 0.1 \
  --noise_strategy hard \
  --seed 42
```

### 3.4 Train trajectory diffusion

```bash
python train_trajectory_diffusion_planner.py \
  --train_path data/rog_processed/webqsp_train.jsonl \
  --valid_path data/rog_processed/webqsp_dev.jsonl \
  --out runs/trajectory_diffusion_webqsp_seed42 \
  --horizon 3 \
  --condition_dropout 0.1 \
  --epochs 20 \
  --seed 42
```

### 3.5 Train the learned fusion head

```bash
python train_fusion_ranker.py \
  --train_path data/rog_processed/webqsp_train.jsonl \
  --valid_path data/rog_processed/webqsp_dev.jsonl \
  --out runs/fusion_ranker_webqsp_seed42 \
  --relation_scorer_ckpt runs/relation_scorer_webqsp_seed42/best.pt \
  --candidate_diffusion_ckpt runs/candidate_diffusion_webqsp_seed42/best.pt \
  --candidate_guidance_scale 1.0 \
  --trajectory_diffusion_ckpt runs/trajectory_diffusion_webqsp_seed42/best.pt \
  --trajectory_guidance_scale 1.0 \
  --ae_ckpt runs/ae_kgqa_torch_real_tune3_noise003/best.pt \
  --planner_ckpt runs/multiseed_cross_infonce_cwq_webqsp/seed_42/mlp_planner/best.pt \
  --value_ckpt runs/final_kgqa_pool48_strong_multiseed/seed_42/value_full_pool_listwise/best.pt \
  --epochs 20 \
  --seed 42
```

## 4. CWQ Transfer and CWQ Retraining

### 4.1 Quick transfer check with the existing WebQSP-trained modules

```bash
python scripts/download_rog_kgqa_data.py --datasets cwq --splits test

python scripts/prepare_rog_kg_env_data.py \
  --datasets cwq \
  --splits test
```

Then evaluate with the same runner you use for WebQSP, swapping the test path to
`data/rog_processed/cwq_test.jsonl`.

### 4.2 Proper CWQ retraining

```bash
python scripts/download_rog_kgqa_data.py --datasets cwq --splits train test

python scripts/prepare_rog_kg_env_data.py \
  --datasets cwq \
  --splits train test \
  --dev_fraction 0.1 \
  --seed 42
```

Then rerun the same four training commands as WebQSP, replacing:

- `webqsp_train.jsonl` -> `cwq_train.jsonl`
- `webqsp_dev.jsonl` -> `cwq_dev.jsonl`
- output prefix `*_webqsp_seed42` -> `*_cwq_seed42`

This is the cleanest way to turn your current CWQ result from a transfer claim into a stronger
same-dataset claim.

## 5. ALFWorld 3000-Trajectory Retraining

This is the cleanest reproducible high-data ALFWorld recipe.

```bash
cd /root/autodl-tmp/DiPLaN
conda activate diplan

export PYTHONPATH=/root/autodl-tmp/DiPLaN
export ALFWORLD_DATA=/root/autodl-tmp/DiPLaN/data/long_horizon/alfworld
export ALFWORLD_CONFIG=$ALFWORLD_DATA/base_config.tw.yaml
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
```

### 5.1 Collect 3000 expert trajectories

```bash
python scripts/collect_alfworld_trajectories.py \
  --data_root "$ALFWORLD_DATA" \
  --config "$ALFWORLD_CONFIG" \
  --episodes 3000 \
  --max_steps 60 \
  --seed 42 \
  --out data/long_horizon/alfworld_processed
```

### 5.2 Train the four modules

```bash
python train_autoencoder_torch.py \
  --config configs/autoencoder_torch_alfworld.json \
  --out runs/alfworld_3000/ae

python train_diffusion_planner_torch.py \
  --config configs/diffusion_torch_alfworld.json \
  --ae_ckpt runs/alfworld_3000/ae/best.pt \
  --out runs/alfworld_3000/diff

python train_value_model_torch.py \
  --config configs/value_torch_alfworld.json \
  --planner_ckpt runs/alfworld_3000/diff/best.pt \
  --out runs/alfworld_3000/value

python train_constraint_model_torch.py \
  --config configs/constraint_torch_alfworld.json \
  --planner_ckpt runs/alfworld_3000/diff/best.pt \
  --out runs/alfworld_3000/constraint
```

### 5.3 Evaluate OOD 134 episodes

```bash
python scripts/run_alfworld_diplan_diffusion.py \
  --data_root "$ALFWORLD_DATA" \
  --config "$ALFWORLD_CONFIG" \
  --split eval_out_of_distribution \
  --episodes 134 \
  --max_steps 50 \
  --seed 42 \
  --use_cuda \
  --ae_ckpt runs/alfworld_3000/ae/best.pt \
  --planner_ckpt runs/alfworld_3000/diff/best.pt \
  --value_ckpt runs/alfworld_3000/value/best.pt \
  --constraint_ckpt runs/alfworld_3000/constraint/best.pt \
  --out results/alfworld_3000_ood134
```

### 5.4 Important ablations

No receding horizon:

```bash
python scripts/run_alfworld_diplan_diffusion.py \
  --data_root "$ALFWORLD_DATA" \
  --config "$ALFWORLD_CONFIG" \
  --split eval_out_of_distribution \
  --episodes 134 \
  --max_steps 50 \
  --seed 42 \
  --use_cuda \
  --ae_ckpt runs/alfworld_3000/ae/best.pt \
  --planner_ckpt runs/alfworld_3000/diff/best.pt \
  --value_ckpt runs/alfworld_3000/value/best.pt \
  --constraint_ckpt runs/alfworld_3000/constraint/best.pt \
  --no_receding \
  --out results/alfworld_3000_no_receding_ood134
```

No value guidance:

```bash
python scripts/run_alfworld_diplan_diffusion.py \
  --data_root "$ALFWORLD_DATA" \
  --config "$ALFWORLD_CONFIG" \
  --split eval_out_of_distribution \
  --episodes 134 \
  --max_steps 50 \
  --seed 42 \
  --use_cuda \
  --ae_ckpt runs/alfworld_3000/ae/best.pt \
  --planner_ckpt runs/alfworld_3000/diff/best.pt \
  --constraint_ckpt runs/alfworld_3000/constraint/best.pt \
  --out results/alfworld_3000_no_value_ood134
```

## 6. Recommended Order

If your budget is limited, run in this order:

1. WebQSP efficiency matched run
2. WebQSP retrain of candidate diffusion + fusion
3. CWQ retraining
4. ALFWorld 3000 retraining

That order gives the strongest paper gain per GPU hour.
