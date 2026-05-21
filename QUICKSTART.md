# DiPLaN MVP Quickstart

## Real KGQA Pipeline (CWQ/WebQSP/GrailQA)

### 0) Prepare real data

```bash
python scripts/prepare_real_kgqa_data.py \
  --cwq /path/to/cwq.json \
  --webqsp /path/to/webqsp.json \
  --grailqa /path/to/grailqa.json \
  --out data/real_processed
```

If `grailqa` is temporarily unavailable in your network environment, you can run with CWQ + WebQSP first:

```bash
python scripts/prepare_real_kgqa_data.py \
  --cwq data/official/cwq.json \
  --webqsp data/official/webqsp.json \
  --out data/real_processed
```

### 1) Train torch models

```bash
python train_autoencoder_torch.py \
  --config configs/autoencoder_torch_kgqa.yaml \
  --out runs/ae_kgqa_torch

python train_diffusion_planner_torch.py \
  --config configs/diffusion_torch_kgqa.yaml \
  --ae_ckpt runs/ae_kgqa_torch/best.pt \
  --out runs/diplan_kgqa_torch

python train_value_model_torch.py \
  --config configs/value_torch_kgqa.yaml \
  --planner_ckpt runs/diplan_kgqa_torch/best.pt \
  --out runs/value_kgqa_torch
```

### 2) Evaluate

```bash
python evaluate_torch.py \
  --config configs/eval_torch_kgqa.yaml \
  --ae_ckpt runs/ae_kgqa_torch/best.pt \
  --planner_ckpt runs/diplan_kgqa_torch/best.pt \
  --value_ckpt runs/value_kgqa_torch/best.pt \
  --out results/main_torch
```

### 3) Ablation

```bash
python run_ablation_torch.py \
  --config configs/ablation_torch_kgqa.yaml \
  --ae_ckpt runs/ae_kgqa_torch/best.pt \
  --planner_ckpt runs/diplan_kgqa_torch/best.pt \
  --value_ckpt runs/value_kgqa_torch/best.pt \
  --out results/ablation_torch
```

## Fast Smoke Test (synthetic KGQA)

## 1) Prepare data

```bash
python scripts/prepare_kgqa_data.py --datasets cwq webqsp grailqa --out data/processed
```

## 2) Train models

```bash
python train_autoencoder.py --config configs/autoencoder_kgqa.yaml --out runs/ae_kgqa
python train_diffusion_planner.py --config configs/diffusion_kgqa.yaml --ckpt runs/ae_kgqa/best.pt --out runs/diplan_kgqa
python train_value_model.py --config configs/value_kgqa.yaml --out runs/value_kgqa
python train_constraint_checker.py --config configs/constraint_kgqa.yaml --out runs/constraint_kgqa
```

## 3) Evaluate

```bash
python evaluate.py \
  --config configs/eval_kgqa.yaml \
  --planner_ckpt runs/diplan_kgqa/best.pt \
  --value_ckpt runs/value_kgqa/best.pt \
  --constraint_ckpt runs/constraint_kgqa/best.pt \
  --benchmarks cwq webqsp grailqa \
  --out results/main
```

## 4) Ablation + statistics

```bash
python run_ablation.py --config configs/ablation_kgqa.yaml --out results/ablation
python stats_report.py --in results/main --out results/stats
```
