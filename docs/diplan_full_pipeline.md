# DiPLaN-complete pipelines (KGQA + ALFWorld)

This is the "truly implements the report" variant. It adds the three pieces the base
KGQA stack was missing and builds the full diffusion stack for ALFWorld.

What was added vs. the base code:
- **Decision-weighted diffusion** (report §7.3.2): `WeightedDiffusionDataset` weights each
  plan by `w_i = exp(α·Normalize(R_i))·1{feasible}` (`src/diplan/torch_pipeline.py`).
- **Learned constraint model `C_ξ`** (§5.3/§7.3.3): `ConstraintModel` +
  `train_constraint_model_torch.py`, applied as a rerank penalty in `evaluate_torch.py`.
- **Prefix-conditioned planning state** (§5.1/§5.5): executed prefix packed into the planner
  condition stream via `<sep>`; vocab unioned in `build_vocabs(prefix_conditioning=True)`.
- **Feasibility projection** (§5.4): `constraints.project_path_to_valid`, wired into
  `_generate_candidates`.
- **Shared modules**: `src/diplan/constraints.py` (one feasibility checker),
  `src/diplan/inference.py` (reusable sampling/decoding), `src/diplan/alfworld_plan.py`
  (plan-token normalizer + tool decoder).

All KGQA changes are **flag-gated and default-off**, so existing configs/ckpts are unchanged.

## Route A — KGQA (CPU-ok, GPU faster)

```bash
python train_autoencoder_torch.py       --config configs/autoencoder_torch_kgqa.diplan_full.json --out runs/kgqa_full/ae
python train_diffusion_planner_torch.py  --config configs/diffusion_torch_kgqa.diplan_full.json   --ae_ckpt runs/kgqa_full/ae/best.pt      --out runs/kgqa_full/diff
python train_value_model_torch.py        --config configs/value_torch_kgqa.diplan_full.json       --planner_ckpt runs/kgqa_full/diff/best.pt --out runs/kgqa_full/value
python train_constraint_model_torch.py   --config configs/constraint_torch_kgqa.diplan_full.json  --planner_ckpt runs/kgqa_full/diff/best.pt --out runs/kgqa_full/constraint
python evaluate_torch.py --config configs/eval_torch_kgqa.diplan_full.json \
    --ae_ckpt runs/kgqa_full/ae/best.pt --planner_ckpt runs/kgqa_full/diff/best.pt \
    --value_ckpt runs/kgqa_full/value/best.pt --constraint_ckpt runs/kgqa_full/constraint/best.pt \
    --out results/kgqa_diplan_full
```

Ablations (toggle in the eval config): `prefix_conditioning`, `use_constraint_model`,
`feasibility_projection_enabled`, `value_guided_sampling`; set diffusion
`decision_weighting:false` for the α=0 ablation.

> Note: AE, planner, value and constraint models **must be trained together** for this
> variant because they share the prefix-unioned vocab (`prefix_conditioning:true`).

## Route B — ALFWorld (must run on the ALFWorld/GPU server)

```bash
export ALFWORLD_DATA=/root/autodl-tmp/DiPLaN/data/long_horizon/alfworld

# 1) Collect handcoded-expert trajectories (train split only exposes the expert plan).
python scripts/collect_alfworld_trajectories.py \
    --data_root "$ALFWORLD_DATA" --config "$ALFWORLD_DATA/base_config.tw.yaml" \
    --episodes 3000 --max_steps 60 --out data/long_horizon/alfworld_processed

# 2) Train the stack on the collected plans (same task-agnostic trainers).
python train_autoencoder_torch.py      --config configs/autoencoder_torch_alfworld.json --out runs/alf/ae
python train_diffusion_planner_torch.py --config configs/diffusion_torch_alfworld.json   --ae_ckpt runs/alf/ae/best.pt      --out runs/alf/diff
python train_value_model_torch.py       --config configs/value_torch_alfworld.json       --planner_ckpt runs/alf/diff/best.pt --out runs/alf/value
python train_constraint_model_torch.py  --config configs/constraint_torch_alfworld.json  --planner_ckpt runs/alf/diff/best.pt --out runs/alf/constraint

# 3) Run the diffusion DiPLaN agent (receding-horizon, real metrics).
python scripts/run_alfworld_diplan_diffusion.py \
    --data_root "$ALFWORLD_DATA" --config "$ALFWORLD_DATA/base_config.tw.yaml" \
    --split eval_out_of_distribution --episodes 20 --max_steps 50 --use_cuda \
    --ae_ckpt runs/alf/ae/best.pt --planner_ckpt runs/alf/diff/best.pt \
    --value_ckpt runs/alf/value/best.pt --constraint_ckpt runs/alf/constraint/best.pt \
    --out results/alfworld_diplan_diffusion_ood20
```

The executor reports **measured** `plan_feasibility` (projectable heads / steps),
`constraint_violation_rate`, and `plan_execution_consistency` — not the hardcoded `1.0`s of
the old heuristic `scripts/run_alfworld_diplan_agent.py`, and `candidate_pool_avg_size` is now
the number of diffusion candidates, not the admissible-command count.
```
