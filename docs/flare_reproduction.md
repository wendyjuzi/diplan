# FLARE reproduction + DiPLaN-diffusion variant (KGQA)

Faithful reproduction of **FLARE** ("Why Reasoning Fails to Plan") as a baseline, plus a
variant that swaps FLARE's MCTS planner for the repo's **diffusion planner** inside the
*identical* environment + evaluative signal. Only the action-selection mechanism changes —
the paper's comparison axis.

## What's implemented
- **`src/diplan/kg_env.py`** — genuine KG traversal env over RoG subgraphs: `KGEnv`
  (`admissible_relations`=A(s), `neighbors`=T(s,a), `answer_reached`=Hits@1), oracle-path
  alignment with BFS repair, and faithful myopic-trap construction (paper appendix E.5).
- **`src/diplan/planners.py`** — one `select_action(env, state, ctx)` interface, 5 planners:
  `single_step`, `beam` (B=8), `lookahead` (k=2), `flare` (MCTS Algorithm 1: UCB selection,
  φ action-pruning k=8, depth-H=3 rollout, trajectory-level eval + `TrajectoryMemory`
  (M=200, δ=0.9), backprop, S=16 sims, c=1.4), and `diplan_diffusion` (diffusion candidate
  generation grounded to the admissible set; same env/signal).
- **`src/diplan/llm_client.py` / `kgqa_prompts.py`** — served OpenAI-compatible LLM as the
  evaluative signal r̂ and proposal φ, with caching and graceful lexical fallback. A `stub`
  lexical scorer is available for offline mechanics validation.
- **`scripts/prepare_rog_kg_env_data.py`** — re-process RoG into adjacency-preserving JSONL.
- **`scripts/run_kgqa_planning_eval.py`** — ToG-style episode loop + metrics
  (Hits@1, Trap@1, First-Error Step, Recovery@First-Error).

## Metrics
- **Hits@1** = answer entity reached in the subgraph (graph-grounded).
- **Trap@1** = selected the constructed myopic trap at step 1 (computed only on rows that have a trap).
- **First-Error Step** / **Recovery@First-Error** = vs the BFS-aligned oracle path.

## End-to-end run

### 1. Download RoG subgraphs (needs `datasets`; one-time)
```bash
pip install -U datasets pyarrow
python scripts/download_rog_kgqa_data.py --datasets webqsp --splits test
# CWQ (large): python scripts/download_rog_kgqa_data.py --datasets cwq --splits test
```

### 2. Re-process into adjacency JSONL (local, no LLM)
```bash
python scripts/prepare_rog_kg_env_data.py --datasets webqsp --splits test
# -> data/rog_processed/webqsp_test.jsonl   (reports align_rate; ~0.94 on WebQSP)
```

### 3a. Offline mechanics check (stub scorer, no endpoint)
```bash
python scripts/run_kgqa_planning_eval.py --config configs/eval_kgqa_planning.json \
    --out results/kgqa_planning_stub \
    --ae_ckpt runs/ae_kgqa_torch_real_tune3/best.pt \
    --planner_ckpt runs/diplan_kgqa_torch_real_tune3/best.pt
```

### 3b. Faithful run with the served LLM
Start a local OpenAI-compatible server (vLLM/Ollama) at `http://127.0.0.1:8000/v1`, set
`llm_model` in `configs/eval_kgqa_planning.llm.json`, then:
```bash
# tiny validation first (S=4, 20 rows) — confirm diag.llm_errors≈0
python scripts/run_kgqa_planning_eval.py --config configs/eval_kgqa_planning.llm.json \
    --out results/kgqa_planning_llm \
    --ae_ckpt runs/ae_kgqa_torch_real_tune3/best.pt \
    --planner_ckpt runs/diplan_kgqa_torch_real_tune3/best.pt
# then scale: set "scorer":"llm", flare.S=16, max_tasks=0 in the config, full WebQSP/CWQ.
```
Outputs `summary_metrics.json`, `summary_by_dataset.json`, `predictions.jsonl`, `diagnostics.json`.

## Known limitations / next steps
- **Diffusion grounding ~5%.** The available diffusion planner (`diplan_kgqa_torch_real_tune3`)
  collapses to a single confident-but-wrong relation per question, so its first-hop relations
  rarely match the subgraph's admissible set (falls back to nearest-admissible). For a
  competitive DiPLaN-vs-FLARE result the diffusion planner should be **retrained to propose
  first-hop relations conditioned on the admissible set**. The harness already grounds and logs
  the hit-rate (`diagnostics.json`).
- **GrailQA deferred** — not in RoG; needs a separate subgraph source (add `from_grailqa_row`).
- **ALFWorld** (paper's tool-use setting) — separate follow-up; runs on the autodl server.
- Oracle paths come from in-subgraph BFS (RoG rows carry no relation chain); Hits@1 is robust to
  this, First-Error/Recovery are relative to that BFS oracle.
# Question-Conditioned Relation Retrieval

The ToG-DiPLaN runner separates three signals:

1. `QueryRelationScorer`: contrastively trained estimate of
   `P(relation | question)`.
2. AE trajectory prior: projection from generated future paths onto ToG's
   legal candidate relations.
3. Value ranker: downstream trajectory-quality estimate.

This separation follows the retrieval-then-reasoning pattern used by KGQA
systems such as UniKGQA and RoG, while preserving FLARE's future-aware action
selection principle. The learned relation scorer should be trained only on the
training split; never train it on WebQSP/CWQ test questions.

Train:

```bash
python train_relation_scorer_torch.py \
  --train_path data/rog_processed/webqsp_train.jsonl \
  --valid_path data/rog_processed/webqsp_dev.jsonl \
  --out runs/relation_scorer_webqsp_seed42 \
  --seed 42
```

Pass the resulting checkpoint to the official-ToG adapted runner:

```bash
--diplan_relation_scorer_ckpt \
  /root/autodl-tmp/DiPLaN/runs/relation_scorer_webqsp_seed42/best.pt
```

Recommended ablations:

```text
tog_only       ToG proposal score
question_only  learned question-relation retrieval
prior_only     question prior + AE trajectory prior
value_only     trajectory value only
prior_value    question/AE prior + trajectory value
fused          ToG + question/AE prior + trajectory value
```

Report the full decision funnel in addition to Hits@1:

```text
oracle_in_pool_rate
oracle_in_keep_rate
oracle_selected_step_rate
oracle_executed_top1_rate
oracle_step_rank_before_mean
oracle_step_rank_after_mean
```

Literature basis:

- UniKGQA (ICLR 2023): question-relation semantic matching shared by retrieval
  and reasoning, https://arxiv.org/abs/2212.00959
- RoG (ICLR 2024): explicit relation paths as plans in a
  planning-retrieval-reasoning pipeline, https://arxiv.org/abs/2310.01061
- ToG: interactive KG exploration with beam search,
  https://arxiv.org/abs/2307.07697
- RE-KBQA: additional relation supervision and relation-guided reranking,
  https://arxiv.org/abs/2305.02118
- Why Reasoning Fails to Plan / FLARE: future-aware evaluation and limited
  commitment, https://arxiv.org/abs/2601.22311
