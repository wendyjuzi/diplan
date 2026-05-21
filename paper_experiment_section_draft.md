# Experimental Setup

## Tasks and Data
We evaluate DiPLaN on KGQA path planning with CWQ, WebQSP, and GrailQA subsets processed into a unified relation-path prediction format. Each sample contains a tokenized query, an oracle relation path, and per-task constraints (e.g., max steps, banned relations).

We report:
1. Aggregate results over the merged test set (available now).
2. Per-dataset results (required for final submission): **[TBD: CWQ/WebQSP/GrailQA breakdown table]**.

## Metrics
Primary metric:
1. `Success Rate` (exact match between executed path and oracle path).

Secondary diagnostics:
1. `First Error Step`.
2. `Recovery at Error`.
3. `Constraint Violation Rate`.
4. `Plan Feasibility`.
5. `Plan Execution Consistency`.
6. `Token Cost` and `Latency Cost` (system-internal efficiency proxies).

## Protocol and Reproducibility
We use controlled ablations where exactly one module changes at a time. All reported numbers below are from the current seed-42 run. For camera-ready robustness:
1. **[TBD: 3-5 seeds, mean ± std for all main rows]**.
2. **[TBD: confidence intervals / significance test on Success Rate deltas]**.


# Main Results

## Core Claim
Our goal is not to claim SOTA absolute accuracy, but to demonstrate that modular generation can be made reliable only when post-processing is correctly designed. The main empirical finding is that post-processing choices dominate end-to-end performance.

## Aggregate Performance (Current)
| Setting | Success Rate | Notes |
|---|---:|---|
| MLP Direct | 1.55% | Pure generator baseline |
| + Memory (no pre-filter) | 1.98% | Recall improves but constraints collapse |
| + Memory (feasibility pre-filter) | 3.95% | Large gain + violations controlled |
| + Memory pre-filter + BCE Value | 4.94% | Ranking helps |
| + Memory pre-filter + Pairwise Value | **7.21%** | Best current setting |

Interpretation: absolute performance is still moderate, but relative lift from 1.55% to 7.21% (4.7x) is substantial and mechanistically interpretable.

## External Baseline Context (to add)
To position this system-level result, we will add:
1. Prompt-only LLM baseline (few-shot / ReAct): **[TBD: X.XX%]**.
2. Search-style baseline (lightweight ToT/MCTS-like): **[TBD: X.XX%]**.
3. Final comparison table with both accuracy and cost: **[TBD]**.


# Ablation Study

## A1: Latent Robustness and Generator Sanity
We observed that deterministic AE latent decoding was brittle under tiny perturbations. Injecting training-time latent noise (`latent_noise_std`) smoothed the latent manifold and removed repetitive-token collapse under controlled sanity settings.

Noise grid (10-sample sanity):
1. `latent_noise_std=0.03` achieved AE strict reconstruction 1.0 with strong planner-side sanity behavior.
2. Larger noise values preserved robustness but slightly reduced exact AE reconstruction.

## A2: Why Memory Needs Feasibility Filtering
Without pre-filtering, memory recall increases hit chance but introduces massive invalid candidates (`constraint_violation_rate=0.7801`).  
With pre-filtering, invalid memory paths are removed before reranking, yielding:
1. `Success Rate`: 1.98% -> 3.95%.
2. `Constraint Violation Rate`: 78.01% -> 0.0186%.

This isolates memory as a useful but potentially dangerous plugin unless constrained by rule-aware gating.

## A3: Why Pairwise Value Beats BCE Value
Under candidate diversity, pairwise ranking improves over BCE:
1. BCE Value: 4.94%.
2. Pairwise Value: 7.21%.

This supports the hypothesis that in long-horizon candidate pools, comparative supervision is more useful than independent binary classification.


# Error Analysis

## Failure Mode Progression
Our debugging pipeline revealed a staged failure progression:
1. Repetition collapse from brittle latent decoding.
2. Candidate ranking degeneration due to weak value supervision.
3. Constraint-unsafe retrieval pollution from memory module.

After targeted fixes:
1. Repetition collapse is strongly reduced in controlled runs.
2. Constraint violations are near-eliminated with memory pre-filtering.
3. Remaining errors increasingly concentrate on retrieval coverage / long-tail reasoning.

## Final Error Taxonomy (to add)
We will include a structured error table (and optional Sankey-style transition chart):
1. Retrieval miss.
2. Near-miss ranking error.
3. Constraint-filter truncation.
4. Length / step-budget mismatch.
5. Other.

Placeholders:
1. **[TBD: per-type counts and percentages for best model]**.
2. **[TBD: before/after transition counts for key fixes]**.


# Computational Cost

## Current Efficiency Signals
Using internal cost proxies:
1. Direct MLP baseline has low cost (`token_cost~5.41`, `latency_cost~0.02`).
2. Memory-enhanced systems increase cost (`latency_cost~0.195`) but deliver substantial gains.
3. Pairwise value adds moderate compute with the largest accuracy gain among post-processing changes.

## Cost Comparison to Search Baselines (to add)
For final submission we will add wall-clock comparisons:
1. Per-query inference latency (ms).
2. Average candidate expansions / node evaluations.
3. Throughput under fixed hardware.

Placeholders:
1. **[TBD: DiPLaN vs ReAct vs ToT/MCTS latency and throughput table]**.
2. **[TBD: hardware details and reproducibility script reference]**.


# Summary of Experimental Claims
This section supports three claims:
1. Modular generation is viable when latent robustness is enforced.
2. Post-processing is a first-order determinant of end-to-end performance.
3. Constraint-aware memory gating + pairwise value ranking is a practical and effective correction path.

These claims are empirically grounded even before SOTA-level absolute scores, and they motivate a principled roadmap toward stronger final performance.
