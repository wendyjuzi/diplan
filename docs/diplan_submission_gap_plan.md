# DiPLaN Submission Gap Plan

## Goal

This document converts the current DiPLaN draft into a submission-oriented execution plan. It focuses on one question:

> What is still missing between the current paper and a realistic AAAI/ACL-level submission?

The answer is organized as a claim-to-evidence checklist so that each missing experiment directly supports one specific paper claim.

## 1. Current Paper Status

### 1.1 What is Already Strong

The current paper already has four strong ingredients:

1. A clear problem framing: the Selection-to-Execution Gap.
2. A concrete method package: candidate diffusion, trajectory diffusion, value model, and learned fusion are all implemented and trainable.
3. A strong full-run diagnosis on WebQSP.
4. A strong efficiency story on a controlled FLARE-MCTS comparison subset.

### 1.2 What Is Still Weak

The current weaknesses are not primarily conceptual. They are evidential:

1. The claim scope is broader than the current evidence scope.
2. The method contribution is not yet protected by full ablations.
3. The efficiency comparison is strong but still vulnerable to fairness questions.
4. The current draft should better integrate existing ALFWorld evidence when justifying broader `agent` language beyond KGQA.

## 2. Priority Ranking

### P0: Must-Have Before Submission

1. Full WebQSP module ablation.
2. Formal definition subsection for diagnostic metrics.
3. One fairness-controlled efficiency table.
4. One qualitative case-study figure.

### P1: Strongly Recommended

1. Strengthen one external agent benchmark section: ALFWorld already serves this role, with tau-bench as a possible further extension.
2. Multi-seed or confidence interval reporting for key results.
3. Failure taxonomy table.

### P2: Nice to Have

1. Additional efficiency ablation on candidate pool size or horizon.
2. Cost-quality Pareto curve.
3. Small generalization study across CWQ / WebQSP / another split.

## 3. Claim-to-Evidence Map

## Claim A

### Claim

In KGQA as an execution-constrained environment, long-horizon failure is not only a candidate-recall problem; it is a gap between selecting locally viable actions and achieving successful executed transitions.

### Current Evidence

Already partially supported by:

- full WebQSP gap diagnostics,
- oracle vs executed gap,
- discussion in the current draft.

### What Is Missing

This claim still needs a more formal experimental presentation:

1. exact metric definitions,
2. a main figure visualizing the gap,
3. one qualitative case where selected answer-reaching action still fails later,
4. one paragraph explaining why this is not merely a renaming of action ranking error.

### Experiments / Outputs Needed

1. **Metric Definition Subsection**
   - Add exact definitions for:
     - Executable Action Recall (Pool)
     - Executable Action Recall (Filtered)
     - Executable Action Selection Rate
     - Executed Top-1 Transition Success
     - Selection-to-Execution Gap
   - Clarify that these are post-hoc diagnostics, not inference-time inputs.

2. **Main Gap Figure**
   - Bar chart:
     - Pool
     - Filtered
     - Selected
     - Executed Top-1
   - Add oracle line or second bar group.

3. **Qualitative Case Study**
   - One example where:
     - candidate pool contains answer-reaching action,
     - selected action remains answer-reaching,
     - later execution still fails.

### Table / Figure

1. Table: diagnostic metrics on full WebQSP.
2. Figure: Selection-to-Execution Gap visualization.
3. Case box: one example trajectory.

### Why It Matters

Without this, the paper's main diagnosis remains interesting but not fully "paperized."

## Claim B

### Claim

DiPLaN's gains come from future-aware action decoding inside the legal action space, not just from having a better reranker.

### Current Evidence

Method description is strong, but experimental protection is weak.

### What Is Missing

You need to show which module contributes what. Otherwise reviewers can say:

> Maybe learned fusion alone is doing the work, and diffusion is decorative.

### Experiments / Outputs Needed

1. **Full WebQSP Module Ablation**

Run:

- ToG base
- ToG + question prior / relation scorer
- + candidate diffusion
- + trajectory diffusion
- + value
- + learned fusion
- full DiPLaN

Metrics:

- Hits@1
- trap@1
- Executed Top-1 Transition Success
- Selection-to-Execution Gap
- LLM calls/task if applicable

2. **Gap Reduction Table**

For each ablation, explicitly report:

- Selection Rate
- Executed Top-1 Success
- Gap = Selection - Execution

This is more informative than Hits@1 alone.

### Table / Figure

1. Table: main ablation.
2. Optional line chart: how the gap shrinks as modules are added.

### Why It Matters

This is the single most important experiment for protecting your method contribution.

## Claim C

### Claim

DiPLaN amortizes future reasoning more efficiently than explicit future search.

### Current Evidence

The FLARE-MCTS smoke20 comparison is strong, but reviewers can still question fairness.

### What Is Missing

You need a fairness-controlled efficiency table.

### Experiments / Outputs Needed

1. **Matched-Budget Efficiency Comparison**

Ensure:

- same candidate pool construction,
- same pruning mode,
- same LLM backend,
- same temperature or decoding setup,
- same horizon budget,
- same split.

Compare:

- FLARE-MCTS
- DiPLaN
- optionally DiPLaN-lite

Required framing:

- present this as `offline amortization versus online search`,
- do not phrase it as a blanket claim that `DiPLaN is faster` without qualification.

Metrics:

- Hits@1
- Executed Top-1 Success
- Time/task
- LLM calls/task
- Tokens/task
- Core time/task
- Relation-pruning calls

2. **Variance or CI**

At least one of:

- 3 seeds,
- bootstrap confidence interval,
- repeated subset runs.

### Table / Figure

1. Main efficiency table.
2. Optional Pareto figure: quality vs latency or quality vs tokens.

### Why It Matters

This is what upgrades your efficiency story from "cool result" to "reviewer-resistant result."

## Claim D

### Claim

The paper studies a long-horizon execution problem through KGQA and externally validates it on ALFWorld, rather than claiming full validation on all long-horizon agents.

### Current Evidence

Currently weak if relying only on WebQSP.

### What Is Missing

Either fully surface the existing ALFWorld validation in the paper, or keep a tighter claim scope that explicitly treats WebQSP KGQA as the main validated setting and ALFWorld as external support.

### Experiments / Outputs Needed

Preferred:

1. **tau-bench**
   - strongest for tool-executor story,
   - state/action/executor alignment is natural,
   - easier to justify agent framing.

Fallback:

2. **ALFWorld**
   - weaker for API-executor story,
   - but still stronger than KGQA-only.

### Minimal Acceptable Version

If full benchmark is too expensive:

1. run a representative subset,
2. define executor-aligned diagnostic metrics,
3. show one table and one case study.

### Table / Figure

1. Cross-domain validation table.
2. One environment trajectory case study.

### Why It Matters

This is the difference between:

- "good KG paper with agent language"
- and
- "actual agent paper"

## 4. Concrete Experiment Queue

### Queue 1: Fastest Highest ROI

Run first:

1. Full WebQSP module ablation.
2. Gap-reduction table.
3. Gap visualization figure.

Why:

- lowest engineering risk,
- directly strengthens core method claim,
- immediately makes the paper look more complete.

### Queue 2: Efficiency Defense

Run second:

1. fairness-controlled FLARE-MCTS vs DiPLaN,
2. add CI or multi-seed,
3. produce final efficiency table.

Why:

- protects your strongest practical claim,
- especially useful if reviewers question smoke20 fairness.

### Queue 3: External Validation

Run third:

1. tau-bench if feasible,
2. otherwise ALFWorld.

Why:

- this is what justifies the broader "agent" framing.

## 5. Recommended Final Paper Structure

If you only finish P0:

Use this title style:

> DiPLaN: Execution-Aligned Future Action Decoding for Long-Horizon Knowledge Graph Reasoning

This keeps the paper honest and strong.

If you finish P0 + P1 external benchmark:

Use this broader title style:

> DiPLaN: Execution-Aligned Future Action Decoding for Long-Horizon Agents

This broad title becomes much easier to defend.

## 6. What to Cut If Needed

If page budget becomes tight, cut in this order:

1. long philosophical discussion about textual plans,
2. extra broad claims about general agents,
3. overly long literature exposition.

Do not cut:

1. diagnostic metric definitions,
2. module ablation,
3. efficiency comparison,
4. one qualitative failure case.

## 7. Submission-Readiness Checklist

Mark these as done before submission:

- [ ] Main claim wording matches evidence scope.
- [ ] Diagnostic metrics are formally defined.
- [ ] Full WebQSP module ablation is included.
- [ ] Gap reduction table is included.
- [ ] FLARE-MCTS efficiency table is fairness-controlled.
- [ ] At least one variance estimate or CI is reported.
- [ ] At least one qualitative success/failure case is shown.
- [ ] Limitations explicitly state current evidence scope.
- [ ] Reproducibility details are complete.
- [ ] Title is aligned with actual evidence breadth.

## 8. My Honest Recommendation

If you want the highest probability path:

1. finish full WebQSP ablation first,
2. finish fairness-controlled efficiency table second,
3. if time remains, add tau-bench,
4. only then keep the broader "agent" title.

If time is short:

1. do not overreach,
2. tighten the title to KG reasoning,
3. make the paper airtight on WebQSP.

That version can still be a strong and credible submission.
