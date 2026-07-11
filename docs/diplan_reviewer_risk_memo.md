# DiPLaN Reviewer-Risk Memo

This memo converts a harsh AAAI/NeurIPS-style review into concrete revision actions for the current DiPLaN paper package.

## Bottom Line

If the current draft were submitted as-is, the most likely outcome is `Weak Reject` or `Borderline Reject`, not because the implementation looks weak, but because the paper currently over-claims relative to its evidence.

The core repair strategy is:

1. narrow the claim scope to KGQA as a representative execution-constrained environment,
2. make the conceptual novelty about `execution-aligned future action decoding`, not about `diffusion modules`,
3. show that the named gap is diagnostically distinct and empirically reduced,
4. neutralize fairness and oracle-supervision objections before reviewers raise them.

## The Five Highest-Risk Reject Points

### 1. Claim Scope Is Still Too Broad

Current risky framing:

- `long-horizon agents`
- `long-horizon reasoning`
- `practical alternative ... in long-horizon agents`

Why this gets attacked:

- the evidence base is still overwhelmingly WebQSP KGQA,
- reviewers can dismiss the paper with `only validated on KGQA`.

Required fix:

- make KGQA the primary claim surface,
- describe it as `one representative execution-constrained environment with an explicit executor`,
- move broad agent language into motivation and limitation paragraphs, not into the main empirical claim.

Safe replacement:

> We study execution-aligned future action decoding in KGQA as a controlled execution-constrained setting, and use it to isolate a measurable selection-to-execution failure mode.

### 2. The Named Gap Can Be Read as a Renaming of Old Problems

Risk:

- `Selection-to-Execution Gap` may be read as action ranking error, greedy failure, or horizon mismatch under a new name.

Required fix:

- explicitly state what existing diagnostics measure and what they miss,
- define the gap operationally using post-hoc metrics,
- show that the paper is not merely renaming `ranking failure`.

Needed text move:

- add a paragraph saying candidate recall, filtering retention, and final Hits@1 do not measure whether a selected action reliably preserves successful executed transitions.

Needed experiment:

- a before/after gap table across ablations,
- not just final Hits@1.

### 3. The Method Currently Looks Like Engineering Composition

Risk:

- candidate diffusion + trajectory diffusion + value + fusion can be read as a stitched pipeline of existing modules.

Required fix:

- define the novelty at the level of problem interface and decoding objective,
- make the modules subordinate to the idea of `future-action decoding in the executor's legal action space`.

Safe framing:

> DiPLaN is not novel because it contains several modules. It is novel to the extent that it operationalizes future-aware action decoding directly inside the executor interface and shows that this reduces a measurable selection-to-execution failure mode.

Implication for writing:

- reduce module-list storytelling,
- increase emphasis on why future signals must be decoded into current legal actions.

### 4. FLARE-MCTS Comparison Is Vulnerable on Fairness

Risk:

- `DiPLaN is faster` is too easy to dismiss because one method amortizes planning offline and the other pays online search cost.

Required fix:

- rewrite the claim as amortization, not raw superiority,
- clearly state that the comparison is a matched-infrastructure efficiency diagnostic on a subset,
- add one fairness checklist table if possible.

Safe wording:

> DiPLaN amortizes part of explicit future search into offline supervised learning, yielding much lower online inference cost under a matched local execution scaffold.

Unsafe wording:

- `DiPLaN is faster than FLARE`
- `DiPLaN is a superior replacement for explicit planning`

### 5. Oracle Supervision Objection Must Be Addressed Up Front

Risk:

- reviewers will object that oracle paths are unrealistic and reduce practical applicability.

Required fix:

- acknowledge that training is oracle-supervised,
- justify it as controlled offline supervision consistent with prior KG path supervision and imitation-style setups,
- avoid presenting this as direct evidence of broad real-world autonomy.

Safe framing:

> Oracle trajectories are used as supervised training signals to study whether future-aware executable action decoding can be learned at all in a controlled setting.

## Concrete Draft Surgery

### Must Change Immediately

1. Remove `code-grounded formulation` as a named contribution.
2. Avoid presenting implementation transparency as a scientific contribution.
3. Replace broad `long-horizon agents` claims with KGQA-scoped claims wherever evidence is only WebQSP.
4. Replace `DiPLaN is 22.47x faster` with amortization language.
5. Add an explicit limitation sentence wherever efficiency results are reported.

### Strongly Recommended

1. Add a short subsection: `Why this is not just action ranking error`.
2. Add a short subsection: `Why oracle supervision is still scientifically useful here`.
3. Add a gap-reduction table across ablations:
   - selection rate
   - executed top-1 transition success
   - gap
4. Add one fairness checklist for FLARE-MCTS:
   - same split
   - same backend
   - same candidate construction
   - same horizon budget
   - same execution scaffold

## Revised Contribution Shape

The paper is strongest when the contributions are:

1. a diagnostic contribution:
   - define and quantify a measurable degradation from selected answer-reaching actions to successful executed transitions in KGQA;
2. a method contribution:
   - propose a supervised execution-aligned future action decoder operating inside the legal action space;
3. an empirical contribution:
   - show better executed transition reliability and lower online planning cost than explicit future search on controlled WebQSP comparisons.

The paper is weaker when the contributions are written as:

- we built several modules,
- we explain our code,
- we are generally about long-horizon agents.

## Submission Standard

If the next revision only rewrites prose, the decision profile likely stays around `Weak Reject`.

To move toward `Borderline Accept`, the minimum package should include:

1. claim-scope surgery,
2. contribution rewrite,
3. gap visualization,
4. gap-reduction ablation,
5. fairness-controlled FLARE table,
6. explicit oracle-supervision justification.
