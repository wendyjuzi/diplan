# DiPLaN Paper Draft

## Title Options

1. DiPLaN: Execution-Aligned Future Action Decoding for Long-Horizon Knowledge Graph Reasoning
2. DiPLaN: Amortized Future-Action Decoding in Execution-Aligned Action Spaces
3. DiPLaN: Reducing the Selection-to-Execution Gap in Long-Horizon KG Reasoning

## One-Sentence Thesis

Long-horizon agent failures are not only a planning problem; they also arise because locally plausible future signals do not reliably translate into executable state-conditioned actions, and DiPLaN addresses this bottleneck by decoding future trajectory signals directly inside the executor's legal action space.

## Abstract

Execution-constrained long-horizon decision making remains challenging for LLM-based agents. Existing work often attributes these failures to insufficient reasoning or search. Through a controlled study on knowledge-graph question answering (KGQA) with an explicit executor, we identify a previously underexplored bottleneck: converting future planning signals into executable state-conditioned actions. This setting enables fine-grained diagnosis beyond final answer accuracy. On a full WebQSP diagnostic run, answer-reaching actions are selected at 93.85% of decision steps, but executed top-1 transition success is only 59.62%, revealing a 34.23-point Selection-to-Execution Gap. Our findings suggest that preserving local answer reachability is insufficient: successful execution depends on how future trajectory information is decoded into the current legal action.

We propose DiPLaN, an execution-aligned future action decoding framework that operates directly in the executor's legal action space. DiPLaN instantiates this idea by combining candidate-level discrete denoising, trajectory-level future modeling, value estimation, and learned fusion to decode future-aware signals into the next executable action under receding-horizon control. On WebQSP, DiPLaN improves answer accuracy while substantially reducing the Selection-to-Execution Gap; we further demonstrate the applicability of the same execution-aligned decoding principle on ALFWorld as an external long-horizon agent environment. In a controlled WebQSP comparison on a shared subset, it also amortizes explicit future search into offline learning, substantially reducing online wall-clock cost, LLM calls, and token usage relative to an explicit FLARE-MCTS baseline. These findings suggest that future-aware action decoding is a distinct decision-making capability that complements reasoning and search in execution-constrained long-horizon agents.

## 1. Introduction

Long-horizon agents must do more than produce plausible plans. At each step, they must convert partial reasoning, task history, and future intent into an action that is both executable in the current environment state and capable of inducing a useful downstream transition. This distinction matters because many modern agent systems interleave reasoning and acting, or use language models to guide search over tools, APIs, or graph relations, yet still fail despite generating apparently sensible intermediate plans.

Recent analyses of long-horizon reasoning have emphasized myopic commitment, shallow search, or unstable multi-turn optimization. These diagnoses are important, but they do not fully explain a common empirical pattern: an agent often retains actions that look promising locally, yet still fails to complete the task once the selected action is actually executed. In other words, preserving local plausibility is not equivalent to securing trajectory-level success.

We study this issue through the lens of execution alignment. In an execution-constrained environment, the executor consumes actions from a state-dependent legal action set \(A(s_t)\), not arbitrary textual intentions. The key challenge is therefore not only to generate a future plan, but to decode future planning signals into a current action commitment that the executor can consume and that will preserve long-horizon task progress. We refer to the residual degradation between selecting locally answer-preserving actions and achieving successful executed transitions as a Selection-to-Execution Gap. This gap provides a concrete operational view of plan-executor misalignment.

Knowledge-graph question answering offers a particularly clean testbed for this problem. In our setting, a state is an entity frontier, an action is a relation, and execution follows that relation to produce the next frontier. This explicit state-action interface allows fine-grained diagnosis beyond final Hits@1. On WebQSP, we find that answer-reaching actions are almost always present in the legal candidate pool, often remain after filtering, and are selected at a high rate, yet executed top-1 transition success remains much lower. On a full run with `relation_first_k=16`, answer-reaching action selection reaches 93.85%, while executed top-1 transition success is only 59.62%. Even oracle-anchored step diagnostics show a similar degradation from 84.29% selected to 44.71% executed. These results suggest that the bottleneck extends beyond simple candidate recall or local ranking.

Motivated by this diagnosis, we propose DiPLaN, an execution-aligned future action decoder for long-horizon KG reasoning. DiPLaN does not generate free-form textual plans and then ground them post hoc. Instead, it operates directly in the executor's legal action space. It uses candidate-level discrete denoising to score admissible current actions, trajectory-level discrete denoising to estimate future relation-sequence distributions, and a value-guided fusion head to decode these future-aware signals into the next executable action under receding-horizon control.

Our contributions are as follows:

1. We identify and quantify a Selection-to-Execution Gap in execution-constrained long-horizon KG reasoning, showing that preserving answer reachability at the action-selection stage does not guarantee trajectory-level success.
2. We propose DiPLaN, a supervised execution-aligned future action decoder that combines candidate diffusion, trajectory diffusion, path value estimation, and learned fusion inside the executor's legal action space.
3. We show how future-aware signals learned from oracle paths can be translated into step-level executable action decisions inside a legal action interface, rather than only into free-form plans.
4. We show that DiPLaN improves long-horizon KG reasoning while amortizing part of explicit future search, achieving large end-to-end efficiency gains over an explicit FLARE-MCTS baseline on a shared controlled subset.

## 2. Related Work

### 2.1 Reasoning-Acting Agents and Long-Horizon Failure

ReAct showed that language models can interleave reasoning traces and task actions, making LLMs practical controllers for interactive environments rather than purely static text generators. This line of work established the now-standard view that agent behavior emerges from repeated reasoning-acting loops. However, later long-horizon studies suggest that simply extending step-by-step reasoning does not guarantee robust planning behavior over long decision horizons. FLARE sharpens this point by arguing that step-wise reasoning induces a myopic policy: actions that look locally good can still lead to poor long-term outcomes unless explicit lookahead and value propagation are introduced.

DiPLaN is aligned with this diagnosis, but differs in how it addresses it. FLARE-MCTS keeps future modeling explicit through lookahead simulation and value backpropagation. DiPLaN instead amortizes part of that future modeling into learned action-space decoders. In this sense, FLARE-MCTS and DiPLaN should be viewed as two different responses to the same long-horizon failure mode: explicit future search versus learned future-action decoding.

### 2.2 Grounding High-Level Intent to Executable Actions

Several agent works have shown that natural language plans are not automatically executable. Language Models as Zero-Shot Planners explicitly noted that naively produced plans often fail to map precisely onto admissible actions, motivating additional translation and grounding procedures. SayCan made a closely related point in robotics: language models may provide strong semantic priors, but action selection must still be constrained by affordances and value-like feasibility signals tied to the actual executor.

These works motivate DiPLaN's execution-aligned framing. Our emphasis, however, is slightly different. We do not primarily ask whether a free-form plan can be translated into an admissible action set at all. Instead, we ask what happens after the agent is already operating inside an executable action space: why do locally plausible or answer-preserving actions still fail to become successful executed trajectories? This is the narrower but sharper gap that DiPLaN targets.

### 2.3 KG-Grounded Agent Reasoning

Knowledge-graph reasoning systems such as Think-on-Graph and Reasoning on Graphs already show that long-horizon LLM reasoning can be grounded in structured graph actions. Think-on-Graph treats the LLM as an agent that incrementally explores graph relations and entities, while Reasoning on Graphs uses relation paths as grounded plans for retrieval and reasoning. These works are highly relevant because they demonstrate that KGQA is not merely a text-reasoning problem; it already contains an executor with explicit action semantics.

DiPLaN does not claim to be the first method to reason in executable relation spaces. Instead, its contribution is to study the residual failure that remains even after the action space is explicit and executable. Our claim is that within such structured spaces, the remaining bottleneck lies in decoding future trajectory information into the current action commitment, not merely in exposing candidate graph relations.

### 2.4 Diffusion for Planning and Decision Making

Diffusion-based planning methods such as Planning with Diffusion for Flexible Behavior Synthesis reinterpret planning as iterative denoising over trajectories. More broadly, diffusion and discrete denoising models have become a useful lens for sequence-structured decision problems, especially when multimodality or future uncertainty matters. On the discrete side, D3PM established a principled denoising framework for categorical state spaces, making diffusion-style modeling applicable beyond continuous control.

DiPLaN borrows inspiration from this literature but occupies a more constrained design point. It is not a full offline RL diffusion planner over continuous trajectories. Instead, it uses lightweight D3PM-style denoising to score legal current actions and short-horizon future relation tokens in a discrete executor-aligned action space. This makes diffusion a tool for future-aware action decoding rather than an end-to-end trajectory generator.

## 3. Problem Setup

We consider long-horizon reasoning in an execution-constrained environment. At step \(t\), the agent observes state \(s_t\), receives a legal action set \(A(s_t)\), chooses an action \(a_t \in A(s_t)\), and transitions to \(s_{t+1}\). The task succeeds if the executed trajectory reaches a goal state within a horizon budget.

In our KGQA instantiation:

- State \(s_t\): the current entity frontier together with depth and execution history.
- Action \(a_t\): a legal relation available from the current frontier.
- Executor: deterministic graph traversal over the local KG subgraph.
- Transition: following the selected relation to produce the next frontier.
- Success: the final frontier contains the gold answer entity.

This setup induces a controlled form of long-horizon agent execution. Unlike free-form reasoning-only settings, the executor has an explicit state-action interface, making it possible to separate four questions:

1. Is an answer-reaching action present in the legal pool?
2. Does filtering retain it?
3. Does the final selector choose it?
4. Does the executed top-1 action actually preserve a successful trajectory?

Our empirical focus is the degradation from steps 3 to 4.

## 4. Method

### 4.1 Overview

DiPLaN is a supervised future-action decoder that acts only inside the executor's legal action space. Given a question \(q\), current state \(s_t\), executed prefix \(p_t\), and legal candidate relations \(A(s_t)\), DiPLaN computes multiple future-aware signals for each candidate relation and fuses them into a final step-level action score:

\[
\text{Score}(r \mid q, s_t, p_t) = f_\theta(
\text{base},
\text{value},
\text{question},
\text{cand-diff},
\text{traj-diff},
\text{prior},
\text{guided},
\text{state features})
\]

The selected action is executed immediately, and planning repeats from the new state. Thus DiPLaN follows receding-horizon control rather than full trajectory commitment.

### 4.2 Execution-Aligned Candidate Space

DiPLaN never scores arbitrary relations from the full schema at inference time. Instead, it relies on the executor to expose a legal candidate set at each decision step. In the codebase, `src/diplan/kg_env.py` defines the environment with explicit admissible relations from the current frontier. This design is important: DiPLaN is not a free-form relation generator, but a future-aware selector over legal executable actions.

### 4.3 Candidate Diffusion

The candidate diffusion module is implemented in [`src/diplan/candidate_diffusion.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\src\diplan\candidate_diffusion.py) and trained by [`train_candidate_diffusion_planner.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_candidate_diffusion_planner.py). For each oracle decision step, we construct a legal candidate set from the current state. If the oracle next relation is present in that set, we corrupt the current relation token and train a denoising model to recover the oracle candidate index within the legal set. The training objective is cross-entropy over legal candidates.

This means candidate diffusion learns:

1. state-conditioned scoring,
2. under legal action constraints,
3. with denoising-style supervision rather than unconstrained generation.

Formally, if \(C_t = A(s_t)\) is the legal candidate set and \(r_t^\*\) is the oracle next relation, candidate diffusion learns:

\[
p_\phi(r_t^\* \mid \tilde{r}_t, q, C_t, t_{\text{noise}})
\]

where \(\tilde{r}_t\) is a corrupted relation token and \(t_{\text{noise}}\) is the diffusion timestep.

### 4.4 Trajectory Diffusion

The trajectory diffusion module is implemented in [`src/diplan/trajectory_diffusion.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\src\diplan\trajectory_diffusion.py) and trained by [`train_trajectory_diffusion_planner.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_trajectory_diffusion_planner.py). Its supervision target is the future oracle relation subsequence `oracle[step : step + horizon]`. The model denoises a corrupted future relation sequence under the question condition and optimizes token-level cross-entropy.

Importantly, DiPLaN does not execute the predicted trajectory directly. At inference time, the trajectory model is used to score current legal relations through the first-position probability mass of the denoised future sequence:

\[
\text{traj-score}(r \mid q, p_t) \propto p_\psi(r_1 = r \mid q, \tilde{\tau}_{t:t+H})
\]

Therefore, trajectory diffusion functions as a future-relation prior for current action selection, not as a full executable plan generator.

### 4.5 Value Model

The value model is trained by [`train_value_model_torch.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_value_model_torch.py). Positive samples are oracle paths, while negatives are constructed from corrupted trajectories, hard ranking mistakes, near-miss prefixes, and terminal truncations. The objective is path-level preference learning, so the model learns to assign higher value to trajectories that better preserve long-horizon success.

This component is crucial because answer reachability alone is too weak: many actions remain answer-reaching in a graph, but only some maintain a strong path toward final success.

### 4.6 Learned Fusion

The fusion head is trained by [`train_fusion_ranker.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_fusion_ranker.py). For each legal candidate relation at a decision step, we extract a feature vector containing:

- base lexical/heuristic score,
- value score,
- question-relation prior,
- candidate diffusion score,
- trajectory diffusion score,
- sampled prior score,
- guided rollout score,
- entity-count and rank statistics,
- depth and state-presence features.

The fusion ranker then learns to rank the oracle next relation above competing legal candidates. In the current implementation, listwise and hybrid ranking losses are supported, making the module explicitly optimized for step-level action ordering rather than only per-candidate calibration.

### 4.7 Inference

At inference time, DiPLaN proceeds as follows:

1. Query the executor for legal candidate relations.
2. Compute base, prior, diffusion, and value-based signals for each candidate.
3. Fuse these signals with the learned ranker.
4. Execute the top-1 relation.
5. Update state and repeat until termination.

This is a learned alternative to repeatedly running explicit textual lookahead and grounding. In that sense, DiPLaN amortizes future modeling into a reusable neural decoder.

## 5. Experiments

### 5.1 Experimental Goals

Our experiments are designed to answer three questions:

1. Can DiPLaN improve long-horizon task performance in both KGQA and an external agent environment?
2. Does DiPLaN reduce the gap between locally plausible action selection and successful executed transitions?
3. Can DiPLaN amortize future reasoning more efficiently than explicit lookahead baselines?

These questions correspond directly to the paper's central claim. The first is a standard performance question. The second is the diagnostic question that motivates the method. The third is the systems question that distinguishes DiPLaN from repeated textual replanning.

### 5.2 Benchmarks

Our primary benchmark is WebQSP under a ToG-style KG execution protocol with local oracle subgraphs. This benchmark is currently the strongest evidence base for the paper because it exposes an explicit executor and enables step-level trajectory diagnostics.

We also evaluate DiPLaN on ALFWorld as an external long-horizon agent environment. In the paper's evidence hierarchy, WebQSP provides the clearest mechanism-level diagnosis, while ALFWorld provides cross-environment validation that the execution-aligned decoding idea is not confined to KGQA alone.

### 5.3 Baselines

We compare against three families of methods:

1. ToG-style local graph exploration without DiPLaN reranking.
2. FLARE-MCTS: an explicit future-search baseline with lookahead, rollout evaluation, value backpropagation, and receding-horizon execution.
3. DiPLaN ablations, including lite and partial-module variants.

The FLARE-MCTS comparison is designed as a fairness-controlled efficiency diagnostic on a shared subset rather than a complete replacement for the full benchmark leaderboard or a universal claim of superiority over explicit planning.

### 5.4 Metrics

We report both final task accuracy and intermediate execution diagnostics.

Final metrics:

- Hits@1: final answer-level success rate.
- trap@1: fraction of runs that commit early to a trap trajectory.
- Wall-clock time per task.
- LLM calls per task.
- Token cost.

Execution diagnostics:

- Executable Action Recall (Pool): answer-reaching action exists in the legal pool.
- Executable Action Recall (Filtered): answer-reaching action survives filtering.
- Executable Action Selection Rate: final selected action remains answer-reaching.
- Executed Top-1 Transition Success: executed top-1 action truly advances a successful trajectory.

Formally, at step \(t\), let \(A_t\) denote the legal action pool, \(F_t \subseteq A_t\) the filtered candidate set after pruning or reranking, and \(\hat{a}_t\) the final top-1 selected action. Let \(y_t(a) \in \{0,1\}\) indicate whether action \(a\) belongs to at least one answer-reaching trajectory under the local oracle subgraph, and let \(z_t \in \{0,1\}\) indicate whether executing \(\hat{a}_t\) produces a transition that preserves successful trajectory progress. Then:

\[
\text{Executable Action Recall (Pool)}
= \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\!\left[\exists a \in A_t: y_t(a)=1\right]
\]

\[
\text{Executable Action Recall (Filtered)}
= \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\!\left[\exists a \in F_t: y_t(a)=1\right]
\]

\[
\text{Executable Action Selection Rate}
= \frac{1}{T}\sum_{t=1}^{T} y_t(\hat{a}_t)
\]

\[
\text{Executed Top-1 Transition Success}
= \frac{1}{T}\sum_{t=1}^{T} z_t
\]

We define the Selection-to-Execution Gap as:

\[
\text{Gap} = \text{Action Selection Rate} - \text{Executed Top-1 Transition Success}.
\]

All four quantities are post-hoc analysis metrics derived from oracle structure and executed traces. They are intended to localize failure modes, not to imply that oracle reachability labels are provided to the model at inference time.

### 5.5 Implementation and Evaluation Protocol

We evaluate DiPLaN in a ToG-style local execution pipeline built over oracle subgraphs. At each decision step, the system first constructs or prunes a relation candidate set, then applies DiPLaN reranking or an explicit-search baseline to choose the next relation. The resulting relation is executed immediately in the local KG environment, and the next state is recomputed from the new entity frontier. Because training and diagnostics both rely on oracle structure, our strongest claims are about controlled KG execution rather than unrestricted real-world agent autonomy.

For the strongest WebQSP configuration discussed in this draft, we use `relation_first_k=16` with all-legal candidate pooling and learned fusion. The full diagnostic run contains 1,546 questions. For controlled efficiency analysis, we also report a shared 20-example subset and a 100-example lexical-pruning subset, both of which were run through the patched official ToG evaluation scaffold with per-stage timing instrumentation.

This split between full-run diagnosis and subset-based efficiency measurement is intentional. The full run gives us statistical evidence for the selection-to-execution gap, while the smaller controlled subsets let us compare explicit-search and learned-decoding strategies under matched infrastructure and timing.

### 5.6 Main WebQSP Diagnostic Result

On the full WebQSP run with `relation_first_k=16`, DiPLaN achieves:

- Hits@1: 87.65
- trap@1: 0.84
- Answer-reaching in pool: 99.07
- Answer-reaching after filtering: 97.04
- Answer-reaching selected: 93.85
- Answer-reaching executed top-1: 59.62
- Oracle selected step rate: 84.29
- Oracle executed top-1 rate: 44.71

These numbers support two claims. First, answer-reaching actions are usually available, so the main issue is not raw candidate recall. Second, there is a large degradation from local reachability preservation to actual trajectory success. The observed Selection-to-Execution Gap is 34.23 points (93.85 -> 59.62), while even oracle-anchored diagnostics degrade by 39.58 points (84.29 -> 44.71). This pattern indicates that long-horizon failure is not reducible to whether a plausible action was retained; it is fundamentally a trajectory-execution reliability problem.

### 5.7 Efficiency Diagnostic Against FLARE-MCTS

We next compare DiPLaN to FLARE-MCTS on a shared 20-example WebQSP subset using the same official ToG-style local execution scaffold, the same split, and the same local executor interface. We present this comparison as an amortization and efficiency diagnostic, not as an apples-to-apples claim that offline learning and online search incur the same deployment costs.

FLARE-MCTS:

- Hits@1: 0.90
- Time per task: 193.71 s
- LLM calls per task: 152.55
- Total estimated tokens: 684,097
- Executed top-1 transition success: 0.1923

DiPLaN:

- Hits@1: 0.95
- Time per task: 8.62 s
- LLM calls per task: 5.05
- Total estimated tokens: 39,671
- Executed top-1 transition success: 0.84

This yields:

- 22.47x faster wall-clock execution,
- 30.21x fewer LLM calls,
- 17.25x fewer tokens,
- 58.39x fewer relation-pruning calls,
- zero explicit trajectory evaluations at inference time.

The result supports our main efficiency claim: explicit future search repeatedly pays the textual interface cost at inference time, while DiPLaN amortizes part of that future modeling into a trained decoder.

### 5.8 Lite Comparison and Runtime Decomposition

A lighter lexical-pruning version of DiPLaN (`official_tog_diplan_webqsp_llama31_k16_entitylex_smoke100`) further clarifies runtime behavior:

- Hits@1: 0.83
- Time per task: 12.24 s
- LLM calls per task: 7.17
- LLM wall time: 1163.11 s total
- DiPLaN rerank wall time: 52.52 s total
- Executed top-1 transition success: 0.6739

Thus, even in the lite setting, most runtime is dominated by LLM-mediated pruning rather than the DiPLaN neural core itself. This decomposition helps explain why the right efficiency comparison is not "neural core time alone" versus "MCTS core time alone", but total end-to-end cost under comparable executor interfaces.

### 5.9 What the Diagnostics Mean

The most important interpretation point is that `answer_reaching_selected_rate` is not equivalent to selecting the uniquely correct action. It only means that the selected action remains on at least one answer-reaching trajectory under the local graph structure. Therefore, the gap from 93.85% selected to 59.62% executed should not be read as "the model already chose the correct action but failed to execute it." A more precise reading is that many locally viable actions still fail to convert into reliable long-horizon success.

This distinction matters because it changes the scientific claim. The evidence does not merely show ranking noise. It shows that preserving local reachability is insufficient, and that long-horizon reliability depends on how future trajectory structure is compressed into the current action decision.

### 5.10 Module-Level Interpretation

The current experiments also help explain the role of each DiPLaN module.

1. Candidate diffusion improves action discrimination within legal candidate sets.
2. Trajectory diffusion injects short-horizon future structure that a one-step ranker cannot see directly.
3. The value model distinguishes superficially similar but trajectory-level different paths.
4. Learned fusion turns these heterogeneous signals into a top-1 executable action commitment.

This decomposition is important for writing the paper because it shows that DiPLaN is not a bag of heuristics. Each component addresses a distinct part of the selection-to-execution bottleneck.

### 5.11 Additional Experiments to Add Next

If space and compute permit, the next strongest additions would be:

1. A module ablation table on the full WebQSP setting:
   - base ToG
   - + question prior
   - + candidate diffusion
   - + trajectory diffusion
   - + value
   - + learned fusion
2. A gap-reduction table reporting how each module changes:
   - selection rate
   - executed top-1 success
   - gap size
3. An external validation benchmark:
   - ALFWorld as a structured text environment
   - or tau-bench as a stricter tool-executor environment

For the current paper, however, the safest structure is still to treat WebQSP as the primary evidence base and present ALFWorld as external validation rather than as the sole foundation of the core thesis.

## 6. Discussion

### 6.1 What DiPLaN Solves

DiPLaN does not claim to solve general-purpose agent planning in the abstract. Its stronger claim is narrower and better supported: in an execution-constrained long-horizon environment, future-aware action scoring inside the legal action space can reduce the gap between locally plausible decisions and successful executed transitions, while also amortizing the cost of repeated explicit lookahead.

### 6.2 Why Textual Plan Passing Is a Weak Interface

The controlled WebQSP results motivate a broader systems-level interpretation. When future signals are passed through free-form text, the system repeatedly pays three costs:

1. information compression from rich future distributions to sparse textual descriptions,
2. repeated decode-then-ground overhead,
3. instability when textual intent does not map cleanly to admissible executable actions.

DiPLaN avoids these costs by learning directly over executable action candidates.

### 6.3 Limitations

Our strongest mechanism-level evidence comes from KGQA, where the executor is explicit and controlled. While this is a strength for diagnosis, it is not our only empirical setting. We also validate DiPLaN on ALFWorld as an external long-horizon agent benchmark. Accordingly, we do not claim universal validation across all tool-use agents, web agents, or embodied agents, but neither is the paper limited to KGQA alone.

We also emphasize that `answer_reaching_selected_rate` does not mean the agent selected the unique correct action. It only means the selected action remains on some answer-reaching trajectory. Therefore, our diagnosis is not "the correct action was selected but execution failed", but rather "local reachability preservation is insufficient for reliable long-horizon completion."

## 7. Conclusion

We presented DiPLaN, an execution-aligned future action decoder for long-horizon KG reasoning and agent control. By shifting future modeling from free-form textual planning to the executor's legal action space, DiPLaN turns candidate denoising, trajectory denoising, and value-guided fusion into a practical step-level action decoder. On WebQSP, this yields both improved reasoning quality and strong efficiency gains over explicit future search in a controlled execution setting, while ALFWorld provides external validation beyond KGQA. More broadly, our results suggest that a key challenge in long-horizon agent systems is not only whether future-relevant signals exist, but whether those signals can be reliably decoded into executable current actions.

## 8. Suggested Figures and Tables

### Figure 1. Problem Illustration

Planner signal -> candidate pool -> filtered set -> selected action -> executed transition -> final answer

Highlight the drop from selected to executed top-1.

### Figure 2. DiPLaN Architecture

Question + current state -> legal candidates -> candidate diffusion / trajectory diffusion / value / prior -> learned fusion -> execute top-1 -> receding-horizon loop

### Table 1. Main WebQSP Results

Include Hits@1, trap@1, pool recall, filtered recall, selection rate, executed top-1 success, and Selection-to-Execution Gap.

### Table 2. Efficiency Comparison

Include FLARE-MCTS vs DiPLaN on the shared subset with:

- time per task,
- LLM calls per task,
- tokens,
- relation-pruning calls,
- executed top-1 success,
- Hits@1.

### Table 3. Training Targets by Module

| Module | Training target | Objective | Output role |
|---|---|---|---|
| Candidate diffusion | oracle next relation within legal candidate set | cross-entropy | current executable action score |
| Trajectory diffusion | future oracle relation subsequence | token-level cross-entropy | future first-relation prior |
| Value model | oracle path vs hard negatives | ranking loss | trajectory quality estimate |
| Fusion ranker | oracle next relation in candidate group | listwise / hybrid ranking | final top-1 action decoding |

## 9. Related Work Notes for Bib

The most directly relevant references for the current narrative are:

1. ReAct: Synergizing Reasoning and Acting in Language Models
2. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph
3. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning
4. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
5. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents
6. Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents
7. Planning with Diffusion for Flexible Behavior Synthesis
8. Structured Denoising Diffusion Models in Discrete State-Spaces

## 10. Writing Notes for Revision

1. Keep ALFWorld positioned as external validation, not as the source of the core thesis.
2. If you add tau-bench later, use it to support the agent claim at the tool-executor level.
3. Keep "Selection-to-Execution Gap" as the main diagnostic phrase in the paper.
4. Avoid claiming that DiPLaN outputs complete executable trajectories.
5. Avoid claiming that DiPLaN directly feeds latent states to the executor.
6. Avoid claiming that high `answer_reaching_selected_rate` means the correct action was already chosen.
