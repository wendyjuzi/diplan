# DiPLaN: Execution-Aligned Future Action Decoding for Long-Horizon Agents

## Abstract

Execution-constrained long-horizon decision making remains challenging for LLM-based agents. Existing work often attributes these failures to insufficient reasoning or search. Through a controlled study on knowledge-graph question answering (KGQA) with an explicit executor, we identify a previously underexplored bottleneck: converting future planning signals into executable state-conditioned actions. This setting enables fine-grained diagnosis beyond final answer accuracy. On a full WebQSP diagnostic run, answer-reaching actions are selected at 93.85% of decision steps, but executed top-1 transition success is only 59.62%, revealing a 34.23-point Selection-to-Execution Gap. Our findings suggest that preserving local answer reachability is insufficient: successful execution depends on how future trajectory information is decoded into the current legal action.

We propose DiPLaN, an execution-aligned future action decoding framework that operates directly in the executor's legal action space. DiPLaN instantiates this idea by combining candidate-level discrete denoising, trajectory-level future modeling, value estimation, and learned fusion to decode future-aware signals into the next executable action under receding-horizon control. On WebQSP, DiPLaN improves answer accuracy while substantially reducing the Selection-to-Execution Gap; we further demonstrate the applicability of the same execution-aligned decoding principle on ALFWorld as an external long-horizon agent environment. In a controlled WebQSP comparison on a shared subset, DiPLaN also amortizes explicit future search into offline learning, substantially reducing online wall-clock cost, LLM calls, and token usage relative to an explicit FLARE-MCTS baseline. These findings suggest that future-aware action decoding is a distinct decision-making capability that complements reasoning and search in execution-constrained long-horizon agents.

## 1. Introduction

Long-horizon agents must do more than produce plausible plans. At every step, they must convert partial reasoning, task history, and future intent into an action that is executable in the current environment state and that preserves progress toward eventual success. This distinction matters because many modern agent systems already interleave reasoning and acting, or use language models to search over tools, APIs, or structured knowledge sources, yet still fail despite generating seemingly sensible intermediate plans [Yao et al., 2022; Sun et al., 2023].

A common explanation for these failures is that the underlying model does not reason deeply enough, search broadly enough, or rank actions accurately enough. While these factors certainly matter, they do not fully explain an empirical pattern that repeatedly appears in long-horizon settings: an agent often retains actions that look plausible locally, yet still fails once the selected action is actually executed. In other words, local plausibility does not guarantee trajectory-level success.

We study this issue through the lens of execution alignment. In an execution-constrained environment, the executor consumes actions from a state-dependent legal action set \(A(s_t)\), not arbitrary textual intentions. The problem is therefore not only to produce a plan, but to decode future-relevant signals into a current action commitment that the executor can actually consume. This view shifts attention from free-form plan quality to the interface between planning and execution.

Knowledge-graph question answering provides a particularly clean setting for studying this issue. In our environment, a state is the current entity frontier, an action is a legal relation, and execution follows that relation to produce the next frontier. Because the executor is explicit, we can decompose long-horizon failure into candidate availability, filtering, final action selection, and executed transition success. This decomposition reveals a consistent gap. On WebQSP, answer-reaching actions are almost always present in the legal pool, frequently survive filtering, and are selected at a high rate, yet executed top-1 transition success remains much lower. In our strongest full-run diagnosis with `relation_first_k=16`, answer-reaching action selection reaches 93.85%, but executed top-1 transition success drops to 59.62%. Even oracle-anchored step diagnostics show a similar degradation from 84.29% selected to 44.71% executed. We refer to this degradation as the **Selection-to-Execution Gap**.

Motivated by this diagnosis, we propose **DiPLaN**, an execution-aligned future action decoder for long-horizon KG reasoning. DiPLaN does not generate free-form textual plans and ground them post hoc. Instead, it operates directly inside the executor's legal action space. It uses candidate-level discrete denoising to score admissible current actions, trajectory-level discrete denoising to model short-horizon future relation sequences, a path value model to estimate trajectory quality, and a learned fusion head to decode these signals into the next executable action under receding-horizon control.

Our central claim is not that DiPLaN solves general-purpose planning in the abstract. Rather, our claim is narrower and more testable: in execution-constrained long-horizon environments, future-aware action decoding inside the legal action space can reduce the gap between locally plausible decisions and successful executed transitions, while also amortizing the cost of explicit future search.

Our contributions are:

1. We identify and quantify a Selection-to-Execution Gap in execution-constrained long-horizon reasoning, showing that preserving answer reachability at the selection stage does not guarantee executed trajectory success.
2. We propose DiPLaN, a supervised execution-aligned future action decoder that combines candidate diffusion, trajectory diffusion, path value estimation, and learned fusion within the executor's legal action space.
3. We show how future-aware signals learned from oracle paths can be translated into step-level executable action decisions inside a legal action interface, rather than only into free-form plans.
4. We show that DiPLaN improves long-horizon KG reasoning while amortizing part of explicit future search, achieving large end-to-end efficiency gains over an explicit FLARE-MCTS baseline on a shared subset.

## 2. Related Work

### 2.1 Reasoning-Acting Agents and Long-Horizon Planning

ReAct established a powerful reasoning-acting paradigm in which language models interleave thought and action, making LLMs practical controllers for interactive tasks [Yao et al., 2022]. Subsequent work extended this paradigm to knowledge graphs, tool environments, and long-horizon benchmarks, showing that language models can guide structured exploration rather than only generate free-form answers [Sun et al., 2023; Luo et al., 2023]. However, stronger reasoning traces do not automatically yield robust long-horizon behavior.

Recent work has increasingly distinguished reasoning from planning. In particular, FLARE argues that step-wise reasoning induces a form of step-wise greedy policy that may be acceptable at short horizons but fails on long-horizon tasks, where early actions must account for delayed consequences [Wang et al., 2026]. This perspective is closely related to ours. The main difference is methodological: FLARE preserves explicit lookahead, rollout evaluation, and value backpropagation, whereas DiPLaN amortizes part of that future modeling into learned action decoders.

### 2.2 Grounding High-Level Intent to Executable Actions

Several lines of work have shown that natural language plans are not automatically executable. Language Models as Zero-Shot Planners demonstrated that plans generated in natural language often require additional translation before they can be mapped to admissible actions [Huang et al., 2022]. SayCan made a similar point in robotics by combining high-level language priors with low-level affordance and value signals grounded in the actual executor [Ahn et al., 2022].

These works motivate execution-aware planning, but our focus is slightly different. We do not primarily study whether a language plan can be mapped into an executable action space at all. Instead, we study the residual failure that remains after the action space is already explicit: why do locally plausible or answer-preserving actions still fail to become successful executed trajectories?

### 2.3 Knowledge-Graph Grounded Reasoning

Think-on-Graph and Reasoning on Graphs demonstrate that KGQA can be framed as structured action-based reasoning rather than ungrounded text generation [Sun et al., 2023; Luo et al., 2023]. Think-on-Graph treats the LLM as an agent that iteratively explores candidate relations and entities, while Reasoning on Graphs uses relation paths as grounded plans for retrieval and reasoning. These methods are directly relevant because they show that KGQA already has an executor with explicit action semantics.

DiPLaN does not claim to be the first system to reason in executable relation spaces. Its contribution is instead to diagnose and reduce the residual gap that remains within such spaces: even after admissible graph actions are explicit, local action plausibility still does not guarantee long-horizon success.

### 2.4 Diffusion for Planning and Decision Making

Diffusion-based planning methods reinterpret planning as iterative denoising over future trajectories [Janner et al., 2022]. In parallel, discrete diffusion models such as D3PM showed how denoising-style modeling can be extended from continuous spaces to structured categorical domains [Austin et al., 2021]. These ideas are relevant because long-horizon decision problems are naturally multi-modal, and a distribution over future trajectories can be more informative than a single greedy rollout.

DiPLaN draws inspiration from this literature but occupies a narrower design point. It is not a full offline RL diffusion planner over continuous trajectories. Instead, it uses lightweight D3PM-style denoising to score legal current actions and short-horizon future relation tokens inside a discrete executor-aligned action space. In our setting, diffusion is a mechanism for future-aware action decoding rather than an end-to-end free-form trajectory generator.

## 3. Problem Setup

We consider an execution-constrained long-horizon decision process. At step \(t\), the agent observes a state \(s_t\), receives a legal action set \(A(s_t)\), chooses an action \(a_t \in A(s_t)\), and transitions to \(s_{t+1}\). The task succeeds if the executed trajectory reaches a goal state within a horizon budget.

In our KGQA instantiation:

- **State** \(s_t\): the current entity frontier together with execution depth and history.
- **Action** \(a_t\): a legal relation available from the current frontier.
- **Executor**: deterministic traversal over the local KG subgraph.
- **Transition**: following the selected relation to produce the next frontier.
- **Success**: the final frontier contains the gold answer entity.

This setup enables a more informative diagnostic decomposition than final answer accuracy alone. For each decision step, we can ask:

1. Does the legal pool contain an answer-reaching action?
2. Does filtering preserve such an action?
3. Does the final selector choose such an action?
4. Does executing the top-1 selected action actually preserve a successful trajectory?

The gap between steps 3 and 4 is the central object of study in this paper.

## 4. Method

### 4.1 Overview

DiPLaN is a supervised future-action decoder that operates only over legal executable actions. Given a question \(q\), current state \(s_t\), executed prefix \(p_t\), and legal candidate relations \(A(s_t)\), DiPLaN computes multiple signals for each candidate relation and fuses them into a final score:

\[
\text{Score}(r \mid q, s_t, p_t) = f_\theta(
\text{base},
\text{question},
\text{cand-diff},
\text{traj-diff},
\text{value},
\text{prior},
\text{guided},
\text{state features}).
\]

The selected relation is executed immediately, after which the system replans from the new state. DiPLaN therefore follows receding-horizon control rather than full-horizon commitment.

### 4.2 Execution-Aligned Candidate Space

At inference time, DiPLaN never scores arbitrary schema relations. It only scores the executor's legal candidate set. In the codebase, [`src/diplan/kg_env.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\src\diplan\kg_env.py) defines an explicit KG executor that exposes admissible relations from the current frontier. This design is essential: DiPLaN is not a free-form relation generator, but a future-aware decoder over admissible executable actions.

### 4.3 Candidate Diffusion

The candidate diffusion module is implemented in [`src/diplan/candidate_diffusion.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\src\diplan\candidate_diffusion.py) and trained by [`train_candidate_diffusion_planner.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_candidate_diffusion_planner.py). For each oracle step, we construct the legal candidate set from the current state. If the oracle next relation is contained in that set, we corrupt the current relation token and train the model to recover the oracle candidate index inside the legal set. The objective is cross-entropy over legal candidates.

Formally, if \(C_t = A(s_t)\) and \(r_t^\*\) is the oracle next relation, candidate diffusion learns:

\[
p_\phi(r_t^\* \mid \tilde{r}_t, q, C_t, t_{\text{noise}}),
\]

where \(\tilde{r}_t\) is a corrupted relation token and \(t_{\text{noise}}\) is the diffusion timestep. This module should therefore be interpreted as a state-conditioned denoising scorer over legal actions, not as an unconstrained action generator.

### 4.4 Trajectory Diffusion

The trajectory diffusion module is implemented in [`src/diplan/trajectory_diffusion.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\src\diplan\trajectory_diffusion.py) and trained by [`train_trajectory_diffusion_planner.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_trajectory_diffusion_planner.py). Its supervision target is the future oracle relation subsequence `oracle[step : step + horizon]`. The model denoises a corrupted future relation sequence under the question condition and is optimized with token-level cross-entropy.

Crucially, DiPLaN does not execute the predicted future sequence directly. At inference time, the trajectory model is used to score current legal relations through the first-position probability mass of the denoised future sequence:

\[
\text{traj-score}(r \mid q, p_t) \propto p_\psi(r_1 = r \mid q, \tilde{\tau}_{t:t+H}).
\]

Thus, trajectory diffusion serves as a future-relation prior for current action selection, not as a full executable plan generator.

### 4.5 Path Value Model

The path value model is trained by [`train_value_model_torch.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_value_model_torch.py). Positive samples are oracle paths. Negative samples are built from corrupted paths, hard ranking errors, near-miss prefixes, and terminal truncations. The objective is path-level preference learning: trajectories that better preserve eventual success should receive higher value scores.

This component is important because answer reachability alone is too weak. In a graph, many actions remain answer-reaching, but only some maintain a strong trajectory toward final success.

### 4.6 Learned Fusion

The fusion head is trained by [`train_fusion_ranker.py`](C:\Users\32782\Downloads\科研\agent\DiPLaN\train_fusion_ranker.py). For each legal candidate relation at a decision step, we build a feature vector that includes:

- base lexical or heuristic score,
- question-relation prior,
- candidate diffusion score,
- trajectory diffusion score,
- value score,
- sampled prior score,
- guided rollout score,
- entity-count, depth, and rank statistics.

The fusion head learns to rank the oracle next relation above competing legal candidates. The current implementation supports listwise and hybrid losses, making the model explicitly optimized for step-level top-1 action selection rather than only per-candidate calibration.

### 4.7 Inference

At inference time, DiPLaN proceeds as follows:

1. Query the executor for legal candidate relations.
2. Compute base, prior, diffusion, and value signals for each candidate.
3. Fuse these signals with the learned ranker.
4. Execute the top-1 relation.
5. Update the state and repeat until termination.

This makes DiPLaN a learned alternative to repeated explicit textual lookahead and grounding. The key systems-level idea is that future modeling is amortized into reusable neural modules rather than recomputed by repeated language-model planning at every step.

## 5. Experiments

### 5.1 Experimental Questions

Our experiments are designed to answer three questions:

1. Does DiPLaN improve long-horizon task performance in both KGQA and an external agent environment?
2. Does DiPLaN reduce the Selection-to-Execution Gap?
3. Does DiPLaN amortize future reasoning more efficiently than explicit lookahead?

### 5.2 Setup

Our primary benchmark is WebQSP under a ToG-style local execution pipeline over oracle subgraphs. This benchmark is particularly useful because it exposes an explicit executor and enables trajectory-level diagnosis. The strongest full-run setting discussed in this paper uses `relation_first_k=16`, all-legal candidate pooling, and learned fusion, covering 1,546 WebQSP questions. Because both supervision and diagnostics use oracle structure, WebQSP serves as the paper's main mechanism-analysis environment.

For efficiency analysis, we also report controlled subset experiments using the patched official ToG scaffold with stage-level timing instrumentation. These include:

- a shared 20-example subset for direct FLARE-MCTS vs DiPLaN comparison,
- a 100-example lexical-pruning subset for runtime decomposition.

Beyond KGQA, we also evaluate DiPLaN on ALFWorld as an external long-horizon agent environment. In the paper narrative, ALFWorld serves as cross-environment validation of the execution-aligned decoding idea, while WebQSP remains the cleanest setting for diagnosing the Selection-to-Execution Gap itself.

This split is deliberate. The full run provides statistically meaningful diagnostic evidence, while the controlled subsets provide clean infrastructure-matched efficiency comparisons.

### 5.3 Baselines

We compare DiPLaN against three families of methods:

1. **ToG-style local exploration** without DiPLaN reranking.
2. **FLARE-MCTS**, an explicit-search baseline combining lookahead, rollout evaluation, backward value propagation, and receding-horizon execution.
3. **DiPLaN ablations**, including lighter pruning variants and partial module combinations.

For this paper, FLARE-MCTS is the most important explicit-search reference point because it represents the strongest competing story: future planning by repeated search and value propagation instead of learned future-action decoding.

### 5.4 Metrics

We report both final task metrics and intermediate execution diagnostics.

**Final metrics**

- **Hits@1**: final answer-level success rate.
- **trap@1**: fraction of tasks that commit to trap trajectories.
- **Wall-clock time per task**
- **LLM calls per task**
- **Token cost**

**Execution diagnostics**

- **Executable Action Recall (Pool)**: an answer-reaching action exists in the legal pool.
- **Executable Action Recall (Filtered)**: an answer-reaching action survives filtering.
- **Executable Action Selection Rate**: the selected action remains answer-reaching.
- **Executed Top-1 Transition Success**: the executed top-1 action truly advances a successful trajectory.

Formally, for a decision step \(t\), let \(A_t\) be the legal action pool, \(F_t \subseteq A_t\) the filtered pool after pruning or reranking, \(\hat{a}_t\) the final top-1 selected action, and \(y_t(a) \in \{0,1\}\) indicate whether action \(a\) lies on at least one answer-reaching trajectory under the local oracle subgraph. Let \(z_t \in \{0,1\}\) indicate whether executing \(\hat{a}_t\) preserves successful trajectory progress in the environment transition sense. We then define:

\[
\text{Pool Recall} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\!\left[\exists a \in A_t: y_t(a)=1\right]
\]

\[
\text{Filtered Recall} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\!\left[\exists a \in F_t: y_t(a)=1\right]
\]

\[
\text{Action Selection Rate} = \frac{1}{T}\sum_{t=1}^{T} y_t(\hat{a}_t)
\]

\[
\text{Executed Top-1 Transition Success} = \frac{1}{T}\sum_{t=1}^{T} z_t
\]

We define the **Selection-to-Execution Gap** as:

\[
\text{Gap} = \text{Action Selection Rate} - \text{Executed Top-1 Transition Success}.
\]

These metrics are post-hoc diagnostics computed from oracle structure and executed traces. They are used to analyze where failure occurs, not as direct inference-time supervision signals during test-time action selection.

### 5.5 Main WebQSP Diagnostic Result

On the full WebQSP run with `relation_first_k=16`, DiPLaN achieves:

- Hits@1: **87.65**
- trap@1: **0.84**
- Executable Action Recall (Pool): **99.07**
- Executable Action Recall (Filtered): **97.04**
- Executable Action Selection Rate: **93.85**
- Executed Top-1 Transition Success: **59.62**
- Oracle selected step rate: **84.29**
- Oracle executed top-1 rate: **44.71**

These numbers support two conclusions. First, answer-reaching actions are usually available, so the main bottleneck is not raw candidate recall. Second, there is a large degradation from local answer-preserving selection to actual trajectory-level success. The observed Selection-to-Execution Gap is **34.23 points** (93.85 -> 59.62). Even oracle-anchored diagnostics show a **39.58-point** degradation (84.29 -> 44.71). This indicates that long-horizon failure is not reducible to whether a plausible action was preserved; it is fundamentally a trajectory-execution reliability problem.

### 5.6 Efficiency Comparison with FLARE-MCTS

We next compare DiPLaN and FLARE-MCTS on a shared 20-example WebQSP subset using the same local ToG-style execution scaffold, the same split, and the same executor interface. We present this result as an amortization and efficiency diagnostic, not as a claim that offline learning and online search incur identical deployment costs.

**FLARE-MCTS**

- Hits@1: **0.90**
- Time per task: **193.71 s**
- LLM calls per task: **152.55**
- Total estimated tokens: **684,097**
- Executed Top-1 Transition Success: **0.1923**

**DiPLaN**

- Hits@1: **0.95**
- Time per task: **8.62 s**
- LLM calls per task: **5.05**
- Total estimated tokens: **39,671**
- Executed Top-1 Transition Success: **0.84**

Relative to FLARE-MCTS, DiPLaN is:

- **22.47x** faster in wall-clock time,
- uses **30.21x** fewer LLM calls,
- uses **17.25x** fewer tokens,
- and achieves much stronger executed transition success.

This result supports our main systems claim: explicit future search repeatedly pays the cost of textual reasoning and grounding, whereas DiPLaN amortizes future modeling into learned action-space decoders.

### 5.7 Runtime Decomposition

The lexical-pruning DiPLaN variant on a 100-example subset further clarifies runtime structure:

- Hits@1: **0.83**
- Time per task: **12.24 s**
- LLM calls per task: **7.17**
- Total LLM wall time: **1163.11 s**
- Total DiPLaN rerank wall time: **52.52 s**
- Executed Top-1 Transition Success: **0.6739**

This shows that, even in the lite setting, most end-to-end runtime is dominated by LLM-based pruning rather than the DiPLaN neural core itself. Therefore, the fairest efficiency comparison is not neural-core time in isolation, but total end-to-end runtime under matched executor interfaces.

### 5.8 Interpretation

A key interpretive point is that `answer_reaching_selected_rate` does **not** mean the system selected the uniquely correct action. It only means the selected action remains on at least one answer-reaching trajectory. Therefore, the gap from 93.85% selected to 59.62% executed should not be read as "the system already chose the correct action but failed to execute it." A more precise interpretation is that many locally viable actions still fail to convert into reliable long-horizon success.

This distinction matters scientifically. The evidence does not merely show noisy ranking. It shows that preserving local answer reachability is insufficient, and that long-horizon reliability depends on how future trajectory structure is compressed into the current action decision.

## 6. Discussion

### 6.1 What DiPLaN Solves

DiPLaN does not claim to solve planning for all agents in all environments. Its strongest supported claim is narrower: in execution-constrained long-horizon environments, future-aware action decoding within the legal action space can reduce the gap between locally plausible action selection and successful executed transitions, while also lowering the cost of repeated explicit future search.

### 6.2 Why Textual Plan Passing Is a Weak Interface

The controlled WebQSP results motivate a broader systems-level interpretation. When future signals are passed through free-form text, the system repeatedly pays three costs:

1. **Lossy compression**: rich future distributions are reduced to sparse textual descriptions.
2. **Repeated decode-ground overhead**: every step requires generating text, interpreting it, and grounding it to executable actions.
3. **Interface instability**: textual intent is not the same object as an admissible executor action.

DiPLaN avoids these costs by learning directly over executable action candidates.

### 6.3 Limitations

Our strongest diagnostic evidence comes from KGQA, where the executor is explicit and controlled. This is useful for mechanism analysis, but it is not the only empirical setting we consider. We also validate DiPLaN on ALFWorld as an external long-horizon agent benchmark. Accordingly, our claim is broader than KGQA alone, but still narrower than a universal statement about all tool-use, web, or embodied agents.

We also do not claim that DiPLaN generates full executable trajectories. The trajectory diffusion component provides a future prior for current action selection; execution still proceeds one step at a time under receding-horizon control.

## 7. Conclusion

We introduced DiPLaN, an execution-aligned future action decoder for long-horizon KG reasoning and agent control. By moving future modeling from free-form textual planning into the executor's legal action space, DiPLaN transforms candidate denoising, trajectory denoising, value estimation, and learned fusion into a practical step-level decision mechanism. On WebQSP, this yields both stronger long-horizon reasoning behavior and large efficiency gains over explicit future search in a controlled execution setting, while ALFWorld provides external validation beyond KGQA. More broadly, our results suggest that an important challenge in long-horizon agent systems is not only whether future-relevant signals exist, but whether those signals can be reliably decoded into executable current actions.

## References to Prepare

The current draft is written to align naturally with the following references:

- Yao et al., 2022. ReAct: Synergizing Reasoning and Acting in Language Models.
- Sun et al., 2023. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
- Luo et al., 2023. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning.
- Ahn et al., 2022. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.
- Huang et al., 2022. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents.
- Wang et al., 2026. Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents.
- Janner et al., 2022. Planning with Diffusion for Flexible Behavior Synthesis.
- Austin et al., 2021. Structured Denoising Diffusion Models in Discrete State-Spaces.

## Suggested Next Edits

1. Add a compact formal notation block for the four diagnostic metrics.
2. Add one module ablation table on the full WebQSP run.
3. Add one figure visualizing the Selection-to-Execution Gap.
4. Convert author-year placeholders into your target venue's citation style.
