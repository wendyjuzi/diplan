# DiPLaN Grounded Paper Narrative

> 目标：这份文档只根据三类证据组织论文叙事：相关文献、当前代码实现、已有实验诊断。它刻意避免把 DiPLaN 描述成尚未实现的端到端轨迹生成器。

## 1. 先给结论

当前最稳的论文叙事不是：

```text
We use diffusion to generate plans for KGQA.
```

而是：

```text
Long-horizon agents often fail because locally plausible action choices do not reliably become successful state transitions. DiPLaN studies this action-selection bottleneck in an execution-aligned state-action space, and uses supervised discrete denoising plus value-guided fusion to convert future trajectory signals into current executable action decisions.
```

中文版本：

```text
长程 Agent 的关键瓶颈不只是“不会规划”，而是当前一步看似合理的动作选择不能稳定转化为长期成功轨迹。DiPLaN 在执行器可消费的状态-动作空间中建模候选动作和未来关系序列，用离散去噪、价值估计和融合排序把未来信号转化为当前可执行动作决策。
```

这句话比“Diffusion Planner”更安全，因为它同时符合代码事实和文献位置。

## 2. 文献事实：别人已经做到哪里

### 2.1 LLM agent 已经在“边推理边行动”

ReAct 提出让 LLM 交替产生 reasoning traces 和 task-specific actions，使动作与外部环境/API/KG 交互。这说明“agent = reasoning + acting loop”已经是成熟基线，不应把“会调 executor”作为 DiPLaN 的新意。

参考：ReAct, arXiv:2210.03629, https://arxiv.org/abs/2210.03629

### 2.2 KGQA 中已有 executable graph search

ToG 把 LLM 当作 agent，在 KG 上迭代探索实体和关系，并用 beam search 找 reasoning paths。因此 KGQA 里的 relation/action search 本身不是新贡献。

参考：Think-on-Graph, arXiv:2307.07697, https://arxiv.org/abs/2307.07697

RoG 进一步把 relation paths 作为 grounded plans，用 KG 检索有效 reasoning paths，再交给 LLM reasoning。这说明“路径作为计划”也已经存在。

参考：Reasoning on Graphs, arXiv:2310.01061, https://arxiv.org/abs/2310.01061

### 2.3 plan/action grounding 是长期问题

SayCan 的核心思想是：语言模型有高层语义知识，但需要由 skill affordance/value grounding 约束到机器人当前能做的动作。它支持我们的上位问题：语言计划必须和执行能力/状态可行性对齐。

参考：SayCan, arXiv:2204.01691, https://arxiv.org/abs/2204.01691

Language Models as Zero-Shot Planners 也指出，LLM naive 生成的 plans often cannot map precisely to admissible actions，因此需要把计划翻译到可执行动作集合。

参考：LMs as Zero-Shot Planners, arXiv:2201.07207, https://arxiv.org/abs/2201.07207

### 2.4 最新长程规划文献强调 myopic commitment

FLARE 的论文把长程失败归因于 step-wise reasoning 诱导的 greedy/myopic policy，并提出 future-aware lookahead 和 reward estimation。这个文献和 DiPLaN 最接近，但 DiPLaN 不应声称同一个贡献；更稳的是说：FLARE 从 planning-centric 角度证明 myopic step-wise decision 是问题，DiPLaN 在 KGQA 的 execution-aligned action space 中实现了一个监督式、可摊销的 future-action scoring 版本。

参考：Why Reasoning Fails to Plan, arXiv:2601.22311, https://arxiv.org/abs/2601.22311

### 2.5 diffusion planning 已经存在，DiPLaN 只能做“离散动作空间借鉴”

Diffuser 把 planning 看作 trajectory denoising；Decision Diffuser 把条件生成模型用于决策。但它们通常面向连续控制/离线 RL 或通用决策建模。DiPLaN 当前实现更接近 D3PM 风格的 discrete denoising over relation/action tokens，而不是完整 Diffuser 式轨迹优化器。

参考：

- Diffuser, arXiv:2205.09991, https://arxiv.org/abs/2205.09991
- Decision Diffuser, arXiv:2211.15657, https://arxiv.org/abs/2211.15657
- D3PM, arXiv:2107.03006, https://arxiv.org/abs/2107.03006

## 3. 代码事实：DiPLaN 当前实际做了什么

### 3.1 executor 是明确的状态-动作系统

`src/diplan/kg_env.py` 把 KGQA 转成确定性环境：

```text
state = frontier entities + depth
action = relation
transition = follow relation from frontier to next entity set
terminal = answer reached / max depth / empty frontier
```

因此 KGQA 在本项目中不是纯文本 reasoning，而是一个受控 state-action executor。这个点可以支撑 execution-aligned action space 的说法。

### 3.2 candidate diffusion 不是自由生成 relation

`src/diplan/candidate_diffusion.py` 的 docstring 已经写得很清楚：

```text
The model does not generate arbitrary KG relations;
it denoises a corrupted current relation back to one relation inside the legal candidate set.
```

训练样本来自 `oracle_path`。每一步取 `env.admissible_relations(state)`，如果 gold relation 在 candidates 中，就把 gold 的 candidate index 作为标签。训练目标在 `train_candidate_diffusion_planner.py` 中是 cross entropy over current candidates。

所以 candidate diffusion 的真实定位是：

```text
state-conditioned legal action denoising scorer
```

不是：

```text
free-form relation generator
```

### 3.3 trajectory diffusion 生成的是未来关系序列分布信号，不是直接执行轨迹

`src/diplan/trajectory_diffusion.py` 构造目标：

```text
oracle[step : step + horizon]
```

它训练一个 discrete denoiser 去恢复未来 horizon 内的 relation tokens。推理时 `score_first_relations(...)` 只取预测未来序列第一个位置的 logits，对当前候选 relations 打分。

所以 trajectory diffusion 的真实定位是：

```text
future relation-sequence prior used to score the current action
```

不是：

```text
generate [r1, r2, r3] and hand the whole plan to executor
```

当前 executor 仍然每次只执行一个 relation/action。

### 3.4 fusion ranker 是监督式 action ranker

`src/diplan/fusion_ranker.py` 的特征包括：

```text
base_score
value_z
question_z
candidate_diffusion_z
trajectory_diffusion_z
prior_z
guided_z
entity_count_log
candidate_rank_frac
depth_frac
has_entities
```

`train_fusion_ranker.py` 里 label 是：

```text
1.0 if rel == gold else 0.0
```

因此 fusion ranker 学的是“在每个 decision step 里把 oracle relation 排前面”，不是直接预测最终答案，也不是 RL policy optimization。

### 3.5 relation scorer 和 candidate diffusion 的区别

`relation_scorer.py` 是 query-relation contrastive scorer，估计：

```text
P(relation | question, executed_prefix)
```

它更像语义匹配/局部先验。

`candidate_diffusion.py` 是在候选动作集合里做 corrupted relation -> gold relation 的去噪恢复，利用 diffusion timestep/noisy relation/candidate set 来学习“当前合法动作集合中哪个动作像 gold action”。

简单说：

```text
relation_scorer = question-relation semantic prior
candidate_diffusion = legal candidate set conditioned denoising action selector
```

## 4. 实验事实：已有结果能支持什么，不能支持什么

目前最强的 WebQSP 全量诊断结果：

```json
{
  "n": 1546,
  "hits@1": 0.8460543337645536,
  "trap@1": 0.007761966364812419,
  "answer_reaching_in_pool_rate": 0.9917733089579525,
  "answer_reaching_in_keep_rate": 0.9424131627056673,
  "answer_reaching_selected_rate": 0.893510054844607,
  "answer_reaching_executed_top1_rate": 0.6101462522851919
}
```

这些数字不要被写成“模型看到了正确树”。更准确的说法是：

```text
这些是 post-hoc diagnostic metrics，用 answer reachability / oracle structure 评估每一步候选动作质量；它们不是推理时给模型看的信号。
```

它能支持的结论：

```text
在这个受控 KG executor 中，answer-reaching action 经常已经存在于 legal/all-legal pool 中，也经常能进入 keep/selected 集合，但 selected action 到 executed top-1 success 仍有明显损失。
```

它不能单独支持的结论：

```text
DiPLaN 已经解决通用 long-horizon agent planning。
DiPLaN 已经生成完整可执行计划。
DiPLaN 已经优于所有 ToG/PoG/FLARE 系统。
```

建议把工程指标改成论文指标：

| 工程指标 | 论文指标 | 含义 |
|---|---|---|
| `answer_reaching_in_pool_rate` | Executable Action Recall (Pool) | 当前候选池中是否存在能到达答案的动作 |
| `answer_reaching_in_keep_rate` | Executable Action Recall (Filtered) | rerank/keep 后是否仍保留 answer-reaching action |
| `answer_reaching_selected_rate` | Executable Action Selection Rate | 策略是否把 answer-reaching action 放入最终执行候选 |
| `answer_reaching_executed_top1_rate` | Executed Top-1 Transition Success | top-1 执行动作是否真的推进到成功 |

核心诊断指标：

```text
Plan-Execution Gap
= Executable Action Selection Rate - Executed Top-1 Transition Success
= 89.35% - 61.01%
= 28.34%
```

这个 gap 是叙事中最有解释力的数字，因为它不是单纯吹 hits@1，而是在量化“选到看起来相关的动作”和“执行成成功轨迹”之间的断层。

## 5. 最推荐的论文主线

### 5.1 问题定义

```text
Long-horizon agent failures are often attributed to weak reasoning or shallow search.
However, in execution-constrained environments, a major bottleneck is the conversion
from locally plausible planning signals to state-conditioned executable actions that
lead to successful future transitions.
```

中文：

```text
长程 Agent 的失败常被归因于推理能力不足或搜索不够深。但在有明确执行器约束的环境中，一个关键瓶颈是：局部看似合理的规划信号，不能稳定转化为当前状态下可执行且能推进未来成功轨迹的动作。
```

### 5.2 方法定义

```text
DiPLaN is a supervised, execution-aligned future action decoder.
It operates inside the executor's legal action space, uses discrete denoising models
to estimate current and future executable relation distributions, and fuses these
signals with value estimates for receding-horizon action selection.
```

中文：

```text
DiPLaN 是一个监督式、执行对齐的未来动作解码器。它不在自由文本计划空间里生成计划，而是在 executor 当前合法动作空间内，用离散去噪模型估计当前动作和未来关系序列分布，再与 value signal 融合，做滚动式当前动作选择。
```

### 5.3 三层架构命名

论文里不要列一堆模块，而要按问题逻辑组织：

```text
Execution-Aligned Action Space
    -> legal action pool / KG executor / relation candidates

Diffusion-based Future Action Modeling
    -> candidate diffusion / trajectory diffusion

Value-Guided Action Decoding
    -> value scoring / learned fusion / receding-horizon selection
```

这样 candidate diffusion、trajectory diffusion、fusion ranker 都不是“堆料”，而是分别服务于：

```text
动作可执行性
未来可预期性
当前动作承诺
```

## 6. 和现有工作的差异边界

### 6.1 对 ToG

ToG 是 LLM-on-KG search/exploration backbone。DiPLaN 不替代 ToG，而是在 ToG/local KG executor 提供的 candidate relation set 上进行 future-aware reranking。

可写：

```text
Unlike ToG, which relies on LLM-guided graph exploration, DiPLaN amortizes part of the low-level action scoring with learned denoising and value signals.
```

不要写：

```text
ToG does not plan in executable space.
```

因为 ToG 确实在 KG relation/entity space 中探索。

### 6.2 对 RoG

RoG 把 relation paths 作为 KG-grounded plans，并用于 retrieval/reasoning。DiPLaN 的差异不是“首次用 path plan”，而是更细粒度地在每个 execution state 上做 current action scoring，并引入 future relation denoising/value fusion。

### 6.3 对 FLARE

FLARE 的核心是 future-aware lookahead + reward estimation，强调 step-wise greedy policy 的长期失败。DiPLaN 可以承认这是强相关文献，然后定位为：

```text
DiPLaN instantiates future-aware decision making in a structured executable action space, using supervised discrete denoising over KG relations rather than general LLM lookahead.
```

### 6.4 对 Diffuser / Decision Diffuser

Diffuser/Decision Diffuser 是轨迹生成/决策建模范式来源。DiPLaN 当前只是借鉴 discrete denoising 的思想，不应说成完整 offline RL diffusion planner。

更稳写法：

```text
Inspired by diffusion-based trajectory modeling, DiPLaN uses lightweight D3PM-style denoising to score discrete executable actions and future relation tokens.
```

## 7. 论文贡献应该怎么写

推荐贡献：

```text
1. We diagnose an action-selection bottleneck in execution-constrained long-horizon KG reasoning: answer-reaching actions can be highly available in the legal action space, yet final executed success remains substantially lower.

2. We propose DiPLaN, a supervised execution-aligned future action decoder that combines candidate-level discrete denoising, trajectory-level denoising, value estimation, and learned fusion for receding-horizon action selection.

3. We provide diagnostic metrics that decompose long-horizon failure into action recall, filtered recall, selected action rate, and executed top-1 transition success, exposing the Plan-Execution Gap hidden by final Hits@1 alone.

4. We empirically evaluate DiPLaN on KGQA with ToG-style execution and propose extension experiments on tool-agent benchmarks such as tau-bench/AppWorld to test whether the same mismatch appears in API executors.
```

注意第 4 点如果还没跑 tau-bench/AppWorld，只能写成 planned extension 或 future validation，不能写成已证明。

## 8. 摘要草稿

```text
Large language model agents are increasingly used for long-horizon tasks, yet their failures are often explained only as weak reasoning or insufficient search. We argue that, in execution-constrained environments, a key bottleneck is the mismatch between locally plausible planning signals and state-conditioned executable actions that produce successful future transitions. We study this problem in knowledge-graph question answering, where the executor exposes an explicit state-action interface: states are entity frontiers and actions are graph relations. Our diagnostics show that answer-reaching actions can frequently appear in the legal action pool and even survive filtering, while top-1 executed transition success remains substantially lower, revealing a Plan-Execution Gap.

We propose DiPLaN, a supervised execution-aligned future action decoder. Rather than generating free-form textual plans, DiPLaN operates inside the executor's legal action space. It uses candidate-level discrete denoising to score current legal actions, trajectory-level denoising to estimate future relation distributions, and value-guided learned fusion to select actions in a receding-horizon manner. Experiments with ToG-style KG execution show that DiPLaN improves long-horizon KG reasoning while providing interpretable diagnostics of where planning signals fail to become successful execution. Our results suggest that future-aware action decoding inside executable spaces is a practical route toward more reliable long-horizon agents.
```

## 9. Introduction 开头草稿

```text
Long-horizon agents must do more than produce plausible plans: they must repeatedly convert planning signals into actions that are executable in the current environment state and that lead to successful downstream transitions. Existing LLM-agent systems often interleave reasoning and acting, or use language models to search over external tools and knowledge graphs. However, such systems may still commit to actions that are locally plausible but globally suboptimal. Recent planning-centric analyses similarly show that step-wise reasoning can induce greedy policies that fail over long horizons.

We study this problem through the lens of execution-aligned planning. In an execution-constrained environment, the executor consumes actions from a state-dependent admissible action set A(s), not arbitrary textual intentions. Therefore, the central challenge is not only to produce a high-level plan, but to decode future planning signals into current executable action commitments. We refer to the residual failure between selected plausible actions and successful executed transitions as the Plan-Execution Gap.

Knowledge-graph question answering provides a controlled testbed for this question. The executor is explicit: a state is a frontier of entities, an action is a relation, and a transition follows that relation to the next frontier. This allows us to decompose failures into action availability, filtering, selection, and executed transition success. Our diagnostics show that answer-reaching actions can be highly available in the candidate pool while executed top-1 success remains much lower, indicating that the bottleneck is not simply candidate recall but future-aware action commitment.
```

## 10. 必须避免的高风险说法

不要写：

```text
DiPLaN generates complete executable trajectories.
```

因为当前 trajectory diffusion 只用未来序列分布给当前候选 relation 打分。

不要写：

```text
DiPLaN directly feeds diffusion latents into the executor.
```

因为 executor 接收的是 discrete relation/action，不接收 latent。

不要写：

```text
DiPLaN solves plan-executor mismatch in general agents.
```

因为目前主要证据来自 KGQA。可以写：

```text
DiPLaN provides a controlled instantiation and diagnostic framework for execution-aligned future action decoding in KGQA, with natural extensions to tool-agent benchmarks.
```

不要写：

```text
We are the first to plan in executable action space.
```

因为 ToG/RoG/SayCan/zero-shot planners 都已经涉及 action grounding 或 executable action constraints。

更稳写：

```text
We focus on the remaining action-selection gap within executable action spaces: how future trajectory signals can be decoded into current action choices.
```

## 11. 下一步实验建议

### 11.1 KGQA 内部补强

必须做的消融：

```text
ToG baseline
ToG + relation_scorer
ToG + candidate_diffusion
ToG + trajectory_diffusion
ToG + value
ToG + learned_fusion
ToG + answer voting / uncertainty trigger
```

必须报告：

```text
Hits@1
Trap@1
Executable Action Recall (Pool)
Executable Action Recall (Filtered)
Executable Action Selection Rate
Executed Top-1 Transition Success
Plan-Execution Gap
LLM calls / task
tokens / task
latency / task
```

### 11.2 工具 Agent 扩展

优先级：

```text
tau-bench > AppWorld > WebArena
```

理由：

```text
tau-bench 有明确 API tools、policy constraints、database final-state evaluation，最适合验证 plan-executor mismatch。
AppWorld 更难但更强，适合作为后续扩展。
WebArena 成本更高且评价噪声更大，不适合作为第一验证场。
```

在 tau-bench 上先做最小闭环：

```text
state = dialogue state + tool history + database summary
action = tool name, later extend to tool name + arguments
executor = API environment
transition = database/tool observation update
success = final database state matches goal
```

不要一开始就做 full argument generation，否则变量太多。先验证：

```text
future-aware action decoding 是否能减少 LLM 每步工具选择调用，并保持或提升 success。
```

## 12. 最后一句定位

DiPLaN 最可信的定位是：

```text
An execution-aligned future action decoder for long-horizon agents.
```

它不是现在就已经完成的通用 Agent planner，而是一个从 KGQA 出发、代码机制清晰、可向 tool-agent benchmark 扩展的研究框架。

