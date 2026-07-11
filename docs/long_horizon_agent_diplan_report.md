# DiPLaN 长程 Agent 规划汇报

## 0. 一句话总论点

现有长程 Agent 的主要失败并不只来自规划能力不足，而来自高层规划表示与执行器可消费的状态-动作接口之间的结构性错配，本文将其定义为 **Plan-Executor Misalignment**。

更具体地说：

```text
Agent 并不是完全找不到正确方向，而是 planner 产生的 plan/rank/future signal
不能稳定转化为 executor 当前状态下可执行、可落地、可推进任务的 action commitment。
```

DiPLaN 的定位不是“用 diffusion 做 KGQA reranker”，而是：

```text
DiPLaN 在 executor 的 executable action space 中建模 future executable trajectories，
再用 value-guided decoder 把未来规划信号转化为当前可执行动作。
```

一句英文版摘要核心：

```text
We study Plan-Executor Misalignment in long-horizon agents: high-level planning signals
may identify plausible actions but fail to translate into state-conditioned executable
commitments. DiPLaN addresses this by modeling future executable trajectories and decoding
them into current actions within the executor's action space.
```

## 1. 为什么这个问题不是“规划能力不足”

很多长程 Agent 论文默认失败原因是：

```text
LLM plan 不够强
search 不够深
ranking 不够准
reasoning 不够好
```

但我们的实验现象更像另一件事：

```text
正确动作已经经常出现在候选空间中，甚至进入 filtered/selected 集合，
但最终 executed trajectory 仍然明显掉分。
```

WebQSP 全量实验观察：

```text
Hits@1 = 84.61%
trap@1 = 0.78%
answer_reaching_in_pool = 99.18%
answer_reaching_in_keep = 94.24%
answer_reaching_selected = 89.35%
answer_reaching_executed_top1 = 61.01%
```

这说明瓶颈不是简单的 candidate recall failure。更准确的瓶颈是：

```text
model can identify answer-reaching actions,
but cannot always execute them into successful trajectories.
```

换句话说：

```text
看起来会做 != 真的做成。
```

这就是 Plan-Executor Misalignment 的实验入口。

## 2. Plan 和 Executor 为什么没有对齐

最关键的一句话是：

```text
plan 描述的是“想完成什么”，executor 需要的是“当前状态下能执行哪个控制变量”。
```

这两者不是同一种表示。

### 2.1 形式化地看

一个 planner 通常做的是：

```text
z_t = Planner(goal, history, abstract_state)
```

其中 `z_t` 可能是自然语言计划、推理文本、未来意图、动作排序、轨迹评分。

但 executor 真正消费的是：

```text
a_t ∈ A(s_t)
s_{t+1}, o_{t+1} = Executor(s_t, a_t)
```

这里：

```text
s_t      = 当前真实环境状态
A(s_t)   = 当前状态下合法、可执行的动作集合
a_t      = 一个具体可执行动作
s_{t+1}  = 执行动作后的真实状态转移
o_{t+1}  = executor 返回的观测、错误、API 结果或实体集合
```

错配发生在：

```text
z_t 不是 a_t
z_t 不一定能映射到 A(s_t)
z_t 认为合理的动作不一定产生目标状态转移
z_t 后续也不一定能吸收 executor feedback 更新自己
```

因此问题不是 planner 没有想法，而是 planner 的输出没有天然满足 executor 的接口约束。

## 3. Plan-Executor Misalignment 的三层框架

原来可以列出五类 mismatch：representation、granularity、state、transition、feedback。为了论文更清楚，建议统一成三层。

| 层级 | 错配形式 | 核心问题 | 例子 |
|---|---|---|---|
| 表示层 Representation Layer | textual intent != executable action；one plan step != one executor step | planner 输出的不是 executor 能直接执行的控制变量 | “检查订单是否可取消”不是 `get_order`、`check_policy`、`cancel_order` 中的某一个具体调用 |
| 状态层 State Layer | abstract plan state != environment state | planner 对任务进度的理解和 executor 的真实环境状态不同步 | planner 以为订单可取消，但数据库显示订单已发货 |
| 动态层 Transition Layer | plausible action != successful transition；feedback != structured replanning signal | 局部合理动作不等于最终成功轨迹，executor 反馈也未必能稳定闭环更新 plan | `cancel_order` 语义合理，但因为 policy/参数/状态条件失败，最终无法完成任务 |

这三层可以直接对应到 DiPLaN 的设计：

| 问题层 | DiPLaN 的对应设计 | 作用 |
|---|---|---|
| 表示层错配 | Execution-Aligned Action Space | 不在自由文本计划空间里做决策，而在 executor 可执行动作空间里建模 |
| 状态层错配 | State-Conditioned Future Proposal | 每一步基于当前实体/工具/环境状态重新产生候选动作和未来轨迹信号 |
| 动态层错配 | Value-Guided Action Decoder + Receding-Horizon Replanning | 不只看当前动作像不像，而看动作后的未来轨迹价值；每执行一步后重新规划 |

## 4. 用例解释：为什么文本 plan 会和 executor 脱节

假设一个 tool agent 的文本计划是：

```text
Find the user's order.
Check whether it can be cancelled.
Cancel the order.
Notify the user.
```

这对人类来说很清楚，但 executor 能执行的是：

```text
get_user_details(user_id)
list_orders(user_id)
get_order(order_id)
cancel_order(order_id, reason)
send_message(user_id, content)
```

中间有四个关键断点。

第一，文本意图不是可执行动作。

```text
“Check whether it can be cancelled” 不是一个 action。
它可能需要 get_order、inspect_status、check_policy 等多个工具调用。
```

第二，一个 plan step 和 executor step 的粒度不一致。

```text
一个高层计划步骤可能对应 0、1、N 个底层动作。
如果没有显式对齐，planner 会以为一个子目标完成了，executor 其实还没产生必要状态转移。
```

第三，planner 的状态是抽象状态，executor 的状态是真实状态。

```text
planner: order checked, cancel possible
executor: order_status = shipped, cancel_order invalid
```

第四，语义合理不等于执行成功。

```text
cancel_order(order_id) 在语言上是正确动作，
但如果 order_id 错、状态不满足、缺少 reason、policy 不允许，就无法推进最终目标。
```

这就是为什么“让 LLM 多想一点”不一定解决问题。需要解决的是：

```text
future planning signal -> executor-consumable action commitment
```

## 5. KGQA 为什么是一个可控验证场

KGQA 不是最终故事的全部，但它是一个很干净的实验环境，因为它把 long-horizon execution 的关键变量显式化了。

| 一般 Agent | KGQA 中的对应物 |
|---|---|
| goal / user request | question |
| state | current entity set / subgraph |
| executable action | relation |
| executor | graph traversal |
| transition | entity --relation--> next entity |
| final success | answer entity reached |
| oracle trajectory | oracle_path |

所以 KGQA 允许我们精确诊断：

```text
正确动作在不在池子里？
正确动作有没有被保留下来？
正确动作有没有被选中？
选中了之后是否真的走成成功轨迹？
```

这比普通 Web/tool agent 更容易观察 plan-executor mismatch 的中间环节。

但论文中需要谨慎表述：

```text
KGQA/ToG/PoG 本来就在 executable relation space 中搜索。
我们的新意不是“第一个在 action space 中规划”，
而是识别并量化 executable action space 内部的 myopic trajectory selection / plan-execution mismatch，
并用 future trajectory modeling + value-guided decoding 来缓解。
```

## 6. 重新命名实验指标

当前工程指标建议改成论文指标：

| 工程名字 | 论文名字 | 含义 |
|---|---|---|
| `answer_reaching_in_pool` | Executable Action Recall (Pool) | 候选池中是否存在能到达答案的可执行动作 |
| `answer_reaching_in_keep` | Executable Action Recall (Filtered) | 过滤/保留后的动作集合中是否仍存在答案相关动作 |
| `answer_reaching_selected` | Executable Action Selection Rate | planner/ranker 是否把答案相关动作选入执行候选 |
| `answer_reaching_executed_top1` | Executed Trajectory Success Rate | top-1 执行动作是否真正推进到成功轨迹 |

建议新增一个核心诊断指标：

```text
Plan-Execution Gap = Executable Action Selection Rate - Executed Trajectory Success Rate
```

用当前结果计算：

```text
89.35% - 61.01% = 28.34%
```

这个数字很关键。它说明：

```text
系统已经较高概率选到了看起来答案相关的动作，
但从“被选中”到“执行成功”之间仍损失了 28.34 个百分点。
```

这比单独报告 Hits@1 更有论文味，因为它直接量化了 plan-executor mismatch。

## 7. DiPLaN 到底解决什么

DiPLaN 不应该被讲成一堆模块：

```text
candidate diffusion + trajectory diffusion + value model + fusion ranker + voting
```

更好的讲法是三阶段：

```text
Search Space Construction
        ↓
Diffusion-based Future Modeling
        ↓
Value-guided Action Decoding
```

### 7.1 Search Space Construction

作用：保证答案相关动作不要太早丢失。

对应模块：

```text
legal action pool
relation scorer
schema/question prior
```

它回答的问题是：

```text
当前状态下哪些动作可执行、值得进入候选空间？
```

### 7.2 Diffusion-based Future Modeling

作用：不只判断当前动作局部是否合理，而是估计它之后的未来可执行轨迹分布。

对应模块：

```text
candidate_diffusion
trajectory_diffusion
```

它回答的问题是：

```text
如果现在选择这个 action，未来几步更可能走向哪些 executable trajectories？
这些轨迹是否更接近成功状态？
```

注意：

```text
当前 DiPLaN 不是直接生成完整文本计划，
也不是把 diffusion latent 直接喂给 executor。
它是在候选 executable actions 上产生 score/distribution，
再辅助选择当前一步动作。
```

### 7.3 Value-guided Action Decoding

作用：把未来轨迹信号转化成当前可执行动作承诺。

对应模块：

```text
value model
fusion ranker
answer voting / uncertainty trigger
```

它回答的问题是：

```text
在当前状态 s_t 和合法动作 A(s_t) 下，
哪个 a_t 最可能产生高长期价值的 trajectory？
```

所以 DiPLaN 的核心不是“生成一个漂亮计划”，而是：

```text
把 future-aware planning signal 解码成 executor 当前能执行的一步动作。
```

## 8. 为什么 diffusion 是合理工具

Diffusion 在这里不是为了炫模型，而是为了两个实际问题。

第一，长程 Agent 每步都调用 LLM 低层规划成本很高。

```text
LLM every step: expensive and slow
DiPLaN: amortized low-level future action proposal
```

第二，单步局部 ranker 容易 myopic。

```text
local action similarity != long-horizon trajectory value
```

Diffusion 的合理角色是：

```text
learn a distribution over future executable actions / trajectories
```

不是：

```text
generate a free-form textual plan
```

更准确的论文表述：

```text
Diffusion is used as a low-cost future proposal mechanism over executable action trajectories,
rather than a free-form plan generator.
```

## 9. 和 ToG / PoG / FLARE 的关系

### 9.1 和 ToG 的关系

ToG 更像：

```text
graph-based action generator + search/executor backbone
```

它一步一步产生可扩展关系，并执行图遍历。

DiPLaN 更像：

```text
future-aware action scorer / action decoder
```

它不替代 ToG，而是在 ToG 的 executable relation space 中增强决策。

### 9.2 和 PoG 的关系

PoG 强调 plan-on-graph 或 path-level planning，关注图结构上的规划。

DiPLaN 的差异可以讲成：

```text
not only planning paths, but diagnosing and reducing the gap between planned/ranked actions and executed successful trajectories.
```

### 9.3 和 FLARE 的关系

FLARE 强调：

```text
future-aware lookahead + reward estimation
```

DiPLaN 可以吸收它的 future-aware 思想，但叙事不要变成复制 FLARE。我们的不同点应放在：

```text
Plan-Executor Misalignment
Execution-aligned future trajectory modeling
Future signal to executable action decoding
Efficiency via amortized diffusion proposal
```

## 10. 接下来为什么要做 tau-bench

如果只做 KGQA，审稿人可能会说：

```text
relation 本来就是 executable action，这是不是 KGQA 特例？
```

因此第二个验证场建议做 tau-bench，因为它更像真实 tool-agent：

```text
state = dialogue + database + tool history
action = API/tool call
executor = tool environment
transition = database/API state update
success = final database state satisfies task goal
```

它可以验证同一个核心命题是否超出 KGQA：

```text
plan/rank 看起来合理，不代表 tool execution 最终成功。
```

最小版本不要一开始做完整参数生成，建议分三阶段：

```text
Stage 1: tool-name selection
Stage 2: tool name + argument schema sketch
Stage 3: full executable tool call
```

这样可以先验证 DiPLaN 的 action-level future proposal 是否有效，再逐步接入真实 executor。

## 11. 推荐实验组织

### Experiment 1: Efficiency

问题：DiPLaN 能否减少高频 LLM 低层规划调用？

对比：

```text
LLM every step
LLM every H steps + cached plan
DiPLaN low-level future proposal
DiPLaN + uncertainty-triggered LLM
```

指标：

```text
success rate
LLM calls / task
tokens / task
latency / task
success / token
success / LLM call
```

### Experiment 2: Mismatch Diagnosis

问题：rank/action recall 高但 final success 低是否普遍存在？

指标：

```text
Executable Action Recall (Pool)
Executable Action Recall (Filtered)
Executable Action Selection Rate
Executed Trajectory Success Rate
Plan-Execution Gap
```

### Experiment 3: Future-to-Action Decoding

问题：future trajectory signal 是否能更好地转化为当前可执行动作？

消融：

```text
Local ranker
Candidate diffusion
Trajectory diffusion
Value-guided fusion
DiPLaN + LLM-on-demand
```

指标：

```text
final success
pass^k
recovery rate
invalid action rate
state transition error
Plan-Execution Gap reduction
```

## 12. 给学长汇报的一页版

可以这样说：

```text
我现在想把 DiPLaN 从 KGQA reranker 提升成一个长程 Agent planning 方法。
核心问题不是“LLM 不会生成计划”，而是 Plan-Executor Misalignment：
高层 planning/ranking 信号和 executor 当前能消费的状态-动作接口之间存在结构性错配。

我们的 WebQSP 结果已经能支持这个观察：answer-reaching action in pool 达到 99.18%，
in keep 达到 94.24%，selected 达到 89.35%，但 executed top-1 success 只有 61.01%。
这说明系统不是完全找不到正确动作，而是从“选到看起来对的动作”到“执行成成功轨迹”之间有明显断层。
我把这个断层定义成 Plan-Execution Gap，目前大约是 28.34%。

DiPLaN 的方法不是生成自由文本计划，而是在 executor 的可执行动作空间中建模 future executable trajectories，
再通过 value-guided decoder 把未来轨迹信号转化为当前一步可执行动作。
因此它解决的是两个问题：第一，减少每步 LLM 低层规划调用；第二，缓解局部 action ranking 与全局 execution success 之间的 mismatch。

接下来我建议用 KGQA 作为可控环境，用 tau-bench 作为真实 tool-agent 环境。
KGQA 证明 action 已经在候选空间但 final success 仍然掉分；tau-bench 用来验证这种 plan-execution mismatch 是否也存在于真实 API executor 和 database-state evaluation 中。
```

## 13. 最重要的写法提醒

不要这样写：

```text
We use diffusion for KGQA planning.
```

应该这样写：

```text
We identify Plan-Executor Misalignment in long-horizon agents and propose DiPLaN,
an execution-aligned future planning framework that models future executable trajectories
and decodes them into current state-conditioned actions.
```

中文版本：

```text
我们研究长程 Agent 中的 Plan-Executor Misalignment：Agent 并不是不会规划，
而是高层规划表示无法稳定映射到执行器的状态条件动作空间。
DiPLaN 通过在 executable action space 中建模 future trajectories，
将未来规划信号解码为当前可执行动作，从而提升长程执行成功率并降低高频 LLM 调用成本。
```
