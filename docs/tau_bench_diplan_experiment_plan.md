# tau-bench 接入 DiPLaN 的最小实验计划

## 目标

验证 DiPLaN 在长程 tool-agent 场景中的两个核心主张：

```text
1. Efficiency:
   diffusion future proposal 可以减少每步 LLM 低层规划调用。

2. Mismatch:
   action/rank 看起来合理，但 final execution success 低，说明存在 plan-execution mismatch。
```

## 为什么选择 tau-bench

tau-bench 是 tool-agent-user interaction benchmark，包含：

```text
domain-specific API tools
simulated user interaction
business rules / policy constraints
database state transition
final task success evaluation
```

这比 KGQA 更像长程 Agent：

```text
KGQA action = relation
tau-bench action = tool call

KGQA executor = graph traversal
tau-bench executor = API tool environment

KGQA success = answer entity reached
tau-bench success = final database state matches goal
```

## 阶段 1：只做 tool-name selection

第一阶段不要直接做完整 tool arguments，否则工程量会变大。

先把动作定义为：

```text
action = tool name
```

例如：

```text
get_user_details
list_orders
update_order_address
cancel_order
```

训练目标：

```text
给定 user goal + dialogue state + tool history，
预测下一步应该调用哪个 tool。
```

这对应当前 KGQA 中：

```text
relation selection
```

## 阶段 2：加入 tool argument sketch

第二阶段再把动作扩展成：

```text
action = tool name + argument schema sketch
```

例如：

```text
update_order_address(order_id=?, address=?)
```

先不要求生成精确参数值，只预测：

```text
需要哪些参数槽位
```

## 阶段 3：完整 tool call

最后才做：

```text
action = executable tool call with concrete arguments
```

这一步最接近真实 agent，但也最难。

## 数据格式建议

把 tau-bench 轨迹转成类似 DiPLaN 的 JSONL：

```json
{
  "task_id": "tau_retail_0001",
  "domain": "retail",
  "goal": "...",
  "state_text": "...",
  "tool_history": [
    "get_user_details",
    "list_orders"
  ],
  "candidate_actions": [
    "get_user_details",
    "list_orders",
    "update_order_address",
    "cancel_order"
  ],
  "oracle_action": "update_order_address",
  "oracle_future_actions": [
    "update_order_address",
    "send_confirmation"
  ],
  "success": true
}
```

这和 KGQA 里的结构一一对应：

```text
question -> goal
executed_prefix -> tool_history
candidate_relations -> candidate_actions
oracle_next_relation -> oracle_action
oracle_path suffix -> oracle_future_actions
```

## 模型复用

可以直接复用 DiPLaN 的三类模块思想：

```text
1. candidate_diffusion
   预测当前候选 tool action 分布。

2. trajectory_diffusion
   预测未来 tool-call trajectory 分布。

3. fusion/action decoder
   把 local score、future score、value score 融合成当前 action ranking。
```

第一版可以不复用 KG relation embedding，而是把 tool/action tokenize 成文本：

```text
tool name + argument schema + natural language description
```

## 实验对照

### Baseline A: LLM every step

每一步都让 LLM 选择下一步 tool。

指标：

```text
success rate
LLM calls / task
tokens / task
latency / task
```

### Baseline B: LLM horizon cache

LLM 每 H 步生成一次 tool plan cache。

例如：

```text
H = 3
```

后续步骤优先从 cache 中选择 action。

### Method C: DiPLaN future proposal

LLM 只提供高层语义或初始 candidate cache。

DiPLaN 每一步做：

```text
candidate action scoring
future trajectory scoring
value-guided decoding
```

### Method D: DiPLaN + LLM-on-demand

当 DiPLaN 不确定时才调用 LLM。

触发条件：

```text
top1-top2 margin < threshold
candidate entropy > threshold
trajectory/value disagreement
dead-end or invalid action risk
```

## 指标

### 效率指标

```text
LLM calls / task
tokens / task
latency / task
success / token
success / LLM call
```

### mismatch 指标

```text
Action Recall@k
Action Selected Rate
Executable Transition Success
Final Task Success
Plan-Execution Gap = Action Recall@k - Final Task Success
```

### 长程稳定性指标

```text
pass^k
recovery rate
invalid tool call rate
state rollback / collateral damage
```

## Plan-Executor Mismatch 如何量化

tau-bench / tool-agent 场景里，可以把 mismatch 拆成四个可量化 gap：

```text
1. Action Feasibility Gap
   planner proposes action
   但该 action 在当前状态下不可执行或参数不满足 precondition。

2. Transition Realization Gap
   action 被成功调用
   但执行后的 database/tool state 没有朝目标状态推进。

3. Local-Global Gap
   当前 tool call 局部合理
   但整条 tool-call trajectory 最终失败。

4. Feedback Update Gap
   executor 返回 error / empty result / policy violation
   但 planner 下一步没有根据反馈修正计划。
```

对应指标：

```text
proposed_action_valid_rate
tool_call_success_rate
state_progress_rate
final_task_success
recovery_after_error_rate
plan_execution_gap = proposed_action_valid_rate - final_task_success
transition_gap = tool_call_success_rate - state_progress_rate
```

这比只报：

```text
success rate
```

更能证明我们的研究问题：

```text
rank/plan 看起来合理，不代表 execution 成功。
```

## 预期论文结果

理想现象不是简单地说 DiPLaN accuracy 最高，而是显示：

```text
1. LLM every step token 最贵。
2. DiPLaN 能减少 LLM 调用。
3. action-level rank/recall 高不等于 final success。
4. 加入 future trajectory + value decoder 后，Plan-Execution Gap 缩小。
```

## 最小实现顺序

```text
Step 1:
下载 tau-bench，先选 retail 或 airline 一个 domain。

Step 2:
导出 tool-call trajectories，统一成 JSONL。

Step 3:
只做 tool-name action selection，不做参数。

Step 4:
训练 candidate_diffusion_tool。

Step 5:
训练 trajectory_diffusion_tool。

Step 6:
训练 fusion/action decoder。

Step 7:
跑 LLM every step vs DiPLaN proposal 的 token/success 对比。
```

## 风险

```text
1. tau-bench 轨迹如果没有 gold tool-call trace，需要从 successful runs 或 reference policy 中抽取。
2. tool arguments 比 tool name 难很多，所以第一版只做 tool name。
3. 如果只做 tool name，审稿人可能说执行不完整，需要在 limitation 里说明。
4. 需要明确 DiPLaN 是 low-level proposal，不是替代完整 LLM agent。
```

## 推荐汇报说法

```text
我准备把 KGQA 作为可控环境，tau-bench 作为真实 tool-agent 环境。
KGQA 证明 answer/action 已在候选空间但 final success 仍低；
tau-bench 用来验证这种 rank-execution mismatch 是否也存在于真实 tool execution。

DiPLaN 的目标不是完全替代 LLM，而是把高频低层 planning amortize 成 diffusion-based future proposal，
在减少 LLM 调用的同时保持或提升 execution success。
```
