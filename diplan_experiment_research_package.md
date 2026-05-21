## Material Passport

- Origin Skill: academic-research-suite (Imbad0202/academic-research-skills-codex)
- Origin Mode: experiment-agent/plan
- Origin Date: 2026-05-21T00:00:00+08:00
- Verification Status: UNVERIFIED
- Version Label: diplan_experiment_research_v1

## Experiment Research Package (DiPLaN)

### 1) Experiment Overview

- **Title**: DiPLaN：面向长程 Agent 推理的 Diffusion 计划分布学习
- **Objective**: 验证“先生成多样化未来计划分布，再价值引导执行”是否优于 step-wise reasoning 和传统搜索式 lookahead。
- **Core Hypotheses**:
  - H1: 在相同或可控计算预算下，DiPLaN 的 Success Rate 高于 step-wise/beam/search baselines。
  - H2: DiPLaN 显著降低 Trap@1，并推迟 First-Error Step。
  - H3: DiPLaN 在 Recovery@Error 与 Plan Feasibility 上显著更优。
  - H4: receding-horizon 执行优于“一次性执行完整计划”。
- **Type**: training + analysis + simulation

### 2) Scope & Stage Design

- **Stage A (MVP, 必做)**: KGQA（CWQ/WebQSP/GrailQA），验证 plan distribution generation 的核心机制。
- **Stage B (增强)**: ALFWorld/ScienceWorld，验证长动作链上的 first-error 与 recovery。
- **Stage C (增强)**: ToolBench/API-Bank/StableToolBench，验证工具选择、参数生成、可执行性。
- **Stage D (增强)**: WebArena/WebVoyager/MiniWoB++，验证动态重规划与真实环境执行偏差。

### 3) Baselines

- **Reasoning baselines**: CoT, ReAct, Reflexion, Self-Consistency
- **Search/planning baselines**: Beam Search, Tree-of-Thoughts, RAP, LATS, MCTS-style lookahead, FLARE
- **Tool-specific baselines**: ReWOO, LLMCompiler, Plan-and-Act, Pre-Act, GoalAct

### 4) Metrics (Primary/Secondary)

- **Primary**:
  - Success Rate (↑)
  - First-Error Step (↑)
  - Recovery@Error (↑)
  - Trap@1 (↓)
- **Secondary**:
  - Plan Feasibility (↑)
  - Constraint Violation Rate (↓)
  - Plan-Execution Consistency (↑)
  - Token/Latency Cost (↓)
  - Diversity-Coverage (适中且越高越好)

### 5) Ablation Matrix

- A1: 去掉 value guidance，仅条件/无条件 diffusion
- A2: 去掉 constraint checker
- A3: 一次性执行完整计划 vs receding-horizon
- A4: latent diffusion vs discrete diffusion vs LLM-only plan sampling
- A5: 固定采样数 N，比较不同 diversity 机制
- A6: oracle evaluator vs learned evaluator
- A7: 低层 action 生成 vs 层级结构化计划生成

### 6) Statistical Analysis Plan

- **比例类指标**（Success Rate/Recovery@Error/Trap@1）:
  - 主检验：两比例检验或 McNemar（配对设置）
  - 稳健性：bootstrap 95% CI（10,000 resamples）
- **连续类指标**（First-Error Step/Latency/Consistency）:
  - 主检验：Wilcoxon signed-rank（非正态时）
  - 报告效应量：Cliff's delta 或 rank-biserial
- **多重比较**: Holm-Bonferroni
- **报告规则**: 同时报告 p 值、效应量、CI，禁止只报显著性

### 7) Reproducibility Protocol

- 固定随机种子：`[13, 23, 42, 3407, 2026]`
- 固定数据切分与评测脚本版本
- 每组实验至少 5 次独立运行
- 统一预算对齐：token 上限、时间上限、工具调用上限
- 输出保存：配置文件、日志、逐步轨迹、环境版本锁定

### 8) Execution Runbook (Template)

> 当前仓库尚未提供可直接执行的训练代码，以下命令为落地模板；待你确认代码结构后可直接替换路径执行。

```bash
# 1) 数据预处理
python scripts/prepare_kgqa_data.py \
  --datasets cwq webqsp grailqa \
  --out data/processed/

# 2) 训练 plan autoencoder
python train_autoencoder.py \
  --config configs/autoencoder_kgqa.yaml \
  --out runs/ae_kgqa/

# 3) 训练 diffusion planner
python train_diffusion_planner.py \
  --config configs/diffusion_kgqa.yaml \
  --ckpt runs/ae_kgqa/best.pt \
  --out runs/diplan_kgqa/

# 4) 训练 value model + 约束检查器
python train_value_model.py --config configs/value_kgqa.yaml --out runs/value_kgqa/
python train_constraint_checker.py --config configs/constraint_kgqa.yaml --out runs/constraint_kgqa/

# 5) 评测
python evaluate.py \
  --planner_ckpt runs/diplan_kgqa/best.pt \
  --value_ckpt runs/value_kgqa/best.pt \
  --constraint_ckpt runs/constraint_kgqa/best.pt \
  --benchmarks cwq webqsp grailqa \
  --out results/main/

# 6) 消融
python run_ablation.py --config configs/ablation_kgqa.yaml --out results/ablation/

# 7) 统计检验
python stats_report.py --in results/ --out results/stats/
```

### 9) 12-Week Implementation Plan

- Week 1-2: 文献整理、任务选择、数据 schema 设计、评测协议冻结
- Week 3-4: 构建 KGQA/ALFWorld 轨迹数据与计划抽取器
- Week 5-6: 训练 plan autoencoder + 初版 diffusion planner
- Week 7-8: 训练 value model/constraint checker，形成离线 planner
- Week 9: 接入 receding-horizon executor
- Week 10: 主实验 + baseline + 消融
- Week 11: 机制分析（Trap@1, First-Error, Recovery, Failure Type）
- Week 12: 论文写作、附录、代码整理与投稿稿

### 10) Risk Register & Mitigation

- **离散动作扩散不稳定**: 先做 KGQA path-token MVP；采用 latent diffusion 与 DDIM
- **计划不可执行**: schema-constrained decoder + feasibility projection
- **评估器不可靠**: 多评估器集成 + oracle 子任务校准
- **采样成本高**: 限步采样、trajectory memory、蒸馏策略
- **计划-执行脱节**: receding-horizon 重规划 + deviation monitor

### 11) Deliverables Checklist

- D1: 数据与计划轨迹构建脚本
- D2: DiPLaN 训练与推理代码
- D3: Baseline 对照与统一评测脚本
- D4: 消融结果表与统计检验报告
- D5: 错误机制诊断图（Trap@1/First-Error/Recovery）
- D6: 可复现实验包（配置、日志、版本锁定）
