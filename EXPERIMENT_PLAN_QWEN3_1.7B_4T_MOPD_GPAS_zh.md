# Qwen3-1.7B 四教师 MOPD：GPAS 精简实验方案

> 版本：2026-09-01  
> 主模型：Qwen3-1.7B  
> 任务：Math、Code、Instruction Following、ScienceQA  
> 核心方法：moment-consistent AdamW + online GPAS / Cost-GPAS

## 1. 实验目标

固定四个任务的 teacher-loss 权重后，实验只回答三个问题：

1. GPAS 能否比 Uniform 和 Gradient-Norm IS 更有效地分配训练 batch？
2. moment-consistent AdamW 能否保持预期的一阶矩和二阶矩目标？
3. Cost-GPAS 能否用更少 GPU 时间达到相同的 weighted teacher loss？

实验分为三个部分：两项小型机制检查、一次真实梯度快照和四个端到端 MOPD 对照。小型检查用于核对公式与实现，论文的主要结论来自端到端 MOPD 结果。

## 2. 方法实现

设任务 `i` 的 objective weight 为 `lambda_i`，采样概率为 `q_i`，importance multiplier 为：

```text
w_i = lambda_i / q_i
```

每步抽取一个任务并生成 task-homogeneous batch。moment-consistent AdamW 使用：

```text
m <- beta1 * m + (1-beta1) * w_i * G_i
v <- beta2 * v + (1-beta2) * w_i * G_i^2
```

任务的预条件梯度尺度和耗时使用在线 EMA：

```text
S_i <- gamma * S_i + (1-gamma) * ||G_i / (sqrt(v_bar)+eps)||^2
C_i <- gamma * C_i + (1-gamma) * step_seconds
```

其中 `v_bar` 是当前 optimizer update 之前的 second moment。proposal 为：

```text
GPAS:       p_i proportional to lambda_i * sqrt(S_i)
Cost-GPAS:  p_i proportional to lambda_i * sqrt(S_i) / sqrt(C_i)
```

四任务 probability floor 使用：

```text
q_i = (1 - 4*rho) * p_i + rho
```

默认 `gamma=0.95`、`rho=0.05`。实现只维护每个任务的 scale EMA 和 time EMA，不增加额外 rollout、teacher call、forward 或 backward。

## 3. 理论与实验的对应关系

| 论文结论 | 实验观测 | 结果用途 |
|---|---|---|
| moment-consistent update 保持目标矩 | 固定 proposal 下比较 standard 与 moment-consistent moment estimate | 核对公式与 optimizer 实现 |
| GPAS 使用与 AdamW 更新更相关的尺度 | 真实 checkpoint 上比较 raw norm、preconditioned norm 和 proposal | 解释任务采样差异 |
| GPAS 改善训练效率 | Uniform、Gradient-Norm IS 与 GPAS 的 loss-token 曲线 | 验证核心方法 |
| Cost-GPAS 改善时间效率 | GPAS 与 Cost-GPAS 的 loss-time 曲线 | 验证成本修正 |

## 4. 两项机制检查

### 4.1 三任务构造例

- 使用三个零均值 Gaussian task gradients，每个任务主导一个坐标。
- 设置不同的 raw gradient scale 与 AdamW-preconditioned scale。
- 比较 Uniform、Gradient-Norm IS、GPAS 和 Cost-GPAS 的 proposal 与 AdamW-scaled estimator error。
- 该结果只用于展示 raw gradient norm 与 optimizer-aware scale 的区别。

### 4.2 AdamW moment 检查

- 选择一个固定非均匀 proposal。
- 使用同一批 task draws 比较 standard second-moment update `w_i^2 G_i^2` 与 moment-consistent update `w_i G_i^2`。
- 同时检查 first-moment estimate。
- 该结果只用于确认代数和代码实现。

## 5. 一次真实梯度快照

从 Uniform MOPD run 的训练中段保存一个 checkpoint。每个任务采集少量 batch，计算：

- raw gradient norm；
- AdamW-preconditioned gradient norm；
- GPAS 与 Cost-GPAS proposal；
- 每个任务的平均完整 step time。

该快照用于展示真实 LLM 训练中四个任务的尺度和耗时差异，并解释 GPAS 的采样概率。无需保存完整训练过程的梯度，也不增加独立训练 run。

## 6. 端到端 MOPD 对照

每个方法执行一次 matched run。所有方法使用同一初始 checkpoint、数据顺序、token budget、optimizer 配置和 objective weights。

| 方法 | Objective | Proposal | 作用 |
|---|---|---|---|
| Uniform | `1/4` | `1/4` | 基线 |
| Gradient-Norm IS | `1/4` | `lambda * raw_norm` | 与常用 importance sampling 比较 |
| GPAS | `1/4` | `lambda * preconditioned_norm` | 核心方法 |
| Cost-GPAS | `1/4` | `lambda * preconditioned_norm / sqrt(step_time)` | GPU 时间版本 |

四个 run 均使用 moment-consistent AdamW。standard AdamW 的差异由第 4.2 节直接检查，不再增加端到端消融 run。

正式训练长度为 400 steps，每步目标为 65,536 valid response tokens。round-robin warm start 使用 8 steps，随后启用对应 proposal。warm start 和正式阶段均计入训练成本。

## 7. 训练设置与指标

- Student、tokenizer、teacher、数据和 prompt template 使用固定版本。
- response cap 为 8,192，temperature 为 1，top-p 为 1。
- shared gradient clip 在 importance correction 之前执行，所有方法使用相同 threshold。
- 每 50 steps 在相同 held-out prompts 上评估 teacher loss。
- 训练结束后评估 MATH-500、LiveCodeBench、IFBench 和 GPQA-Diamond。

主结果只报告：

1. weighted teacher loss vs valid response tokens；
2. weighted teacher loss vs GPU hours；
3. final weighted loss 和四个任务的 final loss；
4. 达到 Uniform final loss 所需的 GPU hours；
5. 四个下游 benchmark 的最终结果。

## 8. 最小日志与复现信息

每步记录：

```text
global_step, task_id,
lambda_i, q_i, lambda_i/q_i,
valid_tokens, total_step_seconds,
raw_grad_norm, preconditioned_grad_norm,
scale_ema, cost_ema,
teacher_loss, clip_flag, overflow_flag
```

run manifest 记录模型、teacher、tokenizer 和数据版本，以及 optimizer 配置、clip threshold、硬件、并行策略、代码 commit 和 evaluation commit。

实现前只需确认三项：

- `q` 的总和为 1，且每个任务满足 probability floor；
- `q=lambda` 时 moment-consistent AdamW 与普通 AdamW 更新一致；
- 中断恢复后 optimizer state 与 proposal state 能正常加载。

## 9. 论文结论标准

- GPAS 的 loss-token 曲线优于 Uniform，支持采样效率结论。
- Cost-GPAS 达到相同 loss 所需的 GPU hours 少于 GPAS 和 Uniform，支持时间效率结论。
- 四个任务的 final loss 与下游 benchmark 用于确认收益来自整体训练进展。
- 摘要只使用真实 MOPD 结果；机制检查只用于解释方法。
