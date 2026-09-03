# Qwen3-1.7B 四教师 MOPD：micro-batch GPAS 实验协议

## 1. 固定项

所有配置使用相同的：

- Qwen3-1.7B 初始 checkpoint、tokenizer、精度、并行和硬件；
- 数学、代码、指令跟随、科学四个教师及各自的 prompt 池；
- 每个 micro-batch `b=4` 个 prompt，每个 prompt 采样一条回复（与 MOPD 的 `N=1` 一致）；
- 每步 `G=16` 个 micro-batch，`m_min=2`，`m_max=8`；
- token → response → micro-batch 的两层平均顺序；
- 总预算 500 步 × 64 条回复 = 32,000 attempted responses，学习率以 optimizer step 为时钟；
- 每个任务独立且固定的 prompt 顺序；
- 每个任务 16,000 个不循环的 prompt；
- 目标权重、计时 EMA decay 和共同 loss threshold；
- 每个配置使用一个训练种子。

每一步都处理四个任务。Uniform 分配是 `[4,4,4,4]`。自适应方法只改变每个任务的 micro-batch 数 `m_i`，且始终满足 `sum(m)=16`。

每个 rollout batch 只做一次更新（`K=1`），教师 log-prob 不跨更新复用。训练共 500 步。即使一个任务始终取 `m_max=8`，也只消耗 `500*8*4=16,000` 个 prompt，因此所有任务都按不放回顺序读取且不循环；Uniform 每任务消耗 8,000 个 prompt。`m_max=8` 限制单个任务最多占每步的一半，也限制计数的单步变化。由于 Uniform 的计数为 4 且 `m_i<=8`，任意非负噪声下都有 `H<=2`。

## 2. 固定目标和 loss scaling

训练目标为

```text
F(theta) = sum_i w_i * ell_i(theta)
w_i = (1 / ell_i(0)) / sum_j (1 / ell_j(0))
```

四个任务的 `ell_i(0)` 都为正，因此实验不使用分母下限。接近零的初始 KL（例如 self-teacher）不在本实验范围内，需要改用 absolute KL 或预先固定的其他尺度。

每个 micro-batch 先独立完成内部平均。任务 `i` 的每个 micro-batch loss 在 backward 前乘 `w_i / m_i`。因此一步的一阶矩观测是

```text
A = sum_i (w_i / m_i) * sum_s g_i,s
```

所有配置的条件期望均为同一个 `sum_i w_i E[g_i]`。不得把整步所有 token 直接合并平均，因为这会把有效任务权重改成接近 `m_i / G`。

## 3. 在线分配

第一步使用 Uniform，并从该步初始化每个任务的噪声和时间统计。GPAS 计算

```text
score_i = w_i * sqrt(e_i)
```

先求与 score 成比例的连续计数，再限制到 `[m_min,m_max]`，最后用最大余数法取整，保证总和为 `G`。

单教师槽的时间模型为

```text
t(m) = C + sum_i m_i * tau_i
J_C(m) = t(m) * sum_i w_i**2 * e_i / m_i
```

`C` 是每步固定的加载、卸载、同步和优化器开销，`tau_i` 是任务 `i` 每个 micro-batch 的边际关键路径时间。墙钟预算为 `T` 时，步数为 `T/t(m)`，平均梯度方差为 `V(m)t(m)/T`，所以 Cost-GPAS 最小化 `J_C`。当 `C=0` 时才有 `m_i ∝ w_i * sqrt(e_i/tau_i)`；实际四任务实验枚举满足边界和总数约束的整数分配。默认使用实测关键路径时间，无法计时时才使用 generated tokens。

## 4. 噪声估计

在本步开始时冻结 `D = 1 / (sqrt(v) + eps)`。任务 `i` 的 unweighted micro-batch 梯度为 `g_i,s`，任务均值为 `a_i = mean_s(g_i,s)`。记录

```text
e_hat_i = (sum_s ||D g_i,s||^2 - m_i ||D a_i||^2) / (m_i - 1)
e_i     = e_hat_i
tau_i   = rho * tau_i + (1-rho) * tau_hat_i
C       = rho * C + (1-rho) * C_hat
```

`e_hat_i` 汇总约 `10^9` 个坐标；独立坐标近似下相对误差量级为 `sqrt(2/d_eff)`。噪声直接使用当步估计，日志报告其逐步相对变化。回复长度重尾，因此 `tau_i` 和 `C` 保留 EMA。

## 5. 可选的 Taskwise AdamW

一阶矩使用 `A`。二阶矩使用

```text
U = sum_i (w_i / m_i) * sum_s g_i,s**2
v_obs = U / G
```

`E[U/G] = (1/G) sum_i w_i E[g_i**2]` 只依赖模型状态，不依赖分配。在比例分配 `m_i=G w_i` 下，其噪声项与 conventional AdamW 完全同尺度；直接使用 `U` 会在噪声主导坐标上把二阶矩放大约 `G=16` 倍，并把有效更新缩小约 4 倍。GPAS 可直接搭配 conventional AdamW，taskwise AdamW 只作为可选的一致性修正；用 matched Uniform/GPAS × Conv./TW 诊断它是否实际重要。实现需要一个持久的参数大小 `U` buffer 和一个跨任务复用的参数大小 `a_i` 临时 buffer。

## 6. 受控检查

受控三任务实验固定 `G=16`、`m_min=2`、`m_max=12`。整数分配为：

| 方法 | `(m_1,m_2,m_3)` | AdamW-scaled variance / Uniform | time × variance / Uniform |
|---|---:|---:|---:|
| Uniform | `(6,5,5)` | 1.000 | 1.000 |
| Raw-noise | `(11,3,2)` | 2.316 | 2.084 |
| GPAS | `(3,3,10)` | 0.679 | 0.913 |
| Cost-GPAS | `(3,5,8)` | 0.734 | 0.857 |

20,000 次 Monte Carlo 的结果与计算值一致。另一个 200,000-step 检查把 Uniform 改为 GPAS 后的 AdamW moment：conventional 二阶矩相对变化为 `0.543`，taskwise 二阶矩变化为 `0.00092`，后者的理论变化为零。

## 7. 端到端配置

| 配置 | `m_i` 规则 | AdamW 二阶矩 |
|---|---|---|
| Uniform-Conv. | `[4,4,4,4]` | conventional |
| GPAS | `w_i * sqrt(e_i)` | conventional |
| Cost-GPAS | `argmin_m J_C(m)` | conventional |
| RawNoise | `w_i * sqrt(Var(g_i))` | conventional |
| Uniform-TW | `[4,4,4,4]` | taskwise，使用 `U/G` |
| GPAS-TW | `w_i * sqrt(e_i)` | taskwise，使用 `U/G` |

主比较为 Uniform、GPAS 和 Cost-GPAS，三者都使用 conventional AdamW。RawNoise 是唯一的分配信号对照；matched Uniform/GPAS × Conv./TW 是唯一的优化器状态诊断，不再扩展其他 allocation ablation。GPAS 对 Uniform 检验同一步数下的下降；Cost-GPAS 对 GPAS 和 Uniform 检验 GPU-hour 效率。

各任务使用独立的 matched prompt stream。配置对任务 `i` 发出的第 `n` 个 prompt 必须相同；自适应配置在同一步到达不同的 per-task stream 前缀是算法本身的结果。匹配 prompt 流用于配对比较。

## 8. 系统执行和记账

每步访问全部四个教师。教师放置、加载顺序和 overlap 策略在配置间相同。若只有一个教师槽，按固定循环顺序处理教师，并保留本步最后一个教师作为下一步的 resident teacher。

每次任务访问记录：

- load、offload、teacher ready 和 transfer tail；
- rollout、teacher scoring、backward、optimizer 时间；
- attempted responses、valid tokens、generated tokens；
- peak HBM 和 GPU 数量。

GPU-hour 使用完整 wall time 乘预留 GPU 数量，包括加载和等待。主系统结论只使用端到端 GPU-hour，不用成本 proxy 代替。

报告 `C / sum_i(m_i * tau_i)`。并发教师槽下，fixed-objective loss scaling 不变，GPAS 不变；理想重叠时 `t(m)=C+max_i(m_i*tau_i)`，Cost-GPAS 必须按该时间模型重新推导和求解。

## 9. 机制检查和主要结果

Uniform trajectory 的 early、middle、late checkpoint 各采集一组冻结的独立 micro-batch。每组分半：一半估计 `e_i` 和分配，另一半测加权梯度方差，再交换两半。报告：

- 当步 `e_hat_i` 与 held-out noise 的误差，以及 `e_hat_i` 的逐步相对变化；
- Uniform、GPAS、Cost-GPAS 的 held-out AdamW-scaled variance；
- `H = V(m_uniform) / V(m_current)` 的时间序列；
- conventional 和 taskwise 二阶矩相对固定 target 的变化；
- 每个任务处于 `m_min` 或 `m_max` 的步数比例。

第一张主要训练图为加权 teacher loss 对 optimizer step，比较 GPAS 与 Uniform。第二张为加权 teacher loss 对 GPU-hour，比较 Cost-GPAS、GPAS 与 Uniform。共同 loss threshold 按本预算下 Uniform 可达到的最终损失预先确定。generated tokens 只进入主表；主表还报告最终总体与分任务 loss、达到共同 loss threshold 所需的 tokens 和 GPU-hour、吞吐、peak HBM、固定开销比、teacher transfer 时间和 step-time quantile。

能力评测报告初始学生、每个 specialist teacher 和所有训练配置的 MATH-500 greedy pass@1、固定 LiveCodeBench slice pass@1、IFBench strict accuracy 和 GPQA-Diamond average@4，并用 `(s-s_init)/(s_teacher-s_init)` 汇总跨域归一化分数。若 MATH-500 或 GPQA-Diamond 对 1.7B 学生缺少 headroom，预先增加 AIME24/25 average@16 或第二个科学集。另报告各任务平均回复长度、8,192 上限下的截断率和实际 prompt 消耗。

如果整段训练的 `H` 都接近 1，结论应是采用 Uniform，而不是把局部差异解释为有效收益。

## 10. 与 D3-MOPD 的边界

D3-MOPD 根据 loss gap 或下降速度改变任务混合比例，因此改变有效目标权重。这里的 `w_i` 固定，`m_i` 只控制每个任务梯度均值的方差。

若 `G < 2M`，本方法不适用；任务子采样需要另行设计 inclusion-probability correction。
