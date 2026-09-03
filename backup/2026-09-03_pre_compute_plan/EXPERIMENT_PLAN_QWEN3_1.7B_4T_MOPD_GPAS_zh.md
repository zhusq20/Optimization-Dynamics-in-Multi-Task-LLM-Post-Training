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
- 回复长度上限 `L_max`（默认 8,192）；截断的回复保留并正常参与 loss。MOPD 的 token 级信号在任意前缀上成立，不需要完整回复；
- 目标权重、计时 EMA decay 和共同 loss threshold；
- 种子：Uniform 与 GPAS 各 3 个（最少 2 个），Cost-GPAS 与 RawNoise 各 1 个。种子改变 prompt 洗牌和采样；同一种子内各方法的 prompt 流保持匹配。

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

同时记录未缩放的 `||g_i,s||^2` 与 `||a_i||^2`，用于离线比较 raw 与 scaled 的任务排序。

## 5. AdamW 二阶矩与分配的关系（不做训练 run）

conventional AdamW 的二阶矩观测 `A**2` 含噪声项 `sum_i w_i**2 Var(g_i) / m_i`，随分配变化，这与任何改变 batch 组成的做法相同。在 `2 <= m_i <= 8`、Uniform 为 4 的边界内，每个任务的噪声项相对 Uniform 在 `[0.5, 2]` 倍之间；二阶矩经 EMA 再开方进入更新，因此噪声主导坐标上更新幅度的变化最坏不超过 `sqrt(2)`。

可选的分配无关观测为 `U = sum_i (w_i/m_i) sum_s g_i,s**2`，`v_obs = U/G`，`E[U/G] = (1/G) sum_i w_i E[g_i**2]` 不依赖 `m`；在比例分配 `m_i = G w_i` 下与 conventional 的噪声尺度一致。受控检查：Uniform 改为 GPAS 后 conventional 二阶矩相对变化 `0.543`，`U/G` 变化 `0.0009`。

决定：所有训练 run 使用 conventional AdamW，不跑 Uniform-TW / GPAS-TW。`U/G` 只作为附录说明。省下的两个 run 用于主对比的种子重复。

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

| 配置 | `m_i` 规则 | 种子数 | 作用 |
|---|---|---:|---|
| Uniform | `[4,4,4,4]` | 3 | 基线；其 checkpoint 供 held-out 方差检查与离线反事实分配 |
| GPAS | `w_i * sqrt(e_i)` | 3 | 同步数下的 held-out loss |
| Cost-GPAS | `argmin_m J_C(m)` | 1 | GPU-hour 效率 |
| RawNoise | `w_i * sqrt(Var(g_i))`（未缩放） | 1，条件触发 | 优化器坐标是否重要 |

所有配置使用 conventional AdamW、相同的学习率与 schedule、相同的计数边界，第一步都是 Uniform。

RawNoise 的触发条件：第 9 节的 held-out 方差检查中，RawNoise 分配的 held-out scaled variance 相对 Uniform 明显高于 GPAS（例如 GPAS < 0.9 而 RawNoise > 1.0），或阶段 0 日志中 raw 与 scaled 排序不一致的步数比例可观。若两者排序基本一致，不跑 RawNoise，论文在 5.1 节用日志说明“两种信号在本设定下给出相同分配”。

最少算力方案：Uniform 与 GPAS 各 2 个种子 + Cost-GPAS 1 个 + RawNoise 1 个 = 6 个 run，与原计划相同。

各任务使用独立的 matched prompt stream。配置对任务 `i` 发出的第 `n` 个 prompt 必须相同；自适应配置在同一步到达不同的 per-task stream 前缀是算法本身的结果。

## 8. 系统执行和记账

每步访问全部四个教师。教师放置、加载顺序和 overlap 策略在配置间相同。若只有一个教师槽，按固定循环顺序处理教师，并保留本步最后一个教师作为下一步的 resident teacher。

每次任务访问记录：

- load、offload、teacher ready 和 transfer tail；
- rollout、teacher scoring、backward、optimizer 时间；
- attempted responses、valid tokens、generated tokens；
- peak HBM 和 GPU 数量。

GPU-hour 使用完整 wall time 乘预留 GPU 数量，包括加载和等待。主系统结论只使用端到端 GPU-hour，不用成本 proxy 代替。

报告 `C / sum_i(m_i * tau_i)` 和线性时间模型 `C + sum_i m_i tau_i` 对实测步时间的拟合残差。

生成时间：每个任务块的生成时间由块内最长回复决定，上限是 `L_max` 个 token 的解码时间，因此进入 `C` 而不是 `tau_i`；教师打分和 backward 与 token 数成正比，进入 `tau_i`。阶段 0 分别记录生成、打分、backward 三部分时间。若生成时间是 `C` 的主要部分且 `C / sum_i m_i tau_i` 偏大，把 `L_max` 降到 4,096 或按 token 长度分位数选取；截断回复保留，所有方法使用同一 `L_max`。并发教师槽下，fixed-objective loss scaling 不变，GPAS 不变；理想重叠时 `t(m)=C+max_i(m_i*tau_i)`，Cost-GPAS 必须按该时间模型重新推导和求解。

## 9. 指标、机制检查和主要结果

### 9.1 主指标：held-out teacher loss

每个任务固定 128 个不在训练流中的 prompt，构成 held-out 集。每 50 步（含第 0 步和第 500 步，共 11 个点）从 checkpoint 离线生成一条回复（固定采样种子、与训练相同的温度），用对应教师打分，得到各任务的 `L_i` 与加权 `F = sum_i w_i L_i`。所有方法、所有种子使用同一 held-out 集和同一采样种子，因此比较在 prompt 级别配对。

训练 batch 上的 loss 只进附录：不同方法在同一步消耗的 prompt 和每任务回复数不同，不可直接比较。

共同 loss 阈值 = Uniform（各种子平均）的最终 held-out `F`。到达阈值的 GPU-hour 在相邻 checkpoint 间线性插值。

### 9.2 机制检查：held-out 梯度方差

取 Uniform 的 early / middle / late checkpoint（约第 50、250、500 步）。每个 checkpoint、每个任务用 held-out prompt 生成 32 个新 micro-batch，`D` 取该 checkpoint 的 AdamW 状态，记录每个 micro-batch 的 `||D g||^2` 与两半的 `||D a||^2`。一半估计 `e_i` 并给出各规则的分配 `m`（Uniform、GPAS、Cost-GPAS、RawNoise），另一半算 `sum_i w_i**2 e_i / m_i`，相对 Uniform 分配报告；交换两半取平均。同时报告 `e_i` 的两半相对误差，并用 2 个和 4 个 micro-batch 的子集模拟在线估计的误差。只存标量范数，不存梯度向量。

### 9.3 主图与主表

- 图 2（噪声与分配动态）：(a) 各任务 scaled 与 raw 的 `e_hat_i(t)`；(b) GPAS 与 Cost-GPAS 的 `m_i(t)`；(c) `H(t)` 与上界 2；(d) `tau_i(t)` 与 `C / sum_i m_i tau_i`。伴随文字给出 `e_hat_i` 的跨任务范围与漂移、raw/scaled 排序不一致的步数比例、各任务触达计数边界的比例、`H` 的中位数与四分位。
- 表（held-out 方差）：三个 checkpoint × {Uniform, RawNoise, GPAS, Cost-GPAS} 的相对方差。
- 图 3（效率）：(a) held-out `F` 对 optimizer step，Uniform 与 GPAS，带种子带；(b) held-out `F` 对 GPU-hour，Uniform、GPAS、Cost-GPAS。
- 主表：初始学生、各教师、Uniform、GPAS、Cost-GPAS（、RawNoise）的 MATH-500 greedy pass@1、固定 LiveCodeBench slice pass@1、IFBench strict accuracy、GPQA-Diamond average@4，归一化增益 `(s - s_init)/(s_teacher - s_init)` 及其平均，最终 held-out `F`，到阈值的 GPU-hour。每个任务相对 Uniform 的差异与平均值一起列出。若 MATH-500 或 GPQA-Diamond 对 1.7B 学生缺少 headroom，预先增加 AIME24/25 average@16 或第二个科学集。
- 附录：各任务 held-out 与训练 loss 曲线、prompt 消耗、回复长度与截断率、计数边界占用、step-time 分位数。

如果 held-out 方差检查中 GPAS 相对 Uniform 的方差比接近 1，说明本设定下各任务噪声差异太小，GPAS 没有作用空间；这一结论直接写进结果。

## 10. 与 D3-MOPD 的边界

D3-MOPD 根据 loss gap 或下降速度改变任务混合比例，因此改变有效目标权重。这里的 `w_i` 固定，`m_i` 只控制每个任务梯度均值的方差。

若 `G < 2M`，本方法不适用；任务子采样需要另行设计 inclusion-probability correction。

## 11. 实验阶梯

按顺序执行，每一阶段的输出决定下一阶段是否需要。

| 阶段 | 内容 | 产出 | 决定 |
|---|---|---|---|
| 0 | 一个 Uniform run（种子 1），开全日志：每个 micro-batch 的 `||D g||^2`、`||g||^2`，每任务的 `||D a||^2`、`||a||^2`，计时；每 50 步存 checkpoint | 离线算出 `e_hat_i(t)`（scaled 与 raw）、GPAS 与 Cost-GPAS 的反事实 `m*(t)`、`H(t)`、`tau_i`、`C`；完成 9.2 的 held-out 方差检查 | `H` 的中位数和 held-out 方差比决定 GPAS 是否有作用空间；raw/scaled 排序一致率决定是否跑 RawNoise；`C / sum m_i tau_i` 和 `tau_i` 的离散度决定 Cost-GPAS 的预期收益；生成时间在 `C` 中的占比决定是否下调 `L_max` |
| 1 | GPAS（种子 1） | 图 3(a) 的第一对曲线 | 与 Uniform 的 held-out 差异方向 |
| 2 | Cost-GPAS（种子 1） | 图 3(b) | — |
| 3 | Uniform 与 GPAS 各再跑 2 个种子 | 种子带与配对区间 | — |
| 4 | RawNoise（种子 1），仅在触发条件满足时 | 主表最后一行 | — |

阶段 0 的日志本身已能产出图 2 的全部面板（GPAS 的实际计数除外），也是引言里“噪声差 X 倍、漂移 Y 倍”这句话的来源。

## 12. 实现清单

- micro-batch 按任务分组；每个 micro-batch 的 loss 先做 token → response 两层平均，再乘 `w_i / m_i`。
- 每个 micro-batch backward 后：用当前 AdamW 状态 `D` 计算并记录 `||D g_i,s||^2`（以及 `||g_i,s||^2`）。在 FSDP / ZeRO 下每个 shard 局部计算，再 all-reduce 一个标量。需要一个参数大小的临时 buffer 累加任务均值 `a_i`，任务块结束时记录 `||D a_i||^2`。
- 若 loss 已乘 `w_i / m_i`，记录前除以 `(w_i / m_i)^2` 还原。
- 每步结束：更新 `e_hat_i`、`tau_i`、`C` 的 EMA，计算下一步的 `m`，日志写入 `m`、`H`、各任务 loss、tokens、attempted responses、计时。
- checkpoint 保存 `m`、`e_hat_i`、EMA 状态、prompt 位置；resume test 验证下一步分配与参数更新可复现。
- 不需要额外的前向或反向；不需要修改优化器。
