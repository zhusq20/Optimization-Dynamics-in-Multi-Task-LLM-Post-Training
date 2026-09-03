# Qwen3-1.7B 四教师 MOPD：micro-batch GPAS 实验协议

> 2026-09-03 修订：按四条约束修订——18 卡（10×48GB + 8×96GB A6000 级）、两周期限（9/17 前完成）、每配置单种子、基线齐全。math 和 IF 教师为 Qwen3-1.7B 学生经领域 RL 得到的同源专家，code 和 science 教师为原版 Qwen3-4B（`enable_thinking=False`，不做额外训练，2026-09-03 定）；学生与教师都用 non-thinking 模式。修订前版本在 `backup/2026-09-03_pre_compute_plan/`。

## 0. 本次修订摘要

| 项 | 原计划 | 现计划 |
|---|---|---|
| 种子 | Uniform、GPAS 各 3，其余 1 | 全部 1；不确定性用 held-out prompt 的配对 bootstrap |
| 教师 | 未定 | math、IF：Qwen3-1.7B 领域 RL 专家，同源；code、science：原版 Qwen3-4B，non-thinking |
| 系统 | 单教师槽轮换加载 | 每个 run 2 卡，四教师常驻推理卡；`C` 不含教师加载 |
| 配置 | 4 个，RawNoise 条件触发 | 6 个，一轮并发：Uniform、GPAS、Cost-GPAS、RawNoise、LossGap、StdMOPD |
| `L_max` | 8,192 | 4,096 |
| 排程 | 四阶段阶梯 | 训练前三项测量 + 冒烟测试 + 单轮并发 |

方法定义量不变：`G=16`、`b=4`、`m∈[2,8]`、500 步、每任务 16,000 prompt、held-out 每任务 128 prompt、每 50 步一个 checkpoint。

## 1. 固定项

所有配置使用相同的：

- 学生：Qwen3-1.7B，non-thinking 模式，固定 checkpoint revision、tokenizer、精度；
- 教师：math 和 IF 教师是 Qwen3-1.7B，由同一学生 checkpoint 经各自领域的 RL 得到（同源、同 tokenizer、non-thinking），所用 checkpoint 步数写进论文 pipeline 表；code 和 science 教师是原版 `Qwen/Qwen3-4B`，`enable_thinking=False`，不做任何额外训练。它与学生 tokenizer.json 逐字节相同、chat template 逐字相同，打分时不需要单独格式化 prompt。选原版而不选 Instruct-2507 的理由是让四个域的教师-学生差距量级接近：按 Qwen3 技术报告的 non-thinking 分数，code 约 +10 到 +14、science 约 +13，与 IF 教师的 +12 和 math 教师的 +5 同量级，避免某个域主导目标。两个 RL 教师的 RL prompt 集、四条 16,000 prompt 训练流、held-out 集、四个 benchmark 两两不重叠；
- 每个 micro-batch `b=4` 个 prompt，每个 prompt 采样一条回复（与 MOPD 的 `N=1` 一致）；
- 每步 `G=16` 个 micro-batch，`m_min=2`，`m_max=8`；
- token → response → micro-batch 的两层平均顺序；
- 总预算 500 步 × 64 条回复 = 32,000 attempted responses，学习率以 optimizer step 为时钟；
- 每个任务独立且固定的 prompt 顺序；每个任务 16,000 个不循环的 prompt；
- 回复长度上限 `L_max = 4,096`；截断的回复保留并正常参与 loss。non-thinking 回复极少超过 2,000 token，上限的作用是封住重复循环回复对生成时间的拖尾。MOPD 的 token 级信号在任意前缀上成立，不需要完整回复；
- 采样温度、目标权重、计时 EMA decay `rho = 0.9`、共同 loss threshold；
- 种子：每个配置 1 个。种子改变 prompt 洗牌和采样；同一种子内各方法的 prompt 流保持匹配。

每一步都处理四个任务。Uniform 分配是 `[4,4,4,4]`。自适应方法只改变每个任务的 micro-batch 数 `m_i`，且始终满足 `sum(m)=16`。

每个 rollout batch 只做一次更新（`K=1`），教师 log-prob 不跨更新复用。即使一个任务始终取 `m_max=8`，也只消耗 `500*8*4=16,000` 个 prompt，因此所有任务都按不放回顺序读取且不循环；Uniform 每任务消耗 8,000 个 prompt。`m_max=8` 限制单个任务最多占每步的一半，也限制计数的单步变化。由于 Uniform 的计数为 4 且 `m_i<=8`，任意非负噪声下都有 `H<=2`。

## 2. 固定目标和 loss scaling

训练目标为

```text
F(theta) = sum_i w_i * ell_i(theta)
w_i = (1 / ell_i(0)) / sum_j (1 / ell_j(0))
```

`ell_i(0)` 在训练前用 held-out 集测量：初始学生对任务 `i` 的 128 个 held-out prompt 各采样一条回复，教师 `i` 打分，取平均 sampled-token reverse KL。math 和 IF 的同源 RL 教师与学生的 KL 可能很小，code 和 science 的 4B 教师 KL 会明显更大，各任务差异可能很大，因此加一条规则：若 `max_i ell_i(0) / min_i ell_i(0) > 10`，改用等权 `w_i = 1/4`，并把论文 Setting 节 "inverse initial loss" 一句改为等权。四个 `ell_i(0)` 和最终采用的 `w` 写进论文 pipeline 表。测量脚本：`experiments/measure_initial_kl.py`（vLLM 生成 + HF 打分，单卡约 10 分钟）。教师 RL 日志里没有对初始策略的 KL（KL 系数为 0），不能替代这一步。

每个 micro-batch 先独立完成内部平均。任务 `i` 的每个 micro-batch loss 在 backward 前乘 `w_i / m_i`。因此一步的一阶矩观测是

```text
A = sum_i (w_i / m_i) * sum_s g_i,s
```

所有配置的条件期望均为同一个 `sum_i w_i E[g_i]`。不得把整步所有 token 直接合并平均，因为这会把有效任务权重改成接近 `m_i / G`；StdMOPD 基线正是故意这样做，见第 7 节。

## 3. 在线分配

第一步使用 Uniform，并从该步初始化每个任务的噪声和时间统计。GPAS 计算

```text
score_i = w_i * sqrt(e_i)
```

先求与 score 成比例的连续计数，再限制到 `[m_min,m_max]`，最后用最大余数法取整，保证总和为 `G`。RawNoise 和 LossGap 只换 score，裁剪与取整相同。

时间模型为

```text
t(m) = C + sum_i m_i * tau_i
J_C(m) = t(m) * sum_i w_i**2 * e_i / m_i
```

`C` 是每步固定的 rollout、权重同步、优化器和日志开销，`tau_i` 是任务 `i` 每个 micro-batch 的打分加 backward 时间。墙钟预算为 `T` 时，步数为 `T/t(m)`，平均梯度方差为 `V(m)t(m)/T`，所以 Cost-GPAS 最小化 `J_C`。当 `C=0` 时才有 `m_i ∝ w_i * sqrt(e_i/tau_i)`；实际四任务实验枚举满足边界和总数约束的整数分配。`tau_i` 与 `C` 用实测时间的 EMA。

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

决定：所有训练 run 使用 conventional AdamW，不跑 Uniform-TW / GPAS-TW。`U/G` 只作为附录说明。

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

| 配置 | 每步分配 / loss | 作用 |
|---|---|---|
| Uniform | `[4,4,4,4]`，固定目标 loss | 基线；其 checkpoint 供 held-out 方差检查与离线反事实分配 |
| GPAS | `m_i ∝ w_i * sqrt(e_i)` | 同步数下的 held-out loss |
| Cost-GPAS | `argmin_m J_C(m)` | GPU-hour 效率 |
| RawNoise | `m_i ∝ w_i * sqrt(Var(g_i))`（未缩放） | 优化器坐标是否重要 |
| LossGap | `m_i ∝ w_i * Lbar_i`，`Lbar_i` 为任务 `i` 训练 batch 教师损失的 EMA（decay 0.9） | gap-following 信号在固定目标下的对照，回答"为什么不按 loss 分配" |
| StdMOPD | 每任务每步 16 个 prompt（与 Uniform 相同），一步内所有有效 token 直接求均值；不做 per-response 平均，不乘 `w_i/m_i` | 现行常用配方；有效任务权重随 token 数变化，目标与 `F` 不同 |

所有配置 1 个种子、conventional AdamW、相同学习率与 schedule、相同 `L_max`、相同计数边界；四个自适应配置第一步都是 Uniform。RawNoise 无条件跑；若日志显示 raw 与 scaled 排序在多数步一致，结果节如实写"两种信号在本设定下给出相近分配"。

LossGap 只用 gap，不用 D3-MOPD 的下降速度项，保持单参数。`w_i * Lbar_i` 在 inverse-initial-loss 权重下等于相对剩余损失 `Lbar_i / ell_i(0)`（差一个常数），即把 micro-batch 推向相对进展最小的任务。

StdMOPD 的实现：rollout 结束后已知本步有效 token 总数 `T`，每个 micro-batch 的 loss 取 token 损失之和除以 `T` 再 backward，累加后即为精确的一步 token-mean。StdMOPD 与 Uniform 每步消耗完全相同的 prompt，二者只差 loss 聚合方式。它的加权 `F` 与其他配置不可比，只比各任务 held-out `L_i` 和 benchmark。

各任务使用独立的 matched prompt stream。配置对任务 `i` 发出的第 `n` 个 prompt 必须相同；自适应配置在同一步到达不同的 per-task stream 前缀是算法本身的结果。

## 8. 系统执行和记账

每个 run 固定占 2 卡，所有 run 的卡型相同：

- 训练卡（96GB）：学生全参微调。bf16 参数 + fp32 master + AdamW 状态约 24 GB；梯度 scratch、任务累加器 `B_i`、步累加器 `A` 三个参数大小张量约 17 GB（fp32）；加激活，总计约 50 GB；
- 推理卡（48GB）：vLLM 学生 rollout（预算约 20 GB；non-thinking 下 64 条回复的 KV 约 5 GB）+ 四个教师常驻：两个 1.7B 共 6.8 GB，两个 4B 共 16 GB，bf16 合计约 23 GB，整卡约 43 GB。若显存紧张，把两个 4B 教师移到训练卡，训练卡余量约 40 GB。教师按固定顺序依次给各自任务的回复打分：prefill-only 前向，取采样 token 的 log-prob。

一步的流程：训练卡把权重同步到推理卡（3.4 GB）→ 64 条回复一次性 rollout → 四教师顺序打分 → 训练卡按任务、按 micro-batch 前反向 → 优化器更新。rollout 时间由最长回复决定，与分配 `m` 基本无关；打分和 backward 与 token 数成正比。因此时间模型保持 `t(m) = C + sum_i m_i tau_i`：`tau_i` = 任务 `i` 每个 micro-batch 的打分 + backward 时间；`C` = rollout + 权重同步 + 优化器 + 日志。Cost-GPAS 按此模型枚举整数分配。报告 `C / sum_i m_i tau_i` 和线性模型对实测步时间的拟合残差。

预期：non-thinking 回复短、教师不超过 4B，`C` 会明显大于 `sum_i m_i tau_i`（估计比值 1.5–2），Cost-GPAS 的解会接近 GPAS，GPU-hour 节省主要来自 GPAS 本身。论文 Cost-GPAS 段和摘要按实测比值写，不预设大幅节省。

每步每任务记录：rollout、teacher scoring、backward、optimizer 时间；attempted responses、valid tokens、generated tokens；peak HBM。GPU-hour = 完整 wall time × 2，包括同步与等待。主系统结论只使用端到端 GPU-hour，不用成本 proxy 代替。

Uniform、GPAS、Cost-GPAS 三个进入 GPU-hour 对比的配置必须在同型 slot 上跑；本方案 8 个 slot 卡型一致，自动满足。

## 9. 指标、机制检查和主要结果

### 9.1 主指标：held-out teacher loss

每个任务固定 128 个不在训练流中的 prompt，构成 held-out 集。每 50 步（含第 0 步和第 500 步，共 11 个点）从 checkpoint 离线生成一条回复（固定采样种子、与训练相同的温度），用对应教师打分，得到各任务的 `L_i` 与加权 `F = sum_i w_i L_i`。所有方法使用同一 held-out 集和同一采样种子，因此比较在 prompt 级别配对。

不确定性：每个 checkpoint 上，方法间 `F` 之差的区间用 512 个 held-out prompt 的配对 bootstrap（1,000 次重采样）给出。它反映 held-out 采样，不反映种子间差异，论文里说明一次。

训练 batch 上的 loss 只进附录：不同方法在同一步消耗的 prompt 和每任务回复数不同，不可直接比较。

共同 loss 阈值 = Uniform 的最终 held-out `F`。到达阈值的 GPU-hour 在相邻 checkpoint 间线性插值。StdMOPD 不参与阈值比较。

### 9.2 机制检查：held-out 梯度方差

取 Uniform 的 early / middle / late checkpoint（第 50、250、500 步）。每个 checkpoint、每个任务用 held-out prompt 生成 32 个新 micro-batch，`D` 取该 checkpoint 的 AdamW 状态，记录每个 micro-batch 的 `||D g||^2` 与两半的 `||D a||^2`。一半估计 `e_i` 并给出各规则的分配 `m`（Uniform、RawNoise、LossGap、GPAS、Cost-GPAS；LossGap 用该 checkpoint 的 held-out `L_i`），另一半算 `sum_i w_i**2 e_i / m_i`，相对 Uniform 分配报告；交换两半取平均。同时报告 `e_i` 的两半相对误差，并用 2 个和 4 个 micro-batch 的子集模拟在线估计的误差。只存标量范数，不存梯度向量。

### 9.3 主图与主表

- 图 2（噪声与分配动态）：(a) 各任务 scaled 与 raw 的 `e_hat_i(t)`；(b) GPAS 与 Cost-GPAS 的 `m_i(t)`；(c) `H(t)` 与上界 2；(d) `tau_i(t)` 与 `C / sum_i m_i tau_i`。伴随文字给出 `e_hat_i` 的跨任务范围与漂移、raw/scaled 排序不一致的步数比例、各任务触达计数边界的比例、`H` 的中位数与四分位。Uniform 的日志本身能产出除 (b) 实际计数外的全部面板。
- 表（held-out 方差）：三个 checkpoint × {Uniform, RawNoise, LossGap, GPAS, Cost-GPAS} 的相对方差。
- 图 3（效率）：(a) held-out `F` 对 optimizer step，Uniform、GPAS、RawNoise、LossGap，带配对 bootstrap 带；(b) held-out `F` 对 GPU-hour，Uniform、GPAS、Cost-GPAS。
- 主表：初始学生、各教师、StdMOPD、Uniform、RawNoise、LossGap、GPAS、Cost-GPAS 的 MATH-500 greedy pass@1、固定 LiveCodeBench slice pass@1、IFBench strict accuracy、GPQA-Diamond average@4，归一化增益 `(s - s_init)/(s_teacher - s_init)` 及其平均，最终 held-out `F`，到阈值的 GPU-hour。每个任务相对 Uniform 的差异与平均值一起列出。StdMOPD 的 `F` 与 GPU-hour 两列留空或标注不可比。
- 附录：各任务 held-out `L_i` 表（含 StdMOPD）、各任务 held-out 与训练 loss 曲线、prompt 消耗、回复长度与截断率、计数边界占用、step-time 分位数。

如果 held-out 方差检查中 GPAS 相对 Uniform 的方差比接近 1，说明本设定下各任务噪声差异太小，GPAS 没有作用空间；这一结论直接写进结果。

### 9.4 训练前的两项测量

- `ell_i(0)`：按第 2 节规则决定 `w`。
- 教师与初始学生的四个 benchmark 分数：若某 benchmark 上 `s_teacher - s_init < 3` 分，归一化增益不可靠，换成或增加一个同域、差距足够的 benchmark，并在论文 Capability 段说明。这一步同时确认 benchmark 与两个 RL 教师的 prompt 集、训练流无重叠。4B 教师的差距按技术报告预期 code +10 以上、science +13，仍要在同一评测器下实测。

## 10. 与 D3-MOPD 的边界

D3-MOPD 根据 loss gap 或下降速度改变任务混合比例，因此改变有效目标权重。这里的 `w_i` 固定，`m_i` 只控制每个任务梯度均值的方差。LossGap 是它的信号在固定目标框架下的对照，不是 D3-MOPD 的复现。

若 `G < 2M`，本方法不适用；任务子采样需要另行设计 inclusion-probability correction。

## 11. 排程

| 日期 | 内容 |
|---|---|
| 9/4–9/5 | 确认 math 和 IF 教师的 checkpoint 步数，下载 `Qwen/Qwen3-4B`；搭 MOPD 训练器 |
| 9/6–9/8 | 训练器、rollout 权重同步、教师打分、held-out 评测脚本完成；教师与初始学生 benchmark；测 `ell_i(0)` 定 `w`；冒烟测试 |
| 9/9 | 冻结配置；8 个 slot 一轮并发启动 |
| 9/10 | 收 run；held-out `F` 曲线；held-out 方差检查；6 个模型 benchmark |
| 9/11–9/12 | 图 2、图 3、各表；填论文占位 |
| 9/13–9/17 | 余量：重跑失败配置；如需多种子，每轮约半天 |

冒烟测试：GPAS 配置跑 20 步，检查每步时间、回复长度分布与截断率、峰值显存、`e_hat_i` 逐步相对变化、resume test。步时超过 2 分钟或截断率超过 5% 时再调 `L_max`。

单轮并发：8 个 slot 同时启动 6 个配置；剩余 2 个 slot 跑 Uniform 与 GPAS 的第二个种子，可选，不进主文，除非后续决定报告多种子。剩余 2 张 48GB 卡做 held-out 评测与 benchmark。

原阶段 0 的判据不再决定跑不跑，只决定结果怎么写：`H` 的中位数和 held-out 方差比决定 GPAS 作用空间的表述；raw/scaled 排序一致率决定 RawNoise 段的表述；`C / sum m_i tau_i` 和 `tau_i` 的离散度决定 Cost-GPAS 段的表述。

## 12. 实现清单

- micro-batch 按任务分组；每个 micro-batch 的 loss 先做 token → response 两层平均，再乘 `w_i / m_i`。StdMOPD 改为 token 损失之和除以本步有效 token 总数。
- micro-batch 内按回复逐条 backward 累加（4 次），避免 4 条回复的 logits 同时驻留；累加完成后再取范数。
- 每个 micro-batch backward 后：用当前 AdamW 状态 `D` 计算并记录 `||D g_i,s||^2`（以及 `||g_i,s||^2`），把 `g_i,s` 加进任务累加器 `B_i`，清零 `.grad`。任务块结束时由 `B_i / m_i` 记录 `||D a_i||^2`，并把 `w_i B_i / m_i` 加进步累加器 `A`。单卡训练，不需要 all-reduce。
- 若 loss 已乘 `w_i / m_i`，记录前除以 `(w_i / m_i)^2` 还原。
- 每步结束：更新 `e_hat_i`、`Lbar_i`、`tau_i`、`C` 的 EMA，计算下一步的 `m`，日志写入 `m`、`H`、各任务 loss、tokens、attempted responses、计时；把权重同步到推理卡。
- checkpoint：每 50 步存权重（3.4 GB）；Uniform 在第 50、250、500 步另存优化器状态（各约 20 GB）供 held-out 方差检查；每个 run 保留最新一份完整状态用于 resume。checkpoint 保存 `m`、`e_hat_i`、EMA 状态、prompt 位置；resume test 验证下一步分配与参数更新可复现。
- 不需要额外的前向或反向；不需要修改优化器。

## 13. 算力与存储预算

| 项目 | 估算 |
|---|---|
| 单个 run | 2 卡：96GB 训练卡 + 48GB 推理卡 |
| 每步时间 | 约 1 分钟：rollout 25–40 s（由最长回复决定，`L_max=4,096` 封顶约 40 s）；四教师打分约 15 s，两个 4B 教师各约为 1.7B 的两倍；学生前反向约 15 s（约 45k token）；同步 + 优化器 + 日志约 10 s |
| 单 run 墙钟 | 500 步约 8–10 h，含每 50 步存档 |
| 一轮 8 slot | 6 配置 + 2 备用种子，约 10 h，约 160 GPU-h |
| 18 卡两周容量 | 约 6,000 GPU-h |
| 存储 | 权重 checkpoint 3.4 GB × 11 × 8 ≈ 300 GB；优化器状态 3 × 20 GB；日志与 held-out 输出可忽略 |

算力不是约束，工程时间是：训练器、权重同步、打分、日志、resume test 必须在 9/8 前跑通。步时与长度分布以冒烟测试实测为准，上表只用于排程。
