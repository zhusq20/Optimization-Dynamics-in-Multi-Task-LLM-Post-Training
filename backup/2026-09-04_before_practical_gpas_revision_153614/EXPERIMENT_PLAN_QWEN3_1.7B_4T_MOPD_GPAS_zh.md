# Qwen3-1.7B 四教师 MOPD：micro-batch GPAS 实验协议

> 2026-09-04 修订：与论文当前方法矩阵同步。全文只有一个 GPAS；`GPAS (per step)` 与 `GPAS (per GPU hour)` 是同一方法的两种分配模式。端到端实验共 8 个单种子配置，并包含 estimator × allocation 的 2×2。在线训练噪声与固定 rollout 后独立重抽 scoring token 的诊断量严格分开。math 和 IF 教师的精确 checkpoint、优化器超参数及尚未实测的结果均保留为 TODO。修订前版本在 `backup/2026-09-03_pre_compute_plan/`。

## 0. 本次修订摘要

| 项 | 原计划 | 现计划 |
|---|---|---|
| 种子 | Uniform、GPAS 各 3，其余 1 | 全部 1；不确定性用 held-out prompt 的配对 bootstrap |
| 教师 | 未定 | math、IF：Qwen3-1.7B 领域 RL 专家，同源；code、science：原版 Qwen3-4B，non-thinking |
| 系统 | 单教师槽轮换加载 | 每个 run 2 卡，四教师常驻推理卡；`C` 不含教师加载 |
| 配置 | 4 个，旧基线条件触发 | 8 个，一轮并发；见第 7 节的完整方法行，其中 GPAS 只有 per-step 与 per-GPU-hour 两种模式 |
| `L_max` | 8,192 | 4,096 |
| 排程 | 四阶段阶梯 | 训练前测量 + 冒烟测试 + 单轮并发 |

方法定义量不变：`G=16`、`b=4`、`m∈[2,8]`、500 步、每任务 16,000 prompt、held-out 每任务 128 prompt、每 50 步一个 checkpoint。

## 1. 固定项

所有配置使用相同的：

- 学生：Qwen3-1.7B，non-thinking 模式；`TODO：填写冻结的 checkpoint revision、tokenizer revision 和精度`；
- 教师：math 和 IF 教师是 Qwen3-1.7B，由同一学生 checkpoint 经各自领域的 RL 得到（同源、同 tokenizer、non-thinking）；code 和 science 教师是原版 `Qwen/Qwen3-4B`，`enable_thinking=False`，不做额外训练。`TODO：填写两个 RL 教师的精确 checkpoint/revision 与 RL recipe，以及 Qwen3-4B 的冻结 revision`。确认所有教师与学生的 tokenizer 文件和 chat template 一致后再冻结配置；
- 数据隔离：两个 RL 教师的 RL prompts、四条 16,000-prompt 训练流、held-out 集和四个 benchmarks 两两不重叠；`TODO：填写数据集名称、版本、许可、过滤规则及去重检查输出`；
- 每个 micro-batch `b=4` 个 prompt，每个 prompt 采样一条回复（与 MOPD 的 `N=1` 一致）；
- 每步 `G=16` 个 micro-batch，`m_min=2`，`m_max=8`；
- token → response → micro-batch 的两层平均顺序；
- 总预算 500 步 × 64 条回复 = 32,000 attempted responses，学习率以 optimizer step 为时钟；
- 每个任务独立且固定的 prompt 顺序；每个任务 16,000 个不循环的 prompt；
- 回复长度上限 `L_max = 4,096`；截断的回复保留并正常参与 loss，上限用于约束异常长回复的生成成本。MOPD 的 token 级信号在任意前缀上成立，不需要完整回复；
- 默认训练估计量为论文定义的 top-k 估计量，`k=16`：在学生 top-k 集上精确求和，真实 rollout token 仅在落到集合外时提供尾部修正。只有方法行明确写 `Sampled-token estimator` 或 `StdMOPD` 时才使用通常的 sampled-token 形式；
- 目标权重、计时 EMA decay `rho = 0.9`、共同 loss threshold；`TODO：填写 rollout/evaluation temperature、完整 decoding 设置和 importance-ratio truncation 规则`；
- 种子：每个配置 1 个。种子改变 prompt 洗牌和采样；同一种子内各方法的 prompt 流保持匹配。

每一步都处理四个任务。Uniform 分配是 `[4,4,4,4]`。自适应方法只改变每个任务的 micro-batch 数 `m_i`，且始终满足 `sum(m)=16`。

每个 rollout batch 只做一次更新（`K=1`），教师 log-prob 不跨更新复用。即使一个任务始终取 `m_max=8`，也只消耗 `500*8*4=16,000` 个 prompt，因此所有任务都按不放回顺序读取且不循环；Uniform 每任务消耗 8,000 个 prompt。`m_max=8` 限制单个任务最多占每步的一半，也限制计数的单步变化。由于 Uniform 的计数为 4 且 `m_i<=8`，任意非负噪声下都有 `H<=2`。

## 2. 固定目标和 loss weighting

训练目标为

```text
F(theta) = sum_i w_i * ell_i(theta)
w_i = (1 / ell_i(0)) / sum_j (1 / ell_j(0))
```

`ell_i(0)` 在训练前用 held-out 集测量：初始学生对任务 `i` 的 128 个 held-out prompt 各采样一条回复，教师 `i` 打分，取平均 sampled-token reverse KL。四个 `ell_i(0)` 和由其倒数归一化得到的固定 `w` 写进论文 pipeline 表。测量脚本为 `experiments/measure_initial_kl.py`；`TODO：填入实测的四个初始损失和最终权重`。教师 RL 日志里没有对初始策略的 KL（KL 系数为 0），不能替代这一步。

每个 micro-batch 先独立完成内部平均。任务 `i` 的每个 micro-batch loss 在 backward 前乘 `w_i / m_i`。除明确写 sampled-token 的两项消融和 StdMOPD 外，下面的 `g_i,s` 都指实际 top-k 训练估计量产生的 micro-batch gradient。因此一步的一阶矩观测是

```text
A = sum_i (w_i / m_i) * sum_s g_i,s
```

对任一固定训练估计量 `a`，Uniform 与所有自适应 count allocations 的条件期望均为同一个 `sum_i w_i E[g_i^a]`，因此 counts 不改变该 estimator 的固定加权 expected semi-gradient。sampled-token 与 top-k 之间只主张 fixed-prefix 条件无偏，不额外声称二者在随机 response length 下的实际 response-normalized trajectory mean 严格相等。不得把整步所有 token 直接合并平均，因为这会让有效任务权重随 token 数变化；StdMOPD 基线正是用这种常见聚合方式，见第 7 节。

## 3. 在线分配

全文只有一个 GPAS。`GPAS (per step)` 与 `GPAS (per GPU hour)` 分别报告它在固定步数和固定 GPU-hour 口径下的两种分配模式，不另起方法名。所有自适应计数配置第一步使用 Uniform，并从该步初始化各任务的在线噪声和时间统计。GPAS 的 per-step 模式计算

```text
score_i = w_i * sqrt(e_i)
```

先求与 score 成比例的连续计数，再限制到 `[m_min,m_max]`，最后用最大余数法取整，保证总和为 `G`。`Counts from unpreconditioned noise` 只在算计数时令 `D=I`；`Counts from loss gap` 使用 `w_i * Lbar_i` 作为 score。二者与 GPAS 共用裁剪和取整过程，但都是描述性基线名，不是 GPAS 的版本。

时间模型为

```text
t(m) = C + sum_i m_i * tau_i
J_C(m) = t(m) * sum_i w_i**2 * e_i / m_i
```

`tau_i` 吸收随任务 `i` 的计数变化的边际 rollout、教师打分和 backward 时间；`C` 只包含与计数无关的权重同步、优化器和日志时间。墙钟预算为 `T` 时，步数为 `T/t(m)`，平均梯度方差为 `V(m)t(m)/T`，所以 GPAS 的 per-GPU-hour 模式枚举所有满足边界和总数约束的整数分配并最小化 `J_C`。只有在连续、无边界且 `C=0` 的松弛问题里才有 `m_i ∝ w_i * sqrt(e_i/tau_i)`；真实四任务实验不使用该闭式解。`tau_i` 与 `C` 由训练 trace 的 EMA 得到，最终系统结论使用包含全部等待的端到端 GPU-hour，并报告加性时间模型的拟合残差。

## 4. 在线训练噪声与独立 scoring-token 诊断

### 4.1 在线训练噪声

在本步开始时冻结 `D = 1 / (sqrt(v_pre) + eps)`。任务 `i` 的 unweighted micro-batch gradient 为 `g_i,s`，任务样本均值为 `a_i = mean_s(g_i,s)`。默认的 Uniform、GPAS 两种模式和两个描述性计数基线均使用实际训练中的 top-k gradient：每个位置的 top-k 集贡献精确求和，真实生成的 rollout token 只有在 top-k 外时才提供尾部修正。在线循环不额外重抽 scoring token。记录

```text
e_hat_i^train = (sum_s ||D g_i,s||^2 - m_i ||D a_i||^2) / (m_i - 1)
e_i           = e_hat_i^train
tau_i   = rho * tau_i + (1-rho) * tau_hat_i
C       = rho * C + (1-rho) * C_hat
```

`e_hat_i^train` 是当前训练估计量在真实 rollout 轨迹上的单 micro-batch 预条件噪声估计，并直接生成下一步的计数。`Sampled-token estimator (GPAS per step)` 必须用本 run 实际 sampled-token gradients 的 `e_hat_i^train`，不能借用 top-k run 的噪声。日志报告在线估计的逐步相对变化；`tau_i` 和 `C` 保留 EMA。

同时记录未预条件的 `||g_i,s||^2` 与 `||a_i||^2`，用于比较未预条件与预条件噪声的任务排序。每步另从 top-k 分数记录三个低成本标量：学生概率加权的 log-ratio 均值（gap level）、同一加权下的 log-ratio 方差（within-position spread）和 `1 - top-k probability mass`（tail mass）。这三个标量只是 token 机制诊断，不是 gradient variance。

### 4.2 固定 rollout 后的独立 scoring-token 诊断

checkpoint 诊断先固定 prompts、完整 responses、全部 prefixes 和 response lengths，再在每个固定 prefix 独立抽一个 `S ~ q(.|z)`；sampled-token 与 top-k 两个估计量共用同一个 `S`，top-k 形式只在 `S` 位于集合外时把它用于尾部修正。该离线重抽样构造 `e_{i,diag}^samp` 与 `e_{i,diag}^topk`，其差

```text
Delta_{i,diag} = e_{i,diag}^samp - e_{i,diag}^topk
```

只度量固定 rollout、prefix 和长度条件下的 scoring-token 方差变化；不预设符号，也不与在线 `e_hat_i^train` 数值等同。在线 top-k 训练仍使用真实 rollout token，不使用这里的独立 `S`。完整 checkpoint 协议见第 9.2 节；端到端 2×2 则由第 7 节明确列出的四个训练 run 构成。

## 5. AdamW 二阶矩与分配的关系（不做训练 run）

conventional AdamW 的逐坐标二阶矩观测 `A ⊙ A` 含噪声项 `sum_i w_i**2 Var(g_i) / m_i`，随分配变化，这与任何改变 batch 组成的做法相同。在 `2 <= m_i <= 8`、Uniform 为 4 的边界内，每个任务的噪声项相对 Uniform 在 `[0.5, 2]` 倍之间；二阶矩经 EMA 再开方进入更新，因此噪声主导坐标上更新幅度的变化最坏不超过 `sqrt(2)`。

可选的分配无关观测为 `U = sum_i (w_i/m_i) sum_s (g_i,s ⊙ g_i,s)`，`v_obs = U/G`，`E[U/G] = (1/G) sum_i w_i E[g_i ⊙ g_i]` 不依赖 `m`；在比例分配 `m_i = G w_i` 下与 conventional 的噪声尺度一致。受控检查：Uniform 改为 GPAS (per step) 后 conventional 二阶矩相对变化 `0.543`，`U/G` 变化 `0.0009`。

决定：所有训练 run 使用 conventional AdamW，不跑 Uniform-TW / GPAS-TW。`U/G` 只作为附录说明。

## 6. 受控检查

受控三任务实验固定 `G=16`、`m_min=2`、`m_max=12`。整数分配为：

| 方法/模式 | `(m_1,m_2,m_3)` | Preconditioned variance / Uniform | Predicted time × variance / Uniform |
|---|---:|---:|---:|
| Uniform | `(6,5,5)` | 1.000 | 1.000 |
| Counts from unpreconditioned noise | `(11,3,2)` | 2.316 | 2.084 |
| GPAS (per step) | `(3,3,10)` | 0.679 | 0.913 |
| GPAS (per GPU hour) | `(3,5,8)` | 0.734 | 0.857 |

这是具有自身任务数、计数边界、合成 task costs 且 `C=0` 的三任务受控计算，不是四教师配置，也不是实测 GPU-hour。20,000 次 Monte Carlo 的结果与计算值一致。另一个 200,000-step 检查把 Uniform 改为 GPAS (per step) 后的 AdamW moment：conventional 二阶矩相对变化为 `0.543`，alternative `U/G` 观测变化为 `0.00092`，后者的理论变化为零。

## 7. 端到端配置

| 训练配置（8 行） | 训练估计量与每步分配 | 作用 |
|---|---|---|
| Uniform | top-k；`[4,4,4,4]`；固定目标 loss | top-k + Uniform 的主基线，也是端到端 2×2 的一个单元 |
| GPAS (per step) | top-k；`m_i ∝ w_i * sqrt(e_i^train,topk)` 后裁剪取整 | 固定 optimizer steps 下的 GPAS 模式；端到端 2×2 的一个单元 |
| GPAS (per GPU hour) | top-k；`argmin_m J_C(m)` | 同一 GPAS 的 cost-aware 模式，用于端到端 GPU-hour 效率 |
| Counts from unpreconditioned noise | top-k；仅算计数时令 `D=I` | 隔离优化器 preconditioner 是否改变计数 |
| Counts from loss gap | top-k；`score_i = w_i * Lbar_i`，`Lbar_i` 为训练 batch 教师损失的 EMA | 固定目标下的 gap-following 描述性基线 |
| Sampled-token estimator (Uniform) | sampled-token；`[4,4,4,4]`；固定目标 loss | sampled-token + Uniform 的端到端 2×2 单元 |
| Sampled-token estimator (GPAS per step) | sampled-token；用本 run 的 `e_i^train,samp` 生成 GPAS counts | sampled-token + GPAS 的端到端 2×2 单元 |
| StdMOPD | sampled-token；每任务每步 16 个 prompt（与 Uniform 相同），一步内所有有效 token 直接求均值；不做 per-response 平均，不乘 `w_i/m_i` | 常见配方；有效任务权重随 token 数变化，目标与 `F` 不同 |

全部 8 个配置各跑 1 个种子，使用 conventional AdamW、相同学习率与 schedule、相同 `L_max`、相同计数边界；所有自适应计数配置第一步都是 Uniform。`TODO：填写 AdamW 超参数、精度和 sharding`。`Counts from unpreconditioned noise` 无条件跑；若日志显示未预条件与预条件噪声的排序在多数步一致，结果节如实写“两种信号在本设定下给出相近分配”。

`Counts from loss gap` 只用 gap，不用 D3-MOPD 的下降速度项。`w_i * Lbar_i` 在 inverse-initial-loss 权重下等于相对剩余损失 `Lbar_i / ell_i(0)`（差一个常数），即把 micro-batch 推向相对进展最小的任务。

以下四个 matched-step runs 构成端到端 estimator × allocation 的 2×2：`{sampled-token, top-k} × {Uniform counts, GPAS (per step) counts}`。每个 GPAS run 只使用本 run 实际训练估计量的在线 micro-batch noise 生成计数；不得把 top-k run 的噪声复用于 sampled-token run，反之亦然。

StdMOPD 的实现：rollout 结束后已知本步有效 token 总数 `T`，每个 micro-batch 的 loss 取 token 损失之和除以 `T` 再 backward，累加后即为精确的一步 token-mean。StdMOPD 与 Uniform 每步消耗完全相同的 prompt，二者只差 loss 聚合方式。它的加权 `F` 与其他配置不可比，只比各任务 held-out `L_i` 和 benchmark。

各任务使用独立的 matched prompt stream。配置对任务 `i` 发出的第 `n` 个 prompt 必须相同；自适应配置在同一步到达不同的 per-task stream 前缀是算法本身的结果。

## 8. 系统执行和记账

每个 run 固定占 2 卡，所有 run 的卡型相同：

- 训练卡（96GB）：学生全参微调。bf16 参数 + fp32 master + AdamW 状态约 24 GB；梯度 scratch、任务累加器 `B_i`、步累加器 `A` 三个参数大小张量约 17 GB（fp32）；加激活，总计约 50 GB；
- 推理卡（48GB）：vLLM 学生 rollout（预算约 20 GB；non-thinking 下 64 条回复的 KV 约 5 GB）+ 四个教师常驻：两个 1.7B 共 6.8 GB，两个 4B 共 16 GB，bf16 合计约 23 GB，整卡约 43 GB。若显存紧张，把两个 4B 教师移到训练卡，训练卡余量约 40 GB。默认 top-k run 中，教师一次前向返回学生 top-k tokens 以及真实 rollout token（若它在集合外）的 log-prob；sampled-token 消融只使用真实 rollout token 的 log-prob。

一步的流程：训练卡把权重同步到推理卡（3.4 GB）→ 64 条回复 rollout → 四教师顺序打分 → 训练卡按任务、按 micro-batch 前反向 → 优化器更新。时间模型保持 `t(m) = C + sum_i m_i tau_i`：`tau_i` 由 trace 拟合并吸收任务 `i` 随计数变化的边际 rollout、打分和 backward 工作；`C` 只吸收与计数无关的同步、优化器和日志。GPAS (per GPU hour) 按该拟合模型枚举整数分配。报告 `C / sum_i m_i tau_i` 和模型对实测 step time 的拟合残差；端到端 GPU-hour 另包含全部等待。若并行执行使加性模型不适用，`TODO：冻结并报告替代的拟合 step-time 函数`。

不预设 `C / sum_i m_i tau_i`、两种 GPAS 模式的接近程度或 GPU-hour 节省。`TODO：训练完成后填入实测 overhead ratio、共同 loss threshold 下的 GPU-hour 差异和时间模型残差`；摘要与结果段只能引用这些实测值。

每步每任务记录：rollout、teacher scoring、backward、optimizer 时间；attempted responses、valid tokens、generated tokens；peak HBM。GPU-hour = 完整 wall time × 2，包括同步与等待。主系统结论只使用端到端 GPU-hour，不用成本 proxy 代替。

Uniform、GPAS (per step)、GPAS (per GPU hour) 三个进入 GPU-hour 对比的配置必须在同型 slot 上跑；本方案 8 个 slot 卡型一致，自动满足。

## 9. 指标、机制检查和主要结果

### 9.1 主指标：held-out teacher loss

每个任务固定 128 个不在训练流中的 prompt，构成 held-out 集。每 50 步（含第 0 步和第 500 步，共 11 个点）从 checkpoint 离线生成一条回复（固定采样种子、与训练相同的温度），用对应教师打分，得到各任务的 `L_i` 与加权 `F = sum_i w_i L_i`。所有方法使用同一 held-out 集和同一采样种子，因此比较在 prompt 级别配对。

不确定性：每个 checkpoint 上，方法间 `F` 之差的区间用 512 个 held-out prompt 的配对 bootstrap（1,000 次重采样）给出。它反映 held-out 采样，不反映种子间差异，论文里说明一次。

训练 batch 上的 loss 只进附录：不同方法在同一步消耗的 prompt 和每任务回复数不同，不可直接比较。

共同 loss 阈值 = Uniform 的最终 held-out `F`。到达阈值的 GPU-hour 在相邻 checkpoint 间线性插值。StdMOPD 不参与阈值比较。

### 9.2 机制检查：held-out 梯度方差

取 Uniform 的 early / middle / late checkpoint（第 50、250、500 步）。每个 checkpoint、每个任务从 held-out prompts 生成 32 个新 micro-batches。先按真实训练构造计算每个 actual top-k gradient：top-k 集精确求和，tail correction 使用该 response 的真实 rollout token。这批 gradients 才供应 Table 1 的 held-out `e_hat_i`、计数选择与方差评分。

将 32 个 micro-batches 随机等分为 A/B 两半。A 半估计 actual top-k training noise，并为 Uniform、Counts from unpreconditioned noise、Counts from loss gap、GPAS (per step) 和 GPAS (per GPU hour) 生成 allocation；Counts from loss gap 使用 checkpoint 的 held-out losses，per-GPU-hour 行使用 checkpoint 前冻结的训练期 `tau_i,C`。B 半仍用 actual top-k gradients 计算

```text
V_B(m_A) = sum_i w_i**2 * e_hat_{i,B}^train,topk / m_{i,A}
```

并相对 Uniform 报告。交换 A/B 后取平均，同时报告 half-sample noise estimates 的相对误差，并用与在线 count range 匹配的子集重复估计。

独立 scoring-token 诊断与上述 allocation comparison 分开计算：固定同一批 sampled responses、prefixes 和 lengths 后，每个 prefix 独立抽 `S ~ q(.|z)`，sampled-token 与 top-k diagnostic gradients 共用同一个 `S`。记录 `e_{i,diag}^samp`、`e_{i,diag}^topk` 和 `Delta_{i,diag}` 的符号与区间，但不由这些诊断量生成 Table 1 的 counts，也不把它们称作在线训练噪声。只存 scalar norms 和三项 gap diagnostics，不存 gradient vectors。

### 9.3 主图与主表

- 图 2（噪声、分配和 token 机制）：(a) 默认 top-k run 各任务在线 `e_hat_i^train,topk(t)` 的预条件与未预条件版本；(b) GPAS (per step) 与 GPAS (per GPU hour) 的 `m_i(t)`；(c) checkpoint 上 fixed-prefix response/scoring-token 诊断，和 gap level、within-position spread、tail mass、教师来源对齐。伴随文字报告跨任务范围与漂移、两种坐标排序不一致的步数比例、计数边界占用、`H` 的中位数与四分位，以及 `Delta_{i,diag}` 的符号和区间。
- 表 1（held-out top-k 方差）：三个 checkpoint × {Uniform, Counts from unpreconditioned noise, Counts from loss gap, GPAS (per step), GPAS (per GPU hour)}，按分半交叉评价后的相对预条件方差。
- 图 3（效率）：(a) estimator × allocation 四个 matched-step runs 的 held-out `F` 对 optimizer step，带配对 bootstrap 带；(b) 同一固定目标下 Uniform、GPAS (per step)、GPAS (per GPU hour) 的 held-out `F` 对 GPU-hour。
- 主表：初始学生、各教师，以及第 7 节全部 8 个训练配置的 MATH-500 greedy pass@1、固定 LiveCodeBench slice pass@1、IFBench strict accuracy、GPQA-Diamond average@4，归一化增益 `(s - s_init)/(s_teacher - s_init)` 及其平均，最终 held-out `F`，到阈值的 GPU-hour。每个任务相对 Uniform 的差异与平均值一起列出。StdMOPD 的 `F` 与 GPU-hour 两列留空或标注不可比。
- Table 3（端到端 matched-step 2×2）：`{sampled-token, top-k} × {Uniform counts, GPAS (per step) counts}` 四个训练 run 的 held-out teacher loss 与配对差异。
- 附录：各任务 held-out `L_i` 表（含 StdMOPD）、各任务 held-out 与训练 loss 曲线、prompt 消耗、回复长度与截断率、计数边界占用、step-time 分位数。

如果 held-out 方差检查中 GPAS 相对 Uniform 的方差比接近 1，说明本设定下各任务噪声差异太小，GPAS 没有作用空间；这一结论直接写进结果。

### 9.4 训练前的两项测量

- `ell_i(0)`：按第 2 节规则决定 `w`。
- 教师与初始学生的四个 benchmark 分数：记录每项归一化增益的分母 `s_teacher - s_init`，但不在看分数后更换 benchmark。`TODO：训练前冻结 benchmark 版本、切片和 decoding，并预先定义分母接近零时只报告原始分数的判据；随后用同一评测器实测教师与初始学生，不使用外部报告值代填。`

## 10. 与 D3-MOPD 的边界

D3-MOPD 根据 loss gap 或下降速度改变任务混合比例，因此改变有效目标权重。这里的 `w_i` 固定，`m_i` 只控制每个任务梯度均值的方差。`Counts from loss gap` 是它的信号在固定目标框架下的描述性对照，不是 D3-MOPD 的复现。

若 `G < 2M`，本方法不适用；任务子采样需要另行设计 inclusion-probability correction。

## 11. 排程

| 日期 | 内容 |
|---|---|
| 9/4–9/5 | 确认 math 和 IF 教师的 checkpoint 步数，下载 `Qwen/Qwen3-4B`；搭 MOPD 训练器 |
| 9/6–9/8 | 训练器、rollout 权重同步、教师打分、held-out 评测脚本完成；教师与初始学生 benchmark；测 `ell_i(0)` 定 `w`；冒烟测试 |
| 9/9 | 冻结配置；8 个 slot 一轮并发启动 |
| 9/10 | 收 8 个配置的 run；held-out `F` 曲线；held-out 方差检查；8 个训练配置最终 checkpoint 的 benchmark |
| 9/11–9/12 | 图 2、图 3、各表；填论文占位 |
| 9/13–9/17 | 余量：重跑失败配置；如需多种子，每轮约半天 |

冒烟测试：GPAS (per step) 配置跑 20 步，检查每步时间、回复长度分布与截断率、峰值显存、在线 `e_hat_i^train,topk` 的逐步相对变化、top-k 尾部修正和 resume test。步时超过 2 分钟或截断率超过 5% 时再调 `L_max`。

单轮并发：8 个 slot 同时启动第 7 节的全部 8 个单种子配置，不预留第二种子 slot。剩余 2 张 48GB 卡做 held-out 评测与 benchmark。若任一配置失败，只在余量期重跑同一冻结配置；增加训练种子属于后续独立决定，不计入本轮方法行。

训练前诊断不再决定跑不跑，只决定结果怎么写：`H` 的中位数和 held-out 方差比决定 GPAS 作用空间的表述；未预条件/预条件排序一致率决定 `Counts from unpreconditioned noise` 段的表述；`C / sum_i m_i tau_i` 和 `tau_i` 的离散度决定 GPAS per-GPU-hour 模式的表述；`Delta_{i,diag}` 只解释 fixed-prefix scoring-token 机制，不替代在线训练噪声。

## 12. 实现清单

- micro-batch 按任务分组；默认 top-k run 在每个 prefix 精确累加 top-k 项，并仅用真实 rollout token 补尾；每个 micro-batch 的 loss 先做 token → response 两层平均，再乘 `w_i / m_i`。两项 sampled-token 消融只用真实 rollout token。StdMOPD 同样用 sampled-token，但把 token 损失之和除以本步有效 token 总数。
- micro-batch 内按回复逐条 backward 累加（4 次），避免 4 条回复的 logits 同时驻留；累加完成后再取范数。
- 每个 micro-batch backward 后：用 pre-step AdamW 状态 `D` 计算并记录当前训练 estimator 的 `||D g_i,s||^2`（以及未预条件的 `||g_i,s||^2`），把 `g_i,s` 加进任务累加器 `B_i`，清零 `.grad`。任务块结束时由 `B_i / m_i` 记录 `||D a_i||^2`，并把 `w_i B_i / m_i` 加进步累加器 `A`。单卡训练，不需要 all-reduce。
- 若 loss 已乘 `w_i / m_i`，记录前除以 `(w_i / m_i)^2` 还原。
- 每步结束：由当前 run 的实际 gradients 更新 `e_hat_i^train`，并更新 `Lbar_i`、`tau_i`、`C` 的 EMA，计算下一步的 `m`；日志写入 `m`、`H`、预条件与未预条件噪声、三项 gap/token 标量、各任务 loss、tokens、attempted responses 和计时，再把权重同步到推理卡。独立 scoring-token 重抽样只出现在 checkpoint 诊断脚本，不进入在线 loop。
- checkpoint：每 50 步存权重（3.4 GB）；Uniform 在第 50、250、500 步另存优化器状态（各约 20 GB）供 held-out 方差检查；每个 run 保留最新一份完整状态用于 resume。checkpoint 保存 `m`、`e_hat_i^train`、EMA 状态、prompt 位置和当前 estimator 类型；resume test 验证下一步分配与参数更新可复现。
- 在线训练不增加模型前向或反向次数，也不修改 optimizer；checkpoint 的独立 scoring-token paired diagnostic 是额外离线 gradient evaluation，其算力单独记账。

## 13. 算力与存储预算

| 项目 | 估算 |
|---|---|
| 单个 run | 2 卡：96GB 训练卡 + 48GB 推理卡 |
| 每步时间 | 约 1 分钟：rollout 25–40 s（由最长回复决定，`L_max=4,096` 封顶约 40 s）；四教师打分约 15 s，两个 4B 教师各约为 1.7B 的两倍；学生前反向约 15 s（约 45k token）；同步 + 优化器 + 日志约 10 s |
| 单 run 墙钟 | 500 步约 8–10 h，含每 50 步存档 |
| 一轮 8 slot | 8 个单种子配置，约 10 h，约 160 GPU-h |
| 18 卡两周容量 | 约 6,000 GPU-h |
| 存储 | 权重 checkpoint 3.4 GB × 11 × 8 ≈ 300 GB；按“每 run 保留 latest、Uniform 另留 early/middle”计，优化器状态约 `(8+2) × 20 GB = 200 GB`；日志与 held-out 输出另计 |

算力不是约束，工程时间是：训练器、权重同步、打分、日志、resume test 必须在 9/8 前跑通。步时与长度分布以冒烟测试实测为准，上表只用于排程。
