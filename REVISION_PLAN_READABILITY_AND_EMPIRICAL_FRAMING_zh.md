# 修改建议：降低理解门槛、突出质量、贴近真实 MOPD 实验论文

> 日期：2026-09-03。针对当前 all-task micro-batch 版本（`sections/*.tex`，PDF 11 页，主文约 8 页）。
> 参照物：MOPD（Ma et al. 2026）、Open-MOPD（Gao et al. 2026）的写法与实验组织。
> 本文档只给建议和可直接粘贴的英文草稿，未改动任何 `.tex`。

---

## 0. 总体判断

当前稿子的读者体验是"推导 + 协议"：主文 8 页中约 4 页是推导（第 3、4 节共 17 个编号公式），约 3 页是"我们将测什么"（第 5 节），真实 MOPD 数字为零；读者看到的第一张图是三任务高斯 toy 的柱状图。这与 MOPD / Open-MOPD 那种"先摆现象、再给方法、再上主表和消融"的实验论文相距很远。

核心思想其实一句话能说清楚：

> 目标权重固定不动；每个任务拿到的 micro-batch 数正比于 `w_i × 该任务在 AdamW 坐标下的单 micro-batch 梯度噪声`；要算时间就再除以 `sqrt(每个 micro-batch 的耗时)`。统计量全部来自本步已经算出的梯度，不多跑一次前向或反向。

但这句话直到 4.1 节才出现，而且被 "Neyman allocation in optimizer-scaled coordinates" 这类术语包住。

四个最大杠杆（按收益排序）：

1. **用真实四教师结果替换协议文字。** 没有这一步，其他修改只能让稿子"读起来像"实验论文。第 4 节给了一个低风险的实验阶梯，第一阶段只需一个带完整日志的 Uniform run 就能产出核心机制图。
2. **把 taskwise AdamW 从"方法前提"降级为"可选的一致性修正"。** 这样第 3 节后半和附录 A.1 都退出主阅读路径，而且避免一个目前没处理的混杂因素：taskwise 二阶矩会把有效学习率缩小约 `sqrt(G) = 4` 倍（见 3.5 节推导）。
3. **重排结构。** 读者应在第 2 页看到方法、第 4 页看到真实曲线。相关工作后移，toy 检查进附录。
4. **主文公式从 17 个压到 6-7 个，符号从约 22 个压到 10 个以内，配一张符号表。**

---

## 1. 三个目标的逐项诊断

### 1.1 理解门槛高在哪里

- **符号负担。** 前 4 页要同时记住 `M, w_i, G, m_i, c, R, g_{i,s}, a_i, A, U, D, e_i, ê_i, V(m), τ_i, C, J_C, H, z, x_i, ρ_τ, m_min, m_max, ℓ_i(0), L_i, F`。没有符号表。
- **未定义术语。** IcePop truncation、ORM terms（`sections/005_theory.tex:167`、`sections/009_appendix.tex:104`）、critical-path time、transfer tail、resident teacher slot、teacher visit、largest-remainder rounding、Neyman allocation、descent lemma / L-smooth、hierarchical mean、bank、manifest。每个都小，叠起来就把非优化背景的 MOPD 读者挡在外面。
- **抽象先于具体。** 摘要第一句之后就是"allocation variable is the number of fixed-size micro-batches"，读者还不知道一个 micro-batch 是什么（4 个来自同一任务的 prompt、各一条回复、一次 backward；见第 7 节）。
- **动机缺现象。** 引言没有任何数字说明"各任务噪声差多少、随训练变多少"，读者不知道为什么均匀分配不够好。
- **第一张图是 toy。** `figures/gpas_main_figure.png` 这张概念总览图画得很清楚，但没被任何 `.tex` 引用；正文 Figure 1 是合成实验柱状图，传达的信号是"理论文 + 玩具实验"。
- **方法核心藏得深。** 分配规则（式 10）在第 4 页；理想情况应在第 1 页的 findings 里用文字给出，第 2 页给出公式。

### 1.2 "高质量"目前靠什么体现、还缺什么

现有优点，值得保留并放到更显眼的位置：

- 固定目标 + `w_i/m_i` 缩放这个估计量设计干净，`m_min` 保证任务覆盖，这两点比 D³-MOPD / gap-following 那种改目标的方法更容易解释。
- 噪声用优化器坐标度量，toy 已经显示 raw norm 会排错序（2.3× 方差），这是"optimizer-aware"最有说服力的证据，但目前只是 toy。
- headroom `H` 是一个可复用的诊断量，应作为结果的主指标之一。
- 配对 prompt stream 和 resume test 都是好习惯。

缺的部分：

- 真实结果（曲线、主表、消融）。
- 对 taskwise AdamW 的学习率尺度效应的处理（3.5 节）。
- 明确的教师、数据、算力披露；归一化分数和教师上限。
- 对"分配信号该用什么"的真实对比：噪声（GPAS）vs. 未缩放噪声。

### 1.3 与真实 MOPD 实验论文的差距

对照 MOPD 4.1 节和 Open-MOPD 第 2、3 节：

| 实验论文常规要素 | 当前稿 |
|---|---|
| 学生起点（Base / SFT / Instruct）及其分数 | 起点为 Qwen3-1.7B-Base，未给初始分 |
| 教师的名称、大小、来源、是否同源 | "Four fixed specialists"，未命名（`005_experiment.tex:75`） |
| 每个域的训练 prompt 集来源 | "16,000 prompts per task"，未说来源 |
| 损失形式（sampled-token reverse KL / top-k）在正文一句话说明 | 只在附录 A.5 |
| 流水线表（Open-MOPD Table 1 风格） | 无 |
| 主结果表（各域 benchmark + 归一化分数 + 教师上限） | 计划中，无数字 |
| 现象图（各域 token share / reward 幅度 / 噪声随训练变化） | 无 |
| 消融表 | 无 |
| 算力披露（GPU 型号、每 run GPU-hour） | 无 |
| 相关工作放在结果之后 | 放在方法之前，占第 2 页 |

---

## 2. 结构重排（ICLR 9 页主文预算）

| 现在 | 建议 | 页数 |
|---|---|---|
| 1 Introduction | 1 Introduction：现象 → 两个旋钮 → 方法一句话 → findings（带数字）→ contributions；Figure 1 = 概念总览 | 1.25 |
| 2 Related Work | 移到 6 | — |
| 3 A Fixed Objective and Micro-Batch Estimator | 2 Setting: Fixed-Weight MOPD with Adaptive Micro-Batch Counts；只留式 (1)(4)；符号表 Table 1 | 0.75 |
| 4 GPAS Micro-Batch Allocation（含 taskwise AdamW 在 3 节） | 3 GPAS：3.1 噪声与分配规则（式 V(m)、GPAS）；3.2 Cost-GPAS（式 J_C 与 C=0 闭式）；3.3 在线估计与 headroom H；3.4 实现细节（算法框 + taskwise 二阶矩一段话 + 开销） | 1.5 |
| 5.1 Controlled checks | 附录 B（保留现 Figure 1） | — |
| 5.2 Predictions tested（Table 1） | 4 Experimental Setup 末尾的 RQ 列表 | — |
| 5.3–5.5 setup / methods / system | 4 Experimental Setup：学生、教师、数据、损失、超参、配置表、指标、算力记账 | 1.0 |
| 5.6 Outcomes（协议文字） | 5 Results：5.1 噪声与分配动态；5.2 loss vs steps / GPU-hours；5.3 主表；5.4 消融 | 2.75 |
| — | 6 Related Work | 0.5 |
| 6 Limitations and Conclusion | 7 Limitations and Conclusion | 0.4 |

附录：A 推导（descent lemma、Cauchy–Schwarz、E[U]）；B 合成检查；C 系统与计时 trace；D 评测细节；E 全部分任务曲线与日志。

---

## 3. 逐节修改建议

### 3.1 标题与摘要

标题可以保留。若想更贴近现象，可用副标题形式：*GPAS: Optimizer-Aware Micro-Batch Allocation for Multi-Teacher On-Policy Distillation*（把 "Task Allocation" 改为 "Micro-Batch Allocation"，与正文一致；目前正文从不"分配任务"，只分配 micro-batch 数）。

摘要问题：`001_abstract.tex:11` 的 "GPAS is Neyman allocation in optimizer-scaled coordinates" 先于任何直观解释；GPAS 缩写未展开；结尾是协议语气（"We specify..."）。可直接替换的草稿（方括号为待填数字）：

```
Multi-teacher on-policy distillation (MOPD) trains one student on its own
rollouts, scored token by token by a specialist teacher for each task. In
current recipes the task mixture plays two roles at once: it defines the
training objective, and it decides how much rollout and teacher compute each
task receives. Adapting the mixture therefore changes what the student is asked
to learn at the same time as it changes how precisely each task gradient is
estimated. We separate the two roles. The objective weights are fixed for the
whole run, every teacher contributes to every optimizer step, and the only
adaptive quantity is the number of fixed-size micro-batches each task receives.
Scaling each micro-batch loss by its task weight divided by its count keeps the
expected update identical under every allocation, so the allocation changes
gradient variance only. Gradient-Preconditioned Adaptive Sampling (GPAS)
assigns micro-batches in proportion to each task's weight times its gradient
noise measured in AdamW's preconditioned coordinates, which is the Neyman
allocation for this estimator; Cost-GPAS further divides by the measured
per-micro-batch time. Both rules use gradients and timings that the trainer
already computes and add no forward or backward passes. In a four-teacher
distillation of Qwen3-1.7B-Base (math, code, instruction following, science), GPAS
reaches [X]% lower weighted teacher loss than uniform allocation at the same
step budget, Cost-GPAS reaches the uniform run's final loss in [Y]% fewer GPU
hours, and no task falls below its uniform counterpart on downstream
benchmarks. A synthetic check confirms the allocation and optimizer-state
calculations.
```

### 3.2 引言

现状：五段都在讲设定，没有现象、没有数字、没有 findings；贡献列表（`002_introduction.tex:40`）是"定义条件、推导规则、给出测量设计"，全是理论/协议贡献。

建议结构：

1. **现象段。** 用 Open-MOPD 的具体数字把"混合比例同时是目标和算力表"说实：token-mean 聚合下 25× 的长度差让 IF 只占 1% 梯度 token；D³-MOPD 用 loss gap 在线改混合比例。落点：这些方法都在同时拧两个旋钮。
2. **两个旋钮段。** "学什么"（权重）与"每个梯度估得多准"（样本数）。分层平均 + `w_i/m_i` 把第一个旋钮焊死，第二个旋钮只影响方差。这里放一个真实数字：四教师 run 中各任务在 AdamW 坐标下的噪声差 [X]×，训练中漂移 [Y]×（指向 Fig 2）。
3. **方法段。** 一句话规则 + 为什么要用优化器坐标（raw norm 会排错序）+ Cost-GPAS 一句。
4. **实现足迹段。** 零额外前向/反向；每个 micro-batch 多两个参数大小的 reduction；标准 MOPD trainer 上约 [N] 行改动。
5. **findings + contributions。**

前两段英文草稿：

```
Multi-teacher on-policy distillation (MOPD) has become a standard way to merge
several reinforcement-learned specialists into one model: the student samples a
response, the teacher for that task scores it token by token, and the student
moves toward the teacher (Ma et al., 2026). Every recipe has to decide a task
mixture, and this decision matters more than it looks. With token-mean
aggregation, a 25x difference in response length leaves instruction following
with about 1% of the gradient tokens despite 20% of the prompts, and Open-MOPD
repairs this by rebalancing the mixture (Gao et al., 2026); D3-MOPD adapts the
mixture online from the teacher-loss gap and its descent rate (Sun et al.,
2026). In each case the mixture is both the training objective and the compute
schedule, so a change in one cannot be separated from a change in the other.

We keep the two apart. The objective is a fixed weighted sum of per-task
teacher losses, with the weights chosen before training. Each optimizer step
contains G fixed-size micro-batches, each drawn from a single task, and the
only decision made online is how many micro-batches each task gets. Because
each micro-batch loss is scaled by its task weight divided by its task's count,
the expected update is the same under every allocation (Section 2). Allocation
therefore decides only how precisely each task gradient is estimated, and the
natural criterion is variance: spend more micro-batches where the gradient is
noisier. In our four-teacher run the per-micro-batch gradient noise, measured
in the coordinates AdamW actually uses, differs by up to [X]x across tasks and
drifts by [Y]x over training (Figure 2), so no fixed split is variance-optimal
throughout.
```

findings 草稿（替换现有贡献列表，数字待填）：

```
Our main findings are:
- Optimizer coordinates matter. Allocation by raw gradient noise and by
  AdamW-scaled noise rank the tasks differently in [Z]% of steps; the raw rule
  [raises] variance relative to uniform, while GPAS lowers it by [..]% (H = [..]).
- GPAS reaches [..]% lower weighted teacher loss at matched optimizer steps,
  with the largest gain on [task].
- Cost-GPAS reaches the uniform run's final loss in [..]% fewer GPU hours, by
  moving micro-batches toward [short-response tasks] without changing their
  objective weight.
- Downstream capability follows the fixed objective: no task regresses
  relative to uniform, and [IF/science] improves by [..] points.
```

contributions 精简为三条：固定目标的 micro-batch 估计量；GPAS / Cost-GPAS 规则及其"免费"在线估计；开放的四教师配方、日志和 headroom 诊断。taskwise 二阶矩作为第三条里的一句话，不单列。

### 3.3 Setting 节（原第 3 节）

- 开头先用一句话定义 micro-batch 和一步：*A micro-batch is 4 prompts from one task, each with one sampled response, processed in one backward pass; a step accumulates G = 16 micro-batches, 64 responses, and takes one AdamW update.* 然后再引入符号。
- 只保留式 (1) 目标和式 (4) 累积梯度 `A`。式 (2) 权重定义、式 (3) 约束用文字。
- `004_preliminary.tex:49` "Averaging all tokens in a step without this factor would instead give task weight approximately m_i/G" 是好句子，直接接上 Open-MOPD 的发现：*This is the length-driven imbalance reported by Open-MOPD; the per-response mean inside each micro-batch and the w_i/m_i factor remove it by construction.* 这一句把你的设定和已知现象接上了，读者立刻知道你在解决什么。
- 符号表（Table 1，10 个符号足够）：

| 符号 | 含义 | 本文取值 |
|---|---|---|
| `M` | 任务（教师）数 | 4 |
| `w_i` | 固定目标权重 | 逆初始损失 |
| `G` | 每步 micro-batch 总数 | 16 |
| `b` | 每个 micro-batch 的 prompt 数，各一条回复 | 4 |
| `m_i` | 任务 i 本步的 micro-batch 数 | `2 ≤ m_i ≤ 8` |
| `g_{i,s}` | 任务 i 第 s 个 micro-batch 的梯度 | — |
| `D` | AdamW 对角预条件 `1/(√v+ε)` | — |
| `e_i` | D 坐标下任务 i 的单 micro-batch 噪声 | 在线估计 |
| `τ_i`, `C` | 每 micro-batch 边际时间、每步固定开销 | 实测 EMA |
| `H` | 相对 Uniform 的方差 headroom | `≤ 2` |

- taskwise AdamW 段（`004_preliminary.tex:58-80`）整体移到方法节 3.4，见 3.5。

### 3.4 方法节：公式瘦身清单

主文保留（6-7 个）：

1. `F = Σ w_i L_i`
2. `A = Σ (w_i/m_i) Σ_s g_{i,s}`
3. `V(m) = Σ w_i² e_i / m_i`（把 `e_i` 的定义并入同一句）
4. `m_i ∝ w_i √e_i`（GPAS）
5. `J_C(m) = (C + Σ m_i τ_i) · V(m)`，紧接 `C=0` 闭式 `m_i ∝ w_i √(e_i/τ_i)`
6. `H = V(m_unif)/V(m)`
7. （可选）`U = Σ (w_i/m_i) Σ_s g_{i,s}²`，放在 3.4 实现细节里

移到附录或改为文字：

- 式 (9) descent lemma（`005_theory.tex:26-35`）：主文一句 *The one-step descent bound shows that, for a fixed step, the allocation enters only through V(m) (Appendix A).*
- 式 (11) clip + largest-remainder：文字 *We clip the continuous counts to [2, 8], rescale the unclipped ones to sum to G, and round by largest remainder.*
- 式 (15) `ê_i`：文字 *ê_i is the unbiased sample variance of the D-scaled micro-batch gradients within the step, summed over coordinates.*
- 式 (16) EMA：文字 *τ_i and C are exponential averages (decay 0.9) of measured times.*
- 式 (6) `E[U]`：附录。
- `005_theory.tex:107` "roughly 10^9 coordinates ... sqrt(2/d_eff)" 这段近似论证，换成真实日志的一句话：*Empirically ê_i changes by [..]% between consecutive steps (Figure 2a), so we use the current estimate without smoothing.*
- `005_theory.tex:92` token-count fallback：脚注或附录。
- 4.2 节关于 `C>0` 无闭式、枚举整数解的两句合并为一句。

算法框：保留，但压成 4 步文字，删掉 "IcePop truncation, ORM terms" 那行（或在 Setting 节用一句话定义损失：*sampled-token reverse KL with importance-ratio truncation [cite]; no outcome reward*）。

### 3.5 taskwise AdamW：降级，并处理尺度问题

**降级理由。** 现稿把它写成"分析的前提"（`004_preliminary.tex:76` "is therefore part of the method, rather than a device for comparing runs"）。实践读者的第一反应是"用 GPAS 得换优化器？"这是采用门槛，也是理解门槛。更合适的写法：GPAS 在标准 AdamW 上即可运行；taskwise 二阶矩是一个可选的一致性修正，在实现细节里一段说明，由现有的 Uniform-Conv 对 Uniform-TW 比较检验它是否重要。

**尺度问题。** 逐坐标看 Uniform 情形（`w_i = 1/4, m_i = 4, G = 16`）：

```
常规 AdamW 二阶矩目标：E[A²] = (Σ w_i h_i)² + Σ w_i² Var_i / m_i
                             = signal² + (1/64) Σ_i Var_i
taskwise 目标：          E[U]  = Σ w_i (h_i² + Var_i)
                             = (1/4) Σ h_i² + (1/4) Σ_i Var_i
```

噪声项之比恰为 `G = 16`，开方后为 4。在噪声主导的坐标（LLM post-training 的常态）上，同一名义学习率下 taskwise 版本的更新幅度约为常规的 1/4。直观地说：常规 AdamW 的 `v` 对应"整步 16 个 micro-batch 均值"的二阶矩，taskwise 的 `v` 对应"单个 micro-batch"的二阶矩。于是 Uniform-Conv vs. Uniform-TW 的对比混杂了一个约 4× 的有效学习率变化，无法归因于"分配无关性"。

**建议修正。** 把观测定义为 `U/G`。此时 `E[U/G]` 的噪声项为 `Σ w_i Var_i / G`；在比例分配 `m_i = G w_i` 下与常规 AdamW 完全相同，而仍与 `m` 无关。信号项仍有差别（`(Σ w_i h_i)²` vs `Σ w_i h_i²/G`），但在噪声主导坐标上可忽略。

写法上，3.4 实现细节里一段即可：*Conventional AdamW's second-moment target contains the allocation-dependent noise term Σ w_i² Var(g_i)/m_i. As an optional refinement we replace A² by U/G, whose expectation depends on the student but not on m and whose scale matches conventional AdamW at the proportional allocation (Appendix A.1). Section 5.4 tests whether this matters.*

### 3.6 实验设置节：要补的"真实论文要素"

- **流水线表**（Open-MOPD Table 1 风格）：学生起点与初始分；四个教师的名称、参数量、来源（自训 RL 专家还是现成模型）、是否与学生同源（引用 Li et al. 2026 "same-origin" 结论作为选择依据）；各域 prompt 集来源与数量；每个 prompt 一条回复、每个 rollout batch 一次更新（K=1）；响应长度上限与截断率；评测集与采样设置。
- **损失形式一句话**：sampled-token reverse KL（policy-gradient form）、是否 top-k、是否截断 importance ratio；每个 prompt 一条回复（第 7 节）。
- **超参一处集中**：LR 与 schedule、β1/β2/ε、clip、weight decay、温度、`b, G, m_min, m_max`、EMA decay、阈值。
- **算力披露**：GPU 型号与数量、每个配置的 GPU-hour、教师服务方式（常驻/换入换出、推理引擎）。
- **RQ 列表**替换现在的 Table 1（`005_experiment.tex:40-64`）：
  - RQ1 各任务在优化器坐标下的噪声差异多大、随训练怎么变？raw 与 scaled 排序是否一致？
  - RQ2 固定目标下，GPAS 是否降低方差和同步数下的加权教师损失？
  - RQ3 Cost-GPAS 是否降低到达同一损失的 GPU-hour？
  - RQ4 taskwise 二阶矩是否重要？
  - RQ5 下游能力是否跟随固定目标，无任务回退？
- **评测**（`005_experiment.tex:181`）：补教师分数、初始学生分数，用 MOPD 论文的归一化分数 `(s − s_init)/(s_teacher − s_init)` 做跨域平均；选有 headroom 的 benchmark（Qwen3-1.7B-Base 在 MATH-500 上可能接近饱和，可加 AIME24/25 avg@16；GPQA-Diamond 对 1.7B 噪声大，avg@8 或加第二个科学集）；按 Open-MOPD 报告各域平均响应长度与截断率，8,192 上限对数学/代码可能截断。
- **系统描述**：单教师槽串行是一种具体系统选择，用一张小图或一句话说明为什么（显存），并给出实测 `C/(Σ m_i τ_i)`。并发教师槽的讨论压成一句，放局限。

### 3.7 结果节：图表规划

- **Fig 1（引言）**：把 `figures/gpas_main_figure.png` 改成矢量、减字，三栏：固定 `w` vs 自适应 `m`；AdamW 坐标下的噪声；分配规则。若有真实数据，右侧加一个 teaser：GPAS run 的 `m_i(t)`。
- **Fig 2（噪声与分配动态，RQ1）**：(a) 各任务 `ê_i(t)`，scaled 与 raw 两条；(b) GPAS 与 Cost-GPAS 的 `m_i(t)`；(c) `H(t)` 与上界 2；(d) `τ_i(t)` 与 `C` 占比。
- **Fig 3（效率，RQ2/RQ3）**：(a) 加权教师损失 vs 步数（GPAS、Uniform-TW、Uniform-Conv）；(b) vs GPU-hour（Cost-GPAS、GPAS、Uniform）。分任务曲线进附录。
- **Table 2（主表，RQ5）**：行 = 初始学生、各教师、Uniform-Conv、Uniform-TW、GPAS、Cost-GPAS；列 = 各域 benchmark、归一化平均、最终加权损失、到阈值的 GPU-hour 与 token。
- **Table 3（消融，RQ1/RQ4）**：计数规则 {Uniform, RawNoise, GPAS}，同一优化器；Uniform-Conv 对 Uniform-TW。
- 现 Figure 1（toy）整体进附录 B，正文一句引用。

一个值得留意的叙事线索：Open-MOPD 报告 IF 的 reward 幅度最大（0.091 vs 数学 0.019）且响应最短（409 vs 10,500 token）。若你的 `e_i` 也是 IF 最高，GPAS 会自动把 micro-batch 推向 IF，Cost-GPAS 会推得更多。这意味着 GPAS 从方差原理出发、不改目标，就复现了 Open-MOPD 靠经验发现的"IF 需要更多预算"。这可以成为引言和结果节的主线之一。

### 3.8 相关工作与局限

- 相关工作移到结果之后，保持半页。加一句：Open-MOPD 的 token-share balancing 在本文设定下由构造满足；gap-following 与 D³-MOPD 改的是目标权重，与本文正交，可以叠加。
- 局限压成一段：单一学生规模、四个任务、一种系统布局、micro-batch 独立假设、逐步的局部准则。删掉 `G < 2M` 那段或并入一句。

---

## 4. 实验层面的补充建议

### 4.1 实验阶梯

1. **阶段 0：一个 Uniform-Conv run，开全日志。** 每个 micro-batch 记录 `||D g||²`、任务均值、计时。离线即可算出 `ê_i(t)`、GPAS 的反事实分配 `m*(t)`、`H(t)`、raw 与 scaled 排序一致率、`τ_i`、`C`。这一步直接产出 Fig 2 的全部面板。
2. **阶段 1**：GPAS vs Uniform（同优化器）。
3. **阶段 2**：Cost-GPAS，需要真实计时 trace。
4. **阶段 3**：RawNoise 计数规则一个 run。

### 4.2 要补的基线

- **RawNoise-GPAS**（未缩放噪声）：toy 里它比 Uniform 差 2.3×，把它做成真实对比是"optimizer-aware"最直接的证据。

---

## 5. 术语与语言

### 5.1 定义或删除

| 术语 | 建议 |
|---|---|
| IcePop truncation, ORM terms | 删除，或在 Setting 节用一句话定义损失 |
| critical-path time, transfer tail, resident teacher slot, teacher visit | 集中在 4 节"系统"段用两句定义，其余地方用 "per-micro-batch time" 和 "fixed per-step overhead" |
| largest-remainder rounding | 文字描述一次 |
| Neyman allocation | 第一次出现时加半句：*the classical rule that samples each stratum in proportion to its standard deviation* |
| descent lemma, L-smooth | 进附录 |
| hierarchical mean | 改为 *per-response mean, then per-prompt mean, then per-micro-batch mean* |
| bank, manifest | *frozen set of micro-batches*，*run configuration file* |
| task vs. domain | MOPD 文献用 domain；建议首次出现写 *task (domain)*，之后统一 |

### 5.2 语气

- 协议语气（"We specify", "We report", "We log", "The first primary plot reports"）在结果到位后全部改为结果语气。
- 每段第一句给结论，再给细节。例：3.3 节现在以 "All statistics needed for the next allocation are obtained during the current step" 开头，这是好的；4.2 节以 "Let τ_i be ..." 开头，应改为 "Long-response tasks cost more per micro-batch, so equal variance is not equal time."

---

## 6. 逐行修改清单

| 位置 | 问题 | 改法 |
|---|---|---|
| `sections/001_abstract.tex:11` | Neyman 术语先于直觉；GPAS 未展开；协议语气 | 用 3.1 草稿 |
| `sections/002_introduction.tex:23-30` | 方法段无直觉、无足迹 | 用 3.2 草稿；加"零额外前向/反向" |
| `sections/002_introduction.tex:40-49` | 贡献全是理论/协议 | 换成 findings + 三条贡献 |
| `sections/004_preliminary.tex:1-6` | 直接进符号 | 先一句定义 micro-batch 与 step |
| `sections/004_preliminary.tex:49-54` | 好句子但没接现象 | 接 Open-MOPD 的 25× / 1% |
| `sections/004_preliminary.tex:58-80` | taskwise AdamW 作为前提，且未处理尺度 | 移到 3.4，改为可选修正，`U/G`，见 3.5 |
| `sections/005_theory.tex:26-35` | descent lemma 占主文 | 一句话 + 附录 |
| `sections/005_theory.tex:45-53` | clip/rounding 公式 | 文字 |
| `sections/005_theory.tex:92-94` | fallback 讨论 | 脚注 |
| `sections/005_theory.tex:104-111` | 10^9 坐标近似论证 | 换成日志数字 |
| `sections/005_theory.tex:113-124` | EMA 公式 | 文字 |
| `sections/005_theory.tex:167-168` | IcePop / ORM | 删除或定义 |
| `sections/005_experiment.tex:14-38` | toy 占主文半页 + 首图 | 附录 B |
| `sections/005_experiment.tex:40-64` | "Predictions tested" 表 | RQ 列表 |
| `sections/005_experiment.tex:73-78` | 教师未命名、学生起点不明 | 流水线表 |
| `sections/005_experiment.tex:139-140` | 否定句 "does not estimate variation across training seeds" | 改成正面陈述：matched streams make the comparison paired |
| `sections/005_experiment.tex:148-172` | 系统段过长 | 两句 + 附录 C |
| `sections/005_experiment.tex:174-190` | 协议文字 | 结果节替换 |
| `sections/007_discussion.tex` | 六段局限 | 压成一段 + 一段结论 |
| `sections/009_appendix.tex:70-71` | 否定句 "not empirical claims about training speed" | 删除 |
| `sections/003_relatedwork.tex` | 位置 | 移到结果之后；加"与 Open-MOPD 正交可叠加"一句 |

---

## 7. 每个 prompt 只采样一条回复：修改点与连带影响

### 7.1 为什么改

- MOPD 原文附录 A 的 Stage 3 设置是 `BS 2048, N = 1`：每个 prompt 一条 rollout；`N = 8` 只用于 Stage 2 的 GRPO 教师。Open-MOPD 的目标定义同样是 "For a prompt x, the student first samples a response y"，每次更新 256 个 prompt、各一条回复。
- 逐 token 的 reverse-KL 监督本身是稠密信号，不需要 GRPO 那种"同一 prompt 多条回复做组内 advantage 归一化"。当前稿子的 `c` 个 prompt group × `R = 4` 是 GRPO 血统的残留。
- 改成一条回复还顺带简化方法：去掉 "prompt group" 与 "hierarchical mean" 两层概念，符号 `c, R` 合并为一个 `b`（每个 micro-batch 的 prompt 数）；micro-batch 之间的独立性假设也更干净，不再有同一 prompt 的四条回复共享 prompt 的相关性。

写进 Setting 节的一句话：

```
Following MOPD and Open-MOPD, each prompt receives one sampled response, and
each rollout batch is used for exactly one optimizer update. The token-level
teacher signal is dense, so no group of responses per prompt is needed, unlike
GRPO-style advantage normalization.
```

### 7.2 预算不变量与本次决定：每任务 16,000 个 prompt、不循环

设每个 micro-batch 有 `b` 个 prompt、各一条回复，每步 `G = 16` 个 micro-batch，共 `S` 步，总回复数 `N_total = 16 b S`。一个任务若始终取 `m_max`，消耗的唯一 prompt 数为

```
S × m_max × b = N_total × (m_max / G) = N_total / 2
```

与 `b` 无关。要求每任务 16,000 个 prompt 且不循环，就意味着 `N_total = 32,000`，`S × b = 2,000`。这比原来的 128,000 条回复少四倍；`R = 4` 时同一个池能支撑四倍回复，是因为每个 prompt 被重复用了四次。

同一预算下 `b` 与 `S` 的划分：

| 划分 | 每 micro-batch | 每步回复 | 步数 | Uniform 每任务每步 prompt | 评价 |
|---|---|---|---|---|---|
| A（采用） | `b = 4` | 64 | 500 | 16 | 对现稿改动最小：500 步、`500×8×4 = 16,000` 的算式、LR schedule、checkpoint 位置全部保留；分配决策次数最多 |
| B | `b = 8` | 128 | 250 | 32 | 每步梯度更稳，步数减半 |
| C | `b = 16` | 256 | 125 | 64 | 与 Open-MOPD 每次更新 256 prompt 对齐，但 125 步太少，自适应分配来不及体现 |

采用 A。理由：GPAS 每步做一次分配决策，损失-步数曲线要有足够多的步才能分开；`b = 4` 在显存上没有问题（4 条 ≤ 8,192 token 的回复一次 backward）；现稿关于 500 步的所有陈述可以原样保留。

另一个可以动的旋钮是 `m_max / G`：它同时决定 prompt 需求量和 headroom 上界 `H ≤ M·m_max/G`。`m_max = 6` 可以把总预算提高到约 42,700 条回复，但 `H` 上界降到 1.5。不建议为了多用数据牺牲 headroom。

### 7.3 预算缩小四倍的连带影响

- **固定开销占比上升。** 每步生成量变小，但教师加载卸载、同步、optimizer 更新这些固定开销 `C` 不变，`C / Σ m_i τ_i` 上升；论文 4.2 节已写明 `C` 主导时 Cost-GPAS 的最优解趋向 GPAS。划分 B 或并发教师槽会降低这个占比。
- **墙钟时间不会按四倍缩短。** 一步的生成时间由最长回复决定，64 条与 256 条回复的生成墙钟差距远小于四倍，GPU 利用率下降。算力披露用实测值。
- **共同 loss threshold 按本预算定。** "到达共同阈值的 GPU-hour 与 token"里的阈值按本预算下 Uniform-TW 能到达的损失来定，例如其最终损失的某个分位。
- **每任务 prompt 消耗。** Uniform 每任务 8,000；始终取 `m_min` 的任务 4,000；始终取 `m_max` 的任务 16,000。日志里报告每个任务实际消耗的 prompt 数，作为分配轨迹的另一种呈现。
- **噪声估计。** micro-batch 只有 4 条回复，`e_i` 的绝对值比 16 条时约大四倍，但分配只看相对值；`ê_i` 在 `m_i = 2` 时的估计质量取决于坐标数，不取决于 `b`。已有 `R = 4` 配置下的日志不能沿用，阶段 0 必须重采。
- **数据层面反而更干净。** 32,000 条回复对应 32,000 个不同的 prompt，每个 prompt 整个训练只出现一次，micro-batch 之间不共享任何 prompt，独立性假设比原来同一 micro-batch 内四条回复共享 prompt 的情形更站得住。不需要"第二遍"策略。

### 7.4 正文逐行修改清单（按划分 A）

| 位置 | 现文 | 改为 |
|---|---|---|
| `sections/004_preliminary.tex:6-8` | "averaged over valid tokens within a response, over the $R$ responses for a prompt, and over prompts" | "averaged over valid tokens within a response and then over prompts, each prompt receiving one sampled response" |
| `sections/004_preliminary.tex:23-25` | "A micro-batch contains $c$ prompt groups, with one prompt and $R$ sampled responses in each group. Its size and hierarchical loss normalization are the same ..." | "A micro-batch contains $b$ prompts from one task, each with one sampled response. Its size and loss normalization are the same ..."，后接 7.1 的那句话（引用 MOPD、Open-MOPD 和 GRPO） |
| `sections/005_theory.tex:142` | "$G,c,R,m_{\min},m_{\max}$" | "$G,b,m_{\min},m_{\max}$" |
| `sections/005_theory.tex:154-156` | "take the next $m_i c$ prompt groups without replacement, generate $R$ responses per group ... form its hierarchical mean loss" | "take the next $m_i b$ prompts without replacement, generate one response per prompt ... form its mean loss" |
| `sections/005_experiment.tex:80-87` | 整段 | 见下方替换段一 |
| `sections/005_experiment.tex:94-99` | "The total budget is 128,000 attempted responses. Because $G$, $c$, and the number of responses per group are fixed, every method takes the same number of optimizer steps. The learning-rate schedule is a function of cumulative attempted responses." | "The total budget is 500 optimizer steps, 32,000 attempted responses in all. Because $G$ and $b$ are fixed, every method takes the same number of optimizer steps and consumes the same number of responses per step. The learning-rate schedule is a function of the optimizer step." |
| `sections/005_experiment.tex:101-103` | "Each task has a fixed stream of 16,000 training prompts. Prompts are consumed without replacement. A task assigned the maximum count for all 500 steps uses exactly $500\times8\times4=16{,}000$ prompts, so no stream cycles." | 见下方替换段二；算式不变 |
| `sections/005_experiment.tex:137` | "the $n$th requested group for a task" | "the $n$th requested prompt for a task" |
| `sections/009_appendix.tex:103-104` | "averages tokens within a response, the $R$ responses within a group, and the $c$ groups within a micro-batch" | "averages tokens within a response and then the $b$ responses within a micro-batch" |
| `sections/008_checklist.tex:16` | "micro-batch hierarchy" | "micro-batch composition" |

替换段一（`sections/005_experiment.tex:80-87`）：

```
One micro-batch contains b = 4 prompts from a single task, each with one
sampled response of at most 8,192 tokens, and is processed in one backward
pass. Each step processes G = 16 micro-batches, 64 responses in all, and takes
one AdamW update, so no rollout is reused across updates. The bounded counts
use m_min = 2 and m_max = 8; Uniform therefore assigns four micro-batches,
16 prompts, to every task. Reverse KL is averaged over valid tokens in each
response and then over the four responses in a micro-batch. Every attempted
response is included in the reported response count and resource totals.
```

替换段二（`sections/005_experiment.tex:101-103`）：

```
Each task has a fixed stream of 16,000 training prompts, shuffled once with a
fixed seed and consumed without replacement. A task assigned the maximum count
for all 500 steps uses exactly 500 x 8 x 4 = 16,000 prompts, so no stream
cycles and no prompt is seen twice; Uniform uses 8,000 prompts per task.
```

### 7.5 实验协议文档的对应修改

`EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md`：

| 行 | 改为 |
|---|---|
| 9 | 每个 micro-batch `b = 4` 个 prompt，每个 prompt 一条回复（与 MOPD 的 `N = 1` 一致） |
| 11 | token → response 两层平均 |
| 12 | 总预算 500 步 × 64 条回复 = 32,000 attempted responses；学习率以 optimizer step 为时钟 |
| 14 | 不变：每个任务 16,000 个不循环的 prompt |
| 20 | 算式 `500*8*4=16,000` 不变；补一句"每个 micro-batch 4 个 prompt、各一条回复；Uniform 每任务消耗 8,000" |
| 107 | "第 `n` 个 prompt" |
| §1 新增 | 每个 rollout batch 只做一次更新（`K = 1`），教师 log-prob 不复用 |
| §9 | 共同 loss threshold 按本预算下 Uniform-TW 的最终损失定 |

### 7.6 顺带收益

- 主文少两个符号（`c, R`）和一个概念（prompt group / hierarchical mean），直接服务第 1 节的降门槛目标。
- 流水线表多一行 "one response per prompt, one update per rollout batch"，与 Open-MOPD 关于 reward staleness 的讨论直接对齐，这是真实 MOPD 论文读者会核对的项。

---

## 8. 优先级

**P0（决定论文性质）**
0. 改为每个 prompt 一条回复，每任务保持 16,000 个不循环 prompt，总预算 32,000 条回复（第 7 节）。
1. 跑阶段 0，用真实 `ê_i(t)`、`m*(t)`、`H(t)` 替换所有协议文字并写进引言。
2. 处理 taskwise 二阶矩的尺度问题（`U/G`），并降级为可选修正。

**P1（决定可读性）**
3. 结构重排；Fig 1 换成概念总览；toy 进附录。
4. 主文公式压到 6-7 个；符号表；术语定义清单。
5. 摘要、引言按草稿重写；findings 带数字。

**P2（打磨）**
6. 教师/数据/算力披露表；归一化分数；benchmark 选择。
7. 补 RawNoise 计数规则对比。
8. 清理否定句与协议语气。
