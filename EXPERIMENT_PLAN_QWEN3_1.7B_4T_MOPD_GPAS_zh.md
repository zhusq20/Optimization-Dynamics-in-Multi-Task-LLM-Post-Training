# GPAS 四域实验：两周核心方案

更新：2026-09-04。本版以两周内完成实验和结果整理为目标，替代此前 13 个训练 runs 与大规模诊断方案。**本轮只做 4 个单种子完整训练 + 1 个共同 checkpoint 的小规模机制对照。**结果围绕 GPAS 的能力收益、计算效率和分配机制展开。

沿用已确定的设置：学生 Qwen3-1.7B-Base；四个域分别使用对应的 Qwen3-1.7B RL teacher；不设置防御性试验、教师资格筛选或 gate 实验。当前文件是执行计划，尚未填入实测结果。

## 1. 三个重点问题与对应实验

| 重点问题 | 直接比较 | 最主要的证据 |
|---|---|---|
| GPAS 是否带来实际收益？ | Uniform vs GPAS，训练目标与样本预算相同 | 第 500 步四域能力、平均分、最差域差值；固定参考 loss 与实测 GPU hours |
| 预条件是否有价值？ | GPAS vs GPAS without preconditioning | 相同 AdamW 优化器下，仅分配统计是否使用 D 的差异 |
| 梯度噪声分配是否优于已有调度信号？ | GPAS vs D³ signal, fixed weights | 相同 loss、任务权重、计数边界下，分配信号的效果差异 |

用一个共同 checkpoint 的 Uniform/GPAS 对照补充机制证据：在相同状态和相同样本预算下，GPAS 是否降低**独立实测的更新方差**，以及这种变化是否伴随实际 loss 下降改善。它使用固定 checkpoint，不按最终效果选择有利时点。

论文的实验叙事收敛为：**方法是否有效 → 相对已有调度信号的表现 → 预条件消融 → 一个直接机制对照**。各域成绩和无效结果完整展示，结论取决于实测，不预写成功方向。

## 2. 唯一的完整训练清单

| ID / run ID | 配置 | 样本分配 | 优化器与目标 | 作用 |
|---|---|---|---|---|
| U / `uniform-s1` | Uniform | 每域 4 个 micro-batches | AdamW；teacher top-64；固定等权 | 主基线 |
| G / `gpas-s1` | GPAS (per step) | 按预条件梯度噪声分配 | 同 U | 主方法 |
| R / `gpas-raw-s1` | GPAS without preconditioning | 按原始梯度噪声分配，统计使用 D=I | 同 U，优化器仍是 AdamW | 唯一机制消融 |
| H / `d3-fixed-s1` | D³ signal, fixed weights | 按 remaining gap × descent velocity 分配 | 同 U | 已有调度信号对照 |

**合计 4 个完整 runs，各运行同一个训练种子 s1 一次，每 run 500 步。**U/G 优先排入，R/H 接续或在有空闲设备时并行。四个配置是本轮全部训练计划，不再追加“有时间就做”的方法。

### 从本轮清单移出的内容

| 移出内容 | 对应范围调整 |
|---|---|
| 4 个单任务 OPD 参照 | 本轮直接比较共享学生的多域能力，不报告相对单任务参照的集成达标率或归一化收益 |
| GPAS cost-aware | 保留实测效率报告，暂不验证额外的 time×variance 调度器 |
| 动态权重 D³ 完整配方、Open-MOPD 配方 | 保留与本方法最直接可比的固定权重 D³ 信号，避免额外目标和实现差异 |
| TA-OPD 两个 runs | 固定一种已有 dense loss，暂不研究跨 loss 泛化 |
| Precise 分支、4 个单任务更新分支、batch-size 扫描 | 局部实验只比较相同预算的 Uniform 与 GPAS |
| 3 checkpoint × 多分支大诊断、方向协方差/K 矩阵交叉拟合、独立预测分类流程 | 只在第 250 步测更新方差与实际一步 loss 变化 |
| 状态分布变化的完整分解、额外种子/模型/teacher 复验 | 保留固定参考 loss、最终 fresh-policy loss 和能力三个观测，不扩展实验轴 |

这些内容移出两周交付范围，不作为开跑或得出当前结果的前置条件。原完整方案保留在备份中。

## 3. 四个 runs 共用的设置

| 项目 | 设置 |
|---|---|
| 学生 | Qwen3-1.7B-Base，non-thinking |
| 数学教师 | 数学 Qwen3-1.7B RL teacher |
| IF 教师 | IF Qwen3-1.7B RL teacher |
| 代码教师 | 代码 Qwen3-1.7B RL teacher |
| 科学教师 | 科学 Qwen3-1.7B RL teacher |
| loss / 权重 | teacher top-64 corrected reverse KL；四域 w_i=1/4 |
| prompts | 每域 16,000 条；共享同种子的域内排列，打乱后无放回消费 |
| micro-batch / step | 每 micro-batch 4 个同域 prompts，每 prompt 1 个 response；每步 G=16，共 64 个 responses |
| 计数 | 每域 2≤m_i≤8，sum_i m_i=16；Uniform 为 (4,4,4,4) |
| 预算 | 每 run 500 步、32,000 条计划 responses，失败重算另计 |
| rollout | 最大 response 长度 4,096；截断回复保留有效 token；每批新 rollout 仅更新一次，K=1 |
| 优化器 | AdamW；学习率、schedule、clipping、精度等直接采用同一份已有训练配方 |
| 种子 | 唯一训练种子 s1；各用途随机源由它派生 |
| 资源 | 每 run 一张 96GB learner GPU + 一张 48GB rollout/teacher GPU；按实际可用 slots 排程 |

四域分别使用对应的 Qwen3-1.7B RL teacher；四个可比 runs 使用相同的实际 teacher revisions，在结果表按域记录。

训练、固定 loss 评估、局部诊断和能力 benchmark 的题目集合分开。每个方法共享初始权重、训练 prompt 顺序和解码设置；分配导致的域曝光差异作为方法行为记录。每步生成使用更新前的学生权重，完成一次更新后同步新权重。

记录实际模型/tokenizer/teacher revisions、数据版本、s1 整数、解码设置、优化器参数、软件/GPU 型号。采用已有配置完成普通实现核对，不另开超参数扫描、先导训练或资格试验。

## 4. 只实现本轮需要的训练差异

### 4.1 默认 loss 与固定权重

    S = teacher_top64(prefix)
    loss_position = sum_{a in S} [p[a] * (log(p[a]) - log(q[a])) - p[a] + q[a]]
    A = sum_i w_i * mean_s(g_i,s)

p/q 保留全词表归一化，top-64 内不重新归一化；对完整 loss 求导，保留 -p 项。教师及采样 prefixes 固定；先对 response 的有效位置平均，再对 response 平均。四个 runs 都按任务均值汇总，等价于每个 micro-batch loss 乘 w_i/m_i；不能按整步 response 平均而把任务权重改为 m_i/G。

### 4.2 GPAS 与原始噪声消融

G 使用更新前的 bias-corrected Adam second moment 定义 D=(sqrt(v_pre)+epsilon)^(-1)，第一步用 D=I 和 Uniform。对未加任务权重、未 clipping 的 micro-batch 梯度做 Welford 统计：

    h = 0; S = 0
    for s, g in task_microbatch_gradients:  # s 从 1 开始
        delta = g - h
        h += delta / s
        S += (s - 1) / s * squared_norm(D * delta)
    e_hat_i = S / (m_i - 1)
    A += w_i * h

同一任务均值 buffer 按域复用；混合精度梯度先 unscale，归约使用 float32，最后对 A 做 clipping 和 AdamW 更新。噪声 EMA decay=0.9，首条观测直接初始化。下一步读取已完成步骤的 EMA 选择计数，当前样本不参与选择自己的计数。

G 枚举满足边界的 **149** 组整数计数，最小化 sum_i w_i²*e_ema_i/m_i。R 采用完全相同流程，仅噪声统计中的 D=I，优化器仍是 AdamW。并列优先选择距离 Uniform 最近的计数，再按 math/code/if/science 字典序确定；全零噪声时用 Uniform。

U 使用常规固定权重梯度累积，H 只记录调度所需的 task loss。**不为 U/H 强加 GPAS 的逐 micro-batch 噪声收集开销**；G/R 的统计耗时和额外内存包含在它们自己的训练成本中。可比的实际更新方差在第 6 节的共同 checkpoint 实验中统一测量。

### 4.3 D³ 固定权重对照

采用论文附录 `app:baselines` 的 remaining gap × descent velocity：初始 loss 取前 5 个观测均值，EMA window=10，窗口 W=10、最多 R=3 个窗口，每 10 步更新，KL denominator floor=0.15，max-normalization，softmax temperature=0.5，probability floor=0.10，batch jitter=0.30。warmup、零信号和操作顺序沿用该来源的固定实现并记录版本。

    gap_i(t) = Lbar_i(t) / max(L_i_initial, 0.15)
    velocity_i(t) = max(0, mean of available window-relative loss decreases)
    signal_i(t) = gap_i(t) * velocity_i(t)

所得概率 p_i 转为计数时，在 149 个可行整数向量中最小化 sum_i(m_i-16*p_i)²，并列沿用 G 的规则。最终仍以 w_i=1/4 汇总任务均值。这一行明确命名为 **D³ signal, fixed weights**，不称为完整 D³-MOPD 配方复现。

## 5. 评估压缩为“一张主表、两条核心曲线、一张机制图”

### 5.1 能力评估

主表统一使用第 500 步，列出：初始学生、实际指定教师、U、G、R、H。指定教师行按域记录四个 Qwen3-1.7B RL teacher。

| 域 | 指标 | 口径 |
|---|---|---|
| Math | MATH-500 greedy pass@1 | 每题一个 greedy 回答，固定答案判定器 |
| Code | 固定 LiveCodeBench 切片 pass@1 | 固定题目 IDs/日期范围、解码、执行环境和超时 |
| IF | IFBench strict accuracy | 固定 evaluator 及 strict 聚合粒度 |
| Science | GPQA-Diamond average@4 | 同题 4 次回答的平均正确率，再对题目平均；不使用任一答对的 pass@4 |

主表同时给出四域百分比分数的算术均值、G 相对 U 的逐域差值和最差域差值 min_i(C_i(G)-C_i(U))。R/H 同样报告逐域与平均变化。能力主比较不使用单任务归一化或未知参照目标。

**完整 benchmark 只评：**初始学生一次；U/G 的第 250 和 500 步；R/H 的第 500 步。共 **7 组学生 checkpoint 评估**，另对各域指定教师做一次对应 benchmark 评估。第 0 步共享，不重复算作四次。U/G 能力曲线只标出 0/250/500 的实测点。

保存题目级输出，使用配对题目 bootstrap 给出评估误差；GPQA 同题四次回答作为一簇。所有完整训练都是单种子，不构造跨训练种子的标准差。

### 5.2 低成本 loss 与效率曲线

每域使用 **64 条固定 held-out prompts**。初始学生只生成一次 reference responses，缓存各位置 teacher top-64 scores，形成四域共 256 个 responses 的公共固定 bank。

- 四个方法在第 **0、100、200、300、400、500** 步计算该固定 bank 的 loss；第 0 步共享一次。
- fresh-policy loss 仅在第 **0 和 500** 步测量：第 0 步复用初始 bank；每个最终模型在同一批 held-out prompts 上各生成一次新 response。保留最终 fresh loss 与固定 loss，不做相邻状态分布完整分解。
- 固定参考 loss 分别对 optimizer steps 和累计 GPU hours 作图；U/G 突出展示，R/H 用同图辅助线。最终能力与实测总训练成本同时报告。

共同目标为 target=F_ref,U(500)，其中 F_ref=sum_i L_i/4。按首次向下穿越目标的相邻评估点插值估算到达成本；未到达标 `unreached`，不外推。它只表示达到同一 loss 的计算效率，能力收益由 benchmark 独立回答。

训练成本记录两张设备的实际占用，包括 rollout、teacher scoring、梯度、统计、同步、等待和保存；不同 GPU 型号分别列出后再汇总。诊断和评估单列，并给出研究总成本。主成本比较包含 G/R 的真实统计开销。

## 6. 一个共同 checkpoint 的机制对照

固定使用 **U 第 250 步**，只比较 Uniform 与 GPAS 两个分支。两者每次都从相同模型、完整 AdamW/LR 状态恢复，使用同一学习率和相同 G=16，做一次实际更新，结果不写回主训练。整个实验仅 **20 次局部更新**。

### 样本与执行

| 用途 | 样本 | 操作 |
|---|---|---|
| Calibration | 每域 16 个独立 micro-batches，共 256 responses | 用固定 pre-step D 估计 e_i，枚举选出一个 GPAS 计数向量；Uniform 固定 (4,4,4,4) |
| Update trials | 每分支 10 次独立抽样，每次 64 responses，共 1,280 responses | 固定所选计数，不再用 trials 重新选择分配；从相同 checkpoint 恢复后执行更新 |
| Evaluation | 每域 64 个独立 prompts，共 256 responses | 由更新前模型生成一次，缓存 teacher scores；所有更新后模型在同一 bank 上评分 |

三个用途从同一域内诊断分布独立抽样，和训练、长期 loss bank、benchmark 分开。update trials 不复用 calibration 的 responses；所有分支共享 evaluation bank。10 次抽样是一步更新的重复，不是额外训练种子。

### 只保留两个观测

1. **更新方差。**对每个 trial 的 clipping 前组装梯度 A_r，在该 checkpoint 的固定 D 下，按分支计算：

       V_hat = sum_r ||D*(A_r - mean_r(A_r))||² / (10 - 1)

   使用 Welford 累积，无需保存全部 trial 梯度。报告 V_hat_G/V_hat_U 与原始值。选择计数的 calibration 和计算 V_hat 的 trials 独立，避免把“在同一批估计上求得最小值”当作机制证据。这里两个分支使用同一个 D，测量开销均计入诊断。

2. **实际一步进展。**在固定 evaluation bank 上记录各域 d_i=L_i(before)-L_i(after)、等权平均下降以及 d_i≤0 的频率；展示全部 10 次 trial 的点和汇总。差值以同一评估 prompt 配对，更新抽样则分支独立处理。共享 bank 的重采样索引在全部 trials 中保持一致。

机制图并列展示：两种分配的 m_i、实际更新方差比、各域一步 loss 变化；用 G 主训练日志补充分配随训练的轨迹。既不需要 evaluation 梯度，也不构造 K、方向协方差、Hessian 或一阶符号分类。

这项对照直接支持“改变分配是否降低更新方差并改善该状态的一步进展”。端到端作用看 U/G 主实验，不由一个 checkpoint 推断长期收敛或全部失效机制。若 calibration 选出的 GPAS 计数恰好等于 Uniform，也保留这个结果，不另选有利 checkpoint 或教师。

## 7. 两周排程与费用

### 7.1 从启动日计 14 天

| 时间 | 主要工作 | 当期产物 |
|---|---|---|
| 第 1–3 天 | 在同一训练器接入默认 loss、固定任务权重、G/R 分配与 H 信号；接入现有评估器；启动 U/G | 可执行的 4 配置、共享数据/teacher manifest、主训练日志 |
| 第 4–6 天 | 完成 U/G，保存第 250/500 步并评估；已有空闲 slot 时启动 R/H | 主比较结果与效率曲线；U/250 完整状态 |
| 第 7–8 天 | 完成 R/H 及最终评估 | 一张包含四种方法的最终能力/成本表 |
| 第 9–10 天 | 执行 U/250 的两个局部分支，各 10 次；补齐公共 loss 与最终 fresh loss | 更新方差和实际一步下降图 |
| 第 11–12 天 | 汇总题目级结果、成本和分配日志，完成论文图表 | 主表、核心曲线、机制图及图表源数据 |
| 第 13–14 天 | 处理故障导致的缺失输出、核对表图与文字、整理产物 | 两周交付版本；不新增配置或实验轴 |

实现与评估并行推进，不以 teacher gap、单任务能力或早期指标达标作为启动条件。计划按一个双卡 slot 也能顺序执行四个 runs 安排，有更多设备时只并行加速现有清单。每日按实测耗时调整顺序，最后两天保留给修复与整理。

### 7.2 预算

| 项目 | 计算 | 工作量 |
|---|---|---:|
| 完整训练 | 4 × 500 × 16 × 4 | **128,000 条计划 responses** |
| 局部 calibration | 4 域 × 16 micro-batches × 4 | 256 条 responses |
| 局部 update trials | 2 分支 × 10 × 16 × 4 | 1,280 条 responses |
| 局部 evaluation bank | 4 域 × 64 | 256 条 responses |
| 局部诊断合计 | 1 个 checkpoint、20 次一步更新 | **1,792 条新生成 responses** |
| 长期固定/fresh loss banks | 共享初始 256 + 四个最终模型各 256 | **1,280 条新生成 responses** |
| 以上生成合计 | 128,000 + 1,792 + 1,280 | **131,072 条计划 responses** |

局部更新后评分额外为 20×256=**5,120 次 response 学生评分前向**；长期固定 bank 每个方法的五个后续评估点另有 4×5×256=5,120 次评分，初始/最终 fresh 评分和局部更新前评分单列。固定 bank 评分不重新生成 response，也不重复请求 teacher scores。

能力 benchmark 费用另算：7 组学生 checkpoint 的题目生成/评分，加各域指定 teacher 一次对应评估；GPQA 每题 4 个回答，其余采用固定单回答口径。Code/IF 题数按最终冻结切片记录，故不预填完整 benchmark token 或 GPU hours。训练与评估故障重算另列。

按之前每 run 8–10 墙钟小时、两张 GPU 的**未验证排程假设**，4 个完整 runs 为 **64–80 GPU hours**（一个双卡 slot 顺序运行约 32–40 小时），实际训练、评估和实现耗时以日志为准。这不是两周完成的硬件保证，排程已为实现、评估与故障处理留出主要时间。

相较上一版，完整训练从 **13 减到 4 runs**，局部更新从 **600 减到 20 次**，诊断生成从 **81,792 减到 1,792 条**。两周内不再添加原方案中的单任务参照或次要配方。

## 8. 交付物与论文同步范围

交付限定为：

1. **主结果表：**初始学生/指定教师/U/G/R/H，四域分数、平均分、相对 U 的逐域和最差域差值、总训练 GPU hours。
2. **核心曲线：**固定参考 loss 对 steps/GPU hours；U/G 能力在 0/250/500 的实测点。最终 fresh loss 作为补充列。
3. **机制与消融：**U/250 的分配、实测更新方差和一步 loss 图；R/H 对照结果直接使用主表，G 的 m_i/e_i 轨迹来自训练日志。
4. **可追溯产物：**配置与实际 RL teacher 标识、逐步 loss/计数/曝光/时间日志、题目级评估、20 次局部更新记录、图表源表。主训练保存第 100/200/300/400/500 步模型和 U/G 第 250 步模型；U/250 额外保存完整优化器/数据/随机状态，另保留最新恢复点。

论文回填按这份缩减范围呈现。正文中既有单任务可迁移参照、所有域均值/方向方差诊断、独立预测验证、cost-aware 和跨 loss 泛化的实测承诺应相应删除或留作后续工作，不用本轮四个 runs 代替未做的证据。理论中的区别仍可保留为分析，但本轮主张聚焦 GPAS 的实际效果、预条件和噪声分配。

本次修改的是实验文档和实验目录说明；论文正文的范围同步与实测结果回填随实验交付完成。当前目录的解析 teaser 与旧合成数据仍只作解析/历史材料，不作为四域实测结果。

论文实现依据：[任务目标与梯度聚合](sections/004_preliminary.tex)、[GPAS 方差与分配](sections/005_theory.tex)、[比较配方](sections/009_appendix.tex)。两周限制、单种子和四域 RL 教师设置以本文件为准。
