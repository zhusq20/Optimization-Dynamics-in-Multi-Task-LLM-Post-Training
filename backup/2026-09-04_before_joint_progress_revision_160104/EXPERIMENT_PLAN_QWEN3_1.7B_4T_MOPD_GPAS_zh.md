# GPAS 四域 MOPD 实验计划

更新：2026-09-04。本版与修订后的正文、附录一致，替代此前以自定义 top-k estimator 为中心的方案。当前为实验规格；端到端训练器和四域实测结果尚未完成。

## 1. 要回答的实际问题

在任务重要性已经确定、dense MOPD 损失已经选定时，怎样分配 micro-batch，让学生更有效地利用教师反馈？

主线是固定任务权重下的计算分配：梯度更不稳定的任务获得更多样本，以更准确地估计它的任务均值。GPAS 用 Adam 的更新尺度衡量这种波动，并可结合任务耗时。结果先展示各域能力与 GPU-hour 效率，再用正常训练产生的轨迹解释分配。

top-k 是既有训练组件，Neyman allocation 是既有统计原则。论文的价值放在把它们变成可用的 MOPD 调度方法，以及对实际训练过程的解释。

## 2. 模型、教师与数据

| 项目 | 配置 |
|---|---|
| 学生 | Qwen3-1.7B，non-thinking |
| 数学教师 | 数学 RL 后的 Qwen3-1.7B checkpoint |
| IF 教师 | IF RL 后的 Qwen3-1.7B checkpoint |
| 代码、科学教师 | 同一个冻结的 Qwen3-4B，两个任务路由共享权重 |
| 任务权重 | 四域始终各 1/4；配方比较中的动态权重单独注明 |
| 训练 prompts | 每域 16,000 条，独立打乱一次，顺序消费，不重复 |
| 每个 micro-batch | 同任务 4 个 prompts，各采样 1 个 response |
| 每步 | 16 个 micro-batches，共 64 个 responses |
| 任务计数 | 2 ≤ m_i ≤ 8，且四域计数之和为 16 |
| 预算 | 每 run 500 步，32,000 个 attempted responses |
| 回复长度 | 上限 4,096；达到上限的回复保留有效 token |
| 更新 | 每批新 rollouts 只更新一次，K=1 |
| 种子 | 本轮每个配置 1 个训练 seed |
| 资源 | 每 run 一张 96GB 训练卡与一张 48GB rollout/teacher 卡 |

这是一组四域、三个不同权重集的教师配置，写作中称 four-domain study，不称四个独立 specialists。教师与初始学生都用本项目的统一评测器测分。

待补入配置文件的内容：精确 checkpoint/revision、模型 tokenizer、训练数据名称与版本、过滤和 held-out 切分、采样温度、EOS 处理、AdamW 参数、精度与显存配置、GPU 型号、训练/推理软件版本。初始 teacher gap 是描述训练状态的量，不决定任务权重或实验是否进行。

## 3. 默认 dense loss

采用 MOPD 的 teacher top-64 corrected reverse-KL：

    S = TopK(q_teacher, k=64)
    loss_position = sum_{a in S} [
        p_student[a] * (log p_student[a] - log q_teacher[a])
        - p_student[a] + q_teacher[a]
    ]

p 和 q 均保留全词表归一化。教师返回 token IDs 与全词表 log-prob；学生计算自己的全词表 log-normalizer，再 gather 对应位置。对上述完整表达式自动微分，包含 -p 项。

每个 response 内对有效位置求均值，再对 micro-batch 中的 response 求均值。核心比较中，先求每任务梯度均值，再乘固定 w_i 汇总。实现等价于每个 micro-batch loss 乘 w_i/m_i。

不再使用 student top-k head + sampled tail correction。标准损失不承担算法新颖性。

## 4. GPAS 在线实现

每步开始时，从已完成步骤的统计量确定 m，再生成当前数据。第一步用 Uniform。Adam 的预条件尺度 D 取更新前的 bias-corrected second moment，第一步尚无状态时用单位尺度。

按任务连续处理 micro-batches，对未加任务权重的梯度做 Welford 统计：

    h = 0
    S = 0
    for s, g in task_microbatch_gradients:
        delta = g - h
        h += delta / s
        S += (s - 1) / s * squared_norm(D * delta)
    e_hat = S / (m_i - 1)
    A += w_i * h

同一任务均值 buffer 循环复用，无需保存一组 micro-batch 梯度。相同递推同时记录 D=I 的噪声。混合精度梯度先 unscale，归约用 float32，组装 A 后再做常规 gradient clipping 和 AdamW 更新。

噪声 EMA 的 decay=0.9，第一条观测直接初始化。计时 EMA 同样为 0.9。GPAS (per step) 按 w_i sqrt(e_ema_i) 做有上下界的比例分配，再用 largest remainder 取整；所有分数为零时用 Uniform。GPAS (cost-aware) 枚举合法整数计数，最小化：

    (C + sum_i m_i * tau_i) * sum_i w_i**2 * e_ema_i / m_i

tau_i 是每任务 micro-batch 的边际时间，C 是固定 step 时间。它们来自实际执行 trace。常规前向、教师打分与反向照常进行；额外向量操作和 buffer 的耗时、显存计入报告。

## 5. 九个训练配置

| ID | 配置 | loss / 权重 | 用途 |
|---|---|---|---|
| 1 | Uniform | teacher top-64 / 固定等权 | 默认 dense MOPD 基线 |
| 2 | GPAS (per step) | 同上 | 梯度噪声驱动分配 |
| 3 | GPAS (cost-aware) | 同上 | 结合实际任务耗时 |
| 4 | Unpreconditioned allocation | 同上 | 计数使用 D=I，优化器仍是 AdamW |
| 5 | D³ signal, fixed weights | 同上 | 完整 gap × velocity 信号用于计数 |
| 6 | D³-MOPD scheduling | teacher top-64 / 随 m_i/G 加权 | 调度配方比较 |
| 7 | Open-MOPD (K=1) | student top-16 / 原配方 token-share 与 gap 权重 | 现有工程配方比较 |
| 8 | TA-OPD + Uniform | teacher top-64 + tail event / 固定等权 | 第二种现有 dense loss |
| 9 | TA-OPD + GPAS | 同 8 / 固定等权 | 同一分配方法迁移到另一损失 |

配置 1–5 共享训练目标，只比较分配。配置 6、7 的任务权重与 loss 聚合方式依配方变化；仍用统一评测指标呈现能力和 teacher agreement。配置 8、9 只有计数策略不同。

### D³ 两行的实现

保留原文 remaining gap × descent velocity，而非仅用原始 loss。初始归一化取前 5 个观测均值；EMA window=10；窗口 W=10，最多 R=3 个窗口；每 10 步更新；max-normalization；softmax temperature=0.5；每域 probability floor=0.10；batch jitter=0.30；KL denominator floor=0.15。history warmup 按原文算法执行。将得到的概率转换为本实验的 micro-batch 计数，并施加共同的 [2,8] 边界。

上述是对本项目 dense loss、micro-batch 单位与预算的受控适配。配置 5 用固定 1/4 汇总任务均值；配置 6 用 m_i/G 汇总。两者各自从本 run 的历史生成信号。

### Open-MOPD 的实现

对齐官方代码的 student top-16 candidate-wise dense loss：每个候选 token 都产生梯度，保留概率加权、token-share balancing 和 gap-aware weighting。四域 prompt 数各占 1/4。K=1 下不额外制造 rollout 复用；reward refresh 没有跨更新的陈旧性需要纠正。

实现参考固定 revision 4809a96cf85a869106ff0ff3f37d0a51e12010ae。集成时记录原实现中 gap smoothing、floor 与 normalization 的实际配置。本行称配方适配，不宣称复现原文模型、所有训练条件或原文结果。

### TA-OPD 两行的实现

使用 teacher top-64，并将集合外概率合并成一个 tail event：

    p_tail = 1 - sum_{a in S} p[a]
    q_tail = 1 - sum_{a in S} q[a]
    loss = sum_{a in S} p[a] * log(p[a]/q[a])
           + p_tail * log(p_tail/q_tail)

保持默认的 response averaging、task weighting、训练预算。用稳定的概率/对数运算实现边界处的 0 log 0 约定。

## 6. 评测与图表

### 能力与学习效率

主表报告初始学生、各任务指定教师，以及九个训练配置的：

- MATH-500 greedy pass@1；
- 固定 LiveCodeBench 切片 pass@1；
- IFBench strict accuracy；
- GPQA-Diamond average@4；
- 四项原始百分比分数的等权平均；
- 统一 held-out teacher top-64 loss；
- 默认损失、固定权重组到共同目标的 GPU hours。

benchmark 版本、代码题日期范围、解码和评测随机种子写入共享配置。避免按看过的最高 checkpoint 分数选择不同任务的结果；主表统一用第 500 步，学习曲线展示中间进展。

第 0、50、100、……、500 步在每域 128 个固定 held-out prompts 上，以相同采样设置和 seed 生成回复。所有方法均计算标准 teacher top-64 loss，再固定等权汇总成 F_eval；不要将各自不同的 training loss 直接作为统一纵轴。

方法间差异用按任务分层、按 prompt 配对的 bootstrap 区间，说明其反映评测样本不确定性。本轮不将其当作跨训练种子的稳定性估计。

默认损失、固定权重组的共同 target 是 Uniform 的最终 F_eval。首次到达时在相邻 checkpoint 间插值；未到达写 unreached。所有方法还报告实际总 GPU hours 和完整 F_eval 曲线，不通过 loss target 隐去不同配方。

### 分配如何变化

每步从正常训练记录：

- 每任务训练 loss、平滑 loss 与近期下降速度；
- preconditioned / unpreconditioned e_hat 和 EMA；
- m_i、消耗的 prompts、有效 token 数；
- response 长度与截断率；
- rollout、teacher scoring、backward、统计量收集、同步和 optimizer 耗时；
- 峰值显存与 step 总时间。

主图按同一横轴对齐 noise、counts、loss/velocity 和 response length。用图解释哪些任务获得更多样本、Adam 尺度是否改变分配、cost-aware 模式如何处理耗时差异。

不增加独立 token 重采样、分半梯度选择—评分、teacher-swap 或额外单任务训练作为前置检查。机制解释从真实训练过程和几个有明确用途的比较中形成。

### 稿件结果顺序

1. 主表和学习曲线：各域能力，matched-step 与 GPU-hour 效率。
2. 分配轨迹：实际预算如何转移，noise 与学习进展何时一致或不同。
3. cost-aware 模式与 TA-OPD 复用：实用收益、实现开销和适用情形。

所有占位段落只写待报告的量，不预先填入改善方向或成功叙事。

## 7. 工程安排与预算

先完成 dense loss、任务均值聚合和正常训练 loop，再接入 GPAS 与已有配方。用短运行确认 loss/backward、计数、显存、日志和 checkpoint resume 工作正常。这里仅处理实现问题，不设置噪声比、gap 大小或预测收益达到某个门槛才继续实验的流程。

按此前 8 个训练 slot 的资源布局，先启动配置 1–8；任一 slot 空闲后启动配置 9。其余评测卡负责共享 held-out 和 benchmarks。各 run 使用同样的 500 步预算；失败 run 在修正具体工程问题后记录修订配置并重跑。

按每 run 8–10 小时的原有排程估算，九个双卡 run 约 144–180 GPU hours，评测另计。这是排程假设，最终报告使用实测时间。checkpoint 每 50 步保存权重，另保留每 run 最新完整状态；完整状态包括 optimizer、prompt 位置、EMA 和 allocation。

本轮不包含模型训练结果的生成；完成训练后将实测 logs、评测输出与图表源数据回填到论文。

## 8. 本次采用的文献与借鉴

- [MOPD](https://arxiv.org/abs/2606.30406)：使用已有 teacher top-k corrected loss。
- [Open-MOPD](https://arxiv.org/abs/2608.19098)：重视 response/token 聚合和各域能力，准确区分 loss weighting 与 sampling。
- [D³-MOPD](https://arxiv.org/abs/2608.24987)：使用完整 gap × velocity 比较，展示随训练变化的预算。
- [TA-OPD](https://arxiv.org/abs/2608.14728)：用现有 tail-aware loss 展示调度复用。
- [Instella-MoE](https://arxiv.org/abs/2609.00791)：借鉴 specialist 与已有能力保留教师共用的工程动机。
- [Rethinking OPD II](https://arxiv.org/abs/2609.04172)：将注意力放在学生吸收反馈的更新效率。
- [EMA-PG](https://arxiv.org/abs/2602.04417) 与 [vOPD](https://arxiv.org/abs/2605.07865)：承认 token estimator 与方差控制已有工作，避免把 top-k 当新贡献。
