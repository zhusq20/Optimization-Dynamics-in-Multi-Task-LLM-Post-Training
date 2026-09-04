# MOPD 联合进展与计算分配：四域实验执行与分析方案

更新：2026-09-04。依据当前论文 *When Does Multi-Teacher On-Policy Distillation Make Joint Progress?*，并按本轮实验决定修订：**所有训练配置使用单种子；代码和科学域用 Qwen3-4B 暂代各自的 Qwen3-1.7B RL teacher；不设置防御性试验或 gate 实验。**这些执行决定优先于稿件中尚未同步的三种子及固定混合教师表述。本文件是待执行规格，尚无四域实测结果。

**执行摘要：**9 个联合训练配置各 1 run，4 个单任务参照各 1 run，合计 **13 个基础 runs**。在同一个 Uniform run 的第 50、250、450 步分析局部机制。主比较是 Uniform 与 GPAS；单任务参照与联合实验并行，用于解释能力获得情况，不作为开跑资格测试。

本版保留直接回答论文问题的能力比较、共同 checkpoint 干预、分配对照与成本分析；不增加教师资格筛选、先导实验通过门槛、额外模型/种子复验或条件触发的新调度器实验。实现参数及产物记录见后文，未知路径和版本按实际运行填写。

## 1. 要回答的问题

当一个学生能分别从各教师获得能力时，教师在实际 top-64 损失中提供的自身更新信号有多强，联合更新是否减弱它？即使不存在负向均值相互作用，可靠的下降也可能很慢。有限样本是否会使本来有用的更新损害某个域？增加样本或改变分配能否改善下降可靠性、下降幅度、实际能力或计算效率？

实验展示单任务与联合训练能力，用同一 checkpoint 的干预区分自身信号弱、合并后进展减少、估计不准确与可靠但缓慢的局部下降，再将诊断与实际能力和计算成本联系起来。这是结果组织顺序，不是串行 gate。单任务可学不证明各域参照分数能由一个模型同时达到；局部冲突也不证明联合目标不可达。

GPAS 是降低总更新方差的一种干预。任务权重和预条件尺度固定时，它不能改变均值更新的一阶符号，也不保证最小化最脆弱任务的方向误差。方向方差用于解释 trace noise 与实际进展的关系，本轮不增加方向分配新方法或额外验证分支。

top-k、Neyman allocation 和概率不等式均为已有组件。拟验证的贡献是：进展余量和方向不确定性是否能预测 MOPD 何时受益于精确采样，以及何时需要其他干预。尚未取得这些结果。

### 1.1 论文论点与所需证据

| 论文依据 | 实验问题与主观测 | 对照或干预 | 允许得出的结论 |
|---|---|---|---|
| `eq:capability_target`；实验节 Individual transfer | 学生是否分别获得各域能力；联合学生距离参照多远 | 初始学生、指定教师、4 个单任务学生、联合学生；匹配域曝光并报告计算 | 观察到的独立可迁移性与集成差距；不证明参照同时可达 |
| `eq:teacher_signal_projection`；`eq:transfer_margins` | 自身信号弱，还是合并后进展减少 | retained masses、masked-logratio variance、K_ii、w_i K_ii、a_i；单域/联合一步更新 | 区分自身进展与跨域贡献；信号标量本身不等于梯度有效性 |
| `eq:zero_conflict_rates`；Common descent can still be slow | 方向有利是否仍然下降很慢 | 绝对/相对 loss decrease；单任务、Uniform、Precise 和预算扫描 | 可靠但幅度小的局部进展；不据此识别真实 Hessian、容量或长期收敛率 |
| `eq:directional_variance`；`eq:directional_reliability` | 正余量是否被有限样本反转 | 独立 calibration、update trials、evaluation；预测反号频率 | 噪声相对余量能否预测采样失败；代入估计值的界不作置信证书 |
| `eq:batch_variance`；`eq:batch_cost_objective` | GPAS 是否改善相关不确定性和实际效率 | 固定目标的 Uniform/GPAS、D=I、D³ signal；实测耗时 | trace 代理的适用范围；方差降低不直接等于速度或能力提高 |
| `eq:policy_drift`；Capability, state distribution, and computation | fresh loss 改善来自旧状态优化还是状态变化；能力是否随之提高 | 相邻 checkpoint 的旧/新 reference banks；独立 benchmark | 分开报告两个 loss 变化项、能力和 GPU hours |

论文来源：[问题与损失](sections/004_preliminary.tex)、[理论](sections/005_theory.tex)、[实验与待填结果](sections/005_experiment.tex)、[实现与推导附录](sections/009_appendix.tex)。公式标识沿用 LaTeX label，避免稿件重新排版后公式编号失效。

### 1.2 预先固定的比较层级

1. **现象与机制：**单任务可迁移性、共同 checkpoint 的自身/合并余量、采样反号和实际 loss 变化。存在不确定或无效结果时仍保留全部域。
2. **端到端主比较：**配置 2 相对配置 1 在第 500 步的四域原始分数、均值、最差域差值，以及共同固定参考 loss 的到达成本，均使用本轮同一个训练种子的结果。
3. **次要比较：**配置 3–9 分别解释成本、预条件、调度信号、现有配方和 loss 的作用。全部方法与参照均为单种子，报告本轮实际差值，不报告跨种子均值/标准差。

GPAS 有三种分别报告的结果：降低 trace 方差；改善局部方向/下降幅度；改善端到端能力或实测计算。只满足前一项时，结论停留在前一项。

## 2. 模型、教师与数据

| 项目 | 配置 |
|---|---|
| 学生 | Qwen3-1.7B，non-thinking |
| 数学教师 | 数学 RL 后的 Qwen3-1.7B checkpoint |
| IF 教师 | IF RL 后的 Qwen3-1.7B checkpoint |
| 代码教师 | **暂用 Qwen3-4B 代替代码 Qwen3-1.7B RL teacher** |
| 科学教师 | **暂用 Qwen3-4B 代替科学 Qwen3-1.7B RL teacher**；当前与代码路由共享同一套 Qwen3-4B 权重 |
| 任务权重 | 四域始终各 1/4；配方比较中的动态权重单独注明 |
| 训练 prompts | 每域 16,000 条；联合训练打乱后顺序消费、不重复；单任务参照耗尽后重新打乱进入第二遍 |
| 每个 micro-batch | 同任务 4 个 prompts，各采样 1 个 response |
| 每步 | 16 个 micro-batches，共 64 个 responses |
| 任务计数 | 2 ≤ m_i ≤ 8，且四域计数之和为 16 |
| 预算 | 每 run 500 步，32,000 条计划 responses；失败重算与请求重试另计 |
| 回复长度 | 上限 4,096；达到上限的回复保留有效 token |
| 更新 | 每批新 rollouts 只更新一次，K=1 |
| 种子 | **所有 9 个联合配置与 4 个单任务参照均为同一个训练种子，各运行一次** |
| 资源 | 每 run 一张 96GB 训练卡与一张 48GB rollout/teacher 卡 |

表中计数范围适用于联合训练。单任务参照使用一个任务、权重 1、每步 16 个 micro-batches，其余基础配置一致；局部诊断的预算另列。

目标教师配置是四域各自的 Qwen3-1.7B RL teacher。当前数学、IF 已使用对应 RL teacher，代码和科学先由 Qwen3-4B **暂代**，这是当前实现状态，不是将“specialist + larger generalist”固定为研究设计。教师与初始学生用同一评测器测分并列入结果；不以 teacher gap 或单任务成绩作为实验启动条件。

每个 run 记录各域实际 teacher ID/revision 和 `temporary_teacher` 标记，主表注明代码/科学使用暂代教师。对应 1.7B RL teacher 就位后在后续完整 runs 中替换，不在一条训练曲线中途换 teacher；替换后的结果标明教师版本，不与暂代版本混写成同配置重复。当前实验无需等待替代结束。

待补入配置文件的内容：精确 checkpoint/revision、模型 tokenizer、训练数据名称与版本、过滤和 held-out 切分、采样温度、EOS 处理、AdamW 参数、精度与显存配置、GPU 型号、训练/推理软件版本。初始 teacher gap 是描述训练状态的量，不决定任务权重或实验是否进行。

### 数据隔离与配对约定

每域预先建立三个独立用途的数据集合：16,000 条训练 prompts、长期 loss 评估 prompts（固定 128 条）、诊断 prompt pool；另行冻结四个能力 benchmark。以题目/来源 ID 和规范化文本检查交叉重复；同一题的变体按同一组切分，过滤和去重计数写入 manifest。benchmark 题目不参与训练、调度、阈值选择或开发调参。

训练池、长期评估池、诊断池、benchmark 之间隔离；诊断池内部 calibration/trial/evaluation 使用同一分布、有放回独立抽样，允许独立抽中同一 prompt。不得用“角色不重用 response”替代数据集合之间的隔离检查。

同一配对 seed 的各方法共享每域训练 prompt 排列与初始权重，分配只改变每步消费数量。随机源按用途拆分：prompt order、rollout、调度 jitter、诊断、benchmark。回复随机种子由 run/域/曝光序号等稳定字段生成，避免某域多抽样改变其他域的随机序列；相同随机种子不意味着模型分叉后 response 仍相同。每步 rollout 服务使用本步更新前的学生版本，完成权重同步后才能生成下一步数据，记录 policy version。

单次数据调用失败按固定重试上限、相同请求内容处理并计入费用；训练有效预算以完整的 64 个 response 样本/步核对，网络重试另计请求 attempts。生成后无有效 token、非有限 loss/gradient、OOM 等情况使该步无效，从更新前状态恢复，不静默丢弃样本后缩小分母或继续更新。正常跑完时每 run 有 32,000 个计划 response 样本；若发生重算，额外生成量另列，不能仍称总 attempts 只有 32,000。达到长度上限的有效回复正常保留。

## 3. 默认 dense loss

采用 MOPD 的 teacher top-64 corrected reverse-KL：

    S = TopK(q_teacher, k=64)
    loss_position = sum_{a in S} [
        p_student[a] * (log p_student[a] - log q_teacher[a])
        - p_student[a] + q_teacher[a]
    ]

p 和 q 均保留全词表归一化。教师返回 top-64 token IDs 与对应的全词表归一化 log-prob；学生计算自己的全词表 log-normalizer，再 gather 对应位置。对上述完整表达式自动微分，包含 -p 项。

每个 response 内对有效位置求均值，再对 micro-batch 中的 response 求均值。核心比较中，先求每任务梯度均值，再乘固定 w_i 汇总。实现等价于每个 micro-batch loss 乘 w_i/m_i。

不再使用 student top-k head + sampled tail correction。标准损失不承担算法新颖性。

实现时冻结 teacher 和生成的 token/prefix，不对采样过程反传；仅 response 位置进入 loss，排除 prompt/padding，EOS 是否计为有效位置在共享配置中固定。tokenizer、词表 ID、chat template、位置对齐和 non-thinking 设置逐项核验。rollout 温度与打分分布温度分开记录；loss 中的 p/q 是所声明打分配置的全词表归一化概率，不能直接拿 top-p 截断后的生成概率替代。按位置/序列分块计算 top-64 项，避免一次保留所有位置的完整教师 logits。

若用未加权梯度做 Welford，则在任务均值汇总时只乘一次 w_i；若从已按 w_i/m_i 缩放的 micro-batch 梯度收集统计，必须先精确还原未加权梯度。二者择一实现，避免重复加权或把采样数变化混入 e_i。常规日志 loss、共同评估 loss 与各配方内部加权目标分别命名。

在已有 top-64 输出上记录实际损失对应的描述性教师信号。对每个固定 prefix，令 z(a)=1[a in S] log(q[a]/p[a])，按完整学生分布 p 计算：

    student_retained_mass = sum_{a in S} p[a]
    teacher_retained_mass = sum_{a in S} q[a]
    mean_z = sum_{a in S} p[a] * log(q[a]/p[a])
    conditional_var_z = sum_{a in S} p[a] * log(q[a]/p[a])**2 - mean_z**2

conditional_var_z 是 E_p[(z-E_p[z])² | prefix]；并非把 p 限制到 S 后重新归一化的方差。按原 loss 的 response 内位置平均、response 间平均记录它和 retained masses。只需要教师已经返回的 top-64 全词表归一化概率，以及学生已有概率；不额外请求教师全词表 logits，不新增大型梯度 buffer。该标量是 Cauchy–Schwarz 信号上界的一个因子，不是参数梯度协方差，不能单独证明梯度有用或教师质量；教师信号经 score features 的映射仍可能很弱。

教师信号到梯度的校验针对这一个 surrogate：-mu_i = E[z_i s_theta] = E[(z_i-E[z_i|prefix])s_theta]，其中 z 在集合外为 0。只在 top-64 集合内减 baseline 会改变更新；它不作为等价实现。保留质量和信号方差使用至少 float32 稳定归约，极小负方差只在规定舍入容差内截为 0，超容差则报错并保留原值。

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

噪声 EMA 的 decay=0.9，第一条观测直接初始化。计时 EMA 同样为 0.9。GPAS (per step) 枚举所有满足 sum_i m_i=16、2≤m_i≤8 的整数计数，最小化 sum_i w_i² e_ema_i/m_i；连续无约束参考比例为 w_i sqrt(e_ema_i)，实际使用精确可行整数解。并列最优时优先选择离 Uniform 最近的计数，再按固定任务顺序确定；所有分数为零时用 Uniform。GPAS (cost-aware) 枚举相同合法整数计数，最小化：

    (C + sum_i m_i * tau_i) * sum_i w_i**2 * e_ema_i / m_i

tau_i 是每任务 micro-batch 的边际时间，C 是固定 step 时间。它们来自实际执行 trace。该目标是时间乘更新方差，不等同于一般情形下的学习进展/秒。常规前向、教师打分与反向照常进行；额外向量操作和 buffer 的耗时、显存计入报告。

执行细化：固定任务顺序为 math、code、if、science；合法计数共有 **149** 组。并列以到 (4,4,4,4) 的平方欧氏距离最小者优先，再按该顺序取字典序最小者；浮点并列容差写入配置。NaN、Inf、负噪声或缺少历史统计属于实现/数值问题，记录后停止排查，不能悄悄偏向某域。

预条件 e_i 的 EMA 混合了历史 checkpoint 的 D 和策略，因此是滞后调度信号，不称为当前状态的无偏 e_i。在线训练池的无放回抽样也不同于诊断的独立协方差模型。记录每域剩余 prompt 数、原始/平滑 e_i 与 D 的状态标识；当前步骤的观测只用于下一步计数。

默认任务块串行执行，tau_i 覆盖该块的 rollout、teacher scoring、学生前向/反向及统计开销；C 覆盖未计入块内的同步、优化器和固定操作。同一耗时只能归属一项，预测时间与实际 step wall time 对比并报告误差。若改为任务重叠执行，时间模型对应实际执行顺序。Uniform 同样记录主协议要求的 trace 统计；收集开销直接从训练计时取得，不另开统计开关对照实验。

在共同状态、同一 e 和当前计数边界下，V_Uniform/V(m)≤2（分母为正）。该上限不适用于跨训练轨迹的实测方差比，也不构成两倍训练加速承诺。

## 5. 单任务参照与九个联合训练配置

四域分别运行单任务 OPD，与联合训练并行安排；从同一初始学生开始，使用当前对应教师和默认 loss，代码/科学也使用同一暂代 Qwen3-4B。每个参照运行 500 步、每步 64 个 responses，总计 32,000 条计划 responses；16,000 条 prompt 池耗尽后重新打乱，第二次访问重新生成 response。记录 unique prompts、总曝光量和重复次数。各域曲线与联合训练在匹配域曝光量处比较，同时给出总 GPU hours；32,000 次单域曝光与 Uniform 的 8,000 次单域曝光不直接用来归因集成损失。参照作为能力比较结果，不用于筛选教师或决定是否继续联合实验。

固定 Uniform 在第 200/400/500 步的单域曝光为 3,200/6,400/8,000，分别对应单任务第 50/100/**125** 步。因此单任务额外保存并评估第 125 步，提供最终 Uniform 的精确曝光匹配点；单任务第 250 步结束首次 16,000 条曝光，第 251 步开始第二遍。GPAS 按实测累计域曝光定位参照曲线，非评估点只作相邻点线性插值并明确标记，不外推，不写成实际测得的相同 checkpoint。相同曝光仍可能有不同更新次数、学习率阶段和其他域历史，不能单独用差值归因冲突。

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

配置 1–9 **每个配置只运行 1 个训练种子**，单任务参照也各 1 个种子。共 **9 个联合 runs + 4 个单任务 runs = 13 runs**；Uniform 与 GPAS 是同种子的主要比较。

运行标识为 `uniform-s1`、`gpas-step-s1`、配置 3–9 各 `*-s1`、`single-{math,code,if,science}-s1`。s1 是唯一训练种子的别名，实际整数写入 manifest。rollout、评估与诊断抽样使用该种子派生的随机源；诊断中 20 次独立抽样是同一 checkpoint 的一步干预重复，不是 20 个训练种子或完整训练 runs。

配置 1–5 共享训练目标，只比较分配。配置 6、7 的任务权重与 loss 聚合方式依配方变化；仍用统一评测指标呈现能力和 teacher agreement。配置 8、9 只有计数策略不同。

### D³ 两行的实现

保留原文 remaining gap × descent velocity，而非仅用原始 loss。初始归一化取前 5 个观测均值；EMA window=10；窗口 W=10，最多 R=3 个窗口；每 10 步更新；max-normalization；softmax temperature=0.5；每域 probability floor=0.10；batch jitter=0.30；KL denominator floor=0.15。history warmup 按原文算法执行。将得到的概率转换为本实验的 micro-batch 计数，并施加共同的 [2,8] 边界。

上述是对本项目 dense loss、micro-batch 单位与预算的受控适配。配置 5 用固定 1/4 汇总任务均值；配置 6 用 m_i/G 汇总。两者各自从本 run 的历史生成信号。

附录 `app:baselines` 对信号的定义为：

    r_i(t) = Lbar_i(t) / max(L_i_initial, 0.15)
    velocity_i(t) = max(0, mean over j=0,...,R'-1 of
        (Lbar_i(t-(j+1)*W) - Lbar_i(t-j*W))
        / max(Lbar_i(t-(j+1)*W), 0.15))
    signal_i(t) = r_i(t) * velocity_i(t)

R' 是当前可用的完整历史窗口数。不把“无历史”当作 velocity=0 后直接做不稳定归一化；history warmup、EMA window 到递推系数的转换、零信号处理、floor/softmax/jitter 的运算顺序须依据固定来源实现并在配置中冻结。概率转计数采用本项目统一适配：在 149 组可行整数中最小化 sum_i(m_i-16*p_i)²，再按第 4 节并列规则处理；报告原概率、最终计数及边界命中率。这不是 GPAS 的方差最小化，也不宣称计数边界仍保留原始概率 floor 的精确数值。

### Open-MOPD 的实现

对齐论文附录指定的 student top-16 candidate-wise dense loss：每个候选 token 都产生梯度，保留概率加权、token-share balancing 和 gap-aware weighting。每步各域固定 4 个 micro-batches，四域 prompt 数各占 1/4，变化的是配方权重。K=1 下不额外制造 rollout 复用；reward refresh 没有跨更新的陈旧性需要纠正。

实现参考固定 revision 4809a96cf85a869106ff0ff3f37d0a51e12010ae。集成时记录原实现中 gap smoothing、floor 与 normalization 的实际配置。本行称配方适配，不宣称复现原文模型、所有训练条件或原文结果。

### TA-OPD 两行的实现

使用 teacher top-64，并将集合外概率合并成一个 tail event：

    p_tail = 1 - sum_{a in S} p[a]
    q_tail = 1 - sum_{a in S} q[a]
    loss = sum_{a in S} p[a] * log(p[a]/q[a])
           + p_tail * log(p_tail/q_tail)

保持默认的 response averaging、task weighting、训练预算。用稳定的概率/对数运算实现边界处的 0 log 0 约定。

## 6. 共同 checkpoint 诊断与预测验证

诊断覆盖 **一个 Uniform run 的第 50、250、450 步，共 3 个 checkpoint**，直接用于检验均值进展、采样精度与实际下降的关系。诊断与训练并行进行，任何诊断结果都不作为后续训练的通过条件。每个 checkpoint 的所有分支使用同一学习率与完整优化器初始状态；Precise 只增大样本数，不按 batch size 放大学习率。

### 数据、状态与分支

从 Uniform 的第 50、250、450 步保存完整状态。每个诊断分支加载相同的学生、AdamW 一阶和二阶矩、更新步数、学习率状态、随机状态、prompt 位置和统计量，再进行一次更新。诊断不写回主训练轨迹。所有诊断 rollout 都由分支更新前的 checkpoint 策略生成。

每个 checkpoint 从各域同一个预先指定的诊断 prompt 分布独立抽取以下三组样本。诊断分布与训练和 benchmark 数据隔离；组间不复用抽样记录或 response，但有放回独立抽样可能选到相同 prompt，不强制 prompt 身份互斥：

1. Calibration：每任务 32 个独立 micro-batches，分成两个各 16 个的独立半组 A/B，用于交叉拟合均值乘积、估计进展余量、trace noise 和方向方差，并预先选定诊断 GPAS 计数。
2. Update trials：每个分支进行 20 次独立更新试验，每次重置到相同完整状态，再指定独立的试验抽样种子；抽样记录与 calibration 分离。
3. Evaluation：每域 128 个独立抽样的 held-out prompts，由当前 checkpoint 生成参考 prefixes，随后固定；所有分支均在这些相同状态上测前后 loss，并用评估梯度估计一阶投影。这些样本不参与 calibration 或计数选择。

诊断的三种用途在每域共享同一个 prompt 分布并有放回独立采样；其中的 mu_i 指该诊断分布下的均值，其与训练效果的关系通过现有端到端结果对照。主训练使用有限池无放回流。记录这一区别，不把独立样本公式称为主训练流的精确协方差；计数选择与效果评估使用独立样本。

每次试验比较：

| 分支 | 单次更新样本 | 用途 |
|---|---|---|
| 四个单任务分支 | 各自使用该任务 16 个 micro-batches，权重 1 | 估计单域更新对全部域的实际影响 |
| Uniform joint | 每域 4 个，共 G=16 | 普通联合更新 |
| GPAS joint | G=16，m_i 在 [2,8]，由 calibration 选择 | 检验 allocation 干预 |
| Precise joint | 每域 32 个，共 G=128，仍固定等权 | 更精确均值的昂贵诊断；不作等成本对手 |

等分配预算扫描取每域 m_i=2、4、8、16、32，对应 G=8、16、32、64、128，每档 20 次更新抽样。**G=16/128 直接复用 Uniform/Precise 主分支结果，只新增 G=8/32/64 三档**；复用结果在图表中引用同一 trial IDs，不重复计算为独立证据。新增档的 update draws 与其他分支独立。局部扫描不使用主训练的 m_i≤8 上限，用于区分改变总样本数与改变分配；其一步收益不等同于固定总计算量下改变更新频率的训练收益。

实际执行的 10 个分支之间、各 trial 之间的 update draws 独立，trial 编号相同不表示复用样本；预算图中的两个复用端点是主分支的别名。所有分支共享 evaluation bank，因此 loss 差值可按同一评估 prompt 配对，而不同分支的更新随机性分别处理。跨 trial 共用的 bank 只计作一份评估数据。

### 记录的诊断量

对 checkpoint 的 pre-step D，计算 K_ij=mu_i^T D mu_j 与 a_i=(Kw)_i。具体使用 K_hat_ij=(mu_hat_i,A^T D mu_hat_j,B + mu_hat_i,B^T D mu_hat_j,A)/2，而不直接平方同一个带噪均值；尤其对角项由独立均值相乘，避免正偏差误判进展。K 中只有一个 D，不能用 D mu_i 和 D mu_j 的余弦替代任务的一阶进展。calibration 估计总 trace 方差以及 v_i(m)=sum_j w_j² Q_ij/m_j，其中 Q_ij=mu_i^T D Sigma_j D mu_i；仅对可靠正余量域报告 v_i/a_i²。

**方向方差的执行估计：**仅把同一个噪声均值投影到独立协方差上仍会产生二次型的均值估计偏差。为此将已有 32 个 micro-batches 的 A/B 再各分为两个 8 个样本的块，得到四个独立块 B1…B4，不额外增加 calibration 样本。对任意不同块 r、s、u，以 B_r、B_s 的任务 i 均值作为两个方向，以 B_u 的任务 j 梯度估计协方差：

    x_b = mu_hat_i,r^T D g_j,u,b
    y_b = mu_hat_i,s^T D g_j,u,b
    Q_hat_ij^(r,s;u) = sum_b (x_b-mean(x))*(y_b-mean(y)) / 7

对 r<s、u 不属于 {r,s} 的 12 个组合取均值。独立同分布和有限矩条件下，该交叉乘积避免两次使用同一噪声方向的正偏差；各组合共享样本，不能作为 12 次独立重复。用于 K 的 A/B 仍各含 16 个样本；e_i 用全部 32 个样本的无偏样本方差。

Q_hat、交叉拟合 K_hat 的对角项都可能因估计误差出现负值，即使总体 Q_ij、K_ii 非负。保留原估计及区间，噪声导致的负值不解释为负“方差”或负自身进展，也不直接代入概率界。a_i 接近零或区间跨零时，优先展示 a_i 与 v_i 原值，归一化风险标为不可稳定估计；这些标记只影响指标展示，不影响任何训练或干预的执行。

另外报告交叉拟合的自身进展 K_ii、加权自身项 w_i K_ii、合并余量 a_i 以及跨任务贡献 c_i=a_i-w_i K_ii。并列比较单任务分支和更精确联合分支的自身域 loss 变化，所有量保留不确定性。c_i<0 表示其他任务削弱了本域一阶进展；a_i<0 才表示当前加权均值方向对本域不利。单纯 a_i<K_ii 也可能来自任务权重，不能一概称为冲突。单任务分支权重为 1，联合中的自身权重为 1/4，实际 loss 差异不能全部归因跨任务干扰。

高维梯度按域和参数块处理，不显式构造参数维度的 Sigma_j。四块交叉拟合与区间估计可使用梯度点积的标量 Gram 记录、诊断阶段的 CPU/offload 或确定性重算，按实际内存选择实现并记录。在线 GPAS 的单任务均值 buffer 与诊断临时存储分别计费。保存样本标识、种子、prefixes、必要均值/点积，重算耗时计入诊断。

对每次 trial，使用独立 evaluation bank 的梯度估计冻结 D 下的投影，再记录实际 AdamW 一步后的各域固定状态 loss 变化及该更新耗时。为独立核验 a_i，将 evaluation 的每域 128 个 response 分成独立的 64/64 半组，按同样的交叉乘积公式得到 K_eval 和 a_eval；它们不参与 calibration 或计数选择。完整 bank 均值可用于每个 trial 的投影，但 evaluation bank 内共用数据的相关性必须保留。

为避免符号混淆，每个 trial 至少保存以下量：

| 量 | 定义 | 正值含义 |
|---|---|---|
| X_i | mu_hat_i,eval^T D A；A 为 clipping 前的组装梯度 | 冻结 D 下预计下降 |
| d_i | L_i^t(theta_before) - L_i^t(theta_after) | 实际 AdamW 一步下降 |
| d_i_relative | d_i / L_i^t(theta_before)，仅分母超过 floor 时 | 相对当前 loss 的下降 |
| P_i_actual | -mu_hat_i,eval^T Delta_theta，Delta_theta 为实际参数变化 | 实际更新方向的一阶预计下降 |
| optimizer_deviation_i | P_i_actual - eta*X_i | 优化器方向/幅度与冻结 D 模型的差异 |
| remainder_i | d_i - P_i_actual | 有限步长非线性与估计/数值余项 |

另存 A 与 Delta_theta 范数、clipping scale、当前学习率和更新步数。这里 remainder_i 不是 Hessian 的直接测量。clipping、momentum、当步矩更新、weight decay 和有限步长都可能影响结果；只看 AdamW loss 上升不能判定理论均值方向有害。把估计值代入 Cantelli 上界不构成有效置信证书。

仅当独立验证的 a_i 估计区间完全小于零时，将该域判为当前加权均值方向不利。即使冻结 D 的余量为正，更精确的实际 AdamW 更新仍可能因 momentum、曲率或其他有限步长效应使 loss 上升；这种结果是模型近似偏差，不自动证明均值冲突。若余量为正但普通投影常反号，GPAS 和增大预算有可检验的目标。余量区间跨过零时标记不确定，不强行归为任一种机制。局部各域同时下降不是长期联合训练成功的必要条件。

“可靠但缓慢的局部进展”通过下降频率与下降幅度并列展示：比较单任务、Uniform、GPAS、Precise 的绝对 d_i 和相对 pre-update loss 的变化，观察提高精度是否只减少反号而没有明显增大下降幅度。相对指标仅在 pre-update loss 高于数值分母下限时计算；它不是相对未知最优值的收敛率。近零变化保留区间，不用人工通过阈值筛选结果。

### 统计与结果解释

全部长期训练采用单种子，主表给出实际分数和相对 Uniform 的差值。题目 bootstrap 用于表达评估样本误差；它不产生跨训练种子标准差。GPQA 同题的 4 次回答作为一簇，各方法按同题配对；四域均值按域分层汇总。

诊断以每个分支的 20 次独立更新抽样报告 X_i≤0 的频率、实际 d_i≤0 的频率、d_i 的均值/分布以及相对变化。calibration 的 micro-batches、update trials 与共用 evaluation bank 各有不同随机性：重采样时，所有分支使用同一组 evaluation prompt 索引，各分支的 trials 分别重采样；保留 calibration 交叉拟合中的共享样本关系。默认展示 95% 评估/抽样区间，区间宽则直接展示，不自动追加试验。

主分析使用连续量，不设“必须达到某阈值才继续”的机制分类流程：

| 要解释的现象 | 对应结果 |
|---|---|
| 自身信号弱 | K_ii、自身单任务更新的下降幅度、masked signal 统计并列展示 |
| 合并减弱进展 | c_i=a_i-w_i K_ii 的符号/区间；a_i 与 w_i K_ii 的差异 |
| 联合均值方向不利 | a_i 的独立估计及区间；区间跨零时保留不确定性 |
| 采样精度影响进展 | 正 a_i 下的 X_i 反号频率、v_i/a_i² 与增大 G/GPAS 后的实际变化 |
| 下降可靠但幅度小 | 下降频率高，同时 d_i 及 d_i/pre-loss 较小；保留完整幅度与区间 |
| 实际优化器与局部模型不一致 | eta*X_i、P_i_actual、d_i 的差异 |

本轮不新增真实 Hessian 测量、额外教师/模型对照、方向调度器或自动补样本实验。Precise 仍是有限样本估计，直接报告其误差与下降幅度。

### 用现有 checkpoint 分析预测关系

在每个 checkpoint 先用 calibration 记录 a_i、v_i/a_i²、trace 代理及 Uniform/GPAS/Precise 的风险排序，再与独立 update trials 的反号频率和实际下降比较。第 50 步形成的解释可在第 250、450 步继续对照；不设置独立预注册、资格验证或先导通过阶段，也不增加 held-out 训练种子。

报告方向得分与实际反号频率的排序关系、GPAS−Uniform 和 Precise−Uniform 的变化及不符合预测的域。单独 V(m) 在同一 checkpoint/分配下对所有域相同，只能比较 checkpoint/分配级风险；另列含域信息的 ||mu_i||² V(m)/a_i² 代理。Cantelli 插件值作为上界型诊断量，不当作真实反号概率或保证。事后形成的解释标为事后分析，不写成预先预测成功。

方向方差仅用于本轮机制解释。论文附录讨论的方向分配扩展不列入当前实验清单或预算。

## 7. 能力、状态分布与计算报告

### 三类评测分别报告

主能力表保持紧凑，报告初始学生、指定教师、四个单任务参照，以及九个联合配置的 MATH-500 greedy pass@1、固定 LiveCodeBench 切片 pass@1、IFBench strict accuracy、GPQA-Diamond average@4 和四项原始百分比分数均值。单任务参照一行来自四个独立学生，不能写成一个共享学生的能力。

统一指标口径如下，精确 evaluator 与 prompt 模板版本仍须冻结：

| 域 | 主指标 | 必须记录 |
|---|---|---|
| Math | MATH-500 greedy pass@1：每题一个 greedy 回答的正确率 | 答案提取/等价判定、最大长度、无效答案处理 |
| Code | 固定 LiveCodeBench 切片 pass@1：每题一个回答 | 题目 ID、发布日期起止、题数、测试环境、执行超时、解码设置 |
| IF | IFBench strict accuracy | 固定 evaluator 的 strict 字段及聚合粒度；prompt-level 与 instruction-level 分开命名，不能混用 |
| Science | GPQA-Diamond average@4 | 每题 4 次独立回答正确率的均值，再对题目平均；不是“4 次任一答对”的 pass@4；题目/选项顺序、解析器和随机种子 |
| 汇总 | 四项百分比分数的算术均值 | 各域等权，保留各域原始分数，不混入 loss |

指定教师行报告各域实际 teacher 的成绩，并在 Code/Science 两列标注“Qwen3-4B 暂代 Qwen3-1.7B RL teacher”；单任务参照行由 4 个学生的域内成绩组成。两行都不作为一个共享模型的联合成绩。

benchmark 版本、代码题日期范围、解码和评测随机种子写入共享配置。主表统一用第 500 步，曲线展示中间 checkpoint，避免对各域分别挑最高点。单任务归一化收益仅在参照增益高于预先指定的正分母下限时报告；下限与能力容差 epsilon_i 均在看结果前写入配置，否则只报告原始增益。始终同时给出原始分数。

能力指标计算固定为（C_i 为百分比分数）：

    acquired_gain_i = C_i(theta) - C_i(theta_initial)
    reference_gain_i = r_i - C_i(theta_initial)
    normalized_gain_i = acquired_gain_i / reference_gain_i
        # 仅 reference_gain_i > reference_gain_floor_i 时定义
    target_shortfall_i = max(0, r_i - epsilon_i - C_i(theta))
    delta_i = C_i(GPAS_s1, step500) - C_i(Uniform_s1, step500)
    delta_mean = mean_i(delta_i)
    delta_worst = min_i(delta_i)

r_i 默认是单任务第 500 步的域内原始分数；曝光匹配分析另用 r_i(E)，不能混合两个分母。归一化增益不裁剪到 [0,1]，大于 1 或负数按原值保留。全域目标同时满足与否只在预先固定 epsilon_i 后描述，并给出不确定性；参照本身只有一个训练 seed，其随机性限制必须保留。教师优于学生不自动等于学生可迁移；teacher gap 不参与决定 w_i。

第 0、50、100、……、500 步在每域 128 个固定 held-out prompts 上测量以下不同指标：

- 固定参考状态 loss：初始学生生成一次 reference prefixes 并固定，所有 checkpoint 在同一 bank 上计算标准 teacher top-64 loss。
- Fresh-policy loss：当前学生在相同 held-out prompt 集上生成新回复，以共享采样配置计算同一 loss。
- 下游能力：使用上述 benchmark 协议，不以 loss 下降代替能力增益。

完整协议在同一组第 0、50、…、500 步执行共享能力评测；单任务第 125 步增加一次。若资源要求使用较稀疏的 benchmark 曲线，必须在开跑前统一修改各方法的评测网格并更新本规格，不能看过结果后只补有利 checkpoint。所有方法初始学生相同，第 0 步能力与 reference bank 可共享一次，明确其为共享测量。训练数据和调度器读取不到 held-out/benchmark 分数。

每 50 步保存该 checkpoint 的 fresh reference prefixes 和 teacher scores。在下一评测点 s=t+50，额外在旧 bank 上测 L_i^t(theta_s)，从而分别估计：旧状态优化项 L_i^t(theta_s)-L_i^t(theta_t)，以及状态分布变化项 L_i^s(theta_s)-L_i^t(theta_s)。这里的 L 是 held-out prompt 分布上的经验评估损失，对应正文分解在评估分布上的版本，不等同于剩余训练池的损失。二者在相同经验 bank 定义下相加为 fresh loss 的 checkpoint 间变化；采样误差单独报告。已保存旧 prefixes/teacher scores，因此额外工作主要是学生评估前向，费用计入评测成本。

每个相邻区间写出以下四列并检查代数闭合（此处负值为 loss 改善，与诊断 d_i 的正值下降约定相反，字段名明确区分）：

    old_state_change = L_old(theta_s) - L_old(theta_t)
    state_change = L_new(theta_s) - L_old(theta_s)
    fresh_total_change = L_new(theta_s) - L_old(theta_t)
    decomposition_residual = fresh_total_change - old_state_change - state_change

residual 仅应有数值归约误差。teacher scores、token masks、response averaging 和同一 bank ID 必须完全一致。状态变化项可以为正或负，也含有限 response 采样误差，不直接命名为能力/覆盖改善。固定初始 bank loss 与 fresh loss 的简单差值不是上述相邻区间分解。

共同 checkpoint 干预另有该 checkpoint 生成的独立 reference bank，分支之间保持固定。长期 initial bank、相邻 checkpoint 分解和局部诊断 bank 分别标注；initial 与 fresh loss 两条曲线的差本身不构成上述时间变化分解。

所有方法和单任务参照统一标注 **single training seed**，展示该 run 的原始分数和方法差值。配对 prompt bootstrap 仅描述评测样本不确定性，不输出跨训练种子的均值、标准差或稳定性结论。

### 成本与日志

默认损失、固定权重组的共同 loss target 为 **唯一 Uniform run 第 500 步的固定参考 loss**。首次到达时在相邻 checkpoint 间插值；未到达写 unreached。所有方法报告实际总 GPU hours，以及分别按每域曝光、steps、GPU hours 绘制的 loss 与能力曲线。最差域相对 Uniform 的分数变化与均值一起报告。

具体令 F_ref(t)=sum_i(1/4)*L_i,initial_bank(theta_t)，target=F_ref,Uniform_s1(500)。配置 1–5 使用同一 target，按第一次向下穿越的相邻评估点插值，不外推；初始已满足记 0，未达到记 `unreached`，曲线非单调仍保留全部观测。该 target 是本轮基线完成后得到的共同参照。配置 6–9 也展示公共 loss 曲线，其配方差异单独解释。

GPU hours 包括两张保留设备上的等待、rollout、teacher scoring、backward、统计量、同步和 checkpoint 时间。额外诊断和 benchmark 评测单列，不能隐去昂贵 precise branches。

每次运行报告 learner GPU hours、rollout/teacher GPU hours、两者之和与墙钟时间；不同型号分别列出 GPU hours，不假设 96GB 与 48GB 是相同计算能力。训练设备等待评测而仍被保留的时间计入训练占用；独立评测设备另计，重叠区间按 device ID 去重。time-to-target 使用到该 checkpoint 的训练设备累计占用，另列包含必要评测和诊断的研究总费用。

正常训练日志保留每域 loss/velocity、preconditioned 与原始 trace noise 及 EMA、m_i、prompt 曝光/唯一数、有效 token、回复长度/截断率、各阶段耗时和峰值显存。共同 checkpoint 追加 K_ii、w_i K_ii、a_i、方向方差、绝对/相对局部 loss decrease、估计误差和实际干预结果；常规 teacher top-64 记录保留概率质量和 centered masked-logratio conditional variance。Cost-aware 结果报告拟合和实测时间及统计开销；TA-OPD 配对仍作为探索性 loss 稳健性比较。

### 稿件结果顺序

1. 单任务可迁移性与联合能力：匹配域曝光和总计算量，报告不确定性。
2. 共同 checkpoint 干预：自身信号与合并进展、可靠但慢的下降、采样失败、精确估计和预算扫描；描述性教师信号单独标注。
3. 预测关系：同一 Uniform run 的早、中、晚 checkpoint 中，诊断量与实际干预的对应和不一致情况。
4. 端到端能力与计算：单种子核心比较、最差域变化、两个状态 loss 和成本。
5. 分配轨迹、cost-aware 与 TA-OPD：解释范围及单 seed 限制。

首图使用明确标注的解析示例：e^{-t} 与 e^{-0.1t} 展示无冲突仍可有不同速率，Gaussian 例子展示均值和采样可靠性的区别。它不报告真实 MOPD 速率，也不替代单任务/联合能力实测图。GPAS 方法示意图保留在附录。所有实测占位段落和图表仅写待测量内容，不填入改善方向、虚构数据或成功结论。

## 8. 工程安排与预算

### 8.1 排程

Uniform、GPAS 与四个单任务参照优先并行安排，配置 3–9 随可用资源排入。保存 Uniform 的第 50、250、450 步，诊断直接从这些 checkpoint 分支运行；无需等单任务参照达标、teacher gap 达标或早期诊断通过才继续训练。

正常实现中核对 loss、梯度聚合、恢复与日志，不额外设置先导训练、资格筛选、合成验证实验或通过后才放行的流程。每个计划实验无论观察到改善、持平还是下降都汇总到对应结果。

此前最多 8 个双卡 slots 仅作排程假设，实际以可用设备安排。临时教师替换状态随每个 run 的 manifest 记录；当前 Qwen3-4B 暂代版本可以直接用于本轮完整比较。

### 8.2 工作量

**基础训练：**9 个联合 runs + 4 个单任务 runs = **13 runs**；合计 6,500 次更新、104,000 个 micro-batches、**416,000 条计划 responses**。按此前每 run 8–10 墙钟小时、各占两张 GPU 的排程假设，基础训练约 **208–260 GPU hours**；实际以各域长度、教师耗时和设备占用记录为准。

**每个诊断 checkpoint：**主分支包含四个单任务、Uniform、GPAS、Precise，共 7 个分支，各 20 次一步更新抽样。等分配扫描的 G=16/128 直接使用 Uniform/Precise 的对应结果，只新增 G=8/32/64 三档，不重复跑等价端点。

| 项目 | 计算 | 新生成 responses |
|---|---|---:|
| Calibration | 4 域 × 32 micro-batches × 4 | 512 |
| 主分支 trials | 20 × (4 个单域分支×16 + Uniform 16 + GPAS 16 + Precise 128) × 4 | 17,920 |
| 新增预算扫描档 | 20 × (8+32+64) × 4 | 8,320 |
| 固定 evaluation bank | 4 域 × 128 prompts × 1 | 512 |
| 单 checkpoint 合计 | 200 次分支更新，evaluation bank 仅生成一次 | **27,264** |
| 3 个 checkpoint 合计 | 1 个训练种子 × 3 时点 | **81,792** |

每个 checkpoint 的 200 个更新后模型分别在 512 个固定 response 上评估，共 **102,400 次 response 学生评分前向**；3 个 checkpoint 合计 **307,200 次**，另计更新前评分、evaluation 梯度与 calibration 点积/重算。teacher scores 可以缓存；固定 response 的评分前向不计为重新生成回复。

基础训练与诊断合计 **497,792 条计划新生成 responses**，另列长期 loss banks、能力 benchmark、单任务第 125 步额外评测和故障重算的费用。诊断是 600 次局部更新，不是新增 600 个训练 runs。训练、诊断、评测分别记录 GPU hours，再给出总费用；不把不同 batch size 的诊断套用普通训练单步价格。

### 8.3 保存与产物

每 50 步保存模型用于能力/曝光曲线，单任务另存第 125 步。Uniform 的第 50、250、450 步保存完整模型、AdamW 一/二阶矩、step counter、LR scheduler、随机状态、数据位置、EMA 和 allocation，另保留最新恢复点。诊断不修改源 run。

故障恢复与额外生成量记录在原 run 的日志中；模型、教师或目标配置发生变化时用新 run ID。结果表逐项列出已完成和未完成的计划实验，不拼接不同教师配置的训练片段。

## 9. 本次采用的文献与借鉴

以下沿用当前稿件的文献定位；实验实现以论文附录及已固定来源版本为准。它们不是本次新增实测证据。集成第三方代码时再记录实际 revision、配置和与本协议的差异。

- [MOPD](https://arxiv.org/abs/2606.30406)：使用已有 teacher top-k corrected loss。
- [Open-MOPD](https://arxiv.org/abs/2608.19098)：借鉴单任务参照与 integration-gap 诊断，同时准确区分 loss weighting 与 sampling。
- [D³-MOPD](https://arxiv.org/abs/2608.24987)：使用完整 gap × velocity 比较，展示随训练变化的预算。
- [TA-OPD](https://arxiv.org/abs/2608.14728)：用现有 tail-aware loss 展示调度复用。
- [Instella-MoE](https://arxiv.org/abs/2609.00791)：借鉴 specialist 与已有能力保留教师共用的工程动机。
- [CaMOPD](https://arxiv.org/abs/2605.27115)：已有更新冲突分析意味着仅画梯度冲突图不足以构成新贡献。
- [Rethinking OPD II](https://arxiv.org/abs/2609.04172)：固定状态下的慢对齐支持研究优化障碍，不证明梯度噪声是原因。
- [EMA-PG](https://arxiv.org/abs/2602.04417) 与 [vOPD](https://arxiv.org/abs/2605.07865)：承认 token estimator 与方差控制已有工作，避免把 top-k 当新贡献。

## 10. 运行配置记录

以下是实际实现需要记录的参数，不是额外实验或审批 gate。沿用已有训练配方填写未知路径/版本，不为每项参数增加资格测试或超参数扫描。

| 组 | 记录内容 |
|---|---|
| 学生与教师 | student、各域 teacher 的路径/revision/哈希、tokenizer、chat template、non-thinking；代码/科学的 `temporary_teacher=true` 及其目标 Qwen3-1.7B RL teacher |
| 数据 | 各域训练/长期评估/诊断数据来源和 IDs、切分与去重、prompt 排列 |
| rollout 与优化器 | 解码/打分温度、长度/EOS、AdamW 学习率/schedule/betas/epsilon/decay、clipping、精度、模型 train/eval mode、policy 同步 |
| 随机源 | 唯一训练种子 s1 的实际整数及 rollout/调度/诊断/评估派生随机源 |
| 比较实现 | GPAS 计数/EMA/计时；D³ 版本、warmup/EMA/零信号与概率处理；Open-MOPD 版本及 gap/token 权重；TA 边界处理 |
| 评估 | benchmark revision、Code 日期范围、IF strict 粒度、各域解码/评分器/随机源、相对指标分母下限；如报告目标达标，再记录 epsilon_i |
| 诊断 | 样本 IDs、A/B 与四块划分、均值/点积保存方式、evaluation bank、实际区间算法与重采样设置 |
| 系统 | GPU 型号/设备 ID、训练/推理/评估版本、内存与耗时、诊断 offload/重算占用 |

若训练包含 dropout 等额外随机源，记录其模式；诊断的均值/梯度应对应同一声明损失函数和随机性定义。

## 11. 结果产物与论文回填

以下是待实现的产物约定，不表示文件已存在。路径均位于未来实验产物根目录，由 artifact manifest 记录真实位置。

| 产物 | 最低字段/内容 | 回填位置 |
|---|---|---|
| `run_manifest` | run ID、方法、唯一训练 seed、各域实际 teacher/revision 与暂代标记、状态、配置/代码/模型/数据哈希、设备、父 checkpoint | 附录配置与可复现性 |
| `training_steps` | run/step/task、policy version、m_i、有效/重试/失败 response 数、exposure/unique/epoch、loss、raw/EMA e、retained masses、masked signal variance、tokens/length/truncation、clipping、各阶段时间/峰值显存 | 分配轨迹、工程开销 |
| `capability_items` / `capability_summary` | 模型 checkpoint、benchmark 版本、题目/样本 seed、原输出与得分、聚合分母、各域分数/均值、原始/归一化增益和有效性 | `tab:main_results`、`fig:individual_joint` |
| `loss_banks` / `state_decomposition` | bank ID、prompt/response IDs、生成模型、teacher scores、mask、loss 配置、旧/新/initial loss、分解各项与闭合误差 | `fig:training_efficiency`、状态分布分析 |
| `diagnostic_calibration` | checkpoint/域/独立块/样本 IDs、K/Q/e、c_i/a_i、区间/有效标记、所选 m、重采样计数稳定性、D 标识 | `fig:checkpoint_diagnostics` |
| `diagnostic_predictions` | calibration 来源、各域风险排序/预期变化、记录时间、对应 trial 结果、是否为事后解释 | 诊断量与干预的关系 |
| `diagnostic_trials` | checkpoint/branch/trial seed、G/m、update 与 evaluation bank IDs、a_eval、X/d/d_relative/P_actual/deviation/remainder、资源用量 | 机制判读、实际 AdamW 对照、预算扫描 |
| `resource_ledger` | device ID/type、用途、起止/占用时间、生成/评分/重算数量、失败/重试、是否与其他事件重叠 | 训练、诊断、评测和总 GPU hours |
| `analysis_spec` / 图表源表 | 指标定义、分母处理、区间方法、版本/哈希、纳入/缺失清单、单种子图表输入 | 结果可追溯与统计口径 |

结果汇总覆盖 13 个单种子基础 runs、3 个诊断 checkpoint，以及每 checkpoint 的 7 个主分支与 3 个新增扫描档（各 20 trials）。预算扫描的 G=16/128 引用已有 Uniform/Precise 结果。同步记录暂代教师版本、共享 bank、各域能力和最差域差值、loss 分解与费用；缺失数据留空说明。

当前目录的 `controlled_optimizer_*`、`random_geometry_stress_*` 和解析 teaser 只支撑实现/示意层面的检查；不能填入四域能力、诊断预测或端到端效率结果。本版完成的是实验文档完善，训练器、实测日志、评测输出和实测图表仍待实现与运行。
