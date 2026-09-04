# MOPD 联合进展与计算分配：四域实验计划

更新：2026-09-04。按第一性原理重构备忘录修订。当前为实验规格；端到端训练器和四域实测结果尚未完成。以下新增单任务参照、共同 checkpoint 干预和独立预测验证，扩大了此前九配置方案的实验范围。

## 1. 要回答的问题

当一个学生能分别从各教师获得能力时，教师在实际 top-64 损失中提供的自身更新信号有多强，联合更新是否减弱它？即使不存在负向均值相互作用，可靠的下降也可能很慢。有限样本是否会使本来有用的更新损害某个域？增加样本或改变分配能否改善下降可靠性、下降幅度、实际能力或计算效率？

实验先确认单任务可迁移性，再用同一 checkpoint 的干预区分自身信号弱、合并后进展减少、估计不准确与可靠但缓慢的局部下降，之后检验预先设定的预测，最后报告能力和计算成本。单任务可学不证明各域参照分数能由一个模型同时达到；局部冲突也不证明联合目标不可达。

GPAS 是降低总更新方差的一种干预。任务权重和预条件尺度固定时，它不能改变均值更新的一阶符号，也不保证最小化最脆弱任务的方向误差。先检验其 trace noise 是否是有用的代理；只有代理失效的实测证据支持时，才做方向分配的离线诊断，不预先增加一种必须成功的新调度器。

top-k、Neyman allocation 和概率不等式均为已有组件。拟验证的贡献是：进展余量和方向不确定性是否能预测 MOPD 何时受益于精确采样，以及何时需要其他干预。尚未取得这些结果。

## 2. 模型、教师与数据

| 项目 | 配置 |
|---|---|
| 学生 | Qwen3-1.7B，non-thinking |
| 数学教师 | 数学 RL 后的 Qwen3-1.7B checkpoint |
| IF 教师 | IF RL 后的 Qwen3-1.7B checkpoint |
| 代码、科学教师 | 同一个冻结的 Qwen3-4B，两个任务路由共享权重 |
| 任务权重 | 四域始终各 1/4；配方比较中的动态权重单独注明 |
| 训练 prompts | 每域 16,000 条；联合训练打乱后顺序消费、不重复；单任务参照耗尽后重新打乱进入第二遍 |
| 每个 micro-batch | 同任务 4 个 prompts，各采样 1 个 response |
| 每步 | 16 个 micro-batches，共 64 个 responses |
| 任务计数 | 2 ≤ m_i ≤ 8，且四域计数之和为 16 |
| 预算 | 每 run 500 步，32,000 个 attempted responses |
| 回复长度 | 上限 4,096；达到上限的回复保留有效 token |
| 更新 | 每批新 rollouts 只更新一次，K=1 |
| 种子 | Uniform、GPAS (per step) 各 3 个配对训练 seeds；其余 7 配置各 1 个探索性 seed；4 个单任务参照各 1 个初始 seed |
| 资源 | 每 run 一张 96GB 训练卡与一张 48GB rollout/teacher 卡 |

表中计数范围适用于联合训练。单任务参照使用一个任务、权重 1、每步 16 个 micro-batches，其余基础配置一致；局部诊断的预算另列。

这是一组四域、三个不同权重集的教师配置，写作中称 four-domain study，不称四个独立 specialists。教师与初始学生都用本项目的统一评测器测分。

待补入配置文件的内容：精确 checkpoint/revision、模型 tokenizer、训练数据名称与版本、过滤和 held-out 切分、采样温度、EOS 处理、AdamW 参数、精度与显存配置、GPU 型号、训练/推理软件版本。初始 teacher gap 是描述训练状态的量，不决定任务权重或实验是否进行。

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

在已有 top-64 输出上记录实际损失对应的描述性教师信号。对每个固定 prefix，令 z(a)=1[a in S] log(q[a]/p[a])，按完整学生分布 p 计算：

    student_retained_mass = sum_{a in S} p[a]
    teacher_retained_mass = sum_{a in S} q[a]
    mean_z = sum_{a in S} p[a] * log(q[a]/p[a])
    conditional_var_z = sum_{a in S} p[a] * log(q[a]/p[a])**2 - mean_z**2

conditional_var_z 是 E_p[(z-E_p[z])² | prefix]；并非把 p 限制到 S 后重新归一化的方差。按原 loss 的 response 内位置平均、response 间平均记录它和 retained masses。只需要教师已经返回的 top-64 全词表归一化概率，以及学生已有概率；不额外请求教师全词表 logits，不新增大型梯度 buffer。该标量是 Cauchy–Schwarz 信号上界的一个因子，不是参数梯度协方差，不能单独证明梯度有用或教师质量；教师信号经 score features 的映射仍可能很弱。

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

## 5. 单任务参照与九个联合训练配置

先对四域分别运行单任务 OPD，从同一初始学生开始，使用对应教师和默认 loss。每个参照运行 500 步、每步 64 个 responses，总计 32,000 次 attempted responses；16,000 条 prompt 池在耗尽后重新打乱，第二次访问仍重新生成 response。记录 unique prompts、总曝光量和重复次数。每域曲线与联合训练在匹配该域曝光量处比较，同时给出总 GPU hours；不要把 32,000 次单域曝光与 Uniform 的 8,000 次单域曝光直接解释为集成损失。初始单任务参照各 1 seed，只建立观察到的可迁移性，不证明其跨训练种子稳定性。

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

配置 1、2 各运行 3 个配对 seeds，是重复验证的主要比较；配置 3–9 各 1 个 seed，只作探索性比较。共 13 个联合训练 runs，另有 4 个单任务参照。

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

## 6. 共同 checkpoint 诊断与预测验证

### 数据、状态与分支

从 Uniform 的第 50、250、450 步保存完整状态。每个诊断分支加载相同的学生、AdamW 一阶和二阶矩、更新步数、学习率状态、随机状态、prompt 位置和统计量，再进行一次更新。诊断不写回主训练轨迹。所有诊断 rollout 都由分支更新前的 checkpoint 策略生成。

每个 checkpoint 从各域同一个预先指定的诊断 prompt 分布独立抽取以下三组样本。诊断分布与训练和 benchmark 数据隔离；组间不复用抽样记录或 response，但有放回独立抽样可能选到相同 prompt，不强制 prompt 身份互斥：

1. Calibration：每任务 32 个独立 micro-batches，分成两个各 16 个的独立半组 A/B，用于交叉拟合均值乘积、估计进展余量、trace noise 和方向方差，并预先选定诊断 GPAS 计数。
2. Update trials：每个分支进行 20 次独立更新试验，每次重置到相同完整状态，再指定独立的试验抽样种子；抽样记录与 calibration 分离。
3. Evaluation：每域 128 个独立抽样的 held-out prompts，由当前 checkpoint 生成参考 prefixes，随后固定；所有分支均在这些相同状态上测前后 loss，并用评估梯度估计一阶投影。这些样本不参与 calibration 或计数选择。

诊断的三种用途在每域共享同一个 prompt 分布并有放回独立采样；其中的 mu_i 指该诊断分布下的均值，向主训练分布的适用性另行检验。主训练使用有限池无放回流。记录这一区别，不能直接把独立样本公式称为主训练流的精确协方差。不要用同一批噪声同时选择分配并评价选择效果。

每次试验比较：

| 分支 | 单次更新样本 | 用途 |
|---|---|---|
| 四个单任务分支 | 各自使用该任务 16 个 micro-batches，权重 1 | 估计单域更新对全部域的实际影响 |
| Uniform joint | 每域 4 个，共 G=16 | 普通联合更新 |
| GPAS joint | G=16，m_i 在 [2,8]，由 calibration 选择 | 检验 allocation 干预 |
| Precise joint | 每域 32 个，共 G=128，仍固定等权 | 更精确均值的昂贵诊断；不作等成本对手 |

另做独立的等分配预算扫描，每域 m_i 依次取 2、4、8、16、32，总 G 为 8、16、32、64、128。局部扫描不使用主训练的 m_i≤8 上限。总预算与分配方式分开改变；一次更新的收益不能直接推断固定总计算量下改变更新频率的训练收益。全部诊断和评测计算另计。

### 记录的诊断量

对 checkpoint 的 pre-step D，计算 K_ij=mu_i^T D mu_j 与 a_i=(Kw)_i。具体使用 K_hat_ij=(mu_hat_i,A^T D mu_hat_j,B + mu_hat_i,B^T D mu_hat_j,A)/2，而不直接平方同一个带噪均值；尤其对角项由独立均值相乘，避免正偏差误判进展。K 中只有一个 D，不能用 D mu_i 和 D mu_j 的余弦替代任务的一阶进展。用 calibration 估计 v_i(m)=sum_j w_j² mu_i^T D Sigma_j D mu_i/m_j、总 trace 方差以及 a_i>0 时的 v_i/a_i²，方向均值与协方差在独立半组之间交叉估计，并显式报告残余估计不确定性。

另外报告交叉拟合的自身进展 K_ii、加权自身项 w_i K_ii、合并余量 a_i 以及交叉任务贡献 a_i-w_i K_ii。并列比较现有单任务分支和更精确联合分支的自身域 loss 变化，所有量保留不确定性。这样区分自身信号已经弱与合并后进展被削弱；单纯 a_i<K_ii 也可能来自任务权重，不能一概称为冲突。沿用 calibration 与 trial 的现有均值统计和逐任务处理，不增加完整训练 runs 或默认同时保存更多大型梯度向量。

对每次 trial，使用独立 evaluation bank 的梯度估计冻结 D 下的投影，再记录实际 AdamW 一步后的各域固定状态 loss 变化及该更新耗时。Evaluation 梯度不参与 calibration 或计数选择；不同 trial 共用 bank 的相关性在置信分析中保留。梯度 clipping、momentum、当步矩更新、weight decay 和有限步长均可造成实际更新与冻结 D 模型不同，必须报告这种偏差。Calibration 的估计误差需要单独量化；把估计值代入 Cantelli 上界不构成有效置信证书。

仅当独立验证的 a_i 估计区间完全小于零时，将该域判为当前加权均值方向不利。即使冻结 D 的余量为正，更精确的实际 AdamW 更新仍可能因 momentum、曲率或其他有限步长效应使 loss 上升；这种结果是模型近似偏差，不自动证明均值冲突。若余量为正但普通投影常反号，GPAS 和增大预算有可检验的目标。余量区间跨过零时标记不确定，不强行归为任一种机制。局部各域同时下降不是长期联合训练成功的必要条件。

增加“可靠但缓慢的局部进展”分类：在独立 trials 中下降具有足够可靠性，但绝对 loss decrease 和相对 pre-update loss 的 decrease 仍低于预先规定阈值。相对指标仅在 pre-update loss 高于指定正下限时计算；它不等于相对未知最优值的收敛率。阈值与置信程序一并预先固定，近零或不确定的变化保留“不确定”标记。比较单任务与更精确联合更新的下降幅度，检验提高精度是否只减少反号而没有消除慢进展。一个 checkpoint 的小下降不能识别曲率、模型容量或长期速率。本轮不承诺测真实 Hessian 或完整收敛；解析指数衰减例子仅证明无冲突仍可慢的可能性。

### 预先设定预测

仅用第一训练 seed 的第 50 步诊断确定规则、回归和慢进展阈值、相对变化分母下限、置信分析和指标，记录后再查看第 250、450 步及其余两个训练 seeds 的诊断结果。预测目标包括：哪些域更易回归，哪些域下降可靠但幅度小，自身进展还是任务合并限制了局部下降，以及 GPAS 或精确估计对反号频率和下降幅度各有多少作用。比较自身项、合并余量大小/符号和方向方差与单独 trace noise 的预测，报告错误和无效干预。验证集不用于重新选择阈值。

若 trace 代理明显失效，可在 calibration 上枚举合法整数计数，离线最小化正余量域的最大 v_i/a_i²，并在独立 trials 中评价。不得悄悄排除负余量域后声称保护了全部任务，也不把离线结果直接写成新的端到端方法胜出。

## 7. 能力、状态分布与计算报告

### 三类评测分别报告

主能力表保持紧凑，报告初始学生、指定教师、四个单任务参照，以及九个联合配置的 MATH-500 greedy pass@1、固定 LiveCodeBench 切片 pass@1、IFBench strict accuracy、GPQA-Diamond average@4 和四项原始百分比分数均值。单任务参照一行来自四个独立学生，不能写成一个共享学生的能力。

benchmark 版本、代码题日期范围、解码和评测随机种子写入共享配置。主表统一用第 500 步，曲线展示中间 checkpoint，避免对各域分别挑最高点。单任务归一化收益仅在参照增益高于预先指定的正分母下限时报告；下限与能力容差 epsilon_i 均在看结果前写入配置，否则只报告原始增益。始终同时给出原始分数。

第 0、50、100、……、500 步在每域 128 个固定 held-out prompts 上测量以下不同指标：

- 固定参考状态 loss：初始学生生成一次 reference prefixes 并固定，所有 checkpoint 在同一 bank 上计算标准 teacher top-64 loss。
- Fresh-policy loss：当前学生在相同 held-out prompt 集上生成新回复，以共享采样配置计算同一 loss。
- 下游能力：使用上述 benchmark 协议，不以 loss 下降代替能力增益。

每 50 步保存该 checkpoint 的 fresh reference prefixes 和 teacher scores。在下一评测点 s=t+50，额外在旧 bank 上测 L_i^t(theta_s)，从而分别估计：旧状态优化项 L_i^t(theta_s)-L_i^t(theta_t)，以及状态分布变化项 L_i^s(theta_s)-L_i^t(theta_s)。这里的 L 是 held-out prompt 分布上的经验评估损失，对应正文分解在评估分布上的版本，不等同于剩余训练池的损失。二者在相同经验 bank 定义下相加为 fresh loss 的 checkpoint 间变化；采样误差单独报告。已保存旧 prefixes/teacher scores，因此额外工作主要是学生评估前向，费用计入评测成本。

共同 checkpoint 干预另有该 checkpoint 生成的独立 reference bank，分支之间保持固定。长期 initial bank、相邻 checkpoint 分解和局部诊断 bank 分别标注；initial 与 fresh loss 两条曲线的差本身不构成上述时间变化分解。

Uniform 与 GPAS 报告每个 seed 结果、三 seed 均值及离散程度。配对 prompt bootstrap 仅描述评测样本不确定性，单独列出，不替代跨训练 seed 差异。其他方法和单任务参照明确标注为单 seed。

### 成本与日志

默认损失、固定权重组的共同 loss target 为三个 Uniform runs 最终固定参考 loss 的均值。首次到达时在相邻 checkpoint 间插值；未到达写 unreached。所有方法报告实际总 GPU hours，以及分别按每域曝光、steps、GPU hours 绘制的 loss 与能力曲线。最差域相对 Uniform 的分数变化与均值一起报告，避免平均分隐去域退化。

GPU hours 包括两张保留设备上的等待、rollout、teacher scoring、backward、统计量、同步和 checkpoint 时间。额外诊断和 benchmark 评测单列，不能隐去昂贵 precise branches。

正常训练日志保留每域 loss/velocity、preconditioned 与原始 trace noise 及 EMA、m_i、prompt 曝光/唯一数、有效 token、回复长度/截断率、各阶段耗时和峰值显存。共同 checkpoint 追加 K_ii、w_i K_ii、a_i、方向方差、绝对/相对局部 loss decrease、估计误差和实际干预结果；常规 teacher top-64 记录保留概率质量和 centered masked-logratio conditional variance。Cost-aware 结果报告拟合和实测时间及统计开销；TA-OPD 配对仍作为探索性 loss 稳健性比较。

### 稿件结果顺序

1. 单任务可迁移性与联合能力：匹配域曝光和总计算量，报告不确定性。
2. 共同 checkpoint 干预：自身信号与合并进展、可靠但慢的下降、采样失败、精确估计和预算扫描；描述性教师信号单独标注。
3. 独立预测验证：后续 checkpoint 和 held-out seeds 的预测准确性与失败情况。
4. 端到端能力与计算：三 seed 核心比较、最差域变化、两个状态 loss 和成本。
5. 分配轨迹、cost-aware 与 TA-OPD：解释范围及单 seed 限制。

首图使用明确标注的解析示例：e^{-t} 与 e^{-0.1t} 展示无冲突仍可有不同速率，Gaussian 例子展示均值和采样可靠性的区别。它不报告真实 MOPD 速率，也不替代单任务/联合能力实测图。GPAS 方法示意图保留在附录。所有实测占位段落和图表仅写待测量内容，不填入改善方向、虚构数据或成功结论。

## 8. 工程安排与预算

先完成默认 loss、固定任务均值聚合、完整 checkpoint 恢复和基本日志。短运行检查实现后，优先运行第一 seed 的 Uniform 及四个单任务参照，并完成第 50 步诊断。锁定预测规则后开展后续诊断及三 seed Uniform/GPAS 核心比较，再完成配置 3–9 的探索性运行。保留已有最多 8 个双卡训练 slots 的资源布局，空闲评测卡处理共享评测；精确诊断资源须另行记录，不沿用九 run 旧预算。

13 个联合 runs 加 4 个单任务 runs，共 17 个基础 runs。按此前每 run 8–10 小时、每 run 两张保留 GPU 的排程假设，基础训练约 272–340 GPU hours，诊断和评测另计；单任务成本可能不同，最终以实测为准。失败 run 修复后重跑并保留配置修订记录。

每 50 步保存模型用于能力/曝光曲线；第 50、250、450 步必须保留完整可复现状态，包括 AdamW moments、step counter、LR scheduler、随机状态、prompt 位置、EMA 和 allocation；另保留最新完整恢复点。单任务 500 步的第二遍曝光显式记录。诊断读取 checkpoint 后另存结果，不修改源 run。

本次修订只定义实验与论文结构；训练器、实测日志、评测输出和图表源数据仍待实现及填入。

## 9. 本次采用的文献与借鉴

- [MOPD](https://arxiv.org/abs/2606.30406)：使用已有 teacher top-k corrected loss。
- [Open-MOPD](https://arxiv.org/abs/2608.19098)：借鉴单任务参照与 integration-gap 诊断，同时准确区分 loss weighting 与 sampling。
- [D³-MOPD](https://arxiv.org/abs/2608.24987)：使用完整 gap × velocity 比较，展示随训练变化的预算。
- [TA-OPD](https://arxiv.org/abs/2608.14728)：用现有 tail-aware loss 展示调度复用。
- [Instella-MoE](https://arxiv.org/abs/2609.00791)：借鉴 specialist 与已有能力保留教师共用的工程动机。
- [CaMOPD](https://arxiv.org/abs/2605.27115)：已有更新冲突分析意味着仅画梯度冲突图不足以构成新贡献。
- [Rethinking OPD II](https://arxiv.org/abs/2609.04172)：固定状态下的慢对齐支持研究优化障碍，不证明梯度噪声是原因。
- [EMA-PG](https://arxiv.org/abs/2602.04417) 与 [vOPD](https://arxiv.org/abs/2605.07865)：承认 token estimator 与方差控制已有工作，避免把 top-k 当新贡献。
