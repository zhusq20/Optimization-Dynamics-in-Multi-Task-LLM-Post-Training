# 从三个中心论点到稳定多任务 OPD：章节与实验建议

2026-09-12。基于当前正文、9 月 12 日证据盘点和用户提供的多领域混训材料。本文件是讨论方案，包含可供改稿的英文章节草稿；没有修改主论文，也没有启动新实验。文中的建议、代数推论和待检验假设均不作为已完成实验结果。

## 1. 最值得借鉴的，是把机制发现变成决策规则

建议在第三个中心论点之后、Related Work 之前加入 **Toward a Stable Multi-Task OPD Recipe**。论文目前已经回答了三类局部问题，但读者仍可能问：“这些观察如何影响下一次训练的选择？”新章节应负责回答这个问题。

用户材料最有用的是“分领域目标—诊断—干预—复评”的闭环。对本论文，最好将其具体化为：

| 当前论点 | 对训练决策的启发 | 证据边界 |
|---|---|---|
| Token balancing | 显式选择域权重与响应权重，避免生成长度无意中改写目标 | DR 更透明，不等于 DR 对所有效用函数最优 |
| Update sparsity / teacher overlap | 观察更新位置、方向及跨域影响，诊断后再决定是否采取冲突干预 | 稀疏、重叠、梯度冲突和能力退化是不同概念 |
| Supervision density | 联合检查候选覆盖、反馈尺度、能力收益与成本 | 增大词表覆盖不自动保证更广参数更新或更稳能力 |
| 新增 recipe | 用预设的分领域能力底线评估和选择上述决策 | 当前只有单训练种子，尚不能宣称稳定性得到普遍验证 |

新章节的定位是三个研究问题的综合产出，无需另立第四个算法贡献。若最终没有增加在线控制实验，标题保留 “Toward”，正文称 “evidence-informed starting recipe”。

## 2. 对第一论点的启发：把“配比”拆成三个不同选择

### 2.1 当前结果已经提供了很好的动机

在共同第 50 步，I64 的三个规约为：

| Reduction | Math | Code | IF | GPQA | Mean | Worst Δ from initialization |
|---|---:|---:|---:|---:|---:|---:|
| GT | 74.20 | 17.97 | 17.00 | 32.95 | 35.53 | −1.33 pp |
| DT | 72.20 | 18.75 | 17.00 | 32.32 | 35.07 | −1.33 pp |
| DR | 69.60 | 17.19 | 19.00 | 31.69 | 34.37 | −0.51 pp |

这是“平均收益”和“跨域能力保护”不同的直接例子。GT 平均分更高，而 DR 的最差领域退化更小。当前结果不能支持“平衡必然让所有领域更好”；可以支持“平均分不能替代保留能力的目标”。数据见 `experiments/aligned_evidence_20260910/normalization_table.tex`。

论文还观察到：相同的 25% prompt 配额，在 GT 中给数学平均 44.05% 的 token 权重，给 IF 14.98%。其 OPD 特性值得更明确地写出：

> 当前学生改变回答长度，回答长度改变 GT 的域权重，新的权重又改变下一步学生。因此，固定 prompt 配比仍可能产生随训练变化的监督分配。

这条反馈关系由规约定义可知；“它造成长期不稳定”则仍需在线实验。DT 去除域间的长度权重；DR 进一步去除域内响应的长度权重。已有固定 batch 结果中，DT–DR 的梯度 cosine 在第 250 步为 0.399–0.786，说明后者不是可以忽略的形式区别。

### 2.2 Recipe 应分别声明配额、目标权重和规约

对于 DR，batch 梯度具有形式：

\[
\widehat g=\sum_d\lambda_d\frac{1}{n_d}
\sum_{i=1}^{n_d}\frac{1}{T_{di}}\sum_t g_{dit}.
\]

- `n_d`：每域生成多少 responses，影响数据曝光、估计方差和计算成本。
- `λ_d`：目标中的显式领域权重。
- `1/T_di`：是否让每条 response 等权。

在每域都被分层采样且正常求域均值的 DR 中，固定 `λ_d` 而只增加 `n_d`，不会直接增加该域条件期望梯度的权重。因此，“某域掉分就多采它”需要说明实际改的是配额、权重还是二者。随机只抽部分领域的实现则还需要明确域选择概率及其校正。等 `λ_d` 也不等于梯度范数相等。

建议结论：**当目标是各域内按 response 计效用时，以 DR 作为可解释起点；如果有意按域内 token 加权，DT 是合理选择。优先消除无意的权重，再讨论有意的权重分配。**

Open-MOPD 的 token-share correction 在相同 batch、mask 与目标权重下对应 DT。该论文也已有 gap weighting，因此本稿的差异应放在域内 response weighting、实际更新诊断和监督覆盖之间的连接，不能把一般动态分配重新包装成新意。[Open-MOPD §4.1–4.2](https://arxiv.org/html/2608.19098v1)

## 3. 对第二论点的启发：从“在哪里更新”走到“是否伤害其他目标”

### 3.1 材料中的梯度冲突解释应作为候选原因

当前证据不支持直接把能力跷跷板归因于严重梯度冲突：

- 联合训练第 250 步，PG/I64 的 BF16 累计非零比例为 5.88%/7.13%，但 FP32 为 95.73%/96.07%。可见权重变化集中不等于只有少量参数参与优化。
- 附录 routed raw-gradient cosine 的均值约为 −0.0020/0.0081；这是少量短响应 bank 的结果，既不能证明普遍冲突，也不能证明无冲突。
- Jaccard 看位置，cosine 看方向；两者都不是实际能力损伤的测量。
- 同一 Adam 历史本身会产生更新。已有零当前梯度控制改变约 0.140% BF16 坐标，四种真实局部 loss 约为 0.149–0.150%。因此 teacher step 相似不一定来自当前教师反馈相似。

教师作用于同一批位置，可能意味着共享可迁移机制，也可能意味着相互抵消，需要方向和功能影响进一步区分。独立训练终点的重叠与共同学生状态的逐教师更新也是不同对象。[Dense Supervision, Sparse Updates](https://arxiv.org/html/2606.13657v3)

### 3.2 最有帮助的新测量：跨领域的一步干预

保留原计划的“同一学生、同一 optimizer 状态的逐教师 step overlap”，并为其增加功能解释，而不是用新任务替代原来的问题。

令 `U_s(g)` 是在保存状态 `s` 下、包含真实 clipping 和 Adam 处理的参数更新。对每个教师计算 `u_d=U_s(λ_d g_d)`，另保存 `u_0=U_s(0)`。各教师使用同样的梯度规约和可比曝光。除了其更新支持与方向，测量固定、独立的各域 probe loss：

\[
C_{de}=L_e^{\mathrm{probe}}(\theta+u_d)
-L_e^{\mathrm{probe}}(\theta+u_0).
\]

这里 θ 与 u 必须在相同参数表示下定义；若评估 BF16 写回后的模型，应按真实 master-to-model 写回构建两个分支，不能把 FP32 step 不加说明地直接加在 BF16 起点上。

`C_de>0` 表示在该局部 probe 上，加入教师 d 的当前反馈比仅保留优化器历史更不利于领域 e。还应记录相对未更新学生的绝对 loss change，避免两个分支都变坏而相对比较显示改善。较便宜的一阶诊断可使用 `g_e^T(u_d−u_0)`，但应使用实际参数位移，并保持 loss 与精度定义一致。

这张矩阵是对局部 surrogate 的干预，不能替代长期 benchmark 或证明“教师 d 导致遗忘”。单步变化可能很小，应报告尺度和重复 bank 的一致性。真实混训中的边际作用可以进一步用混合梯度与删去单域梯度的更新比较；由于 Adam 和 clipping 非线性，不将独立教师 step 的加权和当成真实混合 step。

Recipe 的结论应该是：**只有在权重、覆盖和输入组成得到控制后，跨域损害仍持续出现，才值得测试梯度干预或参数隔离。** PCGrad 确实操作冲突梯度的方向；GradNorm 主要调节梯度尺度和任务训练速度，两者不应笼统写成同一种投影方法。[PCGrad](https://arxiv.org/abs/2001.06782)、[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)

LoRA 本身是低秩参数化，并不自动提供任务之间的隔离；共享一个 LoRA 仍然共享更新参数。独立 adapter、路由或冻结策略需要自己的功能实验，本稿暂无证据把它们列为必需组件。

## 4. 对第三论点的启发：监督密度也是一种隐式分配机制

### 4.1 一个可以从现有公式推导的新连接

当前 I64 在学生 top-64 集合 `S` 上归一化，然后 mask 到师生交集 `I`，没有再归一化到 `I`。令：

\[
r(h)=\frac{p_\theta(I\mid h)}{p_\theta(S\mid h)}.
\]

对非空交集，在固定 prefix、候选集合和 detached 系数的条件下，定义 `g_I-renorm` 为“同样 log-ratio surrogate，但学生权重改成在交集 I 内归一化”的梯度，则：

\[
g_{\mathrm{I64}}(h)=r(h)g_{\mathrm{I\text{-}renorm}}(h).
\]

因为对 `v∈I`，`p(v)/p(S)=[p(I)/p(S)]·[p(v)/p(I)]`，且这些系数均停止梯度，所以该关系是条件于固定计算图的代数恒等式。不能把它扩展成未停止梯度的目标等价；交集为空时，当前 I64 梯度为零，而重新归一化版本未定义。

这说明：**DR 消除了长度导致的响应权重差异，但 I64 仍会通过交集保留质量改变不同 prefix 的监督尺度。** 教师 log-ratio 与模型 Jacobian 还会继续影响梯度大小和方向。因此，长度、候选覆盖和梯度贡献不能用同一组比例概括。

这个连接最适合放在第三节讨论末尾，并在 recipe 表中与第一论点对应。它是新的解释，不是已证明的退化原因。

### 4.2 应补什么，避免误读现有结论

当前 teacher-top64 的 student mass 均值为 99.918%，这是 `p(T)`，不是 `r=p(I)/p(S)`。它说明局部实验处于高覆盖场景，但高概率质量覆盖不自动保证梯度尾项很小：被省略项还含 log-ratio 和 score gradient。

低成本补充可按域报告 `p(S)`、`p(T)`、`p(I)`、`r`、空交集率及实际候选数的分布，尤其看低分位而不只看均值。然后在相同 prefixes 上比较原 I64、交集重归一化版本以及已有 full 参考的 raw-gradient 范数、方向和保存状态 step。范数匹配只作为诊断，用来区分整体尺度与方向差异；不能借此宣称在线能力差距来自某一单独因素。

不默认把交集重归一化当作修复：它也可能放大小交集中的不可靠监督。若覆盖日志已经充分且 r 始终接近 1，这个机制在当前设置中可能影响有限，应如实报告。

PG 在固定 prefix 上给出 full reverse-KL 的无偏梯度估计；I64 改变了词表支持、归一化和估计器。因此现有在线 PG/I64 比较是两种完整监督方式的比较，不能严格命名为“只增加密度”的因果实验。

## 5. 建议写入新章节的候选 recipe

### 5.1 把“稳定”定义为可检查的条件

令 `s_d(θ)` 为方向统一为越高越好的领域指标，并在训练前设置允许退化 `ε_d`。约束为：

\[
s_d(\theta)\ge s_d(\theta_0)-\epsilon_d,\qquad \forall d.
\]

这适合作为开发集上的选择约束，而不是将 benchmark 得分直接放进可反向传播的训练 loss。可以在满足约束的候选 checkpoint 中，选择事先定义的效用最高者。若没有新 checkpoint 合格，保留初始参考并报告没有找到可接受的改进。

需要区分三种性质：

1. **终点保留**：选定模型是否在所有领域通过底线。
2. **过程稳定**：在预定检查点上的最大退化、低于底线的频率或持续区间。少量评估点不能证明两点之间没有退化。
3. **重复运行可靠性**：不同训练种子是否得到相近结果、是否重复过线。

初始学生是能力保留基线，专门教师是学习目标参考，匹配曝光的单教师学生是隔离负迁移的参照。这三种参考不能互相替代；教师分数也不是学生不可超过的理论上界。

### 5.2 默认流程及需要证据才启用的干预

| 步骤 | 可执行默认规则 | 对应论点 |
|---|---|---|
| 训练前定义目标 | 冻结域标签、教师路由、数据版本、初始分数、允许退化和独立开发/测试划分 | 多目标评估 |
| 控制监督权重 | response-level 目标从 DR、`λ_d=1/4`、每域 16 responses 起步；显式记录配额与规约 | 第一论点 |
| 保持 on-policy 语义 | 当前学生重新生成 responses，再由对应教师评分；64 responses 累积完后做一次更新，沿用当前协议 | OPD 特性 |
| 选择监督方式 | 把当前 I64 作为候选，PG 为对照；记录覆盖、反馈尺度、生成和教师评分成本 | 第三论点 |
| 监测并定位 | 固定节奏评估各域；出现持续下降时，依次检查权重、长度/截断、覆盖，再检查逐教师实际更新与跨域影响 | 三个论点 |
| 验收和选模型 | 在开发集过线的 checkpoint 中按预定效用选择，最后用独立测试集一次性评估 | 稳定性闭环 |

当前学习率、Adam 参数和 cap 应作为复现实例完整报告，不宣称这些数值跨模型通用。机制比较按同一更新数和每域 prompt 曝光；计算效率另报实际 token 和时间，不将它们视为同时天然匹配。

动态 λ、动态配额、参考模型 KL、PCGrad 和任务隔离不是上述最小 recipe 的必需组件。若决定新增在线控制，只选一个机制，事先定义触发窗口、调整幅度和最小域曝光；使用开发集而非最终 benchmark 调度。连续下降且超出预设噪声范围才触发，不能承诺固定次数检查即可保证稳定。

材料中的 replay 应改成 **保留旧领域 prompts，由当前学生重新 rollout**。直接复用旧学生 responses 会改变 on-policy 分布，需要另行说明校正和目标。OPD 本身已包含对专门教师的 KL/其 surrogate；若额外锚定初始模型，这是不同参考分布上的保留约束，可能妨碍新能力学习，不能机械地再加一项 KL。

### 5.3 目前 recipe 候选的直接证据

共同第 250 步，M-I64-DR 相对初始学生的点估计变化：

| Domain | Initial | M-I64-DR / 250 | Δ（pp） |
|---|---:|---:|---:|
| Math | 68.80 | 74.80 | +6.00 |
| Code | 16.40625 | 17.96875 | +1.5625 |
| IF | 18.33333 | 26.00 | +7.66667 |
| GPQA | 32.19697 | 33.96465 | +1.76768 |

M-PG/250 的 GPQA 为 30.17677%，相对初始约 −2.02 pp。I64-DR/250 是一个四域点估计均改善的候选终点，但其第 50 步 GPQA 为 31.69%，低于初始的 32.20%。这说明“终点共同改善”和“全程不退化”不能混写。

PG/I64 的已有 endpoint 配对区间条件于同一个训练种子，不能拿题目 bootstrap 宣称跨种子稳定；多个域同时过线还应使用适当的联合不确定性控制。

## 6. 最小且有辨识力的新增实验

### P0：先完成原三问的必要缺口

- 补齐 GT/DT/DR 在共同较晚 checkpoint 的完整四域能力。优先复用存在且协议相同的 checkpoint；不存在的才续训，不能拿历史 Base 或 Student64→I64 结果填入 aligned 比较。
- 补齐单/多教师的同协议参数几何。当前 single 每步 64 个数学 responses、joint 每步 16 个；同一步数不等于同数学曝光，结论须明确比较预算。归因于混训的能力差距需要匹配每域曝光的参照。
- 补齐共同学生与 Adam 状态下逐教师 optimizer-step overlap，及同状态同输入 bank 的 JS 配对。raw-gradient overlap 不能填充这些缺失。

这些是原论文成立所需的对照，不应因为新增 recipe 而被另一套宏大实验替代。

### P1：验证最小 recipe 的 2×2 交叉设计

| | PG | I64 |
|---|---|---|
| GT | **新增 M-PG-GT** | 已有 M-I64-GT，需补齐共同较晚终点 |
| DR | 已有 M-PG | 已有 M-I64-DR |

I64-DT 继续作为机制桥接，区分域间校正和域内响应加权，不必立刻扩成完整 3×2。

该 2×2 可以估计：改变规约的作用、改变监督方式的作用，以及二者是否存在交互。只做 `GT+PG → DR+PG → DR+I64` 的阶梯消融不足以识别交互。每个领域可报告以下差中差：

\[
\mathcal I_d=[s_d(\mathrm{DR,I64})-s_d(\mathrm{GT,I64})]
-[s_d(\mathrm{DR,PG})-s_d(\mathrm{GT,PG})].
\]

建议先将四臂统一到 250 updates，在 0/50/100/250 评测；稳定性统计采用共同的检查网格。随后做至少三个配对训练种子，报告每个种子而不只报均值；三个种子是最低限度的重复性检查，不是充分的统计功效保证。同一 seed 内共享初始化、prompt schedule、教师、cap、optimizer 和评测协议，学生 rollout 内容随训练自然分叉。

主表保留四域分数、mean、worst-domain Δ；过程图显示各域相对初始/底线的轨迹。补充报告观测检查点中的最大退化、越线情况、跨种子过线率，以及生成 token、教师评分开销与实际 GPU 时间。若要作等计算效率声明，应按可比计算预算另行比较。

若预算只够增加少量工作，优先新增 PG-GT 并补齐现有四臂共同终点；这阶段只能验证交互的单种子迹象，不能把稳定性升级为强结论。

### P2：在现有局部实验上补机制连接

1. 逐教师真实步骤测量完成时，同步增加固定独立 probe 上的跨域 loss change 和零当前梯度参照。这可以检验重叠是否伴随局部损害。
2. 在已有 100/250 的 prefix bank 或保存产物可支持的范围内增加交集保留质量统计；必要时做小型同 prefix 的 I64 归一化/尺度诊断。不能用 teacher-top64 覆盖率代替交集质量。

局部 bank 重复用于测量不确定性，训练 seed 重复用于算法结果的不确定性，两者不可混同。优先复用已有实验的保存状态，不扩展成新的 optimizer 扫描、掩码训练或动作数研究。

### P3：只有固定 recipe 仍反复失败时才增加自适应控制

若要把“诊断闭环”升级为实际在线控制器，应加入 `DR+I64+controller`，并与相同累计领域曝光的预设/回放调度比较，区分改善来自总配比还是反馈时机。以一项明确干预开始；PCGrad、LoRA、MoE 同时加入会使三个论点的作用难以解释。

不建议将线上 A/B、安全拒答、多语言或部署路由纳入本稿的必需实验：当前实证范围是 Math、Code、IF、Science，外部材料提到的其他能力没有被本论文验证。

## 7. 新章节的英文草稿

下面草稿可放在第三个中心论点之后。它只使用当前证据，不暗示自适应控制或新消融已经完成。正式插入时加入对应 section/table 引用；实验完成后再更新最后两段的措辞。

### Toward a Stable Multi-Task OPD Recipe

Our three studies inform complementary decisions in multi-task OPD: how to allocate supervision across domains and responses, how to assess interactions between teachers in a shared student, and how to choose vocabulary feedback. Together, they suggest a training recipe that makes these decisions explicit and evaluates them against per-domain retention requirements.

**Specify the intended weighting before changing the mixture.** Prompt quotas, domain loss weights, and within-domain response weights are distinct choices. Global-token averaging allows student-generated lengths to determine domain weights. Domain-token averaging fixes those weights, while domain-response averaging also removes length weighting within each domain. When the intended objective assigns equal importance to responses within a domain, DR provides a transparent starting point. It is not uniformly superior: at update 50, GT achieves the higher mean score, whereas DR has the smaller worst-domain decline. The reduction should therefore follow the intended capability tradeoff rather than be selected from aggregate loss alone.

**Audit feedback coverage together with its scale.** Broader vocabulary feedback and broader parameter changes are different properties. At the shared state studied here, PG, I64, T64, and full-vocabulary supervision produce similar update concentration, under high top-64 probability coverage and a common Adam history. Their online capability profiles nevertheless differ. Moreover, I64 normalizes student weights over the student candidate set before restricting them to the teacher intersection. The retained weight mass can consequently vary across prefixes. Monitoring candidate coverage and feedback magnitude alongside capability helps identify effects that vocabulary size alone does not describe.

**Diagnose teacher interactions before introducing isolation.** Concentrated BF16 checkpoint changes do not establish independent task modules. Coordinate overlap describes where changes occur, while signed directions and cross-domain effects address whether they are compatible. Teacher-specific steps must be compared from a common student and optimizer state, because optimizer history can move parameters even without a current gradient. These distinctions motivate state-controlled diagnostics when a domain deteriorates; the present results do not establish parameter isolation or gradient projection as necessary components of MOPD.

**Use per-domain retention criteria to select checkpoints.** Before training, define a reference score and an acceptable decrease for each domain. Evaluate candidate checkpoints on a separate development set and select among those satisfying these requirements using a prespecified utility. Preserve domain prompt coverage and generate fresh responses with the current student. Reusing historical responses would require additional treatment of the resulting distribution mismatch. Any adaptive changes to domain weights or sampling should be stated separately from the averaging rule and evaluated on held-out data.

Under our current protocol, DR with I64 is a candidate starting configuration: at update 250, its point estimates exceed initialization in all four domains. This endpoint observation does not establish monotonic retention or robustness across training runs. Earlier checkpoints exhibit domain-specific declines, and our experiments use one training seed. A recipe-level evaluation should therefore compare matched training budgets, domain trajectories, and repeated seeds. The proposed recipe is an operational synthesis of the three studies, with its stability assessed through capability retention rather than inferred from loss reduction or parameter sparsity.

## 8. 改稿时的几个具体落点

- 引言：用已有 GT/DR “mean 与 worst Δ 排序不同”引出能力分配与保留；不把严重梯度冲突预设为本文结论。
- 第一节末：加一小段 implication，说明规约先于 sampler 调节，区分 `n_d` 与 `λ_d`。
- 第二节末：保持 sparsity → overlap → JS 的原问题，补充位置、方向与实际损害的区别；用跨域局部测量解释为何需要教师 step，而不是改写成优化器论文。
- 第三节末：加入交集保留质量的条件代数关系，把候选覆盖与第一节隐式权重相连。没有新测量前称可审计机制，不称已经发现的主要瓶颈。
- 第三节后：新增约 1–1.5 页 recipe，包括一张“发现—决策—触发条件”表和可复现的默认流程。正文只保留当前证据支持的建议。
- Discussion：缩短重复总结，集中交代单模型、单 seed、高覆盖局部样本的适用范围，以及何种实验才足以把 candidate recipe 升级为稳定性结论。
- 顺带统一术语：当前摘要写的是 “FP16 model weights”，而正文与记录为 BF16，应改为 BF16。该笔误不影响本方案中的数值，但应在下一次正式改稿中修正。

## 核读依据

本地：`sections/001_abstract.tex`、`002_introduction.tex`、`004_preliminary.tex`、`005_normalization.tex`、`006_teacher_subnetworks.tex`、`007_supervision_density.tex`、`007_discussion.tex`、`005_experiment.tex`、`009_appendix.tex`，以及 `experiments/RESULTS_AUDIT_20260912_zh.md`、当前 aligned 能力表和机制表。9 月 11 日盘点只用于历史上下文，当前结果以 9 月 12 日盘点及正文为准。

外部原始来源：[Open-MOPD](https://arxiv.org/html/2608.19098v1)、[Dense Supervision, Sparse Updates](https://arxiv.org/html/2606.13657v3)、[PCGrad](https://arxiv.org/abs/2001.06782)、[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)。用户粘贴的材料作为工程讨论启发，不作为这些原始论文的替代证据。
