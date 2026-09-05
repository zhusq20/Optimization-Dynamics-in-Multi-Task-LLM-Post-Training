**MOPD 的 top-k 定义、相关工作与 GPAS 修订建议**

检索日期：2026-09-04。审阅对象：用户提供的 `Optimization_Dynamics_in_Multi_Task_LLM_Post_Training (5).pdf`，重点为 Section 2、Section 3、Section 6，以及对应 LaTeX 和实验计划。本文是修订备忘录；没有改动论文正文或实验配置。文献结果均为作者报告，未在本项目中复现。下文明确区分文献事实、数学核查和待验证的研究判断。

**1. 建议先作出的决定**

保留 GPAS 的“给定任务目标下，按优化器坐标中的梯度噪声分配 micro-batch”主线，把 top-k 从方法贡献降为已有训练组件。主实验必须使用已有的 dense top-k 损失，证明 GPAS 对该损失仍有独立收益。现稿的自定义 head-plus-sampled-tail 估计量可移到附录，并补充相关工作；不能以“替代 sampled-token PG”作为主要创新。

这个决定有三条依据：

- MOPD 已明确介绍 teacher top-k 蒸馏，而 Open-MOPD 的公开代码对 student top-k 候选逐项求梯度。因此，“已有 MOPD 只利用一个 token”不成立。
- EMA-PG 已提出 top-k 精确项加 masked sampled-tail 的构造；TA-OPD 进一步研究尾部质量及 sampled correction。现稿估计量与这条已有路线高度重叠。
- GPAS 的分配公式并不依赖自定义 token estimator。只要定义清楚实际 micro-batch 梯度及其条件协方差，标准 top-k、sampled PG、TA-OPD 均可接入。

**2. Section 2 的 top-k 到底是什么**

固定一个前缀，记学生概率为 $p_a$，教师概率为 $q_a$，score 为 $s_a=\nabla_\theta\log p_a$，$r_a=\log(p_a/q_a)s_a$。以下所有梯度都冻结前缀、教师与 support membership；不能自动解释为对整条 on-policy 轨迹分布求完整导数。

| 实现 | support 和计算方式 | 在固定前缀上的含义 |
|---|---|---|
| Sampled-token PG | $Y\sim p$，使用 $r_Y$ | 对当前 token 的 Monte Carlo 估计 |
| MOPD 的 corrected teacher top-k | $S=\operatorname{TopK}(q)$，直接优化 $\sum_{a\in S}[p_a\log(p_a/q_a)-p_a+q_a]$ | 梯度为 $\sum_{a\in S}p_ar_a$，不补尾部；它是截断 surrogate |
| Open-MOPD 的公开 dense 路径 | student top-k，权重为集合内归一化的学生概率；对每个候选使用独立 log-probability 和 advantage | 在一次 on-policy、无 clipping 生效的更新点，梯度形如 $\sum_{a\in S}\tilde p_a r_a$；不是只乘实际生成 token 的 score |
| 现稿 | student top-k；$\hat h=\sum_{a\in S}p_ar_a+\mathbf1\{Y\notin S\}r_Y$ | head 精确求和，加随机尾部修正；固定前缀下期望等于 full-vocabulary token reverse-KL 梯度 |
| TA-OPD | teacher top-k 加一个表示剩余概率质量的 tail token | 对压缩成 $k+1$ 类的分布计算 reverse KL；其 sampled-corrected 版本再估计尾部内部差异 |

MOPD 的公式见 [Section 3.2.2，Eq. 5](https://arxiv.org/html/2606.30406v1#S3.SS2.SSS2)。其中 $-p_a+q_a$ 恢复了 retained coordinates 的正确最小点，**不等于补回完整词表的无偏梯度**。在 $k=|V|$ 时，附加项之和为零；在截断集合上则不能省略。Normalized subset KL 又是另一种损失，不能混称。

Open-MOPD 的论文 Section 2.1 将 position reward 写成候选项之和，容易产生误读；公开代码提供了更明确的证据。在 revision `4809a96cf85a869106ff0ff3f37d0a51e12010ae`，actor 在三维 advantage 路径使用 `topk_log_probs` 作为 loss 输入，PPO 分别计算各候选项后才在 k 维求和。参见 [actor，L1071-L1084](https://github.com/BytedTsinghua-SIA/Open-MOPD/blob/4809a96cf85a869106ff0ff3f37d0a51e12010ae/training/verl/verl/workers/actor/dp_actor.py#L1071-L1084) 和 [loss，L1205-L1243](https://github.com/BytedTsinghua-SIA/Open-MOPD/blob/4809a96cf85a869106ff0ff3f37d0a51e12010ae/training/verl/verl/trainer/ppo/core_algos.py#L1205-L1243)。这不支持现稿 Related Work 中“top-k 只平滑 scalar reward，sampled score 的随机性仍原样保留”的比较。

因此，Section 6 有两项事实性错误，应优先修正：

1. 把 MOPD 写成 student top-k renormalization；其 Eq. 5 实际是 corrected teacher top-k。
2. 把 Open-MOPD 写成只对 sampled token 求梯度；公开 dense 实现不符合这一描述。

此外，sampled-token estimator 不是“第三种 top-k estimator”，这套分类本身也应重写。

**3. 无偏性成立，但不构成这里的主要创新，也不保证降方差**

现稿的 fixed-prefix 无偏性核查通过：

\[
\mathbb E[\hat h\mid c]
=\sum_{a\in S}p_ar_a+\sum_{a\notin S}p_ar_a
=\nabla_\theta D_{\mathrm{KL}}(p\|q).
\]

这里要求 $Y\sim p$、教师概率在所需 token 上可得、系数的 stop-gradient 与文中定义一致。若实际 rollout 使用 temperature、top-p 或 top-k 截断，必须定义真实采样分布，不能继续不加条件地写 $Y\sim\pi_\theta$。重要性校正需要相应 support；零概率截断不能恢复未覆盖区域的无偏估计。

但新颖性已有直接先例。[EMA-PG，Section 5.3](https://arxiv.org/html/2602.04417v1#S5.SS3) 在 2026-02-04 已使用学生 top-k 加 masked sampled-tail。其正文不同展示式的 stop-gradient 写法需要逐式区分，不能只凭名称断言实现完全相同；不过“head 精确、tail 采样”的核心结构显然已有先例。[TA-OPD Appendix E.5，Eq. 22](https://arxiv.org/html/2608.14728v1#A5.SS5) 展示的 detached head-plus-tail surrogate，在相同 support 下求导就得到现稿的梯度结构；其 Section 6 还给出 sample-corrected TA-OPD。把这部分作为新方法不稳妥。

更重要的是，“精确计算一个子集”不能直接套用 Rao-Blackwell 定理宣布总方差下降。令 $X=\mathbf1_S(Y)r_Y$、$Z=\mathbf1_{S^c}(Y)r_Y$，原估计量为 $X+Z$，现稿为 $\mathbb E X+Z$。二者的差是

\[
\operatorname{Var}_D(X+Z)-\operatorname{Var}_D(\mathbb EX+Z)
=\operatorname{Var}_D(X)-2\langle D\mathbb EX,D\mathbb EZ\rangle,
\]

右侧不必非负。这里 $\operatorname{Var}_D(U)=\mathbb E\|D(U-\mathbb EU)\|^2$。

一个已数值核查的两 token 反例：$p=(0.6,0.4)$、$q=(0.9,0.1)$、$k=1$，参数为第一个 logit，故 score 为 $(0.4,-0.6)$。两种估计量的均值均为 $-0.4300223$，sampled-token 方差为 $0.1076044$，现稿方差为 $0.1660446$。这是数学反例，不是模型实验结果。现稿附录已经避免无条件方差保证，摘要、图注及“analytic removal”的叙事也应保持一致。

另一个限定：fixed-prefix 无偏不能自动提升为使用同一 rollout token、随机停止长度及 $1/|y|$ 权重的整条轨迹估计量之间的无偏。未来 token 与停止长度依赖当前 token。当前实验计划对此已有正确限定，正文应明确保留。

**4. 近期论文分别改变了什么判断**

以下是最直接影响本稿的研究，按用途选取，不是穷尽性目录。

| 论文及日期 | 已核实的做法或发现 | 对本稿的具体影响 |
|---|---|---|
| [MOPD，2026-06-29](https://arxiv.org/abs/2606.30406) | PG 与 corrected teacher top-k 两种实现；同源教师实验中表现接近；展示教师来源影响和异步 teacher-prefill 服务 | top-k 是基线组件；先验证同源教师下的剩余问题；测完整服务链开销 |
| [Open-MOPD，2026-08-19](https://arxiv.org/abs/2608.19098) | 诊断 token-share、gap、reward staleness；share/gap 通过 loss weighting 实现；提供 RouteOPD 和开放训练链 | 不能说所有已有方法都把权重与采样绑死；借用单域 OPD reference 和分机制消融 |
| [D³-MOPD，2026-08-25](https://arxiv.org/abs/2608.24987) | 用归一化 remaining gap × descent velocity、平滑、floor 和 batch jitter 调整采样配比；作者报告达到基线最佳表现的步数为 47 对 143 | 是最直接的调度对手；当前只用 loss-gap 的 baseline 不足以代表它；步数加速不能改写成 GPU-hour 加速 |
| [TA-OPD，2026-08-12](https://arxiv.org/abs/2608.14728) | teacher top-k 加 tail mass；另有无偏 sampled correction；研究 normalization 引发的 tail/entropy 漂移 | 引用并考虑作为稳健性底座；不能把尾部修正包装为本稿的新贡献 |
| [vOPD，2026-05-08](https://arxiv.org/abs/2605.07865) | 在 sampled-token OPD 中减去 detached control-variate baseline，可用 top-k 近似 baseline | 若继续研究 token 估计器，要与已有控制变量方法比较；“mean gap 不影响期望”不表示它不影响采样方差 |
| [CaMOPD，2026-05-26](https://arxiv.org/abs/2605.27115) | 在恢复通用能力和保留领域能力的设定中，诊断更新相互抵消，采用交替训练与 gap-based selection | 在自己的设定实测任务均值梯度的关系；不能把能力损失一概归因于噪声 |
| [Rethinking OPD，2026-04-14](https://arxiv.org/abs/2604.13016) | 强调思考模式兼容、教师新增能力及高概率 token 对齐 | 较大教师未必提供可学习增量；需要单域 OPD 验证 teacher headroom |
| [Rethinking OPD II，2026-09-03](https://arxiv.org/abs/2609.04172) | 特定三域实验中，16 个多样化 query/domain 匹配 full-data MOPD；冻结状态仍需长期优化，吸收速度下降原因开放 | 支持研究优化效率；不能据此断言瓶颈就是梯度方差。可借用固定状态实验隔离数据覆盖与优化 |
| [Instella-MoE Technical Report，2026-09-01](https://arxiv.org/abs/2609.00791) | IF 专家加冻结 DPO anchor，sampled-token log-ratio 蒸馏，独立 teacher endpoints，学习新能力同时保留其他能力 | 提供更明确的工程用例，也证明 sampled PG 仍在使用；初始 anchor gap 近零，暴露 inverse-initial-loss 权重的局限 |

补充阅读：[Mismatch Matters / TIDE，2026-08-10](https://arxiv.org/abs/2608.09836) 讨论 teacher-deficit token 注入及训练退化，提醒我们不能只看 KL 下降；[A Token-Level Analysis，2026-08-26](https://arxiv.org/abs/2608.25643) 已分析 sampled K2 的 logit-gradient norm 因子化。后者的单 token 梯度范数不是本稿的 centered micro-batch covariance，二者可以区分，但“gap 与 score 共同决定梯度大小”本身也不是新的机制结论。

**5. 论文动机应该改成什么**

目前“一个 mixture knob 做两件事”的表述有启发性，但不能作为对所有已有 MOPD 的概括。Open-MOPD 的 token-share balancing 明确改变 loss weights 而保持采样频率，已经有部分解耦；D³-MOPD 主要改变采样配比。参见 [Open-MOPD Section 4.1-4.2](https://arxiv.org/html/2608.19098v1#S4.SS1)。

更准确的研究问题是：

> 给定一套蒸馏损失和预先选定的任务目标权重，在同样训练预算下，如何分配各任务的 rollout / micro-batch 数量，使聚合更新更准确？这种分配在 dense top-k 已经消除了固定前缀的 scoring-token 抽样之后，是否仍能改善实际训练效率？

主线只需要三步：

1. 标准 top-k 使固定前缀的候选梯度确定，但 prompts、rollout prefixes、长度和任务分布仍随机，因此 task micro-batch gradients 仍有异质噪声。
2. 用 $w_i/m_i$ 汇总可以让相同训练状态下的加权期望不随 allocation 改变；此时可以单独优化估计精度。
3. 实际更新受 optimizer preconditioner 影响，因此比较原始噪声、预条件噪声和训练时间，验证是否值得付出统计开销。

建议把“teacher gap 导致 token noise，所以我们发明 top-k 去噪”整条主叙事移出摘要和 Introduction。Gap 可以保留为噪声的候选解释变量，与 entropy、length、tail mass、teacher source 一起测试；不要让论文成败依赖预设的单调关系。

此处的理论合理，但创新门槛也要说清：$m_i\propto w_i\sqrt{e_i}$ 是经典 Neyman allocation，逆概率加权也已有充分研究。贡献应来自 MOPD 上的新诊断、可用的低开销统计方案以及标准工程配方上的稳定收益，不能只靠给经典公式换应用场景。

**6. Section 2 可采用的替换框架**

第一段先定义 shared tokenizer/vocabulary、固定 domain teachers、student rollouts 和实际 loss。建议以 MOPD 的 corrected teacher top-k 为清楚可核对的默认数学定义；若最终选择复用 Open-MOPD，则精确写其 dense surrogate，所有主对比共享该实现。两种选择不能混写成同一个目标。

以 corrected teacher top-k 为例，令第 n 步的 rollout policy 为 $\pi_{\theta_n}$，$c=(x,y_{<t})$，$S_i(c)=\operatorname{TopK}_k(\pi_{T_i}(\cdot|c))$：

\[
\ell_i^{(k)}(\theta;c)=\sum_{a\in S_i(c)}
\left[p_\theta(a|c)\log\frac{p_\theta(a|c)}{q_i(a|c)}-p_\theta(a|c)+q_i(a|c)\right].
\]

接着明确定义冻结 rollout 分布的 step-local surrogate：

\[
F_n^{(k)}(\theta)=\sum_iw_i\,
\mathbb E_{x\sim\mathcal D_i,\,y\sim\pi_{\theta_n}(\cdot|x)}
\left[\frac1{|y|}\sum_t\ell_i^{(k)}(\theta;x,y_{<t})\right].
\]

在该 surrogate 上按既定 token → response → micro-batch 平均得到 $g_{i,s}$。预先用历史统计选定本步 $m_i$，然后收集新样本：

\[
A_n=\sum_i\frac{w_i}{m_i}\sum_{s=1}^{m_i}g_{i,s},\qquad
\mathbb E[A_n\mid\mathcal H_n,m]=\nabla F_n^{(k)}(\theta_n).
\]

这里 $\mathcal H_n$ 包含当前学生、优化器状态及过去观测；假设新 micro-batches 条件独立且每任务分布不随本步数量改变。该式保持的是当前训练状态下指定 surrogate 的期望，不声称不同 allocation 的整条训练路径相同，更不声称有限 k 对 full-vocabulary KL 无偏。

可直接用于正文的英文段落：

> We use an existing dense top-k distillation objective on student-generated prefixes. The retained set, normalization, and gradient convention are held fixed across allocation methods. GPAS does not introduce a token-level distillation estimator; it controls the number of micro-batches assigned to each task. Although dense distillation removes scoring-token sampling at a fixed prefix, task gradients remain stochastic because prompts and student-generated trajectories vary. We measure this remaining variability in the optimizer's preconditioned coordinates and allocate micro-batches under fixed task weights. The expectation-preservation statement is conditional on the current training state and refers to the specified step-local surrogate.

Section 3 随后直接保留：

\[
e_i=\mathbb E\|D_n(g_i-\mu_i)\|^2,\quad
V_n(m)=\sum_i\frac{w_i^2e_i}{m_i},\quad
m_i\propto w_i\sqrt{e_i}.
\]

重新安排原 Section 3.2 为“噪声的经验诊断”，把原估计器证明和 scoring-token 实验放附录。预条件器固定时的方差计算是精确的；它对 Adam 的实际更新仅提供局部近似，因为实际二阶矩、动量和 clipping 均影响更新。

**7. 哪些实验最能决定这篇论文能否成立**

先做标准 dense top-k 的短程诊断，再投入完整训练：冻结早、中、晚 checkpoint，在每域收集独立 micro-batches；用一半估计 allocation，另一半评估方差，避免自适应选择造成乐观偏差。优先检验三个问题：噪声是否异质、预条件是否改变排序、这种变化是否对应实际 loss/能力收益。

推荐的主对比均共享同一 dense loss、目标权重、归一化、教师和 optimizer：

| 对比 | 隔离的问题 |
|---|---|
| Uniform | 标准起点 |
| 未预条件的 Neyman allocation | 原始梯度噪声是否已足够 |
| GPAS per-step | optimizer coordinates 是否带来额外收益 |
| 使用完整 gap × velocity 信号的 counts-only 调度 | 学习进展信号与噪声信号谁更适合固定目标；明确这是适配版 D³ 信号 |
| GPAS cost-aware（系统计时支持时） | 方差收益是否转化为端到端时间收益 |

另设原配方比较：完整 D³-MOPD、完整 Open-MOPD，保留各自目标/归一化并报告同一套外部能力与计算预算。不能把自制 counts-from-loss-gap 称为 D³-MOPD，也不能因原配方目标不同就完全不比较。它们可在共同外部指标上比较，但不能把其损失差解释为纯 allocation 效应。

估计器稳健性建议选另一种已有 dense loss（例如 TA-OPD）跑 Uniform/GPAS 两行。sampled PG 也可以作为辅助对照，但不用再把“自定义 top-k × GPAS”作为主贡献消融。

借鉴其他论文的两个高价值设计：

- RouteOPD：每域单独蒸馏，作为任务能力可达性的参考。教师本身强不代表当前学生、prompt、长度设定能学到；这能区分单域教学失败与多域整合失败。
- 固定状态实验：同一批 prefixes 上比较梯度估计误差和局部下降，再进行真实 on-policy 训练。它隔离优化问题；不能替代端到端能力评估。

测量应至少包括任务能力、共同 held-out loss、实际 micro-batch centered variance、均值梯度间 cosine、response length/truncation、entropy/repetition、样本消耗、GPU-hours 和统计额外开销。KL 下降本身不能排除重复生成或状态分布变化带来的退化。

当前单种子 8 配置不足以支持训练稳定性结论。优先让 Uniform 和 GPAS 的关键差异有多个训练种子，再扩展方法数。Prompt bootstrap 只量化评估抽样，不量化训练种子不确定性。

**8. 当前实验方案中需要同步改的细节**

- math/code/IF/science 分别使用对应的 Qwen3-1.7B RL teacher，可称“四个独立 specialist teachers”；主设定为 Qwen3-1.7B-Base student 与四域 domain-RL teachers。
- 教师来源与 domain/size 当前混杂。不能凭 math/IF 对 code/science 的差异归因于 teacher origin；需要至少一个域内的教师替换对照。
- $w_i\propto1/L_i(0)$ 不是无害默认值。接近零的初始 KL 会放大任务权重，有限样本 log-ratio 均值还可能为负。建议均匀 $w_i$ 为主；若保留 inverse-initial-loss，则设稳定化并作敏感性分析。Instella 的初始化 anchor 便是初始 gap 近零的实际场景。
- 4,096 token 上限与 1.7B-Base non-thinking 可以作为预算选择，但必须先证明任务可学、截断不主导结果。Open-MOPD 的 1.7B 长链失败来自另一配置，不能直接推断我们的 non-thinking 配置必然失败。
- “不增加 forward/backward 次数”不等于没有开销。需要逐 micro-batch 暴露梯度、保存均值 buffer、进行参数规模 reduction、处理 sharding。$m_{\min}=2$ 只保证方差可定义，不保证估计可靠。可采用 warmup、跨步平滑/收缩和低频调整，并检验稳定性。
- 响应均值可消除跨任务长度导致的隐式权重漂移，但与 Open-MOPD 每域 token mean 并非完全等价：二者在域内对长短 response 的相对权重不同。
- $T(m)=C+\sum_i m_i\tau_i$ 只在测量支持时适用。异步 teacher 服务、流水重叠和排队会使实际耗时非加性；用实测 $T(m)$ 或系统拟合并留出验证，最终比较完整 GPU-hours。
- 当前 $G=16,m_i\in[2,8]$、Uniform 为 4 时，$V_{\rm unif}/V(m)\le2$。这只是方差比上界，不能解释为“两倍训练速度”；也不能与 D³ 的约三倍步数收益直接比较。
- 固定权重维持的是目标配方，不能保证所有任务能力都不下降。任务冲突、容量、教师错误和近似偏差均可能留下瓶颈。

**9. 建议的改稿顺序与停止条件**

第一步：修正 Related Work 的三类误读（MOPD support、Open-MOPD candidate gradients、share/gap 是否改变 sampling），加入 EMA-PG、TA-OPD、D³-MOPD 和近期 OPD dynamics 文献。

第二步：将摘要、Introduction、贡献列表和主图统一成“已有 dense distillation + optimizer-aware allocation”。保留真实需要检验的经验命题，删除已经预写方向的结果描述，例如默认断言 analytic removal 与 allocation 必然互补。

第三步：按第 6 节重写 Section 2，核心理论与具体 loss 解耦。将 full-KL 无偏、step-local surrogate、固定 task weights 三种不同保证分开。

第四步：先在标准 top-k 上测 residual micro-batch noise 与实际时间。若 held-out allocation ratio 近 1、预条件前后分配相同，或统计开销吃掉收益，就没有证据支撑当前 GPAS 的主要工程动机；应如实缩小结论或转向另一已诊断瓶颈。

若诊断成立，则用“现象 → 分配干预 → 真实效率和逐域能力”建立论文。最有说服力的结果是：在已修正长度权重、采用已有 dense loss 的配方上，GPAS 仍稳定改善计算效率，而且增益确实对应所测的预条件噪声，而不是来自不同任务目标或不同 token estimator。
