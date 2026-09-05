# MOPD 论文三点贡献与实验主线

更新：2026-09-05，已纳入本轮关于“第三点仅做简洁对照、第二点按原顺序展开”的修正。

本文是观察与解释为主的论文。GPAS 不再作为主线，不要求提出新采样算法。当前已有推导与实验设计，但没有这些新实验的实测结果；不把预期写成发现。

## 1. 长度、域内平均与能力平衡：解释已有训练选择

研究问题：不同 token loss 平均方式如何改变各域及域内 response 的权重？这些变化能否解释多任务能力不平衡？

必须区分三种实现：

| 实现 | 域之间权重 | 同一域内 response 权重 |
|---|---|---|
| 全 batch token mean | 该域有效 token 占比 | 正比于 response 长度 |
| 每域 token mean，再按目标域权重平均 | 指定域权重 | 仍正比于 response 长度 |
| 每条 response token mean，再域内与域间平均 | 指定域权重 | 同域内每条 response 等权 |

第二行就是 Open-MOPD TSB 的代数等价形式，不重复包装成新算法。原始 MOPD 的形式目标已有 response 长度平均，不能从 Open-MOPD 的 token-mean 实例推广到所有 MOPD。

解释性推导：在同一固定 batch 内，域 token-mean 梯度 = response-mean 梯度 + 长度与 response-mean 梯度的经验协方差 / 平均长度。它区分跨域 token 份额偏权与域内长度偏权。它不是新的加权平均数学定理，也不保证 response 等权一定最优。

最小证据：同 batch 三种梯度的精确分解，随后配对训练的各域能力曲线。先采用相等 prompt 配额，使目标域先验不偷偷改变。长度 cap 的 1024/4096 对照是辅助观察，记录截断与答案完成率；不让长度 sweep 扩展为主算法。既看训练步数，也看 token 暴露和实际计算成本。

## 2. 稀疏性 → MOPD 子网重叠 → 教师差异与子网差异：论文中心

### 2.1 首先确认 OPD 参数更新的稀疏性

同一个初始 student，分别观察单教师 OPD 和多教师 MOPD。记录初始、早期、中期、最终 checkpoint 相对初始化的参数位移。

报告多个阈值的稀疏度、参数更新范数、承载 90% 更新能量所需的坐标比例。优先用真实 FP32 master weights；同时保留 BF16 checkpoint 指标以便对照已有论文。把 checkpoint delta 的稀疏与单步梯度/更新稀疏分开。

这一步是复现与多教师场景扩展，不能声称首次发现 OPD 稀疏，因为 Dense Supervision, Sparse Updates 已直接研究。

### 2.2 再看不同教师是否更新同一个 MOPD 学生的重叠子网

在某个 MOPD checkpoint θ_t 上固定 student 和优化器状态。各教师分别处理其路由域的诊断 batch，从同一个 θ_t 独立计算梯度与拟执行的 optimizer step。诊断结束恢复状态，不让前一个教师改变后一个教师的起点。

由每位教师的梯度/拟执行更新定义 support。比较各教师 support 的 Jaccard、固定 top 1%/5%/10% 等坐标比例下的重叠，以及同层随机基线。重复若干诊断 batch，避免把某一次采样当稳定子网。

主要输出：几个 MOPD checkpoint 的 teacher/domain overlap 矩阵和对应稀疏度曲线。

关键边界：单教师模型独立训练后的终点 support 只能作为参照，不能替代“各教师对联合学生的更新”。联合 delta 也不能在 Adam/在线轨迹下唯一分解为各教师历史贡献。报告的是同一实际联合 checkpoint 上的 teacher-conditioned update tendency。

重叠本身无好坏方向；更新符号/梯度 alignment 与能力结果只作辅助解释，不把“冲突最小化”或强制不重叠子网变成新主任务。

### 2.3 最后检验：教师分布差异越大，子网差异是否越大？

对同一批 student prefixes，让每个 teacher 都打分，计算 teacher-pair 的 JS 等分布距离。将它与同一 checkpoint 下的子网距离 1−Jaccard 配对，绘制散点及训练中的变化。

同时记录每位 teacher 到当前 student 的分布差异与更新大小。这些是辅助描述，帮助判断观察到的关系是否只是整体更新变大/变小。

核心分析可以使用各教师各自路由域上的更新；增加少量“所有教师使用完全相同 prefixes 计算更新”的检查，帮助区分 teacher 差异与输入域差异。无需先建设复杂的因果辨识项目。若只有少数 teacher pair，就逐对报告当前设置的观察，不把共享教师的 pair 当独立大样本。

当前 code/science 暂时共享一个 Qwen3-4B teacher。二者 teacher 分布距离在相同 context 上为零，路由更新不同则反映 context 差异；不能当作两个独立 teacher。已有教师中间 checkpoint 可提供距离变化，不要求为此重新训练大量专家。

可能得到的结果都可报告：距离越大子网越不同；没有明显关系；关系依赖任务、checkpoint 或 teacher–student gap。不能预先写第一种。

## 3. Supervision density：词表监督范围的简洁对照

这里的 supervision density 明确指每个位置的词表监督范围：full-vocabulary loss、top-k 子集与 sampled-token policy gradient；不指监督覆盖多少生成位置。

在单教师和多教师 checkpoint 上都加入小批公共 prefixes 的 full-vocabulary 梯度/拟执行更新对照，直接测量全词表与 sampled 的更新稀疏度。累积训练轨迹先沿用下表 PG/top-k 比较；局部全词表对照不被写成已经完成全词表在线训练。

不把噪声、MC 次数或 tail correction 的机制研究列为必做：

| 设置 | sampled-token PG | teacher top-64 loss |
|---|---|---|
| 单教师 OPD | 稀疏度曲线与能力背景 | 同样测量 |
| 多教师 MOPD | 稀疏度曲线；各教师 support overlap | 同样测量 |

保持初始化、教师、归一化、长度 cap、优化器、训练预算、参数测量口径一致；在线 rollout 随各自模型生成，不声称全程轨迹相同。明确 top-k 候选来源、概率是否在集合内重新归一化以及裁剪配置。

如果二者在已测范围内没有实质差异，这就是该项结果，不强行追究原因。报告效应大小与不确定性，避免把低精度下没看见差异写成所有条件下严格等价。

只有稳定、有意义的差异出现时，再决定是否用固定前缀 MC、full-vocabulary 或方差控制解释。该解释不是论文重心，也不是完成前两项贡献的前置条件。

若 PG 更稀疏，也不能直接写“因此 MOPD 应使用 PG”；需要能力和代价结果支持。若 PG 与 top-k 稀疏度类似，则可以写“在这些单教师和多教师配置下，增加每位置的词表监督没有明显改变所测参数更新稀疏性”。

## 4. 最小运行复用，不扩大为全面网格搜索

建议初始六种训练配置，先对一个代表性单教师配置完成测量：

| ID | 设置 | 用途 |
|---|---|---|
| S-PG | 代表性单教师，sampled PG | 贡献2.1与贡献3 |
| S-TK | 同教师，top-64 | 贡献2.1与贡献3 |
| M-PG | 多教师，sampled PG，domain-response mean | 贡献2与贡献3 |
| M-TK-DR | 多教师，top-64，domain-response mean | 三项共享主参照 |
| M-TK-DT | 多教师，top-64，domain-token mean | 贡献1 |
| M-TK-GT | 多教师，top-64，global token mean | 贡献1 |

各教师独立训练参考、更多教师/种子和长度辅助实验根据现有 checkpoint 与预算扩展，不默认为六种配置各自又开完整网格。核心论文需据实际覆盖范围限制结论；真实耗时先测，不沿用旧 GPAS 的未经验证估计。

最终主图/表控制为：归一化与各域能力图；single/MOPD稀疏曲线；MOPD教师support重叠矩阵；teacher分布距离与1−J散点；PG/top-k对照表。

## 5. 文稿改动与文献边界

主稿标题已改为 Token Balancing, Parameter Update Sparsity, and Supervision Density in Multi-Teacher On-Policy Distillation。

正文顺序：问题与三贡献 → setting → normalization → sparse OPD / MOPD overlap / teacher distance → PG vs top-k → 实验设计 → related work / discussion。

旧 GPAS 理论文件、脚本与旧计划保留为历史材料，不再从主稿调用或列作当前必做。此前稿件已备份到 backup/2026-09-05_before_three_contribution_restructure_170838。

直接相关来源：

- [MOPD（2026）](https://arxiv.org/abs/2606.30406)
- [Open-MOPD（2026）](https://arxiv.org/abs/2608.19098)
- [Reinforcement Learning Finetunes Small Subnetworks（2025 v2）](https://arxiv.org/abs/2505.11711v2)
- [Dense Supervision, Sparse Updates（2026 v3）](https://arxiv.org/abs/2606.13657v3)
- [On the Geometry of OPD（2026 v3）](https://arxiv.org/abs/2606.07082v3)

已有工作已覆盖 OPD 稀疏与换教师后的部分 overlap。我们的增量应落在多教师共同学生上的观测、教师距离与support差异的系统对应，以及与归一化和loss对照构成的完整实证叙事。
