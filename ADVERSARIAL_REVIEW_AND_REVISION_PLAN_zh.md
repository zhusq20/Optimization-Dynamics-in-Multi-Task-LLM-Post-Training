# 对抗性审稿与修改计划

> 状态：历史记录。当前主文已改为“所有任务每步参与、按任务分配 micro-batch 数”的版本；下文关于 exact-`K` 抽样、probe 和 inclusion probability 的意见只适用于此前的资源受限方案。

## 一、审稿结论

**当前建议：Reject（完成真实 MOPD 实验后可重新评估）。**

论文抓住了一个重要问题：MOPD 的任务权重同时承担“定义目标”和“分配算力”两种作用，导致自适应方法之间很难公平比较。将目标权重固定，再用优化器处理后的梯度尺度分配任务，是一个清楚且有实用价值的方向。当前稿件的主要障碍是证据链不完整。正文提出“每个 response 和每个 GPU-hour 的效率”这一经验问题，却只给出实验协议，没有真实 MOPD checkpoint、训练曲线、系统 trace 或最终结果表。受控合成实验只能验证估计器和 AdamW moment 的计算，无法支持大模型训练效率结论。

## 二、最可能导致拒稿的问题

### P0：提交前必须解决

1. **缺少真实 MOPD 结果。** 摘要、引言和结论目前容易让人以为四教师实验已经完成。论文实际只有 protocol。提交版必须报告 warm-start gradient bank、loss--response 曲线、loss--GPU-hour 曲线、四任务终点指标、peak HBM 和教师切换 trace。
2. **理论主题与实验宽度不匹配。** 论文讨论每次更新包含多少任务，原设计只比较 `K=1` 和 `K=4`。加入 `K=2` 才能检验理论是否能解释中间宽度，也能避免“先选两个端点，再把结果解释成规律”的质疑。
3. **训练稳定性证据不足。** matched prompt stream 能降低数据差异，却无法覆盖初始化、生成和优化路径造成的训练波动。主要效率结论需要最少三次独立重复，或对预注册的关键三组配置做三次确认实验。
4. **缺少常用 MOPD 优化器基线。** `K=1` 原设计全部使用修改后的 taskwise second moment。需要加入 conventional AdamW 的 Uniform `K=1`，才能显示方法相对常用训练方式的净收益，并区分 sampler 与 optimizer-state 改动。
5. **系统代价模型忽略教师驻留状态。** 教师切换是主要成本时，单个任务的平均时间取决于当前 resident teacher。Cost-GPAS 应使用由当前驻留状态预测的 critical-path time；实际结论仍以端到端 GPU-hour 为准。

### P1：明显削弱论文的问题

1. **新颖性边界需要更直接。** `gradient-norm importance sampling` 和 `square-root cost correction` 已有成熟来源。论文的差异应集中在固定 MOPD objective、AdamW taskwise moment、exact-`K` task set 和 streamed-teacher 系统实现的组合，并用真实结果证明这套组合有价值。
2. **主文理论过长。** 独立 inclusion 的完整方差分解、pairwise correction、mixed quota、chi-square identity 和 smoothness 说明同时出现，超过 LLM 论文读者理解方法所需的范围。主文只保留 objective、importance correction、taskwise AdamW moment、GPAS 和 Cost-GPAS 五个公式；证明及 exact-`K` 修正放入附录。
3. **AdamW 几何仍是局部近似。** 当前对角 scaling 来自某个训练状态，gradient clipping、momentum history 和 changing second moment 会改变真实更新。论文应把它称为可测量的局部 allocation score，并通过 frozen checkpoint bank 检验它能否预测 estimator error。
4. **`beta^K` 的 response-clock 处理需要验证。** 该设计会改变一次更新中 moment observation 的时间聚合方式。应在附录给出实现说明，并用 resume/equivalence test 验证累计 decay，而不是在正文展开推导。
5. **超参数没有完整定值。** `A_max=50` 已给出，EMA decay、probability floor、failure penalty 和 common loss threshold 仍需在训练前写进 manifest。
6. **能力保持需要明确判据。** relative teacher loss 是优化目标，MATH、Code、IF、Science capability 仍应设置非退化检查，避免方法只改善教师 surrogate。

### P2：表达与组织问题

1. 正文包含大量防御性否定句，例如 “not a promised loss reduction” 和 “not an exact oracle”。这些句子把读者注意力带到作者没有声称的内容。改成直接说明量的含义和适用范围。
2. `hence`、`thus`、`across`、`yield`、`rather than`、`not ... but ...` 等连接词和绕行句式过多。用明确主语和短句替换。
3. “current workspace lacks ...” 属于内部项目管理信息，不应出现在论文正文或 reproducibility statement。
4. 系统指标分类表只有待填项目，没有数值证据。提交版使用一张真实结果表；实验前将清单保留在内部计划文档。

## 三、理论—实验闭环

| 理论或设计 | 可检验预测 | 对应实验 | 通过标准 |
|---|---|---|---|
| Importance correction | 改变采样概率后，平均梯度目标保持一致 | 受控 Monte Carlo；frozen gradient bank | mean residual 接近 Monte Carlo error |
| Optimizer-aware scale | AdamW-scaled score 比 raw gradient norm 更好地预测 AdamW metric error | 合成受控实验；真实 checkpoint held-out bank | held-out error 低于 Uniform 和 raw-norm proposal |
| Taskwise AdamW moment | second-moment target 对 proposal 保持稳定 | 受控 moment 实验；`K=1/2/4` frozen bank | relative bias 接近采样误差；端到端无能力退化 |
| Increasing `K` | task-selection noise 随 task coverage 增加而下降 | `K=1,2,4` exact-set variance | 实测下降与 frozen-bank 预测方向一致 |
| Cost-GPAS | gradient scale 与 critical-path time 的组合改善单位时间进展 | GPAS、Cost-GPAS、Uniform matched runs | 到共同 loss threshold 的 GPU-hour 更低 |
| Streamed teacher | `K` 的统计收益可能被切换成本抵消 | load/offload trace、tail wait、GPU-hour | 结论由真实 critical path 支持 |

## 四、修改计划

### 已在本轮执行

- 重写摘要和引言，使其只陈述已有的受控证据，并把真实 MOPD 结论标为提交前条件。
- 压缩主文理论，统一使用 `w_i`、`p_i`、`r_i`、`t_i` 四个直观量。
- 删除与主实验无直接关系的主文推导，将 exact-`K` 细节保留在附录。
- 重新运行受控实验并在正文报告可复现数值。
- 建立理论预测到实验指标的显式映射。
- 把大模型实验扩展为 `K=1,2,4`，加入 conventional AdamW `K=1` 基线和关键配置重复验证。
- 系统清理含糊指代、防御性否定句和指定的 AI 风格词汇。

### 需要算力和真实训练资产

1. 冻结四个 teacher checkpoint、数据 revision、prompt template 和硬件 manifest。
2. 采集 warm-start gradient bank，先计算 GPAS headroom 和 `K=1,2,4` 的 held-out estimator error。
3. 跑完整 matched MOPD 配置；关键三组配置至少三次独立重复。
4. 生成两张主曲线和一张汇总表，替换 protocol-only 文字。
5. 根据真实 trace 决定 Cost-GPAS 是否需要 resident-state cost；任何改动在查看训练终点前冻结。

## 五、本轮已运行的受控结果

脚本使用固定随机种子重新生成了 `20,000` 次 estimator trials 和 `1,000,000` 次 AdamW moment draws。受控构造中：

- GPAS 的 AdamW-metric estimator error 是 Uniform 的 `0.717`，下降 `28.3%`。
- Cost-GPAS 的 time-weighted error 是 Uniform 的 `0.889`，下降 `11.1%`。
- raw gradient-norm proposal 的 AdamW-metric error 是 Uniform 的 `6.868`，说明 raw norm 在优化器缩放改变任务排序时会选错任务。
- conventional AdamW second moment 相对固定 target 的偏差是 `0.816`；taskwise moment 的 Monte Carlo 偏差是 `0.00235`。

这些结果验证实现与公式，不能替代真实 MOPD 效率实验。
