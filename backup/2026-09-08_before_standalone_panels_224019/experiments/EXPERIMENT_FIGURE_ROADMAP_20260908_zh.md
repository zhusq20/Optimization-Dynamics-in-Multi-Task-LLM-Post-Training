# 论文实验图路线图：已完成局部探针与剩余图槽

更新日期：2026-09-08。本次接入的 31 组本机机制探针于 **20:45 UTC 全部完成并核验**。这里刷新的是局部机制数据；其他节点的在线能力评测、训练进度和累计扫描没有在本次重新盘点，不能把历史缺口视为实时状态。

## 1. 当前交付与执行范围

已将 **20 组监督密度、7 组归一化、4 组教师对照** 落盘到论文目录，并生成 F1/F2/F5/F6/F7 五组主图，共 15 个主视图子图。按检查点和损失展开后共 15 页 1×3 图（45 个子图实例），它们是同一批证据的不同视图，**不是新增 45 个独立实验**。

- [五页主图 PDF](../figures/local_probe_results_20260908/main_local_probe_figures.pdf)
- [全部十五页 PDF](../figures/local_probe_results_20260908/all_local_probe_figures.pdf)
- [独立数据包与复现说明](local_probe_results_20260908/README_zh.md)
- [31 组源数据与 SHA256](local_probe_results_20260908/data_manifest.json)
- [每张图对应的实验清单](local_probe_results_20260908/figure_manifest.json)
- [完整结果解释](local_probe_results_20260908/RESULTS_zh.md)
- [当前 27 图槽清单](local_probe_results_20260908/figure_plan.csv) / [布局图](../figures/local_probe_results_20260908/figure_roadmap_9x3.pdf)

**用户已取消从 Base 重新训练交集方法及所有新增训练种子重复。** 当前计划没有新增训练或其关联评测；也不因旧正文曾承诺某实验而自动恢复。线上能力部分只接入已有运行的真实结果。多种子与新增 cap 训练从执行排期移除，F9a/F9b 标为 X。

按 24 个核心图槽计，现为 **15 E / 5 P / 4 N**。E 表示可画有明确范围的探索图；P 表示本地旧盘点有部分数据、本次没有补齐；N 表示本次没有目标测量；X 表示已从当前范围取消。状态不等于最终论文验收，也不是研究完成百分比。F3/F4/F8 的 P/N 沿用保守盘点口径，后续接入其他节点已完成产物时再更新。

## 2. 方法、样本和数值身份

- I64 / Overlap64 是当前学生与教师 Top64 的交集。权重仍按原学生 Top64 概率质量归一化，交集上不再次归一化；advantage detach。ST64、teacher-selected Top64、full-vocabulary 各自有不同目标，不能把它们之间的差异归因于仅改变候选数量。
- 密度探针：M-PG250/500、S-PG250、M-I64-DR250、S-I64-250 各四个 bank（42–45）；每域一条 response，cap256，每条至多四个 prefix。每个 bank 有九个分支，额外 PG action draws 在原始数据/CSV 中保留；主图不把它们当训练重复。
- 归一化：M-PG250/500 各两批 cap512；PG500 一批 cap2048；同一个 quota1/2/3/4 bank 分别用 uniform 与 prompt-share 域先验，共七组。DR/DT/GT/zero 使用保存的同一 Adam 状态。
- 教师：M-PG250/500 各两个 bank（42/43）；每位教师在 routed 与 common 输入下做 PG/I64 分支，加 zero-gradient。每 bank 六个共享教师对，不把重复 pair 当独立样本。
- m-i64dr250/s-i64250 含旧 Student64 训练历史，图标题明确标为 mixed history；不是全程交集训练轨迹。
- 全部新图测的是 HF 局部梯度和保存 Adam 下的模拟 BF16 写回；不是历史 Megatron/SGLang 的精确重放、真实在线单步或能力评测。累计变化与局部写回不能混画成同一量。
- 区间只表示诊断 bank 的实测 min–max，不是训练 seed 方差或 95% CI。配对比较共享教师与输入，不做独立样本回归显著性检验。

## 3. 已生成的图与主要读数

| 图组 | 主图 | 完整版本与数据 |
|---|---|---|
| F1 长度与权重 | [PDF](../figures/local_probe_results_20260908/F1_lengths_and_weights.pdf) / [PNG](../figures/local_probe_results_20260908/F1_lengths_and_weights.png) | 四个普通 cap512 bank；不等配额 uniform 对照。逐 response 长度、截断、DR/DT/GT 权重见 plot_data/responses.csv |
| F2 归一化 | [PDF](../figures/local_probe_results_20260908/F2_normalization.pdf) / [PNG](../figures/local_probe_results_20260908/F2_normalization.png) | 全部七组；covariance.csv、branches.csv、heldout.csv；完整三对梯度比较见 normalization_gradients.csv 及各 measurements.json |
| F5 教师更新重叠 | [PDF](../figures/local_probe_results_20260908/F5_teacher_overlap_pg500_Overlap64.pdf) / [PNG](../figures/local_probe_results_20260908/F5_teacher_overlap_pg500_Overlap64.png) | PG250/500 × PG/Overlap64 共四页；pairs.csv 含支持数、原始梯度/写回余弦、阈值和等数量支持检查 |
| F6 教师分布距离 | [PDF](../figures/local_probe_results_20260908/F6_teacher_distance_pg500_Overlap64.pdf) / [PNG](../figures/local_probe_results_20260908/F6_teacher_distance_pg500_Overlap64.png) | PG250/500 × PG/Overlap64 共四页；teacher_js.csv、pairs.csv，JS 来自相同公共前缀 |
| F7 监督密度 | [PDF](../figures/local_probe_results_20260908/F7_density_m-pg500.pdf) / [PNG](../figures/local_probe_results_20260908/F7_density_m-pg500.png) | 五个学生状态各一页；thresholds.csv、branches.csv、pairs.csv。PG500 是主视图，其余全部保留 |

F5/F6 主视图固定用 PG500 和当前交集 loss，与已有主参考检查点一致；不是根据效果大小选择。F7 同样使用 PG500 为主视图。其余状态与 loss 可在全部十五页 PDF 中直接查阅。

可写入结果段落的观察：

1. 四批教师探针中，routed 原始梯度余弦均值为 PG **0.0057**、I64 **−0.0032**；common 分别为 **0.9739、0.9630**。输入是否匹配是解释教师方向差异的关键控制。不能把 routed 的近正交解释成教师独有、互不重叠的参数子网络。
2. 七组归一化的长度–梯度分解最大相对残差约 **1.8e-16**。长回答批次 DR–DT 原始梯度余弦 **0.7063**、DR–GT **0.6583**，但局部 held-out KL 排名并不一致。
3. M-PG500 的四 bank 均值：PG/I64 的原始梯度 L2 为 **52.08/39.44**，BF16 写回 L2 为 **0.03385/0.03395**，精确非零比例为 **0.1699%/0.1712%**。更多监督候选没有对应明显更大的写回范围；不能由范数相近推断方向或能力等价。
4. zero-gradient Adam 对照仍有较大 BF16 写回。额外不执行 Adam 的全坐标检查确认 PG250/500 的 BF16(master) 与保存模型完全一致，排除了这两个检查点已有 master/model 不一致的解释。记录见 no_step_cast_audit.json。

**F2c 只作探索性诊断。** 每域只有一条 held-out response；两组中 zero-gradient 的 KL 下降超过三个 reduction。相同不等配额 bank 的 uniform/prompt 两次运行，其 GT 梯度范数仍有小幅数值差异（6.173284587 vs 6.171707070），KL 读数也有差异，来源尚未定位。因此图中展示全部正负读数，不做规则优劣排序、因果能力结论或显著性声明。

## 4. 当前 8×3 核心图槽与可选第九行

### F1：Token balancing——长度与实际权重（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F1a [E]** | 四个 cap512 固定批次的 response 长度散点；× 标截断，颜色为任务域 | 新图已生成；是受 cap 限制的诊断长度，不代表自然长度或线上完整分布 |
| **F1b [E]** | 同四批次各域有效 token 份额；25% prompt-share 虚线 | 新图已生成；token share 不等于 DR 的域损失系数；旧线上 token 轨迹保留在历史预览 |
| **F1c [E]** | quota1/2/3/4、uniform 域先验下每条 response 的 DR/DT/GT 系数 | 新图已生成；prompt-share 对照也已落盘，完整权重见 responses.csv，不额外占用图槽 |

### F2：Token balancing——局部梯度与 KL（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F2a [E]** | x=七组检查点/bank/干预，按域散点展示 Cov(length,g) 范数除以 mean(length) | 全部七组已画，零值保留；残差只验证固定批次恒等式，不证明分布式 reducer 正确 |
| **F2b [E]** | 七行探针 × 三对 reduction 的 raw-gradient cosine 热图 | 原3×3单批次设计改为全部七批比较；每分支范数和 clip coefficient 在 branches.csv |
| **F2c [E]** | 七行探针 × zero/DR/DT/GT 的宏平均 held-out KL 下降热图，单位1e-3 nats | 原单批次四域热图改为全批次宏平均视图；四域原始值完整保存在 heldout.csv；不能替代 benchmark accuracy |

### F3：Token balancing——在线能力（N/N/N，本次不刷新外部评估）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F3a [N]** | 三 reduction × 四域的真实能力变化热图，250/500 分面 | 本批探针不提供新能力点；只接入其他节点已有训练的核验评测，方法标签保留切换历史 |
| **F3b [N]** | 累计 generated tokens 对四域 macro，按真实检查点标点 | 使用已有运行的评估与去重 token 日志；不恢复新增种子，不画训练置信区间 |
| **F3c [N]** | updates 对 worst-domain change，并记录最差域和截断率 | 与 F3a 复用同一评估，不另跑模型；不能用局部 KL 填空 |

### F4：累计 BF16 变化（P/P/P，本次不补累计扫描）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F4a [P]** | 相对初始化的累计有效变化比例及 exact nonzero，按检查点/暴露作图 | 沿用已核验扫描，后续接入已有250/500扫描；本批局部写回不能填成累计变化 |
| **F4b [P]** | 累计 delta 的能量集中度、energy90 与总 L2 | 旧日志有top1/5/10%离散点；没有连续扫描时不插值为实测全曲线 |
| **F4c [P]** | 方法/检查点 × transformer layers与embedding/head 的累计变化热图 | 复用历史扫描；全模型按参数数加权；保留ST64/I64及混合历史身份 |

### F5：教师更新及输入对照（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F5a [E]** | 同一 checkpoint/loss 的 routed 四教师 BF16 support Jaccard 热图，两 bank 均值 | PG250/500×PG/I64四版全部完成；阈值1e-5；支持数在pairs.csv，热图对角为自身比较1 |
| **F5b [E]** | x=原始梯度余弦，y=BF16写回余弦，圆=routed，三角=common，颜色=教师对 | 由原 Jaccard–cosine 设计调整为直接展示输入控制和优化器变换；两种cosine不可混称 |
| **F5c [E]** | 教师与zero-gradient的Jaccard及逐层随机参照，纵轴对数；误差棒为两bank min–max | 随机量是期望交/期望并之比，不是实测E[Jaccard]；实际支持数与rho=0.01/0.05/0.1%等数量支持结果均已落盘 |

等数量支持只在双方非零坐标足够时定义；ties 按幅度降序、全局坐标升序处理。不用零值补足 top1/5/10% 来宣称 learned subnetworks。现有图只使用绝对阈值；等数量敏感性可直接从 pairs.csv 重画，无需重跑模型。

### F6：教师 JS 与更新位置差异（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F6a [E]** | 同一 checkpoint 两个公共 bank 的teacher–teacher全词表JS均值热图，nats | PG250/500均完成；teacher–student JS另存teacher_js.csv；不同checkpoint bank不同不代表teacher权重变化 |
| **F6b [E]** | x=公共bank JS，y=routed 的1−Jaccard，显示六对教师各两个bank点 | PG/I64、两个checkpoint全部绘制；不拟合回归或把共享pair视为独立样本 |
| **F6c [E]** | 与F6b同一横纵尺度，纵轴改为common输入下1−Jaccard | 同批配对控制已完成；输入相同后梯度高度一致见F5b，不预设分布距离有普遍因果效应 |

### F7：监督目标的局部对照（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F7a [E]** | exact0、1e-8/1e-7/1e-6/1e-5阈值对BF16变化比例；六种目标/控制的均值及bank范围 | 五个学生状态×四bank全部完成，含ST64/I64/teacher64/full/PG/zero；额外PG draws保留CSV |
| **F7b [E]** | x=pre-clip gradient L2（symlog），y=BF16写回L2，显示四bank原始点与zero控制 | 五个学生状态全部完成；坐标分开，不能把两个不同量直接比大小；精确非零比例、FP32 master变化另存CSV |
| **F7c [E]** | 各目标与同bank full-vocabulary BF16提议的余弦；原始点、均值及min–max | 五状态完成；I64学生状态注明mixed history；不声称损失等价或全词表在线训练效果 |

### F8：真实在线更新、能力与成本（N/P/P）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F8a [N]** | 原轨迹实际相邻BF16更新曲线 | 本批没有新增在线单步；旧update字段是FP32。保留缺口，不为补图自动启动新训练 |
| **F8b [P]** | 现有PG/交集运行的单域math与joint四域能力，标注暴露和切换历史 | 本批无新能力点；只接入现有评测，不恢复训练种子实验，不以同step替代匹配暴露 |
| **F8c [P]** | 现有运行macro对真实generated tokens，附GPU时间和最差域变化 | PG旧系列有部分证据；成本按实际运行去重，局部探针开销与训练分开 |

### F9：已取消及范围外扩展（X/X/N）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F9a [X]** | 原计划为配对训练种子的效应图 | 用户明确取消全部新增训练种子；不把诊断draw代替，不再列入待启动任务 |
| **F9b [X]** | 原计划为1024/4096 cap新增训练能力对照 | 已从当前无新增训练的范围移除；局部512/2048探针不能替代；正文若保留旧承诺须删改而非宣称完成 |
| **F9c [N]** | 固定checkpoint的诊断batch-size敏感性 | 未包含在完成的31组中，当前不排队；不等quota控制不是1/4/8 batch-size sweep |

## 5. 复现、校验与后续写作

在论文仓库运行，仅CPU读取已落盘的数据，不加载模型或访问原训练目录：

```bash
python experiments/plot_completed_local_probes.py
python experiments/build_figure_plan.py
```

数据包校验了31个measurements及图所需原始bank、权重、support、manifest、完成标记的SHA256。重新绘图会校验包内全部文件；plot_data各行保留原始job、state与相对source路径。全层测量在raw的measurements.json，未把大概率tensor复制到论文目录，因为当前图不依赖它。

图注必须保留：本地HF、保存Adam、模拟BF16写回、checkpoint/loss/bank数量、cap与prefix范围、mixed history（适用时）、min–max不是CI。F2c特别注明数值与小heldout限制；F5/F6不写独立教师样本显著性；F7不把不同损失当单纯候选数实验。

下一步是将五组图的描述性观察写入正文，并接入其他节点已有能力/累计扫描产物。不要恢复本轮已取消的新训练、多种子或cap训练。论文正文仍有旧teacher-top64定义与更大实验范围的文字，应按实际完成范围单独修订；本次只更新图、数据和作图文档。

01:10/01:50的历史盘点继续保存在figure_audit_20260908和figures/evidence_audit_20260908，不与当前数据混合。修订前路线图与下一批实验清单已备份到backup/2026-09-08_before_local_probe_figures_221426/experiments/。
