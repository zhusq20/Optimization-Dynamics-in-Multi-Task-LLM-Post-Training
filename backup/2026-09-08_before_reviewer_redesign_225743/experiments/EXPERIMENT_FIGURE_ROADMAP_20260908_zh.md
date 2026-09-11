# 论文实验图路线图：已完成局部探针与剩余图槽

更新日期：2026-09-08。本次接入的 31 组本机机制探针于 **20:45 UTC 全部完成并核验**。这里刷新的是局部机制数据；其他节点的在线能力评测、训练进度和累计扫描没有在本次重新盘点，不能把历史缺口视为实时状态。

## 1. 当前交付与执行范围

已将 **20 组监督密度、7 组归一化、4 组教师对照** 落盘并绘制为 **15 张独立实验图**：F1a–c、F2a–c、F5a–c、F6a–c、F7a–c。每张分别导出PDF、PNG、SVG；不再把三个子图拼成一张图片。

按用户反馈，F5/F6把step250、step500和PG/I64放进对应的同一张图，以矩阵列、颜色、点形和空心/实心保留实验身份；F7同时展示五个学生状态和各监督目标。原F5–F7按条件展开的39个子图实例合并为9张汇总图，**只平均同一条件内的bank，不跨条件平均**。

- [独立图浏览页](../figures/local_probe_results_20260908/index.html)
- [逐页审阅PDF：每页仅一张图](../figures/local_probe_results_20260908/all_panels.pdf)
- [各图英文caption草稿](local_probe_results_20260908/FIGURE_CAPTIONS.md)
- [独立数据包与复现说明](local_probe_results_20260908/README_zh.md)
- [31组源数据与SHA256](local_probe_results_20260908/data_manifest.json)
- [每张图的实验及编码清单](local_probe_results_20260908/figure_manifest.json)
- [每个散点/矩阵聚合前的观测](local_probe_results_20260908/panel_values.json)
- [完整结果解释](local_probe_results_20260908/RESULTS_zh.md)
- [当前27图槽清单](local_probe_results_20260908/figure_plan.csv)

### 绘图规范（后续重画沿用）

1. **一张实验图一个文件**。同一问题的checkpoint/loss汇总到同一坐标或矩阵，条件通过必要轴标签和图例区分，不为每个条件复制整张图。
2. 默认字号18pt，轴标签19pt、刻度16pt、矩阵数值15pt；紧凑图例13–14pt。避免将宽三联图缩小后导致文字不可读。
3. **图内不放标题、子图编号、说明段落或脚注**。只保留坐标轴、刻度、数值及必要图例。实验条件、R/C及方法缩写、mixed-history星号、数值限制和解读均写入caption。
4. 颜色、点形、空心/实心明确编码不同条件；矩阵均值只聚合同条件bank。散点保留各bank观测，不把共享教师pair当独立重复。
5. 更完整但不适合主图叠加的阈值/等数量支持结果保留在CSV。F7a主视图统一使用预先选定的1e-5阈值；没有为提高密度把30条阈值曲线叠成一团。

**用户已取消从 Base 重新训练交集方法及所有新增训练种子重复。** 当前计划没有新增训练或其关联评测；也不因旧正文曾承诺某实验而自动恢复。线上能力部分只接入已有运行的真实结果。多种子与新增 cap 训练从执行排期移除，F9a/F9b 标为 X。

按 24 个核心图槽计，现为 **15 E / 5 P / 4 N**。E 表示可画有明确范围的探索图；P 表示本地旧盘点有部分数据、本次没有补齐；N 表示本次没有目标测量；X 表示已从当前范围取消。状态不等于最终论文验收，也不是研究完成百分比。F3/F4/F8 的 P/N 沿用保守盘点口径，后续接入其他节点已完成产物时再更新。

## 2. 方法、样本和数值身份

- I64 / Overlap64 是当前学生与教师 Top64 的交集。权重仍按原学生 Top64 概率质量归一化，交集上不再次归一化；advantage detach。ST64、teacher-selected Top64、full-vocabulary 各自有不同目标，不能把它们之间的差异归因于仅改变候选数量。
- 密度探针：M-PG250/500、S-PG250、M-I64-DR250、S-I64-250 各四个 bank（42–45）；每域一条 response，cap256，每条至多四个 prefix。每个 bank 有九个分支，额外 PG action draws 在原始数据/CSV 中保留；主图不把它们当训练重复。
- 归一化：M-PG250/500 各两批 cap512；PG500 一批 cap2048；同一个 quota1/2/3/4 bank 分别用 uniform 与 prompt-share 域先验，共七组。DR/DT/GT/zero 使用保存的同一 Adam 状态。
- 教师：M-PG250/500 各两个 bank（42/43）；每位教师在 routed 与 common 输入下做 PG/I64 分支，加 zero-gradient。每 bank 六个共享教师对，不把重复 pair 当独立样本。
- m-i64dr250/s-i64250 含旧 Student64 训练历史，图标题明确标为 mixed history；不是全程交集训练轨迹。
- 全部新图测的是 HF 局部梯度和保存 Adam 下的模拟 BF16 写回；不是历史 Megatron/SGLang 的精确重放、真实在线单步或能力评测。累计变化与局部写回不能混画成同一量。
- 矩阵只报告同条件bank均值，散点保留原始观测；min–max可从CSV重建，不解释为训练seed方差或95% CI。配对比较共享教师与输入，不做独立样本回归显著性检验。

## 3. 已生成的图与主要读数

| 图组 | 独立PDF文件 | 合并方式与数据 |
|---|---|---|
| F1 | [F1a](../figures/local_probe_results_20260908/F1a.pdf) · [F1b](../figures/local_probe_results_20260908/F1b.pdf) · [F1c](../figures/local_probe_results_20260908/F1c.pdf) | 四个普通cap512 bank与不等配额对照；逐response长度/截断/权重见responses.csv。 |
| F2 | [F2a](../figures/local_probe_results_20260908/F2a.pdf) · [F2b](../figures/local_probe_results_20260908/F2b.pdf) · [F2c](../figures/local_probe_results_20260908/F2c.pdf) | 七组探针统一展示；F2a横向散点，F2b/c全条件矩阵。 |
| F5 | [F5a](../figures/local_probe_results_20260908/F5a.pdf) · [F5b](../figures/local_probe_results_20260908/F5b.pdf) · [F5c](../figures/local_probe_results_20260908/F5c.pdf) | 两个checkpoint×两个loss×两种输入共八条件；F5a六教师对×八条件，F5b全部96个成对观测，F5c全部zero与随机对照。 |
| F6 | [F6a](../figures/local_probe_results_20260908/F6a.pdf) · [F6b](../figures/local_probe_results_20260908/F6b.pdf) · [F6c](../figures/local_probe_results_20260908/F6c.pdf) | JS不依赖local loss，因此F6a只按六教师对×两个checkpoint展示；F6b/c合并PG/I64和250/500，保持相同坐标尺度。 |
| F7 | [F7a](../figures/local_probe_results_20260908/F7a.pdf) · [F7b](../figures/local_probe_results_20260908/F7b.pdf) · [F7c](../figures/local_probe_results_20260908/F7c.pdf) | 五个学生状态与全部监督目标合并；a=固定阈值变化比例，b=原始梯度/写回范数散点，c=与full参照的方向一致性。 |

所有学生状态、检查点和loss版本均进入对应汇总图；不再选择PG500作为唯一主视图。每个panel的源job和颜色/点形规则记录在figure_manifest.json，具体读数记录在panel_values.json。

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
| **F2a [E]** | x=长度–梯度修正量，y=七组探针，任务域用不同颜色和微小纵向偏移区分 | 独立导出F2a；横向散点避免长条件标签斜排，无类别连接线；零值保留 |
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
| **F5a [E]** | 行=六个非重复教师对，列=250/500×PG/I64×routed/common共八条件；BF16 support Jaccard | 同条件两个bank求均值；不画重复对角/对称单元，支持数与等数量检查保留pairs.csv |
| **F5b [E]** | x=原始梯度余弦，y=BF16写回余弦；颜色=loss，圆/方=250/500，实心/空心=routed/common | 同一图保留全部96个teacher-pair/bank观测；图例只解释编码，实验限制进入caption |
| **F5c [E]** | 八条件中每个教师与zero-gradient的Jaccard及逐层随机参照；纵轴对数，按bank显示散点 | 两个checkpoint、两个loss和两输入已合并；随机参照是期望交/期望并之比，不是E[Jaccard] |

等数量支持只在双方非零坐标足够时定义；ties 按幅度降序、全局坐标升序处理。不用零值补足 top1/5/10% 来宣称 learned subnetworks。现有图只使用绝对阈值；等数量敏感性可直接从 pairs.csv 重画，无需重跑模型。

### F6：教师 JS 与更新位置差异（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F6a [E]** | 行=六教师对，列=250/500；两个公共bank的teacher–teacher全词表JS均值 | 不按loss重复JS，因为教师分布距离与后续local loss选择无关；teacher–student JS仍在CSV |
| **F6b [E]** | x=公共bank JS，y=routed的1−Jaccard；颜色=PG/I64、点形=250/500 | 单图48个观测，保留所有checkpoint/loss；不拟合回归或独立样本显著性 |
| **F6c [E]** | 与F6b相同横纵尺度，纵轴为common输入下1−Jaccard；同样合并250/500和PG/I64 | 单图48个观测，便于与routed直接比较；输入控制的原始梯度结果同时见F5b |

### F7：监督目标的局部对照（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F7a [E]** | 行=五个学生状态，列=六种目标/控制；阈值1e-5的BF16变化比例，同条件四bank均值 | 单张矩阵包含所有状态与loss；数值单位10⁻²%，例如3.52表示0.0352%；其他阈值与bank范围保留CSV |
| **F7b [E]** | x=pre-clip gradient L2（symlog），y=BF16写回L2；颜色=目标，点形=学生状态 | 单图包含五状态×六目标×四bank共120个观测；y明确为观测范围的线性局部坐标；zero重合点不抖动伪造数据 |
| **F7c [E]** | 行=五个学生状态，列=五个非full目标；与同bank full-vocab BF16参照的余弦均值 | 不复制不同state图，也不画恒等于1的full/full列；星号mixed history仅在caption解释 |

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

以下信息统一放在FIGURE_CAPTIONS.md和论文caption，不写进图片：本地HF、保存Adam、模拟BF16写回、checkpoint/loss/bank数量、cap与prefix范围、mixed history（适用时）、诊断bank不是训练seed、条件均值不是统计CI。F2c特别注明数值与小heldout限制；F5/F6不写独立教师样本显著性；F7不把不同损失当单纯候选数实验。

下一步是将五组图的描述性观察写入正文，并接入其他节点已有能力/累计扫描产物。不要恢复本轮已取消的新训练、多种子或cap训练。论文正文仍有旧teacher-top64定义与更大实验范围的文字，应按实际完成范围单独修订；本次只更新图、数据和作图文档。

01:10/01:50的历史盘点继续保存在figure_audit_20260908和figures/evidence_audit_20260908，不与当前数据混合。修订前路线图与下一批实验清单已备份到backup/2026-09-08_before_local_probe_figures_221426/experiments/。

独立面板改版前的三联图、脚本和作图文档归档于backup/2026-09-08_before_standalone_panels_224019/，不再由默认脚本生成。build_figure_plan.py只导出图槽清单，不再生成带大段文字的9×3布局图。
