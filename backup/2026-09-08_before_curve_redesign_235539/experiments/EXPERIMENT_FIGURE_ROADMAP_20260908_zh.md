# 论文实验图路线图：按审稿人问题组织证据

更新日期：2026-09-08。本次接入的 31 组本机机制探针于 **20:45 UTC 全部完成并核验**。这里刷新的是局部机制数据；其他节点的在线能力评测、训练进度和累计扫描没有在本次重新盘点，不能把历史缺口视为实时状态。

## 1. 当前交付与执行范围

已将 **20 组监督密度、7 组归一化、4 组教师对照** 落盘并绘制为 **15 张独立实验图**：F1a–c、F2a–c、F5a–c、F6a–c、F7a–c。每张分别导出PDF、PNG、SVG；不再把三个子图拼成一张图片。

本轮按审稿人阅读方式重画：**先确定每图要回答的问题，再选择比较方式**。F5–F7继续合并step250/500和不同loss，但不再使用密集散点或多重颜色/点形编码。F5用分行成对比较与完整观测范围；F6用同序教师对矩阵；F7直接展示固定学生状态内的目标对照。

建议正文按六张独立图阅读：**F1c → F2b → F5a → F5b → F7a → F7b**，对应平均规则、教师输入控制、监督目标三项问题，每项两张。其余九张是补充诊断；不是要求把15张全部放入正文。

- [独立图浏览页](../figures/local_probe_results_20260908/index.html)
- [六张正文候选：一图一页](../figures/local_probe_results_20260908/main_panels.pdf)
- [全部15张：一图一页](../figures/local_probe_results_20260908/all_panels.pdf)
- [逐图问题、阅读顺序与审稿检查](local_probe_results_20260908/REVIEWER_FIGURE_AUDIT_zh.md)
- [各图英文caption草稿](local_probe_results_20260908/FIGURE_CAPTIONS.md)
- [独立数据包与复现说明](local_probe_results_20260908/README_zh.md)
- [31组源数据与SHA256](local_probe_results_20260908/data_manifest.json)
- [每张图的实验及编码清单](local_probe_results_20260908/figure_manifest.json)
- [每个原始观测与数值转换](local_probe_results_20260908/panel_values.json)
- [每格或区间的均值、范围与样本数](local_probe_results_20260908/panel_summaries.json)
- [完整结果解释](local_probe_results_20260908/RESULTS_zh.md)
- [当前27图槽清单](local_probe_results_20260908/figure_plan.csv)

### 绘图规范（后续重画沿用）

1. 一图回答一个具体问题，并在caption与浏览页写出问题。正文候选与补充诊断分开；不为了填满三格而重复同一结果。
2. **每张实验图独立导出PDF、PNG、SVG**；审阅PDF也一页一图。保留大字号：轴标签19pt、刻度16pt、数字15–16pt、成对比较图例15pt。按论文正文宽度放置，避免把整图再缩成三联图的小格。
3. 图内不放标题、子图编号、说明段落或脚注，只保留必要轴、数值及图例。用完整的Task/Shared inputs、Student/Teacher Top64等标签，减少R/C、ST/T等多重缩写。
4. 比较图最多两系列，安排在分开的纵向位置；点为均值，线为全部观测min–max，绝不称作CI。矩阵一格一个指标，以数值与行列标签作为主编码，颜色只表示该指标的大小。
5. 不跨checkpoint/loss/input合并。F5的汇总单位明确为同条件下全部教师对或教师；F6/F7只在固定单元内平均bank。全部观测和转换参数保留供复核。
6. F6a/b/c共用教师对顺序，b/c共用0–0.4色标；不能用密集散点或未经检验的回归线暗示JS与更新位置的稳定关系。
7. F7a直接用百分比。F7b先对每个目标的四bank范数求均值，再除以同状态PG对应范数均值；F7c先逐bank把cosine转换为角度再平均。转换不是新测量，定义写入caption。
8. 矩阵数字检查对比度与文字重叠；成对标记检查重叠；逐张检查导出的图片。保留原始阈值扫描与等数量支持数据，避免为主图强行叠加所有曲线。

**用户已取消从 Base 重新训练交集方法及所有新增训练种子重复。** 当前计划没有新增训练或其关联评测；也不因旧正文曾承诺某实验而自动恢复。线上能力部分只接入已有运行的真实结果。多种子与新增 cap 训练从执行排期移除，F9a/F9b 标为 X。

按 24 个核心图槽计，现为 **15 E / 5 P / 4 N**。E 表示可画有明确范围的探索图；P 表示本地旧盘点有部分数据、本次没有补齐；N 表示本次没有目标测量；X 表示已从当前范围取消。状态不等于最终论文验收，也不是研究完成百分比。F3/F4/F8 的 P/N 沿用保守盘点口径，后续接入其他节点已完成产物时再更新。

## 2. 方法、样本和数值身份

- I64 / Overlap64 是当前学生与教师 Top64 的交集。权重仍按原学生 Top64 概率质量归一化，交集上不再次归一化；advantage detach。ST64、teacher-selected Top64、full-vocabulary 各自有不同目标，不能把它们之间的差异归因于仅改变候选数量。
- 密度探针：M-PG250/500、S-PG250、M-I64-DR250、S-I64-250 各四个 bank（42–45）；每域一条 response，cap256，每条至多四个 prefix。每个 bank 有九个分支，额外 PG action draws 在原始数据/CSV 中保留；主图不把它们当训练重复。
- 归一化：M-PG250/500 各两批 cap512；PG500 一批 cap2048；同一个 quota1/2/3/4 bank 分别用 uniform 与 prompt-share 域先验，共七组。DR/DT/GT/zero 使用保存的同一 Adam 状态。
- 教师：M-PG250/500 各两个 bank（42/43）；每位教师在 routed 与 common 输入下做 PG/I64 分支，加 zero-gradient。每 bank 六个共享教师对，不把重复 pair 当独立样本。
- m-i64dr250/s-i64250 含旧 Student64 训练历史，状态名加星号，并在caption解释 mixed history；不是全程交集训练轨迹。
- 全部新图测的是 HF 局部梯度和保存 Adam 下的模拟 BF16 写回；不是历史 Megatron/SGLang 的精确重放、真实在线单步或能力评测。累计变化与局部写回不能混画成同一量。
- F5a/b每条件汇总六教师对×两个bank，F5c汇总四教师×两个bank，均值和全观测min–max均可复核。F6/F7矩阵平均同条件bank。所有范围都不解释为训练seed方差或95% CI；共享教师的pair不是独立样本。

## 3. 已生成的图与主要读数

| 图组 | 独立PDF文件 | 如何回答问题 |
|---|---|---|
| F1 | [F1a](../figures/local_probe_results_20260908/F1a.pdf) · [F1b](../figures/local_probe_results_20260908/F1b.pdf) · [F1c](../figures/local_probe_results_20260908/F1c.pdf) | 回答长度逐条入格；token份额直接减prompt参照；三种规则的每response系数并列。 |
| F2 | [F2a](../figures/local_probe_results_20260908/F2a.pdf) · [F2b](../figures/local_probe_results_20260908/F2b.pdf) · [F2c](../figures/local_probe_results_20260908/F2c.pdf) | 同序七条件矩阵；修正量→梯度方向→超过zero对照的局部KL变化。 |
| F5 | [F5a](../figures/local_probe_results_20260908/F5a.pdf) · [F5b](../figures/local_probe_results_20260908/F5b.pdf) · [F5c](../figures/local_probe_results_20260908/F5c.pdf) | a直接比较Task/Shared输入；b直接比较梯度/BF16方向；c比较zero重叠与随机参照。 |
| F6 | [F6a](../figures/local_probe_results_20260908/F6a.pdf) · [F6b](../figures/local_probe_results_20260908/F6b.pdf) · [F6c](../figures/local_probe_results_20260908/F6c.pdf) | 同序六教师对矩阵；a为JS，b/c为两输入的1−Jaccard；全部checkpoint/loss保留。 |
| F7 | [F7a](../figures/local_probe_results_20260908/F7a.pdf) · [F7b](../figures/local_probe_results_20260908/F7b.pdf) · [F7c](../figures/local_probe_results_20260908/F7c.pdf) | 同一学生状态内比较：参数变化百分比、相对PG的两种范数、相对full的方向角。 |

**图槽语义已经更新**：F5a现在是输入对照的梯度余弦，原逐教师对support数据由F6b/c承接；F2c改为相对zero的差值；F7b改为相对PG范数；F7c改为角度。旧图不能与新caption混用。所有原始cosine、范数、KL、Jaccard、阈值数据保持不变。

可写入结果段落的观察：

1. 四批教师探针中，routed 原始梯度余弦均值为 PG **0.0057**、I64 **−0.0032**；common 分别为 **0.9739、0.9630**。输入是否匹配是解释教师方向差异的关键控制。不能把 routed 的近正交解释成教师独有、互不重叠的参数子网络。
2. 七组归一化的长度–梯度分解最大相对残差约 **1.8e-16**。长回答批次 DR–DT 原始梯度余弦 **0.7063**、DR–GT **0.6583**，但局部 held-out KL 排名并不一致。
3. M-PG500 的四 bank 均值：PG/I64 的原始梯度 L2 为 **52.08/39.44**，BF16 写回 L2 为 **0.03385/0.03395**，精确非零比例为 **0.1699%/0.1712%**。更多监督候选没有对应明显更大的写回范围；不能由范数相近推断方向或能力等价。
4. zero-gradient Adam 对照仍有较大 BF16 写回。额外不执行 Adam 的全坐标检查确认 PG250/500 的 BF16(master) 与保存模型完全一致，排除了这两个检查点已有 master/model 不一致的解释。记录见 no_step_cast_audit.json。

**F2c 只作探索性诊断。** 每域只有一条 held-out response；两组中 zero-gradient 的 KL 下降超过三个 reduction。相同不等配额 bank 的 uniform/prompt 两次运行，其 GT 梯度范数仍有小幅数值差异（6.173284587 vs 6.171707070），KL 读数也有差异，来源尚未定位。因此图中展示全部正负读数，不做规则优劣排序、因果能力结论或显著性声明。

## 4. 当前图槽清单（编号分组，不表示必须三联排版）

### F1：Token balancing——长度与实际权重（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F1a [E]** | 八个domain/response行 × 四个checkpoint/bank列，逐条标出长度；不再抖动散点 | 512-token值对应截断，caption限定局部cap512样本；补充诊断 |
| **F1b [E]** | 四域 × 四bank的token share减25% prompt share，单位百分点，正负色标以0为中心 | 重复数值仍各占一格，不再覆盖Math/Science曲线；补充诊断 |
| **F1c [E]** | 十条response × DR/DT/GT权重百分比矩阵，域分组、规则全称直接标出 | 正文候选；同一quota1/2/3/4、uniform先验bank；展示域内和域间权重区别 |

### F2：Token balancing——局部梯度与 KL（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F2a [E]** | 七条件 × 四域的长度–梯度修正L2矩阵，保留零值，统一条件顺序 | 补充诊断；展示修正来源，代数恒等式不替代能力 |
| **F2b [E]** | 七条件 × 三对reduction的原始梯度cosine矩阵，0–1固定尺度 | 正文候选；每行固定同一学生与响应，长回答方向差异直接可读 |
| **F2c [E]** | 七条件 × DR/DT/GT，减去同job zero-gradient的macro KL下降，单位millinats | 补充诊断；正值才超过zero；小heldout和GT数值差异禁止规则排名 |

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
| **F5a [E]** | 四行checkpoint/loss × Task/Shared两输入的梯度cosine成对图 | 正文候选；每点六教师对×两bank均值，线为全观测min–max，不是CI |
| **F5b [E]** | 八行checkpoint/loss/input × raw gradient/BF16 update两种cosine成对图 | 正文候选；两系列分开位置，均值和全范围；保留全部96成对观测的两指标 |
| **F5c [E]** | 八条件的zero-gradient Jaccard与逐层随机参照，横轴对数的成对图 | 补充诊断；每点四教师×两bank，含全范围；随机参照是期望交/期望并 |

等数量支持只在双方非零坐标足够时定义；ties 按幅度降序、全局坐标升序处理。不用零值补足 top1/5/10% 来宣称 learned subnetworks。现有图只使用绝对阈值；等数量敏感性可直接从 pairs.csv 重画，无需重跑模型。

### F6：教师 JS 与更新位置逐对核对（E/E/E，补充诊断）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F6a [E]** | 六教师对 × 两checkpoint的全词表JS均值，按四bank平均JS仅作行排序 | 补充诊断；不复制loss维度；为F6b/c提供相同教师对顺序 |
| **F6b [E]** | 同序六教师对 × 四checkpoint/loss的task输入1−Jaccard矩阵 | 补充诊断；两bank单元均值，和F6c共用0–0.4尺度；不拟合回归 |
| **F6c [E]** | 同序六教师对 × 四checkpoint/loss的shared输入1−Jaccard矩阵 | 补充诊断；完整输入控制，行列与色标和F6b一致 |

### F7：监督目标的范围、尺度与方向（E/E/E）

| 子图 | 画什么、数据点如何定义 | 当前证据与范围 |
|---|---|---|
| **F7a [E]** | 五学生状态 × 六目标的阈值1e-5变化参数百分比，四bank单元均值 | 正文候选；直接显示0.0352%等数值，横向比较固定状态内的目标 |
| **F7b [E]** | 每学生状态Gradient/Update两行 × 五个非PG目标，显示各自四bank范数均值/PG对应均值 | 正文候选；PG=1明确参照；zero保留；不把范数比例误解为方向或能力 |
| **F7c [E]** | 五学生状态 × 五个非full目标，与full BF16写回的方向角（度） | 补充诊断；逐bank arccos后平均，不是mean cosine再arccos；mixed历史保留 |

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
python experiments/validate_reviewer_figures.py
```

数据包校验了31个measurements及图所需原始bank、权重、support、manifest、完成标记的SHA256。重新绘图会校验包内全部文件；plot_data各行保留原始job、state与相对source路径。全层测量在raw的measurements.json，未把大概率tensor复制到论文目录，因为当前图不依赖它。

以下信息统一放在FIGURE_CAPTIONS.md和论文caption，不写进图片：本地HF、保存Adam、模拟BF16写回、checkpoint/loss/bank数量、cap与prefix范围、mixed history（适用时）、诊断bank不是训练seed、条件均值不是统计CI。F2c特别注明数值与小heldout限制；F5/F6不写独立教师样本显著性；F7不把不同损失当单纯候选数实验。

下一步是将五组图的描述性观察写入正文，并接入其他节点已有能力/累计扫描产物。不要恢复本轮已取消的新训练、多种子或cap训练。论文正文仍有旧teacher-top64定义与更大实验范围的文字，应按实际完成范围单独修订；本次只更新图、数据和作图文档。

01:10/01:50的历史盘点继续保存在figure_audit_20260908和figures/evidence_audit_20260908，不与当前数据混合。修订前路线图与下一批实验清单已备份到backup/2026-09-08_before_local_probe_figures_221426/experiments/。

独立面板改版前的三联图、脚本和作图文档归档于backup/2026-09-08_before_standalone_panels_224019/，不再由默认脚本生成。build_figure_plan.py只导出图槽清单，不再生成带大段文字的9×3布局图。

本轮审稿视角重画前的版本归档于backup/2026-09-08_before_reviewer_redesign_225743/。当前数字转换、正文选择理由和逐图问题见REVIEWER_FIGURE_AUDIT_zh.md；validation_report.json记录源数值与导出文件检查。
