# 论文实验盘点与 8×3 / 9×3 作图路线图

盘点日期：2026-09-08。已完成结果快照为 **01:10 UTC**，训练进度复核于 **01:50 UTC**；本文数字不是实时看板。

对象是本仓库 `Optimization-Dynamics-in-Multi-Task-LLM-Post-Training` 的**现行正文**：Token Balancing / Update Sparsity and Teacher Overlap / Supervision Density 三项研究，而非旧 GPAS 方法论文。本文的“24–27 张”指 **8–9 个 figure，每个 figure 一行 (a)(b)(c) 三个子图**；不意味着主文一页容纳全部图。

## 1. 离目标还有多远

**已经有足够的数据做一批有实质内容的探索图；还没有一套口径一致、可以完整支撑三条主线的 24 图证据。最主要的缺口是 TopK 分支能力评估、与当前 overlap loss 对齐的局部诊断、真实在线 BF16 单步更新，以及配对训练种子。**

按下文明确列出的 24 个核心子图计数：

| 子图状态 | 数量 | 含义 |
|---|---:|---|
| E：可画限定范围的探索图 | **12/24** | 本地已有该测量的完整数值；可在图注限定 checkpoint、loss、batch 和 seed 后作图；不等于正式比较矩阵或统计重复齐全 |
| P：有部分数据点 | **8/24** | 可以画部分系列，但目标对照仍缺；不能把空缺曲线补出来 |
| N：缺核心目标数据 | **4/24** | 三种 reduction 的在线能力图 F3a/b/c，以及真实在线 BF16 单步曲线 F8a |
| 第九行扩展 | **另加 3 个 N** | 配对种子效应、长度 cap、batch-size 敏感性；其中验证工作是否必需取决于正文保留的论断，不能因不画第九行就一并省略 |

这里的 12/8/4 是**下文图槽的可用性计数**，不是把整体研究估成完成了 50% 或 83%。按本文完整验收条件，24 个核心图槽均尚未达到最终验收。相互复用的同一批数据也不能按图数算成独立证据。

本轮已交付 **3 行×3 子图的真实数据预览**：[三页 PDF](../figures/evidence_audit_20260908/exploratory_triptychs.pdf)。它们用于核查布局、测量和论点，不冒充最终 24 张图。具体对应 F1b/F2c/F8b、F4a/F4c/F7a、F5a/F5c/F6b-c 的现有部分。

当前排期围绕 **M-PG、S-PG、M-Overlap64-DR/DT/GT、S-Overlap64 六个实验组**：完成已有训练，优先接上三组 overlap 的250步四域评估，利用现成 checkpoint 补归一化、teacher overlap 和监督密度诊断，再安排新增配对种子。当前清单不包含额外四条seed42主训练。

具体到下一批任务、输入路径、程序缺口与本机资源，见 [接下来要跑的实验](NEXT_EXPERIMENTS_20260908_zh.md)。实验组名称用于组织当前工作；每个测量点的方法身份以checkpoint记录为准。

## 2. 本次核验范围与可追溯产物

本次读取论文当前 `.tex`、最近的实验说明、共享盘上的训练日志、provenance、checkpoint 索引、完成标记和逐题评估结果。没有启动/停止训练、调用 teacher、修改原始 run，或重新核实远端进程；“未完成”指快照时没有完整完成证据。

- [机器可读快照](figure_audit_20260908/evidence_snapshot.json)：14 个相关训练目录、21 个 BF16 扫描报告、4 个归一化 probe、5 个局部 BF16 probe、72 行 teacher-pair 记录，以及 209 个已读取源文件的 SHA256。另见 [27图槽 CSV](figure_audit_20260908/figure_plan.csv) 和 [9×3布局图](../figures/evidence_audit_20260908/figure_roadmap_9x3.pdf)。
- [训练台账 CSV](figure_audit_20260908/run_inventory.csv)、[能力台账 CSV](figure_audit_20260908/capability_inventory.csv)、[纠正标签后的 BF16 台账 CSV](figure_audit_20260908/bf16_inventory.csv)。14 个目录包含关联记录，**不是 14 个独立实验**。
- 25 个已完成的“模型/checkpoint×benchmark”结果点通过逐题核验，共 **10,940 条回答**：初始学生 4 点、4 位教师各本域 4 点、训练后 PG 17 点。核验包括完成标记、artifact SHA256、题数/采样数、step、终止状态和逐题 reward 与汇总分数一致性；不是重新运行评分器。
- 评估结果点未发现上述一致性检查失败。21个扫描的L2与对应原训练日志独立比对，最大相对差约1.36e-13；72行教师对的交并计数与Jaccard一致。详见 [验收记录](figure_audit_20260908/validation_report.json)。
- 可复现入口：[audit_figure_evidence.py](audit_figure_evidence.py)、[plot_figure_evidence.py](plot_figure_evidence.py)。刷新快照后需同步本文表格；不要静默覆盖本次历史结论。

旧的 [9 月 6 日 Top16 盘点](STUDENT_TOP16_EXPERIMENT_AUDIT_20260906_zh.md) 和 [工期估计](STUDENT_TOP16_COVERAGE_AND_RUNTIME_20260906_zh.md) 是历史快照，不能再作为当前任务未启动/未完成的依据。当前任务进度以本次复核为准。

## 3. 当前实验组与测量口径

当前四个TopK实验组统一称为 **teacher–student Top64 overlap**，简称 **Overlap64 / I64**；三组多教师分别使用DR、DT、GT归一化，单教师组使用math teacher。图表中的当前实验组名称不替代每个checkpoint的原始测量身份。

记冻结的rollout student选集为S，teacher Top64为T，I=S∩T。当前局部surrogate是：

`L = -Σ(v∈I) stopgrad[ p(v)/Σ(u∈S)p(u) × (log q(v)-log p(v)) ] × log p(v)`。

仅保留交集内项；**分母仍是原student Top64的概率质量，不在交集上重新归一化**。空交集贡献零，仍保留原位置/response的外层分母。

图和原始记录保留 `candidate_source、K、intersection_size、empty_fraction、student_mass、teacher_mass、retained_weight、normalization_support、advantage_detach、clipping、outer_reduction`。I64实际选中数并非恒为64，surrogate/log-ratio可为负，不应简单标成KL loss。

局部诊断中的student-support Top64（ST64）、teacher-selected Top64（T64）与full-vocabulary是各自有明确公式的参照。现有ST64局部结果不能直接替代尚未测量的I64分支；先补I64，再作当前loss的结论。具体数据身份保存在原始provenance和机器可读台账。

## 4. 现在到底有哪些实验数据

### 4.1 当前六个实验组

共同设置为 Qwen3-1.7B Base、四位同架构RL专家、seed42、每次更新64 responses、目标500更新、4096 response cap。多域每域16 responses；单域math为64。Adam的lr=2.5e-7、betas=(0.9,0.98)、eps=1e-8、clip=1、weight_decay=0，具体配置以run manifest为准。

| 实验组 | 已记录更新 / 500 | 当前目录可用HF checkpoint | 下一项依赖 |
|---|---:|---|---|
| M-PG | **500** | 1/50/100/250/500 | 已完成；复用250/500机制诊断 |
| S-PG | **313** | 1/50/100/250 | 完成训练；导出500并评估 |
| M-Overlap64-DR | **235** | 250/500待导出 | 导出250，立即做四域评估与BF16扫描 |
| M-Overlap64-DT | **246** | 250/500待导出 | 导出250，立即做四域评估与BF16扫描 |
| M-Overlap64-GT | **243** | 250/500待导出 | 导出250，立即做四域评估与BF16扫描 |
| S-Overlap64 | **169** | 250/500待导出 | 导出250，立即做四域评估与BF16扫描 |

训练日志步数不等于完整checkpoint。加载前核对`mopd_checkpoint_index.json`、HF索引及全部shards；局部更新另外需要保存的参数/Adam snapshot。进度来源见 [01:50复核快照](figure_audit_20260908/next_experiment_progress.json)。汇总成本时按run与update去重，不累计重复记录。

### 4.2 已完成能力评估

所有分数为百分比；`—` 表示未拿到完整并通过核验的数据，不能补零。MATH-500/LCB/IF 分别为 500/128/300 题；GPQA 为 198 题×4 回答，平均正确率，**不是 pass@4**。

| 学生/参考 | 步数 | MATH-500 | LCB | IFBench strict | GPQA avg@4 | 四域 macro |
|---|---:|---:|---:|---:|---:|---:|
| 初始学生 | 0 | 55.60 | 0.78 | 17.33 | 18.81 | 23.13 |
| M-PG | 100 | 55.40 | 9.38 | — | 23.23 | — |
| M-PG | 250 | 66.20 | 18.75 | 11.67 | 24.75 | 30.34 |
| M-PG | 500 | 66.40 | 13.28 | 14.67 | 24.87 | 29.81 |
| S-PG（只训练 math） | 100 | — | 11.72 | — | 24.24 | — |
| S-PG（只训练 math） | 250 | 66.20 | 21.09 | 12.33 | 20.58 | 30.05 |
| 四个不同教师，各自本域 | — | 75.60 | 23.44 | 29.00 | 34.60 | 不定义为一个模型的 macro |

M-PG/100、S-PG/100 的 math/IF 完整重试在 9 月 8 日启动；[核验状态](../../slime_opd_geometry/outputs/mopd_qwen3/pg100_math_if_retry_20260908/verified_summary.json) 在快照时未完成。M-PG/100 已有前一次完整 math 结果可用；重试不是额外训练 seed，也不要择优挑选分数。

可以如实写的现象：M-PG/500 相对初始数学 +10.80pp，code +12.50pp，IF **−2.67pp**，science +6.06pp；macro 相对 250 下降约 **0.54pp**。M-PG/250 的 worst-domain change 是 IF 的 −5.67pp。一个 seed 不能说明差异稳定，也不能据 S-PG/250 code 较高就认定多教师发生了因果干扰：单/多教师 math 暴露差四倍。

评估截断必须与能力同时检查：M-PG/250→500，LCB 截断 **6.25%→43.75%**，IF **29.33%→74.33%**，math **7.0%→17.2%**，GPQA **0.38%→16.16%**。这与能力变化并列展示，不能把截断响应从分母删掉。教师与初始学生的 prompt suffix 不同，所以教师表是各自部署格式的参考，不是纯权重差异或可共同达到的上界。

### 4.3 BF16 累计变化与局部测量

主指标统一为保存的 BF16 值转换到 FP32 后相减，统计 `|BF16(theta_t)-BF16(theta_0)| > 1e-5`；这样只是计算保存值之间的差，**不会恢复未保存的 FP32 变化**。同时保留 exact nonzero、L2、能量集中度及逐层统计，不以单个阈值替代整套结论。

| 实际配方 | checkpoint | 有效变化参数比例 | 累计 L2 |
|---|---:|---:|---:|
| M-PG | 100 / 250 / 500 | 1.782% / 5.900% / 10.682% | 0.104 / 0.269 / 0.531 |
| S-PG | 100 / 250 | 2.554% / 8.105% | 0.125 / 0.375 |
| ST64 单教师 | 100 | 7.593% | 0.256 |
| ST64 多教师 DR | 100 | 6.669% | 0.233 |
| ST64 多教师 DT | 100 | 7.739% | 0.264 |
| ST64 多教师 GT | 100 | 7.852% | 0.267 |

新的 20 次批量扫描覆盖六组的早期点及 PG/250；另有 M-PG/500，共 21 个报告。训练 `paper/measurements.jsonl` 还保存 BF16 累计能量 top-1/5/10% 和达到 90% 能量所需的参数比例。它们不等同于全分位连续曲线。9 月 6 日旧 T64 的漂移/稀疏性报告继续保留为历史材料，不能混到本表 ST64 系列。

| 机制数据 | 真实完成范围 | 仍不能由此说明的事情 |
|---|---|---|
| normalization | M-PG 的 100/250/500 状态使用 T64 loss；500 另用 I64，合计 4 个完整 probe | 没有 I64 在线学生各时点的归一化结果；没有不等 quota/跨 worker 分母实测 |
| normalization batch | 每域 2 条完整 response，共 8 条；每域 1 条 held-out，共 4 条；cap=512，覆盖这些 response 的全部有效位置 | 不是 64-response、cap=4096 训练 batch；500 的 8 条里 6 条截断，不能作为自然长度分布 |
| S-PG/250、M-PG/250 局部监督 | 每域 1 response，cap128，每 response 抽 2 prefixes；PG / ST64 / full-vocab / 零梯度，共 2 个完整 probe | ST64 不是 I64；只抽位置，不是完整 response 梯度；每状态只有 1 bank |
| M-PG/500 教师 overlap | 3 个诊断 bank（draw42/43/44），每域 1 response、cap256、每条至多 4 prefixes；routed/common 两模式，四教师同状态独立分支 | 3 个 draw 全部来自训练 seed42；不是 3 个训练种子；只有一个训练后 checkpoint 的四教师完整比较 |
| teacher JS | 同一公共 bank 上全词表 teacher–teacher JS、teacher–student JS；4 教师有 6 pairs | 72 CSV 行=3 banks×6 pairs×2 modes×2 thresholds，**不是 72 个独立教师对** |
| 真实在线单步更新 | 当前 logger 的 `quantity="update"` 记录 FP32 optimizer/master 差 | 未记录完整 BF16 单步序列；局部分支写回和跨 checkpoint 差都不能冒充原轨迹单步 |

最后一项可在 [paper_measurements.py](../../slime_opd_geometry/slime_plugins/mopd/paper_measurements.py) 的 `after_step()` 直接核对：`update` 使用 `before["fp32"]`，`delta_bf16` 使用初始 BF16。新测量应另记 `online_update_bf16 = model_bf16_after - model_bf16_before`，保留旧字段身份。

已经值得作图但不宜过度解释的机制现象：

- 4 个归一化 probe 的 length–gradient 分解得到接近浮点舍入误差的残差。这验证了对这些实测 response gradients 的有限 batch 恒等式，**不自动验证分布式训练 reducer 实现正确或能力提升**。
- M-PG/500 的 I64 局部 probe 中，DR/DT/GT 的 raw-gradient cosine 为 0.9665 / 0.9524 / 0.9925（DR–DT / DR–GT / DT–GT）；三种 averaging 确实改变该 batch 的梯度，但 held-out KL 的变化很小，也有域内退步。
- 教师 BF16 support（阈值 1e-5）Jaccard：routed 的 18 次配对范围 **0.668–0.900**，common 为 **0.827–0.996**。零梯度分支与 routed 教师分支的 Jaccard 也达 **0.747–0.998**；当前 weight decay=0，需特别解释共享 Adam 历史和 BF16 舍入的影响。不能把这些高 overlap 直接称为独立子网。
- M-PG/250 的局部 BF16 有效变化比例：零梯度 0.03515%，PG 0.03643%，ST64 0.03681%，full 0.03685%。小差值目前不支持“监督越密，更新必然更稠密”的稳定结论。
- S-PG/250 小 probe 的 PG raw gradient 约 99.94% 坐标精确非零，但没有坐标超过 1e-5，L2 仅约 6.68e-5；其 BF16 写回与零梯度控制相同。这里是小幅度信号，不应称为“原始梯度全零”或直接发现了稀疏子网。

### 4.4 其他已有数据的归属

| 历史/辅助数据 | 建议用途 |
|---|---|
| `experiments/topk_drift_sparsity_20260906.json`、`bf16_effective_changes_20260906.json` | T64 的 retained mass 漂移、累计变化；可选配方稳定性附录，明确与 ST64/I64 区别 |
| `outputs/mopd_core/`、`outputs/mopd_gpas*`、`mopd_precision*` | 旧 GPAS/D3/raw-noise 等工作；不计入这三项研究的完成度，也不要求为凑图重跑 |
| `outputs/weight_spectra/`、`raw_gradient_interference/`、早期 single-task RL/OPD | 可检查优化器/教师训练背景；先核对模型、teacher、数据、暴露及精度，不能替代共同 joint checkpoint 的教师分支 |
| `outputs/mopd_smollm3/implementation-m-pg-v3-s42` | 实现 smoke 产物；尚无第二模型的完整六配置证据矩阵 |
| 现有 `figures/gpas_main_figure.*` | 历史示意图，不是当前论文实验结果；不计入 24 个实验子图 |

## 5. 27 个子图逐张设计

图号 F1–F8 是核心 8 行；F9 是可选第 9 行。颜色固定：domain 用四种颜色；同一方法跨图用同一线型；ST64 与 I64 必须写全，实际测量配方以checkpoint记录标注。最终共享 checkpoint 建议 0/50/250/500；能力基础点 0/250/500，可负担时补 100。只有两三个能力点时使用点线图，不能平滑成“密集训练曲线”。

**所有单点的基本身份**：`run_id + training_seed + checkpoint_hash + loss_version + domain + diagnostic_bank/eval_prompt_id`。局部 proposed step、在线 step、累计 delta 在图标题和纵轴都要区分。

### F1：Token balancing——等 prompt 不等 token，也不等梯度（E/E/E）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F1a [E]** | 四域 response mean length 对 optimizer step 的曲线；每个点为一个训练 batch 的域内均值，附截断率。需要分布时另存逐 response 长度，不能由均值伪造 violin | M-PG 500 步的 `metrics/rollout.jsonl: mopd/task/*/mean_response_length,truncation_rate` 可画。补最终 TopK 三 reduction 同口径曲线 |
| **F1b [E]** | 四域有效 token share 对 step，25% prompt-share 参考线；按窗口汇总 token 数后再求比例 | 已有 M-PG 完整日志。图说明 GT 隐含域权重，并不称为 M-PG 实际 DR 权重。不能套用他文“IF 只占 1%”：本运行 IF 的 token share 在前50/后50更新分别为34.96%/31.23% |
| **F1c [E]** | 固定同 batch，x=每 response 有效长度，y=该 response 的总 loss 系数；DR/DT/GT 三符号、domain 四色 | 现有 `weights.json` 给出 8 个 response×3 reductions=24 点。补真实 4096 cap batch、不等 quota 的 uniform λ 与 prompt-share λ 两个控制；记录所有有效分母 |

要支撑的论点来自 [005_normalization.tex](../sections/005_normalization.tex)：改变外层 averaging 改变域权重和域内长度权重；不是预设某域一定被低估。

### F2：Token balancing——从代数权重到局部梯度和 loss（E/E/E）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F2a [E]** | x=domain/checkpoint，y=‖Cov(T,g)‖/mean(T)；附分解相对残差。每个点由同域完整 response gradients 计算 | 当前 4 probes×4 domains=16 组；相同长度的 covariance=0 要保留。补 I64 早/中/晚及至少 4 独立 batch；残差仅作数值审计 |
| **F2b [E]** | 三 reduction 的 3×3 raw-gradient cosine 热图，旁注各自 pre-clip norm、clip coefficient | 现有 4 个 probe；500 的 I64 是可直接画的探索版。补相同 loss/学生状态/优化器协议的时点，不把 raw-gradient cosine 写成更新 cosine |
| **F2c [E]** | 行=zero-grad/DR/DT/GT，列=四域，颜色=`KL_before−KL_after`；公共 held-out bank 的全词表 reverse KL，正值代表该次局部下降 | 现有每 probe 4×4=16 单元格，macro 另存；预览使用 I64/500。每域现只有 1 held-out response；补更大 held-out bank、重复 batch 和评分重复性。不能据此代替 benchmark accuracy |

要支撑的论点来自正文固定-batch experiment 与 [009_appendix.tex](../sections/009_appendix.tex) 的 finite-batch identity。跨 microbatch/worker 的 sums/denominators 另作审计表，不用“残差接近零”替它背书。

### F3：Token balancing——在线能力是否更均衡（N/N/N）

| 子图 | 画什么、数据点如何定义 | 缺什么实验 |
|---|---|---|
| **F3a [N]** | 三 reductions×四域的能力变化 heatmap；主视图 500，250 可另分面；每格 `score(t)−score(0)` pp，训练种子汇总 | 当前没有TopK的完整能力评估。优先完成三个Overlap64实验组的250/500四域评估；保持同一评估协议，报告其实际得分与变化，不预设归一化优劣 |
| **F3b [N]** | x=累计 student generated tokens，y=四域 macro；三 reduction 点线与配对种子不确定性。每个能力 checkpoint 对应日志累计暴露 | 不用 mean surrogate 代替 capability。至少 0/250/500；可选补 100；不同方法按实际 tokens 标点，不按 64 responses 假定 tokens 相同 |
| **F3c [N]** | x=updates，y=`min_d(score_d(t)−score_d(0))` pp；图旁/附表列当时最差 domain 和截断率 | 同 F3a 的评估即可计算，无需另跑模型。零参考线；不存在已证明的均衡优势时报告零/负结果。不要用“最低绝对分数”替代 worst-domain change |

这行是第一条论文主线最实质的空缺。固定 batch 梯度和 KL 不足以替代它。

### F4：Update sparsity——累计变化是否集中（P/P/P）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F4a [P]** | x=0/1/50/100/250/500，y=BF16 有效累计变化参数比例；S-PG/S-I64/M-PG/M-I64-DR 四主系列，DT/GT 附加；另提供域内 responses/tokens 坐标版 | PG点较全，已有早期扫描按其记录的方法标注。补Overlap64的250/500 checkpoint扫描。主图阈值 1e-5，另给 exact nonzero |
| **F4b [P]** | x=按累计 delta 绝对值排序的参数比例 ρ，y=累计 squared-L2 energy；各主系列比较，附 energy90 fraction 和总 L2 | 已有日志支持 ρ=1/5/10% 三点及 energy90；若要连续曲线，重新流式扫描 checkpoint。能量全为零的点记 undefined，不能画任意 concentration |
| **F4c [P]** | 行=方法/检查点，列=28 transformer layers＋embedding/head；颜色=逐层有效变化比例，配套保存逐层 L2/能量 | 21 个扫描报告有逐层数值；预览是六系列 ST64/PG 的 step100。补最终 I64 250/500 和不同 seeds；全模型比例必须按参数数加权，不能平均层百分比 |

对应 [006_teacher_subnetworks.tex](../sections/006_teacher_subnetworks.tex) 第一问与 [007_supervision_density.tex](../sections/007_supervision_density.tex)。这些图测的是累计变化，不是参数级梯度稀疏性或历史 teacher attribution。

### F5：Teacher overlap——同一个学生里，教师改变哪些位置（E/E/E）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F5a [E]** | 同一个 joint checkpoint 的 4×4 BF16 proposed-step support Jaccard 热图；主阈值 1e-5，列明各教师 support 大小 | M-PG/500 三 banks、routed/common 已有；探索主视图可先用 routed。补 0/250、最终 I64 joint 状态，另补可行的匹配选择数 support |
| **F5b [E]** | x=同一教师对 Jaccard，y=两分支 signed BF16 update cosine，颜色=教师对，符号=routed/common | 现有 6 pairs×3 banks×2 modes=36 点/阈值。补其它 checkpoint；共同位置≠同方向，低 overlap 也不默认更好 |
| **F5c [E]** | 各教师分支与 zero-gradient 分支的 overlap；并列同层匹配随机参照及 support counts，观察共享 Adam 状态影响 | 当前 zero-gradient 和 layer-random ratio-of-expectations 已有。随机参照必须标成“期望交/期望并之比”，不是实测 E[Jaccard]；需要 CI 时另采样随机 masks。补按层/权重幅度分箱参照 |

**选择数控制要适应 BF16 的实际稀疏性。** 本次单步有效变化只有约 0.04% 的参数；机械选择 top-1/5/10% 会包含大量零值。support overlap 的等预算检查建议先试全参数比例 ρ=0.01/0.05/0.1%，且 ρ 不超过所有被比较分支的精确非零比例；任何层/分支不满足时标 undefined 或降低共同预算。top-1/5/10% 可用于**能量**曲线，但不能将用零值补齐的集合称为 learned subnetwork。记录阈值、ties、实际选中数和空集合规则。

### F6：Teacher distance——分布差异是否对应参数位置差异（E/E/E）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F6a [E]** | 4×4 teacher–teacher JS 热图；同一冻结公共 prefix bank、四域等权、全词表、自然对数单位 nats | M-PG/500 三 banks 的六对 JS 已有；不能用各域不同 prefixes 的 teacher KL 拼矩阵。JS 取值范围 [0,ln2]；teacher–student JS 单列辅助表 |
| **F6b [E]** | x=公共 bank teacher JS，y=routed BF16 `1−Jaccard`；六对不同颜色，显示原始 bank 点，主点为各对均值 | 当前有 18 routed 点，允许画不带回归显著性的探索散点。补同协议 0/250/500，用线连接同一教师对，而非把所有重复点当独立样本拟合 |
| **F6c [E]** | 与 F6b 同尺度，但纵轴取完全相同 prefixes、相同 sampled actions 的 common 模式；与 routed 成对连线/比较 | 当前 18 common 点齐。测试去掉输入上下文差异后关联是否还存在；补幅度、zero-gradient 控制与 I64 状态。观察不到关联也是有效结果 |

教师固定时，在**完全不变的全局 prefix bank**上 teacher–teacher JS 不随 student checkpoint 改变；checkpoint-specific bank 上的变化来自 student 生成上下文变化。作图须记录 `bank_id`，不能把这种 JS 轨迹称为“教师权重随训练变了”。四位教师仅支持本设置的描述性关联。

### F7：Supervision density——同一状态与前缀下的局部对照（P/P/P）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F7a [P]** | x=BF16 单步阈值 τ（exact 0、1e-8/1e-7/1e-6/1e-5），y=超过阈值的参数比例；PG/ST64/I64/full 和 zero-grad | 已有 S/M-PG250 与 M-PG500 的 PG/ST64/full；缺 I64 loss 分支及 I64 学生状态。PG tokens 在同一 prefixes 重新采样；保持 prefix weights 不依赖本次 action |
| **F7b [P]** | 同一点并列 raw-gradient L2、BF16 proposed-step L2、有效变化比例；用分组点图或明确分开的坐标，避免把不同单位直接比大小 | 现有 5 probes 可提供 PG/ST64/full。补 I64 和更多 banks；保留小梯度但非零的 S-PG250 示例，以及相同 optimizer 的零梯度分支 |
| **F7c [P]** | x=loss，y=与 full-vocab BF16 proposed step 的 cosine/支持差异；single/joint 状态分组 | 当前 pairwise writebacks 有 PG/ST64/full 关系，缺 I64。至少 0/50/250/500、每点 4 banks，后期覆盖 S-PG/S-I64/M-PG/M-I64-DR 四轨迹；同一 loss 比较内部固定 student、Adam 和 prefixes |

这行检验**实际 loss 选择**，不能将 不同loss的支持规则差异归因于“只改变 vocabulary 数量”。full-vocabulary 是局部参考，未做其在线训练，就不能画 full-vocab 在线能力或累计变化曲线。

### F8：把更新与真实能力、训练暴露和成本连起来（N/P/P）

| 子图 | 画什么、数据点如何定义 | 现有证据与后续数据 |
|---|---|---|
| **F8a [N]** | x=真实训练更新 step，y=BF16 实际单步有效变化比例/能量；与 F4 累计量分开；0→1、50、250、500 附近保存相邻 BF16 写回 | 当前 `update` 是 FP32，缺目标 BF16 序列。训练 logger 在 optimizer 前后记录 BF16；每个窗口建议 5 个连续更新。局部分支不能填空；对已结束 PG，只有用真实 batch、RNG、optimizer、clip 等精确重放并验证轨迹才可能补原始单步 |
| **F8b [P]** | PG 与最终 TopK 的能力比较：单教师本域 math、joint 四域变化；0/250/500，配对 seeds；单域与多域另给匹配 math exposure 的参照 | PG 的已有四域表可画部分版，TopK 能力全缺。S-step125 与 M-step500 的 math responses 同为 8000；要做精确比较应预存对应点，不能把两者 step500 当成相同 math 暴露 |
| **F8c [P]** | x=累计 student tokens，y=macro capability，点大小/附表=GPU hours，曲线=PG/TopK；标注最差域变化避免均值掩盖退步 | 现有 M-PG token/时间日志与 0/250/500 能力可画单系列；缺 TopK/重复种子。报告 learner/rollout/teacher/诊断/评估开销，按节点硬件分开；共享 teacher 的时间不可在每条 run 中全额重复记账 |

F3 比较 **reduction**，F8 比较 **loss 与 single/joint 设置**，避免三行都只是重复一张 accuracy 图。

### F9：可选第九行——结论的适用范围（N/N/N）

| 子图 | 画什么、数据点如何定义 | 实验要求 |
|---|---|---|
| **F9a [N]** | 三个配对 training seeds 的效应森林图：PG–I64 的 BF16 变化差、macro 差；DR–DT/GT 的能力差 | 新增training seeds43/44，预先固定完整协议；现有seed42按manifest核对可合并范围。显示每个 seed 与区间；预先写 material margin。**不画本子图也仍应在核心图报告训练不确定性**；诊断 draws42/43/44 不能顶替 |
| **F9b [N]** | 1024 vs4096 cap 的四域能力变化与截断/答案完成率，固定 M-I64-DR；点为同预算训练 checkpoint | 正文已承诺 cap control；保留承诺至少补一条 1024-cap 500 步训练。若只做 one seed 标探索性；预算不足则正文删去这项实验承诺，而非写成已做 |
| **F9c [N]** | 固定 joint checkpoint 的诊断 batch size=1/4/8 responses per domain，对 overlap、有效更新比例/集中度的影响 | 固定 cap、loss、prefix 取样规则；至少 3 banks/size，记录 combined batch 大小。与 τ/ρ sensitivity 一起检验结论是否依赖测量口径；无需另开在线训练 |

第二模型 SmolLM3 不是当前 Qwen3 shared setting 的硬性缺口。若扩展跨模型结论，应单独配置其教师、任务与评估，不能把一轮 smoke 当“第二模型已覆盖”。不建议用新模型/K sweep/GPAS 填满图数而分散当前证据缺口。

## 6. 接下来需要做的实验与交付顺序

本轮更新的是实验清单与文档，没有启动训练、评估或诊断。详细参数和依赖见 [下一批实验](NEXT_EXPERIMENTS_20260908_zh.md)。

| 编号 / 优先级 | 工作包 | 产物 / 验收 | 覆盖子图 |
|---|---|---|---|
| W0 / P0 | 明确当前Overlap64定义，检查所有测量的checkpoint身份及记录字段 | 保留原始provenance；图表数据点可追溯，缺失值不补造 | 全部 |
| W1 / P0 | 收齐在跑的M/S-PG100 math/IF并去重 | artifact完整、题数/哈希/step/scorer一致；不按最高得分挑重试 | F8b/c |
| W2 / P0 | 完成现有六组；250/500完整导出后流式扫描BF16 | 四组TopK各两点，共8个新增扫描；S-PG500另1点。保存fraction、L2、energy与逐层数据 | F4、F8 |
| W3 / P0，最高GPU优先级 | 三组M-Overlap64/250四域评估，随后五个未完成组的500评估 | 当前缺口8套×1720=13760 responses；前三套直接补归一化能力主线 | F3、F8 |
| W4 / P1 | Overlap64 fixed-batch归一化诊断：先M-PG250/500，后M-Overlap64-DR250/500 | 每状态至少4banks；先沿用8条response、cap512验证，再补长response与不等quota/worker分母检查 | F1、F2 |
| W5 / P0，先补程序入口 | 给局部runner加入Overlap64 loss及teacher-conditioned Overlap64分支；同状态PG/Overlap64/full/zero对照 | 先用M-PG500的3个已缓存banks补Overlap64，再测M-PG250、S-PG250；随后加入Overlap64学生状态。固定prefix/mask/Adam，输出BF16写回与JS/support | F5、F6、F7 |
| W6 / P0，避免漏记 | 记录真实online_update_bf16；在未来250/500附近保留实际相邻更新 | 旧`update`是FP32。仅记录前后BF16真实写回，不用局部分支冒充；加入稳定入口后用于后续保存窗口 | F8a |
| W7 / P1，首轮结果后 | 优先M-PG和M-Overlap64-DR的seeds43/44，共4条训练；再扩其余4配置 | 新增重复固定完整协议；统计聚合前核对每个seed的可比性。六配置完整扩展共12条新运行 | 核心全部 / F9a |
| W8 / P2或正文承诺 | cap1024对照、诊断batch-size敏感性 | 保留正文承诺时完成；第二模型及K sweep后置 | F9b/c |

当前没有另列四条seed42主训练。优先用现有checkpoint回答机制问题、补全能力数据，再依据可分辨的效应安排独立重复。

局部诊断分两轮：第一轮完成已有M-PG250/500、S-PG250状态的当前loss对照；第二轮接上M-Overlap64-DR250/500和S-Overlap64250/500。统一cap256、每response至多4prefixes、同一组bank seeds；旧cap128/2prefix结果仍单独标注。扩展到完整论证时再补0/50与至少4个banks，避免一次性铺开大网格。

## 7. 当前数据缺口与预算

一套四域评估=500+128+300+198×4=**1720 responses**。最小在线闭环：六配置500终点，以及四个多教师配置250中点，共10套。M-PG250/500已完成，当前还缺 **8套=13,760 responses**：

- M-Overlap64-DR/DT/GT250：3套=5160，checkpoint出来后第一批运行。
- M-Overlap64-DR/DT/GT500、S-PG500、S-Overlap64500：5套=8600，跟随终点导出。

S-PG250已完整，可额外复用；S-Overlap64250四域评估是第9套可选补点（1720），有助于单/多教师中期对照。若所有六组都做100/250/500，目标18套=30960；当前已验证PG训练后7500条，尚缺23460条，不把重试回答算成新增覆盖。

| 目标 | 新增工作量 | 优先级 |
|---|---|---|
| 当前六组训练完成 | 剩余 **1294 updates、82,816 responses**，截至01:50 UTC | 保持现有训练推进 |
| 第一批关键能力点 | 3套Overlap64/250评估，5160 responses | 最高 |
| 其余核心能力终点 | 5套评估，8600 responses | checkpoint导出后 |
| 关键比较新增重复 | M-PG与M-Overlap64-DR各seeds43/44：4条×500=2000 updates | 首轮能力/机制对照完成后 |
| 六配置完整新增重复 | seeds43/44共12条×500=6000 updates | 根据结果与算力扩展；包含上面4条，不额外相加 |
| cap支持实验 | 另1条500updates与相应评估 | 保留对应正文承诺时 |

当前各分支最近25条allocation中剔除checkpoint-due点的步时：

| 实验组 | 中位秒/步 | 剩余更新 | 剩余纯步时 |
|---|---:|---:|---:|
| S-PG | 276.1 | 187 | 14.3小时 |
| M-Overlap64-DR | 156.8 | 265 | 11.5小时 |
| M-Overlap64-DT | 153.9 | 254 | 10.9小时 |
| M-Overlap64-GT | 170.7 | 257 | 12.2小时 |
| S-Overlap64 | 208.5 | 331 | 19.2小时 |

合计约 **68.1个训练槽位小时**；槽位包含learner、rollout及可能共享的teacher，不能当作GPU小时。当前五条训练若同时持续推进，最长分支约20小时纯步时；另加导出、恢复和调度时间。

现有轻量CPU批量扫描20点约30.8分钟；两个250局部probe约35/53分钟；500教师overlap每bank约72–73分钟；归一化每bank约16–36分钟。这些是原节点/原协议实测，换到本机A6000或增加当前loss分支后需重新校准。

本机01:50附近GPU0/1/2仅占约2MiB，可作为下一批诊断/评估候选；其余卡已有任务。本轮未预订GPU，启动时需再核对。三张候选卡建议：先准备三个M-Overlap64/250能力评估；checkpoint到齐之前可安排能及时结束的M-PG局部诊断。CPU同时处理已完成的BF16扫描与图表。

若保持现有训练、至少两个评估槽位和一个诊断槽位，首轮补齐当前关键数据预留 **2–4天**；它不是全部统计重复完成的时间。关键四条新增seed训练或六配置完整重复分别另计，先用当前loss与实际节点做吞吐校准，不再将四条额外seed42训练作为工期前提。

## 8. 数据记录协议与验收规则

为后续无需重新训练就能做图，每条新运行/诊断应至少输出以下长表。可以沿用现有JSONL再导出，不要求为论文换训练框架。

| 表 | 每行的粒度 | 必须保留的字段 |
|---|---|---|
| `run_registry` | run / checkpoint | run_id, training_seed, model/tokenizer/teacher revisions, source hash, protocol hash, loss_version, K, reduction, optimizer/clip/decay, node/GPU |
| `train_domain` | update×domain | attempted/completed/invalid/truncated responses, valid/generated/teacher-scored tokens, prompt share, token share, length mean/分位数, denominator, gradient norm/clip, loss各自名称, intersection质量统计 |
| `response_weights` | bank×response×reduction | domain, prompt_id, valid length/mask, response coefficient, λ, quota, microbatch_id, worker_id, sums/denominators |
| `parameter_change` | checkpoint/真实update×quantity×layer×threshold或ρ | BF16 source, before/after hash, quantity=`cumulative/online_step/local_proposal`, n_params, count_changed, L2, total energy, topρ energy, ties/exclusions |
| `teacher_pairs` | checkpoint×bank×teacher_pair×mode×loss×selection | common bank hash, teacher hashes, full JS, support counts/intersection/union/Jaccard, signed cosine, zero-grad/random controls, fixed-count或threshold定义 |
| `capability` | checkpoint×benchmark×prompt×sample | generation config/template/suffix/context/cap, answer/reward, truncation/status, scorer/data revision, artifact hash；GPQA保留question_id和4个sample |
| `costs` | run/update/诊断/评估 | elapsed, active/occupied GPU seconds, teacher共享分摊依据, generated/scored tokens, checkpoint I/O, peak RAM/VRAM；计量定义和节点硬件 |

验收不能只看“程序退出0”或“有图片”：

1. **论点—数据**：每个panel明确当前测量能支持什么，哪些空白仍是未检验假设；正结果、零结果与反例均保留。共同checkpoint局部teacher影响不等于历史联合delta分解。
2. **方法身份**：训练种子/诊断draw/评估sample分开；每个图点连到真正checkpoint与相应loss记录；重复日志去重。
3. **数值口径**：BF16差、真实单步/累计/proposed各自命名；raw gradients的累加精度如实记录（现有probe是BF16反传再CPU FP32汇总，不写成全程FP32反传）。tau与energy一起解释幅度/集中度。
4. **可比较性**：同batch的候选support冻结，比较完整response时覆盖全部有效位置；教师和student共同prefix按实际teacher-only suffix解释。匹配exposure并不意味着整个训练分布相同。
5. **统计单位**：训练主结论优先3个配对seeds；评估按prompt配对bootstrap，GPQA按题聚类4回答；教师pairs共享教师，重复banks/checkpoints不能当独立teacher样本。3个诊断bank的min–max只表示观察到的范围。
6. **决策规则**：在新增正式seed和主要终点评估前写material margin（建议先讨论 BF16 fraction 的绝对pp差、macro/worst-domain的pp差，并给敏感性区间）。现有seed42已看过，不能把后来选的阈值宣称预注册；不得为求显著结果调整指标或挑seed。
7. **可复现作图**：每图落一份panel输入CSV、过滤/聚合规则、源码commit/hash、图注、PDF；保留原始点，不插补未完成评估，不用损失数值推导不存在的能力结果。

## 9. 文稿同步清单

本次不改写正文实证结论；以下改动应与最终主矩阵一起完成：

- [001_abstract.tex](../sections/001_abstract.tex)、[002_introduction.tex](../sections/002_introduction.tex)：仍是teacher top-k与结果待测措辞。可以更新“已有什么”，但I64结果未收齐前不预写其优越性。
- [004_preliminary.tex](../sections/004_preliminary.tex)：区分T64、ST64、I64，给实际主loss；不能只把64替成16或把teacher替成student而保留旧公式。
- [005_normalization.tex](../sections/005_normalization.tex)：把代数恒等式、已完成small-batch审计、尚缺online能力三层分开；cap control做完或删除相应承诺。
- [006_teacher_subnetworks.tex](../sections/006_teacher_subnetworks.tex)：主参数统计统一BF16；FP32现有测量如保留只作说明；修正topρ的零值/tie处理；把高overlap及zero-grad控制一起呈现。
- [007_supervision_density.tex](../sections/007_supervision_density.tex)：当前主线使用teacher–student Top64 overlap；既有局部ST64/T64参照保留各自名称；补齐Overlap64分支后再写相应实证结论。
- [005_experiment.tex](../sections/005_experiment.tex)、[009_appendix.tex](../sections/009_appendix.tex)：补实际训练/评估/诊断预算、数据版本、suffix、precision；移除“协议尚未冻结”“trainer不存在”等过时表述；full vocab只在实做的局部范围内报告。

**排版建议**：先按8行核心讲完整证据链，9行版增加稳健性。根据编译后可读性，把完整阈值网格、所有层、全部教师对轨迹和重复种子原始点移到附录；主文每一行只回答一个问题。不要为了每行三个子图把同一证据换三种画法重复计数。
