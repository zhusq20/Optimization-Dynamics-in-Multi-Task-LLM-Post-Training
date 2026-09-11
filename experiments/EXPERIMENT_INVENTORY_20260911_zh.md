> 本文件保留 2026-09-11 较早的数据盘点。用户指定恢复 7417292 的三个原问题后，下文“实证主线”和“正文取材”的写作建议已被[现行三问说明](../MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md)及[图表路线图](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)取代。Adam、精度、输入和动作采样不能替代原来的稀疏性、教师重叠与监督密度问题；原始数量与来源记录仍供追溯。

**实验盘点与正文取材｜2026-09-11 UTC**

现有结果足以支撑论文的三条实证主线：平均方式分配训练信号，Adam 与精度改变参数变化的支持集，监督候选数改变梯度及更新方向。当前正文已经有 5 张主图和归一化结果表。本次找到一套此前漏入正文的 aligned S-PG/500 完整评测，已补进主图、全结果表和监督密度一节；另将已有 token 份额数据写入归一化正文。

[更新后的论文](../iclr2027_conference.pdf)；[机器可读盘点](completed_inventory_20260911.json)；[数据与图表验证](aligned_evidence_20260910/verification_report.json)。

**一、当前 aligned 主实验：六个配置，13 套完整能力评测**

| 配置 | 已纳入的完整能力 checkpoint | 现有几何 / 机制数据 | 正文位置 |
|---|---|---|---|
| Initial | 0，共享起点 | 初始化参考 | 图 1、附录全结果表 |
| S-PG | 100、250、**500（本次新增）** | 本证据包用其能力轨迹 | 图 1、Section 5 |
| S-I64 | 100、250 | 本证据包用其能力轨迹 | 图 1、Section 5 |
| M-PG | 50、100 | 在线 0/1/50/100；100 步局部机制 | 图 1–5 |
| M-I64-DR | 50、100 | 在线 0/1/50/100 | 图 1–2、归一化表 |
| M-I64-DT | 50 | 在线 0/1/50 | 图 1–2、归一化表 |
| M-I64-GT | 50、100 | 在线 0/1/50/100 | 图 1–2、归一化表 |

13 套共 22,360 条回答。其中 9 套、15,480 条可在本工作区逐回答核验；另 4 套为 S-PG/S-I64 的 100/250 远端已验证摘要。每套包含 Math500 500、LCB 128、IFBench 300、GPQA 198×4，共 1,720 条。另有四位 RL 教师各自领域的兼容基准。

本次新增 S-PG/500 的完成时间是 **2026-09-11 00:43:40 UTC**。逐域成绩为 Math **73.80%**、Code **19.53%**、IF **18.00%**、GPQA **31.82%**。相对共同初始点，Math +5.0pp，配对题目 bootstrap 95% 区间为 [1.4, 8.6]pp；Code +3.125pp，区间 [0.78125, 6.25]pp。正文描述已测轨迹：数学在已评测点中以 100 步的 77.4% 最高，250 和 500 步均为 73.8%。这些区间以训练 checkpoint 为条件。

新增点的证据链是 `raw/S-PG_500_complete.json` → `raw/S-PG_500_index.jsonl` → 四域原始回答哈希及评分重算 → `capability.json` → 图 1 / 全结果表。`raw/S-PG_500_export_verified.json` 记录 310 个参数键及原生 checkpoint 转换后逐元素相等检查；`raw/S-PG_500_job.json` 保存实际评测命令；单教师协议与共同协议的初始化、教师、前缀、停止语义、评估和数据定义一致。新增 792 条 GPQA 用 final-answer-v2 复评分全部一致；累计本机审计为 7,128 条。

能力 coverage 与训练进度分别记账：本次数据截点仍没有 aligned S-I64/500、四组多教师 250/500 或 DT/100 的完整能力产物。这里只汇总已获得的结果；完整训练网格的收尾属于后续实验工作。

**二、最值得保留在正文的证据，以及直接陈述的发现**

| 证据 / 图表 | 可直接写入正文的实测结论 | 本次处理 |
|---|---|---|
| [能力轨迹：图 1](../figures/aligned_evidence_20260910/capability.pdf) + [50 步结果表](aligned_evidence_20260910/normalization_table.tex) | 同一 50 步，GT 数学 74.2%、DR 69.6%；DR 的 IF 19.0%、GT/DT 17.0%。归一化对应早期能力取舍。 | 保留共同 checkpoint 的完整四域表；图 1 延长 S-PG 至 500。 |
| [token 份额：图 10](../figures/aligned_evidence_20260910/all_token_shares.pdf) | GT 前 50 个 rollout 的平均数学 token 份额 44.05%、IF 14.98%，两者 prompt 配额均为 25%。GT 将这一长度差异直接带入 loss 权重。 | 数字提升到 Section 3 正文；完整原始曲线保留在附录。 |
| [在线参数变化：图 2](../figures/aligned_evidence_20260910/cumulative_geometry.pdf) | M-PG/100 的累计非零比例在 FP32 为 94.0%，BF16 为 3.41%；90% 变化能量所需坐标比例分别为 34.46% 与 1.82%。 | 保留为参数变化一节的主结果。 |
| [Adam 状态干预：图 3](../figures/aligned_evidence_20260910/adam_precision.pdf) | 相同保存状态下，零当前梯度也产生 0.1403% 的 BF16 活动；PG 为 0.1486%，reset-m 为 0.0504%，fresh 为 0.5140%。Adam 历史参与决定可见变化。 | 保留独立的零梯度与状态干预读数，围绕优化器如何产生下一步移动来写。 |
| [教师 × 输入：图 4](../figures/aligned_evidence_20260910/teacher_input.pdf) | 固定教师、改变输入域，梯度平均 cosine 接近零；固定输入、改变教师，有同向也有反向。输入组成参与决定教师条件梯度的方向。 | 保留匹配 prefix 数的 crossed 比较；位置重叠与有符号方向各自解释。 |
| [监督密度：图 5](../figures/aligned_evidence_20260910/supervision_density.pdf) | PG/full 梯度 cosine 0.6766；BF16 step cosine 0.9149，而两者活动比例约 0.1486%。固定 prefix 的 PG 动作数 1→16→64 时 cosine 0.5589→0.8962→0.9787。 | 将方向差异作为正面发现；并列给出本 bank 的 Top64 概率质量覆盖。 |

这里的 token 份额是 rollout index 0–49 的逐批份额算术平均，不是把 50 批 token 合并后的比例。数值由 `audit_completed_inventory.py` 重算。GT 的实际 loss 权重与该份额一致；它与能力取舍构成相邻的观察，具体跨步骤机制仍由轨迹和局部测量分别描述。

其余六张附录图承担具体补充作用：全八个归一化配对差值、阈值敏感性、教师 JS 与支持集距离、响应行为、三分支 token 份额、IF 生成诊断。正文保留发现与关键数字，评分、数值控制及测量范围放在方法或对应图注。

**三、已经做过的其他实验：按研究阶段归档**

本次在可见 `outputs/` 中逐个读取了 **218 个 `run_complete.json`**，覆盖 18 个顶层实验目录。该数量包含训练、评测、探针和 smoke 尝试。逐路径、完成状态、步数、时间及 SHA-256 均在机器可读盘点中。历史研究的更细设置见[源仓库历史总账](../../slime_opd_geometry/local/experiment_inventory_20260910/README_zh.md)；下表据完成标记及已有冻结结果归类。

| 实验族 | 已有产物 | 当前论文用途 |
|---|---|---|
| 旧 Base PG + Student64→I64 | 六条 500 步训练轨迹；历史能力评测和原始 scalar 镜像 | 历史协议结果单独保存。四条交集目标具有 Student64 训练前史。 |
| 旧协议的新能力产物 | DR/DT/GT 的 250 步 Math/IF 三组完成；S-I64/500 与 S-PG/500 四域评测完成 | 新增完成状态进入盘点；与 aligned S-PG/500 分开记录。其他终点评估以实际完成标记为准。 |
| 旧 Base 局部探针 | [31 组](local_probe_results_20260908/RESULTS_zh.md)：20 组监督密度、7 组归一化、4 组教师输入控制；15 张历史图 | 保留为探索过程；当前正文已有相同研究问题的 aligned 测量。 |
| 旧 BF16 / Top-k 诊断 | checkpoint 扫描、局部写回、Top16 选集与概率质量、Top64 目标验证 | 为精度与目标实现的来源记录；正式 Top16 完整训练网格未形成。 |
| GPAS v4 | Uniform/GPAS/cost-GPAS/raw-noise 四组 500 步及各自能力评测；50/250/500 variance 测量 | 完成独立采样器研究阶段，与本文三条实证主线分别归档。 |
| GPAS core | Uniform/GPAS/raw/D3-fixed 四组 500 步；6 套不同 cap 的能力评测；common checkpoint 20 步诊断 | 已有训练、能力与机制结果，保留各自配置和评分版本。 |
| GPAS 早期模拟与资源 pilot | controlled 优化器/采样模拟，cost-aware 敏感性；v1–v3、K1/K2/K4、offload 等运行尝试 | 模拟、资源验证与正式 LLM 训练分别记账。 |
| 单域 OPD 与单任务 GRPO | AdamW/SGD 的早期 OPD 谱分析；Math/Code/Science/IF 训练及单任务离线分析 | 作为研究背景与教师来源记录。 |
| GRPO 跨域迁移 | 三任务 6 个 off-diagonal no-thinking 评测，以及 6 个 thinking pilot 完成标记 | 属于跨域迁移实验，可独立组织结果。 |
| 四域混合 GRPO | 1,200 步训练及 final_eval 均有完成标记 | 与 MOPD 的教师监督协议分别归档。 |
| 顺序持续 GRPO | Code→Math→Knowledge→IF；5 个边界评测、20 个教师/任务梯度探针 | 可支撑顺序迁移的独立分析；保留原任务顺序。 |
| GRPO 参数空间分析 | endpoint/layer cosine、PCA、BF16 support、位移尺度与路径效率图表 | 历史几何证据，测量对象是对应 GRPO 轨迹。 |
| 生成与评分诊断 | GPQA 解析审计、IFBench 回放、EOS/尾部梯度、长输出教师复评 | 当前正文使用兼容协议的标准分数与行为图；诊断细节保留来源。 |
| SmolLM3 | MixSFT 的 PG/I64 各 2 步真实管线验证；早期短运行 | 已验证扩展管线，完整第二模型实验仍未形成。 |

**四、本次核验和交付**

对证据包记录的 114 个来源逐一重新计算 SHA-256：112 个与冻结版本相同，2 个 M-PG 在线日志在恢复训练后发生变化。正文继续引用已经冻结并通过哈希核验的历史段；本次仅增量导入新完成的能力评测，避免把恢复段混进原图的 rollout 时钟。局部 433 条主测量、两 bank/8 回答/32 prefix，以及 1/16/64 动作采样和直接 Adam 向量比较均已在正文/附录中找到对应位置。

本次修改包含能力图、全结果表、摘要/引言/设置中的样本总数、监督密度一节的 S-PG 终点结果，以及归一化一节的 token 份额观察。写作采用“测到什么 → 怎样解释 → 对应哪张图”的顺序；具体实验范围保留在设置、图注和讨论中。

复现本次增量整理：

```bash
python experiments/export_aligned_evidence.py --source ../slime_opd_geometry --capability-only
python experiments/plot_aligned_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
python experiments/verify_aligned_evidence.py
python experiments/audit_completed_inventory.py --source ../slime_opd_geometry
```

这里的“当前”指本次盘点快照；持续运行产生的新 checkpoint 可在下一次核验后增量纳入。本次没有启动训练、评测推理或调整已有任务。

最终验证：766 项检查通过，78 个冻结源文件哈希一致；PDF 共 16 页，无未定义引用或 overfull。已检查新增曲线、正文段落及完整结果表所在页的实际渲染。
