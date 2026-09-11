# 2026-09-10 论文证据版本

本版本对应根目录的 `iclr2027_conference.tex` / `iclr2027_conference.pdf`。正文按「损失平均 → 梯度 → Adam → BF16 写回」组织，使用 aligned v8 数据。旧曲线目录保留作历史记录，正文不再引用其中的 Base、Student64→I64、旧 teacher-Top64 或 GPAS 数值。

## 证据覆盖

| 证据 | 数量 / 范围 | 论文用途 |
|---|---|---|
| 对齐能力评测 | 13 套，共 22,360 条回答；共享初始点只计一次 | 能力轨迹、归一化比较、完整结果表 |
| 本机可核验评测 | 9 套，15,480 条回答；其中 GPQA 7,128 条重新评分全部一致 | 配对题目比较、评分审计 |
| 单教师远端评测摘要 | S-PG / S-I64，各 100、250 两个点，共 4 套 | 能力轨迹；原始回答留在评测节点 |
| 兼容教师基准 | 4 个 RL 教师，各自领域 | 教师质量参考表；science 的 792 条 GPQA 重新评分一致 |
| 在线几何 | M-PG、DR、GT 至 100；DT 至 50 | 实际 FP32 / BF16 累积变化、更新与能量集中 |
| 对齐局部机制 | M-PG/100，433 条主测量，2 个 bank，8 条回答、32 个 prefix | 教师×输入、Adam 状态、监督密度 |
| PG 采样控制 | 1 / 16 / 64 actions，2 bank × 2 repeats | 固定 prefix 上的实现采样误差 |
| 直接 Adam 向量比较 | PG / I64 / full，FP32 与 BF16 | 相同密度下的方向差异 |
| 原始训练行为 | M-PG 128、DR 100、DT 100、GT 115 个 rollout clock | token 分配、长度、截断、终止位置监督 |

单教师训练到 500 与单教师 **500 能力评测完成**是不同状态。2026-09-11 增补了已完成的 aligned S-PG/500；S-I64 当前能力曲线止于 250；DT/100 的不完整检查点不进入能力或几何比较。一个 `train/step` 是四响应梯度分片，16 个分片才合成一个 optimizer update；图中的 checkpoint 横轴使用后者。

## 图表与论证的对应

| 图 | 文件 | 回答的问题 |
|---|---|---|
| 1 | `capability.pdf` | 各配置在真实完成的 checkpoint 上怎样分配能力收益？ |
| 2 | `cumulative_geometry.pdf` | 同一累积变化在 FP32 与 BF16 中有多大支持集？ |
| 3 | `adam_precision.pdf` | 保存的 Adam 状态怎样影响下一步可见变化？ |
| 4 | `teacher_input.pdf` | 教师差异与输入差异分别怎样影响方向和位置？ |
| 5 | `supervision_density.pdf` | 词表监督如何影响梯度和拟议更新方向？ |
| 6 | `normalization50.pdf` | 同步 50 点三种归一化的全部配对差值是什么？ |
| 7 | `thresholds.pdf` | 绝对阈值怎样影响局部活动比例？ |
| 8 | `teacher_js.pdf` | 同 bank / 同教师对的分布距离与支持集距离怎样对应？ |
| 9 | `training_dynamics.pdf` | M-PG 的长度、token 分配和响应行为怎样变化？ |
| 10 | `all_token_shares.pdf` | 三种 I64 平均方式对应的原始 token 分配怎样变化？ |
| 11 | `if_generation.pdf` | IF 分数伴随什么完成、截断和重复行为？ |

所有图片在 `../../figures/aligned_evidence_20260910/`，同时提供 PDF、PNG、SVG。四个数据表进入论文（归一化、全能力、教师、局部 coverage）；第五个配对结果表供直接查看。实验覆盖表和平均方式定义表在 LaTeX 中直接给出。

## 复现

在论文根目录运行：

```bash
python experiments/plot_aligned_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
python experiments/verify_aligned_evidence.py
```

作图依赖 Python、NumPy、Matplotlib；PDF 审核另需 PyMuPDF，编译需 LaTeX 与 latexmk。此流程只读取本目录的冻结数据，在 CPU 上完成。`plotting_environment.json` 记录本次版本。不会启动训练、生成回答或访问 GPU。

如需重新从当前实验目录冻结来源，显式运行：

```bash
python experiments/export_aligned_evidence.py --source ../slime_opd_geometry --capability-only
```

此增量命令核验新增完整评测的源回答并重新计算汇总和配对区间，保留已经冻结的训练与局部机制记录。导出针对本次明确列出的运行目录，不自动把未来的新方法或其他初始化加入比较。

## 统计与来源

- `manifest.json`：原文件路径、SHA-256、选入/排除范围。
- `capability.json`：13 套评分和生成诊断，标明逐条核验或远端摘要。
- `prompt_scores.jsonl`：9 套本机评测的紧凑逐回答记录，含 question identity hash、分数、状态、长度，不复制回答正文。
- `paired_comparisons.json`：40 个领域级比较，10,000 次配对 prompt bootstrap，seed 1042。GPQA 的 4 条回答作为题目簇一起重采样。95% 区间未做多重比较校正，条件于训练 checkpoint。
- `gpqa_score_audit.json`：`final-answer-v2` 评分审计。
- `teacher_references.json`：兼容教师基准与 science 复评分。
- `training_configuration.json`：实际运行参数及来源 hash；不复制节点环境变量。
- `raw/`：原始几何、rollout / optimizer 日志、协议、数值验证与局部计算代码。
- `figure_manifest.json`：图与来源文件、绘图脚本 hash。
- `plotted_values.json`：便于查阅的已绘制数值。
- `verification_report.json`：冻结数据、数值、图和论文引用检查结果。
- `EDITORIAL_REVIEW_zh.md`：作者/审稿人视角下的论证取舍与完成检查。

局部诊断的独立单位是 bank；教师对、输入对和采样重复共享模型与数据。它们不等同于独立训练种子。八条局部生成都达到 256-token cap，coverage 表给出其概率质量与梯度尺度。正文直接陈述这些测量支持的结论，范围集中在讨论和实验设置中。

2026-09-11 的更新与各实验族盘点见 [实验盘点](../EXPERIMENT_INVENTORY_20260911_zh.md)。S-PG/500 的模型转换验证、实际评估命令、单教师协议与四域配置也冻结在 `raw/S-PG_500_*`、`raw/single_protocol.json` 和 `raw/single_capability_eval.yaml`。
