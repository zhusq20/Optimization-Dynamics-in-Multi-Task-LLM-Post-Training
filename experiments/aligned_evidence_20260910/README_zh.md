# 当前论文证据包（2026-09-12 更新）

目录名保留首次冻结日期；新增结果使用带 20260912 的文件，原冻结实验记录保持不变。

| 证据 | 当前完整范围 | 文稿用途 |
|---|---|---|
| 能力评测 | 16 套，27,520 回答；12 套本机原始回答核验、4 套远端摘要 | 三问的完整共同 checkpoint 比较 |
| 本机评分核验 | 20,640 回答；9,504 GPQA 回答复评分一致 | 评分与配对差值 |
| 单教师 PG/I64 | 共同 100/250/500 | 能力比较 |
| 联合 PG/I64-DR | 共同 50/100/250 | 能力与参数变化 |
| I64 GT/DT/DR | 共同 50 | 平均方式比较；GT100 单点只保留盘点 |
| 固定 batch 平均方式 | M-PG100/250，各 3 bank，三种 reduction | 正文梯度方向观察、附录六行完整表 |
| 局部监督主实验 | M-PG100，4 bank，16 responses、96 prefixes，4,096-token cap | PG/I64/T64/full 的三列主图 |
| 原两 bank 局部控制 | 8 responses、32 prefixes，256-token cap | 梯度 overlap/JS、Adam、阈值与概率质量附录 |

尚缺的单/多教师匹配参数几何、逐教师 optimizer-step overlap 及与 JS 的配对，在主文保留批注。已有梯度结果作为附录独立诊断，不填进这些图表。完整对照表和曲线只取相应配置共同 checkpoint。全部 16 套原始能力汇总仍保留在 `capability.json`。

## 文件与复现

- `manifest.json`：冻结来源路径和 SHA-256。
- `capability.json` / `prompt_scores.jsonl` / `paired_comparisons.json`：全部能力记录、逐回答分数、68 组配对差值与区间。
- `online_geometry_20260912.json`：两条 joint 分支的完整第 250 步几何。
- `normalization_fixed_batch_20260912.json`：两 checkpoint × 三 bank 的所有三对原始梯度比较。
- `completed_local_followup_20260912.json`：三个完整局部实验；主图使用 `density-mpg100-long-bank1042`，其余记录供追溯。
- `raw/followup/`：新增局部实验的完整原始测量、配置、coverage、完成标记、数值核验及代码来源。
- `figure_manifest.json` / `verification_report.json`：作图数据范围、三列布局与验证结果。

从当前冻结证据编译（CPU）：

```bash
python experiments/plot_aligned_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
python experiments/verify_aligned_evidence.py
```

从源实验重新导出本轮已完成结果：

```bash
python experiments/export_aligned_evidence.py --source ../slime_opd_geometry --capability-only
python experiments/export_followup_evidence.py --source ../slime_opd_geometry
```

这两个命令读取已有产物，不启动训练或模型推理。依赖 Python、NumPy、Matplotlib、PyMuPDF 与 LaTeX/latexmk。每个图表只展示完整比较；新增数据对应的原始来源和数值检查见[本轮结果盘点](../RESULTS_AUDIT_20260912_zh.md)。
