# 按三个原问题组织的论文图表

2026-09-11 修订，恢复 [7417292 正文](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/7417292c57aa0cbe8486cad4986aa5617cd0f681) 的探究顺序。正文各节同时陈述问题、对应实验和实测范围，不再按“梯度—Adam—写回”组织研究贡献。

| 正文位置 | 主图 / 表 | 对应问题 |
|---|---|---|
| Section 3 | `normalization_capability.pdf`；平均方式定义表；共同 50 步结果表 | 三种平均方式如何改变域权重和能力平衡？ |
| Section 4.1 | `cumulative_geometry.pdf` | 联合 PG/I64 是否仍有集中参数变化？单教师对应记录明确列为缺口。 |
| Section 4.2 | `teacher_overlap.pdf` | 同一联合学生中的教师参数选择是否重叠？现有矩阵明确标为梯度。 |
| Section 4.3 | `teacher_js.pdf` | 教师分布距离是否对应参数选择距离？现有散点使用梯度 overlap。 |
| Section 5 | `supervision_density.pdf`；`density_capability_table.tex` | PG/top-k/full 局部更新稀疏性如何比较？PG/top-k 在线能力怎样变化？ |

附录保留六张有直接解释作用的图：`capability.pdf`（完整 13 套能力）、`normalization50.pdf`（配对区间）、`adam_precision.pdf`（优化器控制）、`thresholds.pdf`（稀疏性阈值）、`teacher_input.pdf`（输入与方向背景）、`all_token_shares.pdf`（域权重来源）。

动作数/MC 方向图、独立响应行为图与 RL regularization 理论不进入本版编译稿。历史文件与原始数据保留，不自动成为后续必做实验。

当前图源均在 `aligned_evidence_20260910`。用 `python experiments/plot_aligned_evidence.py` 从冻结数据重画，用 `latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex` 编译，再运行 `python experiments/verify_aligned_evidence.py`。旧 September 8 图集仅作历史记录。
