# 独立实验图与作图数据

31组机制实验的数据保持不变；当前图形版本为15张独立面板：F1a–c、F2a–c、F5a–c、F6a–c、F7a–c。

- [逐张浏览](../../figures/local_probe_results_20260908/index.html)
- [逐页审阅PDF](../../figures/local_probe_results_20260908/all_panels.pdf)：一页一张独立图，不是三联图。
- [英文caption草稿](FIGURE_CAPTIONS.md)：所有实验说明、条件定义与限制在此，不放图内。
- [作图路线图](../EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)

## 组织方式

F5/F6的step250/500、PG/I64和输入条件在同图中明确编码；F6a的JS只按checkpoint和teacher pair展示，不重复loss维度。F7同时显示五个学生状态与六种目标。矩阵只平均同条件bank，不跨checkpoint/loss平均；散点保留原始观测。

所有图单独导出PDF、PNG、SVG。基准字体18pt、轴标签19pt、刻度16pt、矩阵数值15pt；紧凑图例13–14pt。图内无标题、panel编号、脚注或解释段落，只保留坐标、数值与必要图例。图片编号由文件名和论文排版指定。

F7a用固定阈值1e-5的紧凑矩阵取代30条叠加阈值曲线，显示单位为10⁻²%（3.52表示0.0352%）。其他阈值仍完整保存在thresholds.csv。F7b把全部原始梯度与BF16范数放在同一坐标；F7c统一显示相对full-vocabulary参照的余弦。

## 数据文件

- `data_manifest.json`与`raw/JOB/`：原31组测量、bank、权重、support、核验与来源SHA256。189个复制文件不变。
- `plot_data/`：八个CSV，含branches、pairs、thresholds、responses、covariance、heldout、teacher_js、normalization_gradients。
- `panel_values.json`：各独立图中每个散点或矩阵聚合前观测的来源、数值与条件；可复核所有图格均值。
- `figure_manifest.json`：独立面板映射、视觉编码、字号、脚本和数据哈希。
- `FIGURE_CAPTIONS.md`：可移入论文的逐图caption草稿。
- `RESULTS_zh.md` / `results_summary.json`：实验结论与限制；`no_step_cast_audit.json`为无Adam直接转换核查。
- `figure_plan.csv` / `.json`：27图槽，F9a/F9b为已取消X；本轮没有新增实验。
- `validation_report.json`：本次独立图及数值一致性检查。

## 重画

在论文目录运行，仅需NumPy、Matplotlib，无需GPU或原训练输出目录：

```bash
python experiments/plot_completed_local_probes.py
python experiments/build_figure_plan.py
```

默认入口只生成独立实验面板及一图一页的审阅PDF；build_figure_plan只导出清单。旧三联图归档于backup/2026-09-08_before_standalone_panels_224019/，不会混入当前图目录。

所有测量仍是保存Adam上的HF局部模拟，不是真实在线单步或能力评测；mixed-history标记与小heldout数值限制在caption中保留。未恢复新增训练或多种子实验。
