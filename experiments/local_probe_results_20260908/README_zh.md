> 当前正文图已切换为[训练轨迹与机制新版](../../figures/paper_curves_20260908/index.html)，并补入[W&B raw training/eval](../../figures/paper_curves_20260908/raw_curves.html)。下文为31组源探针与上一版15张图的复现记录；旧正文推荐已被替换。

# 独立实验图与作图数据

31组局部机制实验，15张独立PDF/PNG/SVG。当前版本按审稿人问题重画，已移除密集散点与多重颜色/点形编码。

- [按问题逐图阅读](../../figures/local_probe_results_20260908/index.html)
- [六张正文候选，一图一页](../../figures/local_probe_results_20260908/main_panels.pdf)
- [全部15张，一图一页](../../figures/local_probe_results_20260908/all_panels.pdf)
- [逐图问题与审稿检查](REVIEWER_FIGURE_AUDIT_zh.md)
- [英文caption草稿](FIGURE_CAPTIONS.md)
- [作图路线图](../EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)

## 阅读顺序与含义

正文候选为F1c/F2b、F5a/F5b、F7a/F7b，分别回答平均规则、教师输入控制、监督目标的局部影响。其余为补充诊断。图片内只保留轴、数值与至多两系列图例，解释在caption和浏览页图片外。轴19pt、刻度16pt、数字15–16pt、图例15pt。

F5将密集散点改成均值与全观测min–max；不同条件有明确行标签。F6用同一教师对行序，避免不透明点群暗示JS相关性。F7b用同状态PG=1直接比较梯度与BF16两种范数；F7a保留绝对百分比，F7c保留方向差异。

## 数值与复核

- `data_manifest.json`、`raw/JOB/`：31组原测量及来源SHA256，189个源文件保持不变。
- `plot_data/`：八个原始测量CSV，保留绝对范数、cosine、KL、权重和所有已测阈值。
- `panel_values.json`：每个图示值的来源、条件、原数值和转换参数。
- `panel_summaries.json`：矩阵或成对图的均值、最小/最大值、观测数及聚合定义。
- `figure_manifest.json`：独立图、问题、建议位置、字号、脚本与数据哈希。
- `layout_checks.json`：渲染时数字/标记重叠检查；矩阵数字对比度检查。
- `validation_report.json`：当前最终产物、逐值与来源一致性核查。
- `FIGURE_CAPTIONS.md`、`REVIEWER_FIGURE_AUDIT_zh.md`：逐图问题、英文图注、阅读顺序与局限。
- `RESULTS_zh.md`、`results_summary.json`、`no_step_cast_audit.json`：原实验结论与额外转换核查，作为冻结证据保留。
- `figure_plan.csv`、`figure_plan.json`：27个计划图槽及当前视觉定义，F9a/F9b仍取消。

F1b是token share减已知prompt share。F2c是同job KL下降减zero对照。F7b是四bank范数均值之比，非各bank比值的平均。F7c是逐bank转角度后取均值，非均值cosine的角度。F5的范围包含共享教师的依赖观测，不是统计置信区间或训练seed重复。

## 重画

在论文仓库运行，仅需NumPy、Matplotlib，不加载模型或GPU：

```bash
python experiments/plot_completed_local_probes.py
python experiments/build_figure_plan.py
python experiments/validate_reviewer_figures.py
```

每次重画先校验189个源文件。两次绘图历史分别存放于backup/2026-09-08_before_standalone_panels_224019/和backup/2026-09-08_before_reviewer_redesign_225743/。当前入口不生成三联图。

全部测量仍为HF局部模拟，不是实际在线单步或能力评测。mixed-history与小heldout数值限制保留在caption；没有恢复新增训练或多种子实验。
