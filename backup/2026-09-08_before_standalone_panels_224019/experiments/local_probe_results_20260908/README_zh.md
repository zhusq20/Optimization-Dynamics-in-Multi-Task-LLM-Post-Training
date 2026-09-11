# 已完成局部探针：论文作图数据包

31组机制实验在2026-09-08 20:45 UTC完成。本目录包含可独立重画F1/F2/F5/F6/F7的输入，原始实验无需仍在原路径。

- `data_manifest.json`：31个job与所复制文件的SHA256、原始来源。
- `raw/JOB/`：measurements、banks/prefixes、weights/support、完成与核验记录。图不使用大概率tensor，因此不复制distributions.pt。
- `plot_data/`：branches、pairs（两种阈值）、thresholds、responses、covariance、heldout、teacher_js、normalization_gradients CSV；各行包含job、state、相对source。
- `figure_manifest.json`：15页图的panel/job映射、五个主视图、CSV行数及绘图脚本hash。
- `RESULTS_zh.md` / `results_summary.json`：描述性结果与解释限制。
- `no_step_cast_audit.json`：PG250/500的全部坐标无Adam直接转换检查，BF16(master)与保存模型一致。
- `figure_plan.csv` / `.json`：从现行路线图生成的27图槽；X表示取消。

## 看图

[五页主图](../../figures/local_probe_results_20260908/main_local_probe_figures.pdf) · [全部十五页](../../figures/local_probe_results_20260908/all_local_probe_figures.pdf) · [现行路线图](../EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)

主视图F5/F6固定为PG500上的Overlap64，F7为PG500；所有检查点/损失版本均在完整PDF保留。五个图族共15页，45个子图实例不代表45个独立实验。

## 重画

在论文目录执行（NumPy、Matplotlib；不需要GPU、模型或原训练目录）：

```bash
python experiments/plot_completed_local_probes.py
python experiments/build_figure_plan.py
```

重新绘图会验证包内原始文件哈希。不要修改raw测量以调整图形。首次导入使用`--import-from`指向实验campaign的local目录；日常重画不需要这个参数。

## 图注口径

- 全部是HF局部模拟和保存Adam下的BF16写回，不是真实在线单步、累计变化或新增能力评测。
- 教师图每个checkpoint有两个bank、每bank六个共享教师对。范围条是观察到的min–max，不是CI。
- F5a对角为自身Jaccard=1，F6a对角为自身JS=0；矩阵之外没有填造数据。
- F5c随机参照是逐层匹配的期望交/期望并之比。支持数和等数量支持检查在pairs.csv；全层细节在raw。
- F2c展示全部七组macro读数，四域明细在heldout.csv。小heldout和跨运行数值差异使它不适合推断规则的能力排名。
- mixed-history学生状态只作局部检查点；不能称全程交集训练。多种子和新训练已取消。

历史01:10快照及旧三页预览保留在相邻的figure_audit_20260908与figures/evidence_audit_20260908，本数据包没有覆盖它们。
