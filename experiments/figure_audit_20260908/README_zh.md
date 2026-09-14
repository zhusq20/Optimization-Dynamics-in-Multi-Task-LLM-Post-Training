# 2026-09-08 实验与作图数据快照

主说明：[实验进度、27个子图与补实验清单](../EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)。实验结果冻结于2026-09-08 01:10 UTC；另附[01:50训练进度复核](next_experiment_progress.json)与[下一批实验安排](../NEXT_EXPERIMENTS_20260908_zh.md)。不追踪实时进程，不改变训练状态。

## 文件

| 文件 | 内容 |
|---|---|
| `evidence_snapshot.json` | 相关训练、能力点、累计BF16、rollout域统计、局部probe和teacher pairs；209个源文件的读取字节数、SHA256和mtime |
| `run_inventory.csv` | 14个相关训练目录；含关联记录，不能算成14个独立训练；HF只检查索引所需分片存在且非空，没有重新加载所有权重 |
| `capability_inventory.csv` | 25个完整benchmark点、10940条回答；逐题计数、hash、step、终止状态、reward均值与汇总一致性全部通过；没有重新跑评分器 |
| `bf16_inventory.csv` | 21个已完成扫描，保留报告run名与实际checkpoint run两列；实际student-support Top64测量点标为ST64 |
| `next_experiment_progress.json` | 01:50 UTC训练步数、HF/snapshot就绪状态与本机GPU快照；不是已完成结果刷新 |
| `figure_plan.csv` / `.json` | 27个图槽的状态、论点、所需数据、现有证据、待做实验和工作包；根据主说明的逐图表导出 |

原始数据路径均相对于`iclr2027`工作区；snapshot内probe原有输入路径保留以核对provenance。源数据是现有运行产物，不是本轮重新训练或重新测量。M-PG/500的早期独立扫描报告未保存baseline tensor hash，其余20个批量报告保存了相同baseline hash；缺失处记null，不补造哈希。

评估失败、取消或仍在运行的目录不因存在进度条就算完成。25个结果点是完整产物，不代表25个独立模型/训练种子；GPQA四采样按题处理。teacher-pair CSV的72行是六个共享教师pairs的重复测量，不是72个独立样本。

## 复现

在论文目录、使用对输出目录有写权限的工作区用户运行；只用CPU读取文件和Matplotlib，不加载模型、不启动训练。

```bash
# 推荐新目录，保留本次快照，不静默更新历史报告。
python experiments/audit_figure_evidence.py --output /tmp/iclr2027-figure-audit-refresh

# 从冻结的01:10快照重画三个1×3预览，默认输出至figures/evidence_audit_20260908。
python experiments/plot_figure_evidence.py

# 从主说明表格导出27个图槽，并画9×3布局图。
python experiments/build_figure_plan.py
```

刷新脚本针对本次已知campaign目录审计；新campaign应先加入`GROUPS`/diagnostic roots与相应验证。它是可读的本次盘点脚本，不是会自动发现所有未来实验的调度系统。不要在没有同步主文档与图注的情况下替换`evidence_snapshot.json`。

## 预览图注

[图1](../../figures/evidence_audit_20260908/01_balancing_and_capability.pdf)：左为M-PG每25次更新汇总的域有效token份额，虚线是prompt份额，而不是DR实际域系数；中为M-PG/500固定状态下I64局部loss的DR/DT/GT与zero-gradient held-out full-vocab KL下降，8条训练/4条held-out response、cap512；右为单training seed42的M-PG能力，IF/100无完成值，未插值。

[图2](../../figures/evidence_audit_20260908/02_sparsity_and_density.pdf)：前两图是PG与student-support Top64（ST64）checkpoint的BF16累计变化；单/多teacher同一步数的math暴露不同。右为S/M-PG250各自固定状态、PG/ST64/full-vocabulary/zero-gradient的局部BF16写回，cap128、每response抽2prefix；不表示真实在线单步或完整batch梯度。

[图3](../../figures/evidence_audit_20260908/03_teacher_overlap.pdf)：全部基于同一个M-PG500、training seed42；三组draw42/43/44仅为诊断bank。左图显示routed Jaccard的三bank均值；中图的误差棒是三bank观察到的min–max，不是95%置信区间；右图小点为每bank实测值，大点为每teacher pair的bank均值，圆形routed、三角形common；没有拟合趋势、显著性检验或独立teacher样本假设。M/C/I/S分别为math/code/IF/science。

[9×3布局图](../../figures/evidence_audit_20260908/figure_roadmap_9x3.pdf)仅展示作图计划和数据就绪状态，**不含实验测量值**。E是限定范围的探索数据已可画，P是部分数据，N是缺目标数据；均不等于完整论文证据验收。
