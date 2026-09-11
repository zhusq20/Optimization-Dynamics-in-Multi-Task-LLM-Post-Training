# 实验与作图后续工作：2026-09-08 修订

本机简化后的31组机制探针已于20:45 UTC全部完成。实验运行和绘图数据已经落盘；现行范围以[实验图路线图](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)为准。

## 已完成

| 工作 | 完成范围 | 图槽 |
|---|---|---|
| 监督密度 | 五个学生状态×四bank，共20组；PG/ST64/I64/teacher64/full/zero及PG额外draw | F7 |
| 归一化 | PG250/500各两批、一个长回答批次、同一不等quota的两种域权重，共7组 | F1/F2 |
| 教师对照 | PG250/500各两bank，共4组；routed/common、PG/I64、JS、zero和随机参照 | F5/F6 |
| 绘图 | 15张独立图，6张正文候选；F5成对范围、F6同序矩阵、F7明确参照；31组源数据与脚本 | F1/F2/F5/F6/F7 |

[独立图浏览](../figures/local_probe_results_20260908/index.html) · [数据与复现](local_probe_results_20260908/README_zh.md)

## 用户已取消

- 不从Base重新训练交集方法，包括DR/DT/GT和单任务对照。
- 不新增任何训练seed，包括原计划PG/I64的seed43/44。
- 不再为这些新训练安排能力评估或BF16在线记录。
- 当前范围不另开cap1024训练。局部cap512/2048只用于机制检查，不冒充在线cap比较。

旧清单中的D1/D2、种子扩展及新增训练建议已失效；原文备份在backup/2026-09-08_before_local_probe_figures_221426/experiments/。诊断bank的draw42/43不是训练重复。

## 后续数据整理

1. 按F1c/F2b→F5a/F5b→F7a/F7b六张正文候选组织机制结果，其他九张作为补充诊断；参考[逐图审稿检查](local_probe_results_20260908/REVIEWER_FIGURE_AUDIT_zh.md)，保留负结果与限制。
2. 接入其他节点**已有**训练的核验能力结果与累计BF16扫描，补F3/F4/F8；先读完成标记和方法历史，不根据旧步数表重复启动任务。本次没有重新盘点这些外部运行。
3. F8a真实在线BF16单步、F9c batch-size sweep仍没有被这31组覆盖。缺失处保留，不用局部提议或不等quota替代。
4. 正文旧目标定义、配对种子与在线cap承诺应按当前范围删改。本文件是数据整理清单，不启动任何新实验。

绘图格式沿用用户要求：每个panel独立导出，放大字，图内不放标题或解释性文字。F5–F7合并不同checkpoint/loss；不再用密集散点或多重图例编码，解释移入FIGURE_CAPTIONS.md。
