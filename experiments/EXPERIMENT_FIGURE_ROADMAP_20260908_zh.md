# 论文图表与三个问题

2026-09-12 更新。全部复合图按每行三列排版；最后一行不足三张时保留空位，不添加填充指标。

| 位置 | 图 / 表 | 科学问题与数据范围 |
|---|---|---|
| Section3 | normalization_capability；三种平均定义表；50步分数表 | 三种平均方式的共同50步能力 |
| Section4.1 | cumulative_geometry | joint PG/I64共同1/50/100/250参数变化 |
| Section4.1–4.3 | 三个可见补充批注 | 单/多教师几何；逐教师step overlap；step与JS配对 |
| Section5 | supervision_density；PG/I64配对分数表 | 四bank完整局部比较；单教师至500、joint至250的完整能力配对 |
| Appendix | capability、normalization50、all_token_shares | 匹配能力轨迹、全部归一化配对区间、共同前50批token权重 |
| Appendix | teacher_overlap、teacher_js、teacher_input | 教师原始梯度、预测差异与输入诊断 |
| Appendix | adam_precision、thresholds | 原两bank的Adam和阈值控制 |
| Appendix | normalization_fixed_batch_table、endpoint_paired_table | 六个固定batch全部三对梯度；两个端点全部八个分数差值区间 |

共三张正文图、八张附录图。原始16套能力均保留证据包；文中比较表只取共同checkpoint（15套），GT100单点不补入缺DT100的三方实验。动作数、独立响应行为、GPAS与RL regularization不进入本次编译稿。

作图、编译与核验命令见[当前证据指南](aligned_evidence_20260910/README_zh.md)。
