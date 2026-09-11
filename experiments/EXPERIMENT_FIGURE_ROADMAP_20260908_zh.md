# 当前实验图路线图

更新：2026-09-09。本版根据用户指定的两篇MOPD论文，调整为“领域能力轨迹 → 原始训练诊断 → 局部机制控制”。旧15图和27图槽登记保留在原数据包中作历史记录，当前交付以新版manifest为准。

[当前图集](../figures/paper_curves_20260908/index.html) · [正文9张PDF](../figures/paper_curves_20260908/main_figures.pdf) · [W&B raw曲线](../figures/paper_curves_20260908/raw_curves.html) · [逐图配置与依据](paper_curves_20260908/DESIGN_zh.md)

| 证据层 | 当前图 | 形式与范围 |
|---|---|---|
| 结果 | 四领域独立评估曲线 | 初始学生、Joint PG、Math-only PG、各域teacher；真实完整评估点 |
| 训练过程 | 原始log-ratio、token share、length、truncation、reward、loss、gradient | 未平滑；使用各日志自己的clock；不把microbatch当optimizer更新 |
| 归一化机制 | 同bank DT/GT相对DR的gradient cosine | 点图，每行保留实验条件；替代条件热图 |
| 教师输入控制 | Task-specific / identical-input teacher gradient cosine | 成对点与观测范围；共享teacher pair不作独立种子 |
| 监督目标 | 五个学生状态的BF16阈值曲线与PG相对范数 | 有序阈值用线图，无序损失类别用点图 |
| 关系矩阵 | PG500 shared-input教师两两support overlap | 全套唯一热力图；明确checkpoint、loss和阈值 |
| 既有续训 | Student64→I64归一化分支的评估 | 100步是父目标；部分250步评估缺失，保留缺口 |

正文9张含7张折线图、2张点图；全套30张含25折线/4点图/1热图。单张PDF/PNG/SVG独立导出，图内保留坐标与图例，中文问题和英文caption置于图外。

已冻结10条训练记录、52组逐题核验评估attempt、31组原局部探针。原始浏览器可逐run/metric查看，并下载raw CSV；所有数值保留来源文件与行号。详见[数据指南](paper_curves_20260908/README_zh.md)和[验证](paper_curves_20260908/validation_report.json)。

当前缺口：policy entropy未记录；部分续训250步Math/IF评估本地未见完整产物；实际在线BF16逐步测量和跨续训统一GPU成本未在本次补齐。没有把旧FP32 update字段、局部BF16模拟或teacher loss当成这些证据。

没有启动新训练、评测或探针。此前已取消的从Base重训I64、新训练seed、cap训练仍不恢复。本次只整理已有数据和改图。
