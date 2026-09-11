# 作图后续与现行范围

2026-09-09：当前[图集](../figures/paper_curves_20260908/index.html)已改为能力/训练轨迹主导，并补入原始W&B scalar镜像。配置见[设计说明](paper_curves_20260908/DESIGN_zh.md)。

下一步写作可直接使用正文9张候选及其caption。若接入已有运行后续产物，刷新exporter并核对checkpoint身份、完成标记与方法切换，再重画；不根据旧图槽自动启动任务。

仍缺的测量包括policy entropy、部分续训250步Math/IF完整评估、真实在线BF16逐步记录和统一GPU成本核算。当前图明确标注缺口。

此前取消的从Base起I64训练、新训练seed和cap训练仍取消。局部bank重复不替代训练seed，已有局部BF16提议不替代在线训练轨迹。
