# 三个原问题的剩余比较

2026-09-11：以用户指定的 [7417292 正文布局](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/7417292c57aa0cbe8486cad4986aa5617cd0f681) 和[现行三问说明](../MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md)为准。

| 原问题 | 当前已有 | 直接缺口 |
|---|---|---|
| 平均方式与能力平衡 | GT/DT/DR 的共同 50 步能力，DR/GT 的 100 步能力，token 份额与解析恒等式 | 后续共同 checkpoint 的三分支能力；对应固定 batch 的 reduction 测量尚未纳入 aligned 证据 |
| 更新稀疏性、教师重叠、教师距离 | 联合 PG/I64 的在线参数几何；M-PG/100 的逐教师梯度 overlap 和 JS 配对 | 单教师参数几何；同一联合学生和优化器状态下的逐教师拟议更新 overlap，以及它与 JS 的对应 |
| PG/top-k 的稀疏性与能力 | 单教师 100/250、联合 50/100 的配对能力；联合 PG/I64 累计变化；M-PG/100 的局部 full-vocabulary 参照 | 单教师对应的参数变化与 full-vocabulary 局部参照；原分支尚未完成的同 checkpoint 结果 |

优先整理已有分支和 checkpoint。新增测量必须直接回答表中某一项，不把参数支持集分析替换成采样方差、动作数、优化器状态、复杂输入控制或独立算法实验。当前没有启动任何新任务，表中缺口也没有被写成已完成结果。

训练完成标记、能力评测完成标记、在线参数记录与局部拟议更新分开核对。旧 Base / Student64→I64 前史的数据不补进 aligned 比较。局部 bank 重复不当作训练种子，单教师能力结果不当作单教师稀疏性结果。
