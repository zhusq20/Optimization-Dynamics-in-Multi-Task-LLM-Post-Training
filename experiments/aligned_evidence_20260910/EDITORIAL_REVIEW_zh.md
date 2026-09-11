# 恢复三个原问题的写作核对

2026-09-11，依据用户指定的 [7417292 版本正文](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/7417292c57aa0cbe8486cad4986aa5617cd0f681) 修正研究主线。

| 原布局要求 | 本版处理 |
|---|---|
| 引言列出 token balancing、update sparsity、supervision density 三问 | 恢复原版引言背景和三项顺序；摘要、讨论逐项对应。 |
| Section 3：平均方式与 response length，然后实验 | 三行平均方式表、简短恒等式、三分支能力图和共同 checkpoint 表；token 份额与区间在附录。 |
| Section 4：稀疏性 → 教师 overlap → 教师距离 | 恢复三个原问题式小节标题；累计变化、教师 overlap、JS 对应三张正文图。 |
| 参数更新为主，梯度作辅助 | 单步/累计、FP32/BF16 分开；梯度 overlap 图注明确测量对象；单教师几何与逐教师 step overlap 明确缺测。 |
| Section 5：单/多教师 PG/top-k 简洁比较 | 对比已有 online PG/I64，主图改为局部活动比例和能量集中，主表给出同 checkpoint 能力。T64/full 仅作局部参照。 |
| 额外机制不能取代主问题 | Adam 状态和输入控制降到附录；动作数/MC 方向研究、独立生成诊断、RL regularization 推导退出编译稿；取消掩码训练等扩展承诺。 |
| 保留有效实测证据 | 保留 13 套能力评测、原始评分与数据哈希。新增矩阵/表格由已有冻结值汇总，不生成实验结果。 |
| 实验范围同步 | 更新三问说明、实验 README、图表路线图、剩余比较；旧盘点明确标为历史写作建议。 |

本版有五张正文图、六张附录图。沿用现有数据核验脚本，核对来源哈希、评分、图源与论文实际引用，并编译检查引用和排版。最新运行结果见 `verification_report.json`。
