# 论文修改说明

本轮直接修改 LaTeX 正文、附录和图表，不改变实验记录、表格数据或公式。

## 摘要七点对应

1. **研究问题**：改为 loss normalization、vocabulary supervision 和 optimizer dynamics 如何影响参数更新及任务表现。论文固定每个领域的 prompt 数量，比较损失中的有效领域权重；并未系统改变数据采样比例，因此不将研究范围泛化为 data mixing 设计。
2. **研究方法**：摘要用模型与四个任务领域交代研究设置。具体的相同参数/Adam 状态单步对照留在正文解释。
3. **领域与回答权重**：明确“各领域的总 loss 权重”和“同领域内按 token 或按 response 平均”的区别。Shared Adam history 改为累积的一阶、二阶矩；local updates 改为单次 optimizer step 的参数更新。
4. **精度和优化器**：明确 FP32 master-weight change 与 BF16 stored-weight change；97%左右是累积参数变化的非零率，不是 raw-gradient 非零率。解释 Adam 即使当前梯度为零，也会因先前累积的一阶矩更新参数；减去这个基线，是比较加入当前教师梯度后参数更新如何改变，并非线性分解教师贡献。
5. **词表监督**：明确比较的是每个位置纳入 loss 的候选 token 数量，模型词表大小不变。Top-64 的 student–teacher 交集平均覆盖超过99.9%的学生概率质量；末期 I64 相对 PG 的数学和 GPQA 差异分别为+2.60和−1.77个百分点，配对95%置信区间均包含零。
6. **额外测试和模型选择**：从摘要删除低信息量的概括。正文直接说明：retention 约束在10条轨迹中均没有改变所选 checkpoint；PG-DT/Adam 选择250步，但额外测试均分低于500步。选择用 development scores，额外测试用于检验已选模型。
7. **结论**：直接说明 BF16 变化稀疏不代表优化过程只更新少数参数，Adam 更新方向相似不代表当前教师梯度相似；这些统计不能替代逐任务性能评估。

## 全文与图表

- 统一 loss normalization、domain loss weights、response-length weighting、Adam moment estimates、single-step update 等术语，并为单步对照和完整训练结果标明不同解释范围。
- 让段落和小标题优先陈述实测结论，保留训练种子、置信区间、学习率和算力条件等限制。
- 图2、5的原始高度由147.6pt降为123pt，图4由219.6pt降为174pt，分别缩短16.7%、16.7%、20.8%。通过重新定位绘图区、坐标轴、文字和图例实现，保留字号、标记大小和数据数值。
- 图注缩短，核心结论加粗。
- 修正附录原有的一处文表不一致：所引表报告的 instruction-following raw-gradient norm 为6.13×10⁻⁹，正文现与表一致，表格数字未改。

## 表达方式参考

- [DAPO，尤其token-level policy gradient loss](https://arxiv.org/html/2503.14476v2)
- [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/html/2503.20783v2)
- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)

上述论文用于参考术语、论述结构和相关工作定位；本论文的实测数字仍来自仓库内原有结果。
