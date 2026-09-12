# 当前编译稿的引文核验与实验借鉴

核验日期：2026-09-12。范围按 `iclr2027_conference.tex` 的递归 `\input` / `\include` 解析，排除 LaTeX 注释：当前正文及附录实际引用 **10 篇**，均已核实 arXiv 摘要页、题名、作者与年份。指定的两篇论文另读 PDF、HTML 正文及相关实验。现有 BibTeX 共 62 条；其余 52 条未进入当前编译稿，本次不将它们标记为已核验，也不为了扩充参考文献而加入正文。

## 1. 实际引用逐条核验

| BibTeX key | 可查原始来源及核读版本 | 当前用途与核验结果 |
|---|---|---|
| `ma2026mopdmultiteacheronpolicydistillation` | Ma et al., 2026, [MOPD](https://arxiv.org/abs/2606.30406v1) | 领域教师在学生生成轨迹上进行能力整合；题名、13 位作者及年份匹配。 |
| `gao2026openmopddiagnosingfixingcapability` | Gao et al., 2026, [Open-MOPD](https://arxiv.org/abs/2608.19098v1) | token averaging 与能力失衡；宜精确引用 §3.3、§4.1。作者与年份匹配。 |
| `mukherjee2025subnetworks` | Mukherjee et al., 2025, [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711v2) | RL checkpoint 更新稀疏及子网训练；作者、年份及 NeurIPS 2025 说明匹配。 |
| `yu2026denseupdates` | Yu et al., 2026, [Dense Supervision, Sparse Updates](https://arxiv.org/abs/2606.13657v3) | OPD checkpoint 稀疏、支持集重叠；v3 日期为 2026-07-30，六位作者匹配。 |
| `shen2026opdgeometry` | Shen et al., 2026, [On the Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.07082v3) | OPD 参数轨迹与累计更新子空间；v3 日期为 2026-06-14，九位作者匹配。 |
| `qwen2025qwen3` | Yang et al., 2025, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388v1) | 模型来源。原条目用 Qwen Team，现按 arXiv `citation_author` 改为 An Yang 等 60 位正式作者。 |
| `hendrycks2021math` | Hendrycks et al., 2021, [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874v2) | 支持 MATH 数据来源；MATH-500 的具体子集另由数据版本和评测清单确定。 |
| `jain2024livecodebench` | Jain et al., 2024, [LiveCodeBench](https://arxiv.org/abs/2403.07974v2) | 支持 benchmark 来源；v6 与本项目固定 128 题子集是评测协议，不是这篇 2024 论文的结论。 |
| `pyatkin2025ifbench` | Pyatkin et al., 2025, [Generalizing Verifiable Instruction Following](https://arxiv.org/abs/2507.02833v3) | 正确对应 IFBench；八位作者与年份匹配。 |
| `rein2023gpqa` | Rein et al., 2023, [GPQA](https://arxiv.org/abs/2311.12022v1) | 正确对应 GPQA；Diamond 子集与 average@4 的使用方式由本项目协议说明。 |

上述十条均补充 arXiv DOI，并将 URL 固定到核读版本。已有 cite key 保留，避免破坏正文引用。没有新增未经核验的论文。

评测来源可进一步追溯至 [MATH-500 数据卡](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)（test 为 500 行）、[LiveCodeBench 官方仓库](https://github.com/LiveCodeBench/LiveCodeBench)、[IFBench 官方仓库](https://github.com/allenai/IFBench) 和 [GPQA 官方仓库](https://github.com/idavidrein/gpqa)。这些链接标识 benchmark 来源；具体 revision、题目 ID、采样参数仍应从本项目冻结协议追溯。

## 2. 两篇指定论文可借鉴什么

### Yu et al.：稀疏更新的测量和教师重叠

已核读 [v3 正文](https://arxiv.org/html/2606.13657v3)。§4.1 / Table 2 使用 checkpoint 差值和绝对阈值；§4.3 / Table 3 比较不同算法、教师、数据下的终点支持集，其重叠高于相应独立随机基线。§5.1 区分数值满秩与谱能量集中。§4.4 的掩码重训是功能验证，不能由观察到的稀疏度自动推出。

对本稿的编辑建议与推论：

- 第二主线直接延续“哪些参数发生较大变化”这一问题。把 checkpoint 稀疏度与 FP32 优化器变化写清楚，在方法处一次定义，避免在每段重复防御性限定。
- 跨独立终点的支持集重叠，与同一个学生状态下逐教师更新的重叠，是不同的问题。后者是本稿需要自己的数据回答的部分；原论文不能补齐缺测。
- 若展示固定大小的 Jaccard，附录可给独立选择的参考值。对大参数规模、相同选择比例 ρ，随机参考约为 ρ/(2−ρ)；不再另造一组相近的“重叠分数”。
- 第三主线可用其结果动机化“词表监督更广是否改变参数变化”，无需增加 SVD 重构或掩码重训来复制整篇论文。

### Gao et al.：长度通过平均分母改变域权重

已核读 [v1 正文](https://arxiv.org/html/2608.19098v1)。§3.3 明确使用 token-mean aggregation；§4.1 Eq. (8) 对域损失乘以目标份额与实际 token 份额之比。§5 / Table 4 将该机制与后续 gap weighting、reward refresh 分开消融。§3.2 的教师分歧测量是同一 sampled token 的教师 log-prob 范围，并配合 token 干预。

对本稿的编辑建议与推论：

- 在同一批次、同一有效 token mask、同一目标域权重下，将 Eq. (8) 的权重代入全局 token mean，代数上就是本稿 DT。DR 进一步移除域内响应的长度权重。这个连接可以直接用于解释本文三种 reduction 的关系。
- “域权重”“token 份额”“响应长度”足以组织第一主线。不要把 token 份额改称实际梯度贡献；梯度大小和方向还取决于逐位置导数。
- 其 sampled-token 分歧不能代替本稿的梯度方向、参数支持重叠或全词表 JS。不能据此写成一般性的“教师不存在梯度冲突”。
- 不把 reward refresh、动态调度、冲突 token 删除作为本稿新增必做实验；这些机制与三条中心问题的直接联系较弱。

## 3. 引用措辞与证据归属

可写：`Open-MOPD motivates separating domain token weights from prompt counts`；`Prior work reports sparse checkpoint changes under dense OPD supervision`。随后用自己的完整比较支持本文观察。

Shen et al. 的摘要包含减少 update tokens 的控制实验，其对象是响应位置，不能直接引作每个 prefix 减少词表候选的 PG/top-k 对照。该文适合保留在 OPD 参数几何相关工作中。[原文](https://arxiv.org/abs/2606.07082v3)

外部论文的实验数值、趋势和干预效果只以外部观察引用；本稿未完成的实验图表用批注说明所缺条件。引用文献不能替代自己的缺测，也不将他人的定性结论直接写成本文验证所得。

核验检查：62 个 BibTeX key 无重复；以当前 10 个实际引用和仓库 ICLR bst 在临时目录运行 BibTeX，退出码为 0，无警告。
