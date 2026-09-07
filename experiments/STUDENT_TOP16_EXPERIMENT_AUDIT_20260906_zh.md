# Student Top16 切换后的论文实验盘点

更新：2026-09-06。适用论文：**Token Balancing, Update Sparsity, and Supervision Density in Multi-Teacher On-Policy Distillation**。

最新口径：在线 top-k 使用 **student Top16**；参数变化、稀疏性和教师更新重叠**只统计 BF16**。本清单不要求 FP32 参数测量或精度对照。旧 GPAS 实验计划不属于当前必做范围。

本轮建议是：**重跑四组 student Top16，完成两组现有 PG，补齐能力评估和共同 checkpoint 诊断**。本轮完成停止与盘点；下列新训练尚未启动。

补充复核：[论文数据覆盖与工期](STUDENT_TOP16_COVERAGE_AND_RUNTIME_20260906_zh.md) 区分 Qwen3 / SmolLM3 和种子预算，并补列当前文稿承诺的长度上限、batch size 敏感性及真实单步 BF16 记录需求。六组在线训练本身不等于论文数据齐全。

## 1. 已有证据与可以复用的结果

状态核验时间：2026-09-06T23:05:51.492978+00:00。状态来源见 [盘点快照](student_top16_inventory_20260906.json)。训练进度持续变化，以快照时间为准。

| 现有运行 | 最后记录的更新数 | 最近完整 HF checkpoint | 处理 |
|---|---:|---:|---|
| M-PG，seed 42 | 409 | 250 | 继续到 500，保留为新 Top16 对照 |
| S-PG，math，seed 42 | 239 | 100 | 继续到 500，保留为新 Top16 对照 |
| M-TK-DR，旧 teacher Top64 | 444 | 250 | 已停止，保留旧配方结果 |
| M-TK-DT，旧 teacher Top64 | 439 | 250 | 已停止，保留旧配方结果 |
| M-TK-GT，旧 teacher Top64 | 425 | 250 | 已停止，保留旧配方结果 |
| S-TK，旧 teacher Top64 | 278 | 250 | 已停止，巡检已持久停用，旧配方恢复入口已拦截 |

初始 student 四域能力评估与四位 teacher 各自本域的参考评估均已完成，可以复用；计数和评分见 [教师参考结果](teacher_reference_results_20260906.json)。六组训练均未找到完整的训练后能力评估，因此训练 reward 曲线不能替代论文能力结果。历史失败尝试留下的 `provenance_finish.json` 不能用于判断当前 PG 进程是否存活。

已有 [BF16 全参数扫描](bf16_effective_changes_20260906.json) 可以复用。以文献的 `torch.isclose(delta_bf16, 0, atol=1e-5)` 口径，第 250 步有效变化参数比例为：M-PG **5.90%**，旧 M-TK-DR **9.92%**、DT **11.72%**、GT **12.01%**，旧 S-TK **9.92%**；第 100 步 S-PG 为 **2.55%**、旧 S-TK 为 **5.33%**。这些是旧配方的探索性结果，不能直接替代 student Top16 实验。

已有 step-0 公共前缀 probe 完成了第一个 batch，第二个 batch 只有部分记录；进程已因内存压力停止，但 manifest 仍写着 `running`。它使用旧 teacher Top64，不能算作新 Top16 诊断完成。可以保留其前缀、教师分布评分及完整 batch 的历史记录。

Student Top16 已有参数解析、246 项通过的回归记录，以及 35 个位置、560 个选中 token 的真实 teacher 评分对齐验证；这些证据位于 `slime_opd_geometry/local/topk_open_mopd_check_20260906/`。尚未发现新配方完整的在线训练结果。

## 2. 先冻结新 loss 和测量口径

Student 在 rollout 的每个前缀选出 16 个 token，teacher 对同一组 IDs 打分；actor 保留 rollout 选集，重新计算 student 的全词表 log-prob。当前实现为：

```python
advantage = (log_p.softmax(-1) * (log_q - log_p)).detach()
loss = -(advantage * log_p).sum(-1)
```

这使用 Open-MOPD 的 student-support advantage 形式；其上游实现见 [固定版本的 actor 代码](https://github.com/BytedTsinghua-SIA/Open-MOPD/blob/4809a96cf85a869106ff0ff3f37d0a51e12010ae/training/verl/verl/workers/actor/dp_actor.py#L672)。本仓库当前配置不加入 PPO ratio 或 PPO clipping。应明确记录候选来源、K、集合内 student 权重归一化、advantage detach、PG clipping=0、reduction 和概率质量。

主参数统计统一使用保存的 **BF16 student 权重**：

- 累计变化：`delta = theta_bf16(t) - theta_bf16(0)`。
- 主指标：沿用现有 BF16 扫描的 `atol=1e-5` 有效变化比例，同时报告 L2 和逐层比例。若增加阈值曲线，所有配置使用同一组预先固定的阈值。
- 教师局部更新：从同一 student checkpoint 和同一 Adam 状态分别执行教师分支，统计 **BF16 写回之后**的参数差值与 support。
- 单步变化和累计变化分别标记；不得把相隔多个 checkpoint 的差值称为单步更新。Top-1/5/10% support 若含大量零值或 ties，应记录实际选择规则；全零分支不强行选出一个子网。

现有完整 probe 仍按旧参数统计路径输出，需要先对齐为上述 BF16 结果口径。现有 BF16 HF checkpoint 可以直接用于累计变化扫描。固定 checkpoint 的分支恢复需要保留真实优化器状态，但不为其另设参数统计实验。

新运行使用新 output root、run ID、生成配置和 provenance。当前 `local/paper_formal_20260906/generated/protocol.json` 仍记录 teacher Top64，不能只改启动命令后继续沿用这一份冻结协议。旧 Top64 checkpoint 不用于新 Top16 正式训练续跑；两组 PG 则保留自己的原始运行身份。

## 3. 必须补齐的六组在线实验

| 配置 | loss | reduction | 新工作 | 论文用途 |
|---|---|---|---|---|
| S-PG | sampled-token PG | response mean | 完成现有运行 | 单教师稀疏性、监督密度 |
| S-ST16 | student Top16 | response mean | 从共同初始化新跑 | 同上 |
| M-PG | sampled-token PG | domain-response | 完成现有运行 | 多教师稀疏性、监督密度 |
| M-ST16-DR | student Top16 | domain-response | 从共同初始化新跑 | 三项研究的共享参照 |
| M-ST16-DT | student Top16 | domain-token | 从共同初始化新跑 | 归一化对照 |
| M-ST16-GT | student Top16 | global-token | 从共同初始化新跑 | 归一化对照 |

命令入口仍是 `s-tk`、`m-tk-dr`、`m-tk-dt`、`m-tk-gt`；建议显式指定如 `s-st16-s42`、`m-st16-dr-s42` 的新 run ID，以区分历史结果。

保持 Qwen3-1.7B Base、四位现有 RL teachers、math 单教师参照、seed 42、相同 prompt schedule、每步 64 responses、500 更新、训练上限 4096、相同 Adam/学习率/裁剪配置。多域每步各域 16 responses。四组新训练合计 **2,000 次更新、128,000 responses**，另加 PG 剩余预算。

先用四域 Top16 做 2–5 更新的端到端小运行，再用约 20 更新测吞吐；检查每位置 16 个 ID、teacher 对齐、有限 loss、三种 reduction 的累计分母、保存/恢复，以及教师评分时间。已有单次评分验证不能代替这一步。teacher 评分按每 32 个位置请求选集并集，长响应会产生多次请求，因此不能根据 K 从 64 降到 16 就假定训练更快。

每组保存 BF16 权重于 0、1、50、100、250、500；按可复用的 checkpoint 直接计算稀疏曲线。单教师每步 64 条 math，多教师每步 16 条 math：同一步数的本域暴露相差四倍。跨单/多教师比较须同时报告域内 responses/tokens；需要精确配对的点可提前安排额外保存，不必扩展为新训练网格。

## 4. 三项论文研究各缺什么

| 研究 | 必须补的实验 | 应交付的图表 |
|---|---|---|
| Token balancing | 同一个固定 batch 应用 DR/DT/GT；核对有效 token 分母、域权重、长度与 response-gradient 的分解；三条新 Top16 在线分支完成能力评估 | 各域 token 份额与长度；三种平均方式的逐域能力、macro-average、worst-domain change |
| BF16 更新稀疏性 | S-PG/S-ST16/M-PG/M-ST16-DR 的 BF16 checkpoint 扫描；共同联合 checkpoint 上各 teacher 的独立 BF16 更新 | 单/多教师有效变化比例、L2、逐层曲线；教师 support overlap 热图 |
| 教师差异与更新差异 | 同一公共 prefix bank 上四位 teacher 全词表 JS；配对同一 checkpoint 的 BF16 support 距离；补全同 prefixes 的教师分支控制 | 六个 teacher pairs 的 JS 对 `1−Jaccard` 散点及轨迹 |
| Supervision density | 在相同单教师/多教师 checkpoint、prefixes 和优化器状态上比较 PG、student Top16、full-vocabulary reverse KL，并统计各自 BF16 写回变化 | 三种 loss 的 BF16 变化比例、幅度、support 对照表，附对应在线能力 |

归一化的固定-batch 审计还应包括一组不等 quota 检查，分别采用匹配 prompt 份额的域权重和均匀域权重。它是小型诊断，不需要再开两条 500 步训练。默认 probe 每条 response 只抽少量位置；若声称验证整条 response 的训练梯度，要另用能承受的真实固定小 batch 覆盖全部有效位置，不能把抽样位置结果写成完整 response 的测量。

教师分支从相同 checkpoint 独立恢复；增加零梯度优化器分支和按层匹配选择数的随机 support 参照。主 support 由 BF16 参数变化定义。独立单教师训练终点只提供参照，不能代替教师对同一个联合 student 的局部影响。

共同 checkpoint 建议覆盖 0、50、250、500。初始化的模型状态可复用；其后对 S-PG、S-ST16、M-PG、M-ST16-DR 各做早/中/晚三点，共 **1 份可复用初始状态 + 12 个训练后 checkpoint**。每个点复用三种 loss 分支，M-ST16-DR 点再复用三种 reduction。先完成一个小点并测内存，再扩展；每点先做至少 4 个独立诊断 batch，匹配单域 response 数及有效 token 口径。

旧诊断曾产生约 623 GiB 主机内存占用。新 runner 应逐分支/逐层处理并及时释放临时张量，避免同存所有 teacher×loss 的全参数副本。首先验证一个 BF16 诊断点完整结束，再排上述矩阵。

四位 teacher 只有六个 pairs，且 pairs 共享教师；重复 batch 和 checkpoint 不能当作独立教师样本。先逐对展示实际结果，再报告相应抽样不确定性。

Top16 的候选选择与权重归一化都与旧 loss 不同。因此此实验比较实际 loss 配方；只有固定 prefixes、权重和模型状态下的局部对照才能进一步解释差异。`student_top16_normalized_logratio` 可以为负，不能将其当作保证非负的 KL 或直接作为能力改善证据。

## 5. 能力评估与统计重复

优先安排 **10 套训练后四域评估**：六组全部做 step 500，四组多教师配置再做 step 250；初始 student 的 step 0 已完成可复用。这给归一化三分支提供共同的 0/250/500 能力点。若需要更细早期曲线，再补三组 Top16 的 step 100。

每套使用 MATH-500 pass@1（500）、固定 LiveCodeBench slice pass@1（128）、IFBench strict（300）、GPQA-Diamond average@4（198×4），共 **1,720 条回答**；10 套合计 **17,200 条评估回答**。保持现有 decoding、评分器和原生上下文限制，记录截断、答案完成率与 SandboxFusion 结果。

Teacher 与初始 student 的已有参考采用了不同的 prompt suffix。可以复用为各自实际评估配置的参考；若写纯权重差异或 teacher–student 差距的结论，须增加同 prompt 格式控制。student 各实验分支的能力对照必须保持相同模板。

Seed 42 的六组用于第一轮完整发现。要把比较写成稳定结论，建议随后对六组配置各补 seed 43、44，达到 **3 个配对训练种子**，新增 12 条完整训练。预算受限时，先补 M-PG 与 M-ST16-DR，但关于 DT/GT 和单教师的结论仍需标明只有一个种子。评估题目的 bootstrap 不能替代训练种子重复。

每张图标清运行身份、loss、BF16 统计定义、checkpoint、训练 seed、诊断 batch、有效 token 暴露和实测 GPU 时间。实际运行共享 teacher GPU、不同节点硬件及诊断开销需要分别记账。

## 6. 优先级与暂不扩展的项目

1. **立即准备**：冻结 student Top16 新协议和 BF16 统计入口；完成小型在线验证与一个可完成的局部诊断点。
2. **第一优先**：四条新 Top16 + 两条 PG 完成；产出 BF16 checkpoint 曲线和上述能力评估。
3. **同等核心**：利用已保存 checkpoint 完成教师 overlap、teacher JS 和三 loss 的局部对照。这些是论文中心证据，不能仅靠跑满训练代替。
4. **形成可靠结论**：补关键比较的配对种子，再决定是否扩展第二模型。
5. **可选**：SmolLM3 的单/多教师 PG–Top16 四配置；需要跨模型归一化结论时再加 DT/GT。1024/4096 cap、K sweep、student/teacher support 消融、更换教师中间 checkpoint、完整在线 full-vocabulary、MC 方差分解只在相应具体结论需要时追加。

无需重跑 GPAS、D3、raw-noise、PCGrad 等历史方法，也不必为六个 teacher pairs 重新训练一批专家。

## 7. 文稿需要同步的地方

`004_preliminary.tex` 的主 top-k 定义、`005_normalization.tex` 的固定-batch loss、`007_supervision_density.tex` 的表格及 `005_experiment.tex` 的配置说明仍有 teacher Top64。应统一改为 student Top16，并按本次决定统一 BF16 参数统计口径。旧 Top64 的 probability drift 与稀疏性数字保留为明确标注的历史探索性结果。

现有 `experiments/README.md` 中“trainer 与实测结果尚不存在”的表述已经过时，应以本次盘点中的实际产物和完成状态为准。完成新实验之前，不预写 Top16 更稀疏、归一化改善能力或 teacher JS 与子网距离正相关。
