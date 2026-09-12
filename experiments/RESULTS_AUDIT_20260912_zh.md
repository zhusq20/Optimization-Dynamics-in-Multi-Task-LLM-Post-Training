# 实验证据盘点与三问对应（2026-09-12）

本次直接读取源工作区 `../slime_opd_geometry` 的完成标记、原始评分文件和局部测量，没有启动训练、推理评测或探针。9 月 11 日盘点已落后于实际进展；下述新增结果已导入当前论文证据包。图表的完整性按它要回答的**完整比较**判断，而不是按某一分支是否有数字判断。

## 已完成的核心比较

| 问题 | 当前完整证据 | 正文用途与缺口 |
|---|---|---|
| 平均方式改变域权重及能力分配 | GT/DT/DR 在第 50 次更新均有完整四域评分；原始 token 份额轨迹；新增 100/250 更新各三个固定批次的 DR/DT/GT 梯度比较 | 三方能力图限于共同第 50 次更新；固定批次比较直接验证域内长度加权改变梯度方向。DT/100 及后续三方能力比较仍需整组补齐。 |
| MOPD 参数变化集中度 | PG/I64 联合分支在 1/50/100/250 更新具有完整累计 FP32/BF16 几何；250 是本次新增 | 可完整报告联合训练的参数变化。单教师在线几何没有配齐，单/多教师比较应放待补批注。 |
| 同一学生内教师参数重叠，以及教师 JS 与参数选择差异 | 原两批次教师原始梯度的 overlap 与 JS；新增局部实验也仍记录原始梯度 | 原始梯度可以作为附录独立测量。逐教师 optimizer-step 重叠和对应 JS 关系尚未完成，正文整组待补；独立单教师终点不能替代共同学生同一状态的逐教师更新。 |
| 监督密度改变集中度与能力 | 单教师 PG/I64 100/250/**500**，联合 PG/I64 50/100/**250** 完整能力配对；联合 250 累计几何；新增共同 M-PG/100 状态四批次的 PG/I64/T64/full 局部比较 | 新完整长回答局部比较适合替换旧两批次、256-token cap 主图。数学单教师在线几何仍待补。 |

## 新增完整能力结果

[能力数据](aligned_evidence_20260910/capability.json)现有 **16 套、27,520 条回答**；其中本机可逐回答核验 **12 套、20,640 条**，另外四套为保留的远端已验证单教师 100/250 摘要。每套包含 Math500 500、LCB 128、IFBench 300、GPQA 198×4，共 1,720 条。[GPQA 审计](aligned_evidence_20260910/gpqa_score_audit.json)累计重新评分 9,504 条，分歧为零。

| 新增配置 | Math % | Code % | IF % | GPQA % | 完成时间 UTC |
|---|---:|---:|---:|---:|---|
| S-I64 / 500 | 76.00 | 17.96875 | 18.66667 | 30.05051 | 09-11 02:56:52 |
| M-PG / 250 | 73.40 | 16.40625 | 26.00 | 30.17677 | 09-11 09:26:33 |
| M-I64-DR / 250 | 74.80 | 17.96875 | 26.00 | 33.96465 | 09-11 15:02:53 |

证据链为 `raw/{配置}_{更新}_complete.json`、`_index.jsonl`、`_eval.jsonl`、`_job.json`、`_verified_summary.json` → 原始四域回答 SHA-256 与逐题重算 → `capability.json`。新增源目录分别是：

- `outputs/mopd_qwen3_aligned_capability_20260910_sn4622128200/s-intersection64-s42/capability_eval/step_500/`。
- `outputs/mopd_qwen3_aligned_m_pg_20260909/m-pg-s42-g123/capability_eval/step_250/`。
- `outputs/mopd_qwen3_aligned_20260909_sn4622128200/m-intersection64-dr-s42/capability_eval/step_250/`。

单教师和归一化协议的初始化、student、teachers、prompt、response、evaluation、dataset 字段与共享协议一致，各 split 数据哈希一致。S-I64/500 另有 310 个张量键的转换一致性记录，`serialized_tensors_equal_converted_native=true`，见 `raw/S-I64_500_export_verified.json`。

[配对比较](aligned_evidence_20260910/paired_comparisons.json)现为 68 个领域比较，新增同设置同更新 PG/I64 配对。联合 250 更新时 I64−PG 为 Math +1.40、Code +1.5625、IF 0、GPQA +3.7879 个百分点；其 95% 题目 bootstrap 区间分别为 [−1.20, 4.00]、[0, 3.90625]、[−2.66667, 2.66667]、[0, 7.57576]。单教师 500 更新中，I64 的数学较高，PG 的 Code/GPQA 较高。可写观察到的取舍，不写普遍优势或统计显著优势。所有区间均条件于现有训练 checkpoint；四条 GPQA 回答按题目成簇重采样。

## 新增的直接机制证据

**固定批次归一化。** 六组完整结果见[数据](aligned_evidence_20260910/normalization_fixed_batch_20260912.json)和[附录表](aligned_evidence_20260910/normalization_fixed_batch_table.tex)。每个 checkpoint 用三个独立抽取批次，每批每域八条回答，共 32 条，使用完整回答的 I64 loss，cap 为 4,096。先逐回答平均 token loss 求原始梯度，再按 DR/DT/GT 的数学定义加权；表中 cosine 是这些**原始总梯度的夹角余弦**，不是优化器参数变化的夹角。每个 checkpoint 的三批次共用完全相同的保存状态哈希。

| 更新 | DR–DT 梯度 cosine 范围 | DR–GT 范围 | DT–GT 范围 |
|---|---:|---:|---:|
| 100 | 0.5780–0.9110 | 0.5180–0.6659 | 0.7010–0.9745 |
| 250 | 0.3990–0.7864 | 0.3302–0.4072 | 0.4953–0.9612 |

DR/DT 具有相同域权重，方向仍不同，直接支持“域平衡不能消除域内长度加权”。24 个域×批次×checkpoint 的 FP64 协方差恒等式检查最大相对残差为 1.53×10⁻¹⁵；它是实现核验，不需要升级成新指标或独立贡献。源 `normalization_probe.py` 同时检查 HF 权重与保存 BF16 权重逐元素一致、dropout 为零，并从同一 FP32 master 和 Adam 状态计算各分支。原始六组均冻结在 `raw/followup/normalization-mpg{100,250}-bank{1042,1043,1044}/`。

**联合 250 更新的集中度。** [新增在线几何](aligned_evidence_20260910/online_geometry_20260912.json)来自两条运行的完整 `paper/measurements.jsonl`，每分支包含 raw gradient、clipped gradient、actual update、FP32 cumulative delta、BF16 cumulative delta 五类全参数记录。

| 指标（%） | M-PG / 250 | M-I64-DR / 250 |
|---|---:|---:|
| BF16 累计非零坐标 | 5.8792 | 7.1275 |
| BF16 90% 能量所需坐标 | 3.0108 | 3.9363 |
| FP32 累计非零坐标 | 95.7287 | 96.0669 |
| FP32 90% 能量所需坐标 | 32.9427 | 34.8153 |
| 实际 FP32 单步非零坐标 | 86.3527 | 86.7006 |

因此，联合 I64 具有更广的累计 BF16 活动，两者的 BF16 变化仍集中。源原始文件按 `raw/{配置}_paper_20260912.jsonl` 单独冻结，原先在线日志与 `raw/online_context.json` 未覆盖。

**更完整的局部监督密度比较。** `density-mpg100-long-bank1042` 有四批、16 条回答、96 个 prefix；回答长度 81–4,096，仅两条达到 cap。四种 loss、四批、两种精度形成完整 32 条保存状态更新记录，见[冻结汇总](aligned_evidence_20260910/completed_local_followup_20260912.json)。

| loss | BF16 非零坐标均值 % | FP32 非零坐标均值 % | 最大 1% 坐标内 FP32 能量 % |
|---|---:|---:|---:|
| PG | 0.1494333 | 87.98207 | 8.8451 |
| I64 | 0.1501963 | 87.98888 | 8.8271 |
| T64 | 0.1501918 | 87.98883 | 8.8277 |
| Full | 0.1501955 | 87.99883 | 8.8276 |

BF16 最大 1% 坐标含全部变化能量。四种 loss、四批的 BF16 非零比例合计范围为 0.147584–0.151364%；因此正文适合给可读的均值或范围，不需要六位小数。其 routed teacher-top64 student mass 的逐回答均值范围为 99.27905–100%，总体 99.91789%。全部 raw gradient 都经过训练协议的 norm-1 clipping；当前读数不能单独将相近稀疏度归因于 clipping。

新旧 `local_distillation_loss`、`adam_chunk`、`adam` 的 AST 一致；新 `collect` 仅为 activation checkpointing 切换 train/eval，并检查 dropout 为零。新 PG action 在缓存 prefixes 上重新采样。完成标记记录的准备文件、主测量及完成文件 SHA-256 均通过，学生值精确匹配保存 BF16 权重。

另外两项已完成的四批局部数据是 `mechanism-i64dr100` 与 `mechanism-spg500`。它们都在各自 checkpoint 计算四域联合教师 loss，后者不能当作“数学单教师局部 PG/I64”实验。两项完整保存供追溯；它们没有关闭单教师在线几何或逐教师更新归因缺口。

## 全体历史结果与尚未完成的图表

本次扫描 `outputs/` 得到 **225 个完成标记、18 个顶层实验族**，较[9 月 11 日盘点](EXPERIMENT_INVENTORY_20260911_zh.md)的 218 个增加七个：上述三套 aligned 能力、aligned M-PG 500 步训练，以及旧协议 DT/GT/DR 500 步能力三套。完成标记包含运行尝试，不等于独立实验或独立种子。

历史六条 Base / Student64→I64 训练、GPAS v4/core、旧 31 组局部探针、Top16/BF16 诊断、单任务 GRPO/OPD、GRPO 跨域与顺序/混合训练、SmolLM3 两步管线等继续归档。它们的初始化、目标历史或研究问题不同，不补入当前 aligned 主比较。旧协议新增终点评测同样不能与 aligned 曲线合并。

`local/aligned_followup_20260911_sn4622129202/results/` 另有不计入上述 `outputs/` 数目的已完成后续实验：六个归一化批次、三套四批局部机制、三套动作采样诊断。动作次数控制与当前三问弱相关，继续归档。`mechanism-mpg500` 仅部分完成、`mechanism-i64dr500` 仅有配置；不要展示其中的部分 bank 结果。

当前需要整组批注的位置：

1. GT/DT/DR 的 100 及后续共同能力比较；不要只画 DR/GT 延伸。
2. 单教师与多教师的同更新参数变化比较；不要用联合曲线代替缺失的单教师组。
3. 同一学生、同一 optimizer 状态的逐教师真实/拟议参数 step overlap。
4. 与上一项逐教师 step 配对的 teacher JS–参数距离。
5. 单教师 PG/I64 参数集中度及其对应局部比较。

以上缺口只需列出回答原问题必要的测量对象；不应扩展为 Adam 状态扫描、更多动作数、掩码训练或新采样器课题。已完成的梯度 overlap 附录可以直接写：routed top-1% Jaccard 的均值 PG 0.09505、I64 0.09986；common-input 均值 0.38078、0.45771；teacher JS 范围 0.001508–0.027142。不能把这些梯度选择叫作已测得的 optimizer-step 子网。

本次新增导入由 [能力导出脚本](export_aligned_evidence.py)与[后续结果导出脚本](export_followup_evidence.py)复现。新增后 `manifest.json` 共 206 条来源记录，其中 158 个唯一 frozen raw 文件已逐个 SHA-256 核验通过；图、表和编译稿的最终检查见随后更新的 `verification_report.json`。大体积原始输入仍按仓库既有规则留在本地证据目录与源工作区；可读汇总、表格、源文件哈希和导出脚本进入当前文稿工作树。
