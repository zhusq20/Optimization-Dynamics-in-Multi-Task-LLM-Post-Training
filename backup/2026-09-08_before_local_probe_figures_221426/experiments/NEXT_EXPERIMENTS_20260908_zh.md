# 下一批实验：先补能力闭环，再补当前 loss 的机制对照

更新：2026-09-08。训练进度采用 **01:50 UTC** 的[只读复核快照](figure_audit_20260908/next_experiment_progress.json)，已完成结果采用 **01:10 UTC** 的[证据快照](figure_audit_20260908/evidence_snapshot.json)。这是执行计划；本轮未启动新训练、评估或 GPU 诊断。

**最优先的任务是 M-Overlap64-DR/DT/GT 三组 250 步四域评估。** 当前论文最缺的是归一化是否影响实际能力，而已有梯度/局部 KL 测量无法回答这个问题。并行准备 Overlap64 局部对照与 BF16 扫描，现有训练继续到 500；首轮能力与机制结果齐全后，再增加独立训练种子。当前排期不额外增加四条 seed42 主训练。

当前四条 TopK 实验组统一称为 teacher–student Top64 overlap（Overlap64 / I64）。实验组名称用于组织任务；原始 checkpoint、loss 和测量记录保留各自身份。27 个子图的定义、验收条件见[作图路线图](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)。

## 1. 现在能用什么

| 实验组 | 已记录更新 / 500 | 已完整导出的 HF 点 | 下一项 |
|---|---:|---|---|
| M-PG | 500 | 1 / 50 / 100 / 250 / 500 | 复用 250 / 500 的模型与 Adam snapshot |
| S-PG | 313 | 1 / 50 / 100 / 250 | 等 500；250 可立即做局部诊断 |
| M-Overlap64-DR | 235 | 当前目录 250 / 500 待导出 | 250 四域评估 |
| M-Overlap64-DT | 246 | 当前目录 250 / 500 待导出 | 250 四域评估 |
| M-Overlap64-GT | 243 | 当前目录 250 / 500 待导出 | 250 四域评估 |
| S-Overlap64 | 169 | 当前目录 250 / 500 待导出 | 250 机制诊断，500 四域评估 |

按快照计，五条未结束分支还需 **1,294 updates / 82,816 training responses**。M-PG/250、M-PG/500、S-PG/250 的四域评估已经完整；M/S-PG/100 的 math/IF 重试仍在运行。日志中的 update 到达目标不等于模型已导出完成。

下一批主输入目录相对于 `slime_opd_geometry/`：

| 配置 | run 目录 |
|---|---|
| M/S-PG | `outputs/mopd_qwen3/formal_20260906/{m,s}-pg-s42/` |
| M-Overlap64-DR/DT/GT | `outputs/mopd_qwen3/student_top64_intersection_20260907/m-itk64-{dr,dt,gt}-s42/` |
| S-Overlap64 | `outputs/mopd_qwen3/student_top64_single_intersection_20260907/s-itk64-s42/` |

HF 路径从各 run 的 `checkpoints/mopd_checkpoint_index.json` 按 `optimizer_step` 解析，不手算 `iter_*` 名字。局部诊断还需 `paper/checkpoint_step_0250.pt` / `checkpoint_step_0500.pt`；一个完整模型/Adam snapshot 约 24 GB。

## 2. 按这个顺序排任务

| 优先级 / 任务 | 具体实验 | 启动依赖与数量 | 直接补哪些图 |
|---|---|---|---|
| P0 / A1 | M-Overlap64-DR、DT、GT 的 250 步四域能力 | 各自完整 HF 导出；3 套，5,160 responses | **F3a/b/c**，兼 F8b/c |
| P0 / B0 | 给局部 probe 增加 I64 loss、I64 教师分支、读取固定 bank 的入口 | 先完成公式/梯度/Adam 写回数值核验；现有 runner 不能直接改参数运行 I64 | F5–F7 的必要程序准备 |
| P0 / C1 | 新 checkpoint 的 BF16 累计变化扫描 | 四个 Overlap64 配置 × 250/500，加 S-PG/500，共 **9 个新扫描**；CPU 流式读取 | **F4a/b/c** |
| P0 / C2 | 增加真实 BF16 在线单步记录 | 为下一批训练/经过验证的后续启动配置；连续窗口、前后 BF16 值 | **F8a** |
| P1 / B1 | M-PG/500 固定 bank 的 PG / ST64 / I64 / full / zero 对照 | B0 完成；先复用已保存的 3 组 prefix bank，每组重算所需分布 | **F7a/b/c**；I64 routed/common 分支补 F5/F6 |
| P1 / B2 | I64 fixed-batch 的 DR / DT / GT / zero 对照 | M-PG/250、500 可用；每状态 4 banks，先每域 2 responses、cap512 | **F1c、F2a/b/c** |
| P1 / B3 | 时点与学生状态扩展 | M-PG/250、S-PG/250，随后 M-Overlap64-DR 和 S-Overlap64 的 250/500；统一诊断协议 | F5–F7 |
| P0 / A2 | 五个未完成配置的 500 步四域能力 | 三个 M-Overlap64、S-PG、S-Overlap64；5 套，8,600 responses | **F3、F8** |
| P1 / D1 | 关键比较的新增训练种子 | 先 M-PG 与 M-Overlap64-DR 各 seed43/44，共 **4 条 × 500**；正式运行前固定完整协议 | 核心图不确定性、F9a |
| P2 / D2 | DT/GT、单教师的种子扩展，cap 与 batch-size 控制 | 前一轮主矩阵收齐后；正文保留相应主张时完成 | F3/F8 稳健性、F9 |

优先级表示证据价值与时间依赖，B0/C2 是程序准备，不应占用等待 checkpoint 的评估 GPU。A1 与 CPU 扫描可独立推进。现有 PG/100 重试完成后先做逐题核验和去重，不重复占用评估槽位。

## 3. 第一批：三组 250 步能力评估

每套固定 **MATH-500 500 条 + LiveCodeBench 128 条 + IFBench 300 条 + GPQA 198 题 × 4 回答 = 1,720 responses**。使用与已有 PG/250、500 相同的数据、student prompt/template/suffix、采样参数、context/cap、scorer 和依赖版本。GPQA 指标是 avg@4，不是 pass@4。

每题保存 `checkpoint_hash、optimizer_step、benchmark、prompt_id、sample_id、reward、generated_tokens、truncated、termination/status、scorer_version`。输出四域分数、相对初始学生变化、macro、worst-domain change，以及每域截断率；由这些数据一次生成 F3 的三个子图，不另跑三次评估。F3b/F8c 的 token 横轴连接去重后的真实训练 token 日志。

已有 M-PG/250→500 的 macro 从 30.34 降至 29.81，且 code/IF 截断明显增多，因此**能力与长度/截断必须同时记录**。对缺失、报错、截断响应沿用冻结的评分与分母规则，不按终点得分挑选一次重试。

最小主矩阵是六配置 500 终点，加四个多教师配置 250 中点，共 10 套；已完成 M-PG 的两套，尚缺 **8 套 / 13,760 responses**。S-PG/250 已有数据另行复用；S-Overlap64/250 建议在 A1 后补一套（另 1,720 responses），这样单/多教师的 250 步比较更完整。若把六组的 100/250/500 都补齐，则还有 23,460 responses 的缺口，不必作为下一批的前置条件。

## 4. 第二批：用同一状态检验当前 loss

### B1/B3：监督密度与教师 overlap

第一轮使用 M-PG/500 已保存的 draw42/43/44：每域 1 response、cap256、每条至多 4 prefixes。保存的 `prefix_bank.json` 含 prompt/response token IDs、位置和 sampled actions，可以复用；没有持久保存完整 student/teacher 概率数组，仍需重算分布。新增 runner 必须支持加载 bank；仅设置同一个随机种子后重新生成不能保证得到完全相同的 prefix。

在每个 bank 内从**同一个学生参数与同一个 Adam 状态**独立计算 zero-gradient、sampled PG、ST64、I64、full-vocabulary 分支。保留已有 PG/ST64/full 的结果，并核对同协议重算误差。I64 实现应与训练 loss 一致：S 是冻结 student Top64，T 是 teacher Top64，I=S∩T；只在 I 内累加，但权重分母为原 S 的 student 概率质量，advantage detach；空交集为零且不减少外层位置分母。记录交集大小、空交集率和 retained weight，不能用“64 项”代替实测支持数量。

I64 的数值核验至少覆盖完整交集、部分交集、空交集，冻结支持/advantage 与全词表归一化的梯度，以及相同 optimizer/clip 下的 BF16 写回。full-vocabulary 只做局部参考，不新增其在线训练。

同一 joint checkpoint 上另做四位教师的 routed 与 common 两类独立分支；当前 runner 这两类仅实现 PG，I64 教师分支也需要扩展。每模式每 bank 有 6 个教师对；共同前缀 JS、Jaccard、signed cosine、zero-gradient overlap、支持大小和按层随机参照成套保存。随机参照的期望交/期望并之比需如实命名。

第二轮在 M-PG/250、S-PG/250 采用同样的 cap256/4-prefix 协议与 3 banks；单教师状态仅做 math 的 loss 分支，不凭空生成四教师的单域数据。之后接上 M-Overlap64-DR/250 与 S-Overlap64/250，再补 500。既有 cap128/2-prefix pilot 单独保留。正式扩展时补第 4 个 bank 与早期状态；draw seeds 不是 training seeds。

每个分支输出 raw-gradient norm、clip coefficient、FP32 Adam proposed update、BF16 写回的 L2/阈值曲线/逐层数据、与 full-vocab 及 zero-gradient 的距离。support 的等预算检查只选择所有被比较分支均有非零支持的比例；先考察 0.01/0.05/0.1%，不足时降低共同预算或记 undefined，不用零值补足 top-1/5/10%。

### B2：归一化机制

当前归一化 runner **已经支持 `topk_intersection`**。先在 M-PG/250 与 500 各做 4 个独立 fixed-batch banks，然后在 M-Overlap64-DR/250、500 上重复。第一轮每域 2 条完整训练 response，加每域 1 条 held-out，cap512，DR/DT/GT/zero 从同一保存状态出发；初期每个状态是 4 个 probe，不是 4 次训练。

必须输出每条 response 的长度、有效 mask、三种权重、loss sums/denominators，length–gradient covariance 与恒等式残差，raw-gradient cosine/norm、clip，以及四域 held-out full-vocab KL 前后值。这些直接填 F1c/F2。现有 M-PG/500 的一个 I64 bank 只有在输入与协议吻合时才计入 4-bank 目标。

随后用更长 response、更多 held-out 样本和不等 quota 验证。**当前程序强制每域 2 条 response**；batch size=1/4/8 或不等 quota 需要先改采样、权重与分母逻辑，跨 worker 检查还要核对真实分布式 sums/denominators。不能把一次代数残差很小当作这些扩展已完成。

## 5. 与训练并行保存 BF16 数据

C1 的 9 个 CPU 扫描均从同一初始 BF16 基线出发，记录 exact nonzero、`|delta|>1e-5` 的有效变化比例、L2、逐层/embedding/head，以及能量集中度。若 F4b 要连续能量曲线，应扩展流式扫描的分位点；现有 top-1/5/10% 三点不可插补成实测全曲线。每个扫描保存源 checkpoint 与基线 hash。

C2 单独记录 `online_update_bf16 = model_bf16_after − model_bf16_before`。当前 `paper_measurements.py` 的 `update` 是 FP32 optimizer 差，`delta_bf16` 是距初始值的累计差。建议新训练记录 step1、48–52、248–252、496–500 的真实连续更新；完整保存前后值或相应可复核指标，不与局部 proposed step 混用。

修改磁盘源码不会改变正在运行进程中已加载的 logger。当前任务没有插入新的在线记录；若现有运行没有自然进入经过验证的新启动入口，就把这项记录放到 D1 新种子运行中。现有 0/1 相邻保存值可在身份与完整性吻合后核对第一步；其余已过去但未保存的连续窗口仍是缺口。

## 6. 第三批：哪些新训练值得开

首轮得到完整能力与当前 loss 机制对照后，优先新增 **M-PG、M-Overlap64-DR 各 seed43/44，共 4 条训练 / 2,000 updates**。这是监督密度与 joint 能力最直接的关键比较，先用它判断效应幅度和训练方差。各运行预先固定 loss/reduction、数据、optimizer、长度 cap、保存点与评估协议，并在开始前加入 C2。

归一化结论若作为主结论保留，再补 M-Overlap64-DT/GT 的 seed43/44；single/joint 主张还需要 S-PG、S-Overlap64 的 seed43/44。六配置完整新增重复共 **12 条 / 6,000 updates**，包含前面 4 条，不相加为 16 条。现有 seed42 与新运行按 protocol/manifest 核对可比较范围；诊断 draws、评估采样和关联日志不计为训练重复。

在新正式种子前固定主要终点与效应报告：四域 pp 变化、macro、worst-domain change、BF16 fraction 的绝对 pp 差和能量集中度；显示每个训练种子。评估采用按题配对的不确定性估计，GPQA 的 4 回答按题聚类。若第一轮看不到方向一致的差异，也如实扩展/报告不确定性，不按显著性选择保存点。

正文已承诺的 cap1024 vs4096 对照：固定 M-Overlap64-DR，至少增加一条 cap1024 的 500 步探索运行并配套四域评估；如果保留统计性主张，需要对应重复。诊断 batch-size 敏感性用固定 checkpoint，不额外训练模型。第二模型、K sweep、GPAS 扩展暂时后置。

## 7. 程序与资源是否就绪

| 现有入口 | 能复用的部分 | 新任务前还要做什么 |
|---|---|---|
| [`examples/mopd_gpas/_evaluate_paper.sh`](../../slime_opd_geometry/examples/mopd_gpas/_evaluate_paper.sh) | 任意 target；`MOPD_EVAL_STEP` 与显式模型路径，或按 checkpoint index 选择 | 新 campaign 独立配置/输出/端口/GPU，固定完整四域协议；不直接复用旧 target 枚举的 `run_capability_eval.sh` |
| [`local/pg100_math_if_retry_20260908/`](../../slime_opd_geometry/local/pg100_math_if_retry_20260908/) | 已验证评分依赖及逐题核验流程 | 原 plan 只针对 PG100 math/IF；补齐与 PG250/500 相同的 code/GPQA 配置，不把原 `run.py launch` 当通用调度器 |
| [`scan_bf16.py`](../../slime_opd_geometry/local/bf16_checkpoint_scan_20260907_sn4622122392/scan_bf16.py) | CPU 累计 BF16/逐层扫描 | 按新 250/500 index 创建 9 点 manifest；现有批处理清单是旧 20 个点 |
| [`bf16_local_probe.py`](../../slime_opd_geometry/local/mpg500_teacher_overlap_20260907_sn4622122392/bf16_local_probe.py) | 已验证 saved Adam→BF16 写回、PG/ST64/full、PG 教师分支 | loss 列表在代码中固定；先新增 I64 与固定 bank 输入/教师分支，并验证 |
| [`normalization_probe.py`](../../slime_opd_geometry/local/normalization_cuda6_9_20260907_sn4622122392/normalization_probe.py) | I64 fixed-batch、DR/DT/GT/zero，完整 response 梯度 | 4-bank 独立新输出；不等 quota、更大 batch、worker 审计仍需扩展 |
| [`paper_measurements.py`](../../slime_opd_geometry/slime_plugins/mopd/paper_measurements.py) | 已有 before/after BF16 值与统计工具 | 新增真实 BF16 单步 quantity 与窗口，验证后在新进程启用 |

01:50 本机 `exx-B7129F83AV8E4HR-N` 有 10 张 RTX A6000 48GB；GPU0/1/2 当时各约 2MiB，其他卡已有占用。三张卡是**候选槽位**，未预订，启动前按 UUID 和进程重新核对。训练共享盘状态不证明其他节点进程健康。

建议候选 GPU 优先承担 A1 的三套评估；checkpoint 尚未准备好时，一张卡可运行一个有明确结束时间的归一化 probe，其余保留到达即评估的余量。不要刚好在 DT/GT 临近导出时占满三卡做长诊断。CPU 同时准备/扫描已完整导出的 checkpoint。

现有节点上 normalization 每 bank 约 16–36 分钟，teacher overlap 每 bank 约 72–73 分钟；新增 I64 分支与 A6000 上的实际吞吐要重新校准。按现有步时，最长训练分支剩余约 19.2 小时纯步时，另加导出与调度。训练继续且有两个评估槽位、一个诊断槽位时，首轮能力/机制缺口可按 **2–4 天的条件预算** 排程；新增训练种子与全部稳健性实验另计。

交付时更新四类产物：逐题能力长表、当前 loss 的局部分支/教师对长表、BF16 checkpoint/单步长表、27 图槽状态。只有完成产物并核验后才改变状态，不能把“已排队”算成“已有结果”。
