# 图表重排：从能力轨迹到机制解释

更新：2026-09-09。当前入口是[新版图集](../../figures/paper_curves_20260908/index.html)、[正文候选PDF](../../figures/paper_curves_20260908/main_figures.pdf)和[W&B raw curve浏览器](../../figures/paper_curves_20260908/raw_curves.html)。每个导出文件仍只含一张图，图外先写问题和读法。

## 两篇论文提供的具体参照

- [Open-MOPD，Fig.1](https://arxiv.org/pdf/2608.19098v1)：先用领域验证轨迹和教师参考线呈现能力缺口，再解释缺口。它的Fig.3/4用长度、token份额和奖励幅度随训练的变化，分别检验机制。这里采用这种证据顺序，不移植它在SmolLM3上的结论；本项目的IF长度和token份额必须读自己的日志。
- [MOPD，Fig.2/3](https://arxiv.org/pdf/2606.30406)：领域准确率对累计领域训练样本，及accuracy、KL、entropy的训练动态相互配合。这里补充领域评估、数学prompt暴露视图和原始loss/log-ratio；entropy没有记录，明确留空。

## 当前正文候选：9张，其中7张折线图

| 顺序 | 要回答的问题 | 图的形式与内容 | 读图参照 |
|---|---|---|---|
| 1–4 | 联合训练是否保留每个领域的能力？ | 每个benchmark独立折线图；Joint PG / Math-only PG，真实checkpoint分数 | 初始学生点线、领域教师虚线；百分比分数 |
| 5 | 各域逼近教师的速度是否一致？ | Joint PG四域原始teacher_loss随optimizer update变化；log纵轴保留尖峰 | sampled student-minus-teacher log ratio，非精确全词表KL |
| 6 | 均等prompt是否产生均等token份额？ | 四域原始token share对rollout index | 25% prompt份额参考线 |
| 7 | 平均规则是否改变梯度方向？ | 同一批响应下DT/GT相对DR的梯度cosine点图；7个明确条件 | cosine=1；不连接无序类别 |
| 8 | 教师方向差异是否依赖输入？ | task-specific / identical-input成对点图 | 同学生、同Adam、同教师对；均值与观测范围 |
| 9 | 局部写回范围结论是否依赖阈值？ | PG500固定状态的BF16变化比例对阈值折线；PG/I64/full/zero | 阈值是有序量；范围是四bank min–max |

主图先呈现事实，再放解释。数学55.6%→66.4%，IFBench17.3%→14.7%这两个方向不同的结果，比把全部实验铺成色块更容易建立研究问题。Math-only PG的跨域评估是能力保留对照，不能将它称为各域single-teacher oracle。

补充区保留数学训练prompt暴露、raw reward、raw train/loss、sampled log ratio、实际聚合梯度范数、生成长度、截断率、已有Student64→I64续训评估、不同学生状态的阈值曲线，以及局部梯度/写回对照。全套为25张折线图、4张点图、1张热力图。唯一热力图是4×4教师对support overlap；两个轴确实定义两两关系。

## 原始曲线与数据口径

1. 本地`metrics/*.jsonl`由`slime/utils/logging_utils.py::log`在提交`wandb.log(metrics)`前写入。这次直接冻结原始镜像，未使用云端抽样history，也没有声称重新从W&B API下载。每条训练run保留原W&B链接。
2. 静态训练图和浏览器默认均不平滑，保留尖峰、零值和全部记录点。浏览器可加EMA橙线，但原始蓝线始终可见；导出CSV始终是raw值。局部探针的均值/范围另作明确标注。
3. `train/step`是梯度microbatch日志序号；`mopd/update`是完成的optimizer update；`rollout/step`是原rollout index。离线eval用产物的`num_updates`定位checkpoint，不能把eval本地step当训练step。
4. 同一clock的不同worker字段合并；同字段重复日志取最后记录，原始JSONL完整保留。缺失日志区间断线，未生成伪观测点。
5. 所有52组评估attempt均检查源artifact SHA256、样本数、唯一prompt数、重复次数、checkpoint、终止状态和逐题reward均值。50个唯一family/step/dataset点选用最早完成且核验通过的attempt，按时间而非分数选择。重跑点全留在CSV和浏览器，不作为新seed或CI。
6. 主评估曲线里的step0来自已测初始学生。Math-only在500步虽有训练日志，但本次本地没有该checkpoint的完整评估，因此能力曲线止于250步。
7. DR/DT/GT交集目标分别从137/152/151步开始，单任务从101步开始。名为`itk64`的100步评估是Student64父轨迹；250步点含混合历史。不能当作从Base起全程I64训练。
8. 数学prompt横轴来自实际allocation：joint每步16条数学prompt，math-only每步64条。其他领域的单任务暴露为零；不画伪造的公平样本效率曲线。GPU成本尚未作跨续训统一核算，所以未填入推算GPU-hours主图。

## 当前缺口与图能支持的结论

- Policy entropy没有记录；负sampled log-prob不等于entropy。这次未为补图启动训练。
- 部分续训250步只有Code/Science完整评估；Math/IF只画已有100步点。没有将缺失值填0、补插值或延长到末步。
- Raw training reward是训练verifier输出，OPD中`reward_used_in_loss=0`。代码执行器/评分器曾出现的问题仍保留在日志里；该曲线不替代独立benchmark结果。浏览器同时提供已记录的Sandbox错误指标。
- 31组局部探针是固定学生、保存Adam下的HF梯度和模拟BF16写回，不是真实在线逐步更新。bank范围不是训练seed不确定性；teacher pair共享教师，不能用它们假充独立样本。
- 旧15图及27图槽清单保留为历史材料，当前正文顺序、图数与caption以本包为准。旧F3/F4累计扫描、更多种子和新增cap训练不因本次重排自动宣称补齐或恢复。

## 复现与审计

在论文目录执行，重画无需W&B登录、网络、模型或GPU：

```bash
python experiments/plot_paper_figures.py
```

只有主动刷新已有本地日志时才运行：

```bash
python experiments/export_training_curves.py --root ../slime_opd_geometry/outputs/mopd_qwen3
python experiments/plot_paper_figures.py
```

- `raw/training/`、`raw/evaluation/`：完整压缩scalar镜像、评估index/完成标记与紧凑逐题核验证据。
- `source_manifest.csv`：每个冻结流的源路径、原始字节SHA256、大小和记录数。
- `data.json.gz`：重画输入，保留每个metric的原始文件/行号和run身份。
- `raw_curve_values.csv` / `raw_curve_inventory.json`：浏览器全部可选指标的原始值与点数。
- `evaluation_attempts.csv`：所有评估attempt与选择标记；`evaluation_availability.csv`显式列出缺口。
- `plotted_values.json`：静态图中原始点及局部变换的来源；`FIGURE_CAPTIONS.md`说明参照、分母、范围和边界。
- `figure_manifest.json`：正文顺序、图型与生成脚本SHA256。
- `validation_report.json`：最终数据、文件与浏览器核验结果。

冻结时间：2026-09-08T23:58:47.274170+00:00。仍在续跑的run只展示本次冻结到的记录，不将当前图解释为终态。
