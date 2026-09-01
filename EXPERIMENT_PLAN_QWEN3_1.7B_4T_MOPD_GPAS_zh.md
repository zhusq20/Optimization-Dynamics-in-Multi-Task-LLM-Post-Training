# Qwen3-1.7B 四教师 MOPD：GPAS 实验协议

> 版本：2026-09-01  
> Student：Qwen3-1.7B  
> 任务：Math、Code、Instruction Following、Science  
> 主问题：固定 per-response objective 后，哪一种任务宽度和分配方法能用更少 response 与 GPU-hour 达到相同 teacher loss？

## 1. 先冻结的内容

训练前在 manifest 固定：

- student、四个 teacher 和 tokenizer 的精确 revision；
- 每个任务的数据 revision、prompt template 和随机种子；
- `16 prompts × 4 responses = 64 attempted responses` 的 task unit；
- response cap = `8,192` tokens，以及空生成和 teacher-scoring failure 的数值 penalty；
- 初始 loss 分母、EMA decay、inclusion floor、`A_max=50` 和共同 loss threshold；
- GPU 型号、GPU 数量、sharding、dtype、teacher staging 路径和一槽驻留策略。

任何配置都使用同一目标：

```text
relative_loss_i = loss_i / loss_i_0
F_rel = mean_i(relative_loss_i)
```

任务选择只改变执行频率。每个被选任务使用最终 inclusion probability 做 importance correction。
Reverse KL 先在每个 response 的有效 token 上平均，再在同一 prompt 的 4 个 response 上平均，最后在 16 个 prompt 上平均。

## 2. 三层证据

### A. 受控计算检查（已完成）

运行：

```bash
python3 experiments/generate_controlled_figure.py
```

固定 seed 的结果：

- GPAS 的 AdamW-metric error / Uniform = `0.717`；
- Cost-GPAS 的 time-weighted error / Uniform = `0.889`；
- raw gradient-norm proposal 的 AdamW-metric error / Uniform = `6.868`；
- conventional AdamW second-moment relative error = `0.816`；
- taskwise AdamW 的 Monte Carlo relative error = `0.00235`。

这部分只检查公式和实现。

### B. 真实 checkpoint 的 frozen-gradient bank

使用 Uniform trajectory 的 warm、middle、late 三个 checkpoint。每个 checkpoint、每个任务采集 8 个完整 task unit。用一半估计 proposal，另一半评估，再交换两半。报告：

- raw gradient norm 与 AdamW-scaled score；
- `K=1,2,4` 的 exact-set estimator error；
- Uniform、raw-norm、GPAS、Cost-GPAS 的 held-out error；
- conventional 与 taskwise second-moment target error；
- score age、probe 次数和 probe GPU-hour。

四任务的 `K=2` 枚举全部六个 task set。GPAS 使用满足目标 marginals 的 maximum-entropy set distribution。Cost-GPAS 用当前 resident teacher 和固定访问顺序估计每个 set 的 critical-path time，并直接优化 set distribution。

### C. 端到端 MOPD

所有配置从同一个 post-warm-start checkpoint 出发。Warm start 按 round-robin 顺序运行，每个任务贡献 2 个完整 task unit。总预算为 128,000 attempted responses，probe 也计入预算。Learning-rate schedule 使用 attempted responses 作为 clock。

| 配置 | K | Inclusion | AdamW second moment |
|---|---:|---|---|
| Uniform baseline | 1 | uniform | conventional |
| Uniform state control | 1 | uniform | taskwise |
| GPAS | 1 | GPAS | taskwise |
| Cost-GPAS | 1 | Cost-GPAS | taskwise |
| Uniform intermediate | 2 | uniform exact set | taskwise |
| Cost-GPAS intermediate | 2 | set-aware Cost-GPAS | taskwise |
| Streamed mixed state control | 4 | all tasks | taskwise |
| Streamed mixed baseline | 4 | all tasks | conventional |

端到端训练使用 1 个 matched training seed，覆盖表中全部配置。在该 matched seed 下，每个任务的第 `n` 次请求在所有配置中使用相同 prompt group。

## 3. AdamW 与 response clock

若一个 task unit 的 AdamW decay 为 `beta`，一次 `K`-task update 使用 `beta^K`。decoupled weight decay 做相同的复合换算。同时运行两个实现测试：

- clock-equivalence test：检查 moment 和 weight decay 的累计系数；
- resume test：恢复后复现下一 task set、teacher order、moment observation 和 parameter update。

## 4. 一槽 teacher streaming

所有 `K` 使用相同 GPU 和一槽 teacher residency。一次 `K`-task update 在同一 student state 上累积被选任务的梯度：

1. 抽取完整 task set；
2. set 中包含 resident teacher 时先执行该任务；
3. 其余任务按固定 task ID 顺序访问；
4. transfer 可与 rollout 或 backward 重叠；
5. 最后一个 teacher 保持 resident。

每个 task unit 记录 load/offload、teacher ready、rollout、scoring、backward、optimizer、peak HBM、responses 和 tokens。净系统结论使用端到端 GPU-hour。

## 5. 主结果

主图：

1. mean relative teacher loss vs attempted responses；
2. mean relative teacher loss vs GPU hours。

主表：

- final mean relative loss 和四任务 relative loss；
- 达到共同 loss threshold 的 responses 与 GPU hours；
- peak HBM、responses/GPU-hour、tokens/GPU-hour；
- switch rate、transfer tail、p50/p95 update time；
- MATH-500 greedy pass@1、冻结的 LiveCodeBench pass@1 slice、IFBench strict accuracy、GPQA-Diamond average@4。

主图显示全部配置在 matched prompt streams 下的训练曲线。Held-out loss 用相同 evaluation generation seed，并按 prompt 做 paired bootstrap。

在线 GPAS 固定 `A_max=50` processed task units。日志报告 score age、probe 数量和 GPU-hour、最终 bounded inclusion probabilities 以及 importance multipliers。

## 6. 实验产物

完整实验生成：

1. warm-start 与真实 checkpoint 的 frozen-bank 结果；
2. `K=1,2,4` 的两张主曲线；
3. 最终 MOPD outcome table；
4. teacher residency、HBM 和 transfer trace；
5. 全部配置的 run manifest、checkpoint 和按任务日志。
