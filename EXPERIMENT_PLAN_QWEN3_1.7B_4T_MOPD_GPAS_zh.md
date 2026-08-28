# 已归档：Qwen3-1.7B 四教师 MOPD / GPAS 宽范围实验计划

> 当前论文已收缩为 GPAS 单一主线。正文实验以
> `sections/005_experiment.tex` 为准。本文件保留旧的 Local Gaps、mixture
> editing 和 intervention-selection 设计，供后续独立论文使用。

> 固定版本：2026-08-26  
> 主模型：Qwen3-1.7B  
> 教师：Math、Code、Instruction Following、ScienceQA 四个同源 GRPO specialist  
> 随机种子：端到端主方法使用 41/42/43；同状态局部分支使用 seed 42  
> 论文目标：用局部量选择“改善固定 mixture 的优化”或“修改 teacher weights”，并提出固定目标下的高效 MOPD allocation 方法 GPAS。

## 1. 固定研究问题

全文只回答三个相互衔接的问题。

1. **诊断能否选对训练干预**：SoloGap、ConflictGap 和 OptGap 能否预测任务专属分支的剩余误差、固定权重下 GPAS 的收益，以及 response-guided weight edit 的收益？
2. **优化器如何改变任务尺度**：在同一 student 和同一 AdamW state 上，raw task-gradient norm 与 lagged-preconditioned norm 哪一个更准确地预测任务相关的真实参数更新和下一批 teacher loss 下降？
3. **GPAS 是否改善固定 mixture 的训练效率**：在 $w_i=1/4$ 时，optimizer-scaled gradient score 能否比 teacher gap score 和 raw gradient score 更好地分配 batch，并在 importance correction 下降低 estimator error、加快 weighted teacher loss 下降、控制 teacher-mixture drift？

计划中列出的每项实验都对应正文表图或 appendix 的固定结果，不设置根据中间结果决定是否继续的分支。

## 2. 模型谱系和四个 teacher

### 2.1 共享初始化

所有模型使用完全相同的：

- 起点权重：`Qwen/Qwen3-1.7B` 的同一冻结 revision（非 Base checkpoint），只做一次到 Megatron 格式的确定性转换；
- tokenizer、词表和 chat template；
- `enable_thinking=false`；
- base checkpoint hash；
- seed 42 数据顺序和 rollout sampling 约定。

记共享初始化为 $\theta_0$。四个 teacher 都从 $\theta_0$ 独立进行单任务 GRPO：

```text
                              ┌── Math GRPO ─────── θT_math
Qwen3-1.7B θ0 ────────────────┼── Code GRPO ─────── θT_code
                              ├── IF GRPO ───────── θT_if
                              └── ScienceQA GRPO ── θT_science
```

MOPD student 从一份干净的 $\theta_0$ 开始。四个 teacher 在所有 OPD 实验开始前冻结；主实验固定使用各任务 seed-42 GRPO 的 final checkpoint，teacher checkpoint curriculum 保持关闭。

### 2.2 四个 teacher 的固定 GRPO 配方

本计划按四个 specialist 都需要从 $\theta_0$ 正式训练来计数；若这些 checkpoint 已经训练完成，则必须逐项核对并记录真实配置，不以新的结果导向训练替换它们。四条 run 分别只看单一任务数据，统一采用：

| 项目 | 固定设置 |
|---|---|
| train prompts | 每任务 16,000 个唯一 prompt，seed-42 shuffle，完整 1 epoch |
| prompts / optimizer update | 32，因此每个 teacher 恰好 500 updates |
| responses / prompt | 8 |
| GRPO group advantage | group 内 reward 标准化，零方差 group 保留且贡献零 advantage |
| PPO-style clip | 0.2 |
| reference KL | 对 $\theta_0$，$\beta=0.01$ |
| optimizer | AdamW，$\beta_1=0.9,\beta_2=0.95$，weight decay 0 |
| learning rate | $1\times10^{-6}$，前 20 updates linear warmup，之后 constant |
| rollout | 最多 2,048 prompt tokens、8,192 response tokens；temperature 1.0，top-p 1.0 |
| checkpoint | update 0/100/200/300/400/500；OPD 只用 final checkpoint |
| seed | 42 |

不做 reward-threshold filtering，不按验证集选择 checkpoint，也不延长某一个任务的训练；四条 teacher curve 和 final specialist benchmark 都作为 appendix 数据。任务 reward 固定为：

- Math：$0.9\times$答案正确性 $+0.1\times$答案格式正确性；
- Code：$0.9\times$ SandboxFusion unit-test pass fraction $+0.1\times$可解析性；
- IF：逐约束 compliance fraction 的平均值，主 reward 不混入通用 judge；
- ScienceQA：$0.9\times$ exact-choice correctness $+0.1\times$合法选项格式。

所有 reward 都同时记录总分和各 component；对 verifier timeout / infrastructure error 单独标记，不记成模型错误。四条 GRPO teacher run 同时承担三种论文角色：

1. 提供四个 frozen teacher；
2. 提供每个任务的 specialist benchmark reference；
3. 提供 teacher checkpoint、训练数据和 reward provenance。

Teacher scoring endpoint 只接收 student 已生成的 exact token IDs，返回这些 response token 的一维 teacher log-probability；teacher 不重新生成回答，也不返回 full-vocabulary logits。每次运行记录 teacher checkpoint hash、tokenizer hash、endpoint 配置和实际请求 token 数。

## 3. 数据和评测

### 3.1 训练数据

从当前已经准备的四任务数据中，各冻结一个由 16,000 个唯一 prompt 构成的训练池；不从额外来源补齐，也不因某个方法的结果改变训练池内容。GRPO teacher 和对应任务的 OPD student 共用该训练池，但所有 probe 和 benchmark 与训练池严格隔离。

| 任务 | 当前可用规模 | 固定训练池 | GRPO reward / verifier |
|---|---:|---:|---|
| Math | 18,231 | 16,000 | exact-answer / deepscaler |
| Code | 19,169 | 16,000 | unit tests / SandboxFusion |
| IF | 16,575 | 16,000 | IFEvalG-style constraints |
| ScienceQA | 19,670 | 16,000 | GPQA-style exact choice |

另外，每任务冻结 128 个不进入上述训练池、也不属于正式 benchmark 的 prompt，作为所有 checkpoint-local probe 的固定集合。训练池和 probe set 都保存 prompt ID、源数据 revision 和 SHA-256。

由于固定 response-token budget 下 IF 可能消耗更多 prompt，训练源允许在完整遍历 16,000 个 prompt 后按独立 task cursor 进入下一 epoch，并用 seed 42 重新排列；每条 run 记录 unique-prompt coverage 和每任务实际 epoch 数。

### 3.2 正式 benchmark

四个 domain 的总体 macro average 只使用每个域一个预注册主指标，避免某个域因 benchmark 更多而被重复加权。

| Domain | 主指标（进入 4-domain macro） | 补充指标 | 最终评测口径 |
|---|---|---|---|
| Math | MATH-500 | AIME'24 | MATH-500 greedy pass@1；AIME avg@8 |
| Code | LiveCodeBench v5 | LiveCodeBench v6 | 全量题集，greedy pass@1 |
| IF | IFBench strict | IFEval strict-prompt | greedy accuracy / strict compliance |
| ScienceQA | GPQA-Diamond | — | 每题 4 次，报告 avg@4 accuracy |

所有模型共用固定 prompt template、decoding seed、temperature、top-p、response cap、verifier 和代码 sandbox。保存逐题 response、判分结果、错误类型和基础设施状态，不只保存聚合分数。

### 3.3 训练中评测

所有多教师方法每 25 updates（含 update 0）运行一次四任务固定 teacher-loss probe；prompt identity、顺序和 decoding RNG 固定，用这些 25 个时间点计算 loss AUC、tokens-to-target 和 matched-loss trade-off。Probe 只产生评测输出，模型参数保持不变。

在 update $0/150/300/450/600$ 使用同一组正式 online benchmark configs：

- MATH-500 全集，greedy；
- LiveCodeBench v5 固定且与训练零重合的 128 题，greedy；
- GPQA-Diamond，avg@4；
- IFBench strict 全集，greedy；

最终 update 600 额外运行 AIME'24、LiveCodeBench v5/v6 全量、IFEval。四个 single-teacher OPD student 只做起点和最终正式评测。

## 4. 统一的 MOPD 训练合同

### 4.1 训练目标

任务权重固定为

$$
w_{\text{math}}=w_{\text{code}}=w_{\text{if}}=w_{\text{science}}=\frac14.
$$

主实验使用当前代码已经实现的 sampled-token reverse-KL estimator。对 student 采样 token $a_t$，记录

$$
\ell_t=\log\pi_\theta(a_t\mid h_t)-\log\pi_i^T(a_t\mid h_t).
$$

论文和图表始终将其称为 `sampled log-ratio` 或 `sampled reverse-KL estimate`，不写成 full-vocabulary KL。每个任务 loss 先在该任务全部有效 response token 上求均值，再进入任务权重或 importance correction；padding、response length 和每批 prompt 数不改变 $w_i$ 的定义。

### 4.2 Task-homogeneous fixed-token update

一次 optimizer update 只包含一个任务。为尽量使 sampling unit 与理论一致，rollout 以 4 个 prompt 为小组连续生成，直到该 update 获得约 65,536 个非 padding response token；使用完整 response，不切分一条 response，实际 token 数允许在目标附近小幅波动。

固定训练设置：

| 项目 | 设置 |
|---|---|
| student initialization | $\theta_0$ |
| total optimizer updates | 600（包括前 16 个初始化分配步骤） |
| target valid response tokens / update | 65,536 |
| prompt / response cap | 2,048 / 8,192 |
| responses / prompt | 1 |
| rollout sampling | temperature 1.0，top-p 1.0，thinking off |
| optimizer | AdamW，$\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-8}$ |
| learning rate | $1\times10^{-6}$，前 20 updates linear warmup，之后 constant |
| weight decay | 0 |
| checkpoint | update 0/150/300/450/600 |
| seed | 42 |

全部方法匹配 optimizer-update 数和目标 response-token 数，并额外报告实际 student response tokens、teacher-scored tokens、prompt tokens、teacher GPU hours、student GPU hours、wall-clock 和有效 tokens/s。

这里的“截断”有唯一可复算定义：生成达到 8,192 个 response tokens 时仍缺少 EOS。主实验固定该长度上限，逐任务报告命中上限的比例和这些样本的 reward / teacher loss。统一 non-thinking mode 让四任务共享同一输出协议，allocation 实验中的 task-specific thinking switch 保持关闭。

### 4.3 Gradient clipping 与 importance correction 顺序

为使“校正估计器的条件均值等于目标梯度”这一经验声明不被 nonlinear global clipping 破坏，固定采用以下顺序：

1. 计算 task mean-token raw gradient $G_i^{\rm raw}$；
2. 对每个任务使用相同阈值 $C=1.0$ 做 per-task norm clipping，得到 $G_i$；
3. corrected 方法乘 $\rho_i=w_i/a_i$；
4. 不再做第二次 global clipping，直接把 $\rho_iG_i$ 送入 AdamW first/second moments。

因此，校正估计器的条件均值对应 clipped per-task target $\sum_iw_i\mathbb E[G_i]$；这不意味着保持 AdamW 的期望参数更新或训练轨迹。所有方法使用完全相同的 clipping 定义，并报告 raw norm、clip scale 和 clip fraction。

## 5. 完整训练矩阵

### 5.1 四个 single-teacher OPD reference

从同一 $\theta_0$ 分别运行：

| ID | Teacher | Updates | 作用 |
|---|---|---:|---|
| ST-OPD-Math-42 | Math GRPO teacher | 150 | Math 可蒸馏上界 |
| ST-OPD-Code-42 | Code GRPO teacher | 150 | Code 可蒸馏上界 |
| ST-OPD-IF-42 | IF GRPO teacher | 150 | IF 可蒸馏上界 |
| ST-OPD-Science-42 | ScienceQA GRPO teacher | 150 | ScienceQA 可蒸馏上界 |

每条 single-task run 的 150 updates 等于 uniform multi-task run 对该任务的预期 update 数。评测时按 domain 选择对应 single-task student，构成 `Single-teacher OPD oracle (four models)`；表格明确说明它不是一个统一可部署模型。

### 5.2 七种 multi-teacher 训练规则

所有方法使用相同 student、四个 teacher、数据池、token budget、warmup、优化器和评测。

| ID | Task proposal $a_i$ | Correction | 论文角色 |
|---|---|---|---|
| MT-Uniform-42 | $a_i=1/4$ | $w_i/a_i=1$ | 单任务 update、固定 allocation 基线 |
| MT-FullMixture-42 | 每步四任务分层累积 | 直接形成 $\sum_iw_iG_i$ | full-mixture 强基线 |
| MT-GapIS-42 | $a_i\propto w_i\widehat q_i$ | 有 | 固定目标的 gap-based proposal |
| MT-GapNoIS-42 | $a_i\propto w_i\widehat q_i$ | 无 | gap proposal 的 objective-drift 消融 |
| MT-RawGPAS-42 | $a_i\propto w_i\widehat s_i^{\rm raw}$ | 有 | target-gradient-preserving raw-gradient proposal |
| MT-AdamGPAS-42 | $a_i\propto w_i\widehat s_i^{\rm adam}$ | 有 | 完整方法 |
| MT-AdamGPAS-NoIS-42 | 与 AdamGPAS 完全相同 | 无 | importance correction 消融 |

其中：

$$
q_i=\sqrt{\mathbb E[\ell_t^2]},\qquad
s_i^{\rm raw}=\sqrt{\mathbb E\|G_i\|_2^2},\qquad
s_i^{\rm adam}=\sqrt{\mathbb E\|P_tG_i\|_2^2},
$$

$$
P_t=\operatorname{diag}\!\left((\sqrt{v_{t-1}}+\epsilon)^{-1}\right).
$$

统一的 adaptive 设置：

- 六个 single-task-per-update 方法的前 16 updates 使用 task-homogeneous round robin，每任务 4 次；这 16 步计入总预算并完全相同；`MT-FullMixture-42` 从第一个 update 开始使用四任务梯度累积；
- 每任务分别维护 bias-corrected EMA，$\gamma=0.95$；
- probability floor $a_{\min}=0.05$。对 GPAS 使用带下界的精确解
  $$
  a_i=\max\{a_{\min},c w_i\widehat s_i\},
  \qquad \sum_i a_i=1,
  $$
  其中标量 $c$ 通过四任务 water-filling 求得；GapIS 和 GapNoIS 使用同样的下界求解并将 $\widehat s_i$ 换成 $\widehat q_i$；
- 每次成功 optimizer update 后更新被采样任务的统计，并据此决定下一次 task proposal；
- proposal、EMA、task cursor 和 RNG state 随 checkpoint 完整保存；
- 每条 batch 保存被采样任务、$a_i$、$w_i$、$w_i/a_i$、scale estimate 及其更新时间。

`MT-GapIS-42` 和 `MT-GapNoIS-42` 共享 EMA、floor、初始分配和 proposal。两者只在 gradient scale 上不同。GapIS 与 GPAS 的比较隔离 allocation score；GapIS 与 GapNoIS 的比较隔离 importance correction。

`MT-FullMixture-42` 在每个 AdamW step 内依次累积四个 task-homogeneous gradient，每任务目标约 16,384 个有效 response tokens，总计约 65,536 tokens。各任务先独立求 mean-token gradient，再形成 $\sum_iw_iG_i$ 并更新一次 AdamW。该基线匹配总 token 和 teacher-scored-token budget，用于检验在四任务规模下直接消除 task-level sampling noise 的效果。

### 5.3 训练数量

正式训练矩阵为：

- 4 条 single-task GRPO teacher runs（若已有完全匹配的 checkpoint，则登记为既有正式 run）；
- 4 条 single-teacher OPD runs；
- Uniform、FullMixture、GapIS、RawGPAS 和 AdamGPAS 各使用 student seed 41/42/43，共 15 条主训练；
- GapNoIS 和 AdamGPAS-NoIS 使用 seed 42，共 2 条机制消融；
- 合计 4 条 teacher runs 和 21 条 student runs。

全部运行均预先进入正文或 appendix 的结果表、训练曲线或 oracle reference。

可提前复算的核心工作量为：teacher 侧 $4\times16{,}000\times8=512{,}000$ 条 GRPO responses；student 侧 $4\times150+17\times600=10{,}800$ 个正式 OPD updates，约 707.8M 个目标 response tokens。另有 coordinate weight branches 40 条、joint response-guided branches 5 条、task-specific branches 20 条、Adam-GPAS 短分支 5 条、320 个 one-step optimizer clones，以及 4 条 task-order replays。

## 6. 三类 local gap 与干预选择

### 6.1 分析 checkpoint 和缓存

SoloGap / ConflictGap / OptGap 的主诊断使用 `MT-Uniform-42` 的 update $0/150/300/450/600$。所有 seed-42 allocation 方法额外报告五个 checkpoint 的 gap trajectory；其他主方法 seed 在 final checkpoint 报告三类 gap。每个 condition：

1. 固定 student、AdamW state 和四任务 probe prompts；
2. 每任务生成并缓存至少 4,096 个有效 response tokens；
3. 缓存 student/teacher sampled-token log-probs、token mask 和 score-function Jacobian 所需状态；
4. 使用独立的每任务 held-out probe tokens 评估拟合结果，拟合与评估 token 不重合。

### 6.2 Trust-region local projection

对任务 $i$ 的缓存 token 定义 log-ratio residual $r_i$ 和 student log-probability Jacobian $J_i$。局部目标固定为

$$
L_w(\Delta\theta)
=\sum_i\frac{w_i}{N_i}
\left\|r_i+J_i\Delta\theta\right\|_2^2.
$$

通过 JVP/VJP 和 conjugate gradient 隐式求解 joint problem 和四个 task-specific problem。主文报告：

$$
L_{\rm solo}=\sum_iw_i\min_{\|\Delta_i\|\le R}L_i(\Delta_i),
\qquad
L_{\rm shared}=\min_{\|\Delta\|\le R}\sum_iw_iL_i(\Delta),
$$

$$
\mathrm{SoloGap}=\frac{L_{\rm solo}}{L_w(0)},\qquad
\mathrm{ConflictGap}=\frac{L_{\rm shared}-L_{\rm solo}}{L_w(0)},\qquad
\mathrm{OptGap}=\frac{L_w(\Delta_{\rm actual})-L_{\rm shared}}{L_w(0)}.
$$

$R$ 固定为同一方法在该 checkpoint 后 16 个真实 updates 的 task-dependent 参数位移范数：从真实分支终点减去同一初始 model/AdamW state 的 16-step zero-gradient replay，排除已有 momentum 和 weight decay 的共同移动；update 600 使用此前 16 个 updates 的对应位移。$\Delta_{\rm actual}$ 使用同样的 zero-gradient subtraction。最优化在 fit cache 上完成，正文数值在 held-out cache 上计算。三类 gap 仅表示当前 cache、参数化和半径下的短时间尺度。

每个解保存 CG iteration、normal-equation residual、predicted decrease、held-out local loss 和实际参数范数，避免只报告收敛后的一个标量。

### 6.3 固定 weight-perturbation branches

weight response 只从 `MT-Uniform-42` 的上述五个 checkpoint 和完全相同的 optimizer state 分叉。参数化 $w=\operatorname{softmax}(\eta)$，每次只改变一个任务：

$$
\eta_i\leftarrow\eta_i+0.5
\quad\text{或}\quad
\eta_i\leftarrow\eta_i-0.5,
$$

共 $4\times2=8$ 个分支/checkpoint。每个分支：

- 固定 uniform task order $a_i=1/4$；
- 用 $w_i/a_i$ 改变目标权重，从而不改变 cached batch 顺序；
- replay 16 个 cached, task-homogeneous updates；
- 不重新生成 rollout，符合 frozen-prefix semi-gradient 分析；
- 记录四任务 sampled teacher loss、log-ratio、logit response 和参数位移。

总计 40 条 16-update 短分支，全部进入 Figure 1 的 predicted-vs-observed 数据。比较 weight response 对观测变化的：

- per-task loss-change sign accuracy；
- Spearman correlation；
- normalized RMSE；
- predicted-vs-observed scatter；
- ConflictGap 与 empirical trade-off strength 的关系。

每个 checkpoint 还执行一次配方选择：

- 优先任务定义为分支起点 normalized teacher loss 最高的任务；
- response matrix 在 8 个候选分支中选择预测优先任务改善最大、且其他任务满足预注册 loss limit 的分支；
- 报告 top-1 selection accuracy、protected-task violation 和相对最优可行观测分支的 regret；
- 额外运行 5 条 joint response-guided branches，每个 logit 变化限制在 $0.5$；
- 将“直接上调当前 loss 最高任务 $0.5$”作为 loss-following weight baseline。

`empirical trade-off strength` 固定定义为一个任务权重上调后，其他三个任务 held-out teacher loss 正向增加量之和。该量的定义与最终 benchmark 分数分开。

### 6.4 诊断引导的 matched intervention branches

在 `MT-Uniform-42` 的五个 checkpoint 各运行：

- 4 条 task-specific 16-update branches，用于检验 SoloGap 与单任务剩余误差的关系；
- 1 条固定 $w$ 的 Adam-GPAS 16-update branch，用于检验 OptGap 与 allocation 收益的关系；
- 1 条 joint response-guided branch，用于检验 ConflictGap 与 weight-edit 收益的关系。

预注册动作规则：OptGap $>$ ConflictGap 时选 Adam GPAS；ConflictGap $\ge$ OptGap 时选 response-guided weight edit。报告动作选择准确率和相对两个实际分支中较好者的 regret。SoloGap 作为 model、supervision 和 update scale 的审查信号单独报告。

## 7. Optimizer-aware scale 的 counterfactual 实验

### 7.1 Same-state one-step clones

仍使用 `MT-Uniform-42` 的五个 checkpoint。每个 checkpoint、每个任务固定 16 个独立 task-homogeneous token batches，共

$$
5\ \text{checkpoints}\times4\ \text{tasks}\times16\ \text{batches}=320
$$

个 counterfactual observations。

每个 observation 从完全相同的 $(\theta,m,v)$ clone 出发，计算：

- raw clipped norm $\|G_i\|_2$；
- lagged-preconditioned norm $\|P_tG_i\|_2$；
- exact AdamW parameter update；
- zero-gradient AdamW baseline update；
- task-dependent update
  $$
  \Delta_i^{\rm task}
  =\Delta_i^{\rm AdamW}-\Delta^{\rm zero\text{-}grad};
  $$
- 同任务下一 held-out batch 的 sampled teacher loss reduction；
- 四任务 weighted held-out loss reduction。

主要预测目标是 $\|\Delta_i^{\rm task}\|_2$，因为它排除了已有 momentum 和其他任务无关状态带来的共同位移。teacher loss reduction 是次要行为目标。

### 7.2 固定统计输出

raw norm 和 preconditioned norm 分别对两个目标报告：

- Spearman 和 Pearson correlation；
- task-pair ranking accuracy；
- log-log calibration slope；
- checkpoint 分层后的相关性；
- 四任务各自的中位数与 IQR。

同一组 cached gradients 额外离线计算 `AdamW state reset` 和 `SGD/P=I` 两个 appendix 对照；它们不产生新的语言模型训练 run。

### 7.3 Task-order replay

在 update 300 固定 64 个 cached batches（每任务 16 个），从相同 student 和 AdamW state 分别 replay：

1. seed-42 random shuffle；
2. round robin；
3. Math→Code→IF→ScienceQA contiguous blocks；
4. ScienceQA→IF→Code→Math reverse blocks。

四条 replay 的 batch multiset、每任务 token 数、学习率和 update 数完全一致。报告最终 weighted/per-task teacher loss、参数位移、$m/v$ state norm 和 schedule 间离散度；SGD frozen-gradient replay 作为 appendix 参照。这一实验专门支撑“相同任务计数不保证相同 AdamW finite-budget path”。

## 8. GPAS 的端到端评估指标

### 8.1 优化效率

七种 multi-teacher 训练规则均报告：

- weighted sampled teacher loss vs optimizer updates；
- weighted sampled teacher loss vs valid response tokens；
- weighted sampled teacher loss vs teacher-scored tokens；
- weighted sampled teacher loss vs wall-clock；
- update 0--600 的 normalized AUC；
- 达到 `MT-Uniform-42` 最终 improvement 的 80% 和 90% 所需 tokens；
- seed-42 trajectory 的 SoloGap、ConflictGap 和 OptGap，以及其他主方法 seed 的 final gaps。

固定 teacher-loss probe 使用同一批 prompt identity。曲线同时报告 prompt-cluster bootstrap 测量误差和三个 student seeds 的训练差异。

### 8.2 Target-gradient preservation 与 trade-off drift

在五个 checkpoint 定期对四个任务各计算 16 个独立 gradient batches，形成 empirical full-mixture gradient

$$
g_w=\frac14\sum_i g_i.
$$

利用相同梯度样本直接计算每种 proposal 的：

$$
\widehat{\mathbb E}\|P_t\widehat g-P_tg_w\|_2^2,
$$

并报告 corrected estimator 的 empirical mean 与 $g_w$ 的 cosine / relative error。该测量直接验证 variance reduction，不依赖最终 benchmark 是否饱和。

最终 trade-off 报告四任务 teacher-loss vector 和四个主 benchmark vector。为区分“进度不同”和“折中不同”，除同 update 比较外，还在各方法曲线上找到 weighted teacher loss 与 Uniform final 最接近的 checkpoint，比较此时的 per-task vector distance：

$$
D_{\rm tradeoff}
=\left\|\mathbf L_{\rm method}-\mathbf L_{\rm uniform}\right\|_2.
$$

corrected GPAS 的正式经验主张限定为：更低 estimator second moment、更快 weighted-loss 下降，以及相较未校正 allocation 更小的 trade-off drift。AdamW trajectory 的差异通过 moment state 和参数位移单独报告。

### 8.3 下游能力

最终结果同时报告：

- 四个主 benchmark 原始分数；
- AIME'24、LiveCodeBench v6、IFEval 补充分数；
- four-domain macro average；
- 每个 domain 相对 $\theta_0$、对应 GRPO teacher 和 single-teacher OPD student 的差距；
- response length、truncation、invalid output 和 verifier failure。

表格同时展示 macro average 和所有单域分数。

## 9. 随机性和不确定性口径

五个主方法使用 student seed 41/42/43，正文报告 mean 和 standard deviation。两个未校正机制消融只使用 seed 42，表格保留原始数值和明确的 single-seed 标签。

可以报告的统计不确定性来自测量单位本身：

- benchmark：按 prompt 配对 bootstrap 或二项区间；
- 多采样 benchmark：先在 prompt 内求均值，再按 prompt cluster bootstrap；
- teacher-loss probe：按 prompt cluster bootstrap；
- optimizer counterfactual：按 checkpoint×task 分层 bootstrap；
- weight branches：报告全部 paired observations；branch 属于同状态 counterfactual，训练 seed 计数为 1。

所有图注明确写 `single training seed (42)`；bootstrap error bar 不描述为训练稳定性。

## 10. 正文固定表图

### Table 1：Qwen3-1.7B 四教师主结果

行：

- Base student $\theta_0$；
- GRPO specialist oracle（four models）；
- Single-teacher OPD oracle（four models）；
- MT-Uniform（3 seeds）；
- MT-FullMixture（3 seeds）；
- MT-GapIS（3 seeds）；
- MT-GapNoIS-42；
- MT-RawGPAS（3 seeds）；
- MT-AdamGPAS（3 seeds）；
- MT-AdamGPAS-NoIS-42。

列：MATH-500、LiveCodeBench v5、IFBench strict、GPQA-Diamond、4-domain macro、weighted teacher loss、valid response tokens、wall-clock。

### Table 2：优化效率和 objective drift

列：loss AUC、80%/90% tokens-to-target、preconditioned estimator MSE、final OptGap、matched-loss trade-off distance、teacher GPU hours。

### Figure 1：Local fit、OptGap 与 mixture sensitivity

- Uniform 五个 checkpoint 的 SoloGap / ConflictGap / OptGap；
- weight response 的 predicted-vs-observed scatter；
- response-guided branch selection regret 和 protected-task violation；
- gap-guided action selection regret。

### Figure 2：AdamW 如何改变 task scale

- raw norm vs exact task-dependent AdamW update；
- preconditioned norm vs exact task-dependent AdamW update；
- checkpoint/task 分层 correlation；
- task-order replay 的最终 per-task loss。

### Figure 3：GPAS 训练效率

- weighted teacher loss vs valid tokens；
- 七种训练规则的 weighted teacher loss 与 seed-42 gap trajectory；
- empirical preconditioned estimator MSE；
- tokens-to-target。

### Figure 4：Allocation 和最终 trade-off

- 四任务 sampling probability trajectory；
- importance ratio trajectory；
- final per-task teacher-loss vector；
- final four-benchmark capability vector。

### Appendix

- 四个 teacher 的训练配置、benchmark 和 checkpoint hash；
- 四个 single-teacher OPD 完整结果；
- 所有 multi-teacher run 的完整训练曲线；
- coordinate、joint response-guided、task-specific 和 Adam-GPAS 短分支明细；
- 所有 method-checkpoint 的 CG 和三类 gap 求解诊断；
- 320 个 optimizer counterfactual observations；
- AdamW state-reset、SGD/P=I 和 task-order replay；
- AIME、LCBv6、IFEval；
- prompts/responses/tokens/GPU-hours 表；
- 逐题评测与 verifier error summary。

## 11. 需要补齐的实现

现有 zip 中的多任务数据源、teacher routing、sampled-token OPD、exact gradient observer、same-checkpoint raw-gradient probe、benchmark evaluator 和 provenance 可以继续使用。正式运行前需要实现以下固定功能：

1. **Teacher GRPO configs**：为四任务加入固定 reward composition、8-response group、无动态过滤的 500-update 配置和 component-level logging。
2. **Runtime-adaptive task sampler**：训练进程可在每个 update 后更新四任务 proposal；EMA、proposal、RNG 和 cursor 可 checkpoint/resume。
3. **Importance metadata**：采样时把 $a_i,w_i,w_i/a_i$ 写入 batch，训练端在 per-task clipping 后、AdamW state update 前应用 correction。
4. **Preconditioned scale feedback**：用 lagged $v_{t-1}$ 精确计算 $\|P_tG_i\|$，只传回每任务一个标量，不新增 model forward。
5. **Dynamic token accumulation**：task-homogeneous rollout 以小 prompt group 累积到目标有效 response-token 数，并记录实际 overshoot。
6. **Frozen-prefix branch replay**：从指定 checkpoint 和 optimizer state 复制 coordinate weight、joint weight、task-specific 和 Adam-GPAS 分支，复用同一 cached batch order。
7. **Implicit local solver**：为 sampled-token empirical Gauss-Newton 提供 JVP/VJP、joint 与 task-specific trust-region CG、held-out evaluation、weight response 和完整求解日志。
8. **Counterfactual optimizer clone**：从同一 $(\theta,m,v)$ 分别执行 task gradient 和 zero-gradient AdamW step，保存 task-dependent delta。
9. **Paper aggregation**：从固定 schema 直接生成 Table 1--2、Figure 1--4 和 appendix CSV/JSON，不手工摘抄曲线。

## 12. 固定执行顺序

1. 冻结 $\theta_0$、16,000 prompts/task、128 probes/task 和全部 benchmark configs。
2. 按固定 500-update 配方运行四条 GRPO teacher，完成正式 benchmark 和 checkpoint/tokenizer hash 表。
3. 运行四条 150-update single-teacher OPD references。
4. 运行 `MT-Uniform-42`，保存五个主 checkpoint、完整 optimizer state 和 cached probe batches。
5. 基于 `MT-Uniform-42` 完成三类 gap、40 条 coordinate weight branches、5 条 joint response-guided branches、20 条 task-specific branches、5 条 Adam-GPAS 短分支、320 个 optimizer clones 和 task-order replay。
6. 运行 seed-42 的 FullMixture、GapIS、GapNoIS、RawGPAS、AdamGPAS 和 AdamGPAS-NoIS。
7. 运行 seed 41/43 的 Uniform、FullMixture、GapIS、RawGPAS 和 AdamGPAS。
8. 对所有 seed-42 方法补齐五个 checkpoint 的三类 gap，并对其他主方法 seed 的 final checkpoint 计算 gaps。所有 final checkpoints 执行正式评测。
9. 从冻结产物一次性生成正文表图和 appendix；失败或不完整的 run 保留原始状态和失败原因。

## 13. 每条 run 的强制产物

- 完整 config、seed、代码 revision、model/tokenizer/teacher hash；
- 数据 manifest、prompt IDs、task cursor 和 epoch；
- proposal、EMA、importance ratio、token budget；
- sampled prompts、responses、student/teacher log-probs；
- raw/clipped/corrected gradient norm 和 AdamW preconditioned norm；
- optimizer updates、checkpoint 和 optimizer state；
- per-task teacher loss、formal benchmark、length/truncation/error；
- student/teacher tokens、GPU hours、wall-clock；
- `run_complete.json` 或明确的 failure marker。

建议目录：

```text
outputs/qwen3_1.7b_4t_mopd_gpas/
  frozen_inputs/
    model_and_teachers.json
    train_manifest.yaml
    probe_manifest.yaml
    eval_configs/
  teachers/{math,code,if,science}/seed42/
  single_teacher_opd/{math,code,if,science}/seed42/
  multi_teacher/
    uniform/seed42/
    full_mixture/seed42/
    gap_adaptive/seed42/
    raw_gpas/seed42/
    adam_gpas/seed42/
    adam_gpas_no_is/seed42/
  local_analysis/
    projections/
    weight_branches/
    optimizer_clones/
    order_replay/
  paper/
    tables/
    figures/
    appendix/
```

## 14. 论文结论边界

本实验矩阵可以支持以下三类结论：

1. Qwen3-1.7B 四个同源 GRPO teachers 在共享 student 上产生可测量的 SoloGap、ConflictGap 和 OptGap，这些量可用于选择 fixed-mixture optimization 或 weight edit；
2. AdamW lagged-preconditioned task scale 比 raw norm 更贴近真实 optimizer-dependent update 的程度，可以由同状态 counterfactual data 定量判断；
3. corrected GPAS 在固定四任务目标权重下改变训练预算分配，并与 corrected gap proposal 公平比较 estimator MSE、weighted-loss efficiency 和 trade-off drift。

结论将三类 gap 限定在当前 cache、参数化和半径。Sampled log-ratio 始终使用该名称。训练稳定性由三个 student seeds 度量。AdamW trajectory 差异通过 moment state、参数位移和 matched-loss trade-off 报告。
