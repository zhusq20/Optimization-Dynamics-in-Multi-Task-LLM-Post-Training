# Qwen3-1.7B 四教师 MOPD：GPAS / Cost-GPAS 实验计划

> 版本：2026-09-01  
> 主模型：Qwen3-1.7B  
> 任务：Math、Code、Instruction Following、ScienceQA  
> 全局随机种子：42  
> 重复策略：每个实验配置只运行一次，不做独立 seed 重复  
> 当前状态：受控采样实验已完成；E1--E5 大模型实验待运行  
> 原则：每个理论结论对应一个直接测量，端到端结果用于验证训练效率；单 seed 结果按探索性、条件性证据解读。  
> 优化器范围：主训练统一使用 AdamW；SGD 作为 identity-metric 特例在受控实验和 cached-gradient audit 中验证，不建立第二套端到端训练矩阵。

## 1. 论文需要回答的问题

论文集中回答六个问题：

1. **Teacher--student compatibility**：同源 teacher 是否在 student rollout 上保持稳定的 next-token overlap、log-ratio tail 和 task-gradient coherence？
2. **长度稳定的任务更新**：response-normalized task batch 能否避免回答长度隐式改变任务权重？
3. **固定目标下的自适应采样**：乘以 $\lambda_I/q_I$ 后，自适应 proposal 的平均梯度能否保持为 $\sum_i\lambda_i h_i$？
4. **优化器度量下的分配**：对 $\widetilde G_i=A_tG_i$，$q_i\propto\lambda_i s_i$ 能否降低 one-task estimator 在所选度量下的 MSE？SGD 取 $A_t=I$，AdamW 取 lagged preconditioner。
5. **成本感知分配**：$q_i\propto\lambda_i s_i/\sqrt{c_i}$ 能否降低“optimizer-metric second moment $\times$ GPU time”？
6. **端到端效率**：局部 estimator 改善能否转化为更低的 teacher-loss AUC，以及更好的 token / GPU-hour efficiency？

动态 token budget 不进入主算法。所有方法使用相同的 fixed-token task batch。这样可以保持 batch 结构、梯度裁剪和理论目标一致，也避免增加 $b_{\min}$、$b_{\max}$、预算更新周期和额外方差模型。

## 2. 理论—证据映射

| 主张 | 实验 | 直接指标 | 端到端指标 |
|---|---|---|---|
| 同源 teacher 提供可吸收的 OPD 信号 | E2、E3 | reverse KL、top-16 overlap、log-ratio tail、gradient coherence、one-step loss change | 每任务 teacher loss / capability vector |
| 长度归一化移除隐式任务权重 | E1 | raw token share、normalized task coefficient、每任务梯度贡献 | 四任务 loss / capability vector |
| $\lambda_I/q_I$ 保持平均梯度 | 受控采样、E2 | corrected mean 对 full-mixture mean 的 error / cosine | Uncorrected GPAS 的 capability shift |
| GPAS 降低所选优化器度量下的 estimator MSE | 受控采样（SGD / AdamW）、E2（AdamW 为主） | 理论/Monte Carlo $M_I(q)$、$M_P(q)$ | teacher loss vs valid tokens |
| score 应与实际优化器几何匹配 | 受控采样、E3 | SGD/raw 和 AdamW-scaled MSE、task-pair ranking、Spearman correlation | GPAS vs Raw-GPAS 附录消融 |
| Cost-GPAS 改善局部计算代理指标 | 受控采样、E1、E2 | cost EMA error、time-weighted second moment | teacher loss vs GPU hours |
| 方法可迁移到 reward-based update | E5 | GRPO estimator audit | weighted return vs tokens / GPU hours |

证据顺序固定为受控采样实验 → E1 → E2 → E3 → E4 → E5。受控实验检查公式与实现，E1--E3 检查大模型机制，E4 报告 MOPD 训练结果，E5 只在主结论成立后检查迁移性。

已完成的受控实验在 SGD identity metric 下得到 Raw GPAS 理论 / Monte Carlo MSE ratio $0.534/0.534$；在 AdamW metric 下得到 GPAS ratio $0.698/0.697$。

## 3. 四教师 MOPD 共享设置

### 3.1 模型、教师和数据

- Student initialization：冻结 revision 的 Qwen/Qwen3-1.7B。
- 四个 teacher 从同一 initialization 分别进行 Math、Code、IF、ScienceQA 单任务后训练；如需重训 teacher，每个 teacher 也只运行 seed 42 一次。
- MOPD 开始前冻结 teacher。
- 报告 model、tokenizer、teacher checkpoint hash 和四域能力向量。
- 每任务冻结 16,000 个训练 prompt。
- score calibration、gradient evaluation、teacher-loss evaluation 和正式 benchmark 使用互不重叠的 prompt。
- calibration 与 evaluation 使用不同 rollout RNG。

Teacher 的训练结果是实验前提。Teacher checkpoint 的选择不能读取 MOPD 结果。

### 3.2 Task update

每次 optimizer update 只选择一个任务。每条 response 先计算 token mean，再在 task batch 内对 response 求平均：

$$
G_i=\frac1n\sum_{j=1}^n\frac1{L_j}\sum_t g_{j,t}.
$$

固定设置：

| 项目 | 设置 |
|---|---|
| target task weights | $\lambda_i=1/4$ |
| training optimizer | AdamW |
| target valid response tokens / update | 65,536 |
| optimizer updates | 600 |
| shared warm start | 16-step round robin |
| response cap | 8,192 |
| rollout | temperature 1, top-p 1, thinking off |
| score / cost EMA | $\gamma=0.95$ |
| probability floor | $q_{\min}=0.05$ |
| checkpoints | 0/150/300/450/600 |
| global seed | 42 |

下一个 task batch 的 prompt 数根据过去的 response length 估计。当前 batch 的所有完整 response 都进入梯度，实际 token 数和 overshoot 全部计入预算。

每个 rollout batch 只用于一次 optimizer update。这个设置让 reward 与当前 actor 保持一致，并把研究范围集中在 task allocation。

### 3.3 Clipping 和 correction 顺序

1. 计算 response-normalized task gradient $G_i$；
2. 使用所有任务共享的 gradient clip；
3. 用裁剪后的梯度记录 identity/SGD score $\|G_i\|_2$ 和 lagged AdamW score $\|P_tG_i\|_2$；
4. corrected 方法乘 $\lambda_i/q_i$；
5. 调用预先固定的 AdamW。

Correction 后不再添加非线性 gradient clip。理论目标是固定 batch 结构下的 clipped task-update mean。
Correction 本身与优化器无关；主实验的采样度量和训练优化器均为 AdamW。

### 3.4 Cost measurement

每个 task update 记录：

- rollout GPU seconds；
- teacher forward GPU seconds；
- actor forward/backward GPU seconds；
- optimizer GPU seconds；
- total GPU seconds；
- valid response tokens；
- prompt count。

Cost-GPAS 对 total GPU seconds 使用与 score 相同的 EMA。它不增加新的 EMA 系数或 cost multiplier。

### 3.5 单 seed 与可复现策略

- 所有训练、calibration、evaluation、bootstrap 和 Monte Carlo 过程的根 seed 统一为 42。
- 对 `(42, stage, task, checkpoint, prompt_id, sample_id)` 做稳定 hash 派生 RNG stream；方法间的 held-out prompt、decoding stream 和评测顺序保持配对。
- 方法名而非 seed 用于区分 run，例如 `Uniform-s42`、`GPAS-s42`、`Cost-GPAS-s42`。
- 中断后只允许从确定性 checkpoint 续训。若必须从头重跑，旧 run 标记为 superseded，不将两次结果当作独立重复或择优。
- 评测集 bootstrap interval 只描述“给定这一次训练轨迹”的 prompt / batch 抽样不确定性，不代表跨 seed 的训练方差。
- 不报告跨 seed 显著性检验，不宣称已证明稳定性或可复现性。

## 4. 端到端方法

| 优先级 | 方法 | Proposal | Update factor | Run | 作用 |
|---|---|---|---|---|---|
| P0 | Uniform | $q_i=1/4$ | 1 | `Uniform-s42` | 主 baseline |
| P0 | GPAS | $q_i\propto\lambda_i\widehat s_i$ | $\lambda_i/q_i$ | `GPAS-s42` | token-efficiency 方法 |
| P0 | Cost-GPAS | $q_i\propto\lambda_i\widehat s_i/\sqrt{\widehat c_i}$ | $\lambda_i/q_i$ | `Cost-GPAS-s42` | GPU-hour-efficiency 方法 |
| P1 | Raw GPAS | $q_i\propto\lambda_i\widehat r_i$ | $\lambda_i/q_i$ | `Raw-GPAS-s42` | SGD/identity-metric proposal；检验 AdamW scaling |
| P1 | Cost-only | $q_i\propto\lambda_i/\sqrt{\widehat c_i}$ | $\lambda_i/q_i$ | `Cost-only-s42` | 检验只偏向低成本任务是否足够 |
| P1 | Uncorrected GPAS | 与 GPAS 相同 | 1 | `Uncorrected-GPAS-s42` | objective-shift 诊断 |

P0 是主结论所需的最小端到端矩阵，所有方法均使用 AdamW。Raw GPAS 在这里是 cross-metric proposal，不是 SGD 端到端 run；它用于隔离 score geometry 的作用，避免将优化器更换与采样策略混在一起。P1 在 E1--E3 机制检查通过且 P0 轨迹出现对应解释需求时按需选择，不默认全部运行。所有配置均只运行 seed 42 一次。
$\widehat r_i$ 跟踪 identity-metric RMS $s_{i,I}$，$\widehat s_i$ 跟踪 AdamW-metric RMS $s_{i,P}$。

附录 P2 消融（仅在 P0 完整、相关 P1 问题仍未解决时按需运行，每项仍只运行 seed 42 一次）：

- Round robin；
- corrected / uncorrected gap sampler；
- Raw Cost-Aware；
- full-mixture reference：四个 task microbatch 分别归一化后以 $1/4$ 累积。

Gap score 使用 response-normalized teacher loss，并除以 warm-start task mean。Gap sampler 与 GPAS 使用相同的 EMA、warm start 和 probability floor。

## 5. E1：Response length 和 task cost audit

### 操作

在 `Uniform-s42` 的 update 0/150/300/450/600，复用 calibration rollouts，记录：

- 平均 response length；
- fixed-token batch 所需 prompt 数；
- prompt-balanced mixed batch 中的 raw token share；
- response-normalized task coefficient；
- task gradient contribution norm；
- task update 的 GPU-time breakdown。

用同一组 cached rollout 构造两个 estimator：

1. prompt-balanced mixed batch，所有 token 做 global mean；
2. 每个任务先做 response normalization，再以 $1/4$ 合并。

### 预测

- 第一种 estimator 的 task coefficient 应随 response length 改变；
- 第二种 estimator 的显式 task coefficient 应为 $1/4$；
- fixed-token task batch 仍可能表现出不同 GPU time，Cost-GPAS 应使用实测成本。

## 6. E2：Corrected-gradient 和 compute-efficiency audit

主 audit 使用 `Uniform-s42` 的五个冻结 checkpoint。每任务生成 16 个 calibration gradient batch 和 16 个独立 evaluation gradient batch。任一任务的 RMS bootstrap relative standard error 超过 10% 时，所有任务统一增加到 32 个 batch。样本量规则在查看方法间差异之前执行，且 bootstrap 与 Monte Carlo 均使用从全局 seed 42 派生的固定 RNG stream。

Calibration bank 构造：

- Uniform proposal；
- Raw GPAS proposal（$A_t=I$，即 SGD identity metric）；
- GPAS proposal（$A_t=P_t$，即 lagged AdamW metric）；
- Cost-only proposal；
- Cost-GPAS proposal；
- floor-constrained evaluation-bank oracle。

其中 $P_t=\operatorname{Diag}((\sqrt{\bar v_{t-1}}+\epsilon)^{-1})$，$s_{i,A}^2=\mathbb E\|AG_i\|_2^2$。两种 metric 使用同一批 cached gradients，不增加 rollout 或 backward。

Evaluation bank 报告：

1. response-normalized sampled reverse KL；
2. teacher / student top-16 next-token overlap；
3. $|\log(\tau/\pi)|$ 的 95th / 99th percentile；
4. identity/SGD 和 AdamW 度量下的 gradient coherence：

$$
C_i^{(A)}=\frac{\|\operatorname{mean}(A G_i)\|_2^2}{\operatorname{mean}\|A G_i\|_2^2},
\qquad A\in\{I,P_t\};
$$

5. corrected mean 对 full-mixture mean 的 relative $\ell_2$ error；
6. corrected mean 对 full-mixture mean 的 cosine；
7. centered optimizer-metric MSE：

$$
M_A(q)=\mathbb E\|A\widehat g-Ah\|_2^2,
\qquad A\in\{I,P_t\};
$$

8. compute criterion（AdamW metric 为主，identity metric 作诊断）：

$$
J_A(q)=\left(\sum_iq_ic_i\right)
\left(\sum_i\frac{\lambda_i^2s_{i,A}^2}{q_i}\right).
$$

每个 proposal 同时报告两种 metric 下相对于 held-out oracle 的 MSE ratio；compute-criterion ratio 以 AdamW metric 为主结果。
下文简记 $M_P=M_{P_t}$、$J_P=J_{P_t}$。

### 预测

- 同源 teacher 应保持非平凡 top-16 overlap，log-ratio tail 不应随训练恶化；
- reverse KL 下降的任务通常应伴随 $s_{i,P}$ 下降；
- 高 AdamW-metric gradient coherence 应与 E3 中正向的 one-step teacher-loss change 一致；
- Corrected GPAS / Cost-GPAS 的 empirical mean 接近 full-mixture mean；
- Uncorrected proposal 的 mean 朝长期高采样任务偏移；
- Raw GPAS 在 identity metric 下的 held-out $M_I(q)$ 低于 Uniform 和 AdamW-metric GPAS；
- GPAS 在 AdamW metric 下的 held-out $M_P(q)$ 低于 Uniform 和 Raw GPAS；
- Cost-GPAS 的 AdamW-metric compute criterion 低于 Uniform、GPAS 和 Cost-only。

### 机制验收口径

- **Correction**：GPAS 和 Cost-GPAS 的 corrected mean 在至少 4/5 个 checkpoint 上同时满足 cosine $\geq0.99$ 且 relative $\ell_2$ error $\leq0.10$。
- **Estimator**：主 gate 使用实际训练优化器的 AdamW metric：GPAS 的 held-out $M_P(q)$ 低于 Uniform；Cost-GPAS 的 held-out $J_P(q)$ 低于 Uniform，且对比 GPAS 的方向单独报告。Identity-metric 结果作机制校验，不改变 gate。
- **Accounting**：total GPU seconds 与四个分项之和的 relative error $\leq2\%$，valid-token / prompt counter 与原始 log 完全一致。
- **Proposal replay**：用保存的 EMA 状态离线重算每次 $q_i$ 与 $\lambda_i/q_i$，与训练 log 在数值容差内逐 update 一致。

Correction 或 accounting / replay 失败视为实现问题，不启动自适应主 run。Estimator 方向不符合预测则视为可报告的科学结果，但不事后修改 score 或 cost 定义。

## 7. E3：SGD / AdamW metric score 和 online tracking

### SGD identity-metric check

对同一批 cached task gradients 取 $A_t=I$。Vanilla SGD 中当前 task gradient 对一步参数更新的贡献为共享标量乘 $G_i$，因此 raw norm 是每个 batch 的精确当前步尺度，其 RMS 估计 $s_{i,I}$。报告 identity-metric MSE 和 task ranking，作为公式与实现校验；不运行独立 SGD 训练轨迹。

### AdamW score

对每个冻结状态创建两个 clone：

- clone A：应用 task gradient $G_i$；
- clone B：应用 zero gradient，并执行相同的 AdamW moment decay。

两者参数差值表示当前 task gradient 对 AdamW step 的贡献。比较：

- raw gradient norm；
- lagged AdamW-scaled norm；
- realized task-dependent parameter change。

两个 clone 使用同一个缓存的 held-out teacher-loss batch。Teacher logits 在冻结 checkpoint 上缓存一次。报告：

- zero-gradient clone loss 减去 task-gradient clone loss；
- one-step loss change 与 $s_{i,P}$ 的关系；
- one-step loss change 与 AdamW-metric gradient coherence $C_i^{(P)}$ 的关系。

大的 $s_{i,P}$ 同时伴随低 coherence 或负 loss change 时，记录为“update scale 未转化为一致局部进展”，不将其直接解释为可学习的 teacher headroom。

报告每 checkpoint 的 task-pair ranking accuracy 和 Spearman correlation，并与上述 identity-metric 结果并列。

### Online tracking

`Uniform-s42` 主 run 每 25 updates 记录：

- current held-out raw / AdamW score；
- online AdamW-score EMA；
- stale score snapshots；
- warm-start static score；
- current measured task cost；
- cost EMA 对下一次 task cost 的 relative error。

$\gamma=0.95$ 在主训练前冻结，`Uniform-s42` 轨迹只用于报告 tracking error，不用于重选 $\gamma$。

## 8. E4：End-to-end MOPD

每 25 updates 用固定 held-out prompt 和方法间共享的 paired decoding stream 计算：

$$
L(N)=\sum_i\lambda_i\ell_i(N).
$$

主曲线：

1. weighted teacher loss vs valid response tokens；
2. weighted teacher loss vs measured GPU hours。

主比较：

- GPAS vs Uniform：token-efficiency；
- Cost-GPAS vs Uniform / GPAS：GPU-hour-efficiency。

三个 P0 方法均使用 AdamW；E4 比较的是 task proposal，不是 optimizer choice。

不再使用独立 Uniform pilot 选 target。在任何端到端结果可见之前，预注册两个相对 weighted teacher-loss target：$L(N)/L(0)=0.90$ 和 $0.80$。正文报告：

- 达到每个 target 所需的 valid tokens 和 GPU hours；
- 共享预算区间内的 normalized teacher-loss AUC。每个坐标分别取各方法共同覆盖的累积资源区间，对每 25 updates 的评测点线性插值后用梯形法积分，再除以该区间宽度和 $L(0)$。

单 seed 下，每个方法只产生一条训练轨迹。比较报告：

- 相对 `Uniform-s42` 的 teacher-loss AUC 差；
- 达到两个预注册 target 的 token / GPU-hour 差，未达到时明确标记 NR；
- 在共享资源区间 25%/50%/75%/100% 位置的配对 loss 差，作为轨迹方向一致性检查，不将这四个位置当作独立重复；
- teacher-loss 在任务内对 held-out prompt 做 paired stratified bootstrap，E2 estimator 指标对 gradient batch 做配对 bootstrap；两者的 95% interval 都明确标注不包含训练 seed 不确定性。

正文只在以下条件同时满足时写“在 seed 42 上观察到效率改善”：

- 对应资源坐标下的 AUC 优于预先指定的对照：GPAS 对 Uniform，Cost-GPAS 对 Uniform 和 GPAS；
- 四个共享资源位置中至少 3/4 个方向一致；
- E2 的 estimator 指标方向一致；
- 没有额外 teacher calls 或 backward。

若不满足上述条件，如实报告单轨迹结果，不使用“稳定优于”、“显著提升”或“可复现”等跨 seed 表述。

所有方法报告：

- response tokens；
- teacher-scored tokens；
- prompts；
- optimizer updates；
- GPU hours / wall-clock；
- task proposal 和 $\lambda_i/q_i$；
- truncation / invalid output；
- 每任务 teacher loss。

### Capability evaluation

Update 0/300/600 评测：

| Domain | 主指标 |
|---|---|
| Math | MATH-500 greedy pass@1 |
| Code | frozen post-cutoff LiveCodeBench pass@1 |
| IF | IFBench strict |
| ScienceQA | GPQA-Diamond average@4 |

报告四个原始分数、macro average 和最低任务分数。方法间共享 prompt 和 decoding stream；95% paired stratified bootstrap interval 仅反映评测集不确定性。

## 9. E5：Multi-task GRPO transfer

E5 是条件性迁移实验：只在 E2 的 correction / estimator audit 通过且 E4 在 seed 42 上呈现与机制一致的方向时启动。所有 GRPO 配置同样只运行 seed 42 一次。

任务：MATH train、APPS train、ARC-Challenge train。

- target weights：$\lambda_i=1/3$；
- task-homogeneous batch；
- 32,768 valid response tokens / update；
- 8 responses / prompt；
- 400 optimizer updates；
- 12-step shared round-robin warm start；
- 保留 zero-variance reward group。

P0 迁移比较为 Uniform、GPAS 和 Cost-GPAS。Sequence-level Cost-Aware GRPO 作为 P1 reference：它保持 task proposal 为 $q_i=\lambda_i$，在选中任务的 rollout 完成后按照 $|A|/\sqrt{\text{prompt+response length}}$ 重采样 response，并使用 importance correction 和小的 uniform mixture。GPAS / Cost-GPAS 在 rollout 前选择任务。
所有 P0 迁移方法继续使用 AdamW。

主指标为 held-out weighted KL-regularized return vs tokens / GPU hours。报告 MATH-500、APPS test pass@1 和 ARC-Challenge accuracy。Corrected gap sampler 和 task-level Raw Cost-Aware 放入诊断实验。

Multi-Task GRPO 使用其公开 task-weight rule，作为 P1 capability-oriented reference。E5 沿用 E4 的单轨迹报告与措辞规则。

## 10. 执行顺序与停止条件

最小主证据预算是 3 个 P0 MOPD run，共 $3\times600=1{,}800$ 个 optimizer updates，加上 E1--E3 的冻结 checkpoint audit。P1、P2 和 E5 不预留为默认连续队列；它们必须通过下述 gate 再单独批准，以免诊断实验先于主证据消耗算力。

1. **冻结协议**：固定数据 manifest、checkpoint hash、seed 42、RNG 派生规则、评测集和主指标。
2. **完成受控实验复核**：确认 SGD identity metric 和 AdamW metric 下的 correction、MSE、floor 和 cost criterion 与公式一致。
3. **运行 `Uniform-s42`**：同步采集 E1--E3 所需 checkpoint 和缓存，避免额外 baseline / pilot 重跑。
4. **机制 gate**：按 E2 预注册的 correction、accounting 和 proposal-replay 口径验收。任一实现 gate 失败都停止启动自适应主 run，先修复实现；estimator 方向不符合预测则保留为负结果。
5. **运行 P0 自适应方法**：依次完成 `GPAS-s42` 和 `Cost-GPAS-s42`，不中途根据曲线改超参。
6. **冻结 E4 结果**：先产出 P0 表格、曲线和失败案例，再决定哪个 P1 对照能解决具体归因问题。
7. **条件性扩展**：P1/P2 不用于在多个对照中事后挑选最有利结果；E5 只在 MOPD 主证据链完整后运行。

## 11. 执行与发布检查清单

### 任一主 run 启动前

- [ ] 模型、tokenizer、teacher hashes 冻结；
- [ ] 数据 manifests 与去重记录冻结；
- [ ] 全局 seed 42 和所有 RNG namespace 冻结；
- [ ] $\lambda$、$\gamma$、$q_{\min}$、clip 和 token target 冻结；
- [ ] training optimizer 固定为 AdamW，identity / AdamW metric 定义已冻结；
- [ ] protocol version、run ID 和完整配置已保存；
- [ ] raw / AdamW score 均在 $\lambda_i/q_i$ 之前记录；
- [ ] cost 计时边界对所有任务一致；
- [ ] 每个 rollout 只执行一次 optimizer update；
- [ ] calibration / evaluation prompt 与 RNG 分离；

### 自适应主 run 启动前

- [ ] E1--E3 feasibility checks 完成；
- [ ] correction、accounting 和 proposal replay 通过 E2 机制 gate；

### 结果发布前

- [ ] 结果表保留所有 task、checkpoint 和 run；
- [ ] bootstrap interval 标注“不包含训练 seed 不确定性”；
- [ ] 论文措辞明确限定为 seed 42 的单次训练轨迹。
