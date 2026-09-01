# Qwen3-1.7B 四教师 MOPD：GPAS / Cost-GPAS 单 seed 实验方案

> 版本：2026-09-01  
> 主模型：Qwen3-1.7B  
> 任务：Math、Code、Instruction Following、ScienceQA  
> 状态：受控采样实验已完成；大模型实验待运行

## 1. 实验目标

实验只回答四个问题：

1. response normalization 是否避免回答长度隐式改变任务权重；
2. 乘以 $\lambda_I/q_I$ 后，自适应采样是否仍估计固定的目标梯度 $\sum_i\lambda_i h_i$；
3. AdamW 度量下的 GPAS 是否降低 task-gradient estimator 的误差，Cost-GPAS 是否改善时间加权的代理指标；
4. 这些局部改善是否转化为更好的 token 和 GPU-hour 效率。

受控实验检查公式，冻结梯度实验检查大模型上的局部机制，端到端训练回答效率问题。

## 2. 共享设置

### 2.1 模型与数据

- Student 为固定 revision 的 Qwen/Qwen3-1.7B。
- 四个 teacher 从同一 student checkpoint 出发，分别在 Math、Code、IF 和 ScienceQA 上训练；MOPD 期间冻结 teacher。
- 训练和评测使用不重叠的 prompt。
- 报告 model/tokenizer revision、teacher checkpoint 和每个 teacher 的对应任务得分。

### 2.2 Task update

每次 update 选择一个任务。每条 response 先对 valid token 取平均，再在 task batch 内对 response 取平均：

$$
G_i=\frac1n\sum_{j=1}^n\frac1{L_j}\sum_t g_{j,t}.
$$

| 项目 | 设置 |
|---|---|
| target task weights | $\lambda_i=1/4$ |
| optimizer | AdamW |
| valid response tokens / update | 65,536 |
| optimizer updates | 600 |
| warm start | 8-step round robin |
| response cap | 8,192 |
| rollout | temperature 1, top-p 1, thinking off |
| score / cost EMA | $\gamma=0.95$ |
| probability floor | $q_{\min}=0.05$ |
| training seed | 42 |

下一个 task batch 的 prompt 数由该任务近期的平均 response length 估计。当前 batch 中已完成的 response 全部进入梯度，并记录实际 valid token 数。

梯度操作顺序为：构造 response-normalized $G_i$，应用共享的 task-gradient clip，记录采样 score，乘以 $\lambda_i/q_i$，然后调用 AdamW。correction 后不再进行非线性裁剪。

每个 update 记录 task、$q_i$、$\lambda_i/q_i$、valid tokens、GPU seconds、gradient score 和 teacher loss。

## 3. 比较方法

| 方法 | Proposal | Update factor | 作用 |
|---|---|---|---|
| Uniform | $q_i=1/4$ | 1 | 固定频率 baseline |
| Loss-EMA | $q_i\propto\lambda_i\widehat\ell_i$ | $\lambda_i/q_i$ | 简单的难度自适应 baseline |
| GPAS | $q_i\propto\lambda_i\widehat s_i$ | $\lambda_i/q_i$ | token-efficiency 方法 |
| Cost-GPAS | $q_i\propto\lambda_i\widehat s_i/\sqrt{\widehat c_i}$ | $\lambda_i/q_i$ | GPU-hour-efficiency 方法 |

$\widehat\ell_i$ 是 response-normalized teacher loss 的 EMA，$\widehat s_i^2$ 是 $\|P_tG_i\|_2^2$ 的 EMA，$\widehat c_i$ 是 total GPU seconds 的 EMA。四种方法使用相同的 $\gamma$、probability floor、目标权重和训练预算。

Raw GPAS（$P_t=I$）只作为冻结梯度实验中的几何消融。

## 4. 受控采样实验

现有 Gaussian task-gradient 实验保留。它直接比较 SGD identity metric 和 AdamW metric 下的理论/Monte Carlo MSE，以及 Cost-GPAS 的 time-weighted second moment。

已有结果：SGD identity metric 下 Raw GPAS 的理论/Monte Carlo MSE ratio 为 $0.534/0.534$；AdamW metric 下 GPAS 为 $0.698/0.697$。这一实验用于检查公式和实现，不承担大模型效率结论。

## 5. 冻结梯度机制实验

在 `Uniform` 的 update 0、300 和 600 取 checkpoint。每任务采集 16 个 task-gradient batch，用于估计和比较各种 proposal。

### 5.1 Response normalization

用同一批 cached response 比较：

1. prompt-balanced mixed batch 中对所有 token 做 global mean；
2. 每个任务先做 response normalization，再以 $1/4$ 合并。

报告每任务的 response length、global-mean token share 和 response-normalized coefficient。

### 5.2 Proposal quality

比较 Uniform、Loss-EMA、Raw GPAS、GPAS 和 Cost-GPAS，报告：

$$
M_P(q)=\mathbb E\|P_t\widehat g-P_th\|_2^2,
$$

$$
J_P(q)=\left(\sum_iq_ic_i\right)
\left(\sum_i\frac{\lambda_i^2s_{i,P}^2}{q_i}\right).
$$

主结果是相对 Uniform 的 $M_P$ 和 $J_P$ ratio；Raw GPAS 用于显示 identity metric 与 AdamW metric 的差异。

### 5.3 AdamW score

根据保存的 AdamW moment 和 cached $G_i$ 计算下一步的 task-dependent parameter update。报告 raw norm 与 AdamW-metric norm 对 update norm 的 Spearman correlation。

## 6. 端到端 MOPD

四种方法均运行 seed 42 一次。每 50 updates 在共享 held-out prompt 上评估 weighted teacher loss：

$$
L(N)=\sum_i\lambda_i\ell_i(N).
$$

主结果为：

- normalized teacher-loss AUC vs valid response tokens；
- normalized teacher-loss AUC vs measured GPU hours；
- update 600 的 weighted teacher loss 和每任务 teacher loss；
- update 600 的四任务 capability vector。

token 比较使用 GPAS vs Uniform / Loss-EMA；GPU-hour 比较使用 Cost-GPAS vs Uniform / GPAS。AUC 在方法共同覆盖的资源区间内计算。报告 seed 42 的完整曲线和终点结果。

Capability evaluation 包含：

| Domain | 指标 |
|---|---|
| Math | MATH-500 greedy pass@1 |
| Code | frozen post-cutoff LiveCodeBench pass@1 |
| IF | IFBench strict accuracy |
| ScienceQA | GPQA-Diamond average@4 |

报告四个原始分数及 macro average。

## 7. Multi-task GRPO 迁移

迁移实验使用 MATH、APPS 和 ARC-Challenge，只比较 Uniform、GPAS 和 Cost-GPAS。

| 项目 | 设置 |
|---|---|
| target weights | $\lambda_i=1/3$ |
| valid response tokens / update | 32,768 |
| responses / prompt | 8 |
| optimizer updates | 400 |
| warm start | 6-step round robin |
| seed | 42 |

主指标是 held-out weighted KL-regularized return vs tokens / GPU hours，以及最终 MATH-500、APPS test pass@1 和 ARC-Challenge accuracy。

## 8. 执行顺序

1. 运行 seed 42 的 Uniform、Loss-EMA、GPAS 和 Cost-GPAS。
2. 从 Uniform 的三个 checkpoint 完成冻结梯度分析。
3. 生成 token/GPU-hour 曲线、AUC 和 capability 表。
4. 运行 seed 42 的 GRPO 迁移实验。
