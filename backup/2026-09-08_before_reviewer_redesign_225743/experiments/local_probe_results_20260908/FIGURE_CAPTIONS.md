# 实验图 Caption 草稿

每个文件只含一张独立图。图内仅保留轴、刻度、矩阵数值和必要图例；标题、实验条件与解释放在以下 caption。

所有新图均为保存 Adam 状态上的 HF 局部模拟 BF16 写回，不是真实在线单步或能力评测。不同条件保持分开，诊断 bank 不是训练 seed。

## F1a

Response lengths in four local diagnostic banks at M-PG checkpoints 250 and 500 (draws 42/43). Each bank contains two training responses per domain. Crosses denote truncation at the 512-token cap; these are capped diagnostic lengths, not an estimate of the natural online length distribution.

## F1b

Domain token shares in the same four cap512 banks as F1a. Each domain supplies 25% of prompts (dotted line). Token shares differ from prompt shares and must not be interpreted as the domain coefficients of response-normalized DR.

## F1c

Effective response coefficients for domain-response (DR), domain-token (DT), and global-token (GT) averaging on one shared PG500 bank. Domain quotas are 1/2/3/4 for math/code/IF/science, with a uniform domain prior. M, C, I, and S denote these domains. The matched prompt-share prior control is retained in the data package.

## F2a

Magnitude of the length-weighting correction, ||Cov(T,g)|| / mean(T), for each domain in all seven normalization probes. Zero corrections are retained. The long bank uses cap2048; other banks use cap512. Quota/prompt and quota/uniform reuse the same unequal-quota responses. The covariance identity has maximum relative residual about 1.8e-16; this checks a fixed-batch identity, not distributed reducer correctness.

## F2b

Raw-gradient cosine for the three reduction pairs in all seven probes, with the same student, saved Adam state, and responses within each probe. The values describe pre-clipping gradients, not BF16 writebacks. The long-response condition gives DR–DT and DR–GT cosines of approximately 0.706 and 0.658.

## F2c

Macro-average held-out full-vocabulary reverse-KL decrease (before minus after) for the zero-gradient Adam control and DR/DT/GT. All seven positive and negative outcomes are shown. Each domain has only one held-out response. Small cross-run numerical differences were observed even for the theoretically prior-invariant GT control on the reused quota bank, so this panel is exploratory and does not establish a ranking of downstream capability.

## F5a

Teacher-pair overlap of BF16 proposed-writeback supports (absolute threshold 1e-5), jointly showing checkpoints 250/500, PG/I64 losses, and routed/common inputs. Each cell averages the two diagnostic banks 42/43 within its labeled condition; conditions are not pooled. R denotes each teacher’s routed domain; C denotes identical common four-domain prefixes and matched PG actions. Only the six distinct teacher pairs are shown, avoiding redundant diagonal and symmetric cells. Support counts, bank-level ranges and equal-count controls remain in the source tables.

## F5b

Raw-gradient versus BF16 proposed-writeback alignment for every teacher pair, bank, checkpoint and loss. Color identifies PG/I64, circle/square identifies step250/500, and filled/open markers identify routed/common inputs. The 96 plotted comparisons share teachers and contexts and are not independent replicates. Matched inputs yield highly aligned gradients, whereas routed gradients are nearly orthogonal despite substantial BF16 alignment.

## F5c

Overlap with the zero-gradient Adam control for each teacher, checkpoint, loss and input condition. R/C denote routed/common as in F5a. Dots are individual-bank Jaccards; crosses are the layer-matched ratio of expected intersection to expected union, not an empirical expected Jaccard. The logarithmic axis exposes the separation from the random reference. Shared optimizer history must be considered when interpreting BF16 support overlap.

## F6a

Full-vocabulary Jensen–Shannon divergence between each distinct teacher pair, averaged over two common-prefix banks at each student checkpoint. JS is computed before applying either local loss and is therefore shown once rather than duplicated for PG and I64. Teachers are fixed; checkpoint-specific differences reflect different generated prefix banks, not changing teacher weights.

## F6b

Teacher JS on common prefixes versus BF16 support distance under routed inputs, combining both checkpoints and both losses while preserving their labels. Each point represents one teacher pair in one bank. F6b and F6c use identical axis limits for a matched routed/common comparison. Pairwise observations are dependent; no regression, significance test, or causal relationship is asserted.

## F6c

Teacher JS on common prefixes versus BF16 support distance under common inputs, combining both checkpoints and both losses while preserving their labels. Each point represents one teacher pair in one bank. F6b and F6c use identical axis limits for a matched routed/common comparison. Pairwise observations are dependent; no regression, significance test, or causal relationship is asserted.

## F7a

Percentage of parameters whose local BF16 proposed writeback exceeds the prespecified absolute threshold 1e-5, jointly showing all five student states and six objectives/controls. Displayed units are 0.01 percentage points (a cell value of 3.52 represents 0.0352%). Cells average four diagnostic banks (42–45) within each condition. Zero is the zero-gradient Adam control; ST64 selects student Top64, I64 their intersection with teacher Top64, T64 selects teacher Top64, and Full is the full-vocabulary local reference. Asterisks mark states with mixed Student64→I64 training history. Other thresholds and bank ranges remain in the plotting tables; the full threshold sweep is not overlaid as 30 unreadable curves.

## F7b

Pre-clipping raw-gradient L2 versus BF16 proposed-writeback L2 for all five student states, six objectives and four banks (120 observations, with identical zero-control coordinates overlapping). Color denotes objective and marker denotes student state. The gradient axis uses a symmetric logarithmic scale to retain the zero control. The writeback axis is linear and explicitly zoomed to the observed range. The two norms are distinct quantities; similar writeback norms do not establish equal directions or learning effects. Asterisks denote mixed training history.

## F7c

Cosine similarity between each local BF16 proposed writeback and the full-vocabulary reference from the same student, Adam state and prefix bank. Cells average four banks per state/objective. All five states are shown together; the trivial full-versus-full column is omitted. Asterisks retain mixed-history provenance. The full-vocabulary branch is a local reference, not an online-trained capability baseline.

