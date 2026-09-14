# 实验图 Caption 草稿

下列说明用于论文caption；不写入图片。正文候选顺序：F1c → F2b → F5a → F5b → F7a → F7b。其他图作为补充诊断。

**共同范围**：全部为固定学生与保存Adam状态下的HF局部梯度/模拟BF16写回，不是实际线上训练步或能力评测。

I64是学生与教师Top64交集；权重仍按原学生Top64归一化，交集上不再归一化。诊断bank不是训练seed；所有范围均为描述性观测范围，不是置信区间。

## F1c

审稿人问题：DR、DT、GT究竟怎样改变同一批回答的训练权重？

Effective coefficients multiplying each response-mean loss on the same PG500 bank. Math/code/instruction-following/science supply 1/2/3/4 responses and the intended domain prior is uniform. Domain-response (DR) gives equal weight to responses within each domain; domain-token (DT) changes their relative weights with length while preserving the domain total; global-token (GT) also changes domain totals. All columns sum to 100%. The matched prompt-share-prior control is retained in the source data.

## F2b

审稿人问题：只改平均方式，梯度方向是否也会改变？

Cosine similarity of pre-clipping gradients under domain-response (DR), domain-token (DT), and global-token (GT) averaging. Within each row, the student, saved optimizer state and responses are identical; cosine one means equal direction. Row conditions are as in F2a. Direction differences depend on the batch: for example, the long-response row gives 0.706 and 0.658 for DR–DT and DR–GT. These local directions do not rank online capability.

## F5a

审稿人问题：教师梯度的近正交，是否仍存在于相同输入上？

Teacher-pair raw-gradient alignment under task-specific versus shared inputs. PG and I64 are shown at both student checkpoints. Each marker is the mean of all six teacher pairs in two diagnostic banks within the labeled condition; horizontal segments show the full observed minimum–maximum over those 12 dependent pair/bank values, not a confidence interval. Small vertical offsets separate the two input conditions. Routed inputs mix teacher and task differences; the shared-input control uses the same prefix bank and PG actions across teachers. Near-orthogonal routed gradients therefore do not establish disjoint teacher-specific subnetworks.

## F5b

审稿人问题：原始梯度近正交，是否意味着实际写回方向也近正交？

Raw-gradient and BF16 proposed-update cosine on the same teacher pairs, with input conditions in separate rows. Markers and minimum–maximum segments summarize all six pairs across two banks per condition as in F5a; overlapping numerical ranges occupy separate vertical lanes. All local updates start from the same saved Adam state within a checkpoint. Stronger alignment after optimizer processing does not attribute historical updates to individual teachers; zero-current-gradient controls are shown in F5c.

## F7a

审稿人问题：在同一学生状态上，更密的监督是否明显扩大参数写回范围？

Percentage of parameters with absolute local BF16 proposed change greater than 1e-5. Each row fixes the student and saved Adam state; each cell averages four diagnostic banks (42–45). Read across a row to compare objectives, rather than attributing differences between independently trained states to the local loss. Zero gradient retains Adam history; Student Top64, intersection with teacher Top64 (I64), teacher-selected Top64 and full vocabulary are distinct local objectives. Asterisks denote mixed Student64-to-I64 training history, not all-I64 training. Other thresholds and bank ranges are available in the data tables.

## F7b

审稿人问题：目标改变了梯度尺度，是否也按同等比例改变写回尺度？

Pre-clipping raw-gradient and BF16 proposed-update L2 norms, each divided by the corresponding PG norm at the same student checkpoint. Values are ratios of four-bank means, not means of per-bank ratios and not gradient-to-update ratios. The PG column would be exactly one and is omitted; both rows use the same reference level and color scale. The zero-gradient column still shows nonzero writebacks from saved Adam history. Similar update norms do not imply identical update directions or learning outcomes; F7c reports angular differences. Asterisks identify mixed training history as in F7a.

## F1a

审稿人问题：诊断批次中的长度差异有多大，512-token上限影响了哪些观测？

Response lengths in four fixed diagnostic banks (columns: student checkpoint / bank draw). Rows enumerate the two responses per domain within each bank; response indices do not identify matched prompts across banks. Every 512-token response is truncated at the cap; shorter responses completed. These capped local banks are not an estimate of the natural online response-length distribution.

## F1b

审稿人问题：相同prompt配额下，各任务的token份额偏离了多少？

Token share minus prompt share, in percentage points, for the banks in F1a. Each domain contributes 25% of prompts; positive values indicate a larger share of generated tokens and negative values a smaller share. This is the domain weighting induced by global-token averaging on these batches, not the domain coefficient under domain-response averaging. Rows remain separate even when numerical values coincide.

## F2a

审稿人问题：长度加权造成的梯度修正主要出现在哪些任务？

Magnitude of the length-weighting correction, ||Cov(T,g)|| / mean(T), by domain and diagnostic condition. The first four rows use balanced cap512 banks, the fifth cap2048, and the last two share quota1/2/3/4 responses under uniform/prompt-share domain priors. Both quota banks and the long bank use draw42. Displayed 0.00 values are rounded; full precision is retained. The fixed-batch identity has maximum relative residual 1.8e-16 and does not itself establish capability improvement or distributed reducer correctness.

## F2c

审稿人问题：局部KL改善是否超过仅沿用Adam历史的zero-gradient对照？

Additional macro held-out KL decrease relative to the zero-gradient Adam control: 1000 × (KL decrease under the reduction − KL decrease under zero current gradient), in millinats. Positive cells outperform the control on this diagnostic bank; negative cells do not. Each domain has only one held-out response. The matched quota reruns also exhibit small unexplained GT numerical differences, so these exploratory measurements cannot support a rule ranking, significance claim or capability claim. Unadjusted values, including the control, remain in heldout.csv.

## F5c

审稿人问题：不同教师的写回是否大量包含zero-gradient也会移动的参数？

Support overlap with the zero-current-gradient Adam control at absolute BF16 threshold 1e-5. Each point is the mean over four teachers and two banks within one labeled condition, with the full minimum–maximum range. The layer-random reference is the ratio of expected intersection to expected union under matched per-layer counts, not the expected Jaccard itself. The horizontal axis is logarithmic. A separate all-coordinate audit at PG250/500 found BF16(saved master) identical to the saved model before any Adam step, excluding pre-existing master/model disagreement at those two checkpoints.

## F6a

审稿人问题：哪对教师的输出分布更不同？

Full-vocabulary Jensen–Shannon divergence between teachers on shared student prefixes. Cells average two banks within a student checkpoint. Rows are ordered by mean JS across the four banks solely to define a common display order for F6a–c; checkpoint values remain separate. JS is measured before applying the local objective and is not duplicated by PG/I64. The teachers are fixed: checkpoint-specific differences reflect changed student prefix banks.

## F6b

审稿人问题：JS较大的教师对是否也有更不同的更新位置？

BF16 update-set distance under task-specific inputs, at absolute threshold 1e-5. A value of zero means identical selected parameter sets. Each cell averages two banks for one teacher pair, checkpoint and loss. Teacher-pair order is identical to F6a, and F6b/c share the same numerical scale and column order. All PG/I64 and 250/500 conditions are retained. The small teacher pool and shared pairs support a descriptive comparison only; no regression, significance test or universal monotone JS–update relationship is claimed.

## F6c

审稿人问题：共享输入后，教师对的更新位置差异如何变化？

BF16 update-set distance under shared inputs, at absolute threshold 1e-5. A value of zero means identical selected parameter sets. Each cell averages two banks for one teacher pair, checkpoint and loss. Teacher-pair order is identical to F6a, and F6b/c share the same numerical scale and column order. All PG/I64 and 250/500 conditions are retained. The small teacher pool and shared pairs support a descriptive comparison only; no regression, significance test or universal monotone JS–update relationship is claimed.

## F7c

审稿人问题：写回范数相近时，方向是否仍有差异？

Angle between each local BF16 proposed update and the full-vocabulary update from the same student, Adam state and bank. Each cell is the mean of four per-bank angles, computed as arccos(cosine) in degrees; it is not the angle of a mean update or arccos of a mean cosine. Zero degrees means aligned directions. Full-versus-full is zero by construction and omitted. Full vocabulary is a local reference, not a claim of an optimal direction or a completed online capability baseline. Asterisks retain mixed-history provenance.

