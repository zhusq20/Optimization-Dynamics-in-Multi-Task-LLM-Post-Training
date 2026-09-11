# Figure captions

Static training curves retain all raw points. Local-probe envelopes are bank min–max, not training-seed confidence intervals.

## 01_eval_math

MATH-500 pass@1：联合训练的能力随训练如何变化？

Raw MATH-500 pass@1 evaluations at the recorded checkpoints; no smoothing or confidence bands. Blue: joint four-domain PG. Orange: math-only PG, also evaluated on other domains. Both start from the same measured initial checkpoint. Dotted and dashed lines are measured student and domain-teacher references. One training seed; the first complete verified evaluation attempt is selected by completion time, never by score. All attempts are retained in evaluation_attempts.csv and the raw explorer. The math-only series stops at its last available evaluation. Equal updates do not imply equal per-domain exposure; see the math sample-exposure view.

## 01_eval_code

LiveCodeBench pass@1：联合训练的能力随训练如何变化？

Raw LiveCodeBench pass@1 evaluations at the recorded checkpoints; no smoothing or confidence bands. Blue: joint four-domain PG. Orange: math-only PG, also evaluated on other domains. Both start from the same measured initial checkpoint. Dotted and dashed lines are measured student and domain-teacher references. One training seed; the first complete verified evaluation attempt is selected by completion time, never by score. All attempts are retained in evaluation_attempts.csv and the raw explorer. The math-only series stops at its last available evaluation. Equal updates do not imply equal per-domain exposure; see the math sample-exposure view.

## 01_eval_if

IFBench strict：联合训练的能力随训练如何变化？

Raw IFBench strict evaluations at the recorded checkpoints; no smoothing or confidence bands. Blue: joint four-domain PG. Orange: math-only PG, also evaluated on other domains. Both start from the same measured initial checkpoint. Dotted and dashed lines are measured student and domain-teacher references. One training seed; the first complete verified evaluation attempt is selected by completion time, never by score. All attempts are retained in evaluation_attempts.csv and the raw explorer. The math-only series stops at its last available evaluation. Equal updates do not imply equal per-domain exposure; see the math sample-exposure view.

## 01_eval_science

GPQA Diamond avg@4：联合训练的能力随训练如何变化？

Raw GPQA Diamond avg@4 evaluations at the recorded checkpoints; no smoothing or confidence bands. Blue: joint four-domain PG. Orange: math-only PG, also evaluated on other domains. Both start from the same measured initial checkpoint. Dotted and dashed lines are measured student and domain-teacher references. One training seed; the first complete verified evaluation attempt is selected by completion time, never by score. All attempts are retained in evaluation_attempts.csv and the raw explorer. The math-only series stops at its last available evaluation. Equal updates do not imply equal per-domain exposure; see the math sample-exposure view.

## 02_eval_math_exposure

数学能力差异能否由训练样本暴露解释？

The same selected MATH-500 scores as 01_eval_math, against cumulative math prompts from deduplicated allocation records. One response per prompt in these runs. The joint run uses 16 math prompts per update and the math-only run uses 64. This plot compares sample exposure, not wall-time or GPU efficiency.

## 03_train_domain_kl

学生接近各领域教师的速度是否一致？

Joint PG, seed 42. All recorded teacher_loss observations, without smoothing, rebinning, or downsampling. The horizontal clock is mopd/update. Colors identify domains consistently across the training diagnostics. teacher_loss in this PG run estimates the sampled student-minus-teacher log-probability ratio, not an exact full-vocabulary KL. The four series describe one run, not four independent seeds.

## 04_train_token_share

相同prompt份额是否带来相同token份额？

Joint PG, seed 42. All recorded token_share observations, without smoothing, rebinning, or downsampling. The horizontal clock is rollout/step. Colors identify domains consistently across the training diagnostics. teacher_loss in this PG run estimates the sampled student-minus-teacher log-probability ratio, not an exact full-vocabulary KL. The four series describe one run, not four independent seeds.

## A01_response_length

token份额变化是否伴随回答长度变化？

Joint PG, seed 42. All recorded mean_response_length observations, without smoothing, rebinning, or downsampling. The horizontal clock is rollout/step. Colors identify domains consistently across the training diagnostics. teacher_loss in this PG run estimates the sampled student-minus-teacher log-probability ratio, not an exact full-vocabulary KL. The four series describe one run, not four independent seeds.

## A02_truncation

训练回答是否越来越容易触及长度上限？

Joint PG, seed 42. All recorded truncation_rate observations, without smoothing, rebinning, or downsampling. The horizontal clock is rollout/step. Colors identify domains consistently across the training diagnostics. teacher_loss in this PG run estimates the sampled student-minus-teacher log-probability ratio, not an exact full-vocabulary KL. The four series describe one run, not four independent seeds.

## A03_raw_reward_math

math：原始training reward怎样变化？

Raw rollout/reward/math/mean, with every recorded observation. This reward is observed on training prompts and has coefficient zero in the OPD loss. Verifier and infrastructure problems, if present in the original run, remain in the raw values; these curves are not rescored or substituted for benchmark evaluations.

## A03_raw_reward_code

code：原始training reward怎样变化？

Raw rollout/reward/code/mean, with every recorded observation. This reward is observed on training prompts and has coefficient zero in the OPD loss. Verifier and infrastructure problems, if present in the original run, remain in the raw values; these curves are not rescored or substituted for benchmark evaluations.

## A03_raw_reward_if

if：原始training reward怎样变化？

Raw rollout/reward/if/mean, with every recorded observation. This reward is observed on training prompts and has coefficient zero in the OPD loss. Verifier and infrastructure problems, if present in the original run, remain in the raw values; these curves are not rescored or substituted for benchmark evaluations.

## A03_raw_reward_science

science：原始training reward怎样变化？

Raw rollout/reward/science/mean, with every recorded observation. This reward is observed on training prompts and has coefficient zero in the OPD loss. Verifier and infrastructure problems, if present in the original run, remain in the raw values; these curves are not rescored or substituted for benchmark evaluations.

## A04_raw_train_loss

train/loss：未经平滑的训练记录是什么样？

Raw train/loss against train/step, without smoothing or filtering. A train/step increment is a logged gradient microbatch, not an optimizer update. The two PG runs use the same logged metric. PG surrogate loss and sampled KL have distinct meanings. Zero train/grad_norm values from accumulation slices are not used as the aggregate optimizer gradient norm.

## A05_raw_sampled_logratio

train/sampled_reverse_kl_logratio：未经平滑的训练记录是什么样？

Raw train/sampled_reverse_kl_logratio against train/step, without smoothing or filtering. A train/step increment is a logged gradient microbatch, not an optimizer update. The two PG runs use the same logged metric. PG surrogate loss and sampled KL have distinct meanings. Zero train/grad_norm values from accumulation slices are not used as the aggregate optimizer gradient norm.

## A06_aggregate_gradient

mopd/aggregate_grad_norm：未经平滑的训练记录是什么样？

Raw mopd/aggregate_grad_norm against mopd/update, without smoothing or filtering. A train/step increment is a logged gradient microbatch, not an optimizer update. The two PG runs use the same logged metric. PG surrogate loss and sampled KL have distinct meanings. Zero train/grad_norm values from accumulation slices are not used as the aggregate optimizer gradient norm.

## A07_eval_continuations_math

MATH-500 pass@1：已有归一化续训分支有什么评估证据？

Existing joint Student64-to-I64 continuations, with distinct domain-response (DR), domain-token (DT), and global-token (GT) reductions. I64 begins at updates 137/152/151 for DR/DT/GT. Their step-100 evaluations are inherited Student64 checkpoints, not I64-trained endpoints. Only completed verified evaluations are plotted; no extrapolation to unmeasured checkpoints. Dashed connectors cross a change of objective. These are mixed-history trajectories, not from-base controlled I64 ablations.

## A07_eval_continuations_code

LiveCodeBench pass@1：已有归一化续训分支有什么评估证据？

Existing joint Student64-to-I64 continuations, with distinct domain-response (DR), domain-token (DT), and global-token (GT) reductions. I64 begins at updates 137/152/151 for DR/DT/GT. Their step-100 evaluations are inherited Student64 checkpoints, not I64-trained endpoints. Only completed verified evaluations are plotted; no extrapolation to unmeasured checkpoints. Dashed connectors cross a change of objective. These are mixed-history trajectories, not from-base controlled I64 ablations.

## A07_eval_continuations_if

IFBench strict：已有归一化续训分支有什么评估证据？

Existing joint Student64-to-I64 continuations, with distinct domain-response (DR), domain-token (DT), and global-token (GT) reductions. I64 begins at updates 137/152/151 for DR/DT/GT. Their step-100 evaluations are inherited Student64 checkpoints, not I64-trained endpoints. Only completed verified evaluations are plotted; no extrapolation to unmeasured checkpoints. Dashed connectors cross a change of objective. These are mixed-history trajectories, not from-base controlled I64 ablations.

## A07_eval_continuations_science

GPQA Diamond avg@4：已有归一化续训分支有什么评估证据？

Existing joint Student64-to-I64 continuations, with distinct domain-response (DR), domain-token (DT), and global-token (GT) reductions. I64 begins at updates 137/152/151 for DR/DT/GT. Their step-100 evaluations are inherited Student64 checkpoints, not I64-trained endpoints. Only completed verified evaluations are plotted; no extrapolation to unmeasured checkpoints. Dashed connectors cross a change of objective. These are mixed-history trajectories, not from-base controlled I64 ablations.

## 05_normalization_direction

仅改变平均规则，会改变梯度方向吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Each point is one recorded pre-clipping gradient cosine, replacing the previous seven-condition heatmap. The first four rows use cap512; the long-response row uses cap2048. The final two rows use the same unequal quota bank with uniform and prompt-proportional domain priors. The reference at one indicates identical directions. No ordering or interpolation is imposed on these categorical conditions.

## 06_teacher_input_control

教师方向差异来自教师本身，还是来自输入不同？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Means and full minimum–maximum ranges over six teacher pairs and two banks per condition. PG and I64 are local probe objectives on the indicated PG-trained states. Shared inputs change the observed gradient alignment substantially. Ranges show the observed dependent measurements, not confidence intervals.

## A08_gradient_vs_writeback

原始梯度近正交，BF16写回也近正交吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Task-specific inputs. Each marker is the mean over the same six pairs and two banks; segments are observed min–max. Strong update alignment does not identify teacher-specific historical subnetworks.

## 07_threshold_m-pg500

m-pg500：监督目标的变化范围结论依赖阈值吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Student state m-pg500, four banks 42–45. Lines are means; faint envelopes are observed bank min–max, not confidence intervals. Threshold zero means exact nonzero BF16 changes. All thresholds were measured; none are interpolated observations. The main PG500 view focuses on PG, intersection, full vocabulary and zero-current-gradient; Student64 and Teacher64 remain in the source table and other state views. I64-labeled student states have mixed Student64-to-I64 histories. The zero-gradient control retains Adam state. Similar support sizes do not establish equal directions or capabilities.

## 07_threshold_m-pg250

m-pg250：监督目标的变化范围结论依赖阈值吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Student state m-pg250, four banks 42–45. Lines are means; faint envelopes are observed bank min–max, not confidence intervals. Threshold zero means exact nonzero BF16 changes. All thresholds were measured; none are interpolated observations. The main PG500 view focuses on PG, intersection, full vocabulary and zero-current-gradient; Student64 and Teacher64 remain in the source table and other state views. I64-labeled student states have mixed Student64-to-I64 histories. The zero-gradient control retains Adam state. Similar support sizes do not establish equal directions or capabilities.

## 07_threshold_s-pg250

s-pg250：监督目标的变化范围结论依赖阈值吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Student state s-pg250, four banks 42–45. Lines are means; faint envelopes are observed bank min–max, not confidence intervals. Threshold zero means exact nonzero BF16 changes. All thresholds were measured; none are interpolated observations. The main PG500 view focuses on PG, intersection, full vocabulary and zero-current-gradient; Student64 and Teacher64 remain in the source table and other state views. I64-labeled student states have mixed Student64-to-I64 histories. The zero-gradient control retains Adam state. Similar support sizes do not establish equal directions or capabilities.

## 07_threshold_m-i64dr250

m-i64dr250：监督目标的变化范围结论依赖阈值吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Student state m-i64dr250, four banks 42–45. Lines are means; faint envelopes are observed bank min–max, not confidence intervals. Threshold zero means exact nonzero BF16 changes. All thresholds were measured; none are interpolated observations. The main PG500 view focuses on PG, intersection, full vocabulary and zero-current-gradient; Student64 and Teacher64 remain in the source table and other state views. I64-labeled student states have mixed Student64-to-I64 histories. The zero-gradient control retains Adam state. Similar support sizes do not establish equal directions or capabilities.

## 07_threshold_s-i64250

s-i64250：监督目标的变化范围结论依赖阈值吗？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. Student state s-i64250, four banks 42–45. Lines are means; faint envelopes are observed bank min–max, not confidence intervals. Threshold zero means exact nonzero BF16 changes. All thresholds were measured; none are interpolated observations. The main PG500 view focuses on PG, intersection, full vocabulary and zero-current-gradient; Student64 and Teacher64 remain in the source table and other state views. I64-labeled student states have mixed Student64-to-I64 histories. The zero-gradient control retains Adam state. Similar support sizes do not establish equal directions or capabilities.

## A09_objective_norms

监督目标改变了梯度尺度，是否也改变写回尺度？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. PG500, four banks. Ratios use the matched PG mean for each quantity separately; means are ratios of means. Segments show the per-bank range after division by that fixed denominator. Replaces the F7b heatmap.

## A10_teacher_overlap_matrix

同一输入下，四个教师的BF16写回集合有多重叠？

Local HF probes at fixed student checkpoints and saved Adam states; simulated BF16 writeback, not actual online optimizer steps. Diagnostic banks and shared teacher pairs are not training seeds. PG500 state, common inputs, I64 local objective, absolute threshold 1e-5. Each off-diagonal cell averages the two recorded banks. Diagonal entries equal one by identity. This selected controlled condition complements the full PG/I64 and step250/500 input comparisons, not a universal teacher-overlap result.
