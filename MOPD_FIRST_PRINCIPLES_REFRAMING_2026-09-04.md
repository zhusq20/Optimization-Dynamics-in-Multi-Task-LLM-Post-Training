# Reframing the paper around a foundational MOPD question

Research memo, 4 September 2026. Based on the current LaTeX draft, experiment artifacts, revised experiment plan, and primary literature. The mathematical statements below are local deductions under stated assumptions; proposed empirical claims have not been demonstrated in this project. This memo proposes a research direction and does not replace the manuscript or authorize new training expenditure.

**Recommendation:** organize the paper around **when multi-teacher distillation can make reliable joint progress**. Use GPAS as an intervention whose effectiveness and limits the paper explains. The current contribution—optimizer-aware Neyman allocation—is useful, but a foundational claim requires evidence about the mechanism limiting capability integration.

## 1. Start with the actual goal of MOPD

MOPD seeks one student that acquires useful capabilities from several teachers. Reducing a weighted average of teacher losses is an operational training objective; acquiring those capabilities is the outcome that matters.

Let C_i(theta) denote a capability score, and r_i a reference level established by separately distilling teacher i into the same student architecture. A concrete integration goal is to find one theta, within compute budget B, satisfying

\[
C_i(\theta)\ge r_i-\epsilon_i\quad\text{for every domain }i.
\]

The reference is empirical, neither an upper bound nor proof that all references are jointly attainable. Use raw scores alongside reference-normalized gains, and do not normalize by a near-zero or negative gain. Show performance against both per-domain exposure and total compute because specialist and joint training have different resource allocations.

This formulation separates three questions:

1. **Individual transferability:** can this student learn the capability from this teacher under the chosen loss and rollout protocol?
2. **Joint attainability:** can one shared model reach the desired capability vector? Local gradient conflict does not prove global infeasibility.
3. **Finite-compute learnability:** does the actual training process reach that vector efficiently and reliably?

The paper should concentrate on a tractable part of the third question, conditional on evidence for the first. It should investigate joint attainability only when the data make it necessary. A claim to solve all three would require a substantially different project.

## 2. The central question to put in the introduction

> Given capabilities that a student can acquire separately, when does a shared MOPD update improve their distillation objectives together, and how much computation is needed to make that improvement reliable?

This connects capability integration to the existing gradient and allocation machinery. It also permits informative negative findings. Simultaneous improvement of every local loss is a useful diagnostic and a sufficient condition for local joint progress; it is not necessary for successful long-run training. Temporary regressions can be part of an effective trajectory.

The distinction matters because several mechanisms can produce the same apparent plateau:

| Mechanism | What is missing | What more samples can do |
|---|---|---|
| Weak individual transfer | A useful mean update from a particular teacher | Estimate it more accurately; cannot create a missing signal |
| Interference under the chosen mixture | A mean direction helpful to the affected domain | Cannot change the expected first-order sign at fixed weights and optimizer scaling |
| Sampling uncertainty | Reliable estimation of an otherwise useful direction | Can improve reliability; this is GPAS's direct opportunity |
| Poor deterministic conditioning | Effective movement along useful directions | May have little effect once the mean is already accurate |
| Coverage or loss–capability mismatch | Supervision that improves the desired behavior | Cannot be diagnosed from aggregate gradient variance alone |

Teacher agreement on identical prefixes and compatibility of task gradients on different prompt distributions are different quantities. Domain routing chooses a target distribution; shared parameters still couple the learning problems.

## 3. Put task progress before total gradient variance

At a common checkpoint, freeze the rollout distribution and the pre-step positive diagonal scaling D. Let L_i^t be the task's frozen-distribution surrogate, mu_i its gradient at the checkpoint, and Sigma_i the covariance of an independent task-i micro-batch gradient. Counts are selected from history before sampling the step's data.

\[
A=\sum_jw_j\bar g_j,\qquad
\mu_w=\sum_jw_j\mu_j,\qquad
\Sigma(m)=\sum_j\frac{w_j^2}{m_j}\Sigma_j.
\]

For the idealized update theta' = theta - eta D A, define

\[
K_{ij}=\mu_i^\top D\mu_j,\qquad
a_i=(Kw)_i=\mu_i^\top D\mu_w.
\]

The expected first-order task-loss change is -eta a_i. K uses **one D**. Cosines between D mu_i and D mu_j involve D squared and do not directly measure the first-order effect of the update on the task loss.

Three locally distinct cases follow:

- Every a_i is comfortably positive: the chosen weighted mean is a common descent direction.
- Some a_i is positive but small: that domain has little margin against update noise or finite-step curvature.
- Some a_i is nonpositive: changing counts with fixed weights cannot make that domain's expected first-order progress strictly positive in this frozen-D model.

The last statement concerns this mixture at this checkpoint. It does not prove that another mixture fails, that no descent direction exists, or that the model lacks capacity. Nor is it an exact invariance claim for AdamW, whose current moments, momentum, clipping, and training history can change with allocation.

For twice-differentiable frozen losses, a local expansion makes the distinction explicit:

\[
\mathbb E[\Delta L_i^t]
=-\eta a_i+
\frac{\eta^2}{2}
\left[(D\mu_w)^\top H_i(D\mu_w)
+\operatorname{tr}(H_iD\Sigma(m)D)\right]
+\text{higher-order terms}.
\]

This is a local expansion, not an unconditional descent guarantee. Noise matters through its direction and curvature as well as its magnitude. The current GPAS smoothness bound upper-bounds this dependence using total update variance. It is a conservative surrogate for useful progress.

## 4. A finite-compute reliability statement

The uncertainty relevant to domain i is the variance along its own loss gradient:

\[
v_i(m)=\operatorname{Var}(\mu_i^\top D A)
=\sum_j\frac{w_j^2}{m_j}
\mu_i^\top D\Sigma_jD\mu_i.
\]

For a_i > 0, the one-sided Chebyshev–Cantelli inequality gives

\[
\Pr(\mu_i^\top DA\le0)
\le\frac{v_i(m)}{v_i(m)+a_i^2}.
\]

Summing these bounds over domains bounds the probability that at least one task has a non-descent first-order sign. This needs finite second moments and the covariance assumptions above, not Gaussian gradients. It can be loose. Plugging noisy gradient estimates into the formula does not automatically produce a statistically valid certificate.

Write m_j = G q_j for positive sampling fractions summing to one. Then

\[
\frac{v_i(m)}{a_i^2}
=\frac1G\sum_j
\frac{w_j^2\mu_i^\top D\Sigma_jD\mu_i}{q_j a_i^2}.
\]

This supplies a testable prediction: reliability depends on noise **relative to the task's progress margin**. Large raw variance can be harmless if the margin is large; moderate variance can matter near cancellation. With all a_i > 0, a sufficient common per-domain bound of delta/M is obtained when

\[
G\ge\frac{M-\delta}{\delta}
\max_i\sum_j
\frac{w_j^2\mu_i^\top D\Sigma_jD\mu_i}{q_j a_i^2}.
\]

This is a sufficient local budget condition. It ignores integer bounds, estimation error in the statistics, and optimizer/state evolution. Those must be handled before treating it as an operational guarantee. Finite-step descent also requires a step-size or trust-region condition controlling curvature.

These ingredients draw on established stochastic optimization ideas. Inner-product-based sampling tests predate this paper: [Bollapragada, Byrd, and Nocedal](https://arxiv.org/abs/1710.11258) explicitly distinguish directional tests from norm tests. Adaptive sampling for stochastic multiobjective optimization also has direct precedent: [Zhao, Chen, and Yang](https://link.springer.com/article/10.1007/s10957-023-02334-w). The claim to earn is a predictive, causally validated account of MOPD behavior, not invention of probability inequalities or common-descent sampling.

## 5. What happens to GPAS?

The current method minimizes

\[
V(m)=\operatorname{tr}(D\Sigma(m)D).
\]

Since v_i(m) <= ||mu_i||^2 V(m), GPAS controls a conservative bound on directional uncertainty. This provides a principled role in the broader paper. It does not make trace-optimal allocation optimal for every domain's reliability.

A simple analytical example shows the difference. Take two tasks, D = I, equal weights, identical means mu_1 = mu_2 = (1,0), and covariances

\[
\Sigma_1=\operatorname{diag}(1,80),\qquad
\Sigma_2=\operatorname{diag}(9,0).
\]

For G = 40 with no count caps, GPAS sees trace noise (81,9), so allocates (30,10). The directional variance for either task is

\[
v=\tfrac14(1/30+9/10)=7/30.
\]

Allocating (10,30) gives v = 1/10. Much of task 1's noise is orthogonal to both tasks' mean gradients. Under Gaussian micro-batch noise, the latter allocation also has a lower exact probability of a wrong first-order sign. This is an analytical example, not an LLM result, and its budget differs from the four-domain training plan.

There are two defensible research paths:

**Preferred first step:** keep GPAS, test whether its trace statistic is a useful proxy for the directional uncertainties that predict actual regressions, and report the regimes in which it succeeds or fails.

**Conditional extension:** if trace variance is demonstrably misaligned with task progress, investigate

\[
\min_{m:\sum_jm_j=G}\max_{i:a_i>0}\frac{v_i(m)}{a_i^2},
\]

with the existing count bounds. For known positive margins and covariances, the continuous problem is convex in positive m; with four tasks the feasible integer allocations can be enumerated. The normalized risk penalizes uncertainty that threatens a vulnerable task's useful progress. It is not a complete controller when any a_i <= 0: choosing a different mean direction or allowing an explicit task tradeoff is a separate decision. Do not silently exclude harmed domains and call the result joint preservation.

Avoid making a new scheduler the prerequisite for this research. First test whether there is a real MOPD problem that the stronger statistic resolves. Estimating all directional variances adds overhead, and sample splitting or historical estimates are needed to avoid optimistic reuse of the same noisy data.

The current cost criterion T(m)V(m) also needs a narrower interpretation: it measures time times noise, rather than expected learning progress per second in general. The deterministic signal, step size, and curvature also enter a progress calculation. Validate the criterion empirically and identify whether a noise-dominated regime explains its success. With the current four-domain counts, Uniform uses four micro-batches and no task can receive more than eight, giving a variance-improvement ceiling of two under the fixed-state model; this is not a twofold training-speed guarantee.

For exact implementation claims, distinguish independent sampling from the plan's sampling without replacement, and account for finite-population effects where relevant. Largest-remainder rounding is a practical approximation to the continuous per-step solution; it is not always the exact integer optimum. Enumeration is already feasible in the four-domain setting.

## 6. Make the account specific to on-policy distillation

At the next step the student changes the prefixes it visits. Therefore the moving-policy loss satisfies the exact decomposition

\[
L_i^{t+1}(\theta_{t+1})-L_i^t(\theta_t)
=\underbrace{L_i^t(\theta_{t+1})-L_i^t(\theta_t)}_{\text{optimization on fixed states}}
+\underbrace{L_i^{t+1}(\theta_{t+1})-L_i^t(\theta_{t+1})}_{\text{change in visited states}}.
\]

A downward moving-policy KL curve can contain both effects. It cannot, by itself, identify absorption of the earlier supervision. Maintain three separate measurements: fixed-reference-prefix loss, fresh-policy loss, and downstream capability. Reference banks should include disjoint evaluation examples and be refreshed through a prespecified protocol where needed.

For finite top-k, the surrogate is the specified retained-support loss, not unbiased full-vocabulary KL. Keep an existing dense loss as the primary training recipe and use the planned tail-aware pair to test robustness. Teacher source, thinking mode, decoding truncation, and response normalization must remain controlled or explicitly characterized.

This is the scientific link to MOPD: several teacher targets are pulled through shared student parameters while the student changes the state distribution supplying those targets. A generic variance-reduction demonstration alone would not establish that link.

## 7. The smallest decisive empirical program

The current directory contains synthetic studies and an experimental specification; it does not yet contain measured end-to-end MOPD results. The current nine-configuration plan addresses a practical allocation question. The additional diagnostics here are a deliberate expansion for the user's foundational objective.

**Stage A — Establish a phenomenon.** Run the standard Uniform joint recipe and task-specific OPD references, beginning with two well-verified teachers if four full reference runs are too costly. Compare raw capability, each domain's acquired gain, and learning curves against both task exposure and total compute. A strong teacher alone is insufficient evidence of transferability. A joint-versus-specialist gap alone is insufficient evidence of conflict.

**Stage B — Diagnose at common checkpoints.** Save the exact student and optimizer state at early, middle, and late training. From each checkpoint:

1. Estimate task means, transfer margins, trace noise, and directional noise using independent micro-batches; use held-out batches to assess predictions.
2. Clone the checkpoint, apply each task's update and the combined update separately, and evaluate the changes on all domains. Include actual AdamW branch updates so the frozen-D approximation is tested.
3. Compare Uniform, GPAS, and a much more precise weighted gradient. Treat the last as an expensive diagnostic reference and record its cost; it is not a cost-matched competitor.
4. Vary total batch size and allocation separately. A batch-size sweep changes update frequency under fixed compute, so use it for local identification before drawing end-to-end efficiency conclusions.

The central decision is whether reducing estimation noise improves useful descent. If a precise weighted mean still harms a task, variance allocation is not the remedy for its expected local interference. If the mean is useful but ordinary batches frequently reverse a weak task's progress, allocation has a diagnosed target.

**Stage C — Test prediction and intervention.** Freeze the diagnostic rule before examining later checkpoints or another teacher pair. Predict which domains regress, which configurations benefit from GPAS, and where additional samples have little effect. Repeat the decisive Uniform/intervention comparison across training seeds. A prompt bootstrap does not measure training-seed variability.

**Stage D — Confirm capabilities and cost.** Report every domain, worst-domain changes, and elapsed GPU hours. Preserve the planned raw-noise comparison, the closest scheduling recipe, and one dense-loss robustness pair as resources permit. A variance ratio is not a training-speed ratio. Statistics, idle GPUs, teacher scoring, and synchronization count toward the cost.

Conditional branches should follow the diagnosis. If accurate gradients remain slow, examine conditioning and update scaling. If fixed-state loss improves but capability does not, examine coverage and the surrogate. If a residual integration gap persists, a limited task-weight sweep, reduced parameter sharing, or larger student can test particular hypotheses; failure of one configuration does not prove an intrinsic capacity limit.

## 8. Existing literature determines what the paper must add

| Primary work | Relevant established scope | Implication for our claim |
|---|---|---|
| [MOPD](https://arxiv.org/html/2606.30406v1) | Domain-teacher routing and an existing corrected teacher top-k recipe | Treat dense top-k as a training component |
| [Open-MOPD](https://arxiv.org/html/2608.19098v1) | Integration-gap diagnosis, separate-distillation references, and imbalance/reward corrections | Attribute these controls; a joint-versus-separate gap alone is insufficient novelty |
| [D3-MOPD](https://arxiv.org/html/2608.24987v1) | Sampling based on remaining gap and descent velocity | Distinguish allocation signals under fixed weights from full recipe comparisons |
| [CaMOPD](https://arxiv.org/html/2605.27115v1) | Counteracting recovery/preservation updates and within-source weak signals | A gradient-conflict plot alone is insufficient novelty |
| [Rethinking OPD II](https://arxiv.org/html/2609.04172v1) | Slow alignment on fixed states and small diverse query sets matching full-data training in studied settings | Motivates testing optimization bottlenecks; does not establish noise as their cause |

The strongest contribution would be a diagnostic that predicts the right intervention before the intervention is run, including a regime where allocation cannot help. A complete novelty assessment of a new directional scheduler would require deeper comparison with the stochastic multiobjective optimization literature.

## 9. Rewrite the paper around the evidence chain

Suggested working title: **When Does Multi-Teacher On-Policy Distillation Make Joint Progress?**

| Current component | Revision |
|---|---|
| Abstract opens with allocation | Open with the capability-integration question; add the measured mechanism and effect only after results exist |
| Introduction motivates heterogeneous noise | Start with independently transferable gains, observed joint behavior, and competing explanations |
| Preliminary defines the loss and counts | Also distinguish capability targets, frozen-state losses, and fresh-policy losses |
| Theory begins with update variance | Begin with task progress margins; derive directional uncertainty; place GPAS within that account |
| First figure is a method schematic | Use measured individual/joint gains and diagnostic intervention outcomes when available |
| Results begin with a scheduler leaderboard | Begin with the phenomenon, mechanism identification, prediction, then intervention and compute |
| Discussion offers generic limitations | State where the explanation works, where it fails, and which unresolved bottlenecks remain |

The intended contribution sequence is: **define the phenomenon → identify the mechanism → predict its regime → intervene → validate capability and cost**.

Suggested introductory problem statement, with no invented findings:

> Multi-teacher on-policy distillation aims to combine capabilities that a student could otherwise acquire from separate teachers. Yet successful transfer from each teacher does not determine whether their updates can be combined effectively in shared parameters. A task may learn slowly because its expected update is weak, because other tasks oppose that update, or because finite sampling obscures a useful direction. These explanations call for different interventions. We study when a shared distillation update can improve task objectives together and how computation controls the reliability of that progress, while separately measuring the effect of changing student-generated states and downstream capability.

Three claims to earn, rather than prewrite as results:

1. A measured distinction between harmful mean interactions and finite-sample failures predicts observed MOPD behavior.
2. The proposed statistics identify when allocation improves useful joint progress, including settings where trace-noise allocation is inadequate or unnecessary.
3. A targeted intervention improves actual capability integration or compute efficiency across repeated runs, with explicit limits.

The recommended next research action is a standard dense-loss pilot with common-checkpoint transfer and precision interventions. Its result should decide whether the main paper remains centered on GPAS, develops directional allocation, or turns to a different diagnosed bottleneck.
