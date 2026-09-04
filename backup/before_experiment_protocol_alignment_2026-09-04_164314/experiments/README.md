# MOPD joint-progress experiment artifacts

The manuscript distinguishes weak own-task mean signals, reductions after
combining tasks, reliable but slow local progress, and sampling failures in
MOPD. The protocol compares measured loss decreases as well as their signs. GPAS is a
trace-variance allocation intervention within that investigation. The default
training loss remains teacher top-64 corrected reverse KL with equal task
weights.

The current specification is
[the four-domain plan](../EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md).
It requires four single-task OPD references, common-checkpoint update and
precision interventions, prediction tests on held-out checkpoints and seeds,
and end-to-end capability and cost comparisons. Uniform and GPAS (per step)
use three seeds each; seven other configurations use one exploratory seed,
for 13 joint runs plus four initial single-task reference runs. Fixed-reference
loss, fresh-policy loss, and capability are separate measurements. Existing
top-64 outputs also provide retained masses and centered masked-logratio
conditional variance as descriptive teacher-signal statistics; these do not
measure gradient covariance or certify useful teacher signals. This addition
does not require full-vocabulary teacher output or another training run.

The end-to-end MOPD trainer, diagnostic implementation, and their measured
results are not present in this directory yet. Existing synthetic data do not
supply evidence for the proposed MOPD findings. Paper figures and tables that
require these measurements remain explicit placeholders.

## Analytical first figure

    python3 experiments/generate_joint_progress_teaser.py

This generates the analytical teaser in figures/. The curves e^{-t} and
e^{-0.1t} show that zero conflict does not determine the rate of progress;
a Gaussian example separates mean progress from sampling reliability.
These specified examples demonstrate possibilities, not measured MOPD rates,
curvature, or convergence. Empirical individual/joint capability and checkpoint
intervention figures still await data.

## Appendix GPAS schematic

    python3 experiments/generate_gpas_main_figure.py

This regenerates the GPAS method schematic as editable PDF, SVG, and PNG files
in figures/. It appears in the appendix and is not an empirical result.

## Optional initial log-ratio diagnostic

measure_initial_kl.py generates initial-student responses and scores their
sampled tokens under the student and teachers. It reports the response-mean
sampled log ratio, lengths, and truncation rates. This is not the teacher
top-64 training loss and does not set the objective weights or determine
whether an experiment proceeds. Task weights remain equal. The script requires
CUDA, vLLM, Transformers, and the model and prompt files supplied by the caller.

## Earlier synthetic artifacts

generate_controlled_figure.py and controlled_optimizer_*_results.csv
retain the earlier synthetic allocation and second-moment calculations.
They are not used as MOPD results in the revised manuscript.
random_geometry_stress_* belongs to the older task-subsampling formulation.

These artifacts are preserved for reference. The revised protocol adds
single-task references and independent common-checkpoint diagnostics; it does
not infer MOPD mechanisms from the synthetic calculations.
