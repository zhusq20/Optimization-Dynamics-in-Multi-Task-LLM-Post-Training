# MOPD joint-progress experiment artifacts

The current specification is
[the four-domain plan](../EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md).
It follows the manuscript's questions about own-task signal strength,
combined progress, sampling uncertainty, and actual loss/capability changes,
with the user's current execution decisions taking precedence over older
manuscript settings.

## Current experiment decisions

- **One training seed for every configuration:** nine joint runs and four
  single-task references, for **13 base runs**. Uniform and GPAS use the
  same seed; there are no additional training-seed replications.
- **Temporary teachers:** code and science currently share Qwen3-4B as a
  temporary substitute for their respective Qwen3-1.7B RL teachers. The
  intended teacher setup remains domain-specific Qwen3-1.7B RL teachers.
  Record the actual teacher revision and temporary-substitute status in
  every run and result table.
- **No defensive or gate experiments:** single-task references are capability
  comparisons and run alongside joint training. Teacher gaps, reference
  scores, and early diagnostics do not determine whether training proceeds.
  No extra pilot qualification or conditional directional-scheduler study
  is included.

The default loss is teacher top-64 corrected reverse KL with equal task
weights. Fixed-reference loss, fresh-policy loss, and capability are separate
measurements. Existing top-64 outputs also provide retained masses and
centered masked-logratio variance as descriptive teacher-signal statistics.
Single-task update 125 matches Uniform's final 8,000-response domain exposure.

The one Uniform run supplies diagnostic checkpoints at updates 50, 250,
and 450. Each checkpoint has seven main branches with 20 one-step update
draws per branch. The equal-count sweep reuses Uniform and Precise for
G=16 and G=128, adding only G=8, 32, and 64. These local draws are not
additional training seeds. Calibration, update, and evaluation samples have
separate roles; branches share a fixed checkpoint evaluation bank.

Base training uses **416,000 planned responses**. The three diagnostic
checkpoints add **81,792 generated responses** and **307,200 response-level
student scoring forwards** for updated branches. Longitudinal loss banks,
benchmarks, baseline scoring, gradient measurements, and retries are counted
separately. The earlier per-run timing assumption gives **208–260 GPU hours
for base training only**; actual device occupancy determines reported cost.

The end-to-end MOPD trainer, diagnostic implementation, and measured results
are not present in this directory yet. The plan specifies implementation,
analysis, and output records; existing synthetic artifacts are not MOPD
results. Manuscript passages that still describe three training seeds or a
permanent specialist/generalist mixture await synchronization with these
current decisions.

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

## Existing initial log-ratio utility (not scheduled)

The current protocol does not schedule this utility as an experiment or gate.
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
