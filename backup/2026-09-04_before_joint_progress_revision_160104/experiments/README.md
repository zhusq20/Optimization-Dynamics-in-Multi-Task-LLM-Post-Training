# GPAS experiment artifacts

The manuscript studies micro-batch allocation on existing dense distillation
losses. The default is teacher top-64 corrected reverse KL with equal task
weights. GPAS adapts counts using smoothed gradient variance in the optimizer's
coordinates. Its cost-aware mode also uses measured task time.

The current experiment specification is
[the four-domain plan](../EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md).
It specifies nine training configurations, common capability evaluation,
learning curves against steps and GPU hours, and allocation traces collected
during training. The end-to-end MOPD trainer and its measured results are not
present in this directory yet.

## Paper figure

    python3 experiments/generate_gpas_main_figure.py

This regenerates the editable PDF, SVG, and PNG overview in figures/.
It is a schematic, not an empirical result.

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

These existing artifacts are preserved for reference; the current training
plan does not require synthetic or token-estimator validation runs before
the MOPD comparisons.
