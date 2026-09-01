# GPAS experiment artifacts

The controlled checks validate the estimator and AdamW-moment calculations used
in the paper. Run them from the repository root:

```bash
python3 experiments/generate_controlled_figure.py
```

The command regenerates:

- `experiments/controlled_optimizer_sampling_results.csv`
- `experiments/controlled_adamw_moment_results.csv`
- `experiments/random_geometry_stress_scan.csv`
- `experiments/random_geometry_stress_summary.csv`
- `figures/controlled_optimizer_sampling.pdf`
- `figures/random_geometry_stress_scan.pdf`

The main controlled construction uses 20,000 trials with 16 corrected task
draws per trial. A separate AdamW check uses 1,000,000 draws. Seeds, task
geometries, optimizer scaling, and task costs are defined in
`generate_controlled_figure.py`.

Current empirical ratios relative to Uniform are:

| Method or moment | Ratio / error |
|---|---:|
| GPAS, AdamW-metric estimator error | 0.717 |
| Cost-GPAS, time-weighted error | 0.889 |
| Raw gradient-norm sampling, AdamW-metric error | 6.868 |
| Conventional AdamW second-moment relative error | 0.816 |
| Taskwise AdamW second-moment Monte Carlo error | 0.00235 |

These controlled artifacts support the calculation and implementation checks.
The large-model protocol and frozen-checkpoint banks are specified in
`EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md`.
