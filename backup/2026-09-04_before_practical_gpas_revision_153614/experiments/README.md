# GPAS experiment artifacts

The controlled checks validate the all-task micro-batch allocation and
Adam-type moment calculations used in the paper. Run them from the repository
root:

```bash
python3 experiments/generate_controlled_figure.py
```

The command regenerates:

- `experiments/controlled_optimizer_sampling_results.csv`
- `experiments/controlled_optimizer_moment_results.csv`
- `figures/controlled_optimizer_sampling.pdf`

The allocation check uses 20,000 trials, 16 micro-batches per step, and at least
two micro-batches from every task. The separate moment check uses 200,000
simulated steps. The seeds, noise geometry, preconditioner geometry, task costs, count
bounds, and rounding rule are defined in `generate_controlled_figure.py`.

Current empirical ratios relative to Uniform are:

| Method or observation | Ratio / change |
|---|---:|
| GPAS (per step), preconditioned gradient variance | 0.679 |
| GPAS (per GPU hour), predicted time times variance | 0.857 |
| Counts from unpreconditioned noise, preconditioned variance | 2.316 |
| Conventional AdamW second-moment change after reallocation | 0.543 |
| Taskwise AdamW $U/G$ change after reallocation | 0.00092 |

These controlled artifacts check the calculations and implementation. The
four-teacher protocol supplies the end-to-end MOPD efficiency study and is in
`EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md`.

`experiments/measure_initial_kl.py` measures the initial per-task teacher loss
$\ell_i(0)$ on the held-out prompts and prints the inverse-initial-loss weights and the
max/min ratio that decides between inverse-loss and equal weights (plan section 2).

The `random_geometry_stress_*` files are retained as artifacts of the earlier
task-subsampling formulation and are not evidence for the current all-task
micro-batch method.
