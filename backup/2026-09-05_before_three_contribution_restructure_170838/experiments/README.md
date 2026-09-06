# GPAS four-domain experiments: two-week core plan

The current specification is
[the two-week core plan](../EXPERIMENT_PLAN_QWEN3_1.7B_4T_MOPD_GPAS_zh.md).
It replaces the earlier broad experiment program with **four single-seed
training runs and one small common-checkpoint comparison**.

## Active runs

| Run | Allocation | Purpose |
|---|---|---|
| `uniform-s1` | Four micro-batches per domain | Main baseline |
| `gpas-s1` | Preconditioned gradient-noise allocation | Main method |
| `gpas-raw-s1` | Raw gradient-noise allocation; AdamW unchanged | Preconditioning ablation |
| `d3-fixed-s1` | Remaining gap × descent velocity; fixed task weights | Existing scheduling-signal comparison |

All runs share teacher top-64 corrected reverse KL, equal task weights,
G=16 micro-batches of four responses, counts in [2,8], 500 updates, and the
same training seed. Code and science currently share Qwen3-4B **temporarily
in place of their respective Qwen3-1.7B RL teachers**. Record actual teacher
revisions and temporary-substitute status in each run and result table.

Single-task reference runs, cost-aware GPAS, full recipe comparisons,
TA-OPD, precise-gradient branches, batch-size sweeps, additional seeds,
and directional-covariance/prediction studies are outside this two-week
scope. There are no teacher qualification, defensive, or gate experiments.

## Evaluation and mechanism evidence

- Full capability evaluation: the initial student once, Uniform/GPAS at
  updates 250 and 500, and the two comparison runs at update 500. This is
  seven student-checkpoint evaluation sets. Assigned teachers receive one
  corresponding benchmark evaluation per domain.
- Fixed-reference loss: 64 prompts per domain at updates 0, 100, 200, 300,
  400, and 500. The initial reference bank is shared. Fresh-policy loss is
  measured only initially and at the four final checkpoints.
- One common checkpoint: Uniform update 250, comparing Uniform and GPAS
  with ten independent one-step update draws each. Calibration uses 16
  micro-batches per domain; evaluation uses 64 prompts per domain. Counts
  are selected before trial draws. Measure empirical preconditioned update
  variance and actual fixed-bank loss changes; no evaluation gradients,
  K matrix, directional covariance, or precise-gradient branch is required.

The Uniform and D3 baselines do not incur GPAS's per-micro-batch noise
collection overhead during training. GPAS and its raw-noise ablation pay
for their own statistics. Common-checkpoint variance is measured with the
same frozen pre-step scaling for both diagnostic branches.

## Budget and delivery

Four full runs generate **128,000 planned responses**. The local comparison
adds **1,792 generated responses** and **5,120 response-level student scoring
forwards** for updated branches. Longitudinal reference/final-fresh banks
add **1,280 generated responses**, giving **131,072 planned responses before
capability benchmarks and retries**. Cached-bank scoring is counted separately.
The prior, unverified 8–10 hours per run assumption implies **64–80 GPU hours
for base training only** on the two-device layout.

The 14-day schedule prioritizes main training, the two comparison runs,
one local comparison, evaluation, and figures, with the final two days
reserved for missing outputs and packaging. Deliver one capability/cost
table, core learning curves, a local mechanism figure, and source records.
Do not add optional runs to fill the remaining time.

The end-to-end trainer and measured MOPD results are not present in this
directory yet. Manuscript commitments to the earlier broad study must be
narrowed when results are written up; this turn changes the plan and README.

## Existing analytical and historical artifacts

- `generate_joint_progress_teaser.py` generates the analytic first figure.
- `generate_gpas_main_figure.py` generates the GPAS schematic.
- `generate_controlled_figure.py`, `controlled_optimizer_*_results.csv`,
  and `random_geometry_stress_*` retain earlier synthetic calculations.
- `measure_initial_kl.py` is an existing sampled-log-ratio utility and is
  not scheduled as an experiment or gate in this plan.

These existing artifacts are not empirical four-domain results. Earlier
versions of the full experiment plan are preserved under backup/.
