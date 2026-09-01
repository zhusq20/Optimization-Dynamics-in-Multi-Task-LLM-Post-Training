# Controlled experiments

Run every controlled check from the repository root:

```bash
python3 experiments/generate_controlled_figure.py
```

The paper uses these controlled-check outputs:

- `experiments/controlled_optimizer_sampling_results.csv`
- `experiments/controlled_adamw_moment_results.csv`
- `figures/controlled_optimizer_sampling.pdf`

## Three-task construction

The first three panels use a complete synthetic construction defined in the
script. There are three zero-mean Gaussian tasks and three coordinates. Task
`i` has support on coordinate `i` only. The comparison uses the same corrected
task draws for every proposal.

The raw task scales, normalized to sum to one, are
`[0.765259, 0.192307, 0.042434]`. The fixed diagonal AdamW map is the AdamW
scale divided by the raw scale. Its transformed task scales, also normalized,
are `[0.174055, 0.196902, 0.629043]`. These choices give the raw-scale and
GPAS proposals directly. The Cost-GPAS proposal is
`[0.187118, 0.334697, 0.478185]`; task cost `i` is defined as the square of the
AdamW scale divided by that proposal. This makes the Cost-GPAS allocation an
exact consequence of the stated geometry and costs.

The estimator uses target weights `1/3`. The reported raw and AdamW-metric MSE
ratios divide each method's error by the Uniform error. The time-weighted ratio
multiplies AdamW-metric MSE by expected task cost and applies the same Uniform
normalization. Black crosses in the figure show calculated values; bars show
Monte Carlo values.

## AdamW moment check

For an empirical or calculated moment vector `x` and the target-mixture moment
`x_ref`, relative bias means

```text
||x - x_ref||_2 / ||x_ref||_2.
```

The usual AdamW second-moment update squares the full importance-weighted
gradient. The moment-consistent version applies one importance weight to the
squared gradient. Both versions use the same corrected first moment.

The construction and moment comparison only check the sampler calculation.
Evidence about optimization and model quality comes from the MOPD runs.
