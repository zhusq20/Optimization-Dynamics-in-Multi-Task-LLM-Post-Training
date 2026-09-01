"""Controlled check of GPAS and Cost-GPAS.

This local-gradient sanity check compares task proposals for estimating one
fixed, optimizer-scaled multi-task update. End-to-end LLM runs remain the main
test of training efficiency.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


SEED = 20260826
TASKS = 3
DIM = 96
BATCH_SIZE = 16
TRIALS = 20_000


def make_problem():
    rng = np.random.default_rng(SEED)
    weights = np.ones(TASKS) / TASKS

    # Three parameter blocks have very different lagged Adam scales.
    preconditioner = np.r_[np.full(32, 0.005),
                           np.full(32, 0.05),
                           np.full(32, 3.0)]
    means = rng.normal(0.0, 0.03, size=(TASKS, DIM))
    stds = np.full((TASKS, DIM), 0.005)
    stds[0, :32] = 2.0
    stds[1, 32:64] = 0.5
    stds[2, 64:] = 0.1
    costs = np.array([5.0, 2.0, 10.0])
    return weights, preconditioner, means, stds, costs


def proposals(weights, preconditioner, means, stds, costs):
    raw_scale = np.sqrt(np.sum(means**2 + stds**2, axis=1))
    update_scale = np.sqrt(
        np.sum(preconditioner[None, :] ** 2 * (means**2 + stds**2), axis=1)
    )
    uniform = np.ones(TASKS) / TASKS
    raw_gpas = weights * raw_scale / np.sum(weights * raw_scale)
    gpas = weights * update_scale / np.sum(weights * update_scale)
    cost_gpas_score = weights * update_scale / np.sqrt(costs)
    cost_gpas = cost_gpas_score / np.sum(cost_gpas_score)
    return raw_scale, update_scale, {
        "Uniform": uniform,
        "Raw-gradient GPAS": raw_gpas,
        "GPAS": gpas,
        "Cost-GPAS": cost_gpas,
    }


def theoretical_mse(proposal, weights, preconditioner, means, stds):
    target = np.sum(weights[:, None] * means, axis=0)
    second_moments = np.sum(
        preconditioner[None, :] ** 2 * (means**2 + stds**2), axis=1
    )
    single_sample_variance = (
        np.sum(weights**2 * second_moments / proposal)
        - np.sum((preconditioner * target) ** 2)
    )
    return single_sample_variance / BATCH_SIZE


def empirical_mse(proposal, weights, preconditioner, means, stds, seed):
    rng = np.random.default_rng(seed)
    target = np.sum(weights[:, None] * means, axis=0)
    estimates = np.zeros((TRIALS, DIM))
    for _ in range(BATCH_SIZE):
        task = rng.choice(TASKS, size=TRIALS, p=proposal)
        gradient = means[task] + rng.normal(size=(TRIALS, DIM)) * stds[task]
        estimates += (weights[task] / proposal[task])[:, None] * gradient
    estimates /= BATCH_SIZE
    squared_error = np.sum(
        ((estimates - target[None, :]) * preconditioner[None, :]) ** 2,
        axis=1,
    )
    scaled_target = preconditioner * target
    scaled_mean_error = np.linalg.norm(
        preconditioner * (estimates.mean(axis=0) - target)
    ) / np.linalg.norm(scaled_target)
    return (
        squared_error.mean(),
        squared_error.std(ddof=1) / np.sqrt(TRIALS),
        scaled_mean_error,
    )


def compute_criterion(proposal, weights, preconditioner, means, stds, costs):
    second_moments = np.sum(
        preconditioner[None, :] ** 2 * (means**2 + stds**2), axis=1
    )
    expected_cost = np.sum(proposal * costs)
    corrected_second_moment = np.sum(
        weights**2 * second_moments / proposal
    )
    return expected_cost * corrected_second_moment


def main():
    weights, preconditioner, means, stds, costs = make_problem()
    _, _, proposal_map = proposals(
        weights, preconditioner, means, stds, costs
    )

    theory = {}
    empirical = {}
    errors = {}
    mean_errors = {}
    compute = {}
    for index, (name, proposal) in enumerate(proposal_map.items()):
        theory[name] = theoretical_mse(
            proposal, weights, preconditioner, means, stds
        )
        empirical[name], errors[name], mean_errors[name] = empirical_mse(
            proposal, weights, preconditioner, means, stds, SEED + index + 1
        )
        compute[name] = compute_criterion(
            proposal, weights, preconditioner, means, stds, costs
        )

    theory_baseline = theory["Uniform"]
    empirical_baseline = empirical["Uniform"]
    compute_baseline = compute["Uniform"]
    names = list(proposal_map)
    colors = ["#7f8c8d", "#d95f5f", "#2a9d8f", "#3f72af"]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.8))
    x = np.arange(TASKS)
    width = 0.19
    for index, name in enumerate(names):
        axes[0].bar(
            x + (index - 1.5) * width,
            proposal_map[name],
            width,
            label=name,
            color=colors[index],
        )
    axes[0].set_xticks(x, [f"task {i + 1}" for i in x])
    axes[0].set_ylabel("sampling probability")
    axes[0].set_title("Task proposals")
    axes[0].legend(frameon=False, fontsize=7)

    empirical_ratio = np.array(
        [empirical[name] / empirical_baseline for name in names]
    )
    error_ratio = np.array([errors[name] / empirical_baseline for name in names])
    theory_ratio = np.array([theory[name] / theory_baseline for name in names])
    axes[1].bar(
        np.arange(len(names)), empirical_ratio, color=colors,
        yerr=1.96 * error_ratio, capsize=3,
    )
    axes[1].scatter(
        np.arange(len(names)), theory_ratio, color="black", marker="x",
        label="theory", zorder=3,
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    short_names = ["Uniform", "Raw", "GPAS", "Cost"]
    axes[1].set_xticks(np.arange(len(names)), short_names)
    axes[1].set_ylabel("scaled gradient MSE\n(relative to uniform)")
    axes[1].set_title(f"{BATCH_SIZE}-sample estimator")
    axes[1].legend(frameon=False, fontsize=7)

    compute_ratio = np.array([compute[name] / compute_baseline for name in names])
    axes[2].bar(np.arange(len(names)), compute_ratio, color=colors)
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(np.arange(len(names)), short_names)
    axes[2].set_ylabel("second moment $\\times$ cost\n(relative to uniform)")
    axes[2].set_title("Compute criterion")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()

    project_root = Path(__file__).resolve().parents[1]
    figure_output = project_root / "figures" / "controlled_optimizer_sampling.pdf"
    result_output = Path(__file__).with_name(
        "controlled_optimizer_sampling_results.csv"
    )
    fig.savefig(figure_output, bbox_inches="tight")
    with result_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method",
            "prob_task_1",
            "prob_task_2",
            "prob_task_3",
            "theory_mse_ratio",
            "empirical_mse_ratio",
            "relative_mean_error",
            "compute_proxy_ratio",
        ])
        for name in names:
            writer.writerow([
                name,
                *proposal_map[name],
                theory[name] / theory_baseline,
                empirical[name] / empirical_baseline,
                mean_errors[name],
                compute[name] / compute_baseline,
            ])
    print(f"wrote {figure_output}")
    print(f"wrote {result_output}")
    for name in names:
        probs = np.array2string(proposal_map[name], precision=3)
        print(
            f"{name:21s} probabilities={probs} "
            f"theory_ratio={theory[name] / theory_baseline:.3f} "
            f"empirical_ratio={empirical[name] / empirical_baseline:.3f} "
            f"mean_error={mean_errors[name]:.3f} "
            f"compute_ratio={compute[name] / compute_baseline:.3f}"
        )


if __name__ == "__main__":
    main()
