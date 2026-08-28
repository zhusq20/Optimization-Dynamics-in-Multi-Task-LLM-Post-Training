"""Controlled check of optimizer-aware task importance sampling.

This is a local-gradient sanity check, not an LLM training result. It compares
three task proposals for estimating the same preconditioned multi-task update.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    return weights, preconditioner, means, stds


def proposals(weights, preconditioner, means, stds):
    raw_scale = np.sqrt(np.sum(means**2 + stds**2, axis=1))
    update_scale = np.sqrt(
        np.sum(preconditioner[None, :] ** 2 * (means**2 + stds**2), axis=1)
    )
    uniform = np.ones(TASKS) / TASKS
    raw_gpas = weights * raw_scale / np.sum(weights * raw_scale)
    optimizer_gpas = weights * update_scale / np.sum(weights * update_scale)
    return raw_scale, update_scale, {
        "Uniform": uniform,
        "Raw GPAS": raw_gpas,
        "Adam GPAS": optimizer_gpas,
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
    return squared_error.mean(), squared_error.std(ddof=1) / np.sqrt(TRIALS)


def main():
    weights, preconditioner, means, stds = make_problem()
    _, _, proposal_map = proposals(weights, preconditioner, means, stds)

    theory = {}
    empirical = {}
    errors = {}
    for index, (name, proposal) in enumerate(proposal_map.items()):
        theory[name] = theoretical_mse(
            proposal, weights, preconditioner, means, stds
        )
        empirical[name], errors[name] = empirical_mse(
            proposal, weights, preconditioner, means, stds, SEED + index + 1
        )

    baseline = theory["Uniform"]
    names = list(proposal_map)
    colors = ["#7f8c8d", "#d95f5f", "#2a9d8f"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(TASKS)
    width = 0.24
    for index, name in enumerate(names):
        axes[0].bar(
            x + (index - 1) * width,
            proposal_map[name],
            width,
            label=name,
            color=colors[index],
        )
    axes[0].set_xticks(x, [f"task {i + 1}" for i in x])
    axes[0].set_ylabel("sampling probability")
    axes[0].set_title("Task proposals")
    axes[0].legend(frameon=False, fontsize=7)

    empirical_ratio = np.array([empirical[name] / baseline for name in names])
    error_ratio = np.array([errors[name] / baseline for name in names])
    theory_ratio = np.array([theory[name] / baseline for name in names])
    axes[1].bar(
        np.arange(len(names)), empirical_ratio, color=colors,
        yerr=1.96 * error_ratio, capsize=3,
    )
    axes[1].scatter(
        np.arange(len(names)), theory_ratio, color="black", marker="x",
        label="theory", zorder=3,
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(np.arange(len(names)), ["Uniform", "Raw", "Opt.-aware"])
    axes[1].set_ylabel("preconditioned update MSE\n(relative to uniform)")
    axes[1].set_title(f"{BATCH_SIZE}-sample estimator")
    axes[1].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()

    output = Path(__file__).resolve().parents[1] / "figures" / "controlled_optimizer_sampling.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")
    for name in names:
        probs = np.array2string(proposal_map[name], precision=3)
        print(
            f"{name:21s} probabilities={probs} "
            f"theory_ratio={theory[name] / baseline:.3f} "
            f"empirical_ratio={empirical[name] / baseline:.3f}"
        )


if __name__ == "__main__":
    main()
