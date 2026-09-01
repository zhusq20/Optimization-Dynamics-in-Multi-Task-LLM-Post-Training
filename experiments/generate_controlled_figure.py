"""Reproduce the controlled GPAS checks from explicit synthetic geometries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLING_RESULTS = ROOT / "experiments" / "controlled_optimizer_sampling_results.csv"
MOMENT_RESULTS = ROOT / "experiments" / "controlled_adamw_moment_results.csv"
STRESS_RESULTS = ROOT / "experiments" / "random_geometry_stress_scan.csv"
STRESS_SUMMARY = ROOT / "experiments" / "random_geometry_stress_summary.csv"
CONTROLLED_FIGURE = ROOT / "figures" / "controlled_optimizer_sampling.pdf"
STRESS_FIGURE = ROOT / "figures" / "random_geometry_stress_scan.pdf"

# The controlled estimator uses 20,000 independent trials. Each trial averages
# 16 importance-weighted task draws, matching a small task-batch budget.
CONTROLLED_SEED = 20260901
CONTROLLED_TRIALS = 20_000
DRAWS_PER_TRIAL = 16

# The moment check retains its larger Monte Carlo budget.
MOMENT_SEED = 20260901
MOMENT_DRAWS = 1_000_000
MOMENT_CHUNK_SIZE = 20_000

# The stress scan samples every geometry from a fresh deterministic RNG stream.
STRESS_SEED = 20260902
STRESS_GEOMETRIES = 10_000
STRESS_TASKS = 4
STRESS_DIMENSION = 16
LOG_SCALE_LOW = -4.0
LOG_SCALE_HIGH = 4.0

# Three zero-mean Gaussian tasks occupy three separate coordinates. If task i is
# selected, its gradient is raw_scale[i] * Normal(0, 1) * e_i. The arrays sum
# to one, so they also give the raw-scale and AdamW-scale proposals.
RAW_SCALES = np.array(
    [0.7652593561149718, 0.19230656846142996, 0.04243407542359844]
)
ADAMW_SCALES = np.array(
    [0.1740545215177264, 0.1969024054681955, 0.6290430730140781]
)
TARGET_WEIGHTS = np.full(3, 1.0 / 3.0)
UNIFORM_PROPOSAL = np.full(3, 1.0 / 3.0)
COST_PROPOSAL = np.array(
    [0.18711814021008133, 0.3346968374362811, 0.4781850223536376]
)

# Applying this fixed diagonal map to a raw gradient changes its task scale
# from RAW_SCALES to ADAMW_SCALES. The chosen costs make COST_PROPOSAL exactly
# proportional to ADAMW_SCALES / sqrt(TASK_COSTS).
METRIC_DIAGONAL = ADAMW_SCALES / RAW_SCALES
TASK_COSTS = (ADAMW_SCALES / COST_PROPOSAL) ** 2


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a nonempty list of dictionaries with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized(values: np.ndarray) -> np.ndarray:
    """Return a probability vector proportional to positive values."""
    return values / np.sum(values)


def theoretical_mse(
    task_scales: np.ndarray, proposal: np.ndarray, draws: int
) -> float:
    """MSE of the zero-mean, importance-weighted trial average."""
    return float(np.sum((TARGET_WEIGHTS * task_scales) ** 2 / proposal) / draws)


def controlled_sampling_experiment() -> list[dict[str, Any]]:
    """Run the three-task estimator study and return one row per proposal."""
    raw_proposal = normalized(TARGET_WEIGHTS * RAW_SCALES)
    adamw_proposal = normalized(TARGET_WEIGHTS * ADAMW_SCALES)
    recovered_cost_proposal = normalized(
        TARGET_WEIGHTS * ADAMW_SCALES / np.sqrt(TASK_COSTS)
    )
    np.testing.assert_allclose(raw_proposal, RAW_SCALES, atol=1e-15)
    np.testing.assert_allclose(adamw_proposal, ADAMW_SCALES, atol=1e-15)
    np.testing.assert_allclose(recovered_cost_proposal, COST_PROPOSAL, atol=1e-15)
    np.testing.assert_allclose(METRIC_DIAGONAL * RAW_SCALES, ADAMW_SCALES)

    proposals = [
        ("Uniform", UNIFORM_PROPOSAL),
        ("Gradient-Norm IS", raw_proposal),
        ("GPAS", adamw_proposal),
        ("Cost-GPAS", recovered_cost_proposal),
    ]

    # Common random numbers reduce comparison noise while preserving the
    # marginal sampling distribution of every proposal.
    rng = np.random.default_rng(CONTROLLED_SEED)
    task_uniforms = rng.random((CONTROLLED_TRIALS, DRAWS_PER_TRIAL))
    gaussian_draws = rng.standard_normal((CONTROLLED_TRIALS, DRAWS_PER_TRIAL))

    raw_theory = []
    adamw_theory = []
    raw_empirical = []
    adamw_empirical = []
    mean_residual = []

    for _, proposal in proposals:
        sampled_tasks = np.searchsorted(
            np.cumsum(proposal), task_uniforms, side="right"
        )
        estimates = np.zeros((CONTROLLED_TRIALS, 3))
        for task_index in range(3):
            selected = sampled_tasks == task_index
            contribution = (
                selected
                * gaussian_draws
                * RAW_SCALES[task_index]
                * TARGET_WEIGHTS[task_index]
                / proposal[task_index]
            )
            estimates[:, task_index] = np.sum(contribution, axis=1) / DRAWS_PER_TRIAL

        raw_squared_error = np.sum(estimates**2, axis=1)
        adamw_squared_error = np.sum(
            (estimates * METRIC_DIAGONAL[None, :]) ** 2, axis=1
        )
        raw_theory.append(theoretical_mse(RAW_SCALES, proposal, DRAWS_PER_TRIAL))
        adamw_theory.append(
            theoretical_mse(ADAMW_SCALES, proposal, DRAWS_PER_TRIAL)
        )
        raw_empirical.append(float(np.mean(raw_squared_error)))
        adamw_empirical.append(float(np.mean(adamw_squared_error)))
        mean_residual.append(float(np.linalg.norm(np.mean(estimates, axis=0))))

    raw_theory = np.asarray(raw_theory) / raw_theory[0]
    adamw_theory = np.asarray(adamw_theory) / adamw_theory[0]
    raw_empirical = np.asarray(raw_empirical) / raw_empirical[0]
    adamw_empirical = np.asarray(adamw_empirical) / adamw_empirical[0]
    expected_cost = np.array([proposal @ TASK_COSTS for _, proposal in proposals])
    expected_cost /= expected_cost[0]
    cost_theory = adamw_theory * expected_cost
    cost_empirical = adamw_empirical * expected_cost

    rows: list[dict[str, Any]] = []
    for index, (method, proposal) in enumerate(proposals):
        row: dict[str, Any] = {
            "method": method,
            "prob_task_1": float(proposal[0]),
            "prob_task_2": float(proposal[1]),
            "prob_task_3": float(proposal[2]),
            "raw_theory_mse_ratio": float(raw_theory[index]),
            "raw_empirical_mse_ratio": float(raw_empirical[index]),
            "adamw_theory_mse_ratio": float(adamw_theory[index]),
            "adamw_empirical_mse_ratio": float(adamw_empirical[index]),
            "expected_cost_ratio": float(expected_cost[index]),
            "cost_theory_ratio": float(cost_theory[index]),
            "cost_empirical_ratio": float(cost_empirical[index]),
            "estimator_mean_residual": mean_residual[index],
            "seed": CONTROLLED_SEED,
            "trials": CONTROLLED_TRIALS,
            "draws_per_trial": DRAWS_PER_TRIAL,
        }
        for task_index in range(3):
            suffix = task_index + 1
            row[f"raw_scale_task_{suffix}"] = float(RAW_SCALES[task_index])
            row[f"adamw_scale_task_{suffix}"] = float(ADAMW_SCALES[task_index])
            row[f"metric_diagonal_task_{suffix}"] = float(
                METRIC_DIAGONAL[task_index]
            )
            row[f"cost_task_{suffix}"] = float(TASK_COSTS[task_index])
        rows.append(row)
    return rows


def moment_experiment(proposal: np.ndarray) -> list[dict[str, Any]]:
    """Estimate AdamW moment bias under a proposal different from the target.

    For moment vectors x and their target values x_ref, relative bias is
    ||x - x_ref||_2 / ||x_ref||_2. The Monte Carlo columns insert empirical
    moment averages for x; the calculated columns insert exact expectations.
    """
    rng = np.random.default_rng(MOMENT_SEED)
    num_tasks = 3
    dimension = 24
    target = np.full(num_tasks, 1.0 / num_tasks)

    coordinate = np.linspace(0.6, 1.4, dimension)
    means = np.stack(
        [
            0.20 * coordinate,
            0.13 * np.roll(coordinate, 5),
            0.09 * np.roll(coordinate, 11),
        ]
    )
    scales = np.stack(
        [
            1.10 * coordinate,
            0.55 * np.roll(coordinate, 7),
            0.28 * np.roll(coordinate, 13),
        ]
    )
    task_seconds = means**2 + scales**2
    reference_first = np.sum(target[:, None] * means, axis=0)
    reference_second = np.sum(target[:, None] * task_seconds, axis=0)

    first_sum = np.zeros(dimension)
    standard_second_sum = np.zeros(dimension)
    consistent_second_sum = np.zeros(dimension)
    seen = 0

    while seen < MOMENT_DRAWS:
        batch = min(MOMENT_CHUNK_SIZE, MOMENT_DRAWS - seen)
        task = rng.choice(num_tasks, size=batch, p=proposal)
        gradients = means[task] + scales[task] * rng.standard_normal((batch, dimension))
        weight = target[task] / proposal[task]
        first_sum += np.sum(weight[:, None] * gradients, axis=0)
        standard_second_sum += np.sum((weight[:, None] * gradients) ** 2, axis=0)
        consistent_second_sum += np.sum(weight[:, None] * gradients**2, axis=0)
        seen += batch

    empirical_first = first_sum / MOMENT_DRAWS
    empirical_standard_second = standard_second_sum / MOMENT_DRAWS
    empirical_consistent_second = consistent_second_sum / MOMENT_DRAWS
    exact_standard_second = np.sum(
        (target**2 / proposal)[:, None] * task_seconds, axis=0
    )

    def relative_bias(value: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))

    common = {
        "draws": MOMENT_DRAWS,
        "seed": MOMENT_SEED,
        "q_task_1": float(proposal[0]),
        "q_task_2": float(proposal[1]),
        "q_task_3": float(proposal[2]),
    }
    return [
        {
            "optimizer": "Standard AdamW",
            "moment": "First moment",
            "mc_relative_bias": relative_bias(empirical_first, reference_first),
            "calculated_relative_bias": 0.0,
            **common,
        },
        {
            "optimizer": "Moment-consistent AdamW",
            "moment": "First moment",
            "mc_relative_bias": relative_bias(empirical_first, reference_first),
            "calculated_relative_bias": 0.0,
            **common,
        },
        {
            "optimizer": "Standard AdamW",
            "moment": "Second moment",
            "mc_relative_bias": relative_bias(
                empirical_standard_second, reference_second
            ),
            "calculated_relative_bias": relative_bias(
                exact_standard_second, reference_second
            ),
            **common,
        },
        {
            "optimizer": "Moment-consistent AdamW",
            "moment": "Second moment",
            "mc_relative_bias": relative_bias(
                empirical_consistent_second, reference_second
            ),
            "calculated_relative_bias": 0.0,
            **common,
        },
    ]


def stress_scan() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Calculate proposal quality for 10,000 random four-task geometries."""
    rng = np.random.default_rng(STRESS_SEED)
    task_coordinate_seconds = np.exp(
        rng.uniform(
            LOG_SCALE_LOW,
            LOG_SCALE_HIGH,
            size=(STRESS_GEOMETRIES, STRESS_TASKS, STRESS_DIMENSION),
        )
    )
    metric_squared = np.exp(
        rng.uniform(
            LOG_SCALE_LOW,
            LOG_SCALE_HIGH,
            size=(STRESS_GEOMETRIES, STRESS_DIMENSION),
        )
    )

    raw_scales = np.sqrt(np.sum(task_coordinate_seconds, axis=2))
    adamw_scales = np.sqrt(
        np.sum(task_coordinate_seconds * metric_squared[:, None, :], axis=2)
    )
    raw_proposals = raw_scales / np.sum(raw_scales, axis=1, keepdims=True)
    adamw_proposals = adamw_scales / np.sum(adamw_scales, axis=1, keepdims=True)
    uniform_proposals = np.full_like(raw_proposals, 1.0 / STRESS_TASKS)
    target = np.full(STRESS_TASKS, 1.0 / STRESS_TASKS)

    def metric_mse(proposals: np.ndarray) -> np.ndarray:
        return np.sum((target[None, :] * adamw_scales) ** 2 / proposals, axis=1)

    uniform_mse = metric_mse(uniform_proposals)
    raw_mse_ratio = metric_mse(raw_proposals) / uniform_mse
    adamw_mse_ratio = metric_mse(adamw_proposals) / uniform_mse
    adamw_vs_raw_ratio = adamw_mse_ratio / raw_mse_ratio
    proposal_l1 = np.sum(np.abs(raw_proposals - adamw_proposals), axis=1)

    rows: list[dict[str, Any]] = []
    for geometry in range(STRESS_GEOMETRIES):
        row: dict[str, Any] = {
            "geometry_id": geometry,
            "seed": STRESS_SEED,
            "raw_proposal_adamw_mse_ratio": float(raw_mse_ratio[geometry]),
            "adamw_proposal_adamw_mse_ratio": float(adamw_mse_ratio[geometry]),
            "adamw_vs_raw_mse_ratio": float(adamw_vs_raw_ratio[geometry]),
            "proposal_l1_distance": float(proposal_l1[geometry]),
            "metric_squared_min": float(np.min(metric_squared[geometry])),
            "metric_squared_max": float(np.max(metric_squared[geometry])),
        }
        for task_index in range(STRESS_TASKS):
            suffix = task_index + 1
            row[f"raw_scale_task_{suffix}"] = float(raw_scales[geometry, task_index])
            row[f"adamw_scale_task_{suffix}"] = float(
                adamw_scales[geometry, task_index]
            )
            row[f"raw_prob_task_{suffix}"] = float(
                raw_proposals[geometry, task_index]
            )
            row[f"adamw_prob_task_{suffix}"] = float(
                adamw_proposals[geometry, task_index]
            )
        rows.append(row)

    def distribution_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
        return {
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_p05": float(np.quantile(values, 0.05)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_p95": float(np.quantile(values, 0.95)),
            f"{prefix}_max": float(np.max(values)),
        }

    summary = {
        "geometries": STRESS_GEOMETRIES,
        "tasks": STRESS_TASKS,
        "dimension": STRESS_DIMENSION,
        "seed": STRESS_SEED,
        "log_uniform_low": LOG_SCALE_LOW,
        "log_uniform_high": LOG_SCALE_HIGH,
        **distribution_fields("raw_mse_ratio", raw_mse_ratio),
        **distribution_fields("adamw_mse_ratio", adamw_mse_ratio),
        **distribution_fields("adamw_vs_raw_ratio", adamw_vs_raw_ratio),
        **distribution_fields("proposal_l1_distance", proposal_l1),
        "fraction_adamw_better_than_raw": float(
            np.mean(adamw_mse_ratio < raw_mse_ratio)
        ),
        "fraction_adamw_better_than_uniform": float(np.mean(adamw_mse_ratio < 1.0)),
        "fraction_raw_better_than_uniform": float(np.mean(raw_mse_ratio < 1.0)),
    }
    plot_values = {
        "raw_mse_ratio": raw_mse_ratio,
        "adamw_mse_ratio": adamw_mse_ratio,
    }
    return rows, [summary], plot_values


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
        }
    )


def make_controlled_figure(
    sampling: list[dict[str, Any]], moment: list[dict[str, Any]]
) -> None:
    methods = [str(row["method"]) for row in sampling]
    short_methods = ["Uniform", "GradNorm", "GPAS", "Cost"]
    colors = ["#7B8794", "#E69F00", "#3B77B4", "#2A9D8F"]
    configure_plot_style()
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axes_grid.ravel()

    x = np.arange(3)
    width = 0.18
    for method_index, row in enumerate(sampling):
        probability = [float(row[f"prob_task_{task}"]) for task in (1, 2, 3)]
        axes[0].bar(
            x + (method_index - 1.5) * width,
            probability,
            width=width,
            color=colors[method_index],
            label=methods[method_index],
        )
    axes[0].set_title("Task proposals")
    axes[0].set_ylabel("sampling probability")
    axes[0].set_xticks(x, ["task 1", "task 2", "task 3"])
    axes[0].set_ylim(0, 0.84)
    axes[0].legend(frameon=False, loc="upper center", ncol=2)

    adamw_empirical = [float(row["adamw_empirical_mse_ratio"]) for row in sampling]
    adamw_calculated = [float(row["adamw_theory_mse_ratio"]) for row in sampling]
    axes[1].bar(np.arange(4), adamw_empirical, color=colors, width=0.68)
    axes[1].scatter(np.arange(4), adamw_calculated, color="black", marker="x", s=22)
    axes[1].set_title("AdamW-metric estimator error")
    axes[1].set_ylabel("relative to Uniform")
    axes[1].set_xticks(np.arange(4), short_methods, rotation=20)
    axes[1].set_ylim(0, 1.08 * max(adamw_empirical + adamw_calculated))

    cost_empirical = [float(row["cost_empirical_ratio"]) for row in sampling]
    cost_calculated = [float(row["cost_theory_ratio"]) for row in sampling]
    axes[2].bar(np.arange(4), cost_empirical, color=colors, width=0.68)
    axes[2].scatter(np.arange(4), cost_calculated, color="black", marker="x", s=22)
    axes[2].set_title("Time-weighted objective $J(q)$")
    axes[2].set_ylabel("relative to Uniform")
    axes[2].set_xticks(np.arange(4), short_methods, rotation=20)
    axes[2].set_ylim(0, 1.08 * max(cost_empirical + cost_calculated))

    moment_names = ["First moment", "Second moment"]
    optimizer_names = ["Standard AdamW", "Moment-consistent AdamW"]
    moment_x = np.arange(2)
    drift_colors = ["#D55E00", "#2A9D8F"]
    for optimizer_index, optimizer in enumerate(optimizer_names):
        rows = [
            next(
                row
                for row in moment
                if row["optimizer"] == optimizer and row["moment"] == moment_name
            )
            for moment_name in moment_names
        ]
        offset = (optimizer_index - 0.5) * 0.34
        axes[3].bar(
            moment_x + offset,
            [float(row["mc_relative_bias"]) for row in rows],
            width=0.34,
            color=drift_colors[optimizer_index],
            label=optimizer.replace(" AdamW", ""),
        )
        axes[3].scatter(
            moment_x + offset,
            [float(row["calculated_relative_bias"]) for row in rows],
            color="black",
            marker="x",
            s=22,
            zorder=3,
        )
    axes[3].set_title("Expected moment change")
    axes[3].set_ylabel("relative bias")
    axes[3].set_xticks(moment_x, ["first", "second"])
    axes[3].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.75)
        axis.set_axisbelow(True)

    fig.tight_layout(w_pad=1.3, h_pad=1.5)
    CONTROLLED_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONTROLLED_FIGURE, bbox_inches="tight")
    plt.close(fig)


def make_stress_figure(plot_values: dict[str, np.ndarray]) -> None:
    configure_plot_style()
    fig, axis = plt.subplots(figsize=(5.4, 3.15))
    series = [
        ("Raw-scale proposal", plot_values["raw_mse_ratio"], "#E69F00"),
        ("AdamW-scale proposal", plot_values["adamw_mse_ratio"], "#3B77B4"),
    ]
    for label, values, color in series:
        ordered = np.sort(values)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.step(ordered, cumulative, where="post", color=color, label=label)
    axis.axvline(1.0, color="#4F5660", linestyle="--", linewidth=0.9, label="Uniform")
    axis.set_title("Random optimizer geometries")
    axis.set_xlabel("AdamW-metric MSE relative to Uniform")
    axis.set_ylabel("fraction of geometries")
    axis.set_ylim(0, 1.01)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color="#D9DEE3", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    STRESS_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STRESS_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sampling = controlled_sampling_experiment()
    write_rows(SAMPLING_RESULTS, sampling)

    gpas = next(row for row in sampling if row["method"] == "GPAS")
    proposal = np.array([float(gpas[f"prob_task_{task}"]) for task in (1, 2, 3)])
    moment = moment_experiment(proposal)
    write_rows(MOMENT_RESULTS, moment)

    stress_rows, stress_summary, stress_plot_values = stress_scan()
    write_rows(STRESS_RESULTS, stress_rows)
    write_rows(STRESS_SUMMARY, stress_summary)

    make_controlled_figure(sampling, moment)
    make_stress_figure(stress_plot_values)


if __name__ == "__main__":
    main()
