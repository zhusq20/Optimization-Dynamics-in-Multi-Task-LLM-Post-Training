"""Reproduce the controlled checks for all-task micro-batch GPAS."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALLOCATION_RESULTS = ROOT / "experiments" / "controlled_optimizer_sampling_results.csv"
MOMENT_RESULTS = ROOT / "experiments" / "controlled_optimizer_moment_results.csv"
CONTROLLED_FIGURE = ROOT / "figures" / "controlled_optimizer_sampling.pdf"

SEED = 20260902
TRIALS = 20_000
MOMENT_DRAWS = 200_000
MOMENT_CHUNK = 5_000

TASKS = 3
TOTAL_MICROBATCHES = 16
MIN_MICROBATCHES = 2
MAX_MICROBATCHES = 12
TARGET_WEIGHTS = np.full(TASKS, 1.0 / TASKS)

# A fixed diagonal preconditioner reverses the unpreconditioned and
# preconditioned noise rankings.
UNPRECONDITIONED_NOISE_SCALES = np.array(
    [0.7652593561149718, 0.19230656846142996, 0.04243407542359844]
)
PRECONDITIONED_NOISE_SCALES = np.array(
    [0.1740545215177264, 0.1969024054681955, 0.6290430730140781]
)
METRIC_DIAGONAL = PRECONDITIONED_NOISE_SCALES / UNPRECONDITIONED_NOISE_SCALES

# These costs produce a distinct cost-aware integer allocation.
TASK_COSTS = np.array([0.8652444749092914, 0.3460977899637165, 1.730488949818583])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bounded_largest_remainder(scores: np.ndarray) -> np.ndarray:
    """Apply bounded proportional allocation followed by largest remainder."""
    if TOTAL_MICROBATCHES < TASKS * MIN_MICROBATCHES:
        raise ValueError("infeasible lower bound")
    if TOTAL_MICROBATCHES > TASKS * MAX_MICROBATCHES:
        raise ValueError("infeasible upper bound")
    if np.any(scores <= 0):
        raise ValueError("allocation scores must be positive")

    # Find z with sum(clip(score / z, lower, upper)) = G.
    low = np.min(scores) / (MAX_MICROBATCHES * 2.0)
    high = np.max(scores) / (MIN_MICROBATCHES * 0.5)
    for _ in range(100):
        middle = 0.5 * (low + high)
        total = np.clip(
            scores / middle, MIN_MICROBATCHES, MAX_MICROBATCHES
        ).sum()
        if total > TOTAL_MICROBATCHES:
            low = middle
        else:
            high = middle
    continuous = np.clip(
        scores / high, MIN_MICROBATCHES, MAX_MICROBATCHES
    )
    counts = np.floor(continuous + 1e-12).astype(int)
    remainder = TOTAL_MICROBATCHES - int(counts.sum())
    order = np.argsort(-(continuous - counts), kind="stable")
    for index in order:
        if remainder == 0:
            break
        if counts[index] < MAX_MICROBATCHES:
            counts[index] += 1
            remainder -= 1
    if remainder != 0 or counts.sum() != TOTAL_MICROBATCHES:
        raise RuntimeError("bounded rounding failed")
    return counts


def preconditioned_variance(counts: np.ndarray) -> float:
    return float(
        np.sum((TARGET_WEIGHTS * PRECONDITIONED_NOISE_SCALES) ** 2 / counts)
    )


def controlled_allocation_experiment() -> list[dict[str, Any]]:
    """Compare fixed-count stratified allocations using common random numbers."""
    methods = [
        ("Uniform", np.ones(TASKS)),
        ("Counts from unpreconditioned noise", TARGET_WEIGHTS * UNPRECONDITIONED_NOISE_SCALES),
        ("GPAS (per step)", TARGET_WEIGHTS * PRECONDITIONED_NOISE_SCALES),
        (
            "GPAS (per GPU hour)",
            TARGET_WEIGHTS * PRECONDITIONED_NOISE_SCALES / np.sqrt(TASK_COSTS),
        ),
    ]
    allocations = [(name, bounded_largest_remainder(score)) for name, score in methods]

    rng = np.random.default_rng(SEED)
    max_count = max(int(counts.max()) for _, counts in allocations)
    normal = rng.standard_normal((TRIALS, TASKS, max_count))

    theoretical = np.array([preconditioned_variance(counts) for _, counts in allocations])
    empirical = []
    predicted_time = []
    for _, counts in allocations:
        estimate = np.zeros((TRIALS, TASKS))
        for task in range(TASKS):
            estimate[:, task] = (
                TARGET_WEIGHTS[task]
                * UNPRECONDITIONED_NOISE_SCALES[task]
                * normal[:, task, : counts[task]].mean(axis=1)
            )
        preconditioned = estimate * METRIC_DIAGONAL[None, :]
        empirical.append(float(np.mean(np.sum(preconditioned**2, axis=1))))
        predicted_time.append(float(counts @ TASK_COSTS))

    theoretical /= theoretical[0]
    empirical = np.asarray(empirical) / empirical[0]
    predicted_time = np.asarray(predicted_time) / predicted_time[0]
    cost_theory = theoretical * predicted_time
    cost_empirical = empirical * predicted_time

    rows: list[dict[str, Any]] = []
    for method_index, (method, counts) in enumerate(allocations):
        row: dict[str, Any] = {
            "method": method,
            "microbatches_task_1": int(counts[0]),
            "microbatches_task_2": int(counts[1]),
            "microbatches_task_3": int(counts[2]),
            "preconditioned_theory_variance_ratio": float(theoretical[method_index]),
            "preconditioned_empirical_variance_ratio": float(empirical[method_index]),
            "predicted_time_ratio": float(predicted_time[method_index]),
            "cost_theory_ratio": float(cost_theory[method_index]),
            "cost_empirical_ratio": float(cost_empirical[method_index]),
            "seed": SEED,
            "trials": TRIALS,
            "total_microbatches": TOTAL_MICROBATCHES,
            "minimum_microbatches": MIN_MICROBATCHES,
            "maximum_microbatches": MAX_MICROBATCHES,
        }
        for task in range(TASKS):
            suffix = task + 1
            row[f"unpreconditioned_noise_scale_task_{suffix}"] = float(
                UNPRECONDITIONED_NOISE_SCALES[task]
            )
            row[f"preconditioned_noise_scale_task_{suffix}"] = float(
                PRECONDITIONED_NOISE_SCALES[task]
            )
            row[f"metric_diagonal_task_{suffix}"] = float(METRIC_DIAGONAL[task])
            row[f"cost_task_{suffix}"] = float(TASK_COSTS[task])
        rows.append(row)
    return rows


def moment_experiment(uniform: np.ndarray, adaptive: np.ndarray) -> list[dict[str, Any]]:
    """Measure moment changes caused by replacing uniform with adaptive counts."""
    dimension = 24
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
    first_exact = np.sum(TARGET_WEIGHTS[:, None] * means, axis=0)
    taskwise_second_exact = np.sum(
        TARGET_WEIGHTS[:, None] * task_seconds, axis=0
    ) / TOTAL_MICROBATCHES

    def standard_second_exact(counts: np.ndarray) -> np.ndarray:
        return first_exact**2 + np.sum(
            (TARGET_WEIGHTS**2 / counts)[:, None] * scales**2, axis=0
        )

    exact_by_allocation = {
        "Uniform": {
            "First moment": first_exact,
            "Standard second moment": standard_second_exact(uniform),
            "Taskwise second moment": taskwise_second_exact,
        },
        "GPAS (per step)": {
            "First moment": first_exact,
            "Standard second moment": standard_second_exact(adaptive),
            "Taskwise second moment": taskwise_second_exact,
        },
    }

    sums: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(SEED + 1)
    max_count = int(max(uniform.max(), adaptive.max()))
    seen = 0
    for label in ("Uniform", "GPAS (per step)"):
        sums[label] = {
            "First moment": np.zeros(dimension),
            "Standard second moment": np.zeros(dimension),
            "Taskwise second moment": np.zeros(dimension),
        }

    while seen < MOMENT_DRAWS:
        batch = min(MOMENT_CHUNK, MOMENT_DRAWS - seen)
        normal = rng.standard_normal((batch, TASKS, max_count, dimension))
        gradients = means[None, :, None, :] + scales[None, :, None, :] * normal
        for label, counts in (("Uniform", uniform), ("GPAS (per step)", adaptive)):
            first = np.zeros((batch, dimension))
            taskwise_second = np.zeros((batch, dimension))
            for task in range(TASKS):
                selected = gradients[:, task, : counts[task], :]
                first += TARGET_WEIGHTS[task] * selected.mean(axis=1)
                taskwise_second += (
                    TARGET_WEIGHTS[task]
                    * (selected**2).mean(axis=1)
                    / TOTAL_MICROBATCHES
                )
            sums[label]["First moment"] += first.sum(axis=0)
            sums[label]["Standard second moment"] += (first**2).sum(axis=0)
            sums[label]["Taskwise second moment"] += taskwise_second.sum(axis=0)
        seen += batch

    empirical = {
        label: {name: value / MOMENT_DRAWS for name, value in values.items()}
        for label, values in sums.items()
    }

    def relative_change(value: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))

    rows: list[dict[str, Any]] = []
    for name in ("First moment", "Standard second moment", "Taskwise second moment"):
        rows.append(
            {
                "observation": name,
                "mc_relative_change_from_uniform": relative_change(
                    empirical["GPAS (per step)"][name], empirical["Uniform"][name]
                ),
                "calculated_relative_change_from_uniform": relative_change(
                    exact_by_allocation["GPAS (per step)"][name],
                    exact_by_allocation["Uniform"][name],
                ),
                "uniform_counts": "-".join(str(int(value)) for value in uniform),
                "adaptive_counts": "-".join(str(int(value)) for value in adaptive),
                "draws": MOMENT_DRAWS,
                "seed": SEED + 1,
            }
        )
    return rows


def make_figure(
    allocation_rows: list[dict[str, Any]], moment_rows: list[dict[str, Any]]
) -> None:
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
    methods = [
        "Uniform",
        "Counts from unpreconditioned noise",
        "GPAS (per step)",
        "GPAS (per GPU hour)",
    ]
    short_methods = ["Uniform", "Unprecond.\ncounts", "GPAS\nper step", "GPAS\nper GPU h"]
    colors = ["#7B8794", "#E69F00", "#3B77B4", "#2A9D8F"]
    fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axes_grid.ravel()

    x = np.arange(TASKS)
    width = 0.18
    for method_index, row in enumerate(allocation_rows):
        counts = [float(row[f"microbatches_task_{task}"]) for task in (1, 2, 3)]
        axes[0].bar(
            x + (method_index - 1.5) * width,
            counts,
            width=width,
            color=colors[method_index],
            label=methods[method_index],
        )
    axes[0].set_title("Integer micro-batch allocation")
    axes[0].set_ylabel("micro-batches per step")
    axes[0].set_xticks(x, ["task 1", "task 2", "task 3"])
    axes[0].set_ylim(0, 16.5)
    axes[0].set_yticks(np.arange(0, 13, 2))
    axes[0].legend(frameon=False, loc="upper center", ncol=2, fontsize=6.2)

    empirical = [float(row["preconditioned_empirical_variance_ratio"]) for row in allocation_rows]
    calculated = [float(row["preconditioned_theory_variance_ratio"]) for row in allocation_rows]
    axes[1].bar(np.arange(4), empirical, color=colors, width=0.68)
    axes[1].scatter(np.arange(4), calculated, color="black", marker="x", s=22)
    axes[1].set_title("Preconditioned gradient variance")
    axes[1].set_ylabel("relative to Uniform")
    axes[1].set_xticks(np.arange(4), short_methods, rotation=20)

    cost_empirical = [float(row["cost_empirical_ratio"]) for row in allocation_rows]
    cost_calculated = [float(row["cost_theory_ratio"]) for row in allocation_rows]
    axes[2].bar(np.arange(4), cost_empirical, color=colors, width=0.68)
    axes[2].scatter(np.arange(4), cost_calculated, color="black", marker="x", s=22)
    axes[2].set_title("Predicted time $\\times$ variance")
    axes[2].set_ylabel("relative to Uniform")
    axes[2].set_xticks(np.arange(4), short_methods, rotation=20)

    moment_labels = ["First", "Adam-type\n$A\\odot A$", "Alternative\n$U/G$"]
    moment_empirical = [
        float(row["mc_relative_change_from_uniform"]) for row in moment_rows
    ]
    moment_calculated = [
        float(row["calculated_relative_change_from_uniform"]) for row in moment_rows
    ]
    axes[3].bar(np.arange(3), moment_empirical, color=["#7B8794", "#D55E00", "#2A9D8F"], width=0.68)
    axes[3].scatter(np.arange(3), moment_calculated, color="black", marker="x", s=22)
    axes[3].set_title("Change after reallocating counts")
    axes[3].set_ylabel("relative change from Uniform")
    axes[3].set_xticks(np.arange(3), moment_labels)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.75)
        axis.set_axisbelow(True)
    fig.tight_layout(w_pad=1.3, h_pad=1.5)
    CONTROLLED_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONTROLLED_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    allocation = controlled_allocation_experiment()
    write_rows(ALLOCATION_RESULTS, allocation)
    uniform_counts = np.array(
        [allocation[0][f"microbatches_task_{task}"] for task in (1, 2, 3)],
        dtype=int,
    )
    gpas_row = next(
        row for row in allocation if row["method"] == "GPAS (per step)"
    )
    gpas_counts = np.array(
        [gpas_row[f"microbatches_task_{task}"] for task in (1, 2, 3)],
        dtype=int,
    )
    moment = moment_experiment(uniform_counts, gpas_counts)
    write_rows(MOMENT_RESULTS, moment)
    make_figure(allocation, moment)


if __name__ == "__main__":
    main()
