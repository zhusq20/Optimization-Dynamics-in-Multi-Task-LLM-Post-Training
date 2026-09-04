"""Draw exact analytical examples of joint progress; no training data are used.

Panel A uses gradient flow for equally weighted, orthogonal quadratic losses.
Panels B/C use independent Gaussian micro-batch gradients with identity
covariance, equal task weights, and equal counts within each total budget G.
Increasing G is a precision intervention, not fixed-budget GPAS allocation.
"""

from __future__ import annotations

from math import erf, isclose, pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
WIDTH, HEIGHT = 5.5, 3.55  # Native ICLR text width; no font downscaling needed.
INK = "#243342"
MUTED = "#626D77"
BLUE = "#3478A9"
ORANGE = "#C46B24"
TEAL = "#278577"
GRID = "#E4E8EC"
WEIGHTS = np.array([0.5, 0.5])
MEAN_ONE = np.array([1.0, 0.0])
BUDGETS = (4, 64)


def normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / sqrt(2 * pi * variance)


def wrong_direction_probability(mean: float, total_count: int) -> float:
    """P(X <= 0) for X ~ N(mean, 1 / total_count)."""
    return 0.5 * (1 + erf(-mean * sqrt(total_count / 2)))


def verify_examples() -> dict[str, tuple[float, float]]:
    """Check the stated means, covariances, and exact Gaussian probabilities."""
    result = {}
    for name, second_mean, expected_margins in (
        ("harmful", np.array([-2.0, 1.0]), np.array([-0.5, 1.5])),
        ("useful", np.array([-0.6, 1.0]), np.array([0.2, 0.38])),
    ):
        task_means = np.stack((MEAN_ONE, second_mean))
        combined_mean = WEIGHTS @ task_means
        margins = task_means @ combined_mean  # D = identity.
        np.testing.assert_allclose(margins, expected_margins, atol=1e-14)
        for count in BUDGETS:
            counts = np.array([count / 2, count / 2])
            combined_covariance = sum(
                weight**2 * np.eye(2) / task_count
                for weight, task_count in zip(WEIGHTS, counts)
            )
            np.testing.assert_allclose(combined_covariance, np.eye(2) / count)
            variance = MEAN_ONE @ combined_covariance @ MEAN_ONE
            assert isclose(variance, 1 / count)
        result[name] = tuple(
            wrong_direction_probability(float(margins[0]), count)
            for count in BUDGETS
        )
    assert result["harmful"][0] < result["harmful"][1]
    assert result["useful"][0] > result["useful"][1]

    # Orthogonal quadratic flow: theta_i(t) = exp(-w_i * lambda_i * t).
    curvature = np.array([1.0, 0.1])
    time = np.array([0.0, 1.0, 10.0])
    coordinate = np.exp(-WEIGHTS[:, None] * curvature[:, None] * time)
    normalized_loss = coordinate**2
    np.testing.assert_allclose(normalized_loss[0], np.exp(-time))
    np.testing.assert_allclose(normalized_loss[1], np.exp(-0.1 * time))
    return result


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A3ADB6")
    ax.spines[["left", "bottom"]].set_linewidth(0.65)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.65, pad=2)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_axisbelow(True)


def flow_box(fig: plt.Figure, left: float, width: float, lines: str) -> None:
    fig.add_artist(
        Rectangle(
            (left, 0.82), width, 0.15,
            transform=fig.transFigure, facecolor="#F7F9FB",
            edgecolor="#D9E0E6", linewidth=0.65,
        )
    )
    fig.text(
        left + width / 2, 0.895, lines,
        ha="center", va="center", fontsize=8.5, linespacing=1.6,
    )


def draw_density_panel(
    ax: plt.Axes, mean: float, panel_title: str, show_ylabel: bool
) -> None:
    x = np.linspace(-1.8, 1.5, 2001)
    ax.axvspan(-1.8, 0, facecolor="#F4E9E8", alpha=0.8, linewidth=0)
    for count, color in zip(BUDGETS, (BLUE, ORANGE)):
        density = normal_pdf(x, mean, 1 / count)
        ax.plot(x, density, color=color, lw=1.5)
    ax.axvline(0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set(xlim=(-1.8, 1.5), ylim=(0, 3.55), xticks=(-1, 0, 1), yticks=(0, 1, 2, 3))
    ax.set_title(panel_title, pad=8, fontsize=9, weight="semibold")
    ax.set_xlabel(r"Sampled progress $X_1$", labelpad=4)
    if show_ylabel:
        ax.set_ylabel("Density", labelpad=3)
    ax.text(
        0.04, 0.91, rf"$a_1={mean:g}$", transform=ax.transAxes,
        fontsize=8.5, color=INK,
    )
    style_axis(ax)


def main() -> None:
    probabilities = verify_examples()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")

    flow_box(fig, 0.015, 0.285, "Student prefixes\n+ teacher targets")
    flow_box(fig, 0.358, 0.255, "Task gradients\n" + r"$\mu_i$, $\Sigma_i$")
    flow_box(
        fig, 0.671, 0.314,
        r"Fixed $w,D$: mean $a_i$" + "\n" + r"Counts $m$: variance $v_i$",
    )
    for start, stop in ((0.305, 0.352), (0.618, 0.665)):
        fig.add_artist(
            FancyArrowPatch(
                (start, 0.895), (stop, 0.895), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=MUTED,
            )
        )

    # Explicit manual placement leaves room for readable labels at 5.5 inches.
    axes = [
        fig.add_axes([0.108, 0.27, 0.215, 0.38]),
        fig.add_axes([0.418, 0.27, 0.228, 0.38]),
        fig.add_axes([0.754, 0.27, 0.228, 0.38]),
    ]

    time = np.linspace(0, 10, 400)
    axes[0].plot(time, np.exp(-time), color=BLUE, linewidth=1.6, label=r"$e^{-t}$")
    axes[0].plot(time, np.exp(-0.1 * time), color=TEAL, linewidth=1.6, label=r"$e^{-0.1t}$")
    axes[0].set(
        yscale="log", ylim=(3e-5, 1.4), xlim=(0, 10),
        xticks=(0, 5, 10), yticks=(1e-4, 1e-2, 1),
    )
    axes[0].set_title("(a) Unequal rates", pad=8, fontsize=9, weight="semibold")
    axes[0].set_ylabel("Normalized loss", labelpad=3)
    axes[0].set_xlabel(r"Flow time $t$", labelpad=4)
    axes[0].grid(axis="y", color=GRID, linewidth=0.55)
    axes[0].legend(
        loc="lower left", fontsize=8.5, frameon=False,
        handlelength=1.4, handletextpad=0.4, borderaxespad=0.1,
    )
    style_axis(axes[0])

    draw_density_panel(axes[1], -0.5, "(b) Harmful mean", True)
    draw_density_panel(axes[2], 0.2, "(c) Useful mean", False)
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.5, label=r"$G=4$"),
            Line2D([0], [0], color=ORANGE, lw=1.5, label=r"$G=64$"),
        ],
        loc="center", bbox_to_anchor=(0.697, 0.757), ncol=2,
        fontsize=8.5, frameon=False, columnspacing=1.6,
        handlelength=1.8, handletextpad=0.5,
    )
    fig.text(0.2155, 0.757, "Exact mean updates", ha="center", va="center", fontsize=8.5)

    fig.text(0.2155, 0.107, r"All $a_i>0$; zero conflict", ha="center", fontsize=8.5)
    fig.text(0.2155, 0.067, "No sampling noise", ha="center", fontsize=8.5, color=MUTED)
    for center, name in ((0.532, "harmful"), (0.868, "useful")):
        small, large = probabilities[name]
        fig.text(center, 0.107, r"$\Pr(X_1\leq0)$, $G$: 4 $\to$ 64", ha="center", fontsize=8.5)
        digits = 3 if name == "harmful" else 1
        fig.text(
            center, 0.067,
            rf"{100 * small:.1f}% $\to$ {100 * large:.{digits}f}%",
            ha="center", fontsize=8.5, color=MUTED,
        )
    fig.text(
        0.5, 0.015,
        r"Analytical examples, not MOPD measurements. Shading: $X_1\leq0$.",
        ha="center", fontsize=8, color=MUTED,
    )

    OUT.mkdir(exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"joint_progress_teaser.{extension}", dpi=300)
    plt.close(fig)
    print(f"Saved joint_progress_teaser: {WIDTH} x {HEIGHT} inches; smallest font 8 pt.")
    for name, values in probabilities.items():
        print(f"{name}: P(X1 <= 0), G={BUDGETS}: {values}")


if __name__ == "__main__":
    main()
