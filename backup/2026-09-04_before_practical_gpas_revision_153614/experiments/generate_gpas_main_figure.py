"""Generate the overview figure for the GPAS paper.

The figure is intentionally built from vector primitives so every label and
equation remains editable and sharp in the paper PDF.  It is sized to the
5.5-inch text width used by the ICLR 2027 style in this repository.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "figures" / "gpas_main_figure.pdf"
OUT_SVG = ROOT / "figures" / "gpas_main_figure.svg"
OUT_PNG = ROOT / "figures" / "gpas_main_figure.png"


# Existing controlled figures use blue, teal, amber, and neutral gray.  The
# same family is retained here.
NAVY = "#173B57"
BLUE = "#3F7FBF"
TEAL = "#2A9D8F"
AMBER = "#E69F00"
RED = "#D55E00"
GRAY = "#7F8C99"
MID_GRAY = "#A9B3BC"
LIGHT_GRAY = "#D8E0E7"
PANEL = "#F8FAFC"
BLUE_FILL = "#EEF5FC"
TEAL_FILL = "#ECF8F5"
RED_FILL = "#FFF2EC"
WHITE = "#FFFFFF"

TASK_COLORS = [AMBER, BLUE, TEAL, RED]
TASK_SHORT = ["M", "C", "I", "S"]
TASK_NAMES = ["Math", "Code", "Instruction", "Science"]


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LIGHT_GRAY,
    linewidth: float = 0.8,
    radius: float = 0.8,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    linewidth: float = 1.0,
    mutation_scale: float = 7.5,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    zorder: int = 4,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def segmented_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    fractions: list[float],
    *,
    labels: bool,
) -> None:
    cursor = x
    for index, (fraction, color) in enumerate(zip(fractions, TASK_COLORS)):
        segment_width = width * fraction
        ax.add_patch(
            Rectangle(
                (cursor, y),
                segment_width,
                height,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.65,
                zorder=3,
            )
        )
        if labels and segment_width > 2.1:
            ax.text(
                cursor + segment_width / 2,
                y + height / 2,
                TASK_SHORT[index],
                ha="center",
                va="center",
                fontsize=5.4,
                fontweight="bold",
                color=WHITE,
                zorder=4,
            )
        cursor += segment_width
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.35",
            facecolor="none",
            edgecolor=NAVY,
            linewidth=0.75,
            zorder=5,
        )
    )


def draw_lock(ax: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            1.15 * scale,
            0.92 * scale,
            facecolor=GRAY,
            edgecolor=GRAY,
            linewidth=0.6,
            zorder=4,
        )
    )
    ax.add_patch(
        Arc(
            (x + 0.575 * scale, y + 0.92 * scale),
            0.82 * scale,
            1.02 * scale,
            theta1=0,
            theta2=180,
            color=GRAY,
            linewidth=1.25,
            zorder=4,
        )
    )


def panel_title(ax: plt.Axes, x: float, y: float, label: str, title: str) -> None:
    ax.text(
        x,
        y,
        label,
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        x + 3.5,
        y,
        title,
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=NAVY,
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.2,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 2.55))
    fig.patch.set_facecolor(WHITE)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    ax.text(
        50,
        40.4,
        "Fixed objective weights; adaptive micro-batch counts",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=NAVY,
    )

    # Panel containers.
    rounded_box(ax, 1.0, 4.8, 28.5, 33.1, facecolor=PANEL, radius=1.1)
    rounded_box(ax, 31.2, 4.8, 31.8, 33.1, facecolor=PANEL, radius=1.1)
    rounded_box(ax, 64.7, 4.8, 34.2, 33.1, facecolor=PANEL, radius=1.1)

    panel_title(ax, 2.5, 36.1, "(a)", r"Fix $w$; allocate $m$")
    panel_title(ax, 32.7, 36.1, "(b)", "Measure update noise")
    panel_title(ax, 66.2, 36.1, "(c)", "Allocate compute")

    # ------------------------------------------------------------------ (a)
    ax.text(3.0, 32.8, "Fixed task weights", color=GRAY, fontsize=5.6, va="center")
    ax.text(
        3.0,
        30.8,
        r"Objective $w$",
        color=NAVY,
        fontsize=6.7,
        fontweight="bold",
        va="center",
    )
    draw_lock(ax, 23.6, 30.1, scale=0.85)
    ax.text(22.7, 30.8, "fixed", color=GRAY, fontsize=5.4, va="center", ha="right")
    segmented_bar(ax, 3.0, 27.4, 23.9, 2.15, [0.25, 0.25, 0.25, 0.25], labels=True)

    for idx, (name, color) in enumerate(zip(TASK_NAMES, TASK_COLORS)):
        ax.text(
            3.0 + idx * 6.0,
            25.4,
            "Instr." if name == "Instruction" else name,
            color=color,
            fontsize=4.8,
            ha="left",
            va="center",
        )

    ax.text(3.0, 21.9, "Micro-batches in the next step", color=GRAY, fontsize=5.6, va="center")
    ax.text(
        3.0,
        19.8,
        r"Counts $m=(m_1,\ldots,m_M)$",
        color=NAVY,
        fontsize=5.45,
        fontweight="bold",
        va="center",
    )
    ax.text(27.0, 21.0, "adaptive", color=BLUE, fontsize=5.1, va="center", ha="right")
    segmented_bar(ax, 3.0, 16.6, 23.9, 2.15, [0.14, 0.39, 0.29, 0.18], labels=True)

    rounded_box(ax, 3.0, 8.0, 23.9, 5.9, facecolor=WHITE, radius=0.7)
    ax.text(
        8.5,
        11.75,
        r"all tasks present",
        color=NAVY,
        fontsize=5.7,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        8.5,
        9.55,
        r"$m_{\min}\leq m_i\leq m_{\max}$",
        color=GRAY,
        fontsize=4.7,
        ha="center",
        va="center",
    )
    ax.plot([14.5, 14.5], [8.7, 13.2], color=LIGHT_GRAY, linewidth=0.8)
    ax.text(
        21.0,
        11.55,
        r"loss $\times\;w_i/m_i$",
        color=NAVY,
        fontsize=5.6,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(21.0, 9.55, "fixed objective", color=GRAY, fontsize=4.6, ha="center", va="center")
    arrow(ax, (9.0, 16.4), (9.0, 14.05), color=BLUE, linewidth=0.85)
    arrow(ax, (21.0, 27.2), (21.0, 14.05), color=GRAY, linewidth=0.75, linestyle="--")

    ax.text(
        14.9,
        6.55,
        r"$\mathbb{E}[A\mid m]=\sum_i w_i\mu_i$",
        color=TEAL,
        fontsize=6.1,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # ------------------------------------------------------------------ (b)
    ax.text(47.1, 32.6, "reuse gradients from this step", color=GRAY,
            fontsize=5.4, ha="center", va="center")
    rounded_box(ax, 33.2, 27.4, 27.8, 3.5, facecolor=WHITE, radius=0.65)
    ax.text(47.1, 29.15, r"top-$k$ micro-batch gradient $g_{i,s}$", color=NAVY,
            fontsize=5.6, fontweight="bold", ha="center", va="center")

    arrow(ax, (47.1, 27.25), (47.1, 24.9), color=BLUE, linewidth=0.9)
    rounded_box(ax, 33.2, 20.4, 27.8, 4.3, facecolor=BLUE_FILL,
                edgecolor="#A9C9E8", radius=0.65)
    ax.text(47.1, 23.5, "preconditioned gradient", color=BLUE, fontsize=5.4,
            fontweight="bold", ha="center", va="center")
    ax.text(47.1, 21.65, r"$Dg_{i,s}$; $D$ from optimizer (identity for SGD)",
            color=NAVY, fontsize=4.6, ha="center", va="center")

    arrow(ax, (47.1, 20.7), (47.1, 18.7), color=TEAL, linewidth=0.9)
    rounded_box(ax, 33.2, 10.1, 27.8, 8.3, facecolor=TEAL_FILL,
                edgecolor="#99D5C9", radius=0.7)
    ax.text(47.1, 16.5, r"task noise  $e_i=\mathrm{Var}(Dg_i)$",
            color=TEAL, fontsize=6.1, fontweight="bold", ha="center", va="center")
    for idx, (length, color) in enumerate(zip([6.0, 10.5, 4.8, 8.0], TASK_COLORS)):
        y = 14.3 - idx * 1.25
        ax.text(37.0, y, TASK_SHORT[idx], color=color, fontsize=4.8,
                fontweight="bold", ha="right", va="center")
        ax.plot([37.7, 37.7 + length], [y, y], color=color, linewidth=2.1,
                solid_capstyle="round")
    ax.text(47.1, 7.8, "no extra forward or backward pass", color=TEAL,
            fontsize=5.2, fontweight="bold", ha="center", va="center")

    # Horizontal stage transitions.
    arrow(ax, (29.6, 11.0), (31.0, 11.0), color=NAVY, linewidth=1.0)
    arrow(ax, (63.1, 11.0), (64.5, 11.0), color=TEAL, linewidth=1.0)

    # ------------------------------------------------------------------ (c)
    ax.text(66.7, 32.5, "reuse the completed step", color=GRAY, fontsize=5.4, va="center")
    rounded_box(ax, 66.7, 27.5, 13.9, 3.8, facecolor=WHITE, radius=0.65)
    rounded_box(ax, 82.4, 27.5, 14.5, 3.8, facecolor=WHITE, radius=0.65)
    ax.text(73.65, 29.4, "preconditioned\n" + r"noise $e_i$", color=BLUE,
            fontsize=4.65, fontweight="bold", ha="center", va="center", linespacing=0.9)
    ax.text(89.65, 29.4, r"micro-batch cost $\tau_i$", color=AMBER, fontsize=5.2, fontweight="bold", ha="center", va="center")

    # A tiny four-task profile reinforces that the allocation is taskwise.
    bar_x = 67.4
    for idx, (length, color) in enumerate(zip([5.4, 8.6, 4.1, 6.7], TASK_COLORS)):
        y = 25.5 - idx * 1.05
        ax.plot([bar_x, bar_x + length], [y, y], color=color, linewidth=2.0, solid_capstyle="round")
        ax.text(bar_x - 0.55, y, TASK_SHORT[idx], color=color, fontsize=4.6, fontweight="bold", ha="right", va="center")

    rounded_box(ax, 66.7, 16.4, 30.2, 5.2, facecolor=BLUE_FILL, edgecolor="#A9C9E8", radius=0.7)
    ax.text(68.2, 19.7, "GPAS", color=BLUE, fontsize=6.4, fontweight="bold", va="center")
    ax.text(82.5, 19.7, r"$m_i\propto w_i\sqrt{e_i}$", color=NAVY, fontsize=6.15, fontweight="bold", ha="center", va="center")
    ax.text(95.6, 19.7, r"min $V(m)$", color=GRAY, fontsize=4.8, ha="right", va="center")
    ax.text(81.8, 17.6, "optimizer-aware allocation", color=GRAY, fontsize=4.8, ha="center", va="center")

    rounded_box(ax, 66.7, 9.7, 30.2, 5.2, facecolor=TEAL_FILL, edgecolor="#99D5C9", radius=0.7)
    ax.text(81.8, 13.1, "GPAS: cost-aware mode", color=TEAL, fontsize=5.5,
            fontweight="bold", ha="center", va="center")
    ax.text(81.8, 10.95, r"min $J_C(m)$; if $C=0$:  $m_i\propto w_i\sqrt{e_i/\tau_i}$",
            color=NAVY, fontsize=4.55, ha="center", va="center")

    ax.text(81.8, 7.2, "Evaluate: teacher loss vs. steps  •  GPU hours", color=GRAY, fontsize=4.9, fontweight="bold", ha="center", va="center")

    arrow(ax, (80.7, 27.3), (80.7, 21.8), color=BLUE, linewidth=0.85)
    arrow(ax, (89.6, 27.3), (89.6, 15.1), color=AMBER, linewidth=0.85)

    # Online feedback: update m, never w.  The arc ends at the allocation
    # bar in panel (a), and stays below the panel content to avoid crossings.
    feedback_x = [91.8, 97.8, 97.8, 0.4, 0.4]
    feedback_y = [9.4, 3.25, 3.25, 3.25, 17.7]
    ax.plot(feedback_x, feedback_y, color=BLUE, linewidth=0.95, linestyle="--", zorder=6)
    arrow(ax, (0.4, 17.7), (2.8, 17.7), color=BLUE, linewidth=0.95, linestyle="--", zorder=6)
    ax.text(
        52.0,
        1.65,
        r"next step: update $m$ from current noise and cost; keep $w$ fixed",
        color=BLUE,
        fontsize=5.2,
        fontweight="bold",
        ha="center",
        va="center",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
