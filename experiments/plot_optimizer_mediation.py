"""Generate the Section 3 optimizer-mediation figure from frozen records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/optimizer_mediation_20260913"
FIGURES = ROOT / "figures/optimizer_mediation_20260913"
FIGURES.mkdir(parents=True, exist_ok=True)
ROWS = json.loads((DATA / "measurements.json").read_text())["records"]
ONLINE = json.loads((DATA / "online_clip_coefficients.json").read_text())["records"]

REDUCTIONS = ("DR", "DT", "GT")
REDUCTION_COLORS = {"DR": "#D55E00", "DT": "#009E73", "GT": "#CC79A7"}
PAIRS = ("DR_DT", "DR_GT", "DT_GT")
PAIR_LABELS = {"DR_DT": "DR vs DT", "DR_GT": "DR vs GT", "DT_GT": "DT vs GT"}
PAIR_COLORS = {"DR_DT": "#0072B2", "DR_GT": "#D55E00", "DT_GT": "#009E73"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.2,
    "legend.fontsize": 7.6,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 260,
})


def values(path, pair):
    return np.array([row[path][pair]["cosine"] for row in ROWS], dtype=float)


def style(ax):
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.set_axisbelow(True)


def main():
    if len(ROWS) != 6 or len(ONLINE) != 1500:
        raise RuntimeError("Frozen mediation data are incomplete")
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.45), layout="constrained")
    ax_direction, ax_clip, ax_loss = axes

    # (a) Follow the same pair from raw gradient to actual BF16 model change.
    stages = (
        ("raw_gradient_comparisons", "Raw\ngrad."),
        ("clipped_gradient_comparisons", "Clipped\ngrad."),
        ("stateful_adam_master_update_comparisons", "Adam\nstep"),
        ("stateful_adam_bf16_writeback_comparisons", "BF16\nchange"),
    )
    x = np.arange(len(stages))
    for pair in PAIRS:
        matrix = np.stack([values(path, pair) for path, _ in stages], axis=1)
        mean = matrix.mean(axis=0)
        error = matrix.std(axis=0)
        ax_direction.errorbar(
            x,
            mean,
            yerr=error,
            color=PAIR_COLORS[pair],
            marker="o",
            markersize=3.8,
            linewidth=1.35,
            capsize=2,
            label=PAIR_LABELS[pair],
        )
    ax_direction.set(
        title="(a) Adam aligns gradients",
        ylabel="Cosine similarity",
        xticks=x,
        xticklabels=[label for _, label in stages],
        ylim=(0.25, 1.025),
        yticks=np.arange(0.3, 1.01, 0.1),
    )
    ax_direction.legend(frameon=False, loc="lower right")
    style(ax_direction)

    # (b) Online distribution: use the direct, readable fraction retained.
    clip_values = {
        reduction: np.array([
            row["clip_coefficient"] for row in ONLINE if row["branch"] == reduction
        ])
        for reduction in REDUCTIONS
    }
    box = ax_clip.boxplot(
        [clip_values[reduction] for reduction in REDUCTIONS],
        tick_labels=REDUCTIONS,
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, reduction in zip(box["boxes"], REDUCTIONS):
        patch.set(facecolor=REDUCTION_COLORS[reduction], alpha=0.35, edgecolor=REDUCTION_COLORS[reduction])
    rng = np.random.default_rng(1042)
    for index, reduction in enumerate(REDUCTIONS, 1):
        sample = clip_values[reduction][::10]
        ax_clip.scatter(
            index + rng.uniform(-0.16, 0.16, len(sample)),
            sample,
            s=7,
            color=REDUCTION_COLORS[reduction],
            alpha=0.35,
            linewidths=0,
        )
        clipped = 100 * np.mean(clip_values[reduction] < 1.0)
        ax_clip.text(index, 1.055, f"{clipped:.0f}%", ha="center", va="bottom", fontsize=7.4)
    ax_clip.set(
        title="(b) Clipping strength differs",
        ylabel="Fraction retained",
        ylim=(0, 1.13),
        yticks=(0, 0.25, 0.5, 0.75, 1.0),
    )
    style(ax_clip)

    offsets = {"DR": -0.18, "DT": 0.0, "GT": 0.18}
    # (c) Same-norm SGD exposes the raw direction; negative KL change is better.
    categories = (
        ("adam_stateful", 100, "Adam\n100"),
        ("sgd_norm_matched", 100, "SGD\n100"),
        ("adam_stateful", 250, "Adam\n250"),
        ("sgd_norm_matched", 250, "SGD\n250"),
    )
    ax_loss.axhspan(-11, 0, color="#E8F4EA", alpha=0.55, zorder=0)
    for category_index, (optimizer, step, _) in enumerate(categories):
        selected = [row for row in ROWS if row["snapshot_step"] == step]
        for reduction in REDUCTIONS:
            observed = np.array([
                row["branches"][f"{optimizer}/{reduction}"]["heldout_immediate_change"]
                ["full_vocab_reverse_kl"]["macro"]["after_minus_before"] * 1e3
                for row in selected
            ])
            xpos = category_index + offsets[reduction]
            ax_loss.scatter(
                np.full(len(observed), xpos),
                observed,
                facecolors="none",
                edgecolors=REDUCTION_COLORS[reduction],
                s=17,
                linewidths=0.75,
                alpha=0.75,
            )
            ax_loss.errorbar(
                xpos,
                observed.mean(),
                yerr=observed.std(),
                marker="o",
                markersize=3.2,
                color=REDUCTION_COLORS[reduction],
                capsize=1.8,
                linewidth=0.8,
            )
    ax_loss.axhline(0, color="#777777", linestyle=":", linewidth=0.9)
    ax_loss.text(0.02, 0.03, "lower is better", transform=ax_loss.transAxes, fontsize=7.3, color="#35623B")
    ax_loss.set(
        title="(c) Held-out effect",
        ylabel=r"Held-out KL change ($\times 10^{-3}$)",
        xticks=range(len(categories)),
        xticklabels=[label for _, _, label in categories],
        ylim=(-10.5, 8.0),
        xlim=(-0.45, 3.45),
    )
    style(ax_loss)

    reduction_handles = [
        Line2D([0], [0], color=REDUCTION_COLORS[reduction], marker="o", linestyle="", label=reduction)
        for reduction in REDUCTIONS
    ]
    fig.legend(handles=reduction_handles, loc="outside lower center", ncol=3, frameon=False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped_text = []
    for axis in fig.axes:
        for text in [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.get_xticklabels(), *axis.get_yticklabels()]:
            if not text.get_visible() or not text.get_text():
                continue
            bounds = text.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width + 1 or bounds.y1 > fig.bbox.height + 1:
                clipped_text.append((axis.get_title(), text.get_text()))
    if clipped_text:
        raise RuntimeError(f"Figure text outside canvas: {clipped_text}")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"optimizer_mediation.{extension}")
    plt.close(fig)

    # The moment-reset scale control is supporting evidence, so it is rendered
    # separately for the appendix rather than interrupting the main argument.
    moment_fig, ax_norm = plt.subplots(figsize=(3.45, 2.35), layout="constrained")
    conditions = (
        ("adam_stateful", "Saved\nmoments"),
        ("adam_reset_m", "Reset first\nmoment"),
        ("adam_reset_m_v", "Reset both\nmoments"),
    )
    for condition_index, (condition, _) in enumerate(conditions):
        for reduction in REDUCTIONS:
            observed = np.array([
                row["branches"][f"{condition}/{reduction}"]["master_update_l2"] * 1e3
                for row in ROWS
            ])
            xpos = condition_index + offsets[reduction]
            ax_norm.scatter(
                xpos + np.linspace(-0.035, 0.035, len(observed)),
                observed,
                color=REDUCTION_COLORS[reduction],
                s=13,
                alpha=0.48,
                linewidths=0,
            )
            ax_norm.plot(
                xpos,
                observed.mean(),
                marker="_",
                markersize=9,
                color=REDUCTION_COLORS[reduction],
            )
    ax_norm.set(
        title="Adam memory sets the step size",
        ylabel=r"FP32 step norm ($\times 10^{-3}$)",
        xticks=range(len(conditions)),
        xticklabels=[label for _, label in conditions],
        xlim=(-0.45, 2.45),
    )
    style(ax_norm)
    moment_fig.legend(handles=reduction_handles, loc="outside lower center", ncol=3, frameon=False)
    for extension in ("pdf", "png", "svg"):
        moment_fig.savefig(FIGURES / f"optimizer_mediation_moments.{extension}")
    plt.close(moment_fig)

    manifest = {
        "figure": "optimizer_mediation",
        "panels": 3,
        "appendix_figure": "optimizer_mediation_moments",
        "appendix_panels": 1,
        "fixed_bank_jobs": len(ROWS),
        "online_clip_updates": len(ONLINE),
        "measurements_sha256": hashlib.sha256((DATA / "measurements.json").read_bytes()).hexdigest(),
        "online_clip_sha256": hashlib.sha256((DATA / "online_clip_coefficients.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (DATA / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generated the three-panel main figure and moment-reset appendix figure.")


if __name__ == "__main__":
    main()
