"""Task-level three-seed candidate with student and teacher references."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
FIG = ROOT / "figures/three_seed_20260913"
FIG.mkdir(parents=True, exist_ok=True)

summary = list(csv.DictReader((DATA / "estimated_mean_sd.csv").open()))
capability = json.loads(
    (ROOT / "experiments/aligned_evidence_20260910/capability.json").read_text()
)
teacher_rows = json.loads(
    (ROOT / "experiments/aligned_evidence_20260910/teacher_references.json").read_text()
)

models = ["M-I64-DR", "M-I64-DT", "M-I64-GT"]
labels = {"M-I64-DR": "DR", "M-I64-DT": "DT", "M-I64-GT": "GT"}
colors = {"M-I64-DR": "#D55E00", "M-I64-DT": "#009E73", "M-I64-GT": "#CC79A7"}
markers = {"M-I64-DR": "s", "M-I64-DT": "^", "M-I64-GT": "D"}
offsets = {"M-I64-DR": -0.12, "M-I64-DT": 0.0, "M-I64-GT": 0.12}
domains = ["Math", "Code", "IF", "GPQA"]
titles = {
    "Math": "(a) MATH-500",
    "Code": "(b) LiveCodeBench",
    "IF": "(c) IFBench strict",
    "GPQA": "(d) GPQA (avg@4)",
}
teacher_names = {
    "Math": "teacher_math",
    "Code": "teacher_code",
    "IF": "teacher_if",
    "GPQA": "teacher_science",
}
steps = [50, 100, 250, 500]
positions = np.arange(5)
initial = next(r for r in capability if r["model"] == "Initial")["scores"]
teacher = {
    domain: next(r for r in teacher_rows if r["teacher"] == name).get(
        "current_scorer_score",
        next(r for r in teacher_rows if r["teacher"] == name)["score"],
    )
    for domain, name in teacher_names.items()
}


def record(model, step, domain):
    return next(
        r
        for r in summary
        if r["model"] == model
        and int(r["step"]) == step
        and r["measure"] == domain
    )


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.3,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.05), layout="constrained")

for ax, domain in zip(axes.flat, domains):
    student_score = 100 * initial[domain]
    teacher_score = 100 * teacher[domain]

    # These references answer whether training preserves the student and closes
    # the gap to the routed domain teacher; draw them behind the trajectories.
    ax.axhline(student_score, color="#6F6F6F", ls=":", lw=1.15, zorder=0)
    ax.axhline(teacher_score, color="#4B3F72", ls=(0, (5, 2.5)), lw=1.05, zorder=0)

    lower = [student_score, teacher_score]
    upper = [student_score, teacher_score]
    for model in models:
        rows = [record(model, step, domain) for step in steps]
        means = np.array([float(r["mean_pct"]) for r in rows])
        sds = np.array([float(r["sample_sd_pct"]) for r in rows])
        x = positions[1:] + offsets[model]
        ax.plot(
            np.r_[positions[0], x],
            np.r_[student_score, means],
            color=colors[model],
            marker=markers[model],
            ms=3.4,
            lw=1.25,
            zorder=2,
        )
        ax.errorbar(
            x,
            means,
            yerr=sds,
            fmt="none",
            ecolor=colors[model],
            capsize=1.7,
            elinewidth=0.8,
            zorder=1,
        )
        lower.extend(means - sds)
        upper.extend(means + sds)

    lo, hi = min(lower), max(upper)
    pad = max(1.2, 0.12 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.28, 4.55)
    ax.set_title(titles[domain], loc="left", pad=3)
    ax.set_xticks(positions, ["0", "50", "100", "250", "500"])
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.set_axisbelow(True)

    # Direct labels keep both baselines interpretable in every panel.
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.4)
    ax.text(
        4.49,
        teacher_score,
        f"T {teacher_score:.1f}",
        color="#4B3F72",
        ha="right",
        va="bottom",
        fontsize=6.3,
        bbox=label_box,
        zorder=4,
    )
    ax.text(
        4.49,
        student_score,
        f"S {student_score:.1f}",
        color="#5F5F5F",
        ha="right",
        va="top",
        fontsize=6.3,
        bbox=label_box,
        zorder=4,
    )

# (e) Aggregate score with student and teacher references averaged over domains.
ax = axes[1, 1]
student_mean = 100 * np.mean([initial[d] for d in domains])
teacher_mean = 100 * np.mean([teacher[d] for d in domains])
ax.axhline(student_mean, color="#6F6F6F", ls=":", lw=1.15, zorder=0)
ax.axhline(teacher_mean, color="#4B3F72", ls=(0, (5, 2.5)), lw=1.05, zorder=0)
mean_lower = [student_mean, teacher_mean]
mean_upper = [student_mean, teacher_mean]
for model in models:
    rows = [record(model, step, "mean") for step in steps]
    means = np.array([float(r["mean_pct"]) for r in rows])
    sds = np.array([float(r["sample_sd_pct"]) for r in rows])
    x = positions[1:] + offsets[model]
    ax.plot(
        np.r_[positions[0], x],
        np.r_[student_mean, means],
        color=colors[model],
        marker=markers[model],
        ms=3.4,
        lw=1.25,
        zorder=2,
    )
    ax.errorbar(
        x, means, yerr=sds, fmt="none", ecolor=colors[model],
        capsize=1.7, elinewidth=0.8, zorder=1,
    )
    mean_lower.extend(means - sds)
    mean_upper.extend(means + sds)
lo, hi = min(mean_lower), max(mean_upper)
ax.set_ylim(lo - 0.9, hi + 0.9)
ax.set_xlim(-0.28, 4.55)
ax.set_title("(e) Four-domain mean", loc="left", pad=3)
ax.set_xticks(positions, ["0", "50", "100", "250", "500"])
ax.text(
    4.49, teacher_mean, f"T {teacher_mean:.1f}", color="#4B3F72",
    ha="right", va="bottom", fontsize=6.3, bbox=label_box, zorder=4,
)
ax.text(
    4.49, student_mean, f"S {student_mean:.1f}", color="#5F5F5F",
    ha="right", va="top", fontsize=6.3, bbox=label_box, zorder=4,
)

# (f) Minimum domain change tests whether gains sacrifice another task.
ax = axes[1, 2]
ax.axhline(0, color="#6F6F6F", ls=":", lw=1.15, zorder=0)
worst_lower = [0]
worst_upper = [0]
for model in models:
    rows = [record(model, step, "worst_change") for step in steps]
    means = np.array([float(r["mean_pct"]) for r in rows])
    sds = np.array([float(r["sample_sd_pct"]) for r in rows])
    x = positions[1:] + offsets[model]
    ax.plot(
        np.r_[positions[0], x],
        np.r_[0, means],
        color=colors[model],
        marker=markers[model],
        ms=3.4,
        lw=1.25,
        zorder=2,
    )
    ax.errorbar(
        x, means, yerr=sds, fmt="none", ecolor=colors[model],
        capsize=1.7, elinewidth=0.8, zorder=1,
    )
    worst_lower.extend(means - sds)
    worst_upper.extend(means + sds)
lo, hi = min(worst_lower), max(worst_upper)
ax.set_ylim(lo - 0.6, hi + 0.6)
ax.set_xlim(-0.28, 4.55)
ax.set_title("(f) Worst-domain change", loc="left", pad=3)
ax.set_xticks(positions, ["0", "50", "100", "250", "500"])
ax.text(
    4.49, 0, "No regression", color="#5F5F5F", ha="right", va="top",
    fontsize=6.3, bbox=label_box, zorder=4,
)

for ax in axes.flat:
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.set_axisbelow(True)
for ax in axes[:, 0]:
    ax.set_ylabel("Score (%)")
axes[1, 2].set_ylabel("Change (pp)")
for ax in axes[1, :]:
    ax.set_xlabel("Optimizer updates")

method_handles = [
    Line2D(
        [0],
        [0],
        color=colors[m],
        marker=markers[m],
        lw=1.25,
        ms=3.4,
        label=labels[m],
    )
    for m in models
]
reference_handles = [
    Line2D([0], [0], color="#6F6F6F", ls=":", lw=1.15, label="Initial student"),
    Line2D(
        [0],
        [0],
        color="#4B3F72",
        ls=(0, (5, 2.5)),
        lw=1.05,
        label="Domain teacher",
    ),
]
fig.legend(
    handles=method_handles + reference_handles,
    loc="outside upper center",
    ncol=5,
    frameon=False,
    columnspacing=1.4,
    handlelength=2.2,
)

stem = FIG / "normalization_six_panel"
for ext in ["pdf", "png", "svg"]:
    fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
plt.close(fig)

caption = (
    "GT/DT/DR capability (mean and one sample standard deviation across three seeds). Panels a--d "
    "show task scores with initial-student and domain-teacher references; panel e shows their "
    "four-domain mean with averaged references; panel f shows the worst domain change from "
    "initialization, where zero denotes no regression. Horizontal offsets separate overlap."
)
(DATA / "task_reference_figure_caption.txt").write_text(caption + "\n")
print(caption)
