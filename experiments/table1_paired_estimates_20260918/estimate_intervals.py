"""Estimate paired contrasts from reported scores and archived interval widths.

No current question outcomes are reconstructed. The Gaussian parametric bootstrap
is conditional on variance ratios calibrated on older paired-question intervals.
Run with the bundled Python runtime; only NumPy is required.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REF = "origin/archive/before-compile-only-20260914"
DOMAINS = ["Math", "Code", "IF", "GPQA"]
N = np.array([500, 1055, 300, 198])
Z975 = 1.959963984540054
B = 100_000
SEED = 1042


def archived(name):
    path = f"experiments/aligned_evidence_20260910/{name}"
    payload = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=ROOT)
    (OUT / ("calibration_" + name)).write_bytes(payload)
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def fmt(value):
    return str(Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))


def signed(value):
    return ("+" if value >= 0 else "-") + fmt(abs(value))


history, history_hash = archived("paired_comparisons.json")
suites, suites_hash = archived("capability.json")
old_scores = {(r["model"], r["step"]): r["scores"] for r in suites}
latest = {}
for r in history:
    if not r["right"].startswith("M-"):
        continue
    if r["left"] != "Initial" and not r["left"].startswith("M-"):
        continue
    # One latest common checkpoint per joint-training pair and domain; do not
    # treat repeated checkpoints or overlapping questions as independent data.
    key = (r["left"], r["right"], r["domain"])
    if key not in latest or r["right_step"] > latest[key]["right_step"]:
        latest[key] = r

calibration = []
ratios = {}
for group in ["initial", "method"]:
    ratios[group] = []
    for domain in DOMAINS:
        vals = []
        for r in latest.values():
            if r["domain"] != domain or (r["left"] == "Initial") != (group == "initial"):
                continue
            a = old_scores[(r["left"], r["left_step"])][domain]
            b = old_scores[(r["right"], r["right_step"])][domain]
            se = (r["hi_pp"] - r["lo_pp"]) / (100 * 2 * Z975)
            ratio = r["prompts"] * se**2 / (a * (1-a) + b * (1-b))
            vals.append(ratio)
            calibration.append(dict(r, group=group, variance_ratio=ratio))
        ratios[group].append(float(np.median(vals)))

source = ROOT / "experiments/completed_results_20260917/endpoint_table.tex"
scores = {}
optimizer = "Adam"
for line in source.read_text().splitlines():
    if r"\textit{SGD}" in line:
        optimizer = "SGD"
    if line.startswith("Initial student") or re.match(r"^(PG|I16|I64)-", line):
        cells = line.split("&")
        key = "Initial student" if line.startswith("Initial student") else f"{cells[0].strip()} / {optimizer}"
        scores[key] = [float(x.strip()) for x in cells[1:5]]

half_draws = np.random.default_rng(SEED).standard_normal((B // 2, 4))
normal_draws = np.concatenate([half_draws, -half_draws])


def estimate(a, b):
    """A minus B in pp; variance floors enforce the question-score lattice."""
    p = np.array(scores[a]) / 100
    q = np.array(scores[b]) / 100
    delta_pp = [Decimal(str(x))-Decimal(str(y)) for x, y in zip(scores[a], scores[b])]
    delta = np.array([float(x/100) for x in delta_pp])
    group = "initial" if "Initial student" in [a, b] else "method"
    independent_var = (p*(1-p) + q*(1-q)) / N
    # For binary correctness the minimum difference variance is |delta|-delta².
    # GPQA question means lie on a 1/4 lattice, so its difference has that lattice.
    spacing = np.array([1., 1., 1., .25])
    lower = np.floor(delta / spacing) * spacing
    minimum_var = np.maximum(0, (delta-lower)*(lower+spacing-delta)) / N
    # Largest feasible covariance-negative Bernoulli variance; for a [0,1]
    # question mean this remains a conservative upper bound.
    maximum_var = (p + q - 2*np.maximum(0, p+q-1) - delta**2) / N
    scenarios = {}
    for name, factor in [("calibrated", np.array(ratios[group])),
                         ("weaker_pairing", np.minimum(1, 2*np.array(ratios[group]))),
                         ("independent", np.ones(4))]:
        variance = np.minimum(maximum_var, np.maximum(minimum_var, independent_var*factor))
        if a == b:
            variance[:] = 0
        draws = 100*(delta + normal_draws*np.sqrt(variance))
        mean_draws = draws.mean(axis=1)
        scenarios[name] = {
            "domain_variance_pp2": (10_000*variance).tolist(),
            "domain_ci95_pp": np.quantile(draws, [.025, .975], axis=0).T.tolist(),
            "mean_ci95_pp": np.quantile(mean_draws, [.025, .975]).tolist(),
            "mean_se_pp": float(100*np.sqrt(variance.sum())/4),
            "variance_floor_applied": (independent_var*factor < minimum_var - 1e-15).tolist(),
        }
        assert np.all(variance >= -1e-15)
    return {"a": a, "b": b, "calibration_group": group,
            "domain_delta_pp": [float(x) for x in delta_pp],
            "mean_delta_pp": float(sum(delta_pp)/4), "scenarios": scenarios}


contrasts = {f"{a} minus {b}": estimate(a, b) for a in scores for b in scores}


def interval(r, scenario="calibrated"):
    lo, hi = r["scenarios"][scenario]["mean_ci95_pp"]
    return f"[{lo:.2f}, {hi:.2f}]"


baseline = "PG-DR / Adam"
lines = [
    "% Estimated paired intervals: experiments/table1_paired_estimates_20260918/estimate_intervals.py.",
    r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
    r"\caption{\textbf{Endpoint gains coexist with domain-level trade-offs.} Joint-training scores (\%); Mean weights domains equally. $\Delta$ is the mean difference from PG-DR/Adam (percentage points); brackets give estimated paired 95\% intervals$^{\dagger}$.}",
    r"\label{tab:completed_endpoints}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
    r"Configuration & MATH-500 & Code & IF & GPQA & Mean & $\Delta$ [95\% interval]$^{\dagger}$ \\",
    r"\midrule",
]
previous = None
for name, values in scores.items():
    opt = name.split(" / ")[-1] if name != "Initial student" else None
    if opt != previous:
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\textit{{{opt}}}}} \\"]
        previous = opt
    r = contrasts[f"{name} minus {baseline}"]
    cell = "Reference" if name == baseline else f"${signed(r['mean_delta_pp'])}$ {interval(r)}"
    mean = sum(Decimal(str(v)) for v in values) / 4
    lines.append(" & ".join([name.split(" / ")[0]] + [f"{v:.2f}" for v in values] + [fmt(mean), cell]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{2pt}",
          r"{\footnotesize $\dagger$ Estimation procedure and sensitivity checks: Appendix~\ref{app:capability_uncertainty}.}",
          r"\end{table}"]
source.write_text("\n".join(lines) + "\n")

selected = []
for family, opt in [("I64", "Adam"), ("PG", "Adam"), ("PG", "SGD")]:
    for a, b in [("DT", "DR"), ("GT", "DR"), ("DT", "GT")]:
        selected.append((f"{family}/{opt}: {a} $-$ {b}", f"{family}-{a} / {opt}", f"{family}-{b} / {opt}"))
for rule in ["DR", "DT", "GT"]:
    selected.append((f"PG-{rule}: SGD $-$ Adam", f"PG-{rule} / SGD", f"PG-{rule} / Adam"))
selected += [("Adam/DR: I64 $-$ PG", "I64-DR / Adam", baseline),
             ("Adam/DR: I64 $-$ I16", "I64-DR / Adam", "I16-DR / Adam")]
lines = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Small endpoint contrasts remain uncertain under the calibrated model.} Mean differences (percentage points) and estimated paired 95\% intervals; $2\kappa$ weakens the assumed variance reduction, and $\kappa=1$ uses the independent Bernoulli reference. These are pointwise, model-based intervals.}",
         r"\label{tab:paired_contrasts}", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Contrast & $\Delta$ & Calibrated & $\min(2\kappa,1)$ & $\kappa=1$ \\", r"\midrule"]
for label, a, b in selected:
    r = contrasts[f"{a} minus {b}"]
    lines.append(" & ".join([label, f"${signed(r['mean_delta_pp'])}$"] + [interval(r, s) for s in ["calibrated", "weaker_pairing", "independent"]]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "contrast_table.tex").write_text("\n".join(lines) + "\n")

output = {
    "method": "Gaussian parametric bootstrap using archived paired-interval variance ratios; not empirical current-question bootstrap",
    "replicates": B, "seed": SEED, "n_question_clusters": N.tolist(),
    "archive_commit": subprocess.check_output(["git", "rev-parse", REF], cwd=ROOT, text=True).strip(),
    "archive_sha256": {"paired_comparisons.json": history_hash, "capability.json": suites_hash},
    "calibration": calibration, "variance_ratios": ratios, "reported_scores_pp": scores,
    "contrasts": contrasts,
}
(OUT / "estimated_intervals.json").write_text(json.dumps(output, indent=2) + "\n")
print("Variance ratios:", ratios)
for name in scores:
    if name == "Initial student": continue
    r = contrasts[f"{name} minus Initial student"]
    print(name, "minus initial:", f"{r['mean_delta_pp']:+.2f}", interval(r), "independent", interval(r, "independent"))
for label, a, b in selected:
    r = contrasts[f"{a} minus {b}"]
    print(label, f"{r['mean_delta_pp']:+.2f}", interval(r), "weaker", interval(r, "weaker_pairing"))
