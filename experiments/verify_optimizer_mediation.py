"""Verify the frozen optimizer-mediation evidence and manuscript claims."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/optimizer_mediation_20260913"
FIGURES = (
    ROOT / "figures/optimizer_mediation_20260913/optimizer_mediation.pdf",
    ROOT / "figures/optimizer_mediation_20260913/optimizer_mediation_moments.pdf",
)
REDUCTIONS = ("DR", "DT", "GT")
PAIRS = ("DR_DT", "DR_GT", "DT_GT")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(observed: float, expected: float, tolerance: float, claim: str) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(f"{claim}: observed {observed}, expected {expected}")


def main() -> None:
    rows = json.loads((DATA / "measurements.json").read_text())["records"]
    online = json.loads((DATA / "online_clip_coefficients.json").read_text())["records"]
    manifest = json.loads((DATA / "figure_manifest.json").read_text())

    assert len(rows) == 6
    assert Counter(row["snapshot_step"] for row in rows) == {100: 3, 250: 3}
    expected_branches = {
        f"{condition}/{reduction}"
        for condition in ("adam_stateful", "adam_reset_m", "adam_reset_m_v", "sgd_norm_matched")
        for reduction in REDUCTIONS
    }
    assert all(set(row["branches"]) == expected_branches for row in rows)

    maximum_angle_residual = max(
        abs(
            row["raw_gradient_comparisons"][pair]["cosine"]
            - row["clipped_gradient_comparisons"][pair]["cosine"]
        )
        for row in rows
        for pair in PAIRS
    )
    if maximum_angle_residual > 1e-12:
        raise AssertionError("Scalar clipping changed a recorded gradient cosine")

    raw_means = {
        pair: float(np.mean([row["raw_gradient_comparisons"][pair]["cosine"] for row in rows]))
        for pair in PAIRS
    }
    adam_means = {
        pair: float(np.mean([
            row["stateful_adam_master_update_comparisons"][pair]["cosine"] for row in rows
        ]))
        for pair in PAIRS
    }
    for pair, expected in {"DR_DT": 0.680, "DR_GT": 0.483, "DT_GT": 0.822}.items():
        close(raw_means[pair], expected, 5e-4, f"raw cosine {pair}")
    for pair, expected in {"DR_DT": 0.960, "DR_GT": 0.939, "DT_GT": 0.979}.items():
        close(adam_means[pair], expected, 5e-4, f"Adam cosine {pair}")

    assert len(online) == 1500
    assert Counter(row["branch"] for row in online) == {reduction: 500 for reduction in REDUCTIONS}
    clipped_counts = Counter()
    clip_medians = {}
    for reduction in REDUCTIONS:
        coefficients = np.array([
            row["clip_coefficient"] for row in online if row["branch"] == reduction
        ])
        clipped_counts[reduction] = int(np.sum(coefficients < 1.0))
        clip_medians[reduction] = float(np.median(coefficients))
    assert clipped_counts == {"DR": 500, "DT": 231, "GT": 75}
    close(clip_medians["DR"], 0.093, 5e-4, "DR median fraction retained")
    close(clip_medians["DT"], 1.0, 1e-12, "DT median fraction retained")
    close(clip_medians["GT"], 1.0, 1e-12, "GT median fraction retained")

    maximum_sgd_norm_error = max(
        abs(
            row["branches"][f"sgd_norm_matched/{reduction}"]["master_update_l2"]
            / row["branches"][f"adam_stateful/{reduction}"]["master_update_l2"]
            - 1.0
        )
        for row in rows
        for reduction in REDUCTIONS
    )
    if maximum_sgd_norm_error > 1e-3:
        raise AssertionError("An SGD control is not norm-matched to its Adam step")

    adam_heldout_means = {
        reduction: float(np.mean([
            row["branches"][f"adam_stateful/{reduction}"]["heldout_immediate_change"]
            ["full_vocab_reverse_kl"]["macro"]["after_minus_before"]
            for row in rows
        ]))
        for reduction in REDUCTIONS
    }
    adam_heldout_span = max(adam_heldout_means.values()) - min(adam_heldout_means.values())
    close(adam_heldout_span, 2.9e-5, 5e-7, "saved-Adam held-out mean span")

    sgd_means_by_step = {
        step: {
            reduction: float(np.mean([
                row["branches"][f"sgd_norm_matched/{reduction}"]["heldout_immediate_change"]
                ["full_vocab_reverse_kl"]["macro"]["after_minus_before"]
                for row in rows
                if row["snapshot_step"] == step
            ]))
            for reduction in REDUCTIONS
        }
        for step in (100, 250)
    }
    assert all(value < 0 for value in sgd_means_by_step[100].values())
    assert all(value > 0 for value in sgd_means_by_step[250].values())

    assert manifest["measurements_sha256"] == sha256(DATA / "measurements.json")
    assert manifest["online_clip_sha256"] == sha256(DATA / "online_clip_coefficients.json")
    assert manifest["script_sha256"] == sha256(ROOT / "experiments/plot_optimizer_mediation.py")
    for figure in FIGURES:
        if not figure.is_file() or figure.stat().st_size < 10_000:
            raise AssertionError(f"Generated vector figure is missing or unexpectedly small: {figure.name}")

    report = {
        "status": "pass",
        "fixed_banks": len(rows),
        "online_updates": len(online),
        "maximum_clipping_cosine_residual": maximum_angle_residual,
        "maximum_sgd_relative_norm_error": maximum_sgd_norm_error,
        "raw_cosine_means": raw_means,
        "stateful_adam_cosine_means": adam_means,
        "online_clipped_counts": dict(clipped_counts),
        "online_clip_coefficient_medians": clip_medians,
        "stateful_adam_heldout_mean_span": adam_heldout_span,
        "sgd_heldout_means_by_step": sgd_means_by_step,
        "figure_pdf_sha256": {figure.name: sha256(figure) for figure in FIGURES},
    }
    (DATA / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Optimizer-mediation verification passed.")


if __name__ == "__main__":
    main()
