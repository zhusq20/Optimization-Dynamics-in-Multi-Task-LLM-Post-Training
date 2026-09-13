"""Freeze the completed optimizer-mediation experiment into the paper repository."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "slime_opd_geometry/local/optimizer_mediation_20260913"
OUTPUT = ROOT / "experiments/optimizer_mediation_20260913"
REDUCTIONS = ("DR", "DT", "GT")
OPTIMIZERS = ("adam_stateful", "adam_reset_m", "adam_reset_m_v", "sgd_norm_matched")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted((SOURCE / "results").glob("mediation-mpg*-bank*/measurements.json"))
    if len(files) != 6:
        raise RuntimeError(f"Expected six completed measurements, found {len(files)}")
    records = []
    sources = []
    expected_branches = {
        f"{optimizer}/{reduction}" for optimizer in OPTIMIZERS for reduction in REDUCTIONS
    }
    for path in files:
        row = json.loads(path.read_text())
        if row.get("status") != "complete" or set(row["branches"]) != expected_branches:
            raise RuntimeError(f"Incomplete branch population in {path}")
        for pair, raw in row["raw_gradient_comparisons"].items():
            clipped = row["clipped_gradient_comparisons"][pair]
            if abs(raw["cosine"] - clipped["cosine"]) > 1e-12:
                raise RuntimeError(f"Clipping changed the gradient angle in {path}: {pair}")
        for reduction in REDUCTIONS:
            ratio = row["branches"][f"sgd_norm_matched/{reduction}"]["metrics"][
                "actual_to_target_master_update_l2_ratio"
            ]
            if abs(ratio - 1.0) > 1e-3:
                raise RuntimeError(f"SGD norm match failed in {path}: {reduction}")
        records.append({
            "job": row["job"],
            "snapshot_step": row["snapshot_step"],
            "bank_seed": int(row["job"].rsplit("bank", 1)[1]),
            "clip_grad": row["clip_grad"],
            "clip_coefficients": row["clip_coefficients"],
            "raw_gradient_comparisons": row["raw_gradient_comparisons"],
            "clipped_gradient_comparisons": row["clipped_gradient_comparisons"],
            "stateful_adam_master_update_comparisons": row[
                "stateful_adam_master_update_comparisons"
            ],
            "stateful_adam_bf16_writeback_comparisons": row[
                "stateful_adam_bf16_writeback_comparisons"
            ],
            "branches": {
                branch: {
                    "clip_coefficient": value["metrics"]["clip_coefficient"],
                    "gradient_l2": value["metrics"]["gradient_l2_before_clipping"],
                    "master_update_l2": value["metrics"]["master_update"]["l2"],
                    "bf16_writeback_l2": value["metrics"]["bf16_writeback"]["l2"],
                    "heldout_immediate_change": value["heldout_immediate_change"],
                }
                for branch, value in row["branches"].items()
            },
        })
        sources.append({
            "path": str(path.relative_to(ROOT.parent)),
            "sha256": sha256(path),
            "job": row["job"],
        })

    online_path = SOURCE / "online_clip_coefficients.jsonl"
    online = [json.loads(line) for line in online_path.read_text().splitlines()]
    if len(online) != 1500:
        raise RuntimeError(f"Expected 1500 online clip records, found {len(online)}")
    for reduction in REDUCTIONS:
        updates = sorted(row["update"] for row in online if row["branch"] == reduction)
        if updates != list(range(1, 501)):
            raise RuntimeError(f"Online clip series is incomplete for {reduction}")

    write_json(OUTPUT / "measurements.json", {"records": records})
    write_json(OUTPUT / "online_clip_coefficients.json", {"records": online})
    write_json(OUTPUT / "source_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "experiment": "fixed-bank GT/DT/DR optimizer mediation",
        "jobs": len(records),
        "virtual_branches": len(records) * len(expected_branches),
        "online_updates": len(online),
        "sources": sources + [{
            "path": str(online_path.relative_to(ROOT.parent)),
            "sha256": sha256(online_path),
            "records": len(online),
        }],
        "export_script_sha256": sha256(Path(__file__)),
    })
    print(f"Exported {len(records)} fixed-bank jobs and {len(online)} online clip records.")


if __name__ == "__main__":
    main()
