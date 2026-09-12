"""Freeze completed September 11 follow-ups without running a model or probe."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "aligned_evidence_20260910"
FOLLOWUP = "local/aligned_followup_20260911_sn4622129202"
RUNS = {
    "M-PG": "outputs/mopd_qwen3_aligned_m_pg_20260909/m-pg-s42-g123",
    "M-I64-DR": "outputs/mopd_qwen3_aligned_20260909_sn4622128200/m-intersection64-dr-s42",
}


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    src = args.source.resolve()
    manifest = json.loads((OUT / "manifest.json").read_text())
    sources = manifest["sources"]

    def freeze(rel, name):
        data = (src / rel).read_bytes()
        target = OUT / "raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        frozen = str(target.relative_to(OUT))
        sources[:] = [r for r in sources if r.get("frozen") != frozen]
        sources.append({"source": rel, "frozen": frozen,
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return data

    geometry = []
    for model, run in RUNS.items():
        name = f"{model}_paper_20260912.jsonl"
        raw = freeze(run + "/paper/measurements.jsonl", name)
        rows = [json.loads(line) for line in raw.splitlines()]
        selected = [r for r in rows if r.get("step") == 250 and r.get("layer") == "all"]
        assert len(selected) == 5
        assert {r["quantity"] for r in selected} == {
            "raw_gradient", "clipped_gradient", "update", "delta_fp32", "delta_bf16"}
        assert all(r["metrics"]["parameters"] == 1720574976 for r in selected)
        assert all(r["reduction"] == "domain_response" for r in selected)
        geometry.extend({"model": model, **r} for r in selected)
    dump(OUT / "online_geometry_20260912.json", {
        "scope": "Complete paired joint PG/I64 update-250 geometry; original frozen data retained.",
        "geometry": geometry,
        "sources": [r for r in sources if r["frozen"].endswith("_paper_20260912.jsonl")]})

    normalization = []
    for step in [100, 250]:
        states = set()
        for seed in [1042, 1043, 1044]:
            tag = f"normalization-mpg{step}-bank{seed}"
            rel = FOLLOWUP + "/results/" + tag + "/"
            files = {}
            for name in ["run_complete.json", "measurements.json", "probe_manifest.json", "banks.json", "weights.json"]:
                files[name] = json.loads(freeze(rel + name, "followup/" + tag + "/" + name))
            measured = files["measurements.json"]
            assert files["run_complete.json"]["status"] == measured["status"] == "complete"
            assert measured["snapshot_step"] == step and measured["loss"] == "topk_intersection"
            assert set(measured["branches"]) == {"zero_gradient", "DR", "DT", "GT"}
            assert set(measured["gradient_comparisons"]) == {"DR_DT", "DR_GT", "DT_GT"}
            assert len(files["banks.json"]["train"]) == 32
            assert len(files["banks.json"]["heldout"]) == 4
            assert set(measured["decomposition"]) == {"math", "code", "if", "science"}
            assert all(len(r["lengths"]) == 8 for r in measured["decomposition"].values())
            states.add(measured["snapshot_ordered_state_sha256"])
            normalization.append({"step": step, "bank_seed": seed,
                                  "gradient_comparisons": measured["gradient_comparisons"],
                                  "max_identity_relative_residual": max(r["relative_residual"] for r in measured["decomposition"].values()),
                                  "source": "raw/followup/" + tag + "/measurements.json"})
        assert len(states) == 1
    dump(OUT / "normalization_fixed_batch_20260912.json", {"records": normalization})
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Updates & Batch & DR--DT & DR--GT & DT--GT \\", r"\midrule"]
    for row in normalization:
        values = [row["gradient_comparisons"][pair]["cosine"] for pair in ["DR_DT", "DR_GT", "DT_GT"]]
        lines.append(f"{row['step']} & {row['bank_seed'] - 1041} & " + " & ".join(f"{v:.4f}" for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "normalization_fixed_batch_table.tex").write_text("\n".join(lines) + "\n")

    complete_local = []
    for tag in ["density-mpg100-long-bank1042", "mechanism-i64dr100", "mechanism-spg500"]:
        rel = FOLLOWUP + "/results/" + tag + "/"
        marker = json.loads((src / rel / "run_complete.json").read_text())
        assert marker["status"] == "complete"
        for name, digest in marker["artifacts"].items():
            assert hashlib.sha256((src / rel / name).read_bytes()).hexdigest() == digest
        for name in ["run_complete.json", "measure_complete.json", "prepare_complete.json", "measurements.jsonl",
                     "config.json", "coverage.json", "input_manifest.json", "student_verified.json"]:
            freeze(rel + name, "followup/" + tag + "/" + name)
        rows = [json.loads(line) for line in (src / rel / "measurements.jsonl").read_text().splitlines()]
        config = json.loads((src / rel / "config.json").read_text())
        assert sorted({r["bank"] for r in rows if isinstance(r.get("bank"), int)}) == [0, 1, 2, 3]
        selected = [r for r in rows if r["kind"] == "optimizer" and r.get("mode") == "history"
                    and r["loss"] != "zero_gradient"]
        assert len(selected) == 32
        complete_local.append({"experiment": tag, "config": config,
                               "record_counts": dict(Counter(r["kind"] for r in rows)),
                               "optimizer": selected,
                               "source": "raw/followup/" + tag + "/measurements.jsonl"})
    dump(OUT / "completed_local_followup_20260912.json", {"experiments": complete_local})
    for name in ["normalization_probe.py", "bf16_local_probe.py", "mechanism/prepare.py", "mechanism/measure.py",
                 "slime_plugins/mopd/loss.py", "slime_plugins/mopd/topk.py", "slime_plugins/mopd/paper_diagnostics.py"]:
        freeze(FOLLOWUP + "/src/" + name, "followup/src/" + name)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["refresh_scope"] = "Complete capability, paired joint update-250 geometry, and completed fixed-batch follow-ups; original online and local records retained."
    dump(OUT / "manifest.json", manifest)
    print(json.dumps({"new_geometry_records": len(geometry), "normalization_banks": len(normalization),
                      "complete_local_experiments": len(complete_local), "manifest_sources": len(sources)}))


if __name__ == "__main__":
    main()
