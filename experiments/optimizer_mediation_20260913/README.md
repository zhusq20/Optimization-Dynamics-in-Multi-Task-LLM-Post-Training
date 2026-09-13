# Optimizer-mediation evidence for Section 3

This frozen bundle contains six fixed-bank interventions: three response banks at each of the shared M-PG update-100 and update-250 states. GT, DT, and DR use identical responses, routed teacher targets, student parameters, and Adam state within a job.

`measurements.json` retains raw- and clipped-gradient comparisons, saved-state Adam FP32 and BF16 comparisons, moment-reset controls, norm-matched SGD controls, and immediate four-domain held-out loss changes. `online_clip_coefficients.json` contains the exact coefficient reconstructed for every update of the aligned GT, DT, and DR runs. `source_manifest.json` records source hashes; `figure_manifest.json` records the inputs used for the three-panel Section 3 figure and the moment-reset appendix figure.

Regenerate both figures with:

```bash
python experiments/plot_optimizer_mediation.py
```

Verify the frozen measurements and generated PDF with:

```bash
python experiments/verify_optimizer_mediation.py
```

The source-workspace export is reproducible with `python experiments/export_optimizer_mediation.py`; regeneration of the paper figure does not require the source workspace because the compact records are frozen here.
