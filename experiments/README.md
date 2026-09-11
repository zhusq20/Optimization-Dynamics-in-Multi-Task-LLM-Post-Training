# MOPD experiment figures

> **Overleaf cleanup (2026-09-11):** `main` keeps the current manuscript,
> generated tables, PDF figures, summaries, and scripts. Raw inputs, older
> galleries, and compiled paper copies are available in the
> [complete archive snapshot](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/archive/research-data-before-overleaf-cleanup-20260911).
> Run the regeneration/audit commands below from that archive checkout; links
> to removed raw data and older packages refer to paths in that snapshot.
> Building the current paper only requires `latexmk`; see the
> [root guide](../README.md).

The current manuscript is organized by the author's [three original questions](../MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md): loss averaging and capability balance; update sparsity, teacher overlap, and teacher-distribution distance; supervision density and its sparsity/capability comparison.

The [aligned evidence guide (中文)](aligned_evidence_20260910/README_zh.md) maps the **5 main figures + 6 appendix figures** to those questions. Thirteen completed capability suites are retained. Main figures now show the three normalization branches, joint PG/I64 cumulative sparsity, teacher-pair gradient overlap, teacher JS versus selection distance, and local update sparsity across supervision choices.

Single-teacher online parameter geometry and teacher-specific optimizer-step overlaps are missing from this aligned snapshot. The paper states those gaps directly. Available teacher gradient comparisons remain supporting evidence and are not labeled as optimizer-step subnetworks. Adam-state and input controls are in the appendix; action resampling and RL-regularization theory are outside the compiled paper.

The [September 11 inventory](EXPERIMENT_INVENTORY_20260911_zh.md) retains the earlier experiment accounting. Its historical figure priorities are superseded by the [current roadmap](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md). The [remaining comparisons](NEXT_EXPERIMENTS_20260908_zh.md) are restricted to the three questions.

Regenerate the current paper figures on CPU:

```bash
python experiments/plot_aligned_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
python experiments/verify_aligned_evidence.py
```

## Archived September 8 figure package

The package below is retained for provenance. Its mixed initialization/objective histories and previous local probes are superseded for the current manuscript; use the aligned bundle above for current claims.

Start with the [September 8 question-led gallery](../figures/paper_curves_20260908/index.html), [nine-figure main PDF](../figures/paper_curves_20260908/main_figures.pdf), or [raw W&B training/evaluation curve explorer](../figures/paper_curves_20260908/raw_curves.html).

The September 8 sequence starts with four domain evaluation trajectories, then raw teacher-distance and token-share curves, and finally three focused local mechanism charts. The full set has **25 line charts, 4 point charts and 1 teacher-pair heatmap**. Each PDF/PNG/SVG contains one chart; the gallery places its question and reading guide outside the image.

- [Design and paper references (中文)](paper_curves_20260908/DESIGN_zh.md): how Open-MOPD Fig.1/3/4 and MOPD Fig.2/3 inform this layout.
- [Data/reproduction guide](paper_curves_20260908/README_zh.md), [caption drafts](paper_curves_20260908/FIGURE_CAPTIONS.md), [all figures PDF](../figures/paper_curves_20260908/all_figures.pdf).
- [Raw curve CSV](paper_curves_20260908/raw_curve_values.csv), [evaluation attempts](paper_curves_20260908/evaluation_attempts.csv), [availability](paper_curves_20260908/evaluation_availability.csv), [validation](paper_curves_20260908/validation_report.json).
- [September 8 roadmap (中文)](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md), [remaining scope](NEXT_EXPERIMENTS_20260908_zh.md).

The frozen local scalar mirrors contain the values passed to `wandb.log`; the explorer retains every available point and defaults to no smoothing. `train/step` is a gradient-microbatch clock, not an optimizer update. Policy entropy is absent. Mixed Student64-to-I64 histories remain explicit; their step-100 checkpoints were trained with Student64. Repeated evaluation attempts are not independent training seeds.

Regenerate on CPU from the portable data already in this repository:

```bash
python experiments/plot_paper_figures.py
```

To deliberately refresh from existing local run artifacts, run `python experiments/export_training_curves.py` first. This does not start training or evaluation.

The [31-probe source package](local_probe_results_20260908/README_zh.md), its 15 previous charts, and the [earlier evidence snapshot](figure_audit_20260908/README_zh.md) are retained for provenance. Their previous heatmap-led main-figure recommendations are superseded. Local saved-Adam BF16 proposals remain distinct from actual online updates and capability measurements.
