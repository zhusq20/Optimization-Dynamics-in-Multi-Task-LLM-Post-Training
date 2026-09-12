# Optimization Dynamics in Multi-Task LLM Post-Training

The manuscript studies three questions: loss averaging and capability balance;
parameter-update concentration and teacher overlap; and vocabulary supervision
in single-teacher and joint distillation.

The September 12 revision audits existing results, including three newly located
complete capability evaluations, paired joint geometry through update 250, and
six fixed-batch averaging comparisons. The evidence inventory now contains
16 capability suites (27,520 responses). Comparisons use common completed
checkpoints; incomplete experiments are replaced by visible author annotations.
Teacher-gradient diagnostics and numerical controls are in the appendix.
All compound figures use a three-column layout.

Compile the root `iclr2027_conference.tex`:

```bash
python experiments/plot_aligned_evidence.py
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
python experiments/verify_aligned_evidence.py
```

Current source marker: `evidence-complete-comparisons-20260912`.

- [Current result audit (中文)](experiments/RESULTS_AUDIT_20260912_zh.md)
- [Reference verification and related-paper observations (中文)](experiments/REFERENCE_AUDIT_20260912_zh.md)
- [Three-question scope (中文)](MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md)
- [Evidence and reproduction guide](experiments/aligned_evidence_20260910/README_zh.md)

The following sections record the September 11 synchronization history.

## September 11 Overleaf synchronization repair

The project is [here](https://www.overleaf.com/project/6a8a831a5a9b43dfa3f0c368).
In **Integrations → GitHub**, click **Continue** after the repair merge, then
pull if prompted. Select the root main document and recompile from scratch.

The `overleaf-2026-09-11-0312` snapshot at `fdebceb` exactly matches the older
`287eaee` snapshot plus the first 300 file changes returned by GitHub for
`287eaee...2736c5b`. That response excluded the manuscript and every figure.
This reproduces why a merged Git history still left the old paper in Overleaf.
GitHub documents the [300-file comparison limit](https://docs.github.com/en/rest/commits/commits#compare-two-commits).

This repair starts from the actual Overleaf snapshot and replaces its current
paper files with the completed-results version from `be8128e`. The merge uses
the Overleaf snapshot as its first parent and the previous `main` as its second
parent. Its synchronization difference contains fewer than 100 changed files,
including all eleven figure PDFs and all revised paper sections. The two stale
compiled paper PDFs and 45 large archived data files are removed from tracking.

Some historical files are temporarily retained to keep this recovery difference
small. The full cleanup must be resumed in separate synchronization steps after
the corrected paper is visible in Overleaf. `.gitignore` remains in place; it
does not automatically untrack files or filter Overleaf-side uploads.

## Preserved data and compact manuscript snapshot

The complete original research snapshot remains on
[`archive/research-data-before-overleaf-cleanup-20260911`](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/archive/research-data-before-overleaf-cleanup-20260911).
The compact 78-file manuscript snapshot is preserved at
[`be8128e`](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/be8128e54897588e1ed652e7f73a07290bc27762).
Local evidence files are retained. See [the experiment guide](experiments/README.md)
for reproduction instructions using the complete archive checkout.
