# Optimization Dynamics in Multi-Task LLM Post-Training

The current manuscript includes thirteen capability evaluations and eleven
figures. Compile the root **`iclr2027_conference.tex`**:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

All current sections, bibliography/style files, generated table sources, and
PDF figures are present. The validated manuscript has 16 pages. Its abstract
contains “Thirteen capability evaluations”; Figure 1 shows capability
trajectories. The source marker is
`Overleaf source revision: completed-results-20260911`.

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
