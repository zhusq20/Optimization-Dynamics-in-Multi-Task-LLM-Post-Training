# Optimization Dynamics in Multi-Task LLM Post-Training

The current manuscript includes the completed experimental results, thirteen
capability evaluations, and eleven figures. The main file is
**`iclr2027_conference.tex` in the repository root**.

## Compile the paper

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

All required sections, bibliography/style files, generated table sources in
`experiments/aligned_evidence_20260910/`, and eleven PDF figures in
`figures/aligned_evidence_20260910/` are tracked. No raw experiment data or Python
plotting step is required to compile. The validated manuscript has 16 pages.
The output PDF and LaTeX auxiliary files are local build products.

## Overleaf synchronization

This repository is linked to
[the Overleaf project](https://www.overleaf.com/project/6a8a831a5a9b43dfa3f0c368).
GitHub and Overleaf do not synchronize automatically. In Overleaf, open
**Integrations → GitHub**, finish **Continue** if a resolved merge is pending,
and pull the GitHub changes. Confirm the main document is the root
`iclr2027_conference.tex`, then recompile from scratch.

The main file contains the source marker
`Overleaf source revision: completed-results-20260911`.
The abstract contains “Thirteen capability evaluations” and the first figure
is the capability-trajectory chart. If these are absent, the project is still
using an earlier source snapshot or another main document.

The `overleaf-2026-09-11-0155` branch contained the old results-pending draft.
It is already merged into `main`; that merge preserves the completed results.
Do not use that old branch as the current paper source.

## Research data and historical outputs

The complete snapshot before the Overleaf cleanup is preserved in
[`archive/research-data-before-overleaf-cleanup-20260911`](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/archive/research-data-before-overleaf-cleanup-20260911),
at commit `2736c5bdb53623cfb73974f4ba8a8bcd7d157dc3`.
It contains the raw measurements, older figure packages, backups, and build
outputs removed from tracking on `main`. Existing local copies are retained.

For a separate checkout with all evidence and reproduction inputs:

```bash
git fetch origin
git worktree add --detach ../mopd-research-data origin/archive/research-data-before-overleaf-cleanup-20260911
```

Use that checkout for the full figure-generation and evidence-audit workflows.
See [the experiment guide](experiments/README.md) for details.
