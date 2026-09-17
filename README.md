# Optimization Dynamics in Multi-Task LLM Post-Training

The default branch contains only the files needed to compile the current paper,
plus this README and `.gitignore`. Section 3 uses a six-panel figure in two
rows of three; Sections 4 and 5 each use three panels in one row. Supporting
views remain in the appendix, including normalization aggregates. The appendix
uses measured capability and cumulative-geometry trajectories through update
500, update-500 paired intervals, and a compact local-concentration table;
controls already shown in the main figures are referenced rather than repeated. Normalization
capability curves retain their existing means, sample standard deviations, and error
bars. On September 17, 2026, the authors confirmed that their seed-43/44 series
are real experimental data. The source confirmation is recorded in the project
provenance memory; historical filenames do not determine the current data status.
The expanded seed-42 endpoint table, teacher history subtraction, and I16
comparison are accompanied by independent-test, selection, coverage, and
known-overlap sensitivity tables in the appendix.

## Compile

Compile the root `iclr2027_conference.tex` with pdfLaTeX and BibTeX:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The already-generated figure PDFs and table `.tex` files are included. No
experiment data, plotting scripts, Python environment, or GPU is needed to
compile the paper. Standard LaTeX packages are supplied by the TeX installation.

In Overleaf, set the main document to `iclr2027_conference.tex` at the project
root, then use **Recompile from scratch**.

## Research archive and local files

The complete tracked tree immediately before this cleanup is preserved on
[`archive/before-compile-only-20260914`](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/archive/before-compile-only-20260914).
The earlier large research snapshot is on
[`archive/research-data-before-overleaf-cleanup-20260911`](https://github.com/zhusq20/Optimization-Dynamics-in-Multi-Task-LLM-Post-Training/tree/archive/research-data-before-overleaf-cleanup-20260911).

Removed research files remain on disk in the working checkout and are ignored
by Git. The default branch no longer synchronizes them to Overleaf. Git history
has not been rewritten. Add any future compilation dependency explicitly to
`.gitignore` before tracking it.

## Recovering an old Overleaf project

Older Overleaf snapshots can still contain files from before the cleanup.
A successful Git merge does not prove every file reached Overleaf: the GitHub
comparison API returns at most 300 changed files, and the September 14 recovery
comparison had 446. Its response omitted the six-panel figure and its section.

If the existing project remains stale after pulling, importing the current
GitHub repository into a new Overleaf project (or uploading the compact source
ZIP as a new project) obtains the complete compilation-source tree without relying on the
old incremental comparison. The source ZIP should contain no compiled paper
PDF, temporary outputs, or experimental data.
