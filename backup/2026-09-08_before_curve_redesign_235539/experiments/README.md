# MOPD experiment evidence

The current 31 local probes are rendered as **15 standalone charts**, with six suggested main-text figures and nine supporting diagnostics. Start with the [question-led gallery](../figures/local_probe_results_20260908/index.html), [six-figure reading PDF](../figures/local_probe_results_20260908/main_panels.pdf), or [complete PDF](../figures/local_probe_results_20260908/all_panels.pdf). Every page and export contains one chart.

- [Current experiment and figure roadmap (中文)](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md): 27 registered slots, current figure definitions and remaining scope.
- [Reviewer questions and figure rationale (中文)](local_probe_results_20260908/REVIEWER_FIGURE_AUDIT_zh.md), [English caption drafts](local_probe_results_20260908/FIGURE_CAPTIONS.md), and [portable data/reproduction guide](local_probe_results_20260908/README_zh.md).
- [Next work and cancelled experiments (中文)](NEXT_EXPERIMENTS_20260908_zh.md). No new intersection training, training seeds or cap training are queued.
- [Current panel registry](local_probe_results_20260908/figure_plan.csv) and [per-observation plot values](local_probe_results_20260908/panel_values.json).

F5 now uses explicit paired comparisons with observed ranges, F6 uses identically ordered teacher-pair matrices, and F7 compares objectives within a fixed student state using named references. Dense scatter clouds and triptychs are superseded. Explanations are in captions; charts retain only axes, numerical values and necessary legends.

I64 denotes the student/teacher Top64 intersection, with weights normalized on the original student Top64 and no additional intersection renormalization. The local HF gradients and saved-Adam BF16 proposed steps do not replace online capability evaluations or actual online update records. Mixed training histories and the tiny held-out KL limitations remain explicit in the captions.

The [01:10/01:50 evidence snapshot](figure_audit_20260908/README_zh.md), [training inventory](figure_audit_20260908/run_inventory.csv), [capability inventory](figure_audit_20260908/capability_inventory.csv), and [BF16 inventory](figure_audit_20260908/bf16_inventory.csv) are historical records, not a refreshed status report for other nodes. Earlier plot assets remain in the evidence audit and backup directories.

The [three-contribution plan](../MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md), [Top16 audit](STUDENT_TOP16_EXPERIMENT_AUDIT_20260906_zh.md), and [Top16-era estimates](STUDENT_TOP16_COVERAGE_AND_RUNTIME_20260906_zh.md) are historical context. The GPAS schematic generator, sampled-log-ratio utility and superseded GPAS plan do not count as empirical evidence for these studies.
