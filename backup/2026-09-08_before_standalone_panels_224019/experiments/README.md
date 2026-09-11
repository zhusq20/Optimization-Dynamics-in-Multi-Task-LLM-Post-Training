> **2026-09-08：31组局部机制探针已完成并接入实验图。** [五页主图PDF](../figures/local_probe_results_20260908/main_local_probe_figures.pdf)、[完整十五页](../figures/local_probe_results_20260908/all_local_probe_figures.pdf)、[数据与重画](local_probe_results_20260908/README_zh.md)、[更新后的作图路线图](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md)。新增交集训练和全部训练种子重复已取消；下方旧实验计划只作历史背景，以现行路线图为准。

# MOPD experiment evidence

Start with the [2026-09-08 experiment inventory and 8×3 / 9×3 figure roadmap (中文)](EXPERIMENT_FIGURE_ROADMAP_20260908_zh.md). It maps **all 27 proposed panels** to the current manuscript's token-balancing, update-sparsity / teacher-overlap, and supervision-density questions, with measured coverage, missing experiments, measurement identities, and conditional budgets.

- [Next experiments: priorities, inputs, code readiness, and budgets (中文)](NEXT_EXPERIMENTS_20260908_zh.md).
- [27-panel layout](../figures/evidence_audit_20260908/figure_roadmap_9x3.pdf) and [editable panel registry](figure_audit_20260908/figure_plan.csv).
- [Three exploratory triptychs from real data](../figures/evidence_audit_20260908/exploratory_triptychs.pdf): nine preview slots with checkpoint-specific method labels and missing series identified.
- [Frozen evidence snapshot and source hashes](figure_audit_20260908/evidence_snapshot.json), [training inventory](figure_audit_20260908/run_inventory.csv), [verified capability points](figure_audit_20260908/capability_inventory.csv), and [BF16 scans with corrected method identities](figure_audit_20260908/bf16_inventory.csv).
- [Snapshot scope, limitations and reproduction commands](figure_audit_20260908/README_zh.md).

The current plan covers six experiment groups: M-PG, S-PG, M-Overlap64-DR/DT/GT, and S-Overlap64. The [01:50 UTC progress check](figure_audit_20260908/next_experiment_progress.json) records 500 / 313 / 235 / 246 / 243 / 169 updates, respectively. Prioritize three multi-teacher Overlap64 checkpoint-250 capability suites, complete the existing runs, and add overlap-loss local diagnostics before expanding training seeds. Individual measurements retain their checkpoint-specific identities and raw provenance.

The current training recipe uses the intersection of rollout-student and teacher Top64. Weights remain normalized on the original student support; intersection weights are not renormalized. Parameter-change figures use stored BF16 values or actual BF16 writeback. Older teacher Top64 and proposed student Top16 experiments are separate identities. The manuscript still needs corresponding formula and terminology updates.

The [three-contribution plan](../MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md) remains the conceptual background. The [Top16 audit](STUDENT_TOP16_EXPERIMENT_AUDIT_20260906_zh.md) and [Top16-era runtime estimates](STUDENT_TOP16_COVERAGE_AND_RUNTIME_20260906_zh.md) are historical snapshots superseded for execution status by the September 8 inventory.

The GPAS schematic generator (`generate_gpas_main_figure.py`), sampled-log-ratio utility (`measure_initial_kl.py`), and superseded GPAS plan are historical supporting materials. Synthetic calculations and schematics do not count as empirical evidence for these studies. Full-vocabulary probes are local references, not evidence of completed full-vocabulary online training.
