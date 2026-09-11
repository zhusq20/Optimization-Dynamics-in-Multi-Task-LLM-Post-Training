# MOPD: three empirical studies

The active specification is [the three-contribution plan](../MOPD_THREE_CONTRIBUTIONS_2026-09-05_zh.md). It supersedes the GPAS-centered plan.

1. Explain and measure the effect of global token averaging, domain token averaging, and response averaging on capability balance.
2. Verify sparse OPD updates; measure teacher-conditioned support overlap at the same MOPD student checkpoint; compare teacher-pair distribution distance with subnetwork distance. Teacher–student gap is an auxiliary analysis.
3. Compare vocabulary-level supervision density in single-teacher and multi-teacher OPD: sampled-token PG, top-64, and full-vocabulary KL. Small common-prefix full-vocabulary probes measure local updates; PG/top-64 online runs measure cumulative changes. Local probes do not imply completed full-vocabulary online training. Further noise or truncation analysis is conditional on a stable meaningful sparsity difference, not required by default.

The initial design reuses six training configurations: representative single-teacher PG/top-64, joint PG, and joint top-64 with the three reductions. Single-teacher references beyond that representative setting, more teachers, seeds, and a length-cap sweep depend on available checkpoints and measured resources. There is no new required sampling algorithm or disjoint-mask training method.

Teacher-conditioned gradients and proposed optimizer steps start from identical copies of an actual MOPD checkpoint. Independent single-teacher endpoint deltas are reference measurements, not historical attribution inside the joint run. Mathematics, code, instruction following, and science each use their own Qwen3-1.7B teacher trained with reinforcement learning (RL), giving four teacher weight sets.

The end-to-end trainer and measured results for these studies are not yet present. Older analytical figures and synthetic calculations have been moved to an external recovery archive. The retained GPAS schematic generator (`generate_gpas_main_figure.py`), sampled-log-ratio utility (`measure_initial_kl.py`), and superseded GPAS plan are historical supporting materials; none constitute empirical evidence for the revised questions or required experiments in the current plan.
