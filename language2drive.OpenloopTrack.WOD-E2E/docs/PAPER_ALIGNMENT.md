# Paper-to-code alignment: WOD-E2E

The prediction grid and four paired metrics implement Appendix E.2 (Eqs. 34–35) and E.5 (Eqs. 42–54) of the paper's WOD-E2E protocol.

| Paper definition | Implementation |
|---|---|
| 5 s, 4 Hz, 20 future XY points in anchor rear-axle frame | `score_prerelease.normalize_trajectory`, model conversion adapters |
| Validity: finite XY, speed ≤50 m/s, adjacent interval speed change ≤20 m/s² | `score_prerelease.normalize_trajectory` |
| `v_tail = mean(v17..v20)` and `v_late = mean(v12..v20)` | `score_p0_kinematic_proxy.features` |
| Binary own success and opposite rejection | `score_p0_kinematic_proxy.score_arm` |
| STOP_GO and SPEED_CHOICE binary/graded task maps | `score_p0_kinematic_proxy.primitive` |
| Freeze eligible PCUs and private instance parameters | `freeze_paper_contracts.py` and `contracts.jsonl` |
| Equal-PCU, two-arm aggregation; invalid=0 | `score_p0_kinematic_proxy.metric_summary`, `evaluate_paper.py` |
| Separate 441-pair ADE response diagnostic | `score_full_prerelease.py` |

The paper's (P_0) and (P_1) in the arm equations refer to the two opposed **arm predicates**. `expression_id=P0` in data files denotes the single available **wording** for each PCU; there is no second P1 wording in this WOD-E2E release. The paper notation averages wordings within each PCU, then PCUs equally; with one wording this reduces to the implementation's paired mean.

The private speed-choice boundaries are frozen per PCU. Construction requires the normal-reference late speed to exceed the slow-reference late speed by at least 1 m/s, then sets the slow and normal acceptance boundaries at 25% and 75% of that gap. The scorer reads these frozen values and never changes them using model predictions.

The output JSON uses the paper's column names `SemSISR`, `SemCFPS`, `SCS`, `PCS` under `metrics_0to100` while recording the specific WOD-E2E P0 kinematic-proxy profile. This avoids confusing expression IDs, arm predicates, and metric names.

No reference-based best-of-K trajectory selection is used. Native model trajectories are converted to the common grid before scoring. The full candidate registry, qualified PCU set, and geometry analysis have distinct denominators; `summary.json` writes the metric denominator explicitly.
