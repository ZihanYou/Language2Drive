# Paper 1.0 implementation map

The default release scorer is `paper_metrics.py`, profile `realsim-paper-1.0`.
It implements the RealSim track, not the CARLA or open-loop protocols.

| Paper definition | Implementation |
|---|---|
| Section 2.4 / Eq. 1: SISR and CFPS | `aggregate_pairs`: arm mean and binary product, expression-then-PCU means |
| Appendix B.1 / Eq. 7: strict verdict | `score_document`: own, not alternative, safety, progress, deadline, stability |
| Appendix B.2 / Eq. 8: relaxed proxies | Own contract OR the predefined relaxed semantic component |
| Eq. 3: SCS and PCS | Own-request graded completion and paired minimum, not product |
| Eq. 2 and Eq. 11: paired gaps | Exact strict and graded imbalance checks in `aggregate_pairs` |
| Eq. 4–5 / Eq. 18–20: L2D | `l2d_summary`, applied after aggregation at full precision |
| Appendix B.5 / Eq. 16 and G.1: DrivingScore | Arm-specific deadline, frozen reference polyline, 5 m corridor, maximum arc length, fixed 60 slots |
| Appendix C.4–C.5: clocks and task gates | Frozen task annotations and component verifier; Progress is separate from DrivingScore |

The experiment interface enforces 30 unique PCUs, arms R0/R1 and seed 1701.
One expression is evaluated per PCU. Failed or unavailable scheduled executions
contribute zero; retries do not increase the denominator. Missing records and
malformed valid-evidence claims are rejected. Coverage is always returned.
For protocols with more expressions, `aggregate_pairs` keeps equal PCU weights
even when the expression counts differ. DrivingScore's fixed-60 formula refers
specifically to the current RealSim document interface.

The graded task maps are retained: scope-eligible speed/lane satisfaction times
the capped forward-progress factor; branch tasks use first correct commitment
within the decision window. These are separate from the relaxed binary proxies.
Their details are in [METRICS](METRICS.md).

SRG/NBD and trajectory-response statistics are supplementary diagnostics, not
components of the main-table L2D score. The open-loop definitions and cohorts
must not be substituted for RealSim's task maps or scheduled denominator.

Tests check unequal expression counts, partial grades, failed slots, asymmetric
deadlines, invalid evidence, both gap identities, score range and the L2D
arithmetic in all eight displayed RealSim Table 2 rows. The table check permits
rounding error in the published two-decimal inputs; it is an algebra check, not
a fresh model evaluation. Historical versioned scoring APIs remain available
without rewriting their previously produced reports.
