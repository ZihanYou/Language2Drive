# Paper-to-code mapping

The `nuscenes-paper-1.0` profile follows the nuScenes-specific Appendix E.3. The profile's paper identity is recorded by SHA-256 in `config/paper_profile.json`.

| Paper definition | Implementation |
|---|---|
| Eq. (34), exclusive valid own-arm success | `score_lane_pcus.score_one`, `paper_metrics.score_pair` |
| Eq. (35), paired aggregates | `paired_completion`, `collapse_wordings`, `summarize` |
| Eq. (37), speed and displacement | `score_lane_pcus.score_one`, `semantic_metrics.speed_stats` |
| Eq. (38), temporal graded completion | `unified_semantic_metrics.compliance` |
| Eq. (39), distance-based signed response | `trajectory_preference`, `paper_metrics.score_pair` |
| Eq. (40), normalized behavior difference | `normalized_behavior_difference` |
| Table 14, task scopes and acceptance | `score_lane_pcus.py`, `docs/evaluation_parameters.json` |
| Table 15, target-distance helper functions | `semantic_metrics.speed_stats`, `corridor_stats`, `lane_preference`, `paired_response` |
| E.6, scene confidence intervals | `binary_interval`, `aggregate`, `family_macro` |
| Table 9, nuScenes rows | `scripts/reproduce_paper.py` |

The nuScenes graded map uses temporal coverage and the displacement factor. Tolerance-decay maps and minimum conjunctions from other tracks are not substituted for Eq. (38).

The public report uses the paper's `SemSISR` / `SemCFPS` names and 0–100 scale. Low-level compatibility kernels retain fraction-valued quantities. Only the report boundary applies the factor of 100; paired ADE remains in meters.

The headline Table 9 setting is canonical heldout, seed 0. Each additional wording is evaluated as its own paired instance. When a wording set is explicitly requested, the product/minimum is computed within each wording, then wordings are averaged within each PCU, then PCUs are averaged. Masked and shuffled controls are never automatically merged into the wording set.

The legacy coverage-matrix response is explicitly called `SRG_coverage` in `coverage_matrix_metrics`. The paper's `SRG` field always comes from Eq. (39). Invalid trajectories retain zero completion in scheduled denominators; undefined response measurements use JSON `null` and carry coverage.

Original public inputs and evaluation contracts retain their frozen hashes. `provenance.json` binds each new score to the paper profile, prediction file, public/private release hashes, actual map hashes and the portable scoring code. The original release manifest records the original scorer identity; current code hashes are recorded separately.
