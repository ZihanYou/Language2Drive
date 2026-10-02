# Evaluation protocol and native formats

The SDK coordinates evaluators; each native scorer remains authoritative. A report must record track, release, scorer, split, wording set, seed, camera rig, controller, checkpoint revision and expected inventory. Never combine different scorer profiles into one leaderboard cohort.

## PCU and metrics

A PCU holds the initial scene, non-language inputs and evaluator seed fixed. Each arm requests a distinct feasible behavior. Reset recurrent memory across arms. Individual strict success is SISR; paired strict success is CFPS. Strict closed-loop execution additionally considers the native safety/progress/deadline/stability requirements. Closed-loop SemISR / Sem-CFPS are relaxed proxies, not substitutes for strict success. Graded SCS and PCS quantify completion; paraphrase evaluation tests robustness under equivalent wording.

Open-loop SemSISR / SemCFPS use exclusive semantic success over predicted trajectories; they do not establish closed-loop safety. NAVSIM historical scoring and corrected scoring are separate profiles. All displayed tables are on a 0–100 scale, even when native intermediate outputs are fractions.

The native L2D summary balances three language score families: strict individual/paired, relaxed individual/paired and graded individual/paired. It uses the squared mean of square roots of the family scores, combined with a driving component through the native normalized product. It is not a simple arithmetic average of language and driving. CARLA PSM requires additional diagnostic routes; the unified core campaign does not manufacture it.

## Submission files

| Track | Native file | Core fields |
|---|---|---|
| nuScenes | `predictions.jsonl` | `pcu_id`, `arm`, `condition`, `seed`, `public_input_hash`, `instruction`, `trajectory` |
| NAVSIM | `canonical.json`, `rewrites.json` | PCU → arm (`"0"` / `"1"`) → `coordinate_frame`, `reference_point`, `timestamps`, `positions_xy`; rewrites keyed by wording |
| WOD-E2E | `predictions.jsonl` | `request_id`, `trajectory` (20 XY pairs) |
| CARLA | Native results + saved run plan | Original evaluator ledger and per-rollout result artifacts |
| RealSim | Real ASLs → normalized traces + receipts → `experiment.json` | Frozen scene/time/seed binding, driver identity, paired input hashes, 60 scheduled records |

Failed open-loop predictions are exported as invalid trajectories and remain in the expected workload. The native scorer decides their credit. Infrastructure failures, unavailable assets and incomplete inventories must not be presented as successful evaluation. The coordinator returns nonzero on any failed selected track, preserves logs, and continues other tracks. Open-loop request coverage is saved in `inference.json`.

RealSim preserves all 60 seed-1701 slots. Missing/invalid evidence receives zero credit. The SDK checks pinned runtime and unified-driver identity, normalizes ASLs through the existing bridge, and compares actual paired pre-intervention input hashes. Its report is a **custom-model paper-protocol evaluation**, not an automatically approved official submission. The legacy approved-model identity checks remain intact.

## Cohort identity

- CARLA paper: base cohort, both dev and test, seed 1, three expressions; 64 PCUs / 384 core rollouts. The source includes a wider inventory. SDK outputs dev/test scores separately; do not average unequal split sizes. Full native PSM/summary reproduction uses its diagnostic plans.
- RealSim paper: `realsim-paper-1.0`, 30 frozen PCUs, 23 scenes, 60 slots, seed 1701. Do not score a newly compiled larger cohort as this profile.
- nuScenes: verified release manifest; paper held-out 60-PCU canonical profile. The 75-PCU release includes 15 development PCUs.
- NAVSIM: 38 PCUs, 19 recordings, four wordings. The manuscript table uses the historical scorer; SDK custom runs use the corrected scorer.
- WOD-E2E: frozen P0 241-PCU scored subset. Construction workload size is not the scored denominator. Source data needs Waymo access.

The supplied paper table source hashes and copied PDF/figure hashes are recorded in `data/provenance.json`. `data/results.json` contains the corresponding published-page values without recomputing or rounding intermediate scorer values.
