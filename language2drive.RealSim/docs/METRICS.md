# Scoring protocol

## Evaluation unit and isolation

A paired counterfactual unit (PCU) fixes scene, initial state, sensor inputs,
intervention time, decision horizon and non-language settings. Two arms receive
different instructions. Semantic targets, geometric regions and oracle routes
are evaluator-side annotations; they must not be exposed to a tested model.
Before comparing arms, verify equivalent pre-intervention observations and
matching seeds/configuration. An oracle is a feasibility witness, not a tested
language-conditioned model.

The current reporting interface consumes 30 PCUs × two arms × seed 1701.
The library retains other authoring profiles and three-seed oracle qualification
support. Do not infer the experiment's size or seeds from legacy library defaults.
Use explicit frozen source and experiment documents.

## Physical and binary task metrics

The paper entry point is `l2d-score` (also `l2d-review score`). All reported
percentages use fixed cohort denominators: 60 arm slots or 30 pairs. More
generally, expressions are averaged within a PCU before averaging PCUs.

| Metric | Definition |
|---|---|
| SISR | Own contract satisfied, opposing contract not satisfied, physical safety, required progress, deadline and stability all satisfied |
| SemISR | Own-contract success or the task-specific relaxed semantic success predicate |
| Safety | No observed collision and no observed off-road violation in a valid trace |
| Progress | Frozen task-specific minimum progress predicate |
| CFPS | Both arms satisfy SISR |
| Sem-CFPS | Both arms satisfy SemISR |
| DrivingScore | Route-completion percentage, averaged over slots |

The physical-only profile does **not** score red-light violations, speed-limit
violations or wrong-way driving. A commanded speed band is nevertheless a task
target; it remains part of instruction compliance.

DrivingScore projects ego positions onto the frozen arm-specific oracle route,
uses the maximum along-route progress within a 5 m corridor, and divides it by
reference-route length at the arm's task deadline (bounded by the decision
deadline). The alternative arm's later deadline and the oracle tail do not
extend this horizon. It is not the thresholded
Progress indicator or raw distance traveled.

Evidence validity is supplied by the upstream run-verification stage. The
offline scorer validates normalized trajectories, but cannot establish model
identity or simulation provenance from coordinates alone. Missing/invalid
executions receive zero credit and remain in the scheduled denominator. Their
coverage is reported separately. This is a scoring convention, not a claim that
unavailable trajectories were observed. A missing slot or malformed evidence
claimed as valid is an input error. Legacy interval reports remain available via
their versioned library modules, not as the paper's main-table output.

## Paper summary score L2D

All quantities below use the 0–100 scale, without rounding intermediate values:

```
a1 = (SISR + CFPS) / 2
a2 = (SemISR + Sem-CFPS) / 2
a3 = (SCS + PCS) / 2
Lbal = ((sqrt(a1) + sqrt(a2) + sqrt(a3)) / 3)**2
D = (Safety + DrivingScore) / 2
L2D = D * (1 + 0.99 * Lbal) / 100
```

L2D combines population-level metrics; do not average per-rollout L2D values.
The driving term above is RealSim-specific, not CARLA's PSM-based term. With
zero language scores, L2D is at most one point. Report component scores alongside
the summary. SISR − CFPS is half the percentage of one-sided successful pairs.

## Continuous paired semantic metrics

For PCU i, rewrite r and arm j, let C_j(tau) be compliance with target j, including
cross-scoring a trajectory generated under the other instruction. For persistent
speed/lane contracts:

```
C_j(tau) = fraction of eligible samples satisfying target j
           * min(1, max(0, terminal forward progress) / 8 m)
```

The frozen scope and decision/contract deadlines define eligibility. Gate/route
contracts use first commitment to the requested gate within the deadline,
multiplied by the same progress factor. No lane target is invented for a gate
contract. These target definitions cover speed, lane and route PCUs.

```
SCS = 100 * mean_i(mean_r((C_0(tau_0) + C_1(tau_1)) / 2))
PCS = 100 * mean_i(mean_r(min(C_0(tau_0), C_1(tau_1))))
SCS - PCS = 50 * mean_i(mean_r(abs(C_0(tau_0) - C_1(tau_1))))
```

PCS is graded paired compliance, not binary CFPS. The scorer asserts this
identity. The aggregation utility weights rewrites within each PCU equally,
then weights PCUs equally. The current 60-slot document contains one observed
paired realization per PCU, not an unmeasured paraphrase ensemble.

SRG uses normalized preference between the two frozen semantic targets:

```
p_0(tau) = mean_t((d_1(t) - d_0(t)) / (d_0(t) + d_1(t)))
SRG = 100 * mean_i(mean_r((p_0(tau_0) - p_0(tau_1)) / 2))
```

Distances are to speed bands, lane polygons or gate targets, as appropriate.
The comparison uses common eligible samples on a 0.1 s interpolation grid.
When both distances are zero, preference is zero. Empty common support gives
zero SRG by convention; `common_effective_samples` is retained so this case is
distinguishable from an observed zero response. SRG ranges from -100 to 100.

NBD measures direction-agnostic behavioral change:

```
NBD = 100 * mean_i(mean_r(
        sum_t(norm(tau_0(t) - tau_1(t))) /
        sum_t(norm(tau_0(t)) + norm(tau_1(t))) ))
```

XY coordinates use the same scene-start origin; mismatched origins are rejected.
Both trajectories are interpolated within the common decision horizon.
Zero denominator gives zero. NBD ranges from 0 to 100; a larger value alone
does not mean better instruction following.

## Supplemental diagnostics

`supplemental_diagnostics.make_runs` creates normal, repeat-1, repeat-2,
opposite, empty, neutral and shuffled conditions for selected speed/lane/route
PCUs. Repeats hold seed and input settings constant. Shuffling uses a different
PCU's instruction and records its donor. Compare same-instruction repeat NBD
with opposite-instruction NBD, and compare normal/empty/neutral/shuffled
compliance on the original target. Literal empty prompts require explicit
`L2D_DIAGNOSTIC_ALLOW_EMPTY=1`; the normal model path rejects empty instructions.

Failure reporting separates output/control failure, wrong behavior, correct
behavior not completed, and physical safety violation. Unknown evidence and
ambiguous semantic choice remain separate. The mutually exclusive primary
hierarchy is technical failure → unknown evidence → safety violation → wrong
behavior → correct-but-incomplete → complete → ambiguous. Multi-label counts,
overlaps and conditional denominators are also retained. Overlapping labels
must not be summed as a total failure rate.
