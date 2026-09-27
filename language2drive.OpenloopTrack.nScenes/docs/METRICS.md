# nuScenes metric specification

For trajectory points p1,...,p12, insert p0=(0,0) for finite differences. Let `v[k]=||p[k]-p[k-1]||/0.25` and `D=||p12||`. Coordinates are meters in the fixed anchor ego frame. The output declares `ego_at_anchor` and `ego_pose_origin`. Timestamps 0.25,...,3.00 must match within 1e-5 s; all coordinates must be finite.

## Binary predicates: Eqs. (34), (37), Table 14

| Arm | Required samples | Predicate, in addition to D >= 8 m |
|---|---|---|
| Lane change, arm 0 | 2.25,2.50,2.75,3.00 s | Every point in target corridor |
| Lane keep, arm 1 | All 12 points | Every point in current corridor |
| Low speed, arm 0 | Eight intervals ending at 1.25,...,3.00 s | Every speed in [3.2,4.8] m/s |
| High speed, arm 1 | Same eight intervals | Every speed in [7.2,8.8] m/s |

Each lane corridor contains available CAR lane polygons from its contract token set, individually buffered by 0.15 m. `covers` includes the boundary. The first scored speed interval is 1.00–1.25 s. D is Euclidean endpoint displacement.

`B_a = valid * own_predicate * (1-opposite_predicate)`. Both predicates include the displacement requirement. SemSISR is 100 times the mean of `(B_0+B_1)/2`; SemCFPS is 100 times the mean of `B_0*B_1`. Invalid arms remain in the denominator. Incomplete requested result sets are rejected before scoring.

## Graded completion: Eqs. (35), (38)

`C_a(tau) = min(1,D/8) * mean(required membership / speed-band indicators)`; invalid output gives C=0. The own-arm scores are `c0=C_0(tau_0)`, `c1=C_1(tau_1)`.

- SCS = 100 * mean((c0+c1)/2).
- PCS = 100 * mean(min(c0,c1)).

The indicator scopes match Table 14. There is no opposite-target penalty or motion/map multiplier in C. The weaker-arm minimum is taken before wording or PCU averaging.

## Instruction response: Eqs. (39), (40)

At each common-window time, let d0 and d1 be distance to the two lane corridors, or distance to their speed intervals `max(lower-v,0,v-upper)`. Define `p0=(d1-d0)/(d0+d1)`, or zero when the denominator is zero. Average these ratios over the common window, then compute `SRG_i=(mean_p0(tau_0)-mean_p0(tau_1))/2`. The lane common window has four terminal points; the speed window has eight interval endpoints. No epsilon, temperature, clipping or progress gate is added.

`NBD_i = sum_k ||p0k-p1k|| / sum_k (||p0k||+||p1k||)` across all 12 points. A zero denominator yields zero. Normalize within each PCU before cohort averaging. Report SRG and NBD multiplied by 100. Paired ADE is the mean pointwise separation in meters.

These three response measurements use valid pairs and record coverage. Empty valid sets yield null, not zero. Identical paired predictions produce zero SRG/NBD. NBD measures change magnitude; SRG captures its semantic direction.

## Aggregation, conditions and intervals

The declared wording set is averaged within PCU, followed by equal-PCU averaging. The canonical setting uses one wording. All three paraphrases are paired with their matching opposite arm. Controls are evaluated against the original arm target and reported separately. `Paraphrase_Worst_PCU` is the fraction of PCUs whose two arms succeed under every one of the three paraphrases. Control gaps subtract control SemCFPS from canonical SemCFPS and retain negative values.

Headline heldout weighting is 17 lane PCUs / 43 speed PCUs. The family-macro supplement assigns 1/2 to each family. Response coverage and null handling are applied inside each group; additional wording averages average the available paired responses within PCU and retain their measurement counts in coverage.

Binary intervals use 1,000 scene-cluster draws, `random.Random(2026)`, and sorted endpoints at floor(0.025*(B-1)) and floor(0.975*(B-1)). Continuous intervals use 2,000 draws, NumPy `default_rng(2026)` and linear quantiles. Family-macro supplementary intervals use 2,000 independent within-family scene draws and exclude undefined draws. Both arms and all declared wordings remain grouped within scene. There is one PCU per scene in this release.

## Separate quality diagnostics

Motion feasibility uses maximum speed <=17 m/s, initial-speed error <=3 m/s, maximum absolute acceleration <=5 m/s² and absolute wrapped yaw rate <=1 rad/s. Drivable-center checks all 12 reference points with a 0.2 m tolerance. Both rates use all scheduled arms, with invalid arms receiving zero. These diagnostics do not multiply semantic success. The track evaluates open-loop predictions; dynamic collision safety is not inferred from the test metadata.

All construction and scoring constants, plus additional diagnostic helper conventions, appear in `evaluation_parameters.json`.
