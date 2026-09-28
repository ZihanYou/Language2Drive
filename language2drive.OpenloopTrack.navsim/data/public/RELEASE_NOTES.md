# L2D Offline NAVSIM 1.0.0

Release date: 2026-09-23

This is the first frozen benchmark release. It contains 38 human-approved PCUs
from 19 NAVSIM/OpenScene test recordings. The released set contains only
approved PCUs.

## Distribution

- Tier A: 14 PCUs
- Tier B: 9 PCUs
- Tier C: 11 PCUs
- Tier D: 4 PCUs
- Families: lane choice 12, landmark stop 10, speed band 7, order 4,
  object-relative speed 2, branch choice 1, local goal 1, pass/follow 1

## Release gates

- 38/38 released PCUs passed automated construction audit.
- 38/38 passed 0.1-second continuous path collision review.
- 38/38 have recorded human approval decisions.
- Public and evaluator PCU identifiers match exactly.
- Public leakage audit found no private contract, witness, source token, future,
  or map-target fields.
- Repeated builds produced byte-identical benchmark artifacts.
- Oracle evaluator smoke test produced OL-SemSISR = 1.0 and OL-SemCFPS = 1.0
  over all 38 PCUs and 76 arms.

The release remains a compact first version. Its recording-cluster bootstrap
and family breakdowns should accompany aggregate results. The benchmark tests
fixed-observation planning-level language control and does not substitute for
closed-loop evaluation.
