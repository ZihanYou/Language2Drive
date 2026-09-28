# L2D Offline NAVSIM graded compliance addendum

Status: frozen before model inference.
Benchmark set: `l2d-offline-navsim` 1.0.0, 38 released PCUs.
Rewrite set: canonical plus all three released paraphrases per arm

For PCU `i`, rewrite `r`, and instruction arm `j`, let `tau[i,r,j]` be the
eight-pose trajectory produced by independently running instruction `j`.
`C[j](tau)` is the graded completion of candidate contract `j` by `tau`.

## Graded completion

`C` is in `[0,1]` and is implemented by `task_completion` in `verify.py`.
Every trajectory satisfying the released binary contract has `C=1`.
Outside the feasible set, each primitive constraint decays linearly over one
native tolerance or accepted interval width. Conjunctive constraints take the
minimum primitive score. Lane completion uses minimum vehicle-footprint overlap
and terminal progress. Order completion gives 0 when neither event occurs, 0.5
when one occurs, and linearly scores the observed event gap when both occur.
Invalid trajectories receive `C=0` for both candidate contracts.

The evaluator records the full 2-by-2 matrix `C[j](tau[k])`, so the same
function scores a trajectory against its own or the counterfactual instruction.

## Aggregation

Each PCU receives the same weight. Within a PCU, its four rewrites receive the
same weight. There is no family-frequency weighting.

* `SemSISR = 100 mean_(i,r) ((B_0 + B_1) / 2)`, where `B_j` is released
  exclusive binary semantic success for `tau_j`.
* `SemCFPS = 100 mean_(i,r) (B_0 B_1)`.
* `SCS = 100 mean_(i,r) ((C_0(tau_0) + C_1(tau_1)) / 2)`.
* `PCS = 100 mean_(i,r) min(C_0(tau_0), C_1(tau_1))`.

PCS is graded paired compliance. It is distinct from binary SemCFPS.

For every report, the evaluator checks

`SCS - PCS = 50 mean_(i,r) |C_0(tau_0) - C_1(tau_1)|`.

The right side is reported as `paired_imbalance`, together with an
`identity_residual` that must be numerically zero.
