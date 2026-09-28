# WOD-E2E paper scoring profile

This document specifies the WOD-E2E P0 kinematic-proxy metric profile in paper
Appendix E.2/E.5. `freeze_paper_contracts.py` selects eligible pairs and writes
private instance parameters before any model output is read. `evaluate_paper.py`
checks that the release and contract hashes match before scoring predictions.

## Common protocol

Let p_0=(0,0), p_k be the 20 XY points, and dt=0.25 s.

- v_k=||p_k-p_(k-1)||/dt, k=1,...,20.
- L=dt*sum(v_k).
- v_tail=mean(v_17,...,v_20): support 4.00-5.00 s.
- v_late=mean(v_12,...,v_20): nine intervals with end times 3.00,...,5.00 s;
  actual support is 2.75-5.00 s, matching `speeds[11:]`.

Validity requires 20 points, exactly two finite numeric values per point,
excluding booleans; max(v_k)<=50 m/s and
max(|v_k-v_(k-1)|/dt)<=20 m/s^2 for k>=2. These are anomaly filters, without
footprint, collision, heading, jerk or initial-speed-continuity checks.

Define U(z;a,b)=clip((z-a)/(b-a),0,1) and D=1-U. Joint continuous constraints
use `min`. Binary bounds below are inclusive with no extra epsilon.

## STOP_GO

q_stop is the stop reference endpoint; d=||p_20-q_stop||.

| Arm | Binary predicate | Continuous completion |
|---|---|---|
| stop | v_tail<=0.6 and d<=2 | min(D(v_tail;0.6,2), D(d;2,5)) |
| go | v_tail>=2 and x_20>=8 | min(U(v_tail;0.6,2), U(x_20;0,8)) |

Speed units are m/s and distances are meters. The stopping target is frozen
from the stop reference endpoint for each PCU.

## SPEED_CHOICE

Let s and n be the slow and normal reference v_late, with g=n-s>=1 m/s.
Define low=s+0.25g and high=s+0.75g.

| Arm | Binary predicate | Continuous completion |
|---|---|---|
| slow | v_late<=low and L>=2 | min(D(v_late;low,high), U(L;0,2)) |
| normal | v_late>=high and L>=2 | min(U(v_late;low,high), U(L;0,2)) |

These are per-candidate reference-derived speed bounds, frozen in the private
contract file. L is total XY path length.

## Qualification and aggregation

Build the contract before reading predictions. Retain a pair only if both
references satisfy their own binary predicate and fail the opposite predicate.
Unsupported families, insufficient reference speed gap and failed witnesses
are excluded from the proxy denominator with explicit reasons.

For arm j, B_j=P_j(tau_j) AND NOT P_(1-j)(tau_j). C_j scores only the own task.
Missing or invalid predictions have B_j=C_j=0 within the fixed eligible set.
For N eligible pairs, one P0 expression per pair:

```text
SemSISR = 100 * mean_i((B_i0+B_i1)/2)
SemCFPS = 100 * mean_i(B_i0*B_i1)
SCS     = 100 * mean_i((C_i0+C_i1)/2)
PCS     = 100 * mean_i(min(C_i0,C_i1))
```

Every eligible pair has equal weight; this is not equal family weighting.
The JSON output uses these paper column names and identifies the WOD-E2E P0
kinematic-proxy profile in metadata. Scoring emits per-PCU records and coverage.
The set has one P0 wording per PCU; the paper's wording mean is therefore a
single-item mean for this track.

## Separate geometry diagnostics

ADE is mean pointwise Euclidean distance over 20 points.
margin=ADE(prediction,other_ref)-ADE(prediction,own_ref).
Single-arm agreement requires margin>0.5 m (strict); |margin|<=0.5 m is ambiguous.
Paired agreement requires both arms. The geometry scorer uses valid-request and
complete-valid-pair denominators and reports coverage separately. These are not
the fixed-denominator proxy metrics above and do not compute official RFS.

## Separate pilot score

The stop/go pilot scorer uses stop: v_tail<=0.6; go: v_tail>=2 and x_20>=8.
It has no stop endpoint constraint or continuous completion. Its complete-valid
pair rate must not be compared as the same protocol as full proxy scoring.

## Candidate language rules

`draft_semantic_tasks.py` applies these priorities:

1. One reference has v_tail<0.6 and path lengths differ by >=3 m: STOP_GO,
   or TURN_WAIT if the moving reference has a turn label.
2. Opposite turn labels: TURN_CHOICE.
3. One turn label, both paths >=5 m, other heading magnitude <15 degrees:
   TURN_STRAIGHT.
4. Endpoint lateral separation >=2 m: LATERAL_CHOICE. Keep/shift template
   requires |y_stay|<1.2 and |y_shift|>=2.5; otherwise use farther-left/right.
5. Endpoint longitudinal separation >=4 m: SPEED_CHOICE, with the smaller
   path length assigned to slow.
6. Otherwise UNRESOLVED.

A turn label requires path>=5 m, heading magnitude>=30 degrees and
|y_20|>=3 m. Heading is atan2 of displacement p_20-p_16 when its norm>0.3 m,
otherwise zero. These are draft-generation rules, not semantic certification.
