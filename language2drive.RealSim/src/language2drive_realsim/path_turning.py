"""Discrete steering-budget screen; never a controller or safety certificate."""
import math

import numpy as np


def turning_budget(times, positions, headings, intervention_s=1.7):
    """Reject heading changes requiring near-in-place turns after intervention.

    The pinned nonlinear controller bounds front steering to 0.785 rad and
    uses a 2.85 m wheelbase. tan(delta)/L is a deliberately permissive bicycle
    curvature proxy (its low-speed implementation uses delta/L). Chord lengths
    and sampled rig poses are not an exact dynamic reachability calculation.
    A fixed 0.02 rad discretization allowance applies per <=0.2 s interval;
    stationary intervals are checked too, with no division by zero.
    """
    t, p, h = (np.asarray(v, dtype=float) for v in (times, positions, headings))
    if (t.ndim != 1 or len(t) < 3 or p.shape != (len(t), 3) or h.shape != t.shape
        or not all(np.isfinite(v).all() for v in (t, p, h))
        or not math.isfinite(intervention_s) or t[0] != 0
        or intervention_s < 0 or intervention_s >= t[-1]
        or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > 0.200001)
        or not np.any(np.isclose(t, intervention_s, atol=1e-6, rtol=0))):
        raise ValueError("invalid dense path for steering-budget screen")
    curvature = math.tan(0.785) / 2.85
    distance = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    turn = np.abs(np.angle(np.exp(1j * np.diff(h))))
    active = t[:-1] >= intervention_s - 1e-6
    excess = turn - distance * curvature - 0.02
    bad = active & (excess > 1e-9)
    return {"scope": "sampled_bicycle_turn_budget_not_dynamic_reachability_or_qualification",
            "passed": not bool(bad.any()), "wheelbase_m": 2.85,
            "front_steering_bound_rad": 0.785, "curvature_proxy_limit_per_m": curvature,
            "per_interval_heading_allowance_rad": 0.02,
            "maximum_heading_budget_excess_rad": float(max(0., excess[active].max())),
            "violations": [{"start_s": float(t[i]), "end_s": float(t[i+1]),
                            "distance_m": float(distance[i]), "heading_change_rad": float(turn[i])}
                           for i in np.flatnonzero(bad)]}
