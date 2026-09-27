"""Explicit B2 planning timing, without changing scope or safety requirements."""
from itertools import product
import numpy as np


def marker_curvature_grid(cruises, alignment_distances, initial_cruise_deceleration_mps2=None,
                          *, centerline_smoothing_m=None, terminal_standoff_m=None):
    """Bounded explicit B2 hypotheses; preserve legacy parameter sets by default."""
    for values, low, high in ((cruises, 0., 15.), (alignment_distances, 2., 30.)):
        if (not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 3
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
                   or not np.isfinite(v) or v < low or v > high or v <= 0 for v in values)
            or len(set(values)) != len(values)):
            raise ValueError("distinct bounded cruise speeds and spatial alignment distances required")
    deceleration = initial_cruise_deceleration_mps2
    if deceleration is not None and (
        isinstance(deceleration, (bool, np.bool_)) or not isinstance(deceleration, (int, float))
        or not np.isfinite(deceleration) or not 0 < deceleration <= 1.5):
        raise ValueError("initial cruise deceleration must be positive and at most the 1.5 m/s2 planning budget")
    for name, value, upper in (("centerline smoothing", centerline_smoothing_m, 2.),
                               ("terminal standoff", terminal_standoff_m, 10.)):
        if value is not None and (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= upper):
            raise ValueError(f"{name} must be finite and within 0–{upper:g} meters")
    result = []
    for speed, distance in product(cruises, alignment_distances):
        parameters = dict(cruise_mps=speed, alignment_distance_m=distance, align_initial_heading=True,
            lateral_budget_mps2=2., acceleration_mps2=1.5, centerline_smoothing_m=1.,
            speed_step_policy="actual_advance_bisection", intervention_s=1.7)
        if deceleration is not None:
            parameters['initial_cruise_deceleration_mps2'] = deceleration
        if centerline_smoothing_m is not None:
            parameters['centerline_smoothing_m'] = centerline_smoothing_m
        if terminal_standoff_m is not None:
            parameters['terminal_standoff_m'] = terminal_standoff_m
        result.append(parameters)
    return result


def forward_corridor_length(value=180.):
    """Allow explicit longer original-map traversal, never shorten old coverage.

    This is a requested distance, not a claim that the map supplies that much
    connected road. The existing reciprocal/gap/branch checks still apply.
    """
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
        or not np.isfinite(value) or not 180. <= value <= 300.):
        raise ValueError("explicit original forward length must be within 180–300 meters")
    return float(value)


def parse_forward_lane_choices(items):
    """Parse bounded explicit hints; original-map traversal verifies every link."""
    return _parse_lane_choices(items, "forward")


def parse_backward_lane_choices(items):
    """Parse NEXT=PREVIOUS private approach hints, never recorded-history claims."""
    return _parse_lane_choices(items, "backward")


def _parse_lane_choices(items, direction):
    if not isinstance(items, (list, tuple)) or len(items) > 8:
        raise ValueError(f"at most eight explicit {direction} choices required")
    result = {}
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"{direction} choices must be FROM=TO strings")
        pair = item.split("=")
        if (len(pair) != 2 or any(not v or v.strip() != v or v == "-1" for v in pair)
            or pair[0] == pair[1] or pair[0] in result):
            raise ValueError(f"distinct nonempty FROM=TO {direction} choices required")
        result[pair[0]] = pair[1]
    return result


def marker_motion_grid(duration_s, start_timestamp_us, recording_end_timestamp_us):
    """Keep six full seconds after a bounded, explicit diagnostic deadline."""
    if (isinstance(duration_s, (bool, np.bool_))
        or not isinstance(duration_s, (int, float)) or not np.isfinite(duration_s)
        or not 7 <= duration_s <= 30
        or not np.isclose(duration_s * 10, round(duration_s * 10), atol=1e-8, rtol=0)
        or type(start_timestamp_us) is not int or type(recording_end_timestamp_us) is not int
        or start_timestamp_us < 0 or recording_end_timestamp_us <= start_timestamp_us):
        raise ValueError("bounded 10 Hz deadline and exact recording timestamps required")
    horizon = duration_s + 6.
    if start_timestamp_us + round(horizon * 1e6) >= recording_end_timestamp_us:
        raise ValueError("recording must cover the deadline and six full future seconds")
    return np.round(np.arange(round(horizon * 10) + 1) / 10, 6)


def spatial_alignment_completed_s(times, positions, alignment_distance_m):
    """Find completion of an explicit planner alignment, not a lane commitment."""
    t, xyz = np.asarray(times, dtype=float), np.asarray(positions, dtype=float)
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3)
        or not np.isfinite(t).all() or not np.isfinite(xyz).all()
        or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)
        or not np.any(t == 1.7) or isinstance(alignment_distance_m, (bool, np.bool_))
        or not np.isfinite(alignment_distance_m) or not 2 <= alignment_distance_m <= 30):
        raise ValueError("dense finite path and bounded explicit spatial alignment required")
    active = t >= 1.7
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[active, :2], axis=0), axis=1))]
    reached = np.flatnonzero(arc >= alignment_distance_m)
    if not len(reached):
        raise ValueError("path never completes the proposed spatial alignment")
    return float(t[active][reached[0]])
