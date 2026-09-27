"""Curvature-aware exploratory branch timing, with exact recorded input prefix."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from shapely.geometry import LineString, Point

from .lane_path_search import smoothstep5


def offset_planned_path(times, positions, headings, offset_left_m, intervention_s=1.7, transition_s=2.):
    """Offset a proposed path, not its map; preserve the exact observed prefix.

    This is only a planning hypothesis. Recheck original geometry, motion,
    traffic and actual support independently. Positive offset is vehicle-left.
    """
    t, xyz, yaw = [np.asarray(a, dtype=float) for a in (times, positions, headings)]
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or not all(np.isfinite(a).all() for a in (t, xyz, yaw))
        or not np.isfinite([offset_left_m, intervention_s, transition_s]).all()
        or isinstance(offset_left_m, (bool, np.bool_)) or abs(offset_left_m) > .5
        or transition_s < 2. or t[0] != 0 or np.any(np.diff(t) <= 0)
        or np.any(np.diff(t) > .200001) or not np.any(t == intervention_s)
        or not 0 < intervention_s < intervention_s + transition_s <= t[-1]):
        raise ValueError("bounded offset, full dense plan and exact intervention required")
    result, new_yaw = xyz.copy(), yaw.copy()
    if offset_left_m == 0:
        return result, new_yaw
    blend = smoothstep5((t - intervention_s) / transition_s)
    result[:, 0] -= np.sin(yaw) * offset_left_m * blend
    result[:, 1] += np.cos(yaw) * offset_left_m * blend
    velocity = np.gradient(result[:, :2], t, axis=0)
    moving = (t > intervention_s) & (np.linalg.norm(velocity, axis=1) > 1e-8)
    new_yaw[moving] = np.arctan2(velocity[moving, 1], velocity[moving, 0])
    return result, new_yaw


def bounded_speed_step(position_s, speed, dt, arc, limits, acceleration):
    """Solve the next speed at its actual trapezoidal advance, not full acceleration.

    The lower bound enforces maximum braking. Interpolate squared speed, the
    domain of the braking-distance envelope; linear speed interpolation can
    itself demand excess braking between two valid envelope samples.
    """
    lower, upper = max(0., speed - acceleration * dt), speed + acceleration * dt

    def feasible(value, roundoff=0.):
        end = position_s + .5 * (speed + value) * dt
        return end < arc[-1] and value <= np.sqrt(np.interp(end, arc, limits**2)) + roundoff

    if not feasible(lower):
        # Do not include a tolerance in the maximization: repeatedly selecting
        # cap+tolerance accumulates envelope violations during braking.
        if feasible(lower, roundoff=1e-9):
            return lower
        raise ValueError("no bounded-braking step within the confirmed path and speed cap")
    if feasible(upper):
        return upper
    for _ in range(32):
        middle = .5 * (lower + upper)
        if feasible(middle):
            lower = middle
        else:
            upper = middle
    return lower


def curvature_limited_path(times, recorded_xyz, recorded_yaws, corridor_xyz,
                           cruise_mps=8., lateral_budget_mps2=2., acceleration_mps2=1.5,
                           intervention_s=1.7, alignment_s=1.4, centerline_smoothing_m=0.,
                           speed_step_policy="legacy_lookahead", terminal_standoff_m=0.,
                           alignment_distance_m=None, align_initial_heading=False,
                           initial_cruise_deceleration_mps2=None):
    """Limit speed by spatial curvature and braking distance, not recording clock.

    A natural cubic centerline is resampled by planar arc length. Smoothing and
    approximate rig height must still pass independent corridor, dynamics,
    traffic and actual NRE checks. No claim of physical support is made here.
    An explicit initial cruise deceleration treats cruise as a target, not an
    instantaneous cap on the measured prefix speed. Curvature and terminal
    braking envelopes remain hard constraints; this is not speed-limit approval.
    """
    t, xyz, yaw, center = [np.asarray(a, dtype=float) for a in
                           (times, recorded_xyz, recorded_yaws, corridor_xyz)]
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or center.ndim != 2 or center.shape[1] != 3 or len(center) < 3
        or not all(np.isfinite(a).all() for a in (t, xyz, yaw, center))
        or not np.isfinite([cruise_mps, lateral_budget_mps2, acceleration_mps2, intervention_s, alignment_s,
                           centerline_smoothing_m, terminal_standoff_m]).all()
        or not 0 <= centerline_smoothing_m <= 2.
        or not 0 <= terminal_standoff_m <= 10.
        or speed_step_policy not in {"legacy_lookahead", "actual_advance_bisection"}
        or min(cruise_mps, lateral_budget_mps2, acceleration_mps2, alignment_s) <= 0
        or t[0] != 0 or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)
        or not 0 < intervention_s < t[-1] or not np.any(t == intervention_s)):
        raise ValueError("invalid curvature plan inputs or missing exact intervention sample")
    if alignment_distance_m is not None and (isinstance(alignment_distance_m, (bool, np.bool_))
        or not np.isfinite(alignment_distance_m) or not 2 <= alignment_distance_m <= 30):
        raise ValueError("spatial alignment distance must be finite and within 2–30m")
    if type(align_initial_heading) is not bool or (align_initial_heading and alignment_distance_m is None):
        raise ValueError("initial heading alignment requires explicit spatial alignment")
    if initial_cruise_deceleration_mps2 is not None:
        value = initial_cruise_deceleration_mps2
        if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 < value <= acceleration_mps2
            or speed_step_policy != "actual_advance_bisection"):
            raise ValueError("initial cruise deceleration requires a positive bounded budget and actual-advance policy")
    raw_steps = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1)
    if np.any(raw_steps <= 1e-8) or not LineString(center[:, :2]).is_simple:
        raise ValueError("degenerate or self-intersecting centerline")
    raw_arc = np.r_[0., np.cumsum(raw_steps)]
    sample_arc = np.linspace(0., raw_arc[-1], int(np.ceil(raw_arc[-1] / .05)) + 1)
    smooth = CubicSpline(raw_arc, center, bc_type="natural", extrapolate=False)(sample_arc)
    if centerline_smoothing_m:
        # Optional spatial smoothing changes only the proposed driving path,
        # never original road boundaries or the acceptance thresholds. Raw
        # section-center kinks can otherwise induce artificial braking caps.
        # Independent geometry/dynamics/traffic checks remain mandatory.
        smooth = gaussian_filter1d(smooth, centerline_smoothing_m / (sample_arc[1] - sample_arc[0]),
                                   axis=0, mode="nearest")
    steps = np.linalg.norm(np.diff(smooth[:, :2], axis=0), axis=1)
    if np.any(steps <= 1e-8) or not LineString(smooth[:, :2]).is_simple:
        raise ValueError("smoothing created a degenerate or self-intersecting curve")
    arc = np.r_[0., np.cumsum(steps)]
    first = np.gradient(smooth[:, :2], arc, axis=0)
    second = np.gradient(first, arc, axis=0)
    curvature = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / np.linalg.norm(first, axis=1)**3
    # Conservative local envelope covers approximately half a meter of curve.
    curvature = maximum_filter1d(curvature, size=11, mode="nearest")
    cruise_cap = cruise_mps
    if initial_cruise_deceleration_mps2 is not None:
        prefix_index = int(np.flatnonzero(t == intervention_s)[0])
        measured_speed = np.linalg.norm(xyz[prefix_index, :2] - xyz[prefix_index - 1, :2]) / (t[prefix_index] - t[prefix_index - 1])
        if measured_speed > cruise_mps:
            prefix_arc = LineString(smooth[:, :2]).project(Point(xyz[prefix_index, :2]))
            # Squared speed is affine in braking distance. Extending that affine
            # expression behind the intervention avoids a sampled cap kink at
            # the exact prefix boundary; those earlier samples are never driven.
            cruise_cap = np.sqrt(np.maximum(cruise_mps**2,
                measured_speed**2 - 2 * initial_cruise_deceleration_mps2 * (arc - prefix_arc)))
    limits = np.minimum(cruise_cap, np.sqrt(lateral_budget_mps2 / np.maximum(curvature, 1e-8)))
    limits[-1] = 0.
    if terminal_standoff_m:
        if terminal_standoff_m >= arc[-1]:
            raise ValueError("terminal standoff consumes the entire confirmed path")
        stop_index = max(0, int(np.searchsorted(arc, arc[-1] - terminal_standoff_m, side="right")) - 1)
        limits[stop_index:] = 0.
    for i in range(len(limits) - 2, -1, -1):
        limits[i] = min(limits[i], np.sqrt(limits[i + 1]**2 + 2 * acceleration_mps2 * steps[i]))
    index = int(np.flatnonzero(t == intervention_s)[0])
    position_s = LineString(smooth[:, :2]).project(Point(xyz[index, :2]))
    if not 1e-6 < position_s < arc[-1] - 1e-6:
        raise ValueError("intervention projects onto centerline endpoint")
    center_start = np.array([np.interp(position_s, arc, smooth[:, i]) for i in range(3)])
    offset = xyz[index] - center_start
    start_position_s = position_s
    direction_offset = np.zeros(2)
    if align_initial_heading:
        tangent = np.array([np.interp(position_s, arc, first[:, k]) for k in (0, 1)])
        tangent /= np.linalg.norm(tangent)
        direction_offset = np.array([np.cos(yaw[index]), np.sin(yaw[index])]) - tangent
    speed = np.linalg.norm(xyz[index, :2] - xyz[index - 1, :2]) / (t[index] - t[index - 1])
    initial_cap = (np.sqrt(np.interp(position_s, arc, limits**2)) if speed_step_policy == "actual_advance_bisection"
                   else np.interp(position_s, arc, limits))
    if speed > initial_cap + .05:
        raise ValueError("recorded prefix leaves insufficient braking distance for this speed policy: "
                         f"recorded_speed_mps={speed:.6f}, initial_cap_mps={initial_cap:.6f}, "
                         f"cruise_mps={cruise_mps:.6f}, remaining_centerline_m={arc[-1] - position_s:.6f}")
    positions = xyz.copy()
    for i in range(index + 1, len(t)):
        dt = t[i] - t[i - 1]
        if speed_step_policy == "actual_advance_bisection":
            next_speed = bounded_speed_step(position_s, speed, dt, arc, limits, acceleration_mps2)
        else:
            # Preserve legacy plans exactly unless the new policy is explicit.
            lookahead = position_s + speed * dt + .5 * acceleration_mps2 * dt**2
            if lookahead >= arc[-1]:
                raise ValueError("plan reaches the confirmed centerline end")
            cap = np.interp(lookahead, arc, limits)
            next_speed = min(speed + acceleration_mps2 * dt, cap)
            if next_speed < speed - acceleration_mps2 * dt - .05:
                raise ValueError("discrete speed policy requires excessive braking")
        position_s += .5 * (speed + next_speed) * dt
        if position_s >= arc[-1]:
            raise ValueError("plan would extrapolate beyond confirmed centerline")
        positions[i] = [np.interp(position_s, arc, smooth[:, dim]) for dim in range(3)]
        # Distance-based alignment cannot advance while the car is at rest.
        # This modifies the proposed path only; independent turning, geometry,
        # measured acceleration and traffic checks remain mandatory.
        align = smoothstep5((t[i] - intervention_s) / alignment_s if alignment_distance_m is None
                            else (position_s - start_position_s) / alignment_distance_m)
        positions[i, :2] += (1 - align) * offset[:2]
        if align_initial_heading:
            # Match the initial path derivative as well as its position. This
            # is an explicit new curve, not an in-place rotation or map repair.
            positions[i, :2] += (1 - align) * (position_s - start_position_s) * direction_offset
        positions[i, 2] += offset[2]
        speed = next_speed
    velocity = np.gradient(positions[:, :2], t, axis=0)
    headings = np.arctan2(velocity[:, 1], velocity[:, 0])
    headings[t <= intervention_s] = yaw[t <= intervention_s]
    if speed_step_policy == "actual_advance_bisection":
        for i in range(index + 1, len(t)):
            if np.linalg.norm(velocity[i]) < 1e-6:
                headings[i] = headings[i - 1]
    return positions, headings
