"""Explicit common braking hypotheses; never edits or attests a recorded stop."""
import numpy as np


def recorded_path_stop(times, recorded_xyz, recorded_yaws, *, brake_start_s,
                       brake_duration_s, hold_s, intervention_s=1.7):
    """Slow a recording clock smoothly to zero, then retain one pose.

    The integral of 1 - quintic smoothstep gives the reference time exactly.
    A caller may hand this *planned* common approach to a branch planner after
    the hold. It must independently check the stop line, full footprint, road,
    dynamics, traffic, rendering and actual vehicle stop. No legality is implied.
    """
    t, xyz, yaw = [np.asarray(v, dtype=float) for v in (times, recorded_xyz, recorded_yaws)]
    values = (brake_start_s, brake_duration_s, hold_s, intervention_s)
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or not all(np.isfinite(v).all() for v in (t, xyz, yaw))
        or any(isinstance(v, (bool, np.bool_)) for v in values)
        or not np.isfinite(values).all() or t[0] != 0
        or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)
        or not 0 < intervention_s <= brake_start_s
        or not 2 <= brake_duration_s <= 10 or not .5 <= hold_s <= 3):
        raise ValueError("dense recorded source and bounded explicit common stop required")
    stop_s = brake_start_s + brake_duration_s
    handover_s = round(stop_s + hold_s, 6)
    if (handover_s >= t[-1] or any(not np.any(np.isclose(t, s, atol=1e-8, rtol=0))
        for s in (intervention_s, brake_start_s, stop_s, handover_s))):
        raise ValueError("common stop boundaries must be exact samples within the horizon")
    u = np.clip((t - brake_start_s) / brake_duration_s, 0., 1.)
    clock = brake_start_s + brake_duration_s * (u - 2.5*u**4 + 3*u**5 - u**6)
    clock[t <= brake_start_s] = t[t <= brake_start_s]
    planned = np.column_stack([np.interp(clock, t, xyz[:, k]) for k in range(3)])
    headings = np.interp(clock, t, np.unwrap(yaw))
    headings = np.arctan2(np.sin(headings), np.cos(headings))
    # Keep source prefix bit-for-bit, including unwrapped input yaw conventions.
    planned[t <= brake_start_s] = xyz[t <= brake_start_s]
    headings[t <= brake_start_s] = yaw[t <= brake_start_s]
    return planned, headings, {
        "scope": "planned_common_recording_clock_stop_not_recorded_or_legal_stop",
        "qualified": False, "human_reviewed": False, "original_recording_changed": False,
        "stop_legality_verified": False, "actual_stop_verified": False,
        "brake_start_s": float(brake_start_s), "brake_duration_s": float(brake_duration_s),
        "hold_s": float(hold_s), "stop_s": float(stop_s), "handover_s": handover_s,
        "stationary_source_time_s": float(brake_start_s + brake_duration_s / 2),
        "stationary_xyz": planned[np.flatnonzero(t >= stop_s)[0]].tolist(),
        "intervention_s": float(intervention_s),
    }


def recheck_common_stop(search, geometry, times, rows):
    """Rebuild any selected common stop from untouched source before GPU work."""
    parameters = search["selected_parameters"].get("shared_planned_stop")
    diagnostic = search.get("shared_planned_stop")
    if parameters is None and diagnostic is None:
        return
    if not isinstance(parameters, dict) or set(parameters) != {"brake_start_s", "brake_duration_s", "hold_s"}:
        raise ValueError("explicit selected common stop parameters required")
    xyz, yaw, expected = recorded_path_stop(times, geometry["recorded_xyz"], geometry["recorded_yaws"], **parameters)
    if (diagnostic != expected or search.get("shared_recorded_approach_end_s") != parameters["brake_start_s"]
        or expected["handover_s"] >= search["rollout_duration_s"]):
        raise ValueError("common stop source, timing or diagnostic changed")
    reference = np.column_stack((xyz, yaw))
    common = times <= expected["handover_s"]
    if any(not np.array_equal(row[common, 1:], reference[common]) for row in rows.values()):
        raise ValueError("selected plans differ from rebuilt common stopping approach")
