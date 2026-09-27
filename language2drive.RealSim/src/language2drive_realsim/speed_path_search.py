"""Unreviewed B1 path hypotheses; neither legal speeds nor frozen scopes."""
import numpy as np

from .lane_path_search import smoothstep5


def validate_speed_targets_kph(targets_kph, half_width_kph=2.):
    """Validate authoring bands; narrower bands never relax the verifier epsilon."""
    targets = np.asarray(targets_kph, dtype=float)
    if isinstance(half_width_kph, (bool, np.bool_)):
        raise ValueError("speed-band half width must be numeric, not boolean")
    try:
        half = float(half_width_kph)
    except (TypeError, ValueError) as error:
        raise ValueError("finite positive speed-band half width required") from error
    if (targets.shape != (2,) or not np.isfinite(targets).all()
        or not np.isfinite(half) or not 0 < half <= 2
        or targets[0] <= half or targets[1] - targets[0] <= 2 * half):
        raise ValueError("positive finite targets and disjoint bands with half width at most 2 km/h required")
    return targets, half


def validate_terminal_stop_window(window, earliest_s, horizon_s):
    """An optional, explicit stop cannot overlap the caller's active interval."""
    if window is None:
        return None
    try:
        values = np.asarray(window, dtype=object)
        if values.shape != (2,) or any(isinstance(v, (bool, np.bool_)) for v in values):
            raise ValueError("invalid terminal stop window")
        start, end = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid terminal stop window") from error
    if not np.isfinite([start, end, earliest_s, horizon_s]).all() or not earliest_s <= start < end <= horizon_s:
        raise ValueError("terminal stop window must follow the active interval and fit the horizon")
    return float(start), float(end)


def constant_speed_path(times, recorded_xyz, recorded_yaws, target_mps,
                        intervention_s=1.7, transition_s=3.0, *, spatial_xyz=None, spatial_yaws=None,
                        terminal_stop_s=None):
    """Retain the exact prefix and follow the same recorded spatial polyline.

    Integrate a smooth transition from the measured initial speed. Refuse to
    extrapolate beyond recorded geometry or invent motion through a stop.
    An optional terminal stop changes only timing, never extends the geometry.
    The caller must keep it after the task and audit traffic at the NEW timestamps.
    """
    times = np.asarray(times, dtype=float)
    xyz = np.asarray(recorded_xyz, dtype=float)
    yaws = np.asarray(recorded_yaws, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or xyz.shape != (len(times), 3)
        or yaws.shape != times.shape or times[0] != 0 or np.any(np.diff(times) <= 0)
        or not all(np.isfinite(v).all() for v in (times, xyz, yaws))
        or not np.isfinite([target_mps, intervention_s, transition_s]).all()
        or target_mps <= 0 or transition_s <= 0 or intervention_s <= 0
        or intervention_s + transition_s >= times[-1]):
        raise ValueError("invalid speed path inputs")
    indices = np.flatnonzero(np.isclose(times, intervention_s, atol=1e-9, rtol=0))
    if len(indices) != 1 or indices[0] == 0:
        raise ValueError("time grid must include the exact intervention boundary")
    index = int(indices[0])
    distances = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    if np.any(distances <= 1e-6):
        raise ValueError("recorded spatial path contains a stop or duplicate vertex")
    arc = np.concatenate(([0.0], np.cumsum(distances)))
    reference_xyz, reference_yaws, reference_arc = xyz, yaws, arc
    if spatial_xyz is not None or spatial_yaws is not None:
        reference_xyz, reference_yaws = np.asarray(spatial_xyz, dtype=float), np.asarray(spatial_yaws, dtype=float)
        if (reference_xyz.ndim != 2 or reference_xyz.shape[1] != 3 or len(reference_xyz) < len(xyz)
            or reference_yaws.shape != (len(reference_xyz),) or not np.isfinite(reference_xyz).all()
            or not np.isfinite(reference_yaws).all() or not np.array_equal(reference_xyz[:len(xyz)], xyz)
            or not np.array_equal(reference_yaws[:len(yaws)], yaws)):
            raise ValueError("extended recorded geometry must retain the exact original samples")
        reference_distances = np.linalg.norm(np.diff(reference_xyz[:, :2], axis=0), axis=1)
        if np.any(reference_distances <= 1e-6):
            raise ValueError("extended recorded spatial path contains a stop or duplicate vertex")
        reference_arc = np.concatenate(([0.], np.cumsum(reference_distances)))
    initial_speed = (arc[index] - arc[index - 1]) / (times[index] - times[index - 1])
    speed = initial_speed + (target_mps - initial_speed) * smoothstep5((times - intervention_s) / transition_s)
    stop = validate_terminal_stop_window(terminal_stop_s, intervention_s + transition_s, times[-1])
    if stop is not None:
        start, end = stop
        speed *= 1 - smoothstep5((times - start) / (end - start))
    travelled = arc.copy()
    travelled[index + 1:] = arc[index] + np.cumsum(
        np.diff(times[index:]) * (speed[index:-1] + speed[index + 1:]) / 2)
    if travelled[-1] > reference_arc[-1] + 1e-9:
        raise ValueError("speed path would require endpoint extrapolation")
    positions = np.column_stack([np.interp(travelled, reference_arc, reference_xyz[:, a]) for a in range(3)])
    headings = np.interp(travelled, reference_arc, np.unwrap(reference_yaws))
    positions[:index + 1] = xyz[:index + 1]
    headings[:index + 1] = yaws[:index + 1]
    return positions, headings


def speed_band_prefilter(times, positions, lower_mps, upper_mps, active_start_s=5.0, active_end_s=7.0):
    """Check every segment speed in a provisional interval, never its mean."""
    times, positions = np.asarray(times, dtype=float), np.asarray(positions, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 3)
        or not np.isfinite(times).all() or not np.isfinite(positions).all()
        or not np.isfinite([lower_mps, upper_mps, active_start_s, active_end_s]).all()
        or not 0 <= lower_mps < upper_mps or not 0 <= active_start_s < active_end_s
        or times[0] > active_start_s or times[-1] < active_end_s
        or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 0.200001)):
        raise ValueError("invalid or incomplete speed prefilter trace")
    grid = np.unique(np.concatenate(([active_start_s, active_end_s],
                                    times[(times > active_start_s) & (times < active_end_s)])))
    points = np.column_stack([np.interp(grid, times, positions[:, a]) for a in range(3)])
    speed = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) / np.diff(grid)
    failed = (speed < lower_mps) | (speed > upper_mps)
    return {"scope": "sampled_speed_band_not_frozen_spatial_contract", "passed": not bool(failed.any()),
            "active_interval_s": [active_start_s, active_end_s], "band_mps": [lower_mps, upper_mps],
            "minimum_speed_mps": float(speed.min()), "maximum_speed_mps": float(speed.max()),
            "interval_count": len(speed), "failing_interval_start_times_s": grid[:-1][failed].tolist()}
