"""Cheap recorded-viewpoint proximity diagnostics, never a support certificate."""
import numpy as np
from shapely.geometry import LineString, Point


def recorded_path_proximity(times, planned_xyz, recorded_xyz, deadline_s):
    """Compare rig positions on one original time grid without loading an asset.

    Nearest distance uses only the supplied recorded interval, including its
    future samples; it says nothing about unseen reconstruction support.
    No threshold or pass decision is implied by this advisory diagnostic.
    """
    t = np.asarray(times, dtype=float)
    planned = np.asarray(planned_xyz, dtype=float)
    recorded = np.asarray(recorded_xyz, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or planned.shape != (len(t), 3)
        or recorded.shape != planned.shape or not np.isfinite(t).all()
        or not np.isfinite(planned).all() or not np.isfinite(recorded).all()
        or np.any(np.diff(t) <= 0) or t[0] < 0
        or isinstance(deadline_s, bool) or not isinstance(deadline_s, (float, int))
        or not np.isfinite(deadline_s) or not t[0] < deadline_s <= t[-1]):
        raise ValueError("finite matching original rig grids and covered deadline required")
    # A stationary recording is a point, not an invalid zero-length line.
    route = LineString(recorded[:, :2])
    if route.length == 0:
        route = Point(recorded[0, :2])
    nearest = np.array([route.distance(Point(xy)) for xy in planned[:, :2]])
    same_time = np.linalg.norm(planned[:, :2] - recorded[:, :2], axis=1)
    rows = [dict(time_s=float(at), nearest_supplied_recorded_path_m=float(near),
                 same_time_recorded_offset_m=float(offset))
            for at, near, offset in zip(t, nearest, same_time)]
    task = t <= deadline_s
    future = t > deadline_s
    return dict(scope="recorded_path_proximity_advisory_not_reconstruction_support",
        qualified=False, reconstruction_support_verified=False, threshold_applied=False,
        position_reference="rig_origin_planar_xy", recorded_interval_s=[float(t[0]), float(t[-1])],
        deadline_s=float(deadline_s),
        task_max_nearest_recorded_path_m=float(nearest[task].max()),
        future_max_nearest_recorded_path_m=float(nearest[future].max()) if future.any() else None,
        task_max_same_time_recorded_offset_m=float(same_time[task].max()), samples=rows)
