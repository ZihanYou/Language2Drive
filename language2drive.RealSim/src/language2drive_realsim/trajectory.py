"""Shared trajectory math without importing model/driver SDKs."""
import numpy as np


def sample_plan(plan, sample_times_s):
    """Interpolate a local-frame plan, including wrap-safe yaw."""
    waypoints = plan["waypoints"]
    times = np.asarray([float(row["time_s"]) for row in waypoints], dtype=float)
    positions = np.asarray([[float(row[key]) for key in ("x", "y", "z")]
                            for row in waypoints], dtype=float)
    yaws = np.unwrap(np.asarray([float(row["yaw_rad"]) for row in waypoints], dtype=float))
    queries = np.asarray(sample_times_s, dtype=float)
    return (np.column_stack([np.interp(queries, times, positions[:, axis]) for axis in range(3)]),
            np.interp(queries, times, yaws))
