"""Reusable sampled lane traffic/dynamics checks; never a continuous certificate."""
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon


def footprint(matrix, length, width):
    points = np.asarray([[x, y, 0., 1.] for x, y in (
        (-length / 2, -width / 2), (length / 2, -width / 2),
        (length / 2, width / 2), (-length / 2, width / 2))])
    return Polygon((points @ matrix.T)[:, :2])


class PlannedLaneMotionAudit:
    """Load each real actor only once per bounded scene/window search."""
    def __init__(self, artifact, times, timestamps_us, rig_to_aabb):
        self.artifact = artifact
        self.times = np.asarray(times, dtype=float)
        self.timestamps = np.asarray(timestamps_us)
        self.transform = np.asarray(rig_to_aabb, dtype=float)
        vehicle = artifact.rig.vehicle_config
        if (self.times.ndim != 1 or len(self.times) < 3 or self.timestamps.shape != self.times.shape
            or not np.isfinite(self.times).all() or self.times[0] != 0 or self.times[-1] < 13
            or np.any(np.diff(self.times) <= 0) or np.any(np.diff(self.times) > .200001)
            or self.timestamps.dtype.kind not in "iu" or np.any(self.timestamps < 0)
            or self.transform.shape != (4, 4) or not np.isfinite(self.transform).all()
            or not np.allclose(self.transform[3], [0., 0., 0., 1.])
            or vehicle is None or not np.isfinite([vehicle.aabb_x_m, vehicle.aabb_y_m]).all()
            or min(vehicle.aabb_x_m, vehicle.aabb_y_m) <= 0
            or not np.array_equal(self.timestamps, self.timestamps[0] + np.rint(self.times * 1e6).astype(np.uint64))):
            raise ValueError("dense complete 13-second timing and exact vehicle transform required")
        self.traffic = None

    def audit(self, positions, headings):
        xyz, yaw = np.asarray(positions, dtype=float), np.asarray(headings, dtype=float)
        if xyz.shape != (len(self.times), 3) or yaw.shape != self.times.shape or not all(np.isfinite(v).all() for v in (xyz, yaw)):
            raise ValueError("finite complete planned poses required")
        vehicle = self.artifact.rig.vehicle_config
        if self.traffic is None:
            self.traffic = [[(key, footprint(actor.trajectory.interpolate_pose(int(stamp)).as_se3(),
                                             actor.aabb.x, actor.aabb.y))
                             for key, actor in self.artifact.traffic_objects.items()
                             if int(stamp) in actor.trajectory.time_range_us] for stamp in self.timestamps]
        minimum, overlap = float("inf"), []
        for i, (position, heading) in enumerate(zip(xyz, yaw)):
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler("z", heading).as_matrix()
            matrix[:3, 3] = position
            ego = footprint(matrix @ self.transform, vehicle.aabb_x_m, vehicle.aabb_y_m)
            for key, actor in self.traffic[i]:
                minimum = min(minimum, ego.distance(actor))
                if ego.intersection(actor).area > 0:
                    overlap.append({"time_s": float(self.times[i]), "actor_id": key})
        velocity = np.gradient(xyz[:, :2], self.times, axis=0)
        acceleration = np.gradient(velocity, self.times, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        yaw_rate = np.gradient(np.unwrap(yaw), self.times)
        active = (self.times > 1.8) & (self.times < self.times[-1] - .1)
        active_indices = np.flatnonzero(active)
        planar = np.linalg.norm(acceleration[active], axis=1)
        lateral = np.abs(speed[active] * yaw_rate[active])
        at_seven = np.asarray([np.interp(7., self.times, xyz[:, i]) for i in (0, 1)])
        return {"minimum_2d_clearance_m": None if not np.isfinite(minimum) else minimum,
                "audited_horizon_s": float(self.times[-1]),
                "dynamics_open_interval_s": [1.8, float(self.times[-1] - .1)],
                "overlap_sample_count": len(overlap), "first_overlaps": overlap[:3],
                "max_planar_acceleration_mps2": float(planar.max()),
                "max_planar_acceleration_time_s": float(self.times[active_indices[planar.argmax()]]),
                "max_lateral_acceleration_mps2": float(lateral.max()),
                "max_lateral_acceleration_time_s": float(self.times[active_indices[lateral.argmax()]]),
                "progress_at_7s_m": float(np.linalg.norm(at_seven - xyz[0, :2]))}


def sampled_lane_motion_passed(audit):
    return (audit["minimum_2d_clearance_m"] is not None and audit["minimum_2d_clearance_m"] >= .5
            and audit["overlap_sample_count"] == 0 and audit["max_planar_acceleration_mps2"] <= 3.5
            and audit["max_lateral_acceleration_mps2"] <= 3. and audit["progress_at_7s_m"] >= 10.)
