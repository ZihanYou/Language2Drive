"""Native-rule planned footprints, not actual rollout or continuous support proof."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
from shapely.ops import unary_union


def audit_planned_native_road(vector_map, times, positions, headings, rig_to_aabb, length_m, width_m):
    """Use pinned OffRoadScorer on a synthetic plan, never on claimed actual ASL.

    Native offroad is not identical to membership in a lane-polygon union while
    changing lanes. Reject a start outside the original lanes (a documented
    native blind spot), and reject missing/invalid boundaries before the native
    scorer can synthesize or repair lane polygons. Keep all native alarms.
    """
    from alpasim_utils.geometry import Pose
    from eval.scorers.offroad import OffRoadScorer

    t, xyz, yaw, transform = [np.asarray(v, dtype=float) for v in (times, positions, headings, rig_to_aabb)]
    if (t.ndim != 1 or len(t) < 2 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or transform.shape != (4, 4) or not all(np.isfinite(v).all() for v in (t, xyz, yaw, transform))
        or t[0] != 0 or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)
        or not np.isfinite([length_m, width_m]).all() or min(length_m, width_m) <= 0
        or not np.allclose(transform[3], [0., 0., 0., 1.])):
        raise ValueError("complete dense finite planned poses and exact vehicle transform required")
    timestamps = np.rint(t * 1e6).astype(np.uint64)
    if np.any(np.diff(timestamps) == 0):
        raise ValueError("distinct microsecond plan samples required")
    corners = np.asarray([[x, y, 0., 1.] for x, y in (
        (-length_m / 2, -width_m / 2), (length_m / 2, -width_m / 2),
        (length_m / 2, width_m / 2), (-length_m / 2, width_m / 2))])
    poses, polygons, lane_cache = {}, {}, {}
    lane_ids = set()
    starts_on_original_road = False
    for i, timestamp in enumerate(timestamps):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("z", yaw[i]).as_matrix()
        matrix[:3, 3] = xyz[i]
        matrix = matrix @ transform
        pose = Pose(matrix[:3, 3].astype(np.float32), Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32))
        polygon = Polygon((corners @ matrix.T)[:, :2])
        xyzh = np.r_[pose.vec3, pose.yaw()]
        lanes = vector_map.get_current_lane(xyzh, max_dist=6, max_heading_error=2*np.pi)
        originals = []
        for lane in lanes:
            if lane.left_edge is None or lane.right_edge is None:
                raise ValueError("native plan check cannot synthesize missing lane boundaries")
            original = Polygon(np.concatenate((lane.left_edge.points[:, :2], lane.right_edge.points[::-1, :2])))
            if not original.is_valid or original.area <= 0:
                raise ValueError("native plan check cannot repair invalid original lane polygons")
            originals.append(original)
            lane_ids.add(str(lane.id))
        if i == 0:
            starts_on_original_road = bool(originals and unary_union(originals).contains(polygon))
        poses[int(timestamp)], polygons[int(timestamp)] = pose, polygon
        lane_cache[tuple(xyzh)] = lanes

    # The native implementation asks for these exact poses and polygon samples.
    # Intercept its map lookup so every polygon it uses was prevalidated above.
    checked_map = SimpleNamespace(
        get_current_lane=lambda xyzh, **kwargs: lane_cache[tuple(xyzh)],
        get_closest_road_edge=vector_map.get_closest_road_edge)
    sim = SimpleNamespace(timestamps_us=timestamps, vec_map=checked_map,
        actor_trajectories={"EGO": SimpleNamespace(interpolate_poses_list=lambda ts: [poses[int(v)] for v in ts])},
        actor_polygons=SimpleNamespace(
            get_polygon_for_agent_at_time=lambda agent, ts: polygons[int(ts)],
            get_yaw_for_agent_at_time=lambda agent, ts: poses[int(ts)].yaw()))
    metrics = {metric.name: metric for metric in OffRoadScorer(None).calculate(sim)}
    alarms = {name: [float(t[i]) for i, value in enumerate(metrics[name].values) if value]
              for name in ("offroad", "wrong_lane")}
    return {"scope": "planned_native_road_rule_not_actual_ASL_or_continuous_certificate",
        "passed": starts_on_original_road and not any(alarms.values()), "qualified": False,
        "source": "pinned_eval.scorers.offroad.OffRoadScorer", "samples": len(t),
        "interval_s": [float(t[0]), float(t[-1])], "start_footprint_inside_original_lanes": starts_on_original_road,
        "native_alarm_times_s": alarms, "queried_original_lane_ids": sorted(lane_ids),
        "original_geometry_modified": False, "rig_orientation_assumption": "planned_yaw_only_then_exact_rig_to_aabb"}
