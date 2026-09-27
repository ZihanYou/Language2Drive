"""A4 original three-lane bindings and paired plan semantics, not qualification."""
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .lane_path_search import lane_pair_semantic_prefilter
from .bilateral_protocol import validate_dwell_start


def three_lane_prefix_binding(vector_map, middle_ids, left_ids, right_ids, times, positions, yaws):
    """Require a unique original middle lane with reciprocal neighbors at 0/1.7 s."""
    t, xyz, yaw = (np.asarray(v, dtype=float) for v in (times, positions, yaws))
    if (t.ndim != 1 or len(t) < 2 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or not all(np.isfinite(v).all() for v in (t, xyz, yaw))
        or t[0] != 0 or t[-1] < 1.7 or np.any(np.diff(t) <= 0) or np.any(np.diff(t) > .200001)):
        raise ValueError("dense finite original poses through intervention required")
    groups = [set(map(str, ids)) for ids in (middle_ids, left_ids, right_ids)]
    if any(not group for group in groups):
        raise ValueError("three original corridor identities required")

    def original_lane(key):
        lane = vector_map.get_road_lane(key)
        if lane.left_edge is None or lane.right_edge is None or lane.center is None:
            raise ValueError("cannot synthesize missing original lane boundaries")
        edges = [np.asarray(v.xyz, dtype=float) for v in (lane.left_edge, lane.right_edge, lane.center)]
        if any(v.ndim != 2 or v.shape[1] != 3 or len(v) < 2 or not np.isfinite(v).all() for v in edges):
            raise ValueError("finite original lane geometry required")
        polygon = Polygon(np.concatenate((edges[0][:, :2], edges[1][::-1, :2])))
        center = LineString(edges[2][:, :2])
        if not polygon.is_valid or polygon.area <= 0 or center.length <= 0 or not center.is_simple:
            raise ValueError("cannot repair original lane geometry")
        return lane, polygon, center

    def heading_matches(center, point, heading):
        arc = center.project(point)
        a, b = center.interpolate(max(0., arc - .1)), center.interpolate(min(center.length, arc + .1))
        delta = np.arctan2(b.y - a.y, b.x - a.x) - heading
        return abs(np.arctan2(np.sin(delta), np.cos(delta))) <= np.pi / 6

    checks = []
    for stamp in (0., 1.7):
        point = Point([np.interp(stamp, t, xyz[:, i]) for i in (0, 1)])
        heading = float(np.interp(stamp, t, np.unwrap(yaw)))
        matches = []
        for key in sorted(groups[0]):
            lane, polygon, center = original_lane(key)
            if polygon.contains(point) and point.distance(polygon.boundary) >= .15 and heading_matches(center, point, heading):
                matches.append((key, lane))
        if len(matches) != 1:
            raise ValueError(f"no unique original middle lane at {stamp:g} s")
        key, lane = matches[0]
        neighbors = []
        for group, adjacent, reciprocal in ((groups[1], lane.adj_lanes_left, "adj_lanes_right"),
                                             (groups[2], lane.adj_lanes_right, "adj_lanes_left")):
            selected = group.intersection(map(str, adjacent))
            if len(selected) != 1:
                raise ValueError(f"both original immediate side lanes required at {stamp:g} s")
            identifier = selected.pop()
            other, other_polygon, center = original_lane(identifier)
            if key not in set(map(str, getattr(other, reciprocal))) or not heading_matches(center, point, heading):
                raise ValueError("side lane must be reciprocal and same-direction, not an oncoming lane")
            if other_polygon.covers(point) or other_polygon.distance(point) < .15:
                raise ValueError("initial middle-lane pose is ambiguous with a side lane")
            neighbors.append(identifier)
        if len({key, *neighbors}) != 3:
            raise ValueError("three different original lane IDs required")
        checks.append({"time_s": stamp, "middle_lane": key, "left_lane": neighbors[0], "right_lane": neighbors[1]})
    return {"scope": "original_three_lane_initial_and_instruction_binding_not_legality", "passed": True,
            "qualified": False, "checks": checks, "geometry_modified": False}


def bilateral_lane_semantic_prefilter(times, left_xyz, right_xyz, middle_region, left_region, right_region, *, dwell_start_s=6.):
    """Both arms start in the middle and commit to exclusive opposite side lanes."""
    dwell_start_s = validate_dwell_start(dwell_start_s)
    end = dwell_start_s + 1.
    t, left, right = (np.asarray(v, dtype=float) for v in (times, left_xyz, right_xyz))
    if t.ndim != 1 or len(t) < 2 or t[-1] < 7.:
        raise ValueError("complete seven-second task required even with earlier dwell")
    # Reuse the dense, finite, endpoint and whole-line checks. There is no keep
    # arm here: restrict that helper's active occupancy check to the final dwell.
    final = lane_pair_semantic_prefilter(t, left, right, left_region, right_region,
                                         rollout_end_s=end, active_start_s=dwell_start_s, check_linear_segments=True)
    if not middle_region.is_valid or middle_region.is_empty or middle_region.area <= 0:
        raise ValueError("valid original middle-lane region required")
    prefix = t <= 1.7
    if t[0] != 0 or not np.any(t == 1.7):
        raise ValueError("explicit initial and intervention plan samples required")
    same_prefix = bool(np.array_equal(left[prefix], right[prefix]))
    prefix_xy = left[prefix, :2]
    path = Point(prefix_xy[0]) if np.all(prefix_xy == prefix_xy[0]) else LineString(prefix_xy)
    prefix_middle = bool(middle_region.contains(path) and path.distance(middle_region.boundary) >= .15)
    endpoints = [Point(left[i, :2]) for i in (0, int(np.flatnonzero(t == 1.7)[0]))]
    starts_middle = all(middle_region.contains(point) and point.distance(middle_region.boundary) >= .15
                        for point in endpoints)
    grid = np.unique(np.r_[dwell_start_s, t[(t > dwell_start_s) & (t < end)], end])
    outside_middle = {}
    for arm, xyz in (("R0", left), ("R1", right)):
        points = np.column_stack([np.interp(grid, t, xyz[:, i]) for i in (0, 1)])
        dwell = Point(points[0]) if np.all(points == points[0]) else LineString(points)
        outside_middle[arm] = bool(not dwell.intersects(middle_region) and dwell.distance(middle_region) >= .15)
    return {"scope": "bilateral_lane_plan_semantics_not_qualified_or_actual_ASL", "qualified": False,
        "passed": same_prefix and starts_middle and final["passed"] and all(outside_middle.values()),
        "shared_prefix_equal": same_prefix, "prefix_inside_original_middle": prefix_middle,
        "initial_and_instruction_in_original_middle": starts_middle,
        "prefix_road_safety_requires_separate_native_footprint_check": True,
        "exclusive_final_dwell_both_arms": final["exclusive_final_dwell_both_arms"],
        "final_nonexclusive_times_s": {"R0": final["keep_nonexclusive_final_times_s"],
                                       "R1": final["change_nonexclusive_final_times_s"]},
        "final_dwell_outside_middle_by_arm": outside_middle, "dwell_interval_s": [dwell_start_s, end],
        "boundary_margin_m": .15, "linear_segment_check": {
            "R0_left_exclusive": final["linear_segment_check"]["keep_final_exclusive"],
            "R1_right_exclusive": final["linear_segment_check"]["change_final_exclusive"]}}


def actual_bilateral_lane_prefilter(left_trace, right_trace, middle_region, left_region, right_region, *, dwell_start_s=6.):
    """Use exact raw prefix rows, with interpolation only for geometry boundaries.

    Inputs are actual ASL rig-origin [time,x,y,z] samples. This does not prove
    equal camera inputs; the separate actual ASL prefix audit remains mandatory.
    """
    dwell_start_s = validate_dwell_start(dwell_start_s)
    end = dwell_start_s + 1.
    traces = [np.asarray(v, dtype=float) for v in (left_trace, right_trace)]
    if any(v.ndim != 2 or v.shape[1] != 4 or len(v) < 2 or not np.isfinite(v).all()
           or v[0, 0] != 0 or v[-1, 0] < 7 or np.any(np.diff(v[:, 0]) <= 0)
           or np.any(np.diff(v[:, 0]) > .200001) for v in traces):
        raise ValueError("two complete dense actual rig traces required")
    if not middle_region.is_valid or middle_region.is_empty or middle_region.area <= 0:
        raise ValueError("valid original middle geometry required")
    prefixes = [v[v[:, 0] <= 1.7] for v in traces]
    same_prefix = bool(np.array_equal(*prefixes))
    grid = np.unique(np.concatenate([v[:, 0] for v in traces]))
    aligned = [np.column_stack([np.interp(grid, v[:, 0], v[:, i]) for i in (1, 2, 3)]) for v in traces]
    final = lane_pair_semantic_prefilter(grid, *aligned, left_region, right_region,
                                        rollout_end_s=end, active_start_s=dwell_start_s, check_linear_segments=True)
    middle, outside = {}, {}
    for arm, trace in zip(("R0", "R1"), traces):
        middle[arm] = all(middle_region.contains(point) and point.distance(middle_region.boundary) >= .15
            for point in (Point([np.interp(t, trace[:, 0], trace[:, i]) for i in (1, 2)]) for t in (0., 1.7)))
        dwell_t = np.unique(np.r_[dwell_start_s, trace[(trace[:, 0] > dwell_start_s) & (trace[:, 0] < end), 0], end])
        xy = np.column_stack([np.interp(dwell_t, trace[:, 0], trace[:, i]) for i in (1, 2)])
        path = Point(xy[0]) if np.all(xy == xy[0]) else LineString(xy)
        outside[arm] = bool(not path.intersects(middle_region) and path.distance(middle_region) >= .15)
    return {"scope": "actual_ASL_bilateral_rig_geometry_not_full_verifier_or_qualification",
        "qualified": False, "passed": same_prefix and final["passed"] and all(middle.values()) and all(outside.values()),
        "raw_preintervention_pose_rows_equal": same_prefix, "raw_prefix_samples": [len(p) for p in prefixes],
        "camera_and_non_language_prefix_requires_separate_ASL_audit": True,
        "geometry_boundary_interpolation_s": [0., 1.7, dwell_start_s, end],
        "initial_and_instruction_middle_by_arm": middle, "final_dwell_outside_middle_by_arm": outside,
        "final_lane_occupancy": final}


def hold_completed_stop_heading(times, positions, headings, completion_s):
    """Preserve heading through near-stationary pose jitter after a finished blend.

    This is a new plan hypothesis, not a correction of actual vehicle state.
    No positions change. Full turning, motion, road and traffic checks follow.
    """
    t, xyz, yaw = (np.asarray(v, dtype=float) for v in (times, positions, headings))
    if (t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3) or yaw.shape != t.shape
        or not all(np.isfinite(v).all() for v in (t, xyz, yaw)) or np.any(np.diff(t) <= 0)
        or np.any(np.diff(t) > .200001) or not np.isfinite(completion_s) or not 1.7 < completion_s < t[-1]):
        raise ValueError("dense finite plan and a completed post-intervention lane blend required")
    step_speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / np.diff(t)
    result = yaw.copy()
    changed = []
    for i in range(1, len(t)):
        local_speed = max(step_speed[i-1], step_speed[min(i, len(step_speed)-1)])
        if t[i] >= completion_s and local_speed <= .2:
            if result[i] != result[i-1]: changed.append(float(t[i]))
            result[i] = result[i-1]
    return result, {"scope": "planned_completed_stop_heading_hypothesis_not_actual_vehicle_state",
        "qualified": False, "maximum_local_step_speed_mps": .2, "completion_s": completion_s,
        "changed_times_s": changed, "positions_changed": False, "subsequent_safety_checks_required": True}
