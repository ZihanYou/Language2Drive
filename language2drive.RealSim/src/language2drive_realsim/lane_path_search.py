"""Exploratory parallel-lane plans, not legality or reconstruction evidence."""
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .compat import strict_zip


def validate_lane_change_grid(starts_s, duration_s):
    """Bound new timing hypotheses without changing the 1.7/7 s experiment."""
    starts = np.asarray(starts_s, dtype=float)
    if isinstance(duration_s, (bool, np.bool_)):
        raise ValueError("lane-change duration must be numeric, not boolean")
    duration = float(duration_s)
    if (starts.ndim != 1 or not 1 <= len(starts) <= 3 or len(set(starts)) != len(starts)
        or not np.isfinite(starts).all() or not np.isfinite(duration) or duration <= 0
        or np.any(starts < 1.7) or np.any(starts + duration > 7.)):
        raise ValueError("one to three distinct finite lane-change starts after intervention and completion by 7 s required")
    return starts.tolist(), duration


def _observed_predecessor(vector_map, candidates, prefix):
    """Resolve only previously observed approach lanes, never a future branch."""
    matches = []
    for key in candidates:
        lane = vector_map.get_road_lane(key)
        if lane.left_edge is None or lane.right_edge is None:
            raise ValueError("cannot resolve predecessor with missing original edges")
        polygon = Polygon(np.concatenate((lane.left_edge.xyz[:, :2], lane.right_edge.xyz[::-1, :2])))
        center = LineString(lane.center.xyz[:, :2])
        if not polygon.is_valid or polygon.area <= 0 or not center.is_simple or center.length <= 0:
            raise ValueError("cannot resolve predecessor with invalid original geometry")
        observed = []
        for t, x, y, _, yaw in prefix:
            point = Point(x, y)
            if not polygon.contains(point) or point.distance(polygon.boundary) < .15:
                continue
            arc = center.project(point)
            a, b = center.interpolate(max(0., arc - .01)), center.interpolate(min(center.length, arc + .01))
            delta = np.arctan2(b.y - a.y, b.x - a.x) - yaw
            if abs(np.arctan2(np.sin(delta), np.cos(delta))) <= np.pi / 6:
                observed.append(float(t))
        if observed:
            matches.append({"lane_id": key, "matching_preintervention_times_s": observed})
    return matches


def observed_predecessor_paths(vector_map, next_lane_id, candidates, prefix, max_depth=8):
    """Find unique original incoming paths to observed history, or fail closed.

    Intermediate sections are map hypotheses, not claimed observed poses. A
    truncated or cyclic search cannot disambiguate a predecessor. No proximity
    link, unreciprocated link, or endpoint gap above the existing limit is used.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 1 <= max_depth <= 8:
        raise ValueError("predecessor lookback depth must be an integer from 1 to 8")
    stack = [(str(key), [str(next_lane_id)]) for key in candidates]
    paths, unresolved, visits = [], [], 0
    while stack:
        key, downstream = stack.pop()
        visits += 1
        if visits > 100:
            unresolved.append("node_budget_exhausted")
            break
        if key in downstream:
            unresolved.append("cycle")
            continue
        lane = vector_map.get_road_lane(key)
        following = vector_map.get_road_lane(downstream[0])
        if downstream[0] not in {str(v) for v in lane.next_lanes}:
            unresolved.append("nonreciprocal_link")
            continue
        gap = float(np.linalg.norm(lane.center.xyz[-1, :2] - following.center.xyz[0, :2]))
        if not np.isfinite(gap) or gap > .5:
            unresolved.append("unsupported_endpoint_gap")
            continue
        match = _observed_predecessor(vector_map, [key], prefix)
        chain = [key] + downstream
        if match:
            paths.append({"lane_ids_observed_to_next": chain, "observed_match": match[0]})
            continue
        previous = sorted(str(v) for v in lane.prev_lanes if str(v) != "-1")
        if previous and len(downstream) >= max_depth:
            unresolved.append("depth_budget_exhausted")
            continue
        stack.extend((v, chain) for v in previous)
    return {"paths": paths, "unresolved": sorted(set(unresolved)), "visits": visits,
            "resolved": len(paths) == 1 and not unresolved,
            "source": "recorded_prefix_unique_original_incoming_path_not_observed_future"}


def lane_corridor(vector_map, lane_id, minimum_length_m=70.0, *, observed_prefix=None, history_audit=None,
                  predecessor_lookback_depth=0, backward_context_m=60.0, forward_choices=None,
                  backward_choices=None):
    """Use exact edges and unique links, including pre-window lane context."""
    if not np.isfinite(minimum_length_m) or not 0 < minimum_length_m <= 300:
        raise ValueError("corridor length request must be finite and within 300 meters")
    # Explicit privileged-oracle continuation, never inferred from future poses
    # or exposed as an arm-specific route in model inputs. Defaults still stop
    # at an ambiguous successor. Every supplied choice must actually be used.
    choices = {} if forward_choices is None else forward_choices
    if (not isinstance(choices, dict) or len(choices) > 8
        or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
               or k == v or k == "-1" or v == "-1" for k, v in choices.items())):
        raise ValueError("at most eight explicit original forward lane choices required")
    used_choices = set()
    incoming = {} if backward_choices is None else backward_choices
    if (not isinstance(incoming, dict) or len(incoming) > 8
        or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
               or k == v or k == "-1" or v == "-1" for k, v in incoming.items())):
        raise ValueError("at most eight explicit original backward lane choices required")
    used_incoming = set()
    if (isinstance(backward_context_m, (bool, np.bool_)) or not np.isfinite(backward_context_m)
        or not 0 < backward_context_m <= 180):
        raise ValueError("backward context must be finite and within 180 meters")
    if (isinstance(predecessor_lookback_depth, bool) or not isinstance(predecessor_lookback_depth, int)
        or not 0 <= predecessor_lookback_depth <= 8 or (predecessor_lookback_depth and observed_prefix is None)):
        raise ValueError("optional predecessor lookback requires real history and depth 1–8")
    if observed_prefix is not None:
        observed_prefix = np.asarray(observed_prefix, dtype=float)
        if (observed_prefix.ndim != 2 or observed_prefix.shape[1] != 5 or len(observed_prefix) < 2
            or not np.isfinite(observed_prefix).all() or observed_prefix[0, 0] != 0
            or observed_prefix[-1, 0] != 1.7 or np.any(np.diff(observed_prefix[:, 0]) <= 0)
            or np.any(np.diff(observed_prefix[:, 0]) > .200001)):
            raise ValueError("history resolution needs the dense exact recorded 0–1.7 second prefix")
    lanes, identifiers, forward_length = [], [], 0.0
    lane = vector_map.get_road_lane(lane_id)
    for _ in range(20):
        if str(lane.id) in identifiers:
            break
        identifiers.append(str(lane.id))
        lanes.append(lane)
        forward_length += LineString(lane.center.xyz[:, :2]).length
        if forward_length >= minimum_length_m:
            break
        successors = sorted(str(v) for v in lane.next_lanes if str(v) != "-1")
        explicit = str(lane.id) in choices
        if explicit:
            selected = choices[str(lane.id)]
            if selected not in successors or selected in identifiers or len(identifiers) >= 20:
                raise ValueError("explicit forward choice is not an unused original successor")
            successors = [selected]
        if len(successors) != 1:
            break
        following = vector_map.get_road_lane(successors[0])
        gap = np.linalg.norm(lane.center.xyz[-1, :2] - following.center.xyz[0, :2])
        if explicit and (str(lane.id) not in {str(v) for v in following.prev_lanes}
                         or not np.isfinite(gap) or gap > .5):
            raise ValueError("explicit forward choice lacks a reciprocal gap-safe original link")
        if gap > 0.5:
            break
        if explicit:
            used_choices.add(str(lane.id))
        lane = following
    if used_choices != set(choices):
        raise ValueError("unused explicit forward choice: path ended or traversal budget exhausted")
    # Review windows include history before the selected lane segment starts.
    # An optional real preintervention prefix may disambiguate incoming lanes;
    # it must never choose an unobserved forward branch or use future poses.
    backward_length = 0.0
    for _ in range(20):
        previous_ids = sorted(str(v) for v in lanes[0].prev_lanes if str(v) != "-1")
        next_id = str(lanes[0].id)
        explicit_incoming = next_id in incoming
        if explicit_incoming:
            # A privileged counterfactual approach hypothesis, not an assertion
            # that this lane was observed in the shared recorded history.
            selected = incoming[next_id]
            if selected not in previous_ids or selected in identifiers:
                raise ValueError("explicit backward choice is not an unused original predecessor")
            previous_ids = [selected]
        elif len(previous_ids) > 1 and observed_prefix is not None:
            if predecessor_lookback_depth:
                audit = observed_predecessor_paths(vector_map, str(lanes[0].id), previous_ids,
                                                   observed_prefix, predecessor_lookback_depth)
                if history_audit is not None:
                    history_audit.append({"next_lane": str(lanes[0].id), **audit})
                if not audit["resolved"]:
                    break
                previous_ids = [audit["paths"][0]["lane_ids_observed_to_next"][-2]]
            else:
                matches = _observed_predecessor(vector_map, previous_ids, observed_prefix)
                if history_audit is not None:
                    history_audit.append({"next_lane": str(lanes[0].id), "candidate_predecessors": previous_ids,
                                          "matches": matches, "resolved": len(matches) == 1,
                                          "source": "recorded_non_language_prefix_not_counterfactual_future"})
                if len(matches) != 1:
                    break
                selected = matches[0]["lane_id"]
                if str(lanes[0].id) not in {str(i) for i in vector_map.get_road_lane(selected).next_lanes}:
                    raise ValueError("observed predecessor lacks reciprocal directed map link")
                previous_ids = [selected]
        if len(previous_ids) != 1 or previous_ids[0] in identifiers:
            break
        previous = vector_map.get_road_lane(previous_ids[0])
        gap = np.linalg.norm(previous.center.xyz[-1, :2] - lanes[0].center.xyz[0, :2])
        if explicit_incoming and (next_id not in {str(v) for v in previous.next_lanes}
                                  or not np.isfinite(gap) or gap > .5):
            raise ValueError("explicit backward choice lacks a reciprocal gap-safe original link")
        if gap > 0.5:
            break
        if explicit_incoming:
            used_incoming.add(next_id)
            if history_audit is not None:
                history_audit.append(dict(next_lane=next_id, selected_predecessor=str(previous.id),
                    source="explicit_privileged_approach_hypothesis_not_observed_history", observed=False))
        lanes.insert(0, previous)
        identifiers.insert(0, str(previous.id))
        backward_length += LineString(previous.center.xyz[:, :2]).length
        if backward_length >= backward_context_m:
            break
    if used_incoming != set(incoming):
        raise ValueError("unused explicit backward choice: traversal ended or budget exhausted")
    points, polygons = [], []
    for lane in lanes:
        if lane.left_edge is None or lane.right_edge is None:
            raise ValueError("lane corridor lacks authoritative edges")
        polygon = Polygon(np.concatenate((lane.left_edge.xyz[:, :2], lane.right_edge.xyz[::-1, :2])))
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("invalid lane-edge polygon")
        polygons.append(polygon)
        following_points = lane.center.xyz.tolist()
        if points and np.linalg.norm(np.asarray(points[-1])[:2] - lane.center.xyz[0, :2]) <= 0.1:
            following_points = following_points[1:]
        points.extend(following_points)
    return points, identifiers, unary_union(polygons)


def lane_pair_semantic_prefilter(times, keep_xyz, change_xyz, source_region, target_region,
                                 rollout_end_s=7.0, dwell_s=1.0, margin_m=0.15,
                                 active_start_s=1.7, *, check_linear_segments=False):
    """Conservative point-occupancy pilot check, NOT a frozen semantic verifier.

    Require R0 to stay in the source corridor after instruction and both arms
    to occupy mutually exclusive corridors throughout the final dwell. Shared
    successor/merge regions cannot prove two different lane commitments.
    """
    times = np.asarray(times, dtype=float)
    keep_xyz, change_xyz = np.asarray(keep_xyz, dtype=float), np.asarray(change_xyz, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
        or np.any(np.diff(times) > 0.200001)
        or keep_xyz.shape != (len(times), 3) or change_xyz.shape != keep_xyz.shape
        or not all(np.isfinite(v).all() for v in (times, keep_xyz, change_xyz))
        or not np.isfinite([rollout_end_s, dwell_s, margin_m, active_start_s]).all()
        or not 0 < dwell_s <= rollout_end_s or margin_m < 0
        or not 0 <= active_start_s <= rollout_end_s - dwell_s
        or times[0] > 0 or times[-1] < rollout_end_s
        or not source_region.is_valid or not target_region.is_valid
        or source_region.is_empty or target_region.is_empty):
        raise ValueError("invalid lane semantic prefilter inputs")
    # Include exact interval boundaries even if a controller log samples just
    # before/after them; a missing trace horizon is rejected above.
    grid = np.unique(np.concatenate(([0.0, active_start_s, rollout_end_s - dwell_s, rollout_end_s],
                                     times[(times >= 0) & (times <= rollout_end_s)])))
    keep = np.column_stack([np.interp(grid, times, keep_xyz[:, i]) for i in range(3)])
    change = np.column_stack([np.interp(grid, times, change_xyz[:, i]) for i in range(3)])
    def membership(positions, own, other):
        rows = []
        for position in positions:
            point = Point(position[:2])
            inside = own.contains(point) and point.distance(own.boundary) >= margin_m
            exclusive = inside and not other.covers(point) and point.distance(other) >= margin_m
            rows.append((inside, exclusive))
        return np.asarray(rows, dtype=bool)
    r0 = membership(keep, source_region, target_region)
    r1 = membership(change, target_region, source_region)
    final = grid >= rollout_end_s - dwell_s
    active = grid >= active_start_s
    keep_ok = bool(r0[active, 0].all())
    distinct = bool(r0[final, 1].all() and r1[final, 1].all())
    linear = None
    if check_linear_segments:
        # Checking only 10 Hz vertices can miss narrow original-map gaps.
        # This checks the entire interpolated rig-origin polyline, not the
        # continuous physical vehicle footprint or reconstruction support.
        def path_check(positions, own, other=None):
            xy = positions[:, :2]
            path = Point(xy[0]) if np.all(xy == xy[0]) else LineString(xy)
            inside = own.contains(path) and path.distance(own.boundary) >= margin_m
            return bool(inside and (other is None or
                        (not path.intersects(other) and path.distance(other) >= margin_m)))
        linear = {"scope": "piecewise_linear_rig_path_not_vehicle_swept_volume",
                  "keep_active_inside": path_check(keep[active], source_region),
                  "keep_final_exclusive": path_check(keep[final], source_region, target_region),
                  "change_final_exclusive": path_check(change[final], target_region, source_region)}
        keep_ok = keep_ok and linear["keep_active_inside"]
        distinct = distinct and linear["keep_final_exclusive"] and linear["change_final_exclusive"]
    return {"scope": "sampled_lane_occupancy_not_qualification", "passed": keep_ok and distinct,
            "position_reference": "rig_origin_in_nre_local",
            "keep_in_source_throughout": keep_ok, "exclusive_final_dwell_both_arms": distinct,
            "dwell_interval_s": [rollout_end_s - dwell_s, rollout_end_s],
            "keep_active_interval_s": [active_start_s, rollout_end_s],
            "boundary_margin_m": margin_m,
            "linear_segment_check": linear,
            "keep_outside_source_times_s": grid[active & ~r0[:, 0]].tolist(),
            "keep_nonexclusive_final_times_s": grid[final & ~r0[:, 1]].tolist(),
            "change_nonexclusive_final_times_s": grid[final & ~r1[:, 1]].tolist()}


def smoothstep5(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10 + value * (-15 + 6 * value))


def retimed_lane_path(times, recorded_xyz, recorded_yaws, target_xyz,
                      intervention_s, slowdown_s, speed_fraction,
                      lane_change_start_s, lane_change_duration_s, *,
                      return_path_xyz=None, return_start_s=None, return_duration_s=None,
                      target_progress_mode="closest_projection"):
    """Slow along the recording then blend to an explicitly supplied corridor.

    There is no lane extrapolation: active lane-changing samples must project
    inside the supplied corridor. Both path prefix and original time grid are
    retained. The caller must check real traffic, dynamics, map legality and
    NRE support independently before any live qualification.
    """
    times = np.asarray(times, dtype=float)
    recorded_xyz = np.asarray(recorded_xyz, dtype=float)
    recorded_yaws = np.asarray(recorded_yaws, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)
    if target_progress_mode not in ("closest_projection", "recorded_arc_length"):
        raise ValueError("unknown target progress hypothesis")
    if (len(times) < 3 or recorded_xyz.shape != (len(times), 3)
        or recorded_yaws.shape != times.shape or target_xyz.ndim != 2
        or target_xyz.shape[1] != 3 or len(target_xyz) < 2
        or not all(np.isfinite(v).all() for v in (times, recorded_xyz, recorded_yaws, target_xyz))
        or not np.isfinite([intervention_s, slowdown_s, speed_fraction,
                            lane_change_start_s, lane_change_duration_s]).all()
        or times[0] != 0 or np.any(np.diff(times) <= 0)
        or not 0 <= speed_fraction <= 1 or slowdown_s <= 0
        or intervention_s < 0 or lane_change_start_s < intervention_s
        or lane_change_duration_s <= 0):
        raise ValueError("invalid lane path search inputs")
    speed_scale = 1 - (1 - speed_fraction) * smoothstep5((times - intervention_s) / slowdown_s)
    retimed = np.concatenate(([0.0], np.cumsum(np.diff(times) * (speed_scale[:-1] + speed_scale[1:]) / 2)))
    retimed[times <= intervention_s] = times[times <= intervention_s]
    positions = np.column_stack([np.interp(retimed, times, recorded_xyz[:, i]) for i in range(3)])
    blend = smoothstep5((times - lane_change_start_s) / lane_change_duration_s)
    returning = any(v is not None for v in (return_path_xyz, return_start_s, return_duration_s))
    if returning:
        if any(v is None for v in (return_path_xyz, return_start_s, return_duration_s)):
            raise ValueError("complete post-scope return reference and timing required")
        return_path = np.asarray(return_path_xyz, dtype=float)
        if (return_path.shape != positions.shape or not np.isfinite(return_path).all()
            or not np.isfinite([return_start_s, return_duration_s]).all()
            or return_start_s < 7.1 or return_start_s < lane_change_start_s + lane_change_duration_s
            or return_duration_s <= 0 or return_start_s + return_duration_s > times[-1]
            or not np.array_equal(return_path[times <= intervention_s], recorded_xyz[times <= intervention_s])):
            raise ValueError("return must preserve the prefix and start after the complete 7 s task, within the real plan horizon")
        return_weight = smoothstep5((times - return_start_s) / return_duration_s)
        return_delta = return_path - positions
        # A finished return no longer follows the ending target corridor. Any
        # positive residual target weight still needs an original projection.
        blend = blend * (1. - return_weight)
    line = LineString(target_xyz[:, :2])
    if not line.is_simple or line.length <= 0:
        raise ValueError("target corridor is degenerate or self-intersecting")
    target_arcs = None
    if target_progress_mode == "recorded_arc_length":
        # A distinct plan hypothesis: advance on the unchanged target centreline
        # by actual retimed recording distance, not changing nearest projections.
        # This does not smooth/repair the map or extrapolate its endpoints.
        if lane_change_start_s >= times[-1]:
            raise ValueError("arc-length anchoring must precede the plan endpoint")
        traveled = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))]
        anchor = Point([np.interp(lane_change_start_s, times, positions[:, k]) for k in (0, 1)])
        target_arcs = line.project(anchor) + traveled - np.interp(lane_change_start_s, times, traveled)
    for i, weight in enumerate(blend):
        if weight <= 0:
            continue
        distance = line.project(Point(positions[i, :2])) if target_arcs is None else float(target_arcs[i])
        if distance <= 1e-6 or distance >= line.length - 1e-6:
            endpoint = "start" if distance <= 1e-6 else "end"
            raise ValueError("target corridor would require endpoint extrapolation "
                             f"at t={times[i]:.6f}s ({endpoint}; projected={distance:.6f}m; length={line.length:.6f}m)")
        target = np.asarray(line.interpolate(distance).coords[0])
        positions[i, :2] += weight * (target - positions[i, :2])
    if returning:
        positions += return_weight[:, None] * return_delta
    velocity = np.gradient(positions[:, :2], times, axis=0)
    yaws = np.arctan2(velocity[:, 1], velocity[:, 0])
    prefix = times <= intervention_s
    positions[prefix] = recorded_xyz[prefix]
    yaws[prefix] = recorded_yaws[prefix]
    return positions, yaws


def waypoint_plan(times, positions, yaws, controller):
    return {"controller": controller, "coordinate_frame": "nre_local",
            "waypoints": [{"time_s": float(t), "x": float(p[0]), "y": float(p[1]),
                           "z": float(p[2]), "yaw_rad": float(y)}
                          for t, p, y in strict_zip(times, positions, yaws)]}
