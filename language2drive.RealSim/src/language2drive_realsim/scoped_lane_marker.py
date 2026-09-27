"""Shared B2 physical-marker hypotheses, never visual or route approval."""
from copy import deepcopy
import numpy as np
from shapely.geometry import LineString, Point, Polygon


def common_sign_end(corridors_xy, prefix_end_xy, source_sign, source_lanes):
    """Derive ONE transverse end through an unchanged original sign center.

    This is a private spatial hypothesis, not an existing painted road line.
    Callers must retrieve the sign and selected lanes from the actual archive.
    Never infer sign visibility, road legality or a full B2 scope from this.
    """
    if set(corridors_xy) != {"R0", "R1"} or set(source_lanes) != {"R0", "R1"}:
        raise ValueError("two original lane corridors and source records required")
    try:
        key, sign = source_sign["key"], source_sign["traffic_sign"]
        if not all(isinstance(key[k], str) and key[k] for k in
                   ("clip_id", "map_id", "map_id_version", "label_class_id")):
            raise ValueError("original sign identity required")
        if not isinstance(sign["category"], str) or not sign["category"]:
            raise ValueError("original sign category required")
        datum = sign["egomotion_label_class_id"]
        if not isinstance(datum, str) or not datum:
            raise ValueError("original sign coordinate source required")
        if len({r["key"]["map_id"] for r in source_lanes.values()}) != 2:
            raise ValueError("two distinct original lanes required")
        if any(key["clip_id"] != r["key"]["clip_id"]
               or key["map_id_version"] != r["key"]["map_id_version"]
               or datum != r["lane"]["egomotion_label_class_id"] for r in source_lanes.values()):
            raise ValueError("sign and lane raw coordinate sources differ")
        center = np.array([sign["center"][k] for k in ("x", "y", "z")], dtype=float)
    except (KeyError, TypeError) as error:
        raise ValueError("complete original sign and lane records required") from error
    prefix = np.asarray(prefix_end_xy, dtype=float)
    if not np.isfinite(center).all() or prefix.shape != (2,) or not np.isfinite(prefix).all():
        raise ValueError("finite original sign center and intervention position required")
    anchor = center[:2]
    lines, directions = {}, []
    for arm in ("R0", "R1"):
        xy = np.asarray(corridors_xy[arm], dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError("finite original lane center required")
        line = LineString(xy)
        if not line.is_simple or line.length <= 1.:
            raise ValueError("simple nonempty original lane center required")
        arc = line.project(Point(anchor))
        if not .5 < arc < line.length - .5 or line.distance(Point(anchor)) > 12.:
            raise ValueError("nearby interior sign projection required; no endpoint extrapolation")
        tangent = np.array(line.interpolate(arc + .5).coords[0]) - np.array(line.interpolate(arc - .5).coords[0])
        if np.linalg.norm(tangent) < 1e-8:
            raise ValueError("unambiguous local forward direction required")
        directions.append(tangent / np.linalg.norm(tangent))
        lines[arm] = line
    if float(directions[0] @ directions[1]) < .95:
        raise ValueError("both local original directions must agree")
    forward = directions[0] + directions[1]
    forward /= np.linalg.norm(forward)
    transverse = np.array([-forward[1], forward[0]])
    segment_xy = np.array([anchor - 16. * transverse, anchor + 16. * transverse])
    segment = LineString(segment_xy)
    results, points = {}, []
    for arm, line in lines.items():
        crossing = line.intersection(segment)
        if crossing.geom_type != "Point" or crossing.is_empty:
            raise ValueError("one unambiguous shared end crossing per original lane required")
        arc, initial = line.project(crossing), line.project(Point(prefix))
        if not .5 < arc < line.length - .5 or arc <= initial:
            raise ValueError("shared sign end must be ahead and inside both original corridors")
        points.append(np.array(crossing.coords[0]))
        results[arm] = {"instruction_projected_arc_m": initial, "sign_end_arc_m": arc,
                        "distance_ahead_m": arc - initial, "entry_xy": list(crossing.coords[0])}
    separation = float(np.linalg.norm(points[1] - points[0]))
    if not 1. <= separation <= 8.:
        raise ValueError("distinct nearby lane crossings required")
    return {"scope": "original_sign_transverse_end_hypothesis_not_visibility_or_scoped_lane_feasibility",
        "qualified": False, "geometry_modified": False, "derived_end_not_painted_road_line": True,
        "source_traffic_sign": deepcopy(source_sign), "end_anchor_xyz": center.tolist(),
        "end_segment_xy": segment_xy.tolist(), "end_forward_xy": forward.tolist(), "arms": results,
        "same_major_route_verified": False, "visual_grounding_verified": False, "shared_scope_gates_bound": False}


def common_crosswalk_distances(corridors_xy, prefix_end_xy, crosswalk_xy):
    """Require the same unchanged polygon to intersect both forward lane centers."""
    if set(corridors_xy) != {"R0", "R1"}:
        raise ValueError("both original lane corridors required")
    marker = np.asarray(crosswalk_xy, dtype=float)
    prefix = np.asarray(prefix_end_xy, dtype=float)
    if (marker.ndim != 2 or marker.shape[1] != 2 or len(marker) < 3
        or prefix.shape != (2,) or not np.isfinite(marker).all() or not np.isfinite(prefix).all()):
        raise ValueError("finite original XY marker and instruction position required")
    polygon = Polygon(marker)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("cannot repair original crosswalk")
    from shapely.geometry import Point
    results = {}
    for arm, points in corridors_xy.items():
        xy = np.asarray(points, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError("finite original lane center required")
        line = LineString(xy)
        if line.length <= 0 or not line.is_simple:
            raise ValueError("simple original lane center required")
        intersection = line.intersection(polygon)
        if intersection.geom_type != "LineString" or intersection.length <= 0:
            raise ValueError("one shared crosswalk must traverse each lane center")
        arcs = sorted(line.project(p) for p in intersection.boundary.geoms)
        if len(arcs) != 2:
            raise ValueError("unambiguous original crosswalk entry/exit required")
        initial = line.project(Point(prefix))
        if arcs[0] <= initial:
            raise ValueError("common marker must be ahead of the intervention position")
        results[arm] = {"instruction_projected_arc_m": initial,
            "crosswalk_entry_arc_m": arcs[0], "crosswalk_exit_arc_m": arcs[1],
            "distance_ahead_m": arcs[0] - initial,
            "entry_xy": list(line.interpolate(arcs[0]).coords[0])}
    return {"scope": "original_common_crosswalk_geometry_not_visibility_or_scoped_lane_feasibility",
        "qualified": False, "geometry_modified": False, "arms": results,
        "same_major_route_verified": False, "visual_grounding_verified": False,
        "shared_scope_gates_bound": False}
