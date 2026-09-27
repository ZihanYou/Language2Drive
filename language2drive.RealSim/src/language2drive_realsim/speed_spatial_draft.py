"""Map-defined shared B1 segment drafts; no time-window relabeling or road widening."""
import json
import numpy as np
from shapely.geometry import LineString, Point, Polygon, mapping


def original_candidate_scope_geometry(candidate, original_lane_id=None):
    """Select unchanged geometry of either bound lane, independently of B1 arm speed.

    A topology proposal's R0/R1 labels describe lane alternatives, not the two
    speed arms. An explicit choice must still be one of that proposal's lanes.
    This does not authorize a road-level union or an arbitrary map section.
    """
    arms = candidate['selected_arm_pair']['arms']
    ids = [a['lane_id'] for a in arms]
    if len(ids) != 2 or any(not isinstance(k, str) or not k for k in ids) or len(set(ids)) != 2:
        raise ValueError('two distinct bound original candidate lanes required')
    lane_id = ids[0] if original_lane_id is None else original_lane_id
    if lane_id not in ids:
        raise ValueError('scope must be one of the bound original candidate lanes')
    geometry = candidate['lane_geometry_private_not_qualified'].get(lane_id)
    if geometry is None:
        raise ValueError('selected original lane geometry is missing')
    return lane_id, geometry


def downstream_road_geometry(vector_map, start_id, target_id, adjacent_id, pair_kind):
    """Read unchanged edges through a unique, reciprocal forward map chain.

    This selects a private B1 scope, not a target route for the model. Refuse
    ambiguous forward choices, disconnected sections or invented adjacency.
    """
    if pair_kind not in {"keep_vs_left", "keep_vs_right"} or target_id == adjacent_id:
        raise ValueError("distinct known adjacent road lanes required")
    chain = [str(start_id)]
    while chain[-1] != str(target_id):
        lane = vector_map.get_road_lane(chain[-1])
        successors = [str(k) for k in lane.next_lanes if str(k) != "-1"]
        if len(successors) != 1 or successors[0] in chain or len(chain) >= 20:
            raise ValueError("scope must follow a unique bounded directed map chain")
        following = vector_map.get_road_lane(successors[0])
        if (chain[-1] not in {str(k) for k in following.prev_lanes}
            or np.linalg.norm(lane.center.xyz[-1, :2] - following.center.xyz[0, :2]) > .5):
            raise ValueError("downstream map chain is disconnected or nonreciprocal")
        chain.append(successors[0])
    source, adjacent = vector_map.get_road_lane(str(target_id)), vector_map.get_road_lane(str(adjacent_id))
    candidates = source.adj_lanes_left if pair_kind == "keep_vs_left" else source.adj_lanes_right
    reciprocal = adjacent.adj_lanes_right if pair_kind == "keep_vs_left" else adjacent.adj_lanes_left
    if str(adjacent_id) not in {str(k) for k in candidates} or str(target_id) not in {str(k) for k in reciprocal}:
        raise ValueError("actual reciprocal same-direction adjacency required")
    def read(lane):
        if lane.left_edge is None or lane.right_edge is None or lane.center is None:
            raise ValueError("original map edges and centre required")
        left, right, center = (np.asarray(v.xyz, dtype=float) for v in (lane.left_edge, lane.right_edge, lane.center))
        if any(v.ndim != 2 or v.shape[1] != 3 or len(v) < 2 or not np.isfinite(v).all() for v in (left, right, center)):
            raise ValueError("finite original XYZ map vertices required")
        polygon = np.concatenate((left[:, :2], right[::-1, :2])).tolist()
        if not Polygon(polygon).is_valid or Polygon(polygon).area <= 0:
            raise ValueError("valid unchanged original lane region required")
        return {"centerline_xyz": center.tolist(), "left_edge_xyz": left.tolist(), "right_edge_xyz": right.tolist(),
                "lane_region_polygon_xy_not_qualified": polygon}
    source_geometry, adjacent_geometry = read(source), read(adjacent)
    directions = [np.asarray(g["centerline_xyz"][-1])[:2] - np.asarray(g["centerline_xyz"][0])[:2]
                  for g in (source_geometry, adjacent_geometry)]
    if np.dot(*directions) <= 0:
        raise ValueError("adjacent scope lane must have the same travel direction")
    return {"directed_path_from_candidate": chain, "source_lane_id": str(target_id), "adjacent_lane_id": str(adjacent_id),
            "pair_kind": pair_kind, "source_lane_geometry": source_geometry, "adjacent_lane_geometry": adjacent_geometry,
            "geometry_modified": False}


def adjacent_road_scope_boundary(source, adjacent, pair_kind):
    """A B1 road-level scope from original outside edges of known adjacent lanes.

    Unlike a lane occupancy region this includes the intervening separator.
    It does not assert that every interior point is drivable/render-supported;
    actual offroad/native metrics retain those independent checks.
    """
    if pair_kind not in {"keep_vs_left", "keep_vs_right"}:
        raise ValueError("known adjacent lane pair required")
    left = source if pair_kind == "keep_vs_right" else adjacent
    right = adjacent if pair_kind == "keep_vs_right" else source
    boundary = [p[:2] for p in left["left_edge_xyz"]] + [p[:2] for p in reversed(right["right_edge_xyz"])]
    polygon = Polygon(boundary)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("original outer edges do not form a valid road-level scope")
    return boundary


def central_lane_segment(centerline_xy, lane_polygon_xy, entry_fraction=.5, exit_fraction=.85):
    """Clip an original lane between two fixed centreline-fraction cross-sections.

    Fractions are pilot authoring parameters, not adapted to evaluated models.
    The output preserves the original road boundary; it is not render support.
    """
    if not 0 < entry_fraction < exit_fraction < 1:
        raise ValueError("ordered interior fractions required")
    xy = np.asarray(centerline_xy, dtype=float)
    vertices = np.asarray(lane_polygon_xy, dtype=float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all()
        or vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all()):
        raise ValueError("finite original XY geometry required")
    line, original = LineString(xy), Polygon(vertices)
    if line.length <= 0 or not original.is_valid or original.area <= 0:
        raise ValueError("nondegenerate valid lane geometry required")
    extent = max(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 4, 10.)
    cuts = []
    scope = original
    for fraction, sign in ((entry_fraction, 1), (exit_fraction, -1)):
        distance = line.length * fraction
        point = np.asarray(line.interpolate(distance).coords[0])
        before = np.asarray(line.interpolate(max(0., distance - .1)).coords[0])
        after = np.asarray(line.interpolate(min(line.length, distance + .1)).coords[0])
        tangent = after - before
        if np.linalg.norm(tangent) <= 1e-9 or not original.covers(Point(point)):
            raise ValueError("cross-section centre must lie in the original lane")
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        left, right = point + normal * extent, point - normal * extent
        gate = original.intersection(LineString([left, right]))
        if gate.geom_type != "LineString" or gate.length <= 0:
            raise ValueError("one unambiguous original-lane cross-section required")
        scope = scope.intersection(Polygon([left, right, right + sign*tangent*extent, left + sign*tangent*extent]))
        cuts.append([list(gate.coords[0]), list(gate.coords[-1])])
    # GEOS intersections can leave roundoff-sized slivers (observed 1e-15 m²).
    # No buffering/repair is performed. Reject material outside area instead
    # of requiring a bit-exact topological predicate after floating arithmetic.
    outside_area = scope.difference(original).area
    if scope.geom_type != "Polygon" or not scope.is_valid or scope.area <= 0 or outside_area > 1e-12:
        raise ValueError("shared segment must be a connected subset of original lane: "
                         + str((scope.geom_type, scope.is_valid, scope.area, scope.difference(original).area)))
    return {"scope_polygon": json.loads(json.dumps(mapping(scope))),
            "entry_gate": cuts[0], "exit_gate": cuts[1],
            "entry_fraction": entry_fraction, "exit_fraction": exit_fraction,
            "original_centerline_length_m": line.length, "scope_area_m2": scope.area,
            "floating_difference_area_m2": outside_area, "intersection_roundoff_area_limit_m2": 1e-12,
            "map_basis": "fixed original centreline cross-sections, not trajectory timestamps",
            "qualified": False, "geometry_frozen": False, "speed_limit_basis": None}
