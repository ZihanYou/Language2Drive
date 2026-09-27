"""Map-bound spatial diagnostics; never infer render support from polygons."""
import numpy as np
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union

from .compat import strict_zip


def lane_spatial_coverage(traces, lane_polygons, lane_ids_by_arm, start_s=6., end_s=7.):
    """Find original lane sections covering the existing final-dwell interval.

    Inputs are actual ego AABB-centre poses, matching the formal ASL bridge,
    not oracle rig waypoints. Segment containment additionally catches a map
    gap between contained samples; it is not continuous physics/render proof.
    No convex hull, polygon buffer, or inferred road extension is permitted.
    """
    if set(traces) != {"R0", "R1"} or set(lane_ids_by_arm) != {"R0", "R1"}:
        raise ValueError("both actual arms and their lane identities required")
    if not np.isfinite([start_s, end_s]).all() or not 0 <= start_s < end_s:
        raise ValueError("invalid existing dwell interval")
    polygons = {}
    for key, points in lane_polygons.items():
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("original lane vertices must be finite XY")
        polygon = Polygon(values)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("invalid original lane polygon")
        polygons[key] = polygon
    results = {}
    for arm, samples in traces.items():
        rows = np.asarray(samples, dtype=float)
        if (rows.ndim != 2 or rows.shape[1] != 3 or len(rows) < 2
            or not np.isfinite(rows).all() or rows[0, 0] != 0 or rows[-1, 0] < end_s
            or np.any(np.diff(rows[:, 0]) <= 0) or np.any(np.diff(rows[:, 0]) > .200001)):
            raise ValueError("complete dense actual AABB-centre trace required")
        grid = np.unique(np.r_[start_s, rows[(rows[:, 0] > start_s) & (rows[:, 0] < end_s), 0], end_s])
        positions = np.column_stack([np.interp(grid, rows[:, 0], rows[:, a]) for a in (1, 2)])
        line = LineString(positions)
        other_arm = "R1" if arm == "R0" else "R0"
        opposing = unary_union([polygons[key] for key in lane_ids_by_arm[other_arm]])
        candidates = []
        for key in lane_ids_by_arm[arm]:
            polygon = polygons[key]
            outside = [float(t) for t, p in strict_zip(grid, positions) if not polygon.covers(Point(p))]
            candidates.append({"lane_id": key, "sample_outside_times_s": outside,
                               "interpolated_line_covered": bool(polygon.covers(line)),
                               "minimum_boundary_distance_m": float(line.distance(polygon.boundary)),
                               "opposing_region_intersected": bool(opposing.intersects(line)),
                               "covers_exclusive_dwell": bool(polygon.covers(line) and not opposing.intersects(line))})
        results[arm] = {"candidates": candidates,
                        "covering_original_lane_ids": [r["lane_id"] for r in candidates if r["covers_exclusive_dwell"]]}
    unique = all(len(r["covering_original_lane_ids"]) == 1 for r in results.values())
    geometry = None
    if unique:
        geometry = mapping(unary_union([polygons[r["covering_original_lane_ids"][0]] for r in results.values()]))
    return {"scope": "actual_AABB_original_lane_spatial_draft_not_qualification", "qualified": False,
            "position_reference": "ego_aabb_center_in_nre_local", "dwell_interval_s": [start_s, end_s],
            "arms": results, "unique_original_sections_both_arms": unique,
            "shared_scope_proposal_geojson": geometry,
            "scope_geometry_frozen": False, "reconstruction_support_polygon": None,
            "continuous_support_verified": False}
