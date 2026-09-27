"""Explicit tiny joins for exploratory lane regions, never an NRE support claim."""
import math

import numpy as np
from shapely.geometry import MultiPoint, shape
from shapely.ops import unary_union


def section_caps(section):
    """Recover original cap edges from an unmodified rail polygon and centerline.

    AlpaSim centerline endpoints are the averages of corresponding rail ends.
    Require unique matching cap midpoints; never choose a nearby arbitrary edge.
    """
    polygon = shape(section["polygon"])
    center = np.asarray(section["centerline_xyz"], dtype=float)
    if (polygon.geom_type != "Polygon" or not polygon.is_valid or polygon.interiors
        or center.ndim != 2 or len(center) < 2 or center.shape[1] != 3 or not np.isfinite(center).all()):
        raise ValueError("original rail polygon and finite centerline required")
    ring = np.asarray(polygon.exterior.coords)
    caps = []
    for endpoint in center[[0, -1], :2]:
        indices = np.flatnonzero(np.linalg.norm((ring[:-1] + ring[1:]) / 2 - endpoint, axis=1) < 1e-4)
        if len(indices) != 1:
            raise ValueError("no unique original rail cap at centerline endpoint")
        index = int(indices[0])
        cap = ring[index:index + 2].copy()
        # For a polygon created as left rail followed by reversed right rail,
        # the end cap is left/right; the start cap is right/left.
        caps.append(cap)
    return caps[0][::-1], caps[1]


def linked_section_region(sections, directed_links, max_rail_gap_m=.05):
    """Remove internal end-cap seams only at bounded, explicitly linked sections.

    Original polygons are retained. A join uses only their actual four rail
    endpoints, no buffering of outer boundaries. Rejected joins are not filled.
    The result is a diagnostic planning envelope, not the original map or a
    certificate of physical driveability, legality or reconstruction support.
    """
    if not sections or not math.isfinite(max_rail_gap_m) or not 0 < max_rail_gap_m <= .05:
        raise ValueError("require sections and at most five-centimeter rail joins")
    ids = [s["lane_id"] for s in sections]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sections cannot establish a simple corridor")
    links = {tuple(pair) for pair in directed_links}
    polygons = [shape(s["polygon"]) for s in sections]
    if any(p.geom_type != "Polygon" or not p.is_valid or p.area <= 0 for p in polygons):
        raise ValueError("invalid original lane polygon")
    pieces, joins = list(polygons), []
    for previous, following in zip(sections, sections[1:]):
        row = {"from_lane": previous["lane_id"], "to_lane": following["lane_id"], "bridge_added": False}
        if (previous["lane_id"], following["lane_id"]) not in links:
            row["reason"] = "no_authoritative_directed_link"
            joins.append(row)
            continue
        try:
            _, end = section_caps(previous)
            start, _ = section_caps(following)
            gaps = np.linalg.norm(end - start, axis=1)
            row["rail_endpoint_gaps_m"] = gaps.tolist()
            if max(gaps) > max_rail_gap_m:
                row["reason"] = "rail_gap_exceeds_bound"
            else:
                # Slightly rotated caps may cross, making a quadrilateral ring
                # self-intersecting. Their bounded convex hull uses only the
                # same four endpoints and covers both tiny seam triangles.
                bridge = MultiPoint([*end, *start]).convex_hull
                if bridge.area == 0:
                    row["reason"] = "coincident_caps_no_area_to_add"
                elif (bridge.geom_type != "Polygon" or not bridge.is_valid
                      or bridge.area > max_rail_gap_m * (max(np.linalg.norm(end[0] - end[1]),
                                                              np.linalg.norm(start[0] - start[1])) + max_rail_gap_m)):
                    row["reason"] = "cap_hull_exceeds_narrow_join_bound"
                else:
                    pieces.append(bridge)
                    row.update(bridge_added=True, bridge_area_m2=bridge.area,
                               bridge_polygon=bridge.__geo_interface__)
        except ValueError as error:
            row["reason"] = str(error)
        joins.append(row)
    return unary_union(pieces), {
        "scope": "bounded_linked_rail_join_planning_envelope_not_original_map_or_support",
        "join_policy": "bounded_original_cap_hull_v2",
        "maximum_rail_gap_m": max_rail_gap_m, "outer_boundary_buffer_m": 0.,
        "continuous_support_verified": False, "qualified": False, "joins": joins}
