"""Optional original fork surface before both unchanged commitment gates.

This is a planning membership diagnostic, not legality, NRE support or a new
semantic contract. It never buffers a road or fills an original-map gap.
"""
import json

import numpy as np
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from .branch_gate_prefilter import original_branch_gates


POLICY = "original_common_precommit_surface_v1"
FIELD = "shared_precommit_original_surface"


def _upstream_half_plane(gate, upstream, bounds):
    gate, upstream = np.asarray(gate, float), np.asarray(upstream, float)
    if (gate.shape != (2, 2) or upstream.shape != (2,)
        or not np.isfinite(gate).all() or not np.isfinite(upstream).all()):
        raise ValueError("finite original gate and incoming reference required")
    direction = gate[1] - gate[0]
    length = np.linalg.norm(direction)
    if length <= 1e-9:
        raise ValueError("nondegenerate original gate required")
    normal = np.array([-direction[1], direction[0]]) / length
    side = float((upstream - gate[0]) @ normal)
    if abs(side) <= 1e-6:
        raise ValueError("incoming reference must be strictly before the gate")
    normal *= np.sign(side)
    # A computational clipping box only: accepted area remains a subset of
    # the exact original polygons intersected with this half-plane.
    x0, y0, x1, y1 = bounds
    pad = max(x1-x0, y1-y0, 1.)
    vertices = [np.array(v) for v in ((x0-pad, y0-pad), (x1+pad, y0-pad),
                                     (x1+pad, y1+pad), (x0-pad, y1+pad))]
    result = []
    for first, second in zip(vertices, vertices[1:]+vertices[:1]):
        a, b = float((first-gate[0]) @ normal), float((second-gate[0]) @ normal)
        if a >= 0:
            result.append(first)
        if (a >= 0) != (b >= 0):
            result.append(first + a/(a-b)*(second-first))
    if len(result) < 3:
        raise ValueError("original surface has no upstream half-plane")
    return Polygon(result), normal, gate[0]


def original_shared_precommit_surface(candidate, geometry):
    pair = candidate["selected_arm_pair"]
    if pair.get("kind") != "recorded_vs_counterfactual_branch":
        raise ValueError("explicit original fork pair required")
    selected = {v["arm"]: v["lane_id"] for v in pair["arms"]}
    if len(pair["arms"]) != 2 or set(selected) != {"R0", "R1"} or len(set(selected.values())) != 2:
        raise ValueError("two distinct selected branch sections required")
    prefixes, records = {}, {}
    for arm, lane_id in selected.items():
        row = geometry["arms"][arm]
        sections = row["lanes"]
        ids = [v["lane_id"] for v in sections]
        links = [(v["from_lane"], v["to_lane"]) for v in row["joins"]]
        if (len(set(ids)) != len(ids) or ids != row["corridor_lane_ids"]
            or ids.count(lane_id) != 1 or links != list(zip(ids, ids[1:]))):
            raise ValueError("unchanged unique directed original corridor required")
        index = ids.index(lane_id)
        if index == 0:
            raise ValueError("shared original predecessor required")
        prefixes[arm] = sections[:index]
        records[arm] = sections[index]
    if prefixes["R0"] != prefixes["R1"]:
        raise ValueError("identical original incoming chain required")
    predecessor = prefixes["R0"][-1]
    sources = [predecessor, records["R0"], records["R1"]]
    polygons = [shape(v["polygon"]) for v in sources]
    if any(p.geom_type != "Polygon" or not p.is_valid or p.is_empty
           or p.interiors or not np.isfinite(p.bounds).all() for p in polygons):
        raise ValueError("valid unmodified original fork polygons required")
    gates, _ = original_branch_gates(candidate, geometry)
    upstream = np.asarray(predecessor["centerline_xyz"][0], float)[:2]
    source_union = unary_union(polygons)
    surface = source_union
    for arm in ("R0", "R1"):
        half, normal, origin = _upstream_half_plane(gates["draft-gate-"+arm], upstream, source_union.bounds)
        end = np.asarray(records[arm]["centerline_xyz"][-1], float)[:2]
        if not np.isfinite(end).all() or float((end-origin) @ normal) >= -1e-6:
            raise ValueError("selected original branch endpoint must be downstream")
        surface = surface.intersection(half)
    if surface.is_empty or not surface.is_valid or surface.area <= 0:
        raise ValueError("nonempty original precommit surface required")
    # Do not accept disconnected islands as a common junction.
    pieces = list(surface.geoms) if surface.geom_type == "MultiPolygon" else [surface]
    if any(p.geom_type != "Polygon" or not p.intersects(polygons[0]) for p in pieces):
        raise ValueError("common surface must meet its original incoming section")
    audit = dict(policy=POLICY, qualified=False, human_reviewed=False,
        scope="original_fork_precommit_membership_not_legality_or_render_support",
        source_lane_ids=[v["lane_id"] for v in sources],
        original_gate_coordinates=gates, original_map_modified=False,
        boundary_buffer_m=0., original_gap_fill_m=0.,
        surface=json.loads(json.dumps(mapping(surface))))
    return surface, audit


def recheck_shared_precommit_surface(search, candidate, geometry, regions):
    """Rebuild the opt-in surface; never trust a stored pass or polygon alone."""
    if FIELD not in search:
        return regions
    surface, expected = original_shared_precommit_surface(candidate, geometry)
    if search[FIELD] != expected:
        raise ValueError("original shared precommit surface or gates changed")
    if set(regions) != {"R0", "R1"}:
        raise ValueError("both original planning regions required")
    return {arm: unary_union([region, surface]) for arm, region in regions.items()}
