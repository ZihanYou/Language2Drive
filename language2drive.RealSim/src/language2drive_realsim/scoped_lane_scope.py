"""Exact original sections of one B2 lane; no seam repair or qualification."""
from copy import deepcopy
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union


def original_cap_forward_envelope(caps, r0_forward_xy, *, reference_policy="r0_original_cap"):
    """A new common entry after BOTH original start caps, never a map buffer.

    This optional construction uses map geometry only, not observed/planned
    trajectories. Existing R0-cap scopes must remain unchanged. Continuous
    original-lane coverage is still independently required after construction.
    """
    if reference_policy not in {'r0_original_cap', 'joint_original_caps'}:
        raise ValueError('explicit original cap reference policy required')
    if set(caps) != {'R0', 'R1'}:
        raise ValueError('both original caps required')
    arrays = {a: np.asarray(caps[a], dtype=float) for a in ('R0', 'R1')}
    hint = np.asarray(r0_forward_xy, dtype=float)
    if (any(v.shape != (2, 2) or not np.isfinite(v).all() for v in arrays.values())
        or hint.shape != (2,) or not np.isfinite(hint).all() or np.linalg.norm(hint) <= 0):
        raise ValueError('finite original caps and forward direction required')
    anchor = arrays['R0'].mean(axis=0)
    transverse = arrays['R0'][0] - arrays['R0'][1]
    if np.linalg.norm(transverse) <= 0 or np.linalg.norm(arrays['R1'][0]-arrays['R1'][1]) <= 0:
        raise ValueError('nondegenerate original caps required')
    if reference_policy == 'joint_original_caps':
        # The two mapped caps may have different small tilts. Fit their common
        # transverse direction symmetrically, then keep the SAME 5cm bound.
        # Only original map points enter this fit, never either trajectory.
        points = np.concatenate([arrays[a] for a in ('R0', 'R1')])
        anchor = points.mean(axis=0)
        _, axes = np.linalg.eigh((points-anchor).T @ (points-anchor))
        joint_transverse = axes[:, -1]
        if joint_transverse @ transverse < 0:
            joint_transverse = -joint_transverse
        transverse = joint_transverse
    transverse = transverse / np.linalg.norm(transverse)
    forward = np.array([transverse[1], -transverse[0]])
    hint = hint / np.linalg.norm(hint)
    if abs(float(forward @ hint)) < .95:
        raise ValueError('original cap must be transverse to original forward direction')
    if forward @ hint < 0:
        forward = -forward
    projections = {a: ((v-anchor) @ forward).tolist() for a,v in arrays.items()}
    if max(abs(x) for values in projections.values() for x in values) >= .05:
        raise ValueError('original caps exceed existing 5cm common-entry bound')
    shift = max(0., max(x for values in projections.values() for x in values))
    envelope_anchor = anchor + shift * forward
    entry = np.array([envelope_anchor-16*transverse, envelope_anchor+16*transverse])
    policy = ('original_cap_forward_envelope_v1' if reference_policy == 'r0_original_cap'
              else 'joint_original_caps_forward_envelope_v1')
    return entry, dict(policy=policy,
        original_caps_xy={a:v.tolist() for a,v in arrays.items()},
        source_r0_forward_xy=np.asarray(r0_forward_xy, dtype=float).tolist(),
        original_cap_signed_projections_m=projections,forward_xy=forward.tolist(),forward_shift_m=float(shift),
        original_geometry_modified=False,trajectory_used=False,continuous_coverage_still_required=True)


def continuous_scope_coverage(polygon, line):
    """Allow only floating-point roundoff at the two computed gate endpoints.

    Never buffer polygons, tolerate interior seams, or accept a physical lane
    departure. Keep the raw GEOS result and the coordinate-ULP bound visible.
    """
    if (line.geom_type != "LineString" or line.length <= 0 or not line.is_simple
        or not polygon.is_valid or polygon.area <= 0):
        raise ValueError("valid original polygon and simple scope line required")
    scale = max(1., *(abs(v) for v in line.bounds), *(abs(v) for v in polygon.bounds))
    tolerance = float(8 * np.spacing(scale))
    outside = line.difference(polygon)
    parts = list(outside.geoms) if hasattr(outside, "geoms") else [outside]
    endpoints = [Point(line.coords[0]), Point(line.coords[-1])]
    endpoint_only = all(part.is_empty or (part.geom_type in {"Point", "LineString"}
        and any(all(Point(p).distance(endpoint) <= tolerance for p in part.coords)
                for endpoint in endpoints)) for part in parts)
    raw = polygon.covers(line)
    return dict(passed=bool(raw or endpoint_only), raw_original_polygon_covers=bool(raw),
                outside_length_m=float(outside.length), endpoint_roundoff_tolerance_m=tolerance,
                only_gate_endpoint_roundoff=bool(not raw and endpoint_only), original_geometry_modified=False)


def scope_centerline_between_gates(rows, entry_time_s, exit_time_s, entry_gate, exit_gate):
    """Clip only the continuous semantic scope, not the safety/traffic horizon.

    A crossing event is timestamped at the end of a sample interval. Include
    the approach sample and intersect its segment with the frozen entry gate;
    similarly stop at the actual exit intersection, not the post-exit sample.
    """
    times = np.asarray([r["time_s"] for r in rows], dtype=float)
    xy = np.asarray([r["position"][:2] for r in rows], dtype=float)
    if (len(times) < 2 or xy.shape != (len(times), 2) or not np.isfinite(times).all()
        or not np.isfinite(xy).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("finite ordered scoped path samples required")
    starts, ends = np.flatnonzero(times == entry_time_s), np.flatnonzero(times == exit_time_s)
    if len(starts) != 1 or len(ends) != 1 or starts[0] == 0 or ends[0] <= starts[0]:
        raise ValueError("distinct ordered gate crossing intervals required")
    i, j = int(starts[0]), int(ends[0])
    gates = [np.asarray(g, dtype=float) for g in (entry_gate, exit_gate)]
    if any(g.shape != (2, 2) or not np.isfinite(g).all() for g in gates):
        raise ValueError("finite frozen gate segments required")
    crossings = [LineString(xy[k-1:k+1]).intersection(LineString(g))
                 for k, g in zip((i, j), gates)]
    if any(p.is_empty or p.geom_type != "Point" for p in crossings):
        raise ValueError("one unambiguous intersection in each crossing interval required")
    line = LineString(xy[i-1:j+1])
    if not line.is_simple:
        raise ValueError("simple continuous scope path required")
    start, end = [line.project(p) for p in crossings]
    if end <= start:
        raise ValueError("geometrically ordered distinct scope boundaries required")
    return substring(line, start, end)


def original_scope_lane_chain(raw, confirmed_ids, section_ids, source_sign, associations):
    if (not isinstance(section_ids, list) or not 1 <= len(section_ids) <= 8
        or any(not isinstance(k, str) or not k for k in section_ids)
        or len(set(section_ids)) != len(section_ids)):
        raise ValueError("one to eight distinct explicit original scope sections required")
    if len(set(confirmed_ids)) != len(confirmed_ids) or section_ids[0] not in confirmed_ids:
        raise ValueError("unique confirmed original corridor required")
    start = confirmed_ids.index(section_ids[0])
    if confirmed_ids[start:start + len(section_ids)] != section_ids:
        raise ValueError("scope sections must be consecutive in their confirmed original corridor")
    key, datum = source_sign["key"], source_sign["traffic_sign"]["egomotion_label_class_id"]
    records, polygons, rails = [], [], []
    for lid in section_ids:
        record = raw[lid]
        if (record["key"]["map_id"] != lid or record["key"]["clip_id"] != key["clip_id"]
            or record["key"]["map_id_version"] != key["map_id_version"]
            or record["lane"]["egomotion_label_class_id"] != datum):
            raise ValueError("original sign and scope section coordinate sources differ")
        left, right = [np.asarray([[p["x"], p["y"]] for p in record["lane"][name]], dtype=float)
                       for name in ("left_rail", "right_rail")]
        if any(a.ndim != 2 or a.shape[1:] != (2,) or len(a) < 2 or not np.isfinite(a).all()
               for a in (left, right)):
            raise ValueError("finite complete original scope rails required")
        polygon = Polygon(np.r_[left, right[::-1]])
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("cannot repair original scope lane")
        records.append(deepcopy(record)); polygons.append(polygon); rails.append((left, right))
    gaps = []
    for i, (previous, following) in enumerate(zip(section_ids, section_ids[1:])):
        for subject, obj, kind in ((previous, following, "NEXT_LANE"), (following, previous, "PREVIOUS_LANE")):
            links = [r for r in associations if r["key"]["map_id"] == subject and r["key"]["kind"] == kind
                     and r["key"]["clip_id"] == key["clip_id"]]
            if (len(links) != 1 or subject not in links[0]["association"]["subjects"]
                or obj not in links[0]["association"]["objects"]):
                raise ValueError("each scope continuation needs reciprocal original topology")
        gap = [float(np.linalg.norm(rails[i][j][-1] - rails[i+1][j][0])) for j in (0, 1)]
        if max(gap) > .5:
            raise ValueError("original scope section endpoint gap exceeds existing corridor limit")
        gaps.append(dict(from_lane=previous, to_lane=following, rail_endpoint_distances_m=gap))
    # Even micrometre gaps remain: callers must check continuous line coverage.
    polygon = unary_union(polygons)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("valid unchanged original scope polygon union required")
    return dict(polygon=polygon, cap_xy=np.array([rails[0][0][0], rails[0][1][0]]),
                source_records=records, section_ids=list(section_ids), join_diagnostics=gaps,
                original_geometry_modified=False)
