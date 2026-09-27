"""Original speed-limit-sign approach geometry, not visibility or legality."""
import io
from pathlib import Path
import zipfile

import numpy as np
from shapely.geometry import LineString, Point

from .crosswalk_speed_scope import original_reference_lanes
from .speed_spatial_draft import central_lane_segment


def sign_approach_segment(centerlines_xy, lane_polygon_xy, sign_center_xy,
                          far_distance_m=10., near_distance_m=5.):
    """Measure approach arclength from one unambiguous original sign projection.

    The scope remains inside the first original lane. Later original sections
    locate the physical sign; they neither enlarge that lane nor authorize a
    branch. A sign's orientation, visibility and applicable limit remain open.
    """
    if (not np.isfinite([far_distance_m, near_distance_m]).all()
        or not 0 < near_distance_m < far_distance_m or not centerlines_xy):
        raise ValueError("positive ordered distances and original centerlines required")
    lines = [np.asarray(p, dtype=float) for p in centerlines_xy]
    sign = np.asarray(sign_center_xy, dtype=float)
    if (sign.shape != (2,) or not np.isfinite(sign).all()
        or any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all() for p in lines)
        or any(np.any(np.linalg.norm(np.diff(p,axis=0),axis=1) <= 0) for p in lines)):
        raise ValueError("finite nondegenerate original XY geometry required")
    gaps = [float(np.linalg.norm(a[-1]-b[0])) for a,b in zip(lines,lines[1:])]
    if any(g > .5 for g in gaps):
        raise ValueError("original reference has a material gap")
    chunks = [lines[0]] + [b[1:] if np.array_equal(a[-1],b[0]) else b for a,b in zip(lines,lines[1:])]
    points = np.concatenate(chunks)
    reference = LineString(points)
    if not reference.is_simple:
        raise ValueError("simple original directed reference required")
    point = Point(sign)
    distances, projections, offset = [], [], 0.
    for start,end in zip(points,points[1:]):
        edge = LineString([start,end])
        distances.append(edge.distance(point))
        projections.append(offset+edge.project(point))
        offset += edge.length
    nearest = min(distances)
    tied = [s for d,s in zip(distances,projections) if abs(d-nearest) <= 1e-6]
    if nearest > 15. or max(tied)-min(tied) > 1e-4:
        raise ValueError("one nearby unambiguous original sign projection required")
    anchor = reference.project(point)
    if not 1e-6 < anchor < reference.length-1e-6:
        raise ValueError("sign projection cannot use a reference endpoint")
    first_length = LineString(lines[0]).length
    entry,exit_ = anchor-far_distance_m,anchor-near_distance_m
    if not 0 < entry < exit_ < first_length:
        raise ValueError("sign approach must fit inside the first original lane")
    segment = central_lane_segment(lines[0],lane_polygon_xy,entry/first_length,exit_/first_length)
    segment.update(scope_definition="original lane approach before one projected physical speed-limit sign",
        referent_anchor_xy=list(reference.interpolate(anchor).coords[0]),referent_anchor_arclength_m=float(anchor),
        sign_center_xy=sign.tolist(),sign_distance_to_reference_m=float(nearest),
        far_distance_m=float(far_distance_m),near_distance_m=float(near_distance_m),original_join_gaps_m=gaps,
        distance_convention="arclength along explicit original lane centers to unique nearest projection of sign center",
        geometry_modified=False,visual_grounding_verified=False)
    return segment


def read_sign_approach(artifact,lane_ids,sign_id,*,far_distance_m=10.,near_distance_m=5.):
    """Re-read source sign, source rails, map version and coordinate convention."""
    import pyarrow.parquet as pq
    lane_ids = list(map(str,lane_ids))
    path = Path(artifact.source).resolve()
    with zipfile.ZipFile(path) as archive:
        signs = pq.read_table(io.BytesIO(archive.read("clipgt/traffic_sign.parquet")),use_threads=False).to_pylist()
        raw_lanes = pq.read_table(io.BytesIO(archive.read("clipgt/lane.parquet")),use_threads=False).to_pylist()
    selected = [r for r in signs if r["key"]["map_id"] == str(sign_id)]
    if len(selected) != 1 or selected[0]["traffic_sign"]["category"] != "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT":
        raise ValueError("one original regulatory speed-limit sign required")
    original = selected[0]
    clip = artifact.metadata.scene_id.removeprefix("clipgt-")
    vm = artifact.map
    if original["key"]["clip_id"] != clip or vm is None or artifact._map_source != "clipgt":
        raise ValueError("same original clip and actual ClipGT map convention required")
    lanes = original_reference_lanes(vm,lane_ids)
    alignment = []
    for identifier,lane in zip(lane_ids,lanes):
        matches = [r for r in raw_lanes if r["key"]["map_id"] == identifier]
        if len(matches) != 1:
            raise ValueError("one original record per lane required")
        raw = matches[0]
        if (raw["key"]["clip_id"] != clip or raw["key"]["map_id_version"] != original["key"]["map_id_version"]
            or raw["lane"]["egomotion_label_class_id"] != original["traffic_sign"]["egomotion_label_class_id"]):
            raise ValueError("sign/lane clip, map version or coordinate source differs")
        errors = [LineString([[p["x"],p["y"]] for p in raw["lane"][field]]).hausdorff_distance(LineString(edge.xyz[:,:2]))
                  for field,edge in (("left_rail",lane.left_edge),("right_rail",lane.right_edge))]
        if not np.isfinite(errors).all() or max(errors) > 1e-5:
            raise ValueError("original sign/lane XY convention differs from actual map")
        alignment.append(dict(lane_id=identifier,rail_hausdorff_error_m=errors))
    first = lanes[0]
    boundary = np.concatenate((first.left_edge.xyz[:,:2],first.right_edge.xyz[::-1,:2])).tolist()
    centers = [l.center.xyz[:,:2].tolist() for l in lanes]
    center = original["traffic_sign"]["center"]
    segment = sign_approach_segment(centers,boundary,[center["x"],center["y"]],far_distance_m,near_distance_m)
    stat = path.stat()
    return dict(scope="original_speed_limit_sign_approach_not_grounding_or_qualification",qualified=False,human_reviewed=False,
        segment=segment,source_sign=original,source_lane_ids=lane_ids,centerlines_xy=centers,
        original_lane_polygon_xy=boundary,coordinate_alignment=alignment,source_artifact=str(path),
        source_size_bytes=stat.st_size,source_mtime_ns=str(stat.st_mtime_ns),artifact_uuid=artifact.metadata.uuid,
        scene_id_private=artifact.metadata.scene_id,explicit_reference_chain_is_private=True,
        speed_limit_value_or_units_inferred=False,
        remaining=["actual model-resolution sign identity and visibility", "triggered semantic contract binding",
                   "scope language review", "actual paired verification", "legality and continuous support", "formal qualification"])
