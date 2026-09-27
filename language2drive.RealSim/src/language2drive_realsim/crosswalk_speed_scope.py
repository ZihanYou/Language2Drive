"""Original-map crosswalk-relative approach scopes, never visual certificates."""
from pathlib import Path
import io
import zipfile

import numpy as np
from shapely.geometry import LineString, Polygon

from .speed_spatial_draft import central_lane_segment


def original_reference_lanes(vector_map, lane_ids):
    """Validate an explicit original directed chain without choosing a branch."""
    ids = list(map(str, lane_ids))
    if not 1 <= len(ids) <= 20 or len(set(ids)) != len(ids) or any(not k.strip() or k == "-1" for k in ids):
        raise ValueError("bounded distinct explicit lane chain required")
    lanes = [vector_map.get_road_lane(key) for key in ids]
    centers = [np.asarray(lane.center.xyz, dtype=float) for lane in lanes]
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all() for p in centers):
        raise ValueError("finite original reference centers required")
    for index, (previous, current) in enumerate(zip(lanes, lanes[1:])):
        if (str(current.id) not in set(map(str, previous.next_lanes))
            or str(previous.id) not in set(map(str, current.prev_lanes))
            or np.linalg.norm(centers[index][-1, :2] - centers[index+1][0, :2]) > .5):
            raise ValueError("explicit original lane chain must be reciprocal, directed and connected")
    return lanes


def crosswalk_approach_segment(centerlines_xy, lane_polygon_xy, crosswalk_xy,
                              far_distance_m=15., near_distance_m=10.):
    """Locate a fixed approach interval along an explicit original map chain.

    The entire speed scope must fit inside the first original lane section.
    Small recorded endpoint joins contribute to reference arclength only; no
    road polygons are filled and no oracle trajectory is extrapolated.
    """
    if (not np.isfinite([far_distance_m, near_distance_m]).all()
        or not 0 < near_distance_m < far_distance_m or not centerlines_xy):
        raise ValueError("positive ordered approach distances and original centerlines required")
    lines = [np.asarray(points, dtype=float) for points in centerlines_xy]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all() for p in lines):
        raise ValueError("finite original XY centerlines required")
    if any(np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 0) for p in lines):
        raise ValueError("nondegenerate original centerline edges required")
    gaps = [float(np.linalg.norm(a[-1] - b[0])) for a, b in zip(lines, lines[1:])]
    if any(gap > .5 for gap in gaps):
        raise ValueError("original map centerline chain has a material gap")
    points = [lines[0]]
    for prev, current in zip(lines, lines[1:]):
        points.append(current[1:] if np.array_equal(prev[-1], current[0]) else current)
    reference = LineString(np.concatenate(points))
    crosswalk = Polygon(crosswalk_xy)
    if (not reference.is_simple or reference.length <= 0 or not crosswalk.is_valid
        or crosswalk.area <= 0 or not np.isfinite(crosswalk.bounds).all()):
        raise ValueError("simple reference and valid original crosswalk required")
    intersection = reference.intersection(crosswalk)
    if intersection.geom_type != "LineString" or intersection.length <= 0:
        raise ValueError("one unambiguous original crosswalk traversal required")
    anchor = min(reference.project(intersection.boundary.geoms[i]) for i in (0, 1))
    first_length = LineString(lines[0]).length
    entry, exit_ = anchor - far_distance_m, anchor - near_distance_m
    if not 0 < entry < exit_ < first_length:
        raise ValueError("crosswalk-relative speed scope must fit inside the first original lane")
    segment = central_lane_segment(lines[0], lane_polygon_xy, entry / first_length, exit_ / first_length)
    segment.update(scope_definition="original lane approach at fixed map-centerline distances before one crosswalk",
                   referent_anchor_xy=list(reference.interpolate(anchor).coords[0]),
                   referent_anchor_arclength_m=anchor, far_distance_m=float(far_distance_m),
                   near_distance_m=float(near_distance_m), original_join_gaps_m=gaps,
                   distance_convention="arclength along explicit original lane centers to first crosswalk entry",
                   geometry_modified=False, visual_grounding_verified=False)
    return segment


def read_crosswalk_approach(artifact, lane_ids, crosswalk_id, *, far_distance_m=15., near_distance_m=10.):
    """Bind a raw ClipGT referent and map convention before proposing a scope."""
    import pyarrow.parquet as pq

    lane_ids = list(map(str, lane_ids))
    if not 1 <= len(lane_ids) <= 20 or len(set(lane_ids)) != len(lane_ids):
        raise ValueError("bounded distinct explicit lane chain required")
    path = Path(artifact.source).resolve()
    with zipfile.ZipFile(path) as archive:
        crosswalks = pq.read_table(io.BytesIO(archive.read("clipgt/crosswalk.parquet"))).to_pylist()
        raw_lanes = pq.read_table(io.BytesIO(archive.read("clipgt/lane.parquet"))).to_pylist()
    selected = [r for r in crosswalks if r["key"]["map_id"] == str(crosswalk_id)]
    if len(selected) != 1:
        raise ValueError("one original crosswalk ID required")
    original = selected[0]
    if original["crosswalk"]["category"] != "PEDESTRIAN":
        raise ValueError("pedestrian crosswalk source required")
    expected_clip = artifact.metadata.scene_id.removeprefix("clipgt-")
    if original["key"]["clip_id"] != expected_clip:
        raise ValueError("crosswalk source clip differs from the artifact")
    vector_map = artifact.map
    if vector_map is None or artifact._map_source != "clipgt":
        raise ValueError("actual ClipGT map convention required; no cross-source coordinate inference")
    lanes = original_reference_lanes(vector_map, lane_ids)
    alignment = []
    for lane_id, lane in zip(lane_ids, lanes):
        raw = [r for r in raw_lanes if r["key"]["map_id"] == lane_id]
        if len(raw) != 1:
            raise ValueError("one raw source record per lane required")
        raw = raw[0]
        if (raw["key"]["clip_id"] != expected_clip
            or raw["key"]["map_id_version"] != original["key"]["map_id_version"]
            or raw["lane"]["egomotion_label_class_id"] != original["crosswalk"]["egomotion_label_class_id"]):
            raise ValueError("crosswalk/lane clip, map version or coordinate source differs")
        errors = []
        for field, actual in (("left_rail", lane.left_edge), ("right_rail", lane.right_edge)):
            xy = [[p["x"], p["y"]] for p in raw["lane"][field]]
            errors.append(LineString(xy).hausdorff_distance(LineString(actual.xyz[:, :2])))
        if not np.isfinite(errors).all() or max(errors) > 1e-5:
            raise ValueError("original map XY rails differ from the actual NRE map convention")
        alignment.append({"lane_id": lane_id, "rail_hausdorff_error_m": errors})
    first = lanes[0]
    boundary = np.concatenate((first.left_edge.xyz[:, :2], first.right_edge.xyz[::-1, :2])).tolist()
    centerlines = [lane.center.xyz[:, :2].tolist() for lane in lanes]
    crosswalk_xy = [[p["x"], p["y"]] for p in original["crosswalk"]["location"]]
    segment = crosswalk_approach_segment(centerlines, boundary, crosswalk_xy, far_distance_m, near_distance_m)
    stat = path.stat()
    return {"scope": "original_crosswalk_approach_geometry_not_grounding_or_qualification", "qualified": False,
            "human_reviewed": False, "segment": segment, "source_crosswalk": original,
            "source_lane_ids": lane_ids, "centerlines_xy": centerlines, "original_lane_polygon_xy": boundary,
            "coordinate_alignment": alignment, "source_artifact": str(path), "source_size_bytes": stat.st_size,
            "source_mtime_ns": str(stat.st_mtime_ns), "artifact_uuid": artifact.metadata.uuid,
            "scene_id_private": artifact.metadata.scene_id, "explicit_reference_chain_is_private": True,
            "remaining": ["actual model-resolution visual referent identification", "approach language and geometry review",
                          "actual paired contract verification", "legality and rendering review", "formal qualification"]}
