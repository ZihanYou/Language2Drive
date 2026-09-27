"""Recorded-route topology hypotheses using AlpaSim's authoritative map loader.

This module is optional at import time: the lightweight benchmark core does not
depend on NumPy/trajdata.  Run it in the pinned AlpaSim environment so the
OpenDRIVE/ClipGT coordinate transform and lane search match runtime behavior.
Its output is candidate-mining evidence only, never an eligibility decision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List


def _lane_ids(value: Any) -> List[str]:
    # Trajdata uses -1 as a no-neighbor sentinel in some imported OpenDRIVE
    # maps.  It is not a lane ID and must never become a lane-choice proposal.
    return sorted(
        lane_id
        for lane_id in (str(item).strip() for item in (value or []))
        if lane_id and lane_id.lower() not in {"-1", "none", "null"}
    )


def _downsample_xyz(points: Any, limit: int = 64) -> List[List[float]]:
    rows = [[float(value) for value in row[:3]] for row in points]
    if len(rows) <= limit:
        return rows
    indices = sorted(
        {round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)}
    )
    return [rows[index] for index in indices]


def _nearest_xy(points: List[List[float]], target: List[float]) -> List[float]:
    return min(
        points,
        key=lambda point: (point[0] - target[0]) ** 2
        + (point[1] - target[1]) ** 2,
    )[:2]


def _lane_geometry(lane: Any) -> Dict[str, Any]:
    center = _downsample_xyz(lane.center.xyz)
    left = _downsample_xyz(lane.left_edge.xyz) if lane.left_edge is not None else []
    right = _downsample_xyz(lane.right_edge.xyz) if lane.right_edge is not None else []
    geometry: Dict[str, Any] = {
        "centerline_xyz": center,
        "left_edge_xyz": left,
        "right_edge_xyz": right,
        "commitment_gate_xy_not_qualified": None,
        "lane_region_polygon_xy_not_qualified": None,
    }
    if len(center) >= 2:
        center_index = min(max(1, len(center) // 4), len(center) - 1)
        target = center[center_index]
        if left and right:
            gate = [_nearest_xy(left, target), _nearest_xy(right, target)]
        else:
            previous = center[center_index - 1]
            dx, dy = target[0] - previous[0], target[1] - previous[1]
            norm = math.hypot(dx, dy)
            if norm > 1.0e-6:
                nx, ny = -dy / norm, dx / norm
                gate = [
                    [target[0] - 3.0 * nx, target[1] - 3.0 * ny],
                    [target[0] + 3.0 * nx, target[1] + 3.0 * ny],
                ]
            else:
                gate = None
        geometry["commitment_gate_xy_not_qualified"] = gate
    if len(left) >= 2 and len(right) >= 2:
        geometry["lane_region_polygon_xy_not_qualified"] = [
            point[:2] for point in left + list(reversed(right))
        ]
    return geometry


def attach_map_geometry(
    summary: Dict[str, Any], vector_map: Any, trajectory_positions: Any
) -> Dict[str, Any]:
    """Attach review-only lane/gate geometry to summarized hypotheses."""

    summary["recorded_trajectory_xyz_private"] = _downsample_xyz(
        trajectory_positions, limit=101
    )
    for hypothesis in summary.get("route_branch_hypotheses", []) + summary.get(
        "lane_choice_hypotheses", []
    ):
        lane_ids = [str(hypothesis["lane_id"])]
        for key in (
            "legal_next_lanes",
            "adjacent_lanes_left",
            "adjacent_lanes_right",
        ):
            lane_ids.extend(str(value) for value in hypothesis.get(key, []))
        geometry = {}
        errors = []
        for lane_id in dict.fromkeys(lane_ids):
            try:
                geometry[lane_id] = _lane_geometry(vector_map.get_road_lane(lane_id))
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                errors.append("lane_geometry_failed:%s:%s" % (lane_id, type(error).__name__))
        hypothesis["lane_geometry_private_not_qualified"] = geometry
        hypothesis["lane_geometry_errors"] = errors
    return summary


def summarize_lane_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compress lane samples and expose route-reachable decision hypotheses."""

    if not samples:
        return {
            "sample_count": 0,
            "lane_span_count": 0,
            "lane_spans": [],
            "route_branch_hypotheses": [],
            "lane_choice_hypotheses": [],
        }

    origin_us = int(samples[0]["timestamp_us"])
    spans: List[Dict[str, Any]] = []
    for sample in samples:
        lane_id = str(sample["lane_id"])
        if not spans or spans[-1]["lane_id"] != lane_id:
            spans.append(
                {
                    "lane_id": lane_id,
                    "first_timestamp_us": int(sample["timestamp_us"]),
                    "last_timestamp_us": int(sample["timestamp_us"]),
                    "first_offset_us": int(sample["timestamp_us"]) - origin_us,
                    "last_offset_us": int(sample["timestamp_us"]) - origin_us,
                }
            )
        else:
            spans[-1]["last_timestamp_us"] = int(sample["timestamp_us"])
            spans[-1]["last_offset_us"] = int(sample["timestamp_us"]) - origin_us

    branches: List[Dict[str, Any]] = []
    lane_choices: List[Dict[str, Any]] = []
    seen_branch_lanes = set()
    seen_choice_lanes = set()
    for sample_index, sample in enumerate(samples):
        lane_id = str(sample["lane_id"])
        offset_us = int(sample["timestamp_us"]) - origin_us
        base = {
            "lane_id": lane_id,
            "timestamp_us": int(sample["timestamp_us"]),
            "offset_us": offset_us,
            "suggested_start_time_offset_us_not_qualified": max(0, offset_us - 3_000_000),
            "position_xyz": [float(value) for value in sample["position_xyz"]],
        }
        next_lanes = _lane_ids(sample.get("next_lanes"))
        if len(next_lanes) >= 2 and lane_id not in seen_branch_lanes:
            observed_next_sample = next(
                (
                    future
                    for future in samples[sample_index + 1 :]
                    if str(future["lane_id"]) != lane_id
                ),
                None,
            )
            observed_next_lane = str(observed_next_sample["lane_id"]) if observed_next_sample is not None else None
            row = dict(base)
            row.update(
                {
                    "legal_next_lanes": next_lanes,
                    "observed_recorded_next_lane": observed_next_lane,
                    "timing_observation_not_qualified": {
                        "marker_semantics": "approach_lane_first_sample_not_commitment_time",
                        "approach_lane_first_sample_offset_us": offset_us,
                        "next_different_lane_first_sample_offset_us": (
                            int(observed_next_sample["timestamp_us"]) - origin_us
                            if observed_next_sample is not None else None),
                        "next_different_lane_is_listed_successor": observed_next_lane in next_lanes,
                        "exact_gate_crossing_verified": False,
                    },
                    "hypotheses_not_qualified": [
                        "A1_INTERSECTION_BRANCH",
                        "A2_STRAIGHT_VS_TURN",
                    ],
                }
            )
            if len(next_lanes) >= 3:
                row["hypotheses_not_qualified"].append("A3_FORK")
            branches.append(row)
            seen_branch_lanes.add(lane_id)

        left = _lane_ids(sample.get("adj_lanes_left"))
        right = _lane_ids(sample.get("adj_lanes_right"))
        if (left or right) and lane_id not in seen_choice_lanes:
            hypotheses = ["A5_LANE_CHANGE_VS_KEEP", "B2_SCOPED_LANE_PREFERENCE"]
            if left and right:
                hypotheses.insert(0, "A4_LANE_CHANGE_LEFT_VS_RIGHT")
            row = dict(base)
            row.update(
                {
                    "adjacent_lanes_left": left,
                    "adjacent_lanes_right": right,
                    "hypotheses_not_qualified": hypotheses,
                }
            )
            lane_choices.append(row)
            seen_choice_lanes.add(lane_id)

    distance_m = 0.0
    for first, second in zip(samples, samples[1:]):
        p0, p1 = first["position_xyz"], second["position_xyz"]
        distance_m += math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(p0, p1)))
    return {
        "sample_count": len(samples),
        "recording_trajectory_start_timestamp_us": origin_us,
        "recording_trajectory_end_timestamp_us": int(samples[-1]["timestamp_us"]),
        "recorded_path_length_m": distance_m,
        "lane_span_count": len(spans),
        "lane_spans": spans,
        "route_branch_hypotheses": branches,
        "lane_choice_hypotheses": lane_choices,
    }


def _projection_distance(lane: Any, position: Any, np: Any) -> float:
    try:
        projected = lane.center.project_onto(np.asarray(position).reshape(1, 3))[0, :3]
        return float(np.linalg.norm(np.asarray(position)[:3] - projected))
    except (AttributeError, IndexError, TypeError, ValueError):
        return float("inf")


def _sample_route_lanes(vector_map: Any, positions: Any, timestamps: Any, np: Any) -> List[Dict[str, Any]]:
    positions = np.asarray(positions)
    timestamps = np.asarray(timestamps)
    if positions.ndim != 2 or positions.shape[1] < 3 or len(positions) != len(timestamps):
        raise ValueError("invalid recorded trajectory arrays")
    if len(positions) < 2:
        raise ValueError("recorded trajectory requires at least two poses")

    samples: List[Dict[str, Any]] = []
    heading = 0.0
    previous_lane = None
    for index, position in enumerate(positions):
        if index + 1 < len(positions):
            motion = positions[index + 1, :2] - position[:2]
            if float(np.linalg.norm(motion)) > 1.0e-2:
                heading = float(math.atan2(float(motion[1]), float(motion[0])))
        pose_xyzh = np.concatenate((position[:3], np.asarray([heading])))
        candidates = list(vector_map.get_current_lane(pose_xyzh, max_dist=4.0))
        used_closest_fallback = False
        if previous_lane is not None and any(lane.id == previous_lane.id for lane in candidates):
            lane = previous_lane
        else:
            continuation = []
            if previous_lane is not None:
                continuation = [
                    candidate
                    for candidate in candidates
                    if candidate.id in set(previous_lane.next_lanes)
                ]
            pool = continuation or candidates
            if pool:
                lane = min(pool, key=lambda item: _projection_distance(item, position[:3], np))
            else:
                lane = vector_map.get_closest_lane(position[:3])
                used_closest_fallback = True
        samples.append(
            {
                "trajectory_index": index,
                "timestamp_us": int(timestamps[index]),
                "position_xyz": [float(value) for value in position[:3]],
                "lane_id": str(lane.id),
                "next_lanes": _lane_ids(lane.next_lanes),
                "adj_lanes_left": _lane_ids(lane.adj_lanes_left),
                "adj_lanes_right": _lane_ids(lane.adj_lanes_right),
                "used_closest_lane_fallback": used_closest_fallback,
            }
        )
        previous_lane = lane
    return samples


def mine_route_aligned_artifact(path: Path) -> Dict[str, Any]:
    """Mine only topology reached by the real recorded ego trajectory."""

    result: Dict[str, Any] = {
        "available": False,
        "qualified": False,
        "evidence_scope": "authoritative_alpasim_map_plus_recorded_rig_trajectory",
        "errors": [],
    }
    try:
        import numpy as np
        from alpasim_utils.artifact import Artifact
    except ImportError:
        result["errors"].append("alpasim_map_dependencies_unavailable")
        return result

    try:
        artifact = Artifact(source=str(Path(path).resolve()))
        vector_map = artifact.map
        if vector_map is None:
            result["errors"].append("artifact_map_unavailable")
            return result
        trajectory = artifact.rig.trajectory
        samples = _sample_route_lanes(
            vector_map,
            trajectory.positions,
            trajectory.timestamps_us,
            np,
        )
        summary = attach_map_geometry(
            summarize_lane_samples(samples), vector_map, trajectory.positions
        )
        result.update(summary)
        result.update(
            {
                "available": True,
                "map_source": artifact.map_source,
                "closest_lane_fallback_count": sum(
                    bool(row["used_closest_lane_fallback"]) for row in samples
                ),
                "qualification_warning": (
                    "Route reachability does not establish camera visibility, reconstructed "
                    "support for both arms, legality, safety, or oracle feasibility."
                ),
            }
        )
    except Exception as error:  # Preserve per-scene progress while reporting the exact failure type.
        result["errors"].append("route_alignment_failed:%s" % type(error).__name__)
    return result
