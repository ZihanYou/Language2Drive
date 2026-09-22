"""Shared validator for executable Language2Drive exact-parking witnesses.

The v1 generator used to label the lane/slot-axis intersection with the slot
heading even when the live lane heading differed by almost ninety degrees.
That made a static-looking ``approach_pose`` impossible to reach and tempted
the oracle to perform an unsafe forward, full-lock alignment manoeuvre.

Version 4 closes the geometry/control contract.  The approach is a physical
lane pose, while every subsequent forward/reverse pose is serialized,
kinematically bounded in the measured rear-axle frame, checked against the
vertically relevant frozen CARLA level boxes in the actor frame, hashed, and
embedded verbatim as the collapsed actor-pose suffix of ``oracle_path``.  Both
the evaluator loader and the privileged oracle call this validator; neither
is allowed to synthesize a manoeuvre or infer a rear path from actor poses.

Version 5 keeps the physical slot-axis ``approach_pose`` as metadata, but
adds one exact, stored-XODR prelead at ``poses[0]``.  The ordinary route is
proof-bound to that prelead by a dense topology tail.  Runtime consumers must
therefore hand off at the prelead, never at the slot-axis metadata pose.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from team_code.language2drive_signal_free_proof import (
    SignalFreeCorridorProofError,
    stable_hash,
)


PARKING_APPROACH_METHOD = "slot_axis_lane_intersection_v2"
PARKING_REVERSE_SCHEMA = "language2drive.parking_reverse_manoeuvre.v4"
PARKING_REVERSE_METHOD = "audited_reverse_manoeuvre_v4"
PARKING_MANOEUVRE_PROOF_METHOD = (
    "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v4"
)
PARKING_REGISTRY_SCHEMA = "language2drive.parking_manoeuvre_registry.v4"
PARKING_APPROACH_METHOD_V5 = "slot_axis_lane_intersection_v5"
PARKING_REVERSE_SCHEMA_V5 = "language2drive.parking_reverse_manoeuvre.v5"
PARKING_REVERSE_METHOD_V5 = "audited_slot_axis_to_reverse_manoeuvre_v5"
PARKING_MANOEUVRE_PROOF_METHOD_V5 = (
    "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v5"
)
PARKING_REGISTRY_SCHEMA_V5 = "language2drive.parking_manoeuvre_registry.v5"
PARKING_CONTROL_SCHEMA = "language2drive.parking_axle_control_model.v4"
PARKING_VERTICAL_METHOD = "carla_obb_world_z_interval_overlap_v1"
PARKING_SAMPLE_SPACING_M = 0.1
PARKING_SLOT_TOLERANCE_M = 0.2
PARKING_STATIC_CLEARANCE_MIN_M = 0.2
PARKING_VERTICAL_PADDING_M = 0.05
PARKING_MAX_YAW_STEP_DEG = 5.0
PARKING_MAX_MOTION_HEADING_ERROR_DEG = 5.0
PARKING_MAX_NORMALIZED_CONTROL_STEER = 0.8
PARKING_VEHICLE_BLUEPRINT = "vehicle.lincoln.mkz_2020"
PARKING_POSE_REFERENCE = "carla_actor_transform"
PARKING_INTEGRATION_REFERENCE = "rear_axle_center"
PARKING_DYNAMICS_SCHEMA = "language2drive.vehicle_dynamics_evidence.v1"
PARKING_DYNAMICS_SCOPE_STATEMENT = (
    "This immutable asset records one installed CARLA package's Lincoln "
    "wheel geometry and stationary normalized-steer response. Piecewise "
    "interpolation is valid only inside [-0.8,0.8]; live collision-free "
    "multi-seed parking witnesses remain authoritative."
)
PARKING_DYNAMICS_SOURCE_REF = (
    "/root/autodl-tmp/lwq/language2drive.v1/generated/maps/"
    "vehicle_dynamics/lincoln_mkz_2020.dynamics.v1.json"
)
PARKING_DYNAMICS_SOURCE_FILE_SHA256 = (
    "031643aa33e5f234ea6ba4c87a403908aee7942e322802b4b1b0e545d38dde47"
)
PARKING_DYNAMICS_PAYLOAD_SHA256 = (
    "d94392a5c4118dcb8f8660461267cfa1bb44e5f7c9fb111b14057d862d948791"
)
PARKING_WHEEL_GEOMETRY_SHA256 = (
    "55fdf3b68124610b471141f39610ebbf5f21c79e9d53f1464dbd738227285dee"
)
PARKING_NORMALIZED_STEER_SAMPLES_SHA256 = (
    "208e6f69bea0c31037256b266c785b855a697202c4cff1ffc70e70da95974052"
)
PARKING_RAW_STEER_SWEEPS_SHA256 = (
    "b5789ff819d19e4f9d6f36cd76b54a53ac001ad900e98793cf940a78465330ed"
)
PARKING_FRONT_AXLE_X_M = 1.471838832
PARKING_REAR_AXLE_X_M = -1.388632059
PARKING_WHEELBASE_M = 2.860470891
PARKING_FOOTPRINT_LENGTH_M = 4.892382
PARKING_FOOTPRINT_WIDTH_M = 1.836713
PARKING_V5_TOPOLOGY_SCHEMA = (
    "language2drive.parking_v5_stored_xodr_topology.v1"
)
PARKING_V5_TOPOLOGY_METHOD = (
    "stored_xodr_construction_waypoint_sequence_v1"
)
PARKING_V5_TARGET_POSE_INDEX_SPACE = "reverse_manoeuvre.poses"
PARKING_V5_ARM_POSE_INDEX_SPACE = "arm.oracle_path"
PARKING_V5_ROUTE_SCHEMA = "language2drive.parking_v5_route_proof.v1"
PARKING_V5_ROUTE_METHOD = "oracle_path_stored_xodr_tail_exact_suffix_v1"
PARKING_V5_TARGET_ALONG_TRACK_RESIDUAL_MAX_M = 0.1001
PARKING_V5_TARGET_Z_RESIDUAL_MAX_M = 0.1001
PARKING_V5_DENSE_TAIL_MIN_M = 7.0
PARKING_V5_ALIGNED_LEAD_IN_MIN_M = 8.0
PARKING_V5_EPS = 1e-9

APPROACH_POSE_KEYS = frozenset({"x", "y", "z", "yaw_deg"})
APPROACH_PROOF_KEYS = frozenset({
    "method",
    "distance_from_slot_center_m",
    "heading_error_deg",
    "lane_id",
    "lane_yaw_deg",
    "road_id",
    "section_id",
    "reverse",
})
APPROACH_REVERSE_KEYS = frozenset({
    "method",
    "own_slot_tolerance_m",
    "sample_spacing_max_m",
    "samples",
})
REVERSE_MANOEUVRE_KEYS = frozenset({
    "schema_version", "poses", "rear_axle_poses", "proof",
})
REVERSE_POSE_KEYS = frozenset({"x", "y", "z", "yaw_deg", "gear"})
REVERSE_PROOF_KEYS = frozenset({
    "method",
    "pose_reference",
    "integration_reference",
    "poses_sha256",
    "rear_axle_poses_sha256",
    "pose_count",
    "sample_spacing_max_m",
    "rear_axle_sample_spacing_max_m",
    "required_max_rear_axle_curvature_1pm",
    "required_max_abs_normalized_steer",
    "max_normalized_control_steer",
    "vehicle_blueprint",
    "footprint_length_m",
    "footprint_width_m",
    "footprint_tolerance_m",
    "full_footprint_legal_samples",
    "static_clearance_samples",
    "static_collision_samples",
    "static_clearance_min_m",
    "static_bbox_categories",
    "static_bbox_count",
    "static_bboxes_sha256",
    "town",
    "xodr_sha256",
    "lane_graph_sha256",
    "vertical_overlap_proof",
})
REVERSE_PROOF_KEYS_V5 = REVERSE_PROOF_KEYS | frozenset({
    "aligned_lead_in_start_index",
    "slot_axis_approach_index",
    "construction_topology_proof",
})

PARKING_V5_TOPOLOGY_KEYS = frozenset({
    "schema_version", "method", "pose_index_space", "first_pose_index",
    "last_pose_index", "samples", "samples_sha256",
    "canonical_waypoint_transforms_sha256", "runs", "transitions",
})
PARKING_V5_TOPOLOGY_SAMPLE_KEYS = frozenset({
    "pose_index", "road_id", "section_id", "lane_id", "s_m",
})
PARKING_V5_TOPOLOGY_RUN_KEYS = frozenset({
    "first_pose_index", "last_pose_index", "lane_graph_id",
    "travel_s_direction",
})
PARKING_V5_TOPOLOGY_TRANSITION_KEYS = frozenset({
    "from_run_index", "to_run_index",
})
PARKING_V5_ROUTE_PROOF_KEYS = frozenset({
    "schema_version", "method", "oracle_path_sha256",
    "ordinary_tail_start_index", "explicit_suffix_start_index",
    "ordinary_tail_sha256", "explicit_suffix_sha256",
    "target_prelead_sample_sha256", "ordinary_tail_topology_proof",
})

VERTICAL_OVERLAP_PROOF_KEYS = frozenset({
    "method",
    "vertical_padding_m",
    "ego_local_bbox_sha256",
    "pose_count",
    "static_bbox_count",
    "checked_pair_count",
    "vertically_overlapping_pair_count",
    "vertically_disjoint_pair_count",
    "overlap_index_sets_sha256",
})

PARKING_REGISTRY_KEYS = frozenset({"schema_version", "method", "towns"})
PARKING_TOWN_REGISTRY_KEYS = frozenset({
    "town", "bbox_asset", "xodr", "lane_graph", "control_model",
    "vehicle_geometry",
})
PARKING_TOWN_REGISTRY_KEYS_V5 = frozenset({
    *PARKING_TOWN_REGISTRY_KEYS, "slot_collision_occupancy",
})
PARKING_OCCUPANCY_KEYS = frozenset({
    "source_ref", "source_file_sha256", "schema_version", "method", "town",
    "map_name", "vehicle_blueprint", "spawn_z_m", "repeat_count", "slots_sha256",
    "slot_count", "clear_count", "blocked_count", "clear_slots", "blocked_slots",
    "clear_slots_sha256", "blocked_slots_sha256", "source_files", "scope_statement",
})
PARKING_BBOX_REGISTRY_KEYS = frozenset({
    "source_ref", "source_file_sha256", "schema_version", "bboxes_sha256",
    "bbox_count", "categories", "scope_statement", "excluded_scope",
    "quantization_decimals", "canonical_sort_key", "source_api",
})
PARKING_SOURCE_HASH_KEYS = frozenset({"source_ref", "sha256"})
PARKING_CONTROL_REGISTRY_KEYS = frozenset({
    "schema_version", "pose_reference", "integration_reference",
    "vehicle_dynamics_source_ref", "vehicle_dynamics_source_file_sha256",
    "wheel_geometry_sha256", "normalized_steer_samples_sha256",
    "front_axle_x_m", "rear_axle_x_m", "wheelbase_m",
    "max_normalized_control_steer",
})
PARKING_VEHICLE_GEOMETRY_KEYS = frozenset({
    "vehicle_blueprint", "coordinate_frame", "local_bbox",
    "local_bbox_sha256", "source_method", "source_trace_ref",
    "source_trace_sha256", "source_trace_rows", "matching_bbox_rows",
    "actor_reference_z_min_m", "actor_reference_z_max_m",
})
PARKING_LOCAL_BBOX_KEYS = frozenset({"location", "rotation", "extent"})
PARKING_LOCATION_KEYS = frozenset({"x", "y", "z"})
PARKING_ROTATION_KEYS = frozenset({"pitch", "yaw", "roll"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
PARKING_STATIC_BBOX_CATEGORIES = [
    "Buildings",
    "Fences",
    "GuardRail",
    "Other",
    "Poles",
    "Static",
    "TrafficLight",
    "TrafficSigns",
    "Vegetation",
    "Walls",
]

PARKING_BBOX_EXCLUDED_SCOPE = [
    "Roads", "RoadLines", "Ground", "Sidewalks", "Terrain", "Water",
    "Sky", "RailTrack", "Bridge", "Vehicles", "Pedestrians", "Dynamic",
    "Any", "None",
]
PARKING_BBOX_CANONICAL_SORT_KEY = [
    "category", "center.x", "center.y", "center.z", "extent.x",
    "extent.y", "extent.z", "rotation.pitch", "rotation.yaw",
    "rotation.roll",
]
PARKING_BBOX_SCOPE_STATEMENT = (
    "Complete 2D ego footprints are checked against CARLA "
    "world.get_level_bbs for the frozen non-road semantic categories only; "
    "this is not a claim about unlabelled Unreal collision meshes. Live "
    "three-seed collision witnesses remain authoritative."
)

_REGISTRY_CACHE: Dict[tuple[str, str, bool], Dict[str, Any]] = {}
_V5_MAP_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}
_V5_LANE_POINT_CACHE: Dict[tuple[str, float, float, float], bool] = {}


class ParkingManoeuvreProofError(ValueError):
    """A core parking target has no executable, closed parking manoeuvre."""


def _angle_error_degrees(value: float, target: float) -> float:
    return abs((float(value) - float(target) + 180.0) % 360.0 - 180.0)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ParkingManoeuvreProofError(f"{field} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ParkingManoeuvreProofError(
            f"{field} must be finite numeric"
        ) from exc
    if not math.isfinite(result):
        raise ParkingManoeuvreProofError(f"{field} must be finite numeric")
    return result


def _strict_finite(value: Any, field: str) -> float:
    """Accept only a finite JSON number (never bool or a numeric string)."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ParkingManoeuvreProofError(
            f"{field} must be a finite JSON number, not bool"
        )
    return float(value)


def _canonical_yaw(value: Any, decimals: int = 3) -> float:
    result = round((_strict_finite(value, "yaw_deg") + 180.0) % 360.0 - 180.0, decimals)
    return 0.0 if result == 0.0 else result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ParkingManoeuvreProofError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _pose(value: Any, field: str, expected_keys: frozenset[str]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ParkingManoeuvreProofError(
            f"{field} keys must be exactly {sorted(expected_keys)}"
        )
    output: Dict[str, Any] = {
        axis: _finite(value[axis], f"{field}.{axis}")
        for axis in ("x", "y", "z", "yaw_deg")
    }
    if "gear" in expected_keys:
        gear = value["gear"]
        if gear not in {"forward", "reverse"}:
            raise ParkingManoeuvreProofError(
                f"{field}.gear must be forward or reverse"
            )
        output["gear"] = gear
    return output


def _polygon_center(polygon: Any) -> tuple[float, float]:
    if not isinstance(polygon, list) or len(polygon) < 4:
        raise ParkingManoeuvreProofError(
            "parking target polygon must contain at least three vertices"
        )
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    try:
        points = [
            (_finite(point[0], "target.polygon.x"),
             _finite(point[1], "target.polygon.y"))
            for point in vertices
        ]
    except (IndexError, TypeError) as exc:
        raise ParkingManoeuvreProofError(
            "parking target polygon points require x/y"
        ) from exc
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parking_hash(value: Any, field: str) -> str:
    try:
        return stable_hash(value)
    except (SignalFreeCorridorProofError, TypeError, ValueError) as exc:
        raise ParkingManoeuvreProofError(
            f"{field} is not canonical finite JSON"
        ) from exc


def _sha256(value: Any, field: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ParkingManoeuvreProofError(
            f"{field} must be lowercase SHA256"
        )
    return text


def _source_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ParkingManoeuvreProofError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ParkingManoeuvreProofError(f"{field} must be absolute")
    # Preserve the producer's canonical lexical workspace spelling. Path
    # resolution follows shared-inode symlinks (lwq -> simlingo_fzh on some
    # hosts) and would break the exact top-level source manifest identity.
    return Path(os.path.abspath(str(path)))


def _verified_source_bytes(reference: Any, digest: Any, field: str) -> bytes:
    path = _source_path(reference, f"{field}.source_ref")
    expected = _sha256(digest, f"{field}.source_file_sha256")
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise ParkingManoeuvreProofError(
            f"cannot read frozen parking source {path}: {exc}"
        ) from exc
    actual = hashlib.sha256(value).hexdigest()
    if actual != expected:
        raise ParkingManoeuvreProofError(
            f"{field} source SHA256 mismatch"
        )
    return value


def _require_exact_keys(value: Any, keys: frozenset[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ParkingManoeuvreProofError(
            f"{field} keys must be exactly {sorted(keys)}"
        )
    return value


def _signed_angle_radians(value: float, target: float) -> float:
    return math.radians((float(value) - float(target) + 180.0) % 360.0 - 180.0)


def normalized_steer_for_curvature(
    curvature_1pm: float,
    canonical_samples: Sequence[Mapping[str, Any]],
) -> float:
    """Invert the frozen signed steer/curvature map piecewise-linearly.

    No extrapolation is permitted.  This function is shared by proof
    validation and the runtime controller so the required normalized command
    cannot drift from the immutable 33-point qualification envelope.
    """

    curvature = _finite(curvature_1pm, "rear_axle_curvature_1pm")
    if len(canonical_samples) != 33:
        raise ParkingManoeuvreProofError(
            "normalized steer map must contain exactly 33 samples"
        )
    if any(not isinstance(sample, Mapping) for sample in canonical_samples):
        raise ParkingManoeuvreProofError(
            "normalized steer map samples must be objects"
        )
    pairs = [
        (
            _finite(sample.get("ackermann_center_curvature_1pm"),
                    f"normalized_steer_samples[{index}].curvature"),
            _finite(sample.get("normalized_steer"),
                    f"normalized_steer_samples[{index}].normalized_steer"),
        )
        for index, sample in enumerate(canonical_samples)
    ]
    epsilon = 1e-12
    if curvature < pairs[0][0] - epsilon or curvature > pairs[-1][0] + epsilon:
        raise ParkingManoeuvreProofError(
            "rear-axle curvature exceeds normalized-steer qualification envelope"
        )
    if curvature <= pairs[0][0]:
        return pairs[0][1]
    if curvature >= pairs[-1][0]:
        return pairs[-1][1]
    for (first_k, first_u), (second_k, second_u) in zip(pairs, pairs[1:]):
        if first_k - epsilon <= curvature <= second_k + epsilon:
            span = second_k - first_k
            if span <= 0.0:
                raise ParkingManoeuvreProofError(
                    "normalized steer curvature map is not strictly increasing"
                )
            ratio = (curvature - first_k) / span
            return first_u + ratio * (second_u - first_u)
    raise ParkingManoeuvreProofError(
        "rear-axle curvature cannot be inverted in normalized-steer map"
    )


def curvature_for_normalized_steer(
    normalized_steer: float,
    canonical_samples: Sequence[Mapping[str, Any]],
) -> float:
    """Evaluate the frozen steer/curvature map without extrapolation."""

    steer = _finite(normalized_steer, "normalized_steer")
    if len(canonical_samples) != 33:
        raise ParkingManoeuvreProofError(
            "normalized steer map must contain exactly 33 samples"
        )
    if any(not isinstance(sample, Mapping) for sample in canonical_samples):
        raise ParkingManoeuvreProofError(
            "normalized steer map samples must be objects"
        )
    pairs = [
        (
            _finite(sample.get("normalized_steer"),
                    f"normalized_steer_samples[{index}].normalized_steer"),
            _finite(sample.get("ackermann_center_curvature_1pm"),
                    f"normalized_steer_samples[{index}].curvature"),
        )
        for index, sample in enumerate(canonical_samples)
    ]
    epsilon = 1e-12
    if steer < pairs[0][0] - epsilon or steer > pairs[-1][0] + epsilon:
        raise ParkingManoeuvreProofError(
            "normalized steer exceeds qualification envelope"
        )
    if steer <= pairs[0][0]:
        return pairs[0][1]
    if steer >= pairs[-1][0]:
        return pairs[-1][1]
    for (first_u, first_k), (second_u, second_k) in zip(pairs, pairs[1:]):
        if first_u - epsilon <= steer <= second_u + epsilon:
            span = second_u - first_u
            if span <= 0.0:
                raise ParkingManoeuvreProofError(
                    "normalized steer command map is not strictly increasing"
                )
            ratio = (steer - first_u) / span
            return first_k + ratio * (second_k - first_k)
    raise ParkingManoeuvreProofError(
        "normalized steer cannot be evaluated in curvature map"
    )


def _validate_vehicle_dynamics_asset(
    payload: Any, *, verify_sources: bool,
) -> Dict[str, Any]:
    """Recursively close the immutable Lincoln dynamics evidence asset."""

    root_keys = frozenset({
        "schema_version", "vehicle_blueprint", "capture",
        "coordinate_frames", "wheel_geometry", "wheel_geometry_sha256",
        "physics_control", "normalized_steer_response", "scope_statement",
        "source_files",
    })
    asset = _require_exact_keys(payload, root_keys, "vehicle_dynamics")
    if (
        asset["schema_version"] != PARKING_DYNAMICS_SCHEMA
        or asset["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT
        or _parking_hash(asset, "vehicle_dynamics") != PARKING_DYNAMICS_PAYLOAD_SHA256
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics schema/blueprint/payload hash is invalid"
        )

    capture = _require_exact_keys(asset["capture"], frozenset({
        "actor_transform", "client_version", "map_name", "server_version",
        "source_api", "town",
    }), "vehicle_dynamics.capture")
    actor_transform = _require_exact_keys(capture["actor_transform"], frozenset({
        "location", "rotation",
    }), "vehicle_dynamics.capture.actor_transform")
    actor_location = _require_exact_keys(
        actor_transform["location"], PARKING_LOCATION_KEYS,
        "vehicle_dynamics.capture.actor_transform.location",
    )
    actor_rotation = _require_exact_keys(
        actor_transform["rotation"], PARKING_ROTATION_KEYS,
        "vehicle_dynamics.capture.actor_transform.rotation",
    )
    for axis in PARKING_LOCATION_KEYS:
        _finite(actor_location[axis], f"capture.actor_transform.location.{axis}")
    for axis in PARKING_ROTATION_KEYS:
        _finite(actor_rotation[axis], f"capture.actor_transform.rotation.{axis}")
    if (
        not isinstance(capture["town"], str) or not capture["town"]
        or any(
            not isinstance(capture[field], str) or not capture[field]
            for field in (
                "client_version", "server_version", "map_name",
            )
        )
        or capture["source_api"] != [
            "carla.Vehicle.get_physics_control",
            "carla.Vehicle.get_wheel_steer_angle",
        ]
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics capture provenance is invalid"
        )
    frames = _require_exact_keys(asset["coordinate_frames"], frozenset({
        "actor", "raw_wheel_positions",
    }), "vehicle_dynamics.coordinate_frames")
    if frames != {
        "actor": "carla_actor_local_metres",
        "raw_wheel_positions": (
            "carla_WheelPhysicsControl_position_world_centimetres"
        ),
    }:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics coordinate-frame convention is invalid"
        )
    if asset["scope_statement"] != PARKING_DYNAMICS_SCOPE_STATEMENT:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics scope statement is not frozen"
        )

    wheel = _require_exact_keys(asset["wheel_geometry"], frozenset({
        "front_axle_x_m", "front_track_m", "rear_axle_x_m",
        "rear_track_m", "wheelbase_m", "wheels",
    }), "vehicle_dynamics.wheel_geometry")
    if asset["wheel_geometry_sha256"] != _parking_hash(
        wheel, "vehicle_dynamics.wheel_geometry"
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics wheel_geometry_sha256 mismatch"
        )
    wheel_sha = _sha256(
        asset["wheel_geometry_sha256"],
        "vehicle_dynamics.wheel_geometry_sha256",
    )
    if wheel_sha != PARKING_WHEEL_GEOMETRY_SHA256:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics wheel geometry is not the frozen Lincoln asset"
        )
    wheels = wheel["wheels"]
    if not isinstance(wheels, list) or len(wheels) != 4:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics requires exactly four wheels"
        )
    wheel_keys = frozenset({
        "max_steer_angle_deg", "name", "position_actor_local_m",
        "position_world_cm_raw", "radius_cm",
    })
    wheel_names = ["front_left", "front_right", "rear_left", "rear_right"]
    parsed_wheels = []
    for index, (raw_wheel, expected_name) in enumerate(zip(wheels, wheel_names)):
        item = _require_exact_keys(
            raw_wheel, wheel_keys, f"vehicle_dynamics.wheels[{index}]"
        )
        if item["name"] != expected_name:
            raise ParkingManoeuvreProofError(
                "vehicle dynamics wheel order/name is invalid"
            )
        local = _require_exact_keys(
            item["position_actor_local_m"], PARKING_LOCATION_KEYS,
            f"vehicle_dynamics.wheels[{index}].position_actor_local_m",
        )
        raw_position = _require_exact_keys(
            item["position_world_cm_raw"], PARKING_LOCATION_KEYS,
            f"vehicle_dynamics.wheels[{index}].position_world_cm_raw",
        )
        parsed_wheels.append({
            "local": {axis: _finite(local[axis], f"wheel[{index}].local.{axis}")
                      for axis in PARKING_LOCATION_KEYS},
            "raw": {axis: _finite(raw_position[axis], f"wheel[{index}].raw.{axis}")
                    for axis in PARKING_LOCATION_KEYS},
            "max_steer": _finite(item["max_steer_angle_deg"],
                                 f"wheel[{index}].max_steer_angle_deg"),
            "radius": _finite(item["radius_cm"], f"wheel[{index}].radius_cm"),
        })
    front_x = _finite(wheel["front_axle_x_m"], "wheel_geometry.front_axle_x_m")
    rear_x = _finite(wheel["rear_axle_x_m"], "wheel_geometry.rear_axle_x_m")
    wheelbase = _finite(wheel["wheelbase_m"], "wheel_geometry.wheelbase_m")
    front_track = _finite(wheel["front_track_m"], "wheel_geometry.front_track_m")
    rear_track = _finite(wheel["rear_track_m"], "wheel_geometry.rear_track_m")
    if (
        abs(front_x - PARKING_FRONT_AXLE_X_M) > 1e-9
        or abs(rear_x - PARKING_REAR_AXLE_X_M) > 1e-9
        or abs(wheelbase - PARKING_WHEELBASE_M) > 1e-9
        or abs(front_x - rear_x - wheelbase) > 1e-9
        or abs(0.5 * (parsed_wheels[0]["local"]["x"]
                      + parsed_wheels[1]["local"]["x"]) - front_x) > 1e-9
        or abs(0.5 * (parsed_wheels[2]["local"]["x"]
                      + parsed_wheels[3]["local"]["x"]) - rear_x) > 1e-9
        or abs(abs(parsed_wheels[0]["local"]["y"]
                   - parsed_wheels[1]["local"]["y"]) - front_track) > 1e-9
        or abs(abs(parsed_wheels[2]["local"]["y"]
                   - parsed_wheels[3]["local"]["y"]) - rear_track) > 1e-9
        or any(parsed_wheels[index]["max_steer"] <= 0.0 for index in (0, 1))
        or any(abs(parsed_wheels[index]["max_steer"]) > 1e-12 for index in (2, 3))
        or any(item["radius"] <= 0.0 for item in parsed_wheels)
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics axle/wheel geometry does not close"
        )

    physics = _require_exact_keys(asset["physics_control"], frozenset({
        "mass_kg", "steering_curve",
    }), "vehicle_dynamics.physics_control")
    if _finite(physics["mass_kg"], "physics_control.mass_kg") <= 0.0:
        raise ParkingManoeuvreProofError("vehicle dynamics mass must be positive")
    steering_curve = physics["steering_curve"]
    if not isinstance(steering_curve, list) or not steering_curve:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics steering_curve must be non-empty"
        )
    previous_x = -math.inf
    for index, point in enumerate(steering_curve):
        item = _require_exact_keys(
            point, frozenset({"x_axis_raw", "scale"}),
            f"vehicle_dynamics.steering_curve[{index}]",
        )
        x_value = _finite(item["x_axis_raw"], f"steering_curve[{index}].x_axis_raw")
        scale = _finite(item["scale"], f"steering_curve[{index}].scale")
        if x_value <= previous_x or scale <= 0.0:
            raise ParkingManoeuvreProofError(
                "vehicle dynamics steering_curve is not ordered/positive"
            )
        previous_x = x_value

    response = _require_exact_keys(asset["normalized_steer_response"], frozenset({
        "canonical_samples", "canonical_samples_sha256", "command_max",
        "command_min", "command_step", "fixed_delta_seconds",
        "hysteresis_max_deg", "method", "raw_sweeps", "raw_sweeps_sha256",
        "settle_ticks_per_command",
    }), "vehicle_dynamics.normalized_steer_response")
    samples = response["canonical_samples"]
    sample_keys = frozenset({
        "ackermann_center_curvature_1pm",
        "ackermann_outer_angle_residual_deg", "front_left_deg",
        "front_right_deg", "normalized_steer",
    })
    if (
        not isinstance(samples, list) or len(samples) != 33
        or response["canonical_samples_sha256"] != _parking_hash(
            samples, "vehicle_dynamics.normalized_steer_response.canonical_samples"
        )
        or response["canonical_samples_sha256"]
        != PARKING_NORMALIZED_STEER_SAMPLES_SHA256
        or abs(_finite(response["command_min"], "steer.command_min") + 0.8) > 1e-12
        or abs(_finite(response["command_max"], "steer.command_max") - 0.8) > 1e-12
        or abs(_finite(response["command_step"], "steer.command_step") - 0.05) > 1e-12
        or _finite(response["fixed_delta_seconds"], "steer.fixed_delta_seconds") <= 0.0
        or abs(_finite(response["hysteresis_max_deg"], "steer.hysteresis_max_deg")) > 1e-12
        or response["method"] != "stationary_handbraked_bidirectional_sweep_v1"
        or _integer(response["settle_ticks_per_command"],
                    "steer.settle_ticks_per_command", minimum=1) < 1
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics normalized-steer response identity is invalid"
        )
    parsed_samples = []
    for index, sample in enumerate(samples):
        item = _require_exact_keys(
            sample, sample_keys, f"normalized_steer_samples[{index}]"
        )
        parsed = {key: _finite(item[key], f"normalized_steer_samples[{index}].{key}")
                  for key in sample_keys}
        expected_u = round(-0.8 + 0.05 * index, 2)
        if abs(parsed["normalized_steer"] - expected_u) > 1e-12:
            raise ParkingManoeuvreProofError(
                "vehicle dynamics normalized-steer grid is invalid"
            )
        parsed_samples.append(parsed)
    curvatures = [item["ackermann_center_curvature_1pm"] for item in parsed_samples]
    if (
        any(second <= first for first, second in zip(curvatures, curvatures[1:]))
        or abs(curvatures[16]) > 1e-12
        or any(abs(curvatures[index] + curvatures[-1 - index]) > 1e-12
               for index in range(16))
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics curvature map is not strictly signed/antisymmetric"
        )
    # Exercise both interpolation directions while the exact sample grid is in
    # scope; this also rejects a non-invertible map before runtime consumes it.
    for sample in parsed_samples:
        recovered = normalized_steer_for_curvature(
            sample["ackermann_center_curvature_1pm"], parsed_samples
        )
        if abs(recovered - sample["normalized_steer"]) > 1e-12:
            raise ParkingManoeuvreProofError(
                "vehicle dynamics steer/curvature inverse does not close"
            )

    raw_sweeps = response["raw_sweeps"]
    if (
        not isinstance(raw_sweeps, Mapping)
        or set(raw_sweeps) != {"ascending", "descending"}
        or response["raw_sweeps_sha256"] != _parking_hash(
            raw_sweeps, "vehicle_dynamics.normalized_steer_response.raw_sweeps"
        )
        or response["raw_sweeps_sha256"] != PARKING_RAW_STEER_SWEEPS_SHA256
    ):
        raise ParkingManoeuvreProofError(
            "vehicle dynamics raw steer sweep hash/shape is invalid"
        )
    raw_keys = frozenset({
        "ackermann_center_curvature_1pm",
        "ackermann_outer_angle_residual_deg", "last_five_angle_spread_max_deg",
        "normalized_steer", "settled_wheel_angles_deg",
    })
    angle_keys = frozenset({
        "front_left", "front_right", "rear_left", "rear_right",
    })
    for direction, expected_samples in (
        ("ascending", parsed_samples), ("descending", list(reversed(parsed_samples))),
    ):
        sweep = raw_sweeps[direction]
        if not isinstance(sweep, list) or len(sweep) != 33:
            raise ParkingManoeuvreProofError(
                "vehicle dynamics raw steer sweep must have 33 rows"
            )
        for index, (row, expected) in enumerate(zip(sweep, expected_samples)):
            item = _require_exact_keys(row, raw_keys, f"raw_sweeps.{direction}[{index}]")
            angles = _require_exact_keys(
                item["settled_wheel_angles_deg"], angle_keys,
                f"raw_sweeps.{direction}[{index}].settled_wheel_angles_deg",
            )
            if (
                any(abs(_finite(item[key], f"raw_sweeps.{direction}[{index}].{key}")
                        - expected[key]) > 1e-12 for key in (
                            "normalized_steer", "ackermann_center_curvature_1pm",
                            "ackermann_outer_angle_residual_deg",
                        ))
                or abs(_finite(item["last_five_angle_spread_max_deg"],
                               "raw sweep spread")) > 1e-12
                or abs(_finite(angles["front_left"], "raw front_left")
                       - expected["front_left_deg"]) > 1e-12
                or abs(_finite(angles["front_right"], "raw front_right")
                       - expected["front_right_deg"]) > 1e-12
                or abs(_finite(angles["rear_left"], "raw rear_left")) > 1e-12
                or abs(_finite(angles["rear_right"], "raw rear_right")) > 1e-12
            ):
                raise ParkingManoeuvreProofError(
                    "vehicle dynamics raw steer sweep disagrees with canonical map"
                )

    source_files = asset["source_files"]
    if not isinstance(source_files, Mapping) or len(source_files) < 5:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics source_files must close all raw dependencies"
        )
    saw_libcarla = False
    for raw_path, raw_digest in source_files.items():
        path = _source_path(raw_path, "vehicle_dynamics.source_files key")
        digest = _sha256(raw_digest, f"vehicle_dynamics.source_files[{path}]")
        saw_libcarla = saw_libcarla or path.name.startswith("libcarla.")
        if verify_sources and (not path.is_file() or _sha256_file(path) != digest):
            raise ParkingManoeuvreProofError(
                f"vehicle dynamics nested source SHA256 mismatch: {path}"
            )
    if not saw_libcarla:
        raise ParkingManoeuvreProofError(
            "vehicle dynamics source_files omits the loaded libcarla binary"
        )

    return {
        "asset": dict(asset),
        "canonical_samples": [dict(item) for item in parsed_samples],
        "front_axle_x_m": front_x,
        "rear_axle_x_m": rear_x,
        "wheelbase_m": wheelbase,
        "source_files": dict(source_files),
    }


def _local_bbox(value: Any, field: str) -> Dict[str, Dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != PARKING_LOCAL_BBOX_KEYS:
        raise ParkingManoeuvreProofError(
            f"{field} keys must be exactly {sorted(PARKING_LOCAL_BBOX_KEYS)}"
        )
    output: Dict[str, Dict[str, float]] = {}
    for name, keys in (
        ("location", PARKING_LOCATION_KEYS),
        ("rotation", PARKING_ROTATION_KEYS),
        ("extent", PARKING_LOCATION_KEYS),
    ):
        item = value[name]
        if not isinstance(item, Mapping) or set(item) != keys:
            raise ParkingManoeuvreProofError(
                f"{field}.{name} keys must be exactly {sorted(keys)}"
            )
        output[name] = {
            key: _finite(item[key], f"{field}.{name}.{key}") for key in keys
        }
    if any(output["extent"][axis] <= 0.0 for axis in ("x", "y", "z")):
        raise ParkingManoeuvreProofError(
            f"{field}.extent values must be positive"
        )
    return output


def _vertical_half_extent(extent: Mapping[str, Any], rotation: Mapping[str, Any]) -> float:
    pitch = math.radians(_finite(rotation["pitch"], "bbox.rotation.pitch"))
    roll = math.radians(_finite(rotation["roll"], "bbox.rotation.roll"))
    ex = _finite(extent["x"], "bbox.extent.x")
    ey = _finite(extent["y"], "bbox.extent.y")
    ez = _finite(extent["z"], "bbox.extent.z")
    if min(ex, ey, ez) < 0.0:
        raise ParkingManoeuvreProofError("bbox extents cannot be negative")
    return (
        abs(-math.sin(pitch)) * ex
        + abs(math.cos(pitch) * math.sin(roll)) * ey
        + abs(math.cos(pitch) * math.cos(roll)) * ez
    )


def _vertical_overlap_index_sets(
    poses: Sequence[Mapping[str, Any]],
    bboxes: Sequence[Mapping[str, Any]],
    local_bbox: Mapping[str, Mapping[str, Any]],
    *,
    actor_reference_z_min_m: float,
    actor_reference_z_max_m: float,
) -> list[list[int]]:
    ego_half = _vertical_half_extent(
        local_bbox["extent"], local_bbox["rotation"]
    )
    static_intervals = []
    for index, bbox in enumerate(bboxes):
        if not isinstance(bbox, Mapping) or set(bbox) != {
            "category", "center", "extent", "rotation"
        }:
            raise ParkingManoeuvreProofError(
                f"parking bbox asset entry {index} has invalid keys"
            )
        if bbox["category"] not in PARKING_STATIC_BBOX_CATEGORIES:
            raise ParkingManoeuvreProofError(
                f"parking bbox asset entry {index} has unsupported category"
            )
        center = bbox["center"]
        extent = bbox["extent"]
        rotation = bbox["rotation"]
        if (
            not isinstance(center, Mapping)
            or set(center) != PARKING_LOCATION_KEYS
            or not isinstance(extent, Mapping)
            or set(extent) != PARKING_LOCATION_KEYS
            or not isinstance(rotation, Mapping)
            or set(rotation) != PARKING_ROTATION_KEYS
        ):
            raise ParkingManoeuvreProofError(
                f"parking bbox asset entry {index} has invalid transform"
            )
        half = _vertical_half_extent(extent, rotation)
        center_z = _finite(center["z"], f"bbox[{index}].center.z")
        static_intervals.append((center_z - half, center_z + half))

    result: list[list[int]] = []
    local_z = _finite(
        local_bbox["location"]["z"], "vehicle_geometry.local_bbox.location.z"
    )
    for pose in poses:
        pose_z = _finite(pose["z"], "reverse_manoeuvre.pose.z")
        # The authoritative trace records a small actor-reference settling
        # range around the path pose.  Use its conservative union, not merely
        # the nominal pose, so an obstacle cannot be discarded because the
        # spawned Lincoln settles a few centimetres vertically.
        ego_min = (
            pose_z + actor_reference_z_min_m + local_z - ego_half
        )
        ego_max = (
            pose_z + actor_reference_z_max_m + local_z + ego_half
        )
        overlaps = [
            index
            for index, (static_min, static_max) in enumerate(static_intervals)
            if (
                ego_max + PARKING_VERTICAL_PADDING_M
                >= static_min - PARKING_VERTICAL_PADDING_M - 1e-9
                and static_max + PARKING_VERTICAL_PADDING_M
                >= ego_min - PARKING_VERTICAL_PADDING_M - 1e-9
            )
        ]
        result.append(overlaps)
    return result


def _static_horizontal_clearance(
    poses: Sequence[Mapping[str, Any]],
    bboxes: Sequence[Mapping[str, Any]],
    overlap_index_sets: Sequence[Sequence[int]],
    *,
    local_bbox: Mapping[str, Mapping[str, Any]],
) -> float:
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise ParkingManoeuvreProofError(
            "parking v4 source verification requires Shapely"
        ) from exc

    def rotation_matrix(rotation: Mapping[str, Any]) -> list[list[float]]:
        pitch = math.radians(float(rotation["pitch"]))
        yaw = math.radians(float(rotation["yaw"]))
        roll = math.radians(float(rotation["roll"]))
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        # CARLA/Unreal Euler transform frozen for the proof: Rz(yaw) @
        # Ry(pitch) @ Rx(roll).
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
        return tuple(
            sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
            for row in range(3)
        )  # type: ignore[return-value]

    def projected_obb(
        center: Mapping[str, Any], extent: Mapping[str, Any],
        rotation: Mapping[str, Any],
    ) -> Any:
        matrix = rotation_matrix(rotation)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = matvec(matrix, (
                        sx * float(extent["x"]),
                        sy * float(extent["y"]),
                        sz * float(extent["z"]),
                    ))
                    corners.append((
                        float(center["x"]) + offset[0],
                        float(center["y"]) + offset[1],
                    ))
        return Polygon(corners).convex_hull

    local_rotation = rotation_matrix(local_bbox["rotation"])
    local_location = (
        float(local_bbox["location"]["x"]),
        float(local_bbox["location"]["y"]),
        float(local_bbox["location"]["z"]),
    )

    def ego_polygon(pose: Mapping[str, Any]) -> Any:
        yaw = math.radians(float(pose["yaw_deg"]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local_corner = matvec(local_rotation, (
                        sx * float(local_bbox["extent"]["x"]),
                        sy * float(local_bbox["extent"]["y"]),
                        sz * float(local_bbox["extent"]["z"]),
                    ))
                    local_x = local_location[0] + local_corner[0]
                    local_y = local_location[1] + local_corner[1]
                    corners.append((
                        float(pose["x"]) + cosine * local_x - sine * local_y,
                        float(pose["y"]) + sine * local_x + cosine * local_y,
                    ))
        return Polygon(corners).convex_hull

    minimum = math.inf
    bbox_polygons: Dict[int, Any] = {}
    for pose, indices in zip(poses, overlap_index_sets):
        ego = ego_polygon(pose)
        for index in indices:
            if index not in bbox_polygons:
                bbox = bboxes[index]
                bbox_polygons[index] = projected_obb(
                    bbox["center"], bbox["extent"], bbox["rotation"],
                )
            minimum = min(minimum, float(ego.distance(bbox_polygons[index])))
    return minimum


def validate_parking_manoeuvre_registry(
    registry: Any,
    town: str,
    *,
    verify_sources: bool = True,
) -> Dict[str, Any]:
    """Validate and materialize the frozen v4/v5 axle/dynamics registry."""

    try:
        registry_hash = _parking_hash(registry, "parking_manoeuvre_registry")
    except ParkingManoeuvreProofError as exc:
        raise ParkingManoeuvreProofError(
            "parking_manoeuvre_registry is not canonical JSON"
        ) from exc
    cache_key = (registry_hash, str(town), bool(verify_sources))
    cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not isinstance(registry, Mapping) or set(registry) != PARKING_REGISTRY_KEYS:
        raise ParkingManoeuvreProofError(
            "parking_manoeuvre_registry keys must be exactly {}".format(
                sorted(PARKING_REGISTRY_KEYS)
            )
        )
    registry_contracts = {
        PARKING_REGISTRY_SCHEMA: PARKING_MANOEUVRE_PROOF_METHOD,
        PARKING_REGISTRY_SCHEMA_V5: PARKING_MANOEUVRE_PROOF_METHOD_V5,
    }
    if registry_contracts.get(registry["schema_version"]) != registry["method"]:
        raise ParkingManoeuvreProofError(
            "parking_manoeuvre_registry schema/method is unsupported"
        )
    is_registry_v5 = registry["schema_version"] == PARKING_REGISTRY_SCHEMA_V5
    towns = registry["towns"]
    if not isinstance(towns, Mapping) or not towns:
        raise ParkingManoeuvreProofError(
            "parking_manoeuvre_registry.towns must be a non-empty object"
        )
    entry = towns.get(str(town))
    expected_entry_keys = (
        PARKING_TOWN_REGISTRY_KEYS_V5 if is_registry_v5 else PARKING_TOWN_REGISTRY_KEYS
    )
    if not isinstance(entry, Mapping) or set(entry) != expected_entry_keys:
        raise ParkingManoeuvreProofError(
            f"parking registry has no exact town entry for {town}"
        )
    if str(entry["town"]) != str(town):
        raise ParkingManoeuvreProofError("parking registry town key/value mismatch")

    occupancy = entry.get("slot_collision_occupancy")
    if is_registry_v5:
        if not isinstance(occupancy, Mapping) or set(occupancy) != PARKING_OCCUPANCY_KEYS:
            raise ParkingManoeuvreProofError("parking v5 slot occupancy registry is not exact")
        occupancy_path = _source_path(
            occupancy["source_ref"], "slot_collision_occupancy.source_ref"
        )
        if (
            not occupancy_path.is_file()
            or _sha256_file(occupancy_path) != occupancy["source_file_sha256"]
            or occupancy["schema_version"] != "language2drive.parking_slot_collision_occupancy.v1"
            or occupancy["method"] != "carla_try_spawn_release_ego_at_xodr_slot_terminal_v1"
            or occupancy["town"] != str(town)
            or occupancy["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT
            or occupancy["spawn_z_m"] != 0.5
            or occupancy["repeat_count"] != 2
            or not isinstance(occupancy["scope_statement"], str)
        ):
            raise ParkingManoeuvreProofError("parking v5 slot occupancy authority is stale")
        try:
            occupancy_payload = json.loads(occupancy_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ParkingManoeuvreProofError("parking v5 slot occupancy is unreadable") from exc
        rows = occupancy_payload.get("slots") if isinstance(occupancy_payload, Mapping) else None
        clear_slots = sorted(
            row.get("slot_number") for row in rows
            if isinstance(row, Mapping) and row.get("terminal_spawn_clear") is True
        ) if isinstance(rows, list) else []
        blocked_slots = sorted(
            row.get("slot_number") for row in rows
            if isinstance(row, Mapping) and row.get("terminal_spawn_clear") is False
        ) if isinstance(rows, list) else []
        if (
            not isinstance(rows, list)
            or occupancy_payload.get("slots_sha256") != occupancy["slots_sha256"]
            or _parking_hash(rows, "slot occupancy rows") != occupancy["slots_sha256"]
            or occupancy["clear_slots"] != clear_slots
            or occupancy["blocked_slots"] != blocked_slots
            or occupancy["clear_slots_sha256"] != _parking_hash(clear_slots, "clear slots")
            or occupancy["blocked_slots_sha256"] != _parking_hash(blocked_slots, "blocked slots")
            or occupancy_payload.get("summary") != {
                "slot_count": occupancy["slot_count"],
                "clear_count": occupancy["clear_count"],
                "blocked_count": occupancy["blocked_count"],
                "clear_slots_sha256": occupancy["clear_slots_sha256"],
                "blocked_slots_sha256": occupancy["blocked_slots_sha256"],
            }
        ):
            raise ParkingManoeuvreProofError("parking v5 slot occupancy payload closure differs")
        nested_occupancy = occupancy["source_files"]
        if not isinstance(nested_occupancy, Mapping) or set(nested_occupancy) != {
            "slot_registry", "runtime_manifest", "runtime_provenance",
            "available_maps_witness", "capture_tool",
        }:
            raise ParkingManoeuvreProofError("parking v5 occupancy nested sources are not exact")
        for item in nested_occupancy.values():
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
                raise ParkingManoeuvreProofError("parking v5 occupancy nested source shape differs")

    bbox_registry = entry["bbox_asset"]
    if not isinstance(bbox_registry, Mapping) or set(bbox_registry) != PARKING_BBOX_REGISTRY_KEYS:
        raise ParkingManoeuvreProofError(
            "parking bbox registry keys are invalid"
        )
    bbox_count = _integer(
        bbox_registry["bbox_count"], "bbox_asset.bbox_count", minimum=1
    )
    if (
        bbox_registry["schema_version"] != "language2drive.carla_level_bboxes.v1"
        or bbox_registry["categories"] != PARKING_STATIC_BBOX_CATEGORIES
        or bbox_registry["scope_statement"] != PARKING_BBOX_SCOPE_STATEMENT
        or bbox_registry["excluded_scope"] != PARKING_BBOX_EXCLUDED_SCOPE
        or bbox_registry["quantization_decimals"] != 6
        or bbox_registry["canonical_sort_key"] != PARKING_BBOX_CANONICAL_SORT_KEY
        or bbox_registry["source_api"] != "carla.World.get_level_bbs(CityObjectLabel)"
    ):
        raise ParkingManoeuvreProofError(
            "parking bbox registry semantics are not the frozen v4 scope"
        )
    bbox_sha = _sha256(
        bbox_registry["bboxes_sha256"], "bbox_asset.bboxes_sha256"
    )
    _sha256(
        bbox_registry["source_file_sha256"],
        "bbox_asset.source_file_sha256",
    )

    for name in ("xodr", "lane_graph"):
        value = entry[name]
        if not isinstance(value, Mapping) or set(value) != PARKING_SOURCE_HASH_KEYS:
            raise ParkingManoeuvreProofError(
                f"parking registry {name} keys are invalid"
            )
        _source_path(value["source_ref"], f"{name}.source_ref")
        _sha256(value["sha256"], f"{name}.sha256")

    control = entry["control_model"]
    if not isinstance(control, Mapping) or set(control) != PARKING_CONTROL_REGISTRY_KEYS:
        raise ParkingManoeuvreProofError("parking control_model keys are invalid")
    registry_number = _strict_finite if is_registry_v5 else _finite
    control_values = {
        field: registry_number(control[field], f"control_model.{field}")
        for field in (
            "front_axle_x_m", "rear_axle_x_m", "wheelbase_m",
            "max_normalized_control_steer",
        )
    }
    if (
        control["schema_version"] != PARKING_CONTROL_SCHEMA
        or control["pose_reference"] != PARKING_POSE_REFERENCE
        or control["integration_reference"] != PARKING_INTEGRATION_REFERENCE
        or control["vehicle_dynamics_source_ref"]
        != PARKING_DYNAMICS_SOURCE_REF
        or control["vehicle_dynamics_source_file_sha256"]
        != PARKING_DYNAMICS_SOURCE_FILE_SHA256
        or control["wheel_geometry_sha256"] != PARKING_WHEEL_GEOMETRY_SHA256
        or control["normalized_steer_samples_sha256"]
        != PARKING_NORMALIZED_STEER_SAMPLES_SHA256
        or abs(control_values["front_axle_x_m"] - PARKING_FRONT_AXLE_X_M) > 1e-9
        or abs(control_values["rear_axle_x_m"] - PARKING_REAR_AXLE_X_M) > 1e-9
        or abs(control_values["wheelbase_m"] - PARKING_WHEELBASE_M) > 1e-9
        or abs(
            control_values["front_axle_x_m"]
            - control_values["rear_axle_x_m"]
            - control_values["wheelbase_m"]
        ) > 1e-9
        or abs(
            control_values["max_normalized_control_steer"]
            - PARKING_MAX_NORMALIZED_CONTROL_STEER
        ) > 1e-12
    ):
        raise ParkingManoeuvreProofError(
            "parking control_model is not the frozen v4 axle/map contract"
        )
    _sha256(
        control["vehicle_dynamics_source_file_sha256"],
        "control_model.vehicle_dynamics_source_file_sha256",
    )

    vehicle = entry["vehicle_geometry"]
    if not isinstance(vehicle, Mapping) or set(vehicle) != PARKING_VEHICLE_GEOMETRY_KEYS:
        raise ParkingManoeuvreProofError(
            "parking vehicle_geometry keys are invalid"
        )
    local_bbox = _local_bbox(vehicle["local_bbox"], "vehicle_geometry.local_bbox")
    if is_registry_v5:
        for group, values in vehicle["local_bbox"].items():
            for field, value in values.items():
                _strict_finite(
                    value, f"vehicle_geometry.local_bbox.{group}.{field}"
                )
    if (
        vehicle["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT
        or vehicle["coordinate_frame"] != "carla_actor_local"
        or vehicle["source_method"] != "constant_bbox_across_authoritative_trace_v1"
        or vehicle["local_bbox_sha256"] != _parking_hash(
            vehicle["local_bbox"], "vehicle_geometry.local_bbox"
        )
    ):
        raise ParkingManoeuvreProofError(
            "parking vehicle geometry identity/hash is invalid"
        )
    _sha256(vehicle["local_bbox_sha256"], "vehicle_geometry.local_bbox_sha256")
    _sha256(vehicle["source_trace_sha256"], "vehicle_geometry.source_trace_sha256")
    trace_rows = _integer(
        vehicle["source_trace_rows"], "vehicle_geometry.source_trace_rows", minimum=1
    )
    matching_rows = _integer(
        vehicle["matching_bbox_rows"], "vehicle_geometry.matching_bbox_rows", minimum=1
    )
    if matching_rows != trace_rows:
        raise ParkingManoeuvreProofError(
            "parking vehicle geometry must match every source trace row"
        )
    z_min = registry_number(
        vehicle["actor_reference_z_min_m"],
        "vehicle_geometry.actor_reference_z_min_m",
    )
    z_max = registry_number(
        vehicle["actor_reference_z_max_m"],
        "vehicle_geometry.actor_reference_z_max_m",
    )
    if z_min > z_max:
        raise ParkingManoeuvreProofError(
            "parking vehicle geometry actor reference z range is inverted"
        )

    bbox_payload = None
    dynamics_context = None
    xodr_bytes = None
    lane_graph_payload = None
    if verify_sources:
        raw_bbox = _verified_source_bytes(
            bbox_registry["source_ref"], bbox_registry["source_file_sha256"],
            "bbox_asset",
        )
        try:
            bbox_payload = json.loads(raw_bbox.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ParkingManoeuvreProofError(
                "parking bbox asset is not valid JSON"
            ) from exc
        if (
            not isinstance(bbox_payload, Mapping)
            or bbox_payload.get("schema_version") != bbox_registry["schema_version"]
            or bbox_payload.get("town") != str(town)
            or bbox_payload.get("categories") != bbox_registry["categories"]
            or bbox_payload.get("excluded_scope") != bbox_registry["excluded_scope"]
            or bbox_payload.get("scope_statement") != bbox_registry["scope_statement"]
            or bbox_payload.get("canonical_sort_key") != bbox_registry["canonical_sort_key"]
            or bbox_payload.get("quantization_decimals") != 6
            or bbox_payload.get("bbox_count") != bbox_count
            or bbox_payload.get("bboxes_sha256") != bbox_sha
            or not isinstance(bbox_payload.get("bboxes"), list)
            or len(bbox_payload["bboxes"]) != bbox_count
            or _parking_hash(bbox_payload["bboxes"], "bbox_asset.bboxes") != bbox_sha
            or bbox_payload.get("provenance", {}).get("source_api")
            != bbox_registry["source_api"]
        ):
            raise ParkingManoeuvreProofError(
                "parking bbox source does not close its registry semantics"
            )
        for name in ("xodr", "lane_graph"):
            source = entry[name]
            path = _source_path(source["source_ref"], f"{name}.source_ref")
            if not path.is_file() or _sha256_file(path) != source["sha256"]:
                raise ParkingManoeuvreProofError(
                    f"parking registry {name} source SHA256 mismatch"
                )
        try:
            xodr_bytes = _source_path(
                entry["xodr"]["source_ref"], "xodr.source_ref"
            ).read_bytes()
            lane_graph_payload = json.loads(_source_path(
                entry["lane_graph"]["source_ref"], "lane_graph.source_ref"
            ).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ParkingManoeuvreProofError(
                "parking XODR/lane graph source cannot be materialized"
            ) from exc
        if not isinstance(lane_graph_payload, Mapping):
            raise ParkingManoeuvreProofError(
                "parking lane graph source must be a JSON object"
            )
        raw_dynamics = _verified_source_bytes(
            control["vehicle_dynamics_source_ref"],
            control["vehicle_dynamics_source_file_sha256"],
            "control_model.vehicle_dynamics",
        )
        try:
            dynamics_payload = json.loads(raw_dynamics.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ParkingManoeuvreProofError(
                "vehicle dynamics asset is not valid JSON"
            ) from exc
        dynamics_context = _validate_vehicle_dynamics_asset(
            dynamics_payload, verify_sources=True
        )
        if (
            dynamics_payload["capture"]["town"] != str(town)
            or
            control["wheel_geometry_sha256"]
            != dynamics_payload["wheel_geometry_sha256"]
            or control["normalized_steer_samples_sha256"]
            != dynamics_payload["normalized_steer_response"][
                "canonical_samples_sha256"
            ]
            or any(
                abs(control_values[field] - dynamics_context[field]) > 1e-9
                for field in (
                    "front_axle_x_m", "rear_axle_x_m", "wheelbase_m",
                )
            )
        ):
            raise ParkingManoeuvreProofError(
                "parking control_model disagrees with vehicle dynamics asset"
            )
        raw_trace = _verified_source_bytes(
            vehicle["source_trace_ref"], vehicle["source_trace_sha256"],
            "vehicle_geometry.source_trace",
        )
        rows = raw_trace.splitlines()
        if len(rows) != trace_rows:
            raise ParkingManoeuvreProofError(
                "parking vehicle source trace row count mismatch"
            )
        observed_z = []
        observed_bbox = []
        try:
            for raw_row in rows:
                row = json.loads(raw_row.decode("utf-8"))
                ego = row.get("ego", row)
                observed_bbox.append(ego.get("bounding_box", ego.get("bbox")))
                transform = ego.get("transform", {})
                location = transform.get("location", transform)
                observed_z.append(float(location["z"]))
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ParkingManoeuvreProofError(
                "parking vehicle source trace has invalid ego geometry"
            ) from exc
        if (
            sum(value == vehicle["local_bbox"] for value in observed_bbox)
            != matching_rows
            or abs(min(observed_z) - z_min) > 1e-6
            or abs(max(observed_z) - z_max) > 1e-6
        ):
            raise ParkingManoeuvreProofError(
                "parking vehicle source trace disagrees with frozen geometry"
            )

    result = {
        "entry": dict(entry),
        "bbox_asset": dict(bbox_registry),
        "bbox_payload": bbox_payload,
        "local_bbox": local_bbox,
        "dynamics": dynamics_context,
        "registry_schema_version": registry["schema_version"],
        "registry_method": registry["method"],
        "xodr_bytes": xodr_bytes,
        "lane_graph_payload": lane_graph_payload,
    }
    _REGISTRY_CACHE[cache_key] = result
    return result


def validate_parking_registry_source_files(
    registry: Any,
    source_files: Any,
    town: str,
) -> Dict[str, Any]:
    """Bind the sidecar provenance map to every direct parking dependency.

    ``source_files`` is not a substitute for the semantic registry: it proves
    only file identity. Conversely, the registry must not name dependencies
    that the release provenance omits. Requiring both closes that gap while
    still allowing unrelated suite source files in the top-level map.
    """

    context = validate_parking_manoeuvre_registry(
        registry, town, verify_sources=True
    )
    if not isinstance(source_files, Mapping):
        raise ParkingManoeuvreProofError(
            "parking v4/v5 sidecar requires a source_files object"
        )
    entry = context["entry"]
    required = {
        str(_source_path(
            entry["bbox_asset"]["source_ref"], "bbox_asset.source_ref"
        )): str(entry["bbox_asset"]["source_file_sha256"]),
        str(_source_path(
            entry["xodr"]["source_ref"], "xodr.source_ref"
        )): str(entry["xodr"]["sha256"]),
        str(_source_path(
            entry["lane_graph"]["source_ref"], "lane_graph.source_ref"
        )): str(entry["lane_graph"]["sha256"]),
        str(_source_path(
            entry["control_model"]["vehicle_dynamics_source_ref"],
            "control_model.vehicle_dynamics_source_ref",
        )): str(entry["control_model"]["vehicle_dynamics_source_file_sha256"]),
        str(_source_path(
            entry["vehicle_geometry"]["source_trace_ref"],
            "vehicle_geometry.source_trace_ref",
        )): str(entry["vehicle_geometry"]["source_trace_sha256"]),
    }
    occupancy = entry.get("slot_collision_occupancy")
    if isinstance(occupancy, Mapping):
        required[str(_source_path(
            occupancy["source_ref"], "slot_collision_occupancy.source_ref"
        ))] = str(occupancy["source_file_sha256"])
        for name, item in occupancy["source_files"].items():
            required[str(_source_path(
                item["path"], f"slot_collision_occupancy.source_files.{name}.path"
            ))] = str(item["sha256"])
    dynamics = context.get("dynamics")
    if not isinstance(dynamics, Mapping):
        raise ParkingManoeuvreProofError(
            "parking v4/v5 registry did not materialize vehicle dynamics"
        )
    for raw_path, raw_digest in dynamics["source_files"].items():
        required[str(_source_path(
            raw_path, "vehicle_dynamics.source_files key"
        ))] = str(raw_digest)
    for raw_path, raw_digest in source_files.items():
        path = _source_path(raw_path, "source_files key")
        _sha256(raw_digest, f"source_files[{path}]")
    missing_or_stale = sorted(
        path for path, digest in required.items()
        if source_files.get(path) != digest
    )
    if missing_or_stale:
        raise ParkingManoeuvreProofError(
            "parking source_files missing/stale registry dependencies: {}"
            .format(", ".join(missing_or_stale))
        )
    return context


def _parking_v5_map_context(
    registry_context: Mapping[str, Any], town: str,
) -> Dict[str, Any]:
    """Build the offline CARLA map and frozen directed-lane index for v5."""

    entry = registry_context["entry"]
    key = (str(entry["xodr"]["sha256"]), str(entry["lane_graph"]["sha256"]))
    cached = _V5_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    raw_xodr = registry_context.get("xodr_bytes")
    graph = registry_context.get("lane_graph_payload")
    if not isinstance(raw_xodr, bytes) or not isinstance(graph, Mapping):
        raise ParkingManoeuvreProofError(
            "parking v5 requires verified XODR and lane-graph payloads"
        )
    try:
        import carla  # type: ignore

        carla_map = carla.Map(str(town), raw_xodr.decode("utf-8"))
    except (ImportError, OSError, UnicodeError, RuntimeError, TypeError) as exc:
        raise ParkingManoeuvreProofError(
            "parking v5 cannot construct the authoritative offline CARLA map"
        ) from exc
    raw_lanes = graph.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise ParkingManoeuvreProofError(
            "parking v5 lane graph must contain lanes"
        )
    lanes: Dict[str, Mapping[str, Any]] = {}
    by_key: Dict[tuple[int, int, int], str] = {}
    directions: Dict[str, int] = {}
    for index, lane in enumerate(raw_lanes):
        if not isinstance(lane, Mapping) or not isinstance(lane.get("id"), str):
            raise ParkingManoeuvreProofError(
                f"parking lane_graph.lanes[{index}] is invalid"
            )
        lane_id = lane["id"]
        if lane_id in lanes:
            raise ParkingManoeuvreProofError("parking lane graph has duplicate lane id")
        integer_key = tuple(
            _integer(
                lane[field], f"lane_graph.lanes[{index}].{field}",
                minimum=-(2 ** 31),
            )
            for field in ("road_id", "section_id", "lane_id")
        )
        if integer_key[2] == 0 or integer_key in by_key:
            raise ParkingManoeuvreProofError(
                "parking lane graph has invalid/duplicate directed lane tuple"
            )
        samples = lane.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            raise ParkingManoeuvreProofError(
                f"parking lane graph {lane_id} has no directed samples"
            )
        first_s = _strict_finite(samples[0].get("s"), f"lane_graph.{lane_id}.first_s")
        last_s = _strict_finite(samples[-1].get("s"), f"lane_graph.{lane_id}.last_s")
        if abs(last_s - first_s) <= 1e-8:
            raise ParkingManoeuvreProofError(
                f"parking lane graph {lane_id} has no stable s direction"
            )
        successors = lane.get("successors")
        if not isinstance(successors, list) or any(
            not isinstance(item, str) for item in successors
        ):
            raise ParkingManoeuvreProofError(
                f"parking lane graph {lane_id} successors are invalid"
            )
        lanes[lane_id] = lane
        by_key[integer_key] = lane_id
        directions[lane_id] = 1 if last_s > first_s else -1
    if any(
        successor not in lanes
        for lane in lanes.values()
        for successor in lane.get("successors", [])
    ):
        raise ParkingManoeuvreProofError(
            "parking lane graph names an unknown successor"
        )
    result = {
        "carla": carla,
        "map": carla_map,
        "lanes": lanes,
        "by_key": by_key,
        "directions": directions,
    }
    _V5_MAP_CACHE[key] = result
    return result


def _canonical_waypoint_transform(waypoint: Any, pose_index: int) -> Dict[str, Any]:
    return {
        "pose_index": pose_index,
        "x": round(float(waypoint.transform.location.x), 4),
        "y": round(float(waypoint.transform.location.y), 4),
        "z": round(float(waypoint.transform.location.z), 4),
        "yaw_deg": _canonical_yaw(waypoint.transform.rotation.yaw),
    }


def _distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return math.dist(
        tuple(float(first.get(axis, 0.0)) for axis in ("x", "y", "z")),
        tuple(float(second.get(axis, 0.0)) for axis in ("x", "y", "z")),
    )


def _path_length(path: Sequence[Mapping[str, Any]]) -> float:
    return sum(_distance(first, second) for first, second in zip(path, path[1:]))


def _chord_yaw(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return math.degrees(math.atan2(
        float(second["y"]) - float(first["y"]),
        float(second["x"]) - float(first["x"]),
    ))


def _nonzero_chord(
    path: Sequence[Mapping[str, Any]], *, reverse: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    pairs = list(zip(path, path[1:]))
    if reverse:
        pairs.reverse()
    for first, second in pairs:
        if _distance(first, second) > 1e-6:
            return first, second
    raise ParkingManoeuvreProofError("parking path has no nonzero chord")


def _exact_waypoint(carla_map: Any, carla: Any, pose: Mapping[str, Any]) -> Any:
    location = carla.Location(
        x=float(pose["x"]), y=float(pose["y"]),
        z=float(pose.get("z", 0.0)),
    )
    for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
        waypoint = carla_map.get_waypoint(
            location, project_to_road=False, lane_type=lane_type,
        )
        if waypoint is not None:
            return waypoint
    return None


def _validate_v5_topology(
    poses: Sequence[Mapping[str, Any]], proof: Any, *,
    expected_space: str, expected_first: int, expected_last: int,
    map_context: Mapping[str, Any], require_raw_transform: bool,
) -> Dict[str, Any]:
    """Replay the minimal stored-XODR proof; inverse lookup is not authority."""

    proof = _require_exact_keys(
        proof, PARKING_V5_TOPOLOGY_KEYS, "parking_v5 topology proof"
    )
    if (
        proof["schema_version"] != PARKING_V5_TOPOLOGY_SCHEMA
        or proof["method"] != PARKING_V5_TOPOLOGY_METHOD
        or proof["pose_index_space"] != expected_space
    ):
        raise ParkingManoeuvreProofError(
            "parking_v5 topology schema/method/index-space mismatch"
        )
    first = _integer(proof["first_pose_index"], "topology.first_pose_index")
    last = _integer(proof["last_pose_index"], "topology.last_pose_index")
    if (
        (first, last) != (expected_first, expected_last)
        or not first <= last < len(poses)
    ):
        raise ParkingManoeuvreProofError("parking_v5 topology range mismatch")
    samples = proof["samples"]
    if not isinstance(samples, list) or len(samples) != last - first + 1:
        raise ParkingManoeuvreProofError("parking_v5 topology sample count mismatch")
    if _parking_hash(samples, "topology.samples") != _sha256(
        proof["samples_sha256"], "topology.samples_sha256"
    ):
        raise ParkingManoeuvreProofError("parking_v5 topology samples hash mismatch")

    carla = map_context["carla"]
    carla_map = map_context["map"]
    lanes = map_context["lanes"]
    by_key = map_context["by_key"]
    directions = map_context["directions"]
    transforms = []
    replay = []
    for expected_index, sample in enumerate(samples, first):
        sample = _require_exact_keys(
            sample, PARKING_V5_TOPOLOGY_SAMPLE_KEYS,
            f"topology.samples[{expected_index}]",
        )
        pose_index = _integer(
            sample["pose_index"], f"topology.samples[{expected_index}].pose_index"
        )
        if pose_index != expected_index:
            raise ParkingManoeuvreProofError("parking_v5 topology sample index mismatch")
        road_id = _integer(
            sample["road_id"], f"topology.samples[{expected_index}].road_id",
            minimum=-(2 ** 31),
        )
        section_id = _integer(
            sample["section_id"], f"topology.samples[{expected_index}].section_id",
            minimum=-(2 ** 31),
        )
        lane_id = _integer(
            sample["lane_id"], f"topology.samples[{expected_index}].lane_id",
            minimum=-(2 ** 31),
        )
        if lane_id == 0:
            raise ParkingManoeuvreProofError("parking_v5 topology lane_id cannot be zero")
        s_m = _strict_finite(
            sample["s_m"], f"topology.samples[{expected_index}].s_m"
        )
        waypoint = carla_map.get_waypoint_xodr(road_id, lane_id, s_m)
        if waypoint is None or int(waypoint.section_id) != section_id:
            raise ParkingManoeuvreProofError(
                f"parking_v5 stored-XODR reconstruction failed at {expected_index}"
            )
        if waypoint.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
            raise ParkingManoeuvreProofError(
                f"parking_v5 stored-XODR sample is not a legal lane at {expected_index}"
            )
        key = (road_id, section_id, lane_id)
        lane_graph_id = by_key.get(key)
        if lane_graph_id is None:
            raise ParkingManoeuvreProofError(
                f"parking_v5 topology lane is absent from graph: {key}"
            )
        transform = _canonical_waypoint_transform(waypoint, pose_index)
        transforms.append(transform)
        pose = poses[pose_index]
        public_pose = {
            field: pose[field] for field in ("x", "y", "z", "yaw_deg")
        }
        canonical_pose = {
            field: transform[field] for field in ("x", "y", "z", "yaw_deg")
        }
        if require_raw_transform and public_pose != canonical_pose:
            raise ParkingManoeuvreProofError(
                f"parking_v5 ordinary tail is not raw XODR transform at {pose_index}"
            )
        heading = math.radians(float(transform["yaw_deg"]))
        dx = float(pose["x"]) - float(transform["x"])
        dy = float(pose["y"]) - float(transform["y"])
        along = dx * math.cos(heading) + dy * math.sin(heading)
        z_residual = float(pose["z"]) - float(transform["z"])
        if not require_raw_transform and (
            abs(along) > PARKING_V5_TARGET_ALONG_TRACK_RESIDUAL_MAX_M + PARKING_V5_EPS
            or abs(z_residual) > PARKING_V5_TARGET_Z_RESIDUAL_MAX_M + PARKING_V5_EPS
        ):
            raise ParkingManoeuvreProofError(
                f"parking_v5 target topology sample is unbound at {pose_index}"
            )
        replay.append({
            "pose_index": pose_index, "lane_graph_id": lane_graph_id,
            "s_m": s_m, "along": along, "z_residual": z_residual,
        })
    if _parking_hash(transforms, "topology.canonical_waypoint_transforms") != _sha256(
        proof["canonical_waypoint_transforms_sha256"],
        "topology.canonical_waypoint_transforms_sha256",
    ):
        raise ParkingManoeuvreProofError(
            "parking_v5 canonical waypoint transform hash mismatch"
        )
    if expected_space == PARKING_V5_TARGET_POSE_INDEX_SPACE:
        expected_prelead = {
            key: transforms[0][key] for key in ("x", "y", "z", "yaw_deg")
        }
        actual_prelead = {
            key: poses[0][key] for key in ("x", "y", "z", "yaw_deg")
        }
        if actual_prelead != expected_prelead:
            raise ParkingManoeuvreProofError(
                "parking_v5 poses[0] is not the raw canonical prelead"
            )

    expected_runs = []
    for item in replay:
        lane_graph_id = item["lane_graph_id"]
        if not expected_runs or expected_runs[-1]["lane_graph_id"] != lane_graph_id:
            expected_runs.append({
                "first_pose_index": item["pose_index"],
                "last_pose_index": item["pose_index"],
                "lane_graph_id": lane_graph_id,
                "travel_s_direction": directions[lane_graph_id],
            })
        else:
            expected_runs[-1]["last_pose_index"] = item["pose_index"]
    runs = proof["runs"]
    if not isinstance(runs, list) or len(runs) != len(expected_runs):
        raise ParkingManoeuvreProofError("parking_v5 topology run count mismatch")
    for index, (claimed, expected) in enumerate(zip(runs, expected_runs)):
        claimed = _require_exact_keys(
            claimed, PARKING_V5_TOPOLOGY_RUN_KEYS, f"topology.runs[{index}]"
        )
        for field in ("first_pose_index", "last_pose_index"):
            _integer(claimed[field], f"topology.runs[{index}].{field}")
        direction = claimed["travel_s_direction"]
        if (
            isinstance(direction, bool) or not isinstance(direction, int)
            or direction not in (-1, 1) or dict(claimed) != expected
        ):
            raise ParkingManoeuvreProofError(
                f"parking_v5 topology run[{index}] mismatch"
            )
        run_samples = replay[
            expected["first_pose_index"] - first:
            expected["last_pose_index"] - first + 1
        ]
        for before, after in zip(run_samples, run_samples[1:]):
            if (after["s_m"] - before["s_m"]) * direction <= 1e-8:
                raise ParkingManoeuvreProofError(
                    f"parking_v5 topology s is not strict-forward in {expected['lane_graph_id']}"
                )
    transitions = proof["transitions"]
    expected_transitions = [
        {"from_run_index": index, "to_run_index": index + 1}
        for index in range(max(0, len(expected_runs) - 1))
    ]
    if not isinstance(transitions, list) or len(transitions) != len(expected_transitions):
        raise ParkingManoeuvreProofError("parking_v5 topology transition count mismatch")
    for index, (claimed, expected) in enumerate(zip(transitions, expected_transitions)):
        claimed = _require_exact_keys(
            claimed, PARKING_V5_TOPOLOGY_TRANSITION_KEYS,
            f"topology.transitions[{index}]",
        )
        _integer(claimed["from_run_index"], f"topology.transitions[{index}].from")
        _integer(claimed["to_run_index"], f"topology.transitions[{index}].to")
        if dict(claimed) != expected:
            raise ParkingManoeuvreProofError(
                f"parking_v5 topology transition[{index}] index mismatch"
            )
        before = expected_runs[index]["lane_graph_id"]
        after = expected_runs[index + 1]["lane_graph_id"]
        if after not in lanes[before].get("successors", []):
            raise ParkingManoeuvreProofError(
                f"parking_v5 topology transition is not direct: {before}->{after}"
            )
    return {
        "samples": samples,
        "sample0_sha256": _parking_hash(samples[0], "topology.samples[0]"),
        "transforms": transforms,
        "runs": expected_runs,
    }


def _validate_v5_full_obb(
    poses: Sequence[Mapping[str, Any]], registry_context: Mapping[str, Any],
    map_context: Mapping[str, Any], *, own_slot_polygon: Optional[Any] = None,
) -> float:
    """Replay the <=0.2m full-OBB exact-lane/static audit for v5."""

    try:
        from shapely.geometry import Point, Polygon
    except ImportError as exc:
        raise ParkingManoeuvreProofError(
            "parking v5 full-OBB replay requires shapely"
        ) from exc
    local = registry_context["entry"]["vehicle_geometry"]["local_bbox"]
    ex = float(local["extent"]["x"])
    ey = float(local["extent"]["y"])
    lx = float(local["location"]["x"])
    ly = float(local["location"]["y"])

    def lattice(half: float) -> list[float]:
        count = max(1, int(math.ceil(2.0 * half / 0.2)))
        return [-half + (2.0 * half * index / count) for index in range(count + 1)]

    local_samples = [
        (lx + dx, ly + dy) for dx in lattice(ex) for dy in lattice(ey)
    ]
    own_slot = None
    if own_slot_polygon is not None:
        own_slot = Polygon(own_slot_polygon).buffer(PARKING_SLOT_TOLERANCE_M)
    carla = map_context["carla"]
    carla_map = map_context["map"]
    map_sha = str(registry_context["entry"]["xodr"]["sha256"])
    for pose_index, pose in enumerate(poses):
        yaw = math.radians(float(pose["yaw_deg"]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        for local_x, local_y in local_samples:
            world_x = float(pose["x"]) + cosine * local_x - sine * local_y
            world_y = float(pose["y"]) + sine * local_x + cosine * local_y
            if own_slot is not None and own_slot.covers(Point(world_x, world_y)):
                continue
            key = (
                map_sha, round(world_x, 4), round(world_y, 4),
                round(float(pose["z"]), 3),
            )
            legal = _V5_LANE_POINT_CACHE.get(key)
            if legal is None:
                legal = _exact_waypoint(carla_map, carla, {
                    "x": world_x, "y": world_y, "z": float(pose["z"]),
                }) is not None
                _V5_LANE_POINT_CACHE[key] = legal
            if not legal:
                raise ParkingManoeuvreProofError(
                    f"parking v5 full OBB leaves exact lane/own slot at pose {pose_index}"
                )
    bbox_payload = registry_context.get("bbox_payload")
    if not isinstance(bbox_payload, Mapping):
        raise ParkingManoeuvreProofError(
            "parking v5 full-OBB replay requires frozen static bboxes"
        )
    overlap = _vertical_overlap_index_sets(
        poses,
        bbox_payload["bboxes"],
        registry_context["local_bbox"],
        actor_reference_z_min_m=float(
            registry_context["entry"]["vehicle_geometry"]["actor_reference_z_min_m"]
        ),
        actor_reference_z_max_m=float(
            registry_context["entry"]["vehicle_geometry"]["actor_reference_z_max_m"]
        ),
    )
    clearance = _static_horizontal_clearance(
        poses, bbox_payload["bboxes"], overlap,
        local_bbox=registry_context["local_bbox"],
    )
    if clearance < PARKING_STATIC_CLEARANCE_MIN_M - PARKING_V5_EPS:
        raise ParkingManoeuvreProofError(
            "parking v5 full OBB violates frozen static clearance"
        )
    return clearance


def _validate_v5_ingress(
    selected: Mapping[str, Any], raw_poses: Sequence[Mapping[str, Any]],
    raw_rear_poses: Sequence[Mapping[str, Any]], proof: Mapping[str, Any],
    approach: Mapping[str, Any], approach_proof: Mapping[str, Any],
    center: tuple[float, float], slot_heading: float,
    registry_context: Mapping[str, Any], map_context: Mapping[str, Any],
) -> Dict[str, Any]:
    aligned = _integer(
        proof["aligned_lead_in_start_index"],
        "reverse_manoeuvre.proof.aligned_lead_in_start_index",
    )
    slot_axis = _integer(
        proof["slot_axis_approach_index"],
        "reverse_manoeuvre.proof.slot_axis_approach_index",
    )
    if not 0 < aligned < slot_axis < len(raw_poses) - 1:
        raise ParkingManoeuvreProofError(
            "parking v5 ingress indices must satisfy 0 < aligned < slot-axis < terminal"
        )
    topology = _validate_v5_topology(
        raw_poses, proof["construction_topology_proof"],
        expected_space=PARKING_V5_TARGET_POSE_INDEX_SPACE,
        expected_first=0, expected_last=slot_axis,
        map_context=map_context, require_raw_transform=False,
    )
    line = raw_rear_poses[aligned:slot_axis + 1]
    actor_line = raw_poses[aligned:slot_axis + 1]
    if any(pose["gear"] != "forward" for pose in line):
        raise ParkingManoeuvreProofError(
            "parking v5 aligned lead-in must be entirely forward"
        )
    start, end = line[0], line[-1]
    dx = float(end["x"]) - float(start["x"])
    dy = float(end["y"]) - float(start["y"])
    baseline = math.hypot(dx, dy)
    if baseline <= 1e-8:
        raise ParkingManoeuvreProofError("parking v5 aligned lead-in has no baseline")
    ux, uy = dx / baseline, dy / baseline
    baseline_yaw = math.degrees(math.atan2(uy, ux))
    accumulated = 0.0
    maximum_lateral = 0.0
    maximum_yaw_error = 0.0
    previous_progress = -math.inf
    for absolute_index, (rear_pose, actor_pose) in enumerate(
        zip(line, actor_line), aligned,
    ):
        rx = float(rear_pose["x"]) - float(start["x"])
        ry = float(rear_pose["y"]) - float(start["y"])
        progress = rx * ux + ry * uy
        lateral = abs(rx * (-uy) + ry * ux)
        if absolute_index > aligned and progress <= previous_progress + 1e-9:
            raise ParkingManoeuvreProofError(
                "parking v5 aligned lead-in is not strict-forward"
            )
        previous_progress = progress
        maximum_lateral = max(maximum_lateral, lateral)
        maximum_yaw_error = max(
            maximum_yaw_error,
            _angle_error_degrees(rear_pose["yaw_deg"], baseline_yaw),
            _angle_error_degrees(actor_pose["yaw_deg"], baseline_yaw),
        )
        if absolute_index < slot_axis:
            following = raw_rear_poses[absolute_index + 1]
            segment = _distance(rear_pose, following)
            if not 1e-6 < segment <= PARKING_SAMPLE_SPACING_M + PARKING_V5_EPS:
                raise ParkingManoeuvreProofError(
                    "parking v5 aligned lead-in spacing must be in (0,0.1]m"
                )
            accumulated += segment
            maximum_yaw_error = max(
                maximum_yaw_error,
                _angle_error_degrees(_chord_yaw(rear_pose, following), baseline_yaw),
            )
    if (
        baseline < PARKING_V5_ALIGNED_LEAD_IN_MIN_M - PARKING_V5_EPS
        or accumulated < PARKING_V5_ALIGNED_LEAD_IN_MIN_M - PARKING_V5_EPS
        or maximum_lateral > 0.001 + PARKING_V5_EPS
        or maximum_yaw_error > 1.0 + PARKING_V5_EPS
    ):
        raise ParkingManoeuvreProofError(
            "parking v5 aligned lead-in fails 8m/straight/forward qualification"
        )
    topology_samples = topology["samples"]
    maximum_lane_yaw_error = 0.0
    for pose_index in range(aligned, slot_axis + 1):
        sample = topology_samples[pose_index]
        waypoint = map_context["map"].get_waypoint_xodr(
            sample["road_id"], sample["lane_id"], float(sample["s_m"]),
        )
        if waypoint is None:
            raise ParkingManoeuvreProofError(
                "parking v5 aligned topology sample cannot reconstruct"
            )
        maximum_lane_yaw_error = max(
            maximum_lane_yaw_error,
            _angle_error_degrees(waypoint.transform.rotation.yaw, baseline_yaw),
        )
    if maximum_lane_yaw_error > 1.0 + PARKING_V5_EPS:
        raise ParkingManoeuvreProofError(
            "parking v5 aligned lead-in lane yaw differs by more than 1 degree"
        )

    expected_approach = {
        key: raw_poses[slot_axis][key] for key in ("x", "y", "z", "yaw_deg")
    }
    if selected.get("approach_pose") != expected_approach:
        raise ParkingManoeuvreProofError(
            "parking v5 approach_pose must be the raw slot-axis actor pose"
        )
    slot_sample = topology_samples[slot_axis]
    slot_waypoint = map_context["map"].get_waypoint_xodr(
        slot_sample["road_id"], slot_sample["lane_id"], float(slot_sample["s_m"]),
    )
    if slot_waypoint is None:
        raise ParkingManoeuvreProofError("parking v5 slot-axis sample cannot reconstruct")
    lane_yaw = _canonical_yaw(slot_waypoint.transform.rotation.yaw)
    for field in ("road_id", "section_id", "lane_id"):
        _integer(
            approach_proof[field], f"approach_pose_proof.{field}",
            minimum=-(2 ** 31),
        )
    if (
        approach_proof["lane_id"] == 0
        or approach_proof["road_id"] != slot_sample["road_id"]
        or approach_proof["section_id"] != slot_sample["section_id"]
        or approach_proof["lane_id"] != slot_sample["lane_id"]
        or _strict_finite(
            approach_proof["lane_yaw_deg"], "approach_pose_proof.lane_yaw_deg"
        ) != lane_yaw
    ):
        raise ParkingManoeuvreProofError(
            "parking v5 approach lane identity/yaw disagrees with slot-axis sample"
        )
    heading_error = _angle_error_degrees(lane_yaw, slot_heading)
    claimed_heading_error = _strict_finite(
        approach_proof["heading_error_deg"], "approach_pose_proof.heading_error_deg"
    )
    claimed_distance = _strict_finite(
        approach_proof["distance_from_slot_center_m"],
        "approach_pose_proof.distance_from_slot_center_m",
    )
    measured_distance = math.hypot(
        float(approach["x"]) - center[0], float(approach["y"]) - center[1]
    )
    heading = math.radians(slot_heading)
    hx, hy = math.cos(heading), math.sin(heading)
    dx = float(approach["x"]) - center[0]
    dy = float(approach["y"]) - center[1]
    along = dx * hx + dy * hy
    cross = dx * (-hy) + dy * hx
    if (
        abs(claimed_heading_error - heading_error) > 1e-3
        or abs(claimed_distance - measured_distance) > 1e-3
        or along <= 0.0
        or abs(cross) > 0.001 + PARKING_V5_EPS
        or abs(along - claimed_distance) > 0.001 + PARKING_V5_EPS
    ):
        raise ParkingManoeuvreProofError(
            "parking v5 approach proof is not the positive slot-axis intersection"
        )
    reverse = _require_exact_keys(
        approach_proof["reverse"], APPROACH_REVERSE_KEYS,
        "approach_pose_proof.reverse",
    )
    if dict(reverse) != {
        "method": PARKING_REVERSE_METHOD_V5,
        "own_slot_tolerance_m": PARKING_SLOT_TOLERANCE_M,
        "sample_spacing_max_m": PARKING_SAMPLE_SPACING_M,
        "samples": len(raw_poses) - slot_axis,
    }:
        raise ParkingManoeuvreProofError(
            "parking v5 slot-axis-to-reverse nested proof mismatch"
        )
    clearance = _validate_v5_full_obb(
        raw_poses, registry_context, map_context,
        own_slot_polygon=selected.get("polygon"),
    )
    return {
        "aligned_lead_in_start_index": aligned,
        "slot_axis_approach_index": slot_axis,
        "prelead_pose": {
            key: raw_poses[0][key] for key in ("x", "y", "z", "yaw_deg")
        },
        "prelead_rear_pose": dict(raw_rear_poses[0]),
        "target_prelead_sample_sha256": topology["sample0_sha256"],
        "construction_topology_proof": topology,
        "full_obb_static_clearance_m": clearance,
    }


def _validate_v5_compact_prefix(
    path: Sequence[Mapping[str, Any]], suffix_start: int,
    map_context: Mapping[str, Any],
) -> None:
    """Fail closed on the compact ordinary route before its dense tail."""

    ordinary = path[:suffix_start + 1]
    if len(ordinary) < 2:
        raise ParkingManoeuvreProofError(
            "parking v5 ordinary prefix must contain at least two poses"
        )
    carla = map_context["carla"]
    carla_map = map_context["map"]
    for index, pose in enumerate(ordinary):
        if _exact_waypoint(carla_map, carla, pose) is None:
            raise ParkingManoeuvreProofError(
                f"parking v5 ordinary pose {index} has no exact lane membership"
            )
    for index, (first, second) in enumerate(zip(ordinary, ordinary[1:])):
        gap = _distance(first, second)
        if gap > 4.0 + PARKING_V5_EPS:
            raise ParkingManoeuvreProofError(
                f"parking v5 compact ordinary gap exceeds 4m at {index}"
            )
        # Stable GRP compaction can leave a sub-5cm de-duplication connector.
        # It has no reliable chord direction and is ignored by the producer's
        # directed-polyline proof as well.
        if gap <= 0.05:
            continue
        chord = _chord_yaw(first, second)
        count = max(1, int(math.ceil(gap / 0.1)))
        for sample_index in range(count + 1):
            alpha = sample_index / count
            point = {
                axis: float(first.get(axis, 0.0))
                + alpha * (
                    float(second.get(axis, 0.0)) - float(first.get(axis, 0.0))
                )
                for axis in ("x", "y", "z")
            }
            waypoint = _exact_waypoint(carla_map, carla, point)
            if waypoint is None or (
                not waypoint.is_junction
                and _angle_error_degrees(
                    chord, waypoint.transform.rotation.yaw
                ) > 90.0 + PARKING_V5_EPS
            ):
                raise ParkingManoeuvreProofError(
                    f"parking v5 compact ordinary route is not directed at {index}"
                )
    checked_yaw = 0
    for index, pose in enumerate(ordinary[:-1]):
        following = next(
            (
                candidate for candidate in ordinary[index + 1:]
                if _distance(pose, candidate) > 0.01 + PARKING_V5_EPS
            ),
            None,
        )
        if following is None:
            continue
        checked_yaw += 1
        if _angle_error_degrees(pose["yaw_deg"], _chord_yaw(pose, following)) > 3.0 + PARKING_V5_EPS:
            raise ParkingManoeuvreProofError(
                f"parking v5 ordinary serialized yaw is not forward at {index}"
            )
    if checked_yaw == 0:
        raise ParkingManoeuvreProofError(
            "parking v5 ordinary prefix has no meaningful forward chord"
        )


def _validate_v5_route_proof(
    route: Mapping[str, Any], selected: Mapping[str, Any],
    raw_poses: Sequence[Mapping[str, Any]], executable_poses: Sequence[Mapping[str, Any]],
    ingress: Mapping[str, Any], registry_context: Mapping[str, Any],
    map_context: Mapping[str, Any],
) -> Dict[str, Any]:
    route_proof = _require_exact_keys(
        route.get("parking_v5_route_proof"), PARKING_V5_ROUTE_PROOF_KEYS,
        "parking_v5_route_proof",
    )
    if (
        route_proof["schema_version"] != PARKING_V5_ROUTE_SCHEMA
        or route_proof["method"] != PARKING_V5_ROUTE_METHOD
    ):
        raise ParkingManoeuvreProofError(
            "parking_v5_route_proof schema/method mismatch"
        )
    for field in (
        "oracle_path_sha256", "ordinary_tail_sha256", "explicit_suffix_sha256",
        "target_prelead_sample_sha256",
    ):
        _sha256(route_proof[field], f"parking_v5_route_proof.{field}")
    path = route.get("oracle_path")
    if not isinstance(path, list) or len(path) < 4:
        raise ParkingManoeuvreProofError("parking v5 oracle_path is invalid")
    for index, pose in enumerate(path):
        _require_exact_keys(
            pose, APPROACH_POSE_KEYS, f"oracle_path[{index}]"
        )
        for field in ("x", "y", "z", "yaw_deg"):
            _strict_finite(pose[field], f"oracle_path[{index}].{field}")
        if (
            float(pose["x"]) != round(float(pose["x"]), 4)
            or float(pose["y"]) != round(float(pose["y"]), 4)
            or float(pose["z"]) != round(float(pose["z"]), 4)
            or float(pose["yaw_deg"]) != _canonical_yaw(pose["yaw_deg"])
        ):
            raise ParkingManoeuvreProofError(
                f"parking v5 oracle_path precision/yaw mismatch at {index}"
            )
    if _parking_hash(path, "oracle_path") != route_proof["oracle_path_sha256"]:
        raise ParkingManoeuvreProofError("parking v5 oracle_path hash mismatch")
    tail_start = _integer(
        route_proof["ordinary_tail_start_index"],
        "parking_v5_route_proof.ordinary_tail_start_index",
    )
    suffix_start = _integer(
        route_proof["explicit_suffix_start_index"],
        "parking_v5_route_proof.explicit_suffix_start_index",
    )
    if not 0 < tail_start < suffix_start < len(path):
        raise ParkingManoeuvreProofError(
            "parking v5 route indices must satisfy 0 < tail < suffix < path"
        )
    exact_suffix = [
        {key: pose[key] for key in ("x", "y", "z", "yaw_deg")}
        for pose in executable_poses
    ]
    if path[suffix_start:] != exact_suffix:
        raise ParkingManoeuvreProofError(
            "parking v5 explicit suffix differs from collapsed target poses"
        )
    if _parking_hash(
        path[suffix_start:], "parking_v5 explicit suffix"
    ) != route_proof["explicit_suffix_sha256"]:
        raise ParkingManoeuvreProofError("parking v5 explicit suffix hash mismatch")
    if _parking_hash(
        path[tail_start:suffix_start + 1], "parking_v5 ordinary tail"
    ) != route_proof["ordinary_tail_sha256"]:
        raise ParkingManoeuvreProofError(
            "parking v5 ordinary tail hash must include the shared prelead once"
        )
    if route_proof["target_prelead_sample_sha256"] != ingress["target_prelead_sample_sha256"]:
        raise ParkingManoeuvreProofError(
            "parking v5 route is not bound to target construction sample0"
        )
    intended = _strict_finite(
        route.get("intended_path_length_m"), "intended_path_length_m"
    )
    if abs(intended - round(_path_length(path), 3)) > PARKING_V5_EPS:
        raise ParkingManoeuvreProofError(
            "parking v5 intended_path_length_m is not the once-counted path"
        )
    if path[suffix_start] != ingress["prelead_pose"]:
        raise ParkingManoeuvreProofError(
            "parking v5 explicit suffix must begin at the unique prelead"
        )

    topology = _validate_v5_topology(
        path, route_proof["ordinary_tail_topology_proof"],
        expected_space=PARKING_V5_ARM_POSE_INDEX_SPACE,
        expected_first=tail_start, expected_last=suffix_start - 1,
        map_context=map_context, require_raw_transform=True,
    )
    last_sample = topology["samples"][-1]
    target_sample = ingress["construction_topology_proof"]["samples"][0]
    last_key = (
        last_sample["road_id"], last_sample["section_id"], last_sample["lane_id"],
    )
    target_key = (
        target_sample["road_id"], target_sample["section_id"], target_sample["lane_id"],
    )
    by_key = map_context["by_key"]
    directions = map_context["directions"]
    lanes = map_context["lanes"]
    last_graph = by_key[last_key]
    target_graph = by_key[target_key]
    if last_graph == target_graph:
        if (
            float(target_sample["s_m"]) - float(last_sample["s_m"])
        ) * directions[last_graph] <= 1e-8:
            raise ParkingManoeuvreProofError(
                "parking v5 tail-to-prelead s is not strict-forward"
            )
    elif target_graph not in lanes[last_graph].get("successors", []):
        raise ParkingManoeuvreProofError(
            "parking v5 tail-to-prelead is not a direct lane successor"
        )
    last_waypoint = map_context["map"].get_waypoint_xodr(
        last_sample["road_id"], last_sample["lane_id"], float(last_sample["s_m"]),
    )
    target_waypoint = map_context["map"].get_waypoint_xodr(
        target_sample["road_id"], target_sample["lane_id"], float(target_sample["s_m"]),
    )
    cross_gap = _distance(path[suffix_start - 1], path[suffix_start])
    cross_chord = _chord_yaw(path[suffix_start - 1], path[suffix_start])
    if (
        last_waypoint is None or target_waypoint is None
        or cross_gap > PARKING_SAMPLE_SPACING_M + PARKING_V5_EPS
        or max(
            _angle_error_degrees(cross_chord, last_waypoint.transform.rotation.yaw),
            _angle_error_degrees(cross_chord, target_waypoint.transform.rotation.yaw),
        ) > 3.0 + PARKING_V5_EPS
    ):
        raise ParkingManoeuvreProofError(
            "parking v5 tail-to-prelead gap/chord qualification failed"
        )
    tail_with_prelead = path[tail_start:suffix_start + 1]
    tail_samples_with_prelead = [*topology["samples"], target_sample]
    for chord_index, (first, second) in enumerate(
        zip(tail_with_prelead, tail_with_prelead[1:])
    ):
        spacing = _distance(first, second)
        if not 1e-8 < spacing <= PARKING_SAMPLE_SPACING_M + PARKING_V5_EPS:
            raise ParkingManoeuvreProofError(
                "parking v5 dense tail spacing must be in (0,0.1]m"
            )
        first_sample = tail_samples_with_prelead[chord_index]
        second_sample = tail_samples_with_prelead[chord_index + 1]
        first_waypoint = map_context["map"].get_waypoint_xodr(
            first_sample["road_id"], first_sample["lane_id"],
            float(first_sample["s_m"]),
        )
        second_waypoint = map_context["map"].get_waypoint_xodr(
            second_sample["road_id"], second_sample["lane_id"],
            float(second_sample["s_m"]),
        )
        chord = _chord_yaw(first, second)
        if (
            first_waypoint is None or second_waypoint is None
            or max(
                _angle_error_degrees(
                    chord, first_waypoint.transform.rotation.yaw
                ),
                _angle_error_degrees(
                    chord, second_waypoint.transform.rotation.yaw
                ),
            ) > 3.0 + PARKING_V5_EPS
        ):
            raise ParkingManoeuvreProofError(
                "parking v5 dense-tail chord differs from exact lane by more than 3 degrees"
            )
    tail_length = _path_length(tail_with_prelead)
    if tail_length < PARKING_V5_DENSE_TAIL_MIN_M - PARKING_V5_EPS:
        raise ParkingManoeuvreProofError("parking v5 dense tail is shorter than 7m")
    prefix_gap = _distance(path[tail_start - 1], path[tail_start])
    if prefix_gap > 1.0 + PARKING_V5_EPS:
        raise ParkingManoeuvreProofError(
            "parking v5 compact-prefix to dense-tail gap exceeds 1m"
        )
    prefix_seam = _angle_error_degrees(
        _chord_yaw(*_nonzero_chord(path[:tail_start + 1], reverse=True)),
        _chord_yaw(*_nonzero_chord(tail_with_prelead)),
    )
    explicit_seam = _angle_error_degrees(
        cross_chord, _chord_yaw(*_nonzero_chord(path[suffix_start:])),
    )
    if prefix_seam > 3.0 + PARKING_V5_EPS or explicit_seam > 3.0 + PARKING_V5_EPS:
        raise ParkingManoeuvreProofError("parking v5 protected route seam exceeds 3 degrees")
    _validate_v5_compact_prefix(path, suffix_start, map_context)
    tail_clearance = _validate_v5_full_obb(
        [dict(pose, gear="forward") for pose in tail_with_prelead],
        registry_context, map_context,
    )
    return {
        "ordinary_tail_start_index": tail_start,
        "explicit_suffix_start_index": suffix_start,
        "ordinary_tail_length_m": tail_length,
        "tail_to_prelead_gap_m": cross_gap,
        "prefix_to_tail_gap_m": prefix_gap,
        "prefix_to_tail_seam_deg": prefix_seam,
        "ordinary_to_explicit_seam_deg": explicit_seam,
        "ordinary_tail_static_clearance_m": tail_clearance,
        "ordinary_tail_topology_proof": topology,
    }


def resolve_parking_target(route: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the statically selected parking target, if this is one."""

    target = route.get("target")
    if not isinstance(target, Mapping):
        return None
    if target.get("type") == "parking_slot":
        return target
    if target.get("type") != "if_else":
        return None
    condition = target.get("condition")
    if not isinstance(condition, str) or not condition.startswith("occupied:"):
        return None
    slot = condition.split(":", 1)[1]
    setup = route.get("actor_setup")
    occupancy = setup.get("slot_occupancy") if isinstance(setup, Mapping) else None
    selected = occupancy.get(slot) if isinstance(occupancy, Mapping) else None
    if not isinstance(selected, bool):
        raise ParkingManoeuvreProofError(
            "parking if_else has no boolean frozen slot occupancy"
        )
    branch = target.get("then" if selected else "else")
    if not isinstance(branch, Mapping):
        raise ParkingManoeuvreProofError(
            "parking if_else selected branch must be an object"
        )
    return branch if branch.get("type") == "parking_slot" else None


def validate_parking_manoeuvre(
    route: Mapping[str, Any],
    target: Optional[Mapping[str, Any]] = None,
    *,
    registry: Any = None,
    require_oracle_suffix: bool = True,
    verify_sources: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate a selected core parking target and return compiled geometry.

    Safety may omit the witness because it is an independent refusal track.
    Any supplied v4/v5 object is nevertheless validated.  Every A--D parking
    route must carry the complete proof and exact ``oracle_path`` suffix.
    """

    selected = target if target is not None else resolve_parking_target(route)
    if selected is None or selected.get("type") != "parking_slot":
        return None
    layer = str(route.get("layer", ""))
    raw_manoeuvre = selected.get("reverse_manoeuvre")
    if raw_manoeuvre is None and layer == "Safety":
        return None
    if raw_manoeuvre is None and layer in {"A", "B", "C", "D"}:
        raise ParkingManoeuvreProofError(
            "core parking target requires reverse_manoeuvre v4/v5"
        )
    if not isinstance(raw_manoeuvre, Mapping) or set(raw_manoeuvre) != REVERSE_MANOEUVRE_KEYS:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre keys must be exactly {}".format(
                sorted(REVERSE_MANOEUVRE_KEYS)
            )
        )
    manoeuvre_schema = raw_manoeuvre["schema_version"]
    if manoeuvre_schema not in {PARKING_REVERSE_SCHEMA, PARKING_REVERSE_SCHEMA_V5}:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre has unsupported schema_version"
        )
    is_v5 = manoeuvre_schema == PARKING_REVERSE_SCHEMA_V5
    registry_context = validate_parking_manoeuvre_registry(
        registry, str(route.get("town", "")), verify_sources=verify_sources
    )
    if is_v5:
        occupancy = registry_context["entry"]["slot_collision_occupancy"]
        slot_number = selected.get("slot_number")
        if not isinstance(slot_number, str) or slot_number not in occupancy["clear_slots"]:
            raise ParkingManoeuvreProofError(
                "parking target slot is not terminal-spawn-clear in the frozen CARLA occupancy asset"
            )
    expected_registry_schema = (
        PARKING_REGISTRY_SCHEMA_V5 if is_v5 else PARKING_REGISTRY_SCHEMA
    )
    if registry_context["registry_schema_version"] != expected_registry_schema:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre and registry versions disagree"
        )

    if is_v5:
        raw_approach = _require_exact_keys(
            selected.get("approach_pose"), APPROACH_POSE_KEYS,
            "target.approach_pose",
        )
        for field in ("x", "y", "z", "yaw_deg"):
            _strict_finite(raw_approach[field], f"target.approach_pose.{field}")
    approach = _pose(
        selected.get("approach_pose"),
        "target.approach_pose",
        APPROACH_POSE_KEYS,
    )
    raw_approach_proof = selected.get("approach_pose_proof")
    if not isinstance(raw_approach_proof, Mapping) or set(raw_approach_proof) != APPROACH_PROOF_KEYS:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof keys must be exactly {}".format(
                sorted(APPROACH_PROOF_KEYS)
            )
        )
    approach_proof = dict(raw_approach_proof)
    expected_approach_method = (
        PARKING_APPROACH_METHOD_V5 if is_v5 else PARKING_APPROACH_METHOD
    )
    if approach_proof["method"] != expected_approach_method:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof has unsupported method"
        )
    for field in ("road_id", "section_id", "lane_id"):
        _integer(approach_proof[field], f"approach_pose_proof.{field}", minimum=-2**31)
    if int(approach_proof["lane_id"]) == 0:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof.lane_id cannot be zero"
        )
    number = _strict_finite if is_v5 else _finite
    lane_yaw = number(
        approach_proof["lane_yaw_deg"], "approach_pose_proof.lane_yaw_deg"
    )
    if _angle_error_degrees(approach["yaw_deg"], lane_yaw) > 1.0:
        raise ParkingManoeuvreProofError(
            "approach_pose yaw must match physical lane_yaw_deg within 1 degree"
        )
    try:
        slot_heading = number(selected["heading_deg"], "target.heading_deg")
    except KeyError as exc:
        raise ParkingManoeuvreProofError(
            "parking target requires heading_deg"
        ) from exc
    expected_heading_error = _angle_error_degrees(lane_yaw, slot_heading)
    claimed_heading_error = number(
        approach_proof["heading_error_deg"],
        "approach_pose_proof.heading_error_deg",
    )
    if abs(claimed_heading_error - expected_heading_error) > 1e-3:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof heading_error_deg disagrees with lane/slot yaw"
        )
    center = _polygon_center(selected.get("polygon"))
    if is_v5:
        polygon = selected.get("polygon")
        if not isinstance(polygon, list):
            raise ParkingManoeuvreProofError("parking v5 polygon must be a list")
        for index, point in enumerate(polygon):
            if not isinstance(point, list) or len(point) != 2:
                raise ParkingManoeuvreProofError(
                    f"parking v5 polygon[{index}] must be exact [x,y]"
                )
            _strict_finite(point[0], f"target.polygon[{index}].x")
            _strict_finite(point[1], f"target.polygon[{index}].y")
    expected_distance = math.hypot(
        approach["x"] - center[0], approach["y"] - center[1]
    )
    claimed_distance = number(
        approach_proof["distance_from_slot_center_m"],
        "approach_pose_proof.distance_from_slot_center_m",
    )
    if abs(claimed_distance - expected_distance) > 1e-3:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof distance disagrees with slot center"
        )

    raw_reverse_proof = approach_proof["reverse"]
    if not isinstance(raw_reverse_proof, Mapping) or set(raw_reverse_proof) != APPROACH_REVERSE_KEYS:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof.reverse keys must be exactly {}".format(
                sorted(APPROACH_REVERSE_KEYS)
            )
        )
    expected_reverse_method = (
        PARKING_REVERSE_METHOD_V5 if is_v5 else PARKING_REVERSE_METHOD
    )
    if raw_reverse_proof["method"] != expected_reverse_method:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof.reverse has unsupported method"
        )
    own_tolerance = number(
        raw_reverse_proof["own_slot_tolerance_m"],
        "approach_pose_proof.reverse.own_slot_tolerance_m",
    )
    reverse_spacing = number(
        raw_reverse_proof["sample_spacing_max_m"],
        "approach_pose_proof.reverse.sample_spacing_max_m",
    )
    if abs(own_tolerance - PARKING_SLOT_TOLERANCE_M) > 1e-9:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof.reverse own-slot tolerance must equal 0.2m"
        )
    if abs(reverse_spacing - PARKING_SAMPLE_SPACING_M) > 1e-9:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof.reverse sample spacing must equal 0.1m"
        )

    raw_poses = raw_manoeuvre["poses"]
    raw_rear_poses = raw_manoeuvre["rear_axle_poses"]
    if (
        not isinstance(raw_poses, list)
        or not isinstance(raw_rear_poses, list)
        or len(raw_poses) < 5
        or len(raw_rear_poses) != len(raw_poses)
    ):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre actor/rear pose lists must have equal length >= 5"
        )
    poses = [
        _pose(value, f"reverse_manoeuvre.poses[{index}]", REVERSE_POSE_KEYS)
        for index, value in enumerate(raw_poses)
    ]
    rear_poses = [
        _pose(
            value, f"reverse_manoeuvre.rear_axle_poses[{index}]",
            REVERSE_POSE_KEYS,
        )
        for index, value in enumerate(raw_rear_poses)
    ]
    if is_v5:
        for sequence_name, sequence in (
            ("poses", raw_poses), ("rear_axle_poses", raw_rear_poses),
        ):
            xy_decimals = 4 if sequence_name == "poses" else 6
            for index, pose in enumerate(sequence):
                for field in ("x", "y", "z", "yaw_deg"):
                    _strict_finite(
                        pose[field], f"reverse_manoeuvre.{sequence_name}[{index}].{field}"
                    )
                if (
                    float(pose["x"]) != round(float(pose["x"]), xy_decimals)
                    or float(pose["y"]) != round(float(pose["y"]), xy_decimals)
                    or float(pose["yaw_deg"]) != _canonical_yaw(pose["yaw_deg"])
                ):
                    raise ParkingManoeuvreProofError(
                        f"reverse_manoeuvre.{sequence_name}[{index}] v5 precision/yaw is not frozen"
                    )
    for sequence_name, sequence in (
        ("poses", poses), ("rear_axle_poses", rear_poses),
    ):
        for index, pose in enumerate(sequence):
            if not -180.0 <= float(pose["yaw_deg"]) < 180.0:
                raise ParkingManoeuvreProofError(
                    "reverse_manoeuvre.{}[{}].yaw_deg is not canonical"
                    .format(sequence_name, index)
                )
    for index, (raw_actor, raw_rear) in enumerate(
        zip(raw_poses, raw_rear_poses)
    ):
        if (
            raw_rear["x"] != round(float(raw_rear["x"]), 6)
            or raw_rear["y"] != round(float(raw_rear["y"]), 6)
            or raw_rear["z"] != 0.0
            or raw_rear["yaw_deg"] != round(float(raw_rear["yaw_deg"]), 3)
        ):
            raise ParkingManoeuvreProofError(
                f"rear_axle_poses[{index}] precision/frame is not frozen"
            )
        yaw = math.radians(float(raw_rear["yaw_deg"]))
        expected_actor = {
            "x": round(
                float(raw_rear["x"])
                + (-PARKING_REAR_AXLE_X_M) * math.cos(yaw),
                4,
            ),
            "y": round(
                float(raw_rear["y"])
                + (-PARKING_REAR_AXLE_X_M) * math.sin(yaw),
                4,
            ),
            "z": raw_rear["z"],
            "yaw_deg": raw_rear["yaw_deg"],
            "gear": raw_rear["gear"],
        }
        if raw_actor != expected_actor:
            raise ParkingManoeuvreProofError(
                f"poses[{index}] is not the exact frozen rear-to-actor transform"
            )
    if poses[0]["gear"] != "forward" or poses[-1]["gear"] != "reverse":
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre must begin forward and end reverse"
        )
    raw_first_pose = {
        axis: raw_poses[0][axis] for axis in ("x", "y", "z", "yaw_deg")
    }
    if not is_v5 and raw_first_pose != selected.get("approach_pose"):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre first pose disagrees exactly with approach_pose"
        )

    transitions = [
        index
        for index, (first, second) in enumerate(zip(poses, poses[1:]))
        if first["gear"] != second["gear"]
    ]
    rear_transitions = [
        index
        for index, (first, second) in enumerate(zip(rear_poses, rear_poses[1:]))
        if first["gear"] != second["gear"]
    ]
    if len(transitions) != 1 or rear_transitions != transitions:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre actor/rear paths require one shared gear transition"
        )
    transition = transitions[0]
    if poses[transition]["gear"] != "forward" or poses[transition + 1]["gear"] != "reverse":
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre gear transition must be forward-to-reverse"
        )
    if any(pose["gear"] != "forward" for pose in poses[:transition + 1]) or any(
        pose["gear"] != "reverse" for pose in poses[transition + 1:]
    ):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre gear phases are not contiguous"
        )

    measured_max_curvature = 0.0
    required_max_abs_steer = 0.0
    dynamics_context = registry_context.get("dynamics")
    if not isinstance(dynamics_context, Mapping):
        raise ParkingManoeuvreProofError(
            "parking v4 requires a verified dynamics asset"
        )
    steer_samples = dynamics_context["canonical_samples"]
    for index, ((first, second), (rear_first, rear_second)) in enumerate(zip(
        zip(poses, poses[1:]), zip(rear_poses, rear_poses[1:]),
    )):
        actor_distance = math.sqrt(sum(
            (second[axis] - first[axis]) ** 2 for axis in ("x", "y", "z")
        ))
        rear_distance = math.sqrt(sum(
            (rear_second[axis] - rear_first[axis]) ** 2
            for axis in ("x", "y", "z")
        ))
        signed_yaw_step = _signed_angle_radians(
            rear_second["yaw_deg"], rear_first["yaw_deg"]
        )
        yaw_step = abs(math.degrees(signed_yaw_step))
        if index == transition:
            if (
                any(
                    raw_poses[index][axis] != raw_poses[index + 1][axis]
                    for axis in ("x", "y", "z", "yaw_deg")
                )
                or any(
                    raw_rear_poses[index][axis]
                    != raw_rear_poses[index + 1][axis]
                    for axis in ("x", "y", "z", "yaw_deg")
                )
            ):
                raise ParkingManoeuvreProofError(
                    "reverse_manoeuvre actor/rear gear transition must repeat raw-exact poses"
                )
            continue
        if (
            actor_distance <= 1e-6
            or actor_distance > PARKING_SAMPLE_SPACING_M + 1e-9
            or rear_distance <= 1e-6
            or rear_distance > PARKING_SAMPLE_SPACING_M + 1e-9
        ):
            raise ParkingManoeuvreProofError(
                "reverse_manoeuvre actor/rear moving spacing must be in (0,0.1]m"
            )
        if yaw_step > PARKING_MAX_YAW_STEP_DEG + 1e-9:
            raise ParkingManoeuvreProofError(
                "reverse_manoeuvre adjacent yaw change exceeds 5 degrees"
            )
        tangent = math.degrees(math.atan2(
            rear_second["y"] - rear_first["y"],
            rear_second["x"] - rear_first["x"],
        ))
        expected_motion_yaw = (
            rear_first["yaw_deg"]
            if rear_first["gear"] == "forward"
            else rear_first["yaw_deg"] + 180.0
        )
        if _angle_error_degrees(tangent, expected_motion_yaw) > PARKING_MAX_MOTION_HEADING_ERROR_DEG + 1e-9:
            raise ParkingManoeuvreProofError(
                "reverse_manoeuvre rear pose yaw disagrees with gear-directed motion"
            )
        signed_distance = rear_distance * (
            -1.0 if rear_first["gear"] == "reverse" else 1.0
        )
        signed_curvature = signed_yaw_step / signed_distance
        normalized_steer = normalized_steer_for_curvature(
            signed_curvature, steer_samples
        )
        measured_max_curvature = max(
            measured_max_curvature, abs(signed_curvature)
        )
        required_max_abs_steer = max(
            required_max_abs_steer, abs(normalized_steer)
        )

    expected_terminal = {
        "x": round(center[0], 4),
        "y": round(center[1], 4),
        "z": 0.0,
        "yaw_deg": (
            _canonical_yaw(slot_heading)
            if is_v5
            else round((slot_heading + 180.0) % 360.0 - 180.0, 3)
        ),
        "gear": "reverse",
    }
    if raw_poses[-1] != expected_terminal:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre terminal actor pose is not the exact slot pose"
        )

    # ``poses`` retains the exact duplicate transform at the gear transition:
    # the forward entry and reverse entry are separate controller states.  A
    # CARLA route polyline, however, must never contain consecutive duplicate
    # waypoints.  Its executable suffix therefore keeps the forward staging
    # pose and omits only the immediately following reverse-labelled duplicate.
    executable_poses = poses[:transition + 1] + poses[transition + 2:]
    suffix_start = None
    if require_oracle_suffix:
        oracle_path = route.get("oracle_path")
        if (
            not isinstance(oracle_path, list)
            or len(oracle_path) <= len(executable_poses)
        ):
            raise ParkingManoeuvreProofError(
                "parking oracle_path requires a forward prefix plus the manoeuvre suffix"
            )
        suffix_start = len(oracle_path) - len(executable_poses)
        raw_executable_poses = (
            raw_poses[:transition + 1] + raw_poses[transition + 2:]
        )
        exact_suffix = [
            {axis: pose[axis] for axis in ("x", "y", "z", "yaw_deg")}
            for pose in raw_executable_poses
        ]
        # Validate complete transform shape before comparing serialized values.
        for index, path_point in enumerate(oracle_path[suffix_start:]):
            _pose(
                path_point,
                f"oracle_path[{suffix_start + index}]",
                APPROACH_POSE_KEYS,
            )
        if oracle_path[suffix_start:] != exact_suffix:
            raise ParkingManoeuvreProofError(
                "oracle_path suffix disagrees exactly with serialized "
                "reverse_manoeuvre poses"
            )

    raw_proof = raw_manoeuvre["proof"]
    expected_proof_keys = REVERSE_PROOF_KEYS_V5 if is_v5 else REVERSE_PROOF_KEYS
    if not isinstance(raw_proof, Mapping) or set(raw_proof) != expected_proof_keys:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof keys must be exactly {}".format(
                sorted(expected_proof_keys)
            )
        )
    proof = dict(raw_proof)
    expected_proof_method = (
        PARKING_MANOEUVRE_PROOF_METHOD_V5
        if is_v5 else PARKING_MANOEUVRE_PROOF_METHOD
    )
    if proof["method"] != expected_proof_method:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof has unsupported method"
        )
    if proof["poses_sha256"] != _parking_hash(raw_poses, "reverse_manoeuvre.poses"):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof poses_sha256 mismatch"
        )
    if proof["rear_axle_poses_sha256"] != _parking_hash(
        raw_rear_poses, "reverse_manoeuvre.rear_axle_poses"
    ):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof rear_axle_poses_sha256 mismatch"
        )
    if (
        proof["pose_reference"] != PARKING_POSE_REFERENCE
        or proof["integration_reference"] != PARKING_INTEGRATION_REFERENCE
    ):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre proof pose/integration references are unsupported"
        )
    pose_count = _integer(
        proof["pose_count"], "reverse_manoeuvre.proof.pose_count", minimum=5
    )
    if pose_count != len(poses):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof pose_count mismatch"
        )
    expected_nested_samples = pose_count
    if is_v5:
        slot_axis_for_count = _integer(
            proof["slot_axis_approach_index"],
            "reverse_manoeuvre.proof.slot_axis_approach_index",
        )
        expected_nested_samples = pose_count - slot_axis_for_count
    if _integer(
        raw_reverse_proof["samples"],
        "approach_pose_proof.reverse.samples",
        minimum=1,
    ) != expected_nested_samples:
        raise ParkingManoeuvreProofError(
            "approach_pose_proof.reverse.samples count mismatch"
        )
    spacing = _finite(
        proof["sample_spacing_max_m"],
        "reverse_manoeuvre.proof.sample_spacing_max_m",
    )
    if abs(spacing - PARKING_SAMPLE_SPACING_M) > 1e-9:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof sample spacing must equal 0.1m"
        )
    rear_spacing = _finite(
        proof["rear_axle_sample_spacing_max_m"],
        "reverse_manoeuvre.proof.rear_axle_sample_spacing_max_m",
    )
    if abs(rear_spacing - PARKING_SAMPLE_SPACING_M) > 1e-9:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof rear-axle sample spacing must equal 0.1m"
        )
    numeric_positive = (
        "footprint_length_m",
        "footprint_width_m",
    )
    if is_v5:
        for field in numeric_positive + (
            "sample_spacing_max_m", "rear_axle_sample_spacing_max_m",
            "required_max_rear_axle_curvature_1pm",
            "required_max_abs_normalized_steer",
            "max_normalized_control_steer", "footprint_tolerance_m",
            "static_clearance_min_m",
        ):
            _strict_finite(proof[field], f"reverse_manoeuvre.proof.{field}")
    numeric = {
        field: _finite(proof[field], f"reverse_manoeuvre.proof.{field}")
        for field in numeric_positive + (
            "required_max_rear_axle_curvature_1pm",
            "required_max_abs_normalized_steer",
            "max_normalized_control_steer",
            "footprint_tolerance_m",
            "static_clearance_min_m",
        )
    }
    if any(numeric[field] <= 0.0 for field in numeric_positive):
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre proof dimensions/steering must be positive"
        )
    exact_frozen = {
        "max_normalized_control_steer": PARKING_MAX_NORMALIZED_CONTROL_STEER,
        "footprint_length_m": PARKING_FOOTPRINT_LENGTH_M,
        "footprint_width_m": PARKING_FOOTPRINT_WIDTH_M,
    }
    if any(
        abs(numeric[field] - expected) > 1e-6
        for field, expected in exact_frozen.items()
    ):
        raise ParkingManoeuvreProofError(
            "parking manoeuvre proof vehicle/control constants disagree with "
            "the frozen Lincoln v4 contract"
        )
    if numeric["required_max_rear_axle_curvature_1pm"] < 0.0:
        raise ParkingManoeuvreProofError(
            "required_max_rear_axle_curvature_1pm cannot be negative"
        )
    if (
        abs(
            numeric["required_max_rear_axle_curvature_1pm"]
            - measured_max_curvature
        ) > 1e-9
    ):
        raise ParkingManoeuvreProofError(
            "required_max_rear_axle_curvature_1pm disagrees with rear poses"
        )
    if (
        abs(
            numeric["required_max_abs_normalized_steer"]
            - required_max_abs_steer
        ) > 1e-9
    ):
        raise ParkingManoeuvreProofError(
            "required_max_abs_normalized_steer disagrees with piecewise map inverse"
        )
    if (
        numeric["required_max_abs_normalized_steer"]
        > numeric["max_normalized_control_steer"] + 1e-9
    ):
        raise ParkingManoeuvreProofError(
            "parking manoeuvre exceeds the normalized-steer qualification envelope"
        )
    if not (
        0.0
        < numeric["max_normalized_control_steer"]
        <= PARKING_MAX_NORMALIZED_CONTROL_STEER
    ):
        raise ParkingManoeuvreProofError(
            "parking max_normalized_control_steer must be in (0,0.8]"
        )
    if proof["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT:
        raise ParkingManoeuvreProofError(
            "parking manoeuvre proof must use vehicle.lincoln.mkz_2020"
        )
    if abs(numeric["footprint_tolerance_m"] - PARKING_SLOT_TOLERANCE_M) > 1e-9:
        raise ParkingManoeuvreProofError(
            "parking footprint_tolerance_m must equal 0.2"
        )
    for field in ("full_footprint_legal_samples", "static_clearance_samples"):
        if _integer(proof[field], f"reverse_manoeuvre.proof.{field}", minimum=1) != pose_count:
            raise ParkingManoeuvreProofError(
                f"reverse_manoeuvre.proof.{field} must equal pose_count"
            )
    if _integer(
        proof["static_collision_samples"],
        "reverse_manoeuvre.proof.static_collision_samples",
    ) != 0:
        raise ParkingManoeuvreProofError(
            "parking manoeuvre static_collision_samples must equal zero"
        )
    if numeric["static_clearance_min_m"] < PARKING_STATIC_CLEARANCE_MIN_M - 1e-9:
        raise ParkingManoeuvreProofError(
            "parking manoeuvre static clearance must be at least 0.2m"
        )
    if proof["static_bbox_categories"] != PARKING_STATIC_BBOX_CATEGORIES:
        raise ParkingManoeuvreProofError(
            "parking static_bbox_categories must equal the frozen non-road "
            "CARLA level-label list"
        )
    _integer(
        proof["static_bbox_count"],
        "reverse_manoeuvre.proof.static_bbox_count",
        minimum=1,
    )
    if _SHA256.fullmatch(str(proof["static_bboxes_sha256"])) is None:
        raise ParkingManoeuvreProofError(
            "reverse_manoeuvre.proof.static_bboxes_sha256 must be lowercase SHA256"
        )
    if str(proof["town"]) != str(route.get("town", "")):
        raise ParkingManoeuvreProofError(
            "parking manoeuvre proof town mismatch"
        )
    for field in ("xodr_sha256", "lane_graph_sha256"):
        if _SHA256.fullmatch(str(proof[field])) is None:
            raise ParkingManoeuvreProofError(
                f"reverse_manoeuvre.proof.{field} must be lowercase SHA256"
            )

    registry_entry = registry_context["entry"]
    bbox_registry = registry_context["bbox_asset"]
    control_registry = registry_entry["control_model"]
    vehicle_registry = registry_entry["vehicle_geometry"]
    if (
        proof["method"] != registry.get("method")
        or proof["pose_reference"] != control_registry["pose_reference"]
        or proof["integration_reference"]
        != control_registry["integration_reference"]
        or proof["static_bbox_categories"] != bbox_registry["categories"]
        or int(proof["static_bbox_count"]) != int(bbox_registry["bbox_count"])
        or proof["static_bboxes_sha256"] != bbox_registry["bboxes_sha256"]
        or proof["xodr_sha256"] != registry_entry["xodr"]["sha256"]
        or proof["lane_graph_sha256"] != registry_entry["lane_graph"]["sha256"]
        or proof["vehicle_blueprint"] != vehicle_registry["vehicle_blueprint"]
    ):
        raise ParkingManoeuvreProofError(
            "parking manoeuvre proof disagrees with source registry"
        )
    for field in ("max_normalized_control_steer",):
        if abs(float(proof[field]) - float(control_registry[field])) > 1e-6:
            raise ParkingManoeuvreProofError(
                f"parking proof {field} disagrees with control registry"
            )

    vertical = proof["vertical_overlap_proof"]
    if not isinstance(vertical, Mapping) or set(vertical) != VERTICAL_OVERLAP_PROOF_KEYS:
        raise ParkingManoeuvreProofError(
            "vertical_overlap_proof keys must be exactly {}".format(
                sorted(VERTICAL_OVERLAP_PROOF_KEYS)
            )
        )
    if vertical["method"] != PARKING_VERTICAL_METHOD:
        raise ParkingManoeuvreProofError(
            "vertical_overlap_proof method is unsupported"
        )
    if is_v5:
        _strict_finite(
            vertical["vertical_padding_m"],
            "vertical_overlap_proof.vertical_padding_m",
        )
    padding = _finite(
        vertical["vertical_padding_m"],
        "vertical_overlap_proof.vertical_padding_m",
    )
    if abs(padding - PARKING_VERTICAL_PADDING_M) > 1e-9:
        raise ParkingManoeuvreProofError(
            "vertical_overlap_proof padding must equal 0.05m"
        )
    local_bbox_sha = _sha256(
        vertical["ego_local_bbox_sha256"],
        "vertical_overlap_proof.ego_local_bbox_sha256",
    )
    if local_bbox_sha != vehicle_registry["local_bbox_sha256"]:
        raise ParkingManoeuvreProofError(
            "vertical_overlap_proof ego bbox disagrees with registry"
        )
    vertical_pose_count = _integer(
        vertical["pose_count"], "vertical_overlap_proof.pose_count", minimum=5
    )
    vertical_bbox_count = _integer(
        vertical["static_bbox_count"],
        "vertical_overlap_proof.static_bbox_count",
        minimum=1,
    )
    checked_pairs = _integer(
        vertical["checked_pair_count"],
        "vertical_overlap_proof.checked_pair_count",
        minimum=1,
    )
    overlapping_pairs = _integer(
        vertical["vertically_overlapping_pair_count"],
        "vertical_overlap_proof.vertically_overlapping_pair_count",
    )
    disjoint_pairs = _integer(
        vertical["vertically_disjoint_pair_count"],
        "vertical_overlap_proof.vertically_disjoint_pair_count",
    )
    if (
        vertical_pose_count != pose_count
        or vertical_bbox_count != int(proof["static_bbox_count"])
        or checked_pairs != vertical_pose_count * vertical_bbox_count
        or overlapping_pairs + disjoint_pairs != checked_pairs
    ):
        raise ParkingManoeuvreProofError(
            "vertical_overlap_proof counts do not close"
        )
    overlap_hash = _sha256(
        vertical["overlap_index_sets_sha256"],
        "vertical_overlap_proof.overlap_index_sets_sha256",
    )
    bbox_payload = registry_context["bbox_payload"]
    if bbox_payload is None:
        raise ParkingManoeuvreProofError(
            "parking v4 requires verified bbox source payload"
        )
    overlap_index_sets = _vertical_overlap_index_sets(
        poses,
        bbox_payload["bboxes"],
        registry_context["local_bbox"],
        actor_reference_z_min_m=float(
            vehicle_registry["actor_reference_z_min_m"]
        ),
        actor_reference_z_max_m=float(
            vehicle_registry["actor_reference_z_max_m"]
        ),
    )
    recomputed_overlaps = sum(len(indices) for indices in overlap_index_sets)
    if (
        _parking_hash(
            overlap_index_sets, "vertical_overlap_proof.overlap_index_sets"
        ) != overlap_hash
        or recomputed_overlaps != overlapping_pairs
        or checked_pairs - recomputed_overlaps != disjoint_pairs
    ):
        raise ParkingManoeuvreProofError(
            "vertical_overlap_proof index hash/counts disagree with sources"
        )
    actual_clearance = _static_horizontal_clearance(
        poses,
        bbox_payload["bboxes"],
        overlap_index_sets,
        local_bbox=registry_context["local_bbox"],
    )
    claimed_clearance = numeric["static_clearance_min_m"]
    expected_clearance = (
        PARKING_STATIC_CLEARANCE_MIN_M
        if not math.isfinite(actual_clearance)
        else math.floor(actual_clearance * 1_000_000.0) / 1_000_000.0
    )
    if (
        expected_clearance < PARKING_STATIC_CLEARANCE_MIN_M - 1e-9
        or abs(claimed_clearance - expected_clearance) > 1e-6
    ):
        raise ParkingManoeuvreProofError(
            "parking v4 horizontal clearance disagrees with vertically "
            "overlapping bbox geometry"
        )

    ingress = None
    route_evidence = None
    if is_v5:
        map_context = _parking_v5_map_context(
            registry_context, str(route.get("town", ""))
        )
        ingress = _validate_v5_ingress(
            selected, raw_poses, raw_rear_poses, proof,
            approach, approach_proof, center, slot_heading,
            registry_context, map_context,
        )
        if require_oracle_suffix:
            route_evidence = _validate_v5_route_proof(
                route, selected, raw_poses, executable_poses, ingress,
                registry_context, map_context,
            )
            if suffix_start != route_evidence["explicit_suffix_start_index"]:
                raise ParkingManoeuvreProofError(
                    "parking v5 inferred suffix start differs from route proof"
                )

    return {
        "schema_version": manoeuvre_schema,
        "is_v5": is_v5,
        "approach_pose": approach,
        "approach_pose_proof": approach_proof,
        "poses": poses,
        "rear_axle_poses": rear_poses,
        "executable_poses": executable_poses,
        "proof": proof,
        "control_model": dict(control_registry),
        "normalized_steer_samples": [dict(item) for item in steer_samples],
        "transition_index": transition,
        "oracle_path_suffix_start": suffix_start,
        "slot_center": center,
        "slot_heading_deg": slot_heading,
        "v5_ingress": ingress,
        "v5_route_proof": route_evidence,
    }
