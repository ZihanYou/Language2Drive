from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from .frozen_paths import ProvenancePath as Path
from typing import Any

from .io import sha256_file, stable_hash
from .maps import _import_carla
from .parking import (
    BBOX_CATEGORIES as PARKING_BBOX_CATEGORIES,
    FOOTPRINT_LENGTH_M as PARKING_FOOTPRINT_LENGTH_M,
    FOOTPRINT_TOLERANCE_M as PARKING_FOOTPRINT_TOLERANCE_M,
    FOOTPRINT_WIDTH_M as PARKING_FOOTPRINT_WIDTH_M,
    FRONT_AXLE_X_M as PARKING_FRONT_AXLE_X_M,
    MANOEUVRE_SPACING_M as PARKING_MANOEUVRE_SPACING_M,
    MAX_NORMALIZED_CONTROL_STEER as PARKING_MAX_NORMALIZED_CONTROL_STEER,
    REAR_AXLE_X_M as PARKING_REAR_AXLE_X_M,
    VEHICLE_BLUEPRINT as PARKING_VEHICLE_BLUEPRINT,
    WHEELBASE_M as PARKING_WHEELBASE_M,
    _actor_pose_from_rear as _parking_actor_pose_from_rear,
    _normalized_steer_for_curvature as _parking_normalized_steer_for_curvature,
)

PREDICATES = {
    "enter_gate", "inside_region", "vehicle_footprint_inside_slot",
    "lane_relation", "speed_band", "headway_band", "passing_side",
    "actor_order", "progress", "dwell", "safe_legal",
}
OPERATORS = {
    "eventually", "always", "within", "until", "sequence", "if_else",
    "bind_reference",
}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SchemaError(ValueError):
    pass


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SchemaError(f"{path} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise SchemaError(f"{path} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise SchemaError(f"{path} must be <= {maximum}")
    return result


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or "__" in value:
        raise SchemaError(f"{path} must be a path-safe identifier without '__'")
    return value


def _polygon(value: Any, path: str) -> None:
    if not isinstance(value, list) or len(value) < 3:
        raise SchemaError(f"{path} must contain at least three vertices")
    vertices: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise SchemaError(f"{path}[{index}] must contain x and y")
        vertices.append((_number(point[0], f"{path}[{index}][0]"), _number(point[1], f"{path}[{index}][1]")))
    unique = set(vertices[:-1] if vertices[0] == vertices[-1] else vertices)
    if len(unique) < 3:
        raise SchemaError(f"{path} must contain at least three unique vertices")


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path} must be a non-empty string")
    return value


def _contract_contains(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contract_contains(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contract_contains(child, key) for child in value)
    return False


def _angle_error(value: float, target: float) -> float:
    return abs((float(value) - float(target) + 180.0) % 360.0 - 180.0)


def _parking_targets(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("type") == "parking_slot":
            result.append(value)
        else:
            for child in value.values():
                result.extend(_parking_targets(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_parking_targets(child))
    return result


def _selected_parking_target(target: Any, actor_setup: Any) -> dict[str, Any] | None:
    if not isinstance(target, dict):
        return None
    if target.get("type") == "parking_slot":
        return target
    if target.get("type") != "if_else":
        return None
    condition = target.get("condition")
    if not isinstance(condition, str) or not condition.startswith("occupied:"):
        return None
    slot = condition.split(":", 1)[1]
    occupancy = actor_setup.get("slot_occupancy") if isinstance(actor_setup, dict) else None
    occupied = occupancy.get(slot) if isinstance(occupancy, dict) else None
    if not isinstance(occupied, bool):
        raise SchemaError("parking if_else lacks a frozen boolean occupancy")
    selected = target.get("then" if occupied else "else")
    return selected if isinstance(selected, dict) and selected.get("type") == "parking_slot" else None


def _parking_registry_town(pcu: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and return the exact frozen registry entry used by proofs."""

    registry = pcu.get("_source_parking_registry")
    if not registry:
        return None
    if not isinstance(registry, dict) or set(registry) != {"schema_version", "method", "towns"}:
        raise SchemaError("parking_manoeuvre_registry root is not exact")
    if (registry["schema_version"] != "language2drive.parking_manoeuvre_registry.v5"
            or registry["method"] != "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v5"
            or not isinstance(registry["towns"], dict)):
        raise SchemaError("parking_manoeuvre_registry version/method is unsupported")
    town = registry["towns"].get("parkinglotfish-new3")
    expected_town = {
        "town", "bbox_asset", "xodr", "lane_graph", "control_model",
        "vehicle_geometry", "slot_collision_occupancy",
    }
    if not isinstance(town, dict) or set(town) != expected_town or town["town"] != "parkinglotfish-new3":
        raise SchemaError("parking_manoeuvre_registry town entry is not exact")
    bbox = town["bbox_asset"]
    bbox_keys = {
        "source_ref", "source_file_sha256", "schema_version", "bboxes_sha256", "bbox_count",
        "categories", "scope_statement", "excluded_scope", "quantization_decimals",
        "canonical_sort_key", "source_api",
    }
    if not isinstance(bbox, dict) or set(bbox) != bbox_keys:
        raise SchemaError("parking_manoeuvre_registry bbox_asset is not exact")
    if (bbox["schema_version"] != "language2drive.carla_level_bboxes.v1"
            or bbox["bbox_count"] != 409 or bbox["categories"] != PARKING_BBOX_CATEGORIES
            or bbox["scope_statement"] != (
                "Complete 2D ego footprints are checked against CARLA world.get_level_bbs "
                "for the frozen non-road semantic categories only; this is not a claim about "
                "unlabelled Unreal collision meshes. Live three-seed collision witnesses "
                "remain authoritative."
            )
            or bbox["excluded_scope"] != [
                "Roads", "RoadLines", "Ground", "Sidewalks", "Terrain", "Water", "Sky",
                "RailTrack", "Bridge", "Vehicles", "Pedestrians", "Dynamic", "Any", "None",
            ]
            or bbox["quantization_decimals"] != 6
            or bbox["canonical_sort_key"] != [
                "category", "center.x", "center.y", "center.z", "extent.x", "extent.y",
                "extent.z", "rotation.pitch", "rotation.yaw", "rotation.roll",
            ]
            or bbox["source_api"] != "carla.World.get_level_bbs(CityObjectLabel)"):
        raise SchemaError("parking_manoeuvre_registry bbox_asset frozen semantics mismatch")
    control = town["control_model"]
    if not isinstance(control, dict) or set(control) != {
        "schema_version", "pose_reference", "integration_reference",
        "vehicle_dynamics_source_ref", "vehicle_dynamics_source_file_sha256",
        "wheel_geometry_sha256", "normalized_steer_samples_sha256",
        "front_axle_x_m", "rear_axle_x_m", "wheelbase_m",
        "max_normalized_control_steer",
    }:
        raise SchemaError("parking_manoeuvre_registry control_model is not exact")
    if (
        control["schema_version"] != "language2drive.parking_axle_control_model.v4"
        or control["pose_reference"] != "carla_actor_transform"
        or control["integration_reference"] != "rear_axle_center"
    ):
        raise SchemaError("parking registry control-model frame/version mismatch")
    for field, expected in {
        "front_axle_x_m": PARKING_FRONT_AXLE_X_M,
        "rear_axle_x_m": PARKING_REAR_AXLE_X_M,
        "wheelbase_m": PARKING_WHEELBASE_M,
        "max_normalized_control_steer": PARKING_MAX_NORMALIZED_CONTROL_STEER,
    }.items():
        if abs(_number(control[field], f"parking registry control.{field}") - expected) > 1e-9:
            raise SchemaError(f"parking registry control.{field} mismatch")
    for field in ("xodr", "lane_graph"):
        if not isinstance(town[field], dict) or set(town[field]) != {"source_ref", "sha256"}:
            raise SchemaError(f"parking_manoeuvre_registry {field} is not exact")
    vehicle = town["vehicle_geometry"]
    vehicle_keys = {
        "vehicle_blueprint", "coordinate_frame", "local_bbox", "local_bbox_sha256",
        "source_method", "source_trace_ref", "source_trace_sha256", "source_trace_rows",
        "matching_bbox_rows", "actor_reference_z_min_m", "actor_reference_z_max_m",
    }
    if not isinstance(vehicle, dict) or set(vehicle) != vehicle_keys:
        raise SchemaError("parking_manoeuvre_registry vehicle_geometry is not exact")
    if (vehicle["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT
            or vehicle["coordinate_frame"] != "carla_actor_local"
            or vehicle["source_method"] != "constant_bbox_across_authoritative_trace_v1"
            or vehicle["local_bbox"] != {
                "location": {"x": -0.0063, "y": 0.0, "z": 0.7488},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "extent": {"x": 2.4462, "y": 0.9184, "z": 0.7451},
            }
            or vehicle["local_bbox_sha256"] != stable_hash(vehicle["local_bbox"])
            or vehicle["source_trace_rows"] != 490 or vehicle["matching_bbox_rows"] != 490
            or abs(float(vehicle["actor_reference_z_min_m"]) - (-0.0609)) > 1e-9
            or abs(float(vehicle["actor_reference_z_max_m"]) - 0.0095) > 1e-9):
        raise SchemaError("parking_manoeuvre_registry vehicle_geometry closure mismatch")
    occupancy = town["slot_collision_occupancy"]
    occupancy_keys = {
        "source_ref", "source_file_sha256", "schema_version", "method", "town",
        "map_name", "vehicle_blueprint", "spawn_z_m", "repeat_count", "slots_sha256",
        "slot_count", "clear_count", "blocked_count", "clear_slots", "blocked_slots",
        "clear_slots_sha256", "blocked_slots_sha256", "source_files", "scope_statement",
    }
    if not isinstance(occupancy, dict) or set(occupancy) != occupancy_keys:
        raise SchemaError("parking registry slot occupancy entry is not exact")
    occupancy_path = Path(occupancy["source_ref"])
    if (
        not occupancy_path.is_absolute() or not occupancy_path.is_file()
        or sha256_file(occupancy_path) != occupancy["source_file_sha256"]
        or occupancy["schema_version"] != "language2drive.parking_slot_collision_occupancy.v1"
        or occupancy["method"] != "carla_try_spawn_release_ego_at_xodr_slot_terminal_v1"
        or occupancy["town"] != "parkinglotfish-new3"
        or occupancy["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT
        or occupancy["spawn_z_m"] != 0.5
        or occupancy["repeat_count"] != 2
        or not isinstance(occupancy["scope_statement"], str)
    ):
        raise SchemaError("parking registry slot occupancy authority is stale")
    occupancy_payload = json.loads(occupancy_path.read_text(encoding="utf-8"))
    if (
        not isinstance(occupancy_payload, dict)
        or occupancy_payload.get("slots_sha256") != occupancy["slots_sha256"]
        or occupancy_payload.get("summary", {}).get("slot_count") != occupancy["slot_count"]
        or occupancy_payload.get("summary", {}).get("clear_count") != occupancy["clear_count"]
        or occupancy_payload.get("summary", {}).get("blocked_count") != occupancy["blocked_count"]
        or occupancy_payload.get("summary", {}).get("clear_slots_sha256") != occupancy["clear_slots_sha256"]
        or occupancy_payload.get("summary", {}).get("blocked_slots_sha256") != occupancy["blocked_slots_sha256"]
        or occupancy_payload.get("scope_statement") != occupancy["scope_statement"]
        or stable_hash(occupancy_payload.get("slots")) != occupancy["slots_sha256"]
        or occupancy["clear_slots"] != sorted(
            row.get("slot_number") for row in occupancy_payload.get("slots", [])
            if row.get("terminal_spawn_clear") is True
        )
        or occupancy["blocked_slots"] != sorted(
            row.get("slot_number") for row in occupancy_payload.get("slots", [])
            if row.get("terminal_spawn_clear") is False
        )
        or stable_hash(occupancy["clear_slots"]) != occupancy["clear_slots_sha256"]
        or stable_hash(occupancy["blocked_slots"]) != occupancy["blocked_slots_sha256"]
    ):
        raise SchemaError("parking registry slot occupancy payload closure mismatch")
    direct_sources: dict[str, str] = {}
    for path_field, hash_field, body in (
        ("source_ref", "source_file_sha256", bbox),
        ("source_ref", "sha256", town["xodr"]),
        ("source_ref", "sha256", town["lane_graph"]),
        ("vehicle_dynamics_source_ref", "vehicle_dynamics_source_file_sha256", control),
        ("source_trace_ref", "source_trace_sha256", vehicle),
    ):
        source = Path(body[path_field])
        if not source.is_absolute() or not source.is_file() or sha256_file(source) != body[hash_field]:
            raise SchemaError(f"parking registry source is missing/stale: {source}")
        direct_sources[str(source)] = body[hash_field]
    direct_sources[str(occupancy_path)] = occupancy["source_file_sha256"]
    asset = json.loads(Path(bbox["source_ref"]).read_text(encoding="utf-8"))
    if (asset.get("schema_version") != bbox["schema_version"]
            or asset.get("town") != town["town"]
            or asset.get("categories") != bbox["categories"]
            or asset.get("scope_statement") != bbox["scope_statement"]
            or asset.get("excluded_scope") != bbox["excluded_scope"]
            or asset.get("quantization_decimals") != bbox["quantization_decimals"]
            or asset.get("canonical_sort_key") != bbox["canonical_sort_key"]
            or asset.get("provenance", {}).get("source_api") != bbox["source_api"]
            or len(asset.get("bboxes", [])) != bbox["bbox_count"]
            or stable_hash(asset.get("bboxes")) != bbox["bboxes_sha256"]):
        raise SchemaError("parking registry canonical bbox list closure mismatch")
    dynamics_path = Path(control["vehicle_dynamics_source_ref"])
    dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
    wheel = dynamics.get("wheel_geometry")
    response = dynamics.get("normalized_steer_response")
    if (
        control["vehicle_dynamics_source_file_sha256"]
        != "031643aa33e5f234ea6ba4c87a403908aee7942e322802b4b1b0e545d38dde47"
        or stable_hash(dynamics)
        != "d94392a5c4118dcb8f8660461267cfa1bb44e5f7c9fb111b14057d862d948791"
        or control["wheel_geometry_sha256"]
        != "55fdf3b68124610b471141f39610ebbf5f21c79e9d53f1464dbd738227285dee"
        or control["normalized_steer_samples_sha256"]
        != "208e6f69bea0c31037256b266c785b855a697202c4cff1ffc70e70da95974052"
        or response.get("raw_sweeps_sha256")
        != "b5789ff819d19e4f9d6f36cd76b54a53ac001ad900e98793cf940a78465330ed"
        or
        dynamics.get("schema_version") != "language2drive.vehicle_dynamics_evidence.v1"
        or dynamics.get("vehicle_blueprint") != PARKING_VEHICLE_BLUEPRINT
        or dynamics.get("capture", {}).get("town") != town["town"]
        or not isinstance(wheel, dict)
        or not isinstance(response, dict)
        or stable_hash(wheel) != dynamics.get("wheel_geometry_sha256")
        or stable_hash(response.get("canonical_samples"))
        != response.get("canonical_samples_sha256")
        or control["wheel_geometry_sha256"] != dynamics.get("wheel_geometry_sha256")
        or control["normalized_steer_samples_sha256"]
        != response.get("canonical_samples_sha256")
    ):
        raise SchemaError("parking dynamics asset canonical hash/version mismatch")
    if (
        abs(float(wheel.get("front_axle_x_m", math.nan)) - PARKING_FRONT_AXLE_X_M) > 1e-9
        or abs(float(wheel.get("rear_axle_x_m", math.nan)) - PARKING_REAR_AXLE_X_M) > 1e-9
        or abs(float(wheel.get("wheelbase_m", math.nan)) - PARKING_WHEELBASE_M) > 1e-9
        or abs((float(wheel["front_axle_x_m"]) - float(wheel["rear_axle_x_m"]))
               - float(wheel["wheelbase_m"])) > 1e-9
    ):
        raise SchemaError("parking dynamics axle geometry does not close")
    wheels = wheel.get("wheels")
    if (
        not isinstance(wheels, list)
        or [item.get("name") for item in wheels]
        != ["front_left", "front_right", "rear_left", "rear_right"]
        or any(not isinstance(item.get("position_actor_local_m"), dict) for item in wheels)
        or any(abs(float(item["max_steer_angle_deg"])) > 1e-12 for item in wheels[2:])
        or any(float(item["max_steer_angle_deg"]) <= 0 for item in wheels[:2])
    ):
        raise SchemaError("parking dynamics per-wheel geometry is invalid")
    samples = response.get("canonical_samples")
    expected_commands = [round(-0.8 + index * 0.05, 2) for index in range(33)]
    if (
        not isinstance(samples, list) or len(samples) != 33
        or [float(item.get("normalized_steer", math.nan)) for item in samples]
        != expected_commands
        or response.get("command_min") != -0.8
        or response.get("command_max") != 0.8
        or response.get("command_step") != 0.05
        or response.get("hysteresis_max_deg") != 0.0
    ):
        raise SchemaError("parking dynamics steer response grid is not frozen")
    curvatures = [float(item["ackermann_center_curvature_1pm"]) for item in samples]
    if (
        any(right <= left for left, right in zip(curvatures, curvatures[1:]))
        or curvatures[16] != 0.0
        or any(abs(curvatures[index] + curvatures[-index - 1]) > 1e-12
               for index in range(16))
        or stable_hash(response.get("raw_sweeps")) != response.get("raw_sweeps_sha256")
    ):
        raise SchemaError("parking dynamics steer response monotonic/symmetry closure failed")
    raw = response.get("raw_sweeps", {})
    if (
        not isinstance(raw, dict) or set(raw) != {"ascending", "descending"}
        or len(raw["ascending"]) != 33 or len(raw["descending"]) != 33
        or [float(item["normalized_steer"]) for item in raw["ascending"]]
        != expected_commands
        or [float(item["normalized_steer"]) for item in raw["descending"]]
        != list(reversed(expected_commands))
    ):
        raise SchemaError("parking dynamics bidirectional raw sweep closure failed")
    nested_sources = dynamics.get("source_files")
    if not isinstance(nested_sources, dict) or len(nested_sources) < 5:
        raise SchemaError("parking dynamics source-files provenance is incomplete")
    top_sources = pcu.get("_source_files")
    if top_sources is not None and not isinstance(top_sources, dict):
        raise SchemaError("parking top-level source_files must be an object")
    occupancy_sources = occupancy["source_files"]
    if not isinstance(occupancy_sources, dict) or set(occupancy_sources) != {
        "slot_registry", "runtime_manifest", "runtime_provenance",
        "available_maps_witness", "capture_tool",
    }:
        raise SchemaError("parking slot occupancy nested source map is not exact")
    for item in occupancy_sources.values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise SchemaError("parking slot occupancy nested source shape differs")
    all_registry_sources = {
        **direct_sources, **nested_sources,
        **{str(item["path"]): str(item["sha256"]) for item in occupancy_sources.values()},
    }
    for source_ref, digest in all_registry_sources.items():
        source = Path(source_ref)
        if (
            not source.is_absolute() or not source.is_file()
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or sha256_file(source) != digest
            or isinstance(top_sources, dict) and top_sources.get(str(source)) != digest
        ):
            raise SchemaError(f"parking dynamics source is missing/stale: {source}")
    observed_z: list[float] = []
    matching = 0
    trace_rows = 0
    with Path(vehicle["source_trace_ref"]).open(encoding="utf-8") as stream:
        for raw in stream:
            trace_rows += 1
            try:
                row = json.loads(raw)
                ego = row.get("ego", row)
                matching += ego.get("bounding_box", ego.get("bbox")) == vehicle["local_bbox"]
                transform = ego.get("transform", {})
                location = transform.get("location", transform)
                observed_z.append(float(location["z"]))
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SchemaError("parking vehicle source trace has invalid ego geometry") from error
    if (trace_rows != vehicle["source_trace_rows"]
            or matching != vehicle["matching_bbox_rows"]
            or not observed_z
            or abs(min(observed_z) - float(vehicle["actor_reference_z_min_m"])) > 1e-6
            or abs(max(observed_z) - float(vehicle["actor_reference_z_max_m"])) > 1e-6):
        raise SchemaError("parking vehicle source trace disagrees with frozen registry geometry")
    return town


def _parking_overlap_index_sets(poses: list[dict[str, Any]], town: dict[str, Any]) -> list[list[int]]:
    asset = json.loads(Path(town["bbox_asset"]["source_ref"]).read_text(encoding="utf-8"))
    vehicle = town["vehicle_geometry"]
    local = vehicle["local_bbox"]

    def half(extent: dict[str, Any], rotation: dict[str, Any]) -> float:
        pitch = math.radians(float(rotation["pitch"])); roll = math.radians(float(rotation["roll"]))
        return (abs(-math.sin(pitch)) * float(extent["x"])
                + abs(math.cos(pitch) * math.sin(roll)) * float(extent["y"])
                + abs(math.cos(pitch) * math.cos(roll)) * float(extent["z"]))

    ego_half = half(local["extent"], local["rotation"])
    static_intervals = []
    for bbox in asset["bboxes"]:
        bbox_half = half(bbox["extent"], bbox["rotation"])
        center = float(bbox["center"]["z"])
        static_intervals.append((center - bbox_half, center + bbox_half))
    result = []
    for pose in poses:
        ego_min = (float(pose["z"]) + float(vehicle["actor_reference_z_min_m"])
                   + float(local["location"]["z"]) - ego_half)
        ego_max = (float(pose["z"]) + float(vehicle["actor_reference_z_max_m"])
                   + float(local["location"]["z"]) + ego_half)
        result.append([
            index for index, (static_min, static_max) in enumerate(static_intervals)
            if (ego_max + 0.05 >= static_min - 0.05 - 1e-9
                and static_max + 0.05 >= ego_min - 0.05 - 1e-9)
        ])
    return result


def _parking_static_horizontal_clearance(
    poses: list[dict[str, Any]], town: dict[str, Any], overlap_sets: list[list[int]],
) -> float:
    """Recompute the v3 complete-OBB horizontal clearance exactly."""

    try:
        from shapely.geometry import Polygon
    except ImportError as error:
        raise SchemaError("parking v3 schema validation requires Shapely") from error
    asset = json.loads(Path(town["bbox_asset"]["source_ref"]).read_text(encoding="utf-8"))
    local_bbox = town["vehicle_geometry"]["local_bbox"]

    def matrix(rotation: dict[str, Any]) -> list[list[float]]:
        pitch = math.radians(float(rotation["pitch"])); yaw = math.radians(float(rotation["yaw"]))
        roll = math.radians(float(rotation["roll"]))
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    def matvec(value: list[list[float]], vector: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(sum(value[row][column] * vector[column] for column in range(3))
                     for row in range(3))  # type: ignore[return-value]

    def projected(center: dict[str, Any], extent: dict[str, Any], rotation: dict[str, Any]) -> Any:
        transform = matrix(rotation); corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = matvec(transform, (
                        sx * float(extent["x"]), sy * float(extent["y"]), sz * float(extent["z"]),
                    ))
                    corners.append((float(center["x"]) + offset[0], float(center["y"]) + offset[1]))
        return Polygon(corners).convex_hull

    local_transform = matrix(local_bbox["rotation"])
    local_location = tuple(float(local_bbox["location"][axis]) for axis in ("x", "y", "z"))

    def ego_polygon(pose: dict[str, Any]) -> Any:
        yaw = math.radians(float(pose["yaw_deg"])); cosine, sine = math.cos(yaw), math.sin(yaw)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner = matvec(local_transform, (
                        sx * float(local_bbox["extent"]["x"]),
                        sy * float(local_bbox["extent"]["y"]),
                        sz * float(local_bbox["extent"]["z"]),
                    ))
                    local_x, local_y = local_location[0] + corner[0], local_location[1] + corner[1]
                    corners.append((float(pose["x"]) + cosine * local_x - sine * local_y,
                                    float(pose["y"]) + sine * local_x + cosine * local_y))
        return Polygon(corners).convex_hull

    cached: dict[int, Any] = {}; minimum = math.inf
    for pose, indices in zip(poses, overlap_sets):
        ego = ego_polygon(pose)
        for index in indices:
            if index not in cached:
                bbox = asset["bboxes"][index]
                cached[index] = projected(bbox["center"], bbox["extent"], bbox["rotation"])
            minimum = min(minimum, float(ego.distance(cached[index])))
    return minimum


_PARKING_V5_TOPOLOGY_KEYS = {
    "schema_version", "method", "pose_index_space", "first_pose_index",
    "last_pose_index", "samples", "samples_sha256",
    "canonical_waypoint_transforms_sha256", "runs", "transitions",
}
_PARKING_V5_SAMPLE_KEYS = {"pose_index", "road_id", "section_id", "lane_id", "s_m"}
_PARKING_V5_RUN_KEYS = {
    "first_pose_index", "last_pose_index", "lane_graph_id", "travel_s_direction",
}
_PARKING_V5_TRANSITION_KEYS = {"from_run_index", "to_run_index"}
_PARKING_V5_ROUTE_KEYS = {
    "schema_version", "method", "oracle_path_sha256", "ordinary_tail_start_index",
    "explicit_suffix_start_index", "ordinary_tail_sha256", "explicit_suffix_sha256",
    "target_prelead_sample_sha256", "ordinary_tail_topology_proof",
}


def _strict_index(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaError(f"{path} must be a nonnegative strict integer")
    return value


def _strict_integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaError(f"{path} must be a strict integer")
    return value


def _canonical_yaw(value: Any) -> float:
    result = round((float(value) + 180.0) % 360.0 - 180.0, 3)
    return 0.0 if result == 0.0 else result


def _parking_lane_graph_index(town: dict[str, Any]) -> tuple[
    dict[tuple[int, int, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]
]:
    graph = json.loads(Path(town["lane_graph"]["source_ref"]).read_text(encoding="utf-8"))
    by_key = {
        (int(row["road_id"]), int(row["section_id"]), int(row["lane_id"])): row
        for row in graph["lanes"]
    }
    by_id = {str(row["id"]): row for row in graph["lanes"]}
    directions = {}
    for identifier, row in by_id.items():
        samples = row.get("samples", [])
        if len(samples) < 2:
            raise SchemaError(f"parking lane {identifier} has no direction samples")
        delta = float(samples[-1]["s"]) - float(samples[0]["s"])
        if abs(delta) <= 1e-9:
            raise SchemaError(f"parking lane {identifier} has ambiguous s direction")
        directions[identifier] = 1 if delta > 0 else -1
    return by_key, by_id, directions


def _parking_offline_map(town: dict[str, Any]) -> tuple[Any, Any]:
    carla = _import_carla()
    return carla, carla.Map(
        str(town["town"]), Path(town["xodr"]["source_ref"]).read_text(encoding="utf-8"),
    )


def _parking_v5_topology(
    proof: Any, poses: list[dict[str, Any]], expected_space: str,
    expected_first: int, expected_last: int, path: str, town: dict[str, Any],
    *, require_raw_transform: bool,
) -> dict[str, Any]:
    """Replay the exact minimal stored-XODR topology contract."""

    if not isinstance(proof, dict) or set(proof) != _PARKING_V5_TOPOLOGY_KEYS:
        raise SchemaError(f"{path} topology exact10 keys differ")
    if (proof["schema_version"] != "language2drive.parking_v5_stored_xodr_topology.v1"
            or proof["method"] != "stored_xodr_construction_waypoint_sequence_v1"
            or proof["pose_index_space"] != expected_space):
        raise SchemaError(f"{path} topology schema/method/index-space differs")
    first = _strict_index(proof["first_pose_index"], f"{path}.first_pose_index")
    last = _strict_index(proof["last_pose_index"], f"{path}.last_pose_index")
    if first != expected_first or last != expected_last or last >= len(poses):
        raise SchemaError(f"{path} topology range differs")
    samples = proof["samples"]
    if not isinstance(samples, list) or len(samples) != last - first + 1:
        raise SchemaError(f"{path} topology sample inventory differs")
    if stable_hash(samples) != proof["samples_sha256"]:
        raise SchemaError(f"{path} topology sample hash differs")
    for field in ("samples_sha256", "canonical_waypoint_transforms_sha256"):
        if not isinstance(proof[field], str) or re.fullmatch(r"[0-9a-f]{64}", proof[field]) is None:
            raise SchemaError(f"{path}.{field} is not lowercase SHA256")
    by_key, by_id, directions = _parking_lane_graph_index(town)
    carla, carla_map = _parking_offline_map(town)
    transforms: list[dict[str, Any]] = []
    graph_ids: list[str] = []
    for expected_index, sample in enumerate(samples, first):
        if not isinstance(sample, dict) or set(sample) != _PARKING_V5_SAMPLE_KEYS:
            raise SchemaError(f"{path}.samples[{expected_index}] exact5 keys differ")
        _strict_index(sample["pose_index"], f"{path}.samples[{expected_index}].pose_index")
        for field in ("road_id", "section_id", "lane_id"):
            _strict_integer(sample[field], f"{path}.samples[{expected_index}].{field}")
        if sample["pose_index"] != expected_index or sample["lane_id"] == 0:
            raise SchemaError(f"{path}.samples[{expected_index}] index/lane differs")
        s_m = _number(sample["s_m"], f"{path}.samples[{expected_index}].s_m")
        waypoint = carla_map.get_waypoint_xodr(
            int(sample["road_id"]), int(sample["lane_id"]), s_m,
        )
        if waypoint is None or int(waypoint.section_id) != int(sample["section_id"]):
            raise SchemaError(f"{path}.samples[{expected_index}] cannot reconstruct")
        key = (int(sample["road_id"]), int(sample["section_id"]), int(sample["lane_id"]))
        graph = by_key.get(key)
        if graph is None:
            raise SchemaError(f"{path}.samples[{expected_index}] lane absent from graph")
        graph_ids.append(str(graph["id"]))
        transform = {
            "pose_index": expected_index,
            "x": round(float(waypoint.transform.location.x), 4),
            "y": round(float(waypoint.transform.location.y), 4),
            "z": round(float(waypoint.transform.location.z), 4),
            "yaw_deg": _canonical_yaw(waypoint.transform.rotation.yaw),
        }
        transforms.append(transform)
        pose = poses[expected_index]
        dx = float(pose["x"]) - float(waypoint.transform.location.x)
        dy = float(pose["y"]) - float(waypoint.transform.location.y)
        yaw = math.radians(float(waypoint.transform.rotation.yaw))
        along = abs(dx * math.cos(yaw) + dy * math.sin(yaw))
        vertical = abs(float(pose["z"]) - float(waypoint.transform.location.z))
        if along > 0.1001 + 1e-9 or vertical > 0.1001 + 1e-9:
            raise SchemaError(f"{path}.samples[{expected_index}] pose residual exceeds v5 bound")
        public = {name: pose[name] for name in ("x", "y", "z", "yaw_deg")}
        canonical = {name: transform[name] for name in ("x", "y", "z", "yaw_deg")}
        if (require_raw_transform or (expected_space == "reverse_manoeuvre.poses" and expected_index == 0)) \
                and public != canonical:
            raise SchemaError(f"{path}.samples[{expected_index}] is not raw canonical transform")
        membership = None
        location = carla.Location(x=float(pose["x"]), y=float(pose["y"]), z=float(pose["z"]))
        for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
            membership = carla_map.get_waypoint(location, False, lane_type)
            if membership is not None:
                break
        if membership is None:
            raise SchemaError(f"{path}.samples[{expected_index}] lacks exact lane membership")
    if stable_hash(transforms) != proof["canonical_waypoint_transforms_sha256"]:
        raise SchemaError(f"{path} canonical waypoint transform hash differs")

    runs = proof["runs"]
    if not isinstance(runs, list) or not runs:
        raise SchemaError(f"{path}.runs is empty")
    cursor = first
    for run_index, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != _PARKING_V5_RUN_KEYS:
            raise SchemaError(f"{path}.runs[{run_index}] exact4 keys differ")
        run_first = _strict_index(run["first_pose_index"], f"{path}.runs[{run_index}].first")
        run_last = _strict_index(run["last_pose_index"], f"{path}.runs[{run_index}].last")
        direction = _strict_integer(
            run["travel_s_direction"], f"{path}.runs[{run_index}].direction",
        )
        if run_first != cursor or run_last < run_first or run_last > last \
                or direction not in (-1, 1):
            raise SchemaError(f"{path}.runs[{run_index}] range/direction differs")
        lane_graph_id = run["lane_graph_id"]
        if not isinstance(lane_graph_id, str) or lane_graph_id not in by_id \
                or lane_graph_id != graph_ids[run_first - first] \
                or any(value != lane_graph_id for value in graph_ids[run_first - first:run_last - first + 1]) \
                or run["travel_s_direction"] != directions[lane_graph_id]:
            raise SchemaError(f"{path}.runs[{run_index}] lane binding differs")
        selected = samples[run_first - first:run_last - first + 1]
        for left, right in zip(selected, selected[1:]):
            if (float(right["s_m"]) - float(left["s_m"])) * directions[lane_graph_id] <= 1e-8:
                raise SchemaError(f"{path}.runs[{run_index}] reverses s")
        cursor = run_last + 1
    if cursor != last + 1:
        raise SchemaError(f"{path}.runs do not cover the topology range")
    transitions = proof["transitions"]
    if not isinstance(transitions, list) or len(transitions) != len(runs) - 1:
        raise SchemaError(f"{path}.transitions inventory differs")
    for index, transition in enumerate(transitions):
        if (not isinstance(transition, dict) or set(transition) != _PARKING_V5_TRANSITION_KEYS
                or _strict_index(transition["from_run_index"], f"{path}.transition.from") != index
                or _strict_index(transition["to_run_index"], f"{path}.transition.to") != index + 1
                or runs[index + 1]["lane_graph_id"]
                not in by_id[runs[index]["lane_graph_id"]].get("successors", [])):
            raise SchemaError(f"{path}.transitions[{index}] is not a direct successor")
    return {"samples": samples, "transforms": transforms, "carla_map": carla_map, "carla": carla}


def _parking_v5_compact_prefix(
    oracle_path: list[dict[str, Any]], suffix_start: int, path: str, town: dict[str, Any],
) -> None:
    from .generator import _validate_directed_polyline

    carla, carla_map = _parking_offline_map(town)
    ordinary = oracle_path[:suffix_start + 1]
    for index, (first, second) in enumerate(zip(ordinary, ordinary[1:])):
        if math.dist(
            tuple(float(first[name]) for name in ("x", "y", "z")),
            tuple(float(second[name]) for name in ("x", "y", "z")),
        ) > 4.0 + 1e-9:
            raise SchemaError(f"{path} compact ordinary gap exceeds 4m at {index}")
    for index, pose in enumerate(ordinary):
        location = carla.Location(x=float(pose["x"]), y=float(pose["y"]), z=float(pose["z"]))
        if not any(carla_map.get_waypoint(location, False, lane_type) is not None
                   for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking)):
            raise SchemaError(f"{path} compact ordinary pose {index} lacks exact membership")
    directed = _validate_directed_polyline(
        carla_map, ordinary, sample_spacing_m=0.1, maximum_heading_error_deg=90.0,
    )
    if not directed["valid"]:
        raise SchemaError(f"{path} compact ordinary directed replay failed: {directed}")
    for index, pose in enumerate(ordinary[:-1]):
        following = next((value for value in ordinary[index + 1:] if math.dist(
            tuple(float(pose[name]) for name in ("x", "y", "z")),
            tuple(float(value[name]) for name in ("x", "y", "z")),
        ) > 0.01 + 1e-9), None)
        if following is None:
            continue
        tangent = math.degrees(math.atan2(
            float(following["y"]) - float(pose["y"]),
            float(following["x"]) - float(pose["x"]),
        ))
        if _angle_error(float(pose["yaw_deg"]), tangent) > 3.0 + 1e-9:
            raise SchemaError(f"{path} ordinary serialized yaw differs at {index}")


def _parking_v5_route_proof(
    arm: dict[str, Any], selected_target: dict[str, Any], path: str,
    town: dict[str, Any],
) -> None:
    proof = arm.get("parking_v5_route_proof")
    if not isinstance(proof, dict) or set(proof) != _PARKING_V5_ROUTE_KEYS:
        raise SchemaError(f"{path}.parking_v5_route_proof exact9 keys differ")
    if (proof["schema_version"] != "language2drive.parking_v5_route_proof.v1"
            or proof["method"] != "oracle_path_stored_xodr_tail_exact_suffix_v1"):
        raise SchemaError(f"{path}.parking_v5_route_proof schema/method differs")
    oracle = arm["oracle_path"]
    for index, pose in enumerate(oracle):
        if not isinstance(pose, dict) or set(pose) != {"x", "y", "z", "yaw_deg"}:
            raise SchemaError(f"{path}.oracle_path[{index}] exact4 keys differ")
        for field in ("x", "y", "z", "yaw_deg"):
            _number(pose[field], f"{path}.oracle_path[{index}].{field}")
        if (float(pose["x"]) != round(float(pose["x"]), 4)
                or float(pose["y"]) != round(float(pose["y"]), 4)
                or float(pose["z"]) != round(float(pose["z"]), 4)
                or float(pose["yaw_deg"]) != _canonical_yaw(pose["yaw_deg"])):
            raise SchemaError(f"{path}.oracle_path[{index}] precision/canonical yaw differs")
    for field in (
        "oracle_path_sha256", "ordinary_tail_sha256", "explicit_suffix_sha256",
        "target_prelead_sample_sha256",
    ):
        if not isinstance(proof[field], str) or re.fullmatch(r"[0-9a-f]{64}", proof[field]) is None:
            raise SchemaError(f"{path}.parking_v5_route_proof.{field} is not SHA256")
    if proof["oracle_path_sha256"] != stable_hash(oracle):
        raise SchemaError(f"{path}.parking_v5_route_proof whole-path hash differs")
    tail_start = _strict_index(proof["ordinary_tail_start_index"], f"{path}.tail_start")
    suffix_start = _strict_index(proof["explicit_suffix_start_index"], f"{path}.suffix_start")
    if not 0 < tail_start < suffix_start < len(oracle):
        raise SchemaError(f"{path}.parking_v5_route_proof index order differs")
    manoeuvre = selected_target["reverse_manoeuvre"]
    collapsed = []
    for pose in manoeuvre["poses"]:
        public = {name: pose[name] for name in ("x", "y", "z", "yaw_deg")}
        if not collapsed or public != collapsed[-1]:
            collapsed.append(public)
    if oracle[suffix_start:] != collapsed:
        raise SchemaError(f"{path}.parking_v5_route_proof explicit suffix differs")
    if (proof["ordinary_tail_sha256"] != stable_hash(oracle[tail_start:suffix_start + 1])
            or proof["explicit_suffix_sha256"] != stable_hash(oracle[suffix_start:])):
        raise SchemaError(f"{path}.parking_v5_route_proof suffix hashes differ")
    target_topology = manoeuvre["proof"]["construction_topology_proof"]
    if proof["target_prelead_sample_sha256"] != stable_hash(target_topology["samples"][0]):
        raise SchemaError(f"{path}.parking_v5_route_proof target prelead binding differs")
    target_replay = _parking_v5_topology(
        target_topology, manoeuvre["poses"], "reverse_manoeuvre.poses", 0,
        int(manoeuvre["proof"]["slot_axis_approach_index"]), f"{path}.target_topology",
        town, require_raw_transform=False,
    )
    tail_replay = _parking_v5_topology(
        proof["ordinary_tail_topology_proof"], oracle, "arm.oracle_path",
        tail_start, suffix_start - 1, f"{path}.ordinary_tail_topology", town,
        require_raw_transform=True,
    )
    ordinary_last = tail_replay["samples"][-1]
    target_first = target_replay["samples"][0]
    by_key, by_id, directions = _parking_lane_graph_index(town)
    last_key = (ordinary_last["road_id"], ordinary_last["section_id"], ordinary_last["lane_id"])
    first_key = (target_first["road_id"], target_first["section_id"], target_first["lane_id"])
    last_graph, first_graph = by_key[last_key]["id"], by_key[first_key]["id"]
    if last_graph == first_graph:
        if (float(target_first["s_m"]) - float(ordinary_last["s_m"])) * directions[last_graph] <= 1e-8:
            raise SchemaError(f"{path} tail-to-prelead s is not forward")
    elif first_graph not in by_id[last_graph].get("successors", []):
        raise SchemaError(f"{path} tail-to-prelead is not a direct successor")
    cross_gap = math.dist(
        tuple(float(oracle[suffix_start - 1][name]) for name in ("x", "y", "z")),
        tuple(float(oracle[suffix_start][name]) for name in ("x", "y", "z")),
    )
    if not 1e-8 < cross_gap <= 0.1 + 1e-9:
        raise SchemaError(f"{path} tail-to-prelead gap differs")
    cross_chord = math.degrees(math.atan2(
        float(oracle[suffix_start]["y"]) - float(oracle[suffix_start - 1]["y"]),
        float(oracle[suffix_start]["x"]) - float(oracle[suffix_start - 1]["x"]),
    ))
    last_waypoint = tail_replay["carla_map"].get_waypoint_xodr(
        int(ordinary_last["road_id"]), int(ordinary_last["lane_id"]),
        float(ordinary_last["s_m"]),
    )
    target_waypoint = target_replay["carla_map"].get_waypoint_xodr(
        int(target_first["road_id"]), int(target_first["lane_id"]),
        float(target_first["s_m"]),
    )
    if (last_waypoint is None or target_waypoint is None
            or max(_angle_error(cross_chord, last_waypoint.transform.rotation.yaw),
                   _angle_error(cross_chord, target_waypoint.transform.rotation.yaw)) > 3.0 + 1e-9):
        raise SchemaError(f"{path} tail-to-prelead chord/yaw differs")
    dense = oracle[tail_start:suffix_start + 1]
    gaps = [math.dist(
        tuple(float(first[name]) for name in ("x", "y", "z")),
        tuple(float(second[name]) for name in ("x", "y", "z")),
    ) for first, second in zip(dense, dense[1:])]
    if any(not 1e-8 < value <= 0.1 + 1e-9 for value in gaps) or sum(gaps) < 7.0 - 1e-9:
        raise SchemaError(f"{path} dense ordinary tail spacing/length differs")
    prefix_gap = math.dist(
        tuple(float(oracle[tail_start - 1][name]) for name in ("x", "y", "z")),
        tuple(float(oracle[tail_start][name]) for name in ("x", "y", "z")),
    )
    if prefix_gap > 1.0 + 1e-9:
        raise SchemaError(f"{path} compact-prefix splice exceeds 1m")

    def nonzero_chord(values: list[dict[str, Any]], *, reverse: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        pairs = list(zip(values, values[1:]))
        if reverse:
            pairs.reverse()
        for first, second in pairs:
            if math.dist(
                tuple(float(first[name]) for name in ("x", "y", "z")),
                tuple(float(second[name]) for name in ("x", "y", "z")),
            ) > 1e-6:
                return first, second
        raise SchemaError(f"{path} splice has no nonzero chord")

    def chord_yaw(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
        first, second = pair
        return math.degrees(math.atan2(
            float(second["y"]) - float(first["y"]),
            float(second["x"]) - float(first["x"]),
        ))

    prefix_seam = _angle_error(
        chord_yaw(nonzero_chord(oracle[:tail_start + 1], reverse=True)),
        chord_yaw(nonzero_chord(dense)),
    )
    explicit_seam = _angle_error(
        cross_chord, chord_yaw(nonzero_chord(oracle[suffix_start:])),
    )
    if prefix_seam > 3.0 + 1e-9 or explicit_seam > 3.0 + 1e-9:
        raise SchemaError(f"{path} protected parking-v5 splice yaw differs")

    _parking_v5_compact_prefix(oracle, suffix_start, path, town)
    measured = sum(math.dist(
        tuple(float(first[name]) for name in ("x", "y", "z")),
        tuple(float(second[name]) for name in ("x", "y", "z")),
    ) for first, second in zip(oracle, oracle[1:]))
    if abs(_number(arm.get("intended_path_length_m"), f"{path}.intended_path_length_m")
           - round(measured, 3)) > 1e-9:
        raise SchemaError(f"{path}.intended_path_length_m is not final once-counted length")

    # Full OBB/exact/static authority begins at the protected dense tail.  The
    # compact upstream GRP prefix remains directed/exact-lane checked above
    # and is guarded dynamically by the ordinary collision controller.
    from . import parking as parking_module
    geometry = parking_module._geometry(Path(__file__).resolve().parents[2])
    dense_forward = [dict(pose, gear="forward") for pose in dense]
    overlap = parking_module._overlap_index_sets(geometry, dense_forward)
    if any(
        parking_module._footprint(
            geometry, float(pose["x"]), float(pose["y"]),
            math.radians(float(pose["yaw_deg"])),
        ).distance(geometry["obstacles"][bbox_index]) < 0.2 - 1e-9
        for pose, indices in zip(dense_forward, overlap)
        for bbox_index in indices
    ):
        raise SchemaError(f"{path} dense parking tail static clearance differs")
    remote_polygon = [
        [1_000_000.0, 1_000_000.0], [1_000_001.0, 1_000_000.0],
        [1_000_000.0, 1_000_001.0], [1_000_000.0, 1_000_000.0],
    ]
    if not parking_module._exact_map_footprint_audit(
        geometry, tail_replay["carla_map"], dense_forward, remote_polygon,
    ):
        raise SchemaError(f"{path} dense parking tail leaves exact lane surface")


def _validate_parking_target(
    target: dict[str, Any], oracle_path: Any, path: str,
    registry_town: dict[str, Any] | None,
) -> None:
    """Fail closed on every active and inactive conditional parking branch."""

    if target.get("type") == "parking_slot" and registry_town is not None:
        slot_number = target.get("slot_number")
        occupancy = registry_town["slot_collision_occupancy"]
        if not isinstance(slot_number, str) or slot_number not in occupancy["clear_slots"]:
            raise SchemaError(
                f"{path}.slot_number is not terminal-spawn-clear in the frozen CARLA occupancy asset"
            )

    approach = target.get("approach_pose")
    if not isinstance(approach, dict) or set(approach) != {"x", "y", "z", "yaw_deg"}:
        raise SchemaError(f"{path}.approach_pose requires exactly x/y/z/yaw_deg")
    for field in approach:
        _number(approach[field], f"{path}.approach_pose.{field}")
    approach_proof = target.get("approach_pose_proof")
    expected_approach = {
        "method", "distance_from_slot_center_m", "heading_error_deg", "lane_id",
        "lane_yaw_deg", "road_id", "section_id", "reverse",
    }
    if not isinstance(approach_proof, dict) or set(approach_proof) != expected_approach:
        raise SchemaError(f"{path}.approach_pose_proof keys must be {sorted(expected_approach)}")
    if approach_proof["method"] != "slot_axis_lane_intersection_v5":
        raise SchemaError(f"{path}.approach_pose_proof method is unsupported")
    for field in ("road_id", "section_id", "lane_id"):
        if not isinstance(approach_proof[field], int) or isinstance(approach_proof[field], bool):
            raise SchemaError(f"{path}.approach_pose_proof.{field} must be integer")
    if approach_proof["lane_id"] == 0:
        raise SchemaError(f"{path}.approach_pose_proof.lane_id cannot be zero")
    lane_yaw = _number(approach_proof["lane_yaw_deg"], f"{path}.approach_pose_proof.lane_yaw_deg")
    if _angle_error(float(approach["yaw_deg"]), lane_yaw) > 1.0:
        raise SchemaError(f"{path}.approach_pose yaw disagrees with physical lane yaw")
    heading = _number(target.get("heading_deg"), f"{path}.heading_deg")
    heading_error = _number(approach_proof["heading_error_deg"], f"{path}.approach_pose_proof.heading_error_deg")
    if abs(heading_error - _angle_error(lane_yaw, heading)) > 1e-3:
        raise SchemaError(f"{path}.approach_pose_proof heading error is inconsistent")
    polygon = target.get("polygon"); _polygon(polygon, f"{path}.polygon")
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    center = (sum(float(point[0]) for point in vertices) / len(vertices),
              sum(float(point[1]) for point in vertices) / len(vertices))
    distance_claim = _number(
        approach_proof["distance_from_slot_center_m"],
        f"{path}.approach_pose_proof.distance",
    )
    heading_radians = math.radians(heading)
    dx = float(approach["x"]) - center[0]
    dy = float(approach["y"]) - center[1]
    along = dx * math.cos(heading_radians) + dy * math.sin(heading_radians)
    cross = dx * (-math.sin(heading_radians)) + dy * math.cos(heading_radians)
    if (along <= 0.0 or abs(cross) > 0.001 + 1e-9
            or abs(distance_claim - along) > 1e-3):
        raise SchemaError(f"{path}.approach_pose_proof positive slot-axis ray is inconsistent")

    reverse = approach_proof["reverse"]
    reverse_keys = {"method", "own_slot_tolerance_m", "sample_spacing_max_m", "samples"}
    if (not isinstance(reverse, dict) or set(reverse) != reverse_keys
            or reverse["method"] != "audited_slot_axis_to_reverse_manoeuvre_v5"):
        raise SchemaError(f"{path}.approach_pose_proof.reverse is not frozen v5")
    if abs(_number(reverse["own_slot_tolerance_m"], f"{path}.reverse.tolerance") - PARKING_FOOTPRINT_TOLERANCE_M) > 1e-9:
        raise SchemaError(f"{path}.reverse own-slot tolerance must equal 0.2")
    if abs(_number(reverse["sample_spacing_max_m"], f"{path}.reverse.spacing") - PARKING_MANOEUVRE_SPACING_M) > 1e-9:
        raise SchemaError(f"{path}.reverse spacing must equal 0.1")

    manoeuvre = target.get("reverse_manoeuvre")
    if not isinstance(manoeuvre, dict) or set(manoeuvre) != {
        "schema_version", "poses", "rear_axle_poses", "proof",
    }:
        raise SchemaError(f"{path}.reverse_manoeuvre keys are invalid")
    if manoeuvre["schema_version"] != "language2drive.parking_reverse_manoeuvre.v5":
        raise SchemaError(f"{path}.reverse_manoeuvre schema is unsupported")
    poses = manoeuvre["poses"]
    rear_poses = manoeuvre["rear_axle_poses"]
    if (not isinstance(poses, list) or not isinstance(rear_poses, list)
            or len(poses) < 5 or len(rear_poses) != len(poses)):
        raise SchemaError(f"{path}.reverse_manoeuvre requires at least five poses")
    expected_pose = {"x", "y", "z", "yaw_deg", "gear"}
    for sequence_name, sequence in (("poses", poses), ("rear_axle_poses", rear_poses)):
        for index, pose in enumerate(sequence):
            if not isinstance(pose, dict) or set(pose) != expected_pose or pose["gear"] not in {"forward", "reverse"}:
                raise SchemaError(f"{path}.reverse_manoeuvre.{sequence_name}[{index}] is invalid")
            for field in ("x", "y", "z", "yaw_deg"):
                _number(pose[field], f"{path}.reverse_manoeuvre.{sequence_name}[{index}].{field}")
            if not -180.0 <= float(pose["yaw_deg"]) < 180.0:
                raise SchemaError(
                    f"{path}.reverse_manoeuvre.{sequence_name}[{index}].yaw_deg is not canonical"
                )
            if sequence_name == "rear_axle_poses" and (
                pose["x"] != round(float(pose["x"]), 6)
                or pose["y"] != round(float(pose["y"]), 6)
                or pose["yaw_deg"] != round(float(pose["yaw_deg"]), 3)
            ):
                raise SchemaError(f"{path}.rear_axle_poses[{index}] precision is not frozen")
    for index, (actor_pose, rear_pose) in enumerate(zip(poses, rear_poses)):
        if _parking_actor_pose_from_rear(rear_pose) != actor_pose:
            raise SchemaError(f"{path}.poses[{index}] is not exact rear-to-actor transform")
    # These are serialized controller states, not noisy measurements.  Raw
    # JSON equality is part of the v3 closure so even a sub-micrometre drift
    # cannot pass source validation and then disagree with the runtime loader.
    transitions = [index for index, pair in enumerate(zip(rear_poses, rear_poses[1:])) if pair[0]["gear"] != pair[1]["gear"]]
    if len(transitions) != 1 or poses[transitions[0]]["gear"] != "forward" or poses[transitions[0] + 1]["gear"] != "reverse":
        raise SchemaError(f"{path}.reverse_manoeuvre requires exactly one forward-to-reverse transition")
    maximum_curvature = 0.0
    required_max_steer = 0.0
    steer_samples = json.loads(
        Path(registry_town["control_model"]["vehicle_dynamics_source_ref"])
        .read_text(encoding="utf-8")
    )["normalized_steer_response"]["canonical_samples"] if registry_town else []
    for index, ((first, second), (rear_first, rear_second)) in enumerate(zip(
        zip(poses, poses[1:]), zip(rear_poses, rear_poses[1:]),
    )):
        actor_separation = math.dist(tuple(float(first[a]) for a in ("x", "y", "z")),
                                     tuple(float(second[a]) for a in ("x", "y", "z")))
        rear_separation = math.dist(tuple(float(rear_first[a]) for a in ("x", "y", "z")),
                                    tuple(float(rear_second[a]) for a in ("x", "y", "z")))
        signed_yaw_step = ((float(rear_second["yaw_deg"]) - float(rear_first["yaw_deg"])
                            + 180.0) % 360.0 - 180.0)
        yaw_step = abs(signed_yaw_step)
        if index == transitions[0]:
            if (any(first[field] != second[field] for field in ("x", "y", "z", "yaw_deg"))
                    or any(rear_first[field] != rear_second[field]
                           for field in ("x", "y", "z", "yaw_deg"))):
                raise SchemaError(f"{path}.reverse_manoeuvre transition must repeat raw exact pose")
            continue
        if (not 1e-6 < actor_separation <= 0.1 + 1e-9
                or not 1e-6 < rear_separation <= 0.1 + 1e-9
                or yaw_step > 5.0 + 1e-9):
            raise SchemaError(f"{path}.reverse_manoeuvre spacing/yaw bound failed")
        tangent = math.degrees(math.atan2(
            float(rear_second["y"]) - float(rear_first["y"]),
            float(rear_second["x"]) - float(rear_first["x"]),
        ))
        expected = float(rear_first["yaw_deg"]) + (180 if rear_first["gear"] == "reverse" else 0)
        if _angle_error(tangent, expected) > 5.0 + 1e-9:
            raise SchemaError(f"{path}.reverse_manoeuvre motion/gear yaw mismatch")
        signed_distance = rear_separation * (-1.0 if rear_first["gear"] == "reverse" else 1.0)
        signed_curvature = math.radians(signed_yaw_step) / signed_distance
        try:
            normalized = _parking_normalized_steer_for_curvature(steer_samples, signed_curvature)
        except ValueError as error:
            raise SchemaError(f"{path}.rear curvature is outside frozen steer response") from error
        maximum_curvature = max(maximum_curvature, abs(signed_curvature))
        required_max_steer = max(required_max_steer, abs(normalized))
    terminal_expected = {
        "x": round(center[0], 4), "y": round(center[1], 4), "z": 0.0,
        "yaw_deg": _canonical_yaw(heading),
        "gear": "reverse",
    }
    if poses[-1] != terminal_expected:
        raise SchemaError(f"{path}.reverse_manoeuvre terminal actor pose is not exact slot pose")

    proof = manoeuvre["proof"]
    expected_proof = {
        "method", "poses_sha256", "pose_count", "sample_spacing_max_m",
        "pose_reference", "integration_reference", "rear_axle_poses_sha256",
        "rear_axle_sample_spacing_max_m", "required_max_rear_axle_curvature_1pm",
        "required_max_abs_normalized_steer",
        "max_normalized_control_steer", "vehicle_blueprint", "footprint_length_m",
        "footprint_width_m", "footprint_tolerance_m", "full_footprint_legal_samples",
        "static_clearance_samples", "static_collision_samples", "static_clearance_min_m",
        "static_bbox_categories", "static_bbox_count", "static_bboxes_sha256", "town",
        "xodr_sha256", "lane_graph_sha256", "vertical_overlap_proof",
        "aligned_lead_in_start_index", "slot_axis_approach_index",
        "construction_topology_proof",
    }
    if not isinstance(proof, dict) or set(proof) != expected_proof:
        raise SchemaError(f"{path}.reverse_manoeuvre.proof keys must be {sorted(expected_proof)}")
    if (proof["method"] != "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v5"
            or proof["poses_sha256"] != stable_hash(poses)
            or proof["rear_axle_poses_sha256"] != stable_hash(rear_poses)
            or proof["pose_reference"] != "carla_actor_transform"
            or proof["integration_reference"] != "rear_axle_center"):
        raise SchemaError(f"{path}.reverse_manoeuvre proof method/hash mismatch")
    integer_count = len(poses)
    for field in ("pose_count", "full_footprint_legal_samples", "static_clearance_samples"):
        if (not isinstance(proof[field], int) or isinstance(proof[field], bool)
                or proof[field] != integer_count):
            raise SchemaError(f"{path}.reverse_manoeuvre.proof.{field} must equal pose count")
    aligned_index = _strict_index(
        proof["aligned_lead_in_start_index"], f"{path}.proof.aligned_lead_in_start_index",
    )
    slot_axis_index = _strict_index(
        proof["slot_axis_approach_index"], f"{path}.proof.slot_axis_approach_index",
    )
    if not 0 < aligned_index < slot_axis_index < len(poses) - 1:
        raise SchemaError(f"{path}.reverse_manoeuvre v5 ingress index order differs")
    if any(poses[slot_axis_index][field] != approach[field]
           for field in ("x", "y", "z", "yaw_deg")):
        raise SchemaError(f"{path}.reverse_manoeuvre slot-axis pose disagrees with approach")
    if (not isinstance(reverse["samples"], int) or isinstance(reverse["samples"], bool)
            or reverse["samples"] != integer_count - slot_axis_index
            or not isinstance(proof["static_collision_samples"], int)
            or isinstance(proof["static_collision_samples"], bool)
            or proof["static_collision_samples"] != 0):
        raise SchemaError(f"{path}.reverse_manoeuvre proof sample counts mismatch")
    frozen = {
        "sample_spacing_max_m": PARKING_MANOEUVRE_SPACING_M,
        "rear_axle_sample_spacing_max_m": PARKING_MANOEUVRE_SPACING_M,
        "max_normalized_control_steer": PARKING_MAX_NORMALIZED_CONTROL_STEER,
        "footprint_length_m": PARKING_FOOTPRINT_LENGTH_M,
        "footprint_width_m": PARKING_FOOTPRINT_WIDTH_M,
        "footprint_tolerance_m": PARKING_FOOTPRINT_TOLERANCE_M,
    }
    for field, expected in frozen.items():
        if abs(_number(proof[field], f"{path}.proof.{field}") - expected) > 1e-6:
            raise SchemaError(f"{path}.reverse_manoeuvre proof frozen {field} mismatch")
    if proof["vehicle_blueprint"] != PARKING_VEHICLE_BLUEPRINT or proof["static_bbox_categories"] != PARKING_BBOX_CATEGORIES:
        raise SchemaError(f"{path}.reverse_manoeuvre proof vehicle/categories mismatch")
    measured = _number(
        proof["required_max_rear_axle_curvature_1pm"],
        f"{path}.proof.rear_curvature", minimum=0,
    )
    if abs(measured - maximum_curvature) > 1e-9:
        raise SchemaError(f"{path}.reverse_manoeuvre measured curvature mismatch")
    claimed_steer = _number(
        proof["required_max_abs_normalized_steer"],
        f"{path}.proof.normalized_steer", minimum=0,
    )
    if (abs(claimed_steer - required_max_steer) > 1e-9
            or claimed_steer > PARKING_MAX_NORMALIZED_CONTROL_STEER + 1e-9):
        raise SchemaError(f"{path}.reverse_manoeuvre required normalized steering mismatch")
    if _number(proof["static_clearance_min_m"], f"{path}.proof.clearance") < 0.2 - 1e-9:
        raise SchemaError(f"{path}.reverse_manoeuvre static clearance below 0.2m")
    if (not isinstance(proof["static_bbox_count"], int)
            or isinstance(proof["static_bbox_count"], bool) or proof["static_bbox_count"] < 1):
        raise SchemaError(f"{path}.reverse_manoeuvre static bbox count invalid")
    for field in ("static_bboxes_sha256", "xodr_sha256", "lane_graph_sha256"):
        if not isinstance(proof[field], str) or re.fullmatch(r"[0-9a-f]{64}", proof[field]) is None:
            raise SchemaError(f"{path}.reverse_manoeuvre.proof.{field} is not SHA256")

    vertical = proof["vertical_overlap_proof"]
    vertical_keys = {
        "method", "vertical_padding_m", "ego_local_bbox_sha256", "pose_count",
        "static_bbox_count", "checked_pair_count", "vertically_overlapping_pair_count",
        "vertically_disjoint_pair_count", "overlap_index_sets_sha256",
    }
    if not isinstance(vertical, dict) or set(vertical) != vertical_keys:
        raise SchemaError(f"{path}.vertical_overlap_proof keys are invalid")
    if vertical["method"] != "carla_obb_world_z_interval_overlap_v1":
        raise SchemaError(f"{path}.vertical_overlap_proof method is unsupported")
    if abs(_number(vertical["vertical_padding_m"], f"{path}.vertical_padding_m") - 0.05) > 1e-9:
        raise SchemaError(f"{path}.vertical overlap padding is not frozen")
    for field in (
        "pose_count", "static_bbox_count", "checked_pair_count",
        "vertically_overlapping_pair_count", "vertically_disjoint_pair_count",
    ):
        if not isinstance(vertical[field], int) or isinstance(vertical[field], bool) or vertical[field] < 0:
            raise SchemaError(f"{path}.vertical_overlap_proof.{field} must be a nonnegative integer")
    if (vertical["pose_count"] != len(poses)
            or vertical["static_bbox_count"] != proof["static_bbox_count"]
            or vertical["checked_pair_count"] != len(poses) * proof["static_bbox_count"]
            or vertical["vertically_overlapping_pair_count"]
            + vertical["vertically_disjoint_pair_count"] != vertical["checked_pair_count"]):
        raise SchemaError(f"{path}.vertical overlap counts do not close")
    for field in ("ego_local_bbox_sha256", "overlap_index_sets_sha256"):
        if not isinstance(vertical[field], str) or re.fullmatch(r"[0-9a-f]{64}", vertical[field]) is None:
            raise SchemaError(f"{path}.vertical_overlap_proof.{field} is not SHA256")
    if registry_town is None:
        raise SchemaError(f"{path} parking v5 target lacks source parking_manoeuvre_registry")
    bbox_registry = registry_town["bbox_asset"]
    vehicle_registry = registry_town["vehicle_geometry"]
    if (proof["town"] != registry_town["town"]
            or proof["static_bbox_count"] != bbox_registry["bbox_count"]
            or proof["static_bbox_categories"] != bbox_registry["categories"]
            or proof["static_bboxes_sha256"] != bbox_registry["bboxes_sha256"]
            or proof["xodr_sha256"] != registry_town["xodr"]["sha256"]
            or proof["lane_graph_sha256"] != registry_town["lane_graph"]["sha256"]
            or vertical["ego_local_bbox_sha256"] != vehicle_registry["local_bbox_sha256"]
            or proof["pose_reference"] != registry_town["control_model"]["pose_reference"]
            or proof["integration_reference"]
            != registry_town["control_model"]["integration_reference"]
            or abs(float(proof["max_normalized_control_steer"])
                   - float(registry_town["control_model"]["max_normalized_control_steer"])) > 1e-12):
        raise SchemaError(f"{path}.reverse_manoeuvre proof disagrees with parking registry")
    overlap_sets = _parking_overlap_index_sets(poses, registry_town)
    overlap_count = sum(len(indices) for indices in overlap_sets)
    if (vertical["overlap_index_sets_sha256"] != stable_hash(overlap_sets)
            or vertical["vertically_overlapping_pair_count"] != overlap_count
            or vertical["vertically_disjoint_pair_count"]
            != vertical["checked_pair_count"] - overlap_count):
        raise SchemaError(f"{path}.vertical overlap hash/count differs from frozen assets")
    actual_clearance = _parking_static_horizontal_clearance(poses, registry_town, overlap_sets)
    expected_clearance = (
        0.2 if not math.isfinite(actual_clearance)
        else math.floor(actual_clearance * 1_000_000.0) / 1_000_000.0
    )
    if (expected_clearance < 0.2 - 1e-9
            or abs(float(proof["static_clearance_min_m"]) - expected_clearance) > 1e-6):
        raise SchemaError(f"{path}.parking v5 horizontal clearance disagrees with source geometry")

    topology_replay = _parking_v5_topology(
        proof["construction_topology_proof"], poses,
        "reverse_manoeuvre.poses", 0, slot_axis_index,
        f"{path}.reverse_manoeuvre.proof.construction_topology_proof",
        registry_town, require_raw_transform=False,
    )
    topology_samples = topology_replay["samples"]
    slot_sample = topology_samples[slot_axis_index]
    slot_waypoint = topology_replay["carla_map"].get_waypoint_xodr(
        int(slot_sample["road_id"]), int(slot_sample["lane_id"]),
        float(slot_sample["s_m"]),
    )
    if slot_waypoint is None:
        raise SchemaError(f"{path}.slot-axis construction waypoint cannot reconstruct")
    slot_lane_yaw = _canonical_yaw(slot_waypoint.transform.rotation.yaw)
    if (
        approach_proof["road_id"] != slot_sample["road_id"]
        or approach_proof["section_id"] != slot_sample["section_id"]
        or approach_proof["lane_id"] != slot_sample["lane_id"]
        or float(approach_proof["lane_yaw_deg"]) != slot_lane_yaw
    ):
        raise SchemaError(f"{path}.approach proof differs from slot-axis construction sample")

    aligned_rear = rear_poses[aligned_index:slot_axis_index + 1]
    aligned_actor = poses[aligned_index:slot_axis_index + 1]
    if any(pose["gear"] != "forward" for pose in aligned_rear):
        raise SchemaError(f"{path}.aligned lead-in is not entirely forward")
    start, end = aligned_rear[0], aligned_rear[-1]
    dx = float(end["x"]) - float(start["x"])
    dy = float(end["y"]) - float(start["y"])
    baseline = math.hypot(dx, dy)
    if baseline <= 1e-8:
        raise SchemaError(f"{path}.aligned lead-in has no endpoint baseline")
    ux, uy = dx / baseline, dy / baseline
    baseline_yaw = math.degrees(math.atan2(uy, ux))
    accumulated = 0.0
    previous_progress = -math.inf
    maximum_lateral = 0.0
    maximum_yaw_error = 0.0
    for relative, (rear_pose, actor_pose) in enumerate(zip(aligned_rear, aligned_actor)):
        offset_x = float(rear_pose["x"]) - float(start["x"])
        offset_y = float(rear_pose["y"]) - float(start["y"])
        progress = offset_x * ux + offset_y * uy
        lateral = abs(offset_x * (-uy) + offset_y * ux)
        if relative and progress <= previous_progress + 1e-9:
            raise SchemaError(f"{path}.aligned lead-in is not strictly progressive")
        previous_progress = progress
        maximum_lateral = max(maximum_lateral, lateral)
        maximum_yaw_error = max(
            maximum_yaw_error,
            _angle_error(rear_pose["yaw_deg"], baseline_yaw),
            _angle_error(actor_pose["yaw_deg"], baseline_yaw),
        )
        if relative + 1 < len(aligned_rear):
            following = aligned_rear[relative + 1]
            segment = math.dist(
                tuple(float(rear_pose[name]) for name in ("x", "y", "z")),
                tuple(float(following[name]) for name in ("x", "y", "z")),
            )
            if not 1e-6 < segment <= 0.1 + 1e-9:
                raise SchemaError(f"{path}.aligned lead-in spacing differs")
            accumulated += segment
            tangent = math.degrees(math.atan2(
                float(following["y"]) - float(rear_pose["y"]),
                float(following["x"]) - float(rear_pose["x"]),
            ))
            maximum_yaw_error = max(maximum_yaw_error, _angle_error(tangent, baseline_yaw))
    if baseline < 8.0 - 1e-9 or accumulated < 8.0 - 1e-9:
        raise SchemaError(f"{path}.aligned lead-in is shorter than 8m")
    if maximum_lateral > 0.001 + 1e-9 or maximum_yaw_error > 1.0 + 1e-9:
        raise SchemaError(f"{path}.aligned lead-in straight/yaw bound differs")
    for index in range(aligned_index, slot_axis_index + 1):
        sample = topology_samples[index]
        waypoint = topology_replay["carla_map"].get_waypoint_xodr(
            int(sample["road_id"]), int(sample["lane_id"]), float(sample["s_m"]),
        )
        if waypoint is None or _angle_error(
            waypoint.transform.rotation.yaw, baseline_yaw,
        ) > 1.0 + 1e-9:
            raise SchemaError(f"{path}.aligned lead-in lane yaw differs")

    # Repeat the complete footprint qualification for every active and
    # inactive branch.  The lane-graph polygon remains only a fast screen;
    # exact offline Driving/Parking lookup is the final legality authority.
    from . import parking as parking_module
    geometry = parking_module._geometry(Path(__file__).resolve().parents[2])
    geometry_overlap = parking_module._overlap_index_sets(geometry, poses)
    audit = parking_module._audit(geometry, poses, rear_poses, polygon, geometry_overlap)
    if audit is None:
        exact_context = dict(geometry)
        exact_context["lane_surface"] = geometry["Polygon"]([
            (-1000.0, -1000.0), (1000.0, -1000.0),
            (1000.0, 1000.0), (-1000.0, 1000.0),
        ])
        audit = parking_module._audit(
            exact_context, poses, rear_poses, polygon, geometry_overlap,
        )
    if audit is None or not parking_module._exact_map_footprint_audit(
        geometry, topology_replay["carla_map"], poses, polygon,
    ):
        raise SchemaError(f"{path}.parking v5 full actor-footprint qualification failed")
    if (abs(float(audit[0]) - measured) > 1e-9
            or abs(float(audit[1]) - claimed_steer) > 1e-9):
        raise SchemaError(f"{path}.parking v5 audit/control proof differs")

    if isinstance(oracle_path, list):
        suffix = []
        for pose in poses:
            value = {key: pose[key] for key in ("x", "y", "z", "yaw_deg")}
            if not suffix or value != suffix[-1]:
                suffix.append(value)
        if len(oracle_path) <= len(suffix) or oracle_path[-len(suffix):] != suffix:
            raise SchemaError(f"{path}.reverse_manoeuvre collapsed path is not exact oracle_path suffix")


def _contract_speed_bands(value: Any) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    if isinstance(value, dict):
        body = value.get("speed_band")
        if isinstance(body, dict) and "min_mps" in body and "max_mps" in body:
            result.append((
                _number(body["min_mps"], "contract.speed_band.min_mps", minimum=0.0),
                _number(body["max_mps"], "contract.speed_band.max_mps", minimum=0.0),
            ))
        for child in value.values():
            result.extend(_contract_speed_bands(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_contract_speed_bands(child))
    return result


def _validate_speed_proof(arm: dict[str, Any], path: str, town: str) -> None:
    needs_proof = _contract_contains(arm.get("contract"), "speed_band")
    has_limit = any(key in arm for key in (
        "legal_speed_mps", "legal_speed_proof", "speed_feasibility_cap_proof",
    ))
    if not needs_proof and not has_limit:
        return
    if not needs_proof:
        raise SchemaError(f"{path} has an unneeded legal speed proof")
    limit = _number(arm.get("legal_speed_mps"), f"{path}.legal_speed_mps", minimum=0.000001)
    proof = arm.get("legal_speed_proof")
    if not isinstance(proof, dict):
        raise SchemaError(f"{path}.legal_speed_proof must be an object")
    expected = {
        "method", "samples", "lane_types", "sample_spacing_max_m",
        "minimum_positive_mps", "town", "xodr_sha256",
    }
    if set(proof) != expected:
        raise SchemaError(f"{path}.legal_speed_proof keys must be {sorted(expected)}")
    if proof["method"] != "oracle_path_exact_waypoint_min_v1":
        raise SchemaError(f"{path}.legal_speed_proof.method is unsupported")
    if not isinstance(proof["samples"], int) or isinstance(proof["samples"], bool) or proof["samples"] <= 0:
        raise SchemaError(f"{path}.legal_speed_proof.samples must be a positive integer")
    if proof["lane_types"] != ["Driving", "Parking"]:
        raise SchemaError(f"{path}.legal_speed_proof.lane_types must preserve Driving/Parking precedence")
    _number(proof["sample_spacing_max_m"], f"{path}.legal_speed_proof.sample_spacing_max_m", minimum=0.000001, maximum=0.1)
    minimum = _number(proof["minimum_positive_mps"], f"{path}.legal_speed_proof.minimum_positive_mps", minimum=0.000001)
    if abs(minimum - limit) > 1e-6:
        raise SchemaError(f"{path}.legal_speed_proof minimum disagrees with legal_speed_mps")
    if proof["town"] != town:
        raise SchemaError(f"{path}.legal_speed_proof town disagrees with PCU")
    digest = proof["xodr_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SchemaError(f"{path}.legal_speed_proof.xodr_sha256 must be lowercase SHA256")
    cap_proof = arm.get("speed_feasibility_cap_proof")
    if not isinstance(cap_proof, dict):
        raise SchemaError(f"{path}.speed_feasibility_cap_proof must be an object")
    cap_expected = {"method", "cap_mps", "control_margin_mps"}
    if set(cap_proof) != cap_expected:
        raise SchemaError(
            f"{path}.speed_feasibility_cap_proof keys must be {sorted(cap_expected)}"
        )
    if cap_proof["method"] != "carla_default_limit_conservative_cap_v1":
        raise SchemaError(f"{path}.speed_feasibility_cap_proof.method is unsupported")
    cap = _number(cap_proof["cap_mps"], f"{path}.speed_feasibility_cap_proof.cap_mps", minimum=0.000001)
    margin = _number(
        cap_proof["control_margin_mps"],
        f"{path}.speed_feasibility_cap_proof.control_margin_mps",
        minimum=0.000001,
    )
    if abs(cap - 8.333333) > 1e-6:
        raise SchemaError(f"{path}.speed_feasibility_cap_proof.cap_mps must equal frozen 8.333333")
    if abs(margin - 0.35) > 1e-9:
        raise SchemaError(f"{path}.speed_feasibility_cap_proof.control_margin_mps must equal frozen 0.35")
    effective = min(limit, cap)
    for low, high in _contract_speed_bands(arm.get("contract")):
        if low > high:
            raise SchemaError(f"{path} contains an inverted speed band [{low}, {high}]")
        if high + margin > effective + 1e-6:
            raise SchemaError(
                f"{path} speed band max {high} plus control margin {margin} "
                f"exceeds effective XODR/CARLA cap {effective}"
            )
    signal_proof = arm.get("signal_free_corridor_proof")
    if not isinstance(signal_proof, dict):
        raise SchemaError(f"{path}.signal_free_corridor_proof must be an object")
    signal_expected = {
        "method", "lane_id", "path_start_index_inclusive",
        "path_end_index_inclusive", "buffer_start_index_inclusive",
        "buffer_end_index_inclusive", "road_s_min", "road_s_max",
        "corridor_length_m", "before_buffer_length_m",
        "after_buffer_length_m", "sample_spacing_max_m",
        "samples_checked", "junction_sample_count", "road_signal_count",
        "road_signal_reference_count", "road_stop_control_count",
        "map_road_signal_count", "other_road_signal_clearance_m", "execution_path_sha256",
        "safety_buffer_path_sha256", "lane_graph_sha256", "xodr_sha256",
        "audit_sha256",
    }
    if set(signal_proof) != signal_expected:
        raise SchemaError(
            f"{path}.signal_free_corridor_proof keys must be {sorted(signal_expected)}"
        )
    if signal_proof["method"] != "exact_signal_stop_free_lane_slice_v2":
        raise SchemaError(f"{path}.signal_free_corridor_proof.method is unsupported")
    lane_identity = _nonempty_string(
        signal_proof["lane_id"], f"{path}.signal_free_corridor_proof.lane_id"
    )
    lane_parts = lane_identity.rsplit(":", 3)
    if len(lane_parts) != 4 or lane_parts[0] != town:
        raise SchemaError(f"{path}.signal_free_corridor_proof.lane_id town mismatch")
    try:
        tuple(int(value) for value in lane_parts[1:])
    except ValueError as error:
        raise SchemaError(
            f"{path}.signal_free_corridor_proof.lane_id must encode town:road:section:lane"
        ) from error
    indices = {}
    for field in (
        "path_start_index_inclusive", "path_end_index_inclusive",
        "buffer_start_index_inclusive", "buffer_end_index_inclusive",
    ):
        value = signal_proof[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SchemaError(f"{path}.signal_free_corridor_proof.{field} must be a nonnegative integer")
        indices[field] = value
    if not (
        indices["buffer_start_index_inclusive"] < indices["path_start_index_inclusive"]
        < indices["path_end_index_inclusive"] < indices["buffer_end_index_inclusive"]
    ):
        raise SchemaError(f"{path}.signal_free_corridor_proof slice indices are not nested")
    oracle_path = arm.get("oracle_path")
    if (not isinstance(oracle_path, list)
            or indices["path_end_index_inclusive"] - indices["path_start_index_inclusive"] + 1
            != len(oracle_path)):
        raise SchemaError(
            f"{path}.signal_free_corridor_proof path slice size must equal oracle_path"
        )
    for field in (
        "junction_sample_count", "road_signal_count",
        "road_signal_reference_count", "road_stop_control_count",
    ):
        value = signal_proof[field]
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise SchemaError(f"{path}.signal_free_corridor_proof.{field} must equal integer zero")
    s_min = _number(signal_proof["road_s_min"], f"{path}.signal_free_corridor_proof.road_s_min")
    s_max = _number(signal_proof["road_s_max"], f"{path}.signal_free_corridor_proof.road_s_max")
    if s_min > s_max:
        raise SchemaError(f"{path}.signal_free_corridor_proof road s bounds are inverted")
    corridor_length = _number(
        signal_proof["corridor_length_m"],
        f"{path}.signal_free_corridor_proof.corridor_length_m",
        minimum=55.0, maximum=65.0,
    )
    for field in ("before_buffer_length_m", "after_buffer_length_m"):
        _number(signal_proof[field], f"{path}.signal_free_corridor_proof.{field}", minimum=20.0)
    spacing = _number(
        signal_proof["sample_spacing_max_m"],
        f"{path}.signal_free_corridor_proof.sample_spacing_max_m",
        minimum=0.000001, maximum=0.1,
    )
    if abs(spacing - 0.1) > 1e-9:
        raise SchemaError(f"{path}.signal_free_corridor_proof.sample_spacing_max_m must equal 0.1")
    checked = signal_proof["samples_checked"]
    if not isinstance(checked, int) or isinstance(checked, bool) or checked <= 0:
        raise SchemaError(f"{path}.signal_free_corridor_proof.samples_checked must be positive integer")
    clearance = signal_proof["other_road_signal_clearance_m"]
    map_signal_count = signal_proof["map_road_signal_count"]
    if (not isinstance(map_signal_count, int) or isinstance(map_signal_count, bool)
            or map_signal_count < 0):
        raise SchemaError(
            f"{path}.signal_free_corridor_proof.map_road_signal_count must be nonnegative integer"
        )
    if clearance is not None:
        _number(
            clearance, f"{path}.signal_free_corridor_proof.other_road_signal_clearance_m",
            minimum=40.0,
        )
        if map_signal_count <= 0:
            raise SchemaError(f"{path}.signal_free_corridor_proof clearance needs map signals")
    elif map_signal_count != 0:
        raise SchemaError(
            f"{path}.signal_free_corridor_proof null clearance requires zero map signals"
        )
    for field in (
        "execution_path_sha256", "safety_buffer_path_sha256",
        "lane_graph_sha256", "xodr_sha256", "audit_sha256",
    ):
        digest = signal_proof[field]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SchemaError(f"{path}.signal_free_corridor_proof.{field} must be lowercase SHA256")
    if signal_proof["execution_path_sha256"] != stable_hash(arm.get("oracle_path")):
        raise SchemaError(f"{path}.signal_free_corridor_proof does not bind oracle_path")
    actual_length = 0.0
    oracle_path = arm.get("oracle_path", [])
    if isinstance(oracle_path, list):
        actual_length = sum(
            math.hypot(float(second["x"]) - float(first["x"]),
                       float(second["y"]) - float(first["y"]))
            for first, second in zip(oracle_path, oracle_path[1:])
            if isinstance(first, dict) and isinstance(second, dict)
        )
    if abs(actual_length - corridor_length) > 0.01:
        raise SchemaError(f"{path}.signal_free_corridor_proof length does not match oracle_path")


def _validate_oracle_path(value: Any, path: str) -> None:
    """Validate the evaluator-private executable polyline.

    Release routes use the private path directly, including its heading at
    spawn and at audited parking/region terminal poses.  Requiring every point
    to carry the same complete transform prevents an all-or-none yaw mismatch
    from being discovered only after CARLA has started.
    """

    if not isinstance(value, list) or len(value) < 2:
        raise SchemaError(f"{path} must contain at least two points")
    previous: tuple[float, float, float] | None = None
    for index, point in enumerate(value):
        point_path = f"{path}[{index}]"
        if not isinstance(point, dict):
            raise SchemaError(f"{point_path} must be an object")
        missing = {"x", "y", "z", "yaw_deg"} - set(point)
        if missing:
            raise SchemaError(f"{point_path} is missing {sorted(missing)}")
        coordinates = tuple(_number(point[axis], f"{point_path}.{axis}") for axis in ("x", "y", "z"))
        _number(point["yaw_deg"], f"{point_path}.yaw_deg")
        if previous is not None and math.dist(previous, coordinates) <= 1e-6:
            raise SchemaError(f"{point_path} duplicates the preceding oracle waypoint")
        previous = coordinates


def _allowed_keys(body: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(body) - allowed
    if unknown:
        raise SchemaError(f"{path} has unknown arguments {sorted(unknown)}")


def _validate_predicate(name: str, body: dict[str, Any], path: str, depth: int) -> None:
    if name == "enter_gate":
        _allowed_keys(body, {"id"}, path)
        gate_id = body.get("id")
        if isinstance(gate_id, dict) and set(gate_id) == {"$ref"}:
            _nonempty_string(gate_id["$ref"], f"{path}.id.$ref")
        elif isinstance(gate_id, str) and gate_id.startswith("$"):
            _nonempty_string(gate_id[1:], f"{path}.id")
        else:
            _nonempty_string(gate_id, f"{path}.id")
    elif name == "inside_region":
        _allowed_keys(body, {"polygon", "tolerance_m", "stable_seconds"}, path)
        _polygon(body.get("polygon"), f"{path}.polygon")
        if "tolerance_m" in body:
            _number(body["tolerance_m"], f"{path}.tolerance_m", minimum=0)
    elif name == "vehicle_footprint_inside_slot":
        _allowed_keys(body, {"polygon", "tolerance_m", "heading_deg", "target_heading_deg",
                             "heading_tolerance_deg", "max_speed_mps", "stable_seconds"}, path)
        _polygon(body.get("polygon"), f"{path}.polygon")
        if "tolerance_m" in body:
            _number(body["tolerance_m"], f"{path}.tolerance_m", minimum=0)
        heading = body.get("heading_deg", body.get("target_heading_deg"))
        if heading is not None:
            headings = heading if isinstance(heading, list) else [heading]
            if not headings:
                raise SchemaError(f"{path}.heading_deg cannot be empty")
            for index, value in enumerate(headings):
                _number(value, f"{path}.heading_deg[{index}]")
        if "heading_tolerance_deg" in body:
            _number(body["heading_tolerance_deg"], f"{path}.heading_tolerance_deg", minimum=0, maximum=180)
        if "max_speed_mps" in body:
            _number(body["max_speed_mps"], f"{path}.max_speed_mps", minimum=0)
    elif name == "lane_relation":
        _allowed_keys(body, {"value", "stable_seconds", "scope"}, path)
        _nonempty_string(body.get("value"), f"{path}.value")
    elif name == "speed_band":
        _allowed_keys(body, {"min_mps", "max_mps", "stable_seconds", "scope"}, path)
        low = _number(body.get("min_mps"), f"{path}.min_mps", minimum=0)
        high = _number(body.get("max_mps"), f"{path}.max_mps", minimum=0)
        if low > high:
            raise SchemaError(f"{path}.min_mps cannot exceed max_mps")
        if "stable_seconds" in body:
            _number(body["stable_seconds"], f"{path}.stable_seconds", minimum=0)
    elif name == "headway_band":
        _allowed_keys(body, {"min_seconds", "max_seconds", "actor_id", "stable_seconds"}, path)
        low = _number(body.get("min_seconds"), f"{path}.min_seconds", minimum=0)
        high = _number(body.get("max_seconds"), f"{path}.max_seconds", minimum=0)
        if low > high:
            raise SchemaError(f"{path}.min_seconds cannot exceed max_seconds")
        if "actor_id" in body:
            actor_id = body["actor_id"]
            if not isinstance(actor_id, str) or not actor_id.strip():
                raise SchemaError(f"{path}.actor_id must be a non-empty actor id or reference")
        if "stable_seconds" in body:
            _number(body["stable_seconds"], f"{path}.stable_seconds", minimum=0)
    elif name == "passing_side":
        _allowed_keys(body, {"value", "stable_seconds"}, path)
        _nonempty_string(body.get("value"), f"{path}.value")
    elif name == "actor_order":
        _allowed_keys(body, {"id", "value", "order", "stable_seconds"}, path)
        if "order" in body:
            if not isinstance(body["order"], list) or not body["order"]:
                raise SchemaError(f"{path}.order must be a non-empty list")
            for index, value in enumerate(body["order"]):
                _nonempty_string(value, f"{path}.order[{index}]")
        elif "id" in body:
            value = body["id"]
            if not (isinstance(value, str) and (value.startswith("$") or value.strip())) and not (
                isinstance(value, dict) and set(value) == {"$ref"} and isinstance(value["$ref"], str)
            ):
                raise SchemaError(f"{path}.id must be a non-empty id or reference")
        elif "value" in body:
            _nonempty_string(body["value"], f"{path}.value")
        else:
            raise SchemaError(f"{path} requires id, value, or order")
    elif name == "progress":
        _allowed_keys(body, {"min", "stable_seconds"}, path)
        if "min" in body:
            _number(body["min"], f"{path}.min", minimum=0)
    elif name == "dwell":
        _allowed_keys(body, {"key", "condition", "min_seconds", "seconds", "max_speed_mps"}, path)
        has_key, has_condition = "key" in body, "condition" in body
        if has_key == has_condition:
            raise SchemaError(f"{path} requires exactly one of key or condition")
        if has_key:
            _nonempty_string(body["key"], f"{path}.key")
        else:
            validate_contract(body["condition"], f"{path}.condition", _depth=depth + 1)
        seconds_key = "min_seconds" if "min_seconds" in body else "seconds"
        if seconds_key in body:
            _number(body[seconds_key], f"{path}.{seconds_key}", minimum=0)
        if "max_speed_mps" in body:
            _number(body["max_speed_mps"], f"{path}.max_speed_mps", minimum=0)
    elif name == "safe_legal":
        _allowed_keys(body, {"forbidden_events"}, path)
        if "forbidden_events" in body:
            events = body["forbidden_events"]
            if not isinstance(events, list) or not events:
                raise SchemaError(f"{path}.forbidden_events must be a non-empty list")
            for index, value in enumerate(events):
                _nonempty_string(value, f"{path}.forbidden_events[{index}]")

    if name not in {"speed_band", "headway_band", "dwell"} and "stable_seconds" in body:
        _number(body["stable_seconds"], f"{path}.stable_seconds", minimum=0)
    if "scope" in body:
        _nonempty_string(body["scope"], f"{path}.scope")


def validate_contract(node: Any, path: str = "contract", *, _depth: int = 0) -> None:
    if _depth > 64:
        raise SchemaError(f"{path} exceeds maximum nesting depth")
    if not isinstance(node, dict) or len(node) != 1:
        raise SchemaError(f"{path} must contain exactly one operator or predicate")
    name, body = next(iter(node.items()))
    if name in PREDICATES:
        if not isinstance(body, dict):
            raise SchemaError(f"{path}.{name} must be an object")
        _validate_predicate(name, body, f"{path}.{name}", _depth)
        return
    if name not in OPERATORS:
        raise SchemaError(f"{path}: unknown expression {name!r}")
    if name in {"eventually", "always"}:
        validate_contract(body, f"{path}.{name}", _depth=_depth + 1)
    elif name == "within":
        if not isinstance(body, dict) or set(body) != {"condition", "seconds"}:
            raise SchemaError(f"{path}.within requires condition and seconds")
        _number(body["seconds"], f"{path}.within.seconds", minimum=0)
        validate_contract(body["condition"], f"{path}.within.condition", _depth=_depth + 1)
    elif name == "until":
        if not isinstance(body, dict) or set(body) != {"invariant", "condition"}:
            raise SchemaError(f"{path}.until requires invariant and condition")
        validate_contract(body["invariant"], f"{path}.until.invariant", _depth=_depth + 1)
        validate_contract(body["condition"], f"{path}.until.condition", _depth=_depth + 1)
    elif name == "sequence":
        if not isinstance(body, list) or not body:
            raise SchemaError(f"{path}.sequence must be a non-empty list")
        for index, child in enumerate(body):
            validate_contract(child, f"{path}.sequence[{index}]", _depth=_depth + 1)
    elif name == "if_else":
        if not isinstance(body, dict) or set(body) != {"condition", "then", "else"}:
            raise SchemaError(f"{path}.if_else requires condition, then, and else")
        for key in ("condition", "then", "else"):
            validate_contract(body[key], f"{path}.if_else.{key}", _depth=_depth + 1)
    else:
        if not isinstance(body, dict):
            raise SchemaError(f"{path}.bind_reference must be an object")
        required, allowed = {"reference", "body"}, {"reference", "body", "source", "value", "when"}
        if not required.issubset(body) or not set(body).issubset(allowed):
            raise SchemaError(f"{path}.bind_reference requires reference/body and optional source/value/when")
        if "source" in body and "value" in body:
            raise SchemaError(f"{path}.bind_reference source and value are mutually exclusive")
        # A reference may be a human-readable semantic label (for example a
        # complete landmark name), not merely a filesystem/run identifier.
        _nonempty_string(body["reference"], f"{path}.bind_reference.reference")
        if "source" in body:
            _nonempty_string(body["source"], f"{path}.bind_reference.source")
        if "when" in body:
            validate_contract(body["when"], f"{path}.bind_reference.when", _depth=_depth + 1)
        validate_contract(body["body"], f"{path}.bind_reference.body", _depth=_depth + 1)


_LANE_TERMINAL_MAP_CACHE: dict[tuple[str, str], Any] = {}


def _validate_lane_terminal_commitment(pcu: dict[str, Any], path: str) -> None:
    """Bind A/lane outcomes to their proven terminal commitment window."""

    if pcu.get("layer") != "A" or pcu.get("category") != "lane":
        return
    arms = pcu["arms"]
    current = [arm for arm in arms if arm.get("id") == "current"]
    change = [arm for arm in arms if arm.get("id") != "current"]
    if len(current) != 1 or len(change) != 1:
        raise SchemaError(f"{path} lane PCU requires one current and one adjacent arm")
    for index, arm in enumerate(arms):
        target = arm.get("target")
        if not isinstance(target, dict) or set(target) != {"type", "id"} or target.get("type") != "lane":
            raise SchemaError(f"{path}.arms[{index}].target must be an exact lane target")
        lane_id = _nonempty_string(target.get("id"), f"{path}.arms[{index}].target.id")
        expected = {"sequence": [
            {"progress": {"min": 0.9}},
            {"dwell": {
                "condition": {"lane_relation": {"value": lane_id}},
                "seconds": 2.0,
            }},
        ]}
        if arm.get("contract") != expected:
            raise SchemaError(
                f"{path}.arms[{index}].contract must use the exact terminal lane commitment"
            )

    change_path = change[0]["oracle_path"]
    current_path = current[0]["oracle_path"]
    if len(change_path) != 4 or len(current_path) != 3:
        raise SchemaError(f"{path} lane paths must retain the sparse 4/3-point construction")

    def planar_length(points: list[dict[str, Any]]) -> float:
        return sum(
            math.hypot(float(end["x"]) - float(start["x"]),
                       float(end["y"]) - float(start["y"]))
            for start, end in zip(points, points[1:])
        )

    total = planar_length(change_path)
    connector = planar_length(change_path[:3])
    if total <= 1e-9:
        raise SchemaError(f"{path} lane-change path has no planar length")
    connector_progress = connector / total
    precommitment_margin = 0.9 * total - connector
    suffix_length = planar_length(change_path[2:])
    if connector_progress > 0.88 + 1e-12:
        raise SchemaError(f"{path} lane-change connector ends after progress .88")
    if precommitment_margin + 1e-12 < 2.0:
        raise SchemaError(f"{path} lane-change connector lacks 2m precommitment margin")

    metadata = pcu.get("metadata")
    proof = metadata.get("lane_terminal_commitment_proof") if isinstance(metadata, dict) else None
    proof_keys = {
        "schema_version", "method", "commitment_progress", "connector_max_progress",
        "minimum_precommitment_margin_m", "exact_suffix_step_m",
        "change_connector_end_progress", "change_precommitment_margin_m",
        "change_terminal_suffix_m", "source_lane_id", "target_lane_id",
        "exact_lane_sample_counts", "xodr_source_ref", "xodr_sha256",
        "lane_graph_source_ref", "lane_graph_sha256",
    }
    if not isinstance(proof, dict) or set(proof) != proof_keys:
        raise SchemaError(f"{path}.metadata lane terminal proof must have exact keys")
    expected_scalars = {
        "schema_version": "language2drive.lane_terminal_commitment.v1",
        "method": "serialized_xy_connector_terminal_exact_lane_v1",
        "commitment_progress": 0.9,
        "connector_max_progress": 0.88,
        "minimum_precommitment_margin_m": 2.0,
        "exact_suffix_step_m": 0.1,
        "change_connector_end_progress": round(connector_progress, 9),
        "change_precommitment_margin_m": round(precommitment_margin, 6),
        "change_terminal_suffix_m": round(suffix_length, 6),
        "source_lane_id": current[0]["target"]["id"],
        "target_lane_id": change[0]["target"]["id"],
    }
    for key, expected in expected_scalars.items():
        value = proof.get(key)
        if isinstance(expected, float):
            _number(value, f"{path}.metadata.lane_terminal_commitment_proof.{key}")
        if value != expected:
            raise SchemaError(f"{path}.metadata lane terminal proof {key} disagrees")
    counts = proof.get("exact_lane_sample_counts")
    expected_counts = {
        "change_terminal_window": 101,
        "keep_terminal_window": 101,
        "raw_serialized_poses": 7,
        "change_connector_to_terminal": max(1, int(math.ceil(suffix_length / 0.1))) + 1,
    }
    if not isinstance(counts, dict) or counts != expected_counts or any(
        isinstance(value, bool) or not isinstance(value, int) for value in counts.values()
    ):
        raise SchemaError(f"{path}.metadata lane terminal exact sample counts disagree")

    source_files = pcu.get("_source_files")
    if not isinstance(source_files, dict):
        raise SchemaError(f"{path} lane terminal proof lacks source_files authority")
    bound_paths: dict[str, Path] = {}
    for kind in ("xodr", "lane_graph"):
        source_ref = proof.get(f"{kind}_source_ref")
        digest = proof.get(f"{kind}_sha256")
        if (not isinstance(source_ref, str) or not source_ref
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise SchemaError(f"{path}.metadata lane terminal {kind} source binding is invalid")
        source = Path(source_ref).resolve()
        if source_files.get(str(source)) != digest:
            raise SchemaError(f"{path}.metadata lane terminal {kind} is absent from source_files")
        if not source.is_file() or sha256_file(source) != digest:
            raise SchemaError(f"{path}.metadata lane terminal {kind} source hash is stale")
        bound_paths[kind] = source

    lane_graph = json.loads(bound_paths["lane_graph"].read_text(encoding="utf-8"))
    if lane_graph.get("town") != pcu.get("town"):
        raise SchemaError(f"{path}.metadata lane graph town disagrees")
    graph_lanes = {
        lane.get("id"): lane
        for lane in lane_graph.get("lanes", [])
        if isinstance(lane, dict) and isinstance(lane.get("id"), str)
    }
    graph_lane_ids = set(graph_lanes)
    if {proof["source_lane_id"], proof["target_lane_id"]} - graph_lane_ids:
        raise SchemaError(f"{path}.metadata terminal lane identities are absent from lane graph")
    source_graph_lane = graph_lanes[proof["source_lane_id"]]
    target_graph_lane = graph_lanes[proof["target_lane_id"]]
    adjacent_side = None
    if source_graph_lane.get("left_adjacent") == proof["target_lane_id"]:
        adjacent_side = "left"
    elif source_graph_lane.get("right_adjacent") == proof["target_lane_id"]:
        adjacent_side = "right"
    permission = str(source_graph_lane.get("lane_change", "none")).casefold()
    if (adjacent_side is None or permission not in {adjacent_side, "both"}
            or source_graph_lane.get("is_junction") is not False
            or target_graph_lane.get("is_junction") is not False):
        raise SchemaError(f"{path}.metadata target lane is not a permitted adjacent lane")

    def lane_tuple(identifier: str) -> tuple[int, int, int]:
        pieces = identifier.rsplit(":", 3)
        if len(pieces) != 4 or pieces[0] != pcu.get("town"):
            raise SchemaError(f"{path}.metadata terminal lane identity is malformed")
        try:
            return tuple(int(value) for value in pieces[1:])  # type: ignore[return-value]
        except ValueError as error:
            raise SchemaError(f"{path}.metadata terminal lane identity is malformed") from error

    cache_key = (str(pcu["town"]), proof["xodr_sha256"])
    carla_map = _LANE_TERMINAL_MAP_CACHE.get(cache_key)
    if carla_map is None:
        carla = _import_carla()
        carla_map = carla.Map(
            str(pcu["town"]), bound_paths["xodr"].read_text(encoding="utf-8")
        )
        _LANE_TERMINAL_MAP_CACHE[cache_key] = carla_map
    else:
        carla = _import_carla()

    def fraction_point(points: list[dict[str, Any]], fraction: float) -> dict[str, float]:
        lengths = [
            math.hypot(float(end["x"]) - float(start["x"]),
                       float(end["y"]) - float(start["y"]))
            for start, end in zip(points, points[1:])
        ]
        total_length = sum(lengths)
        target_length = fraction * total_length
        along = 0.0
        for index, length in enumerate(lengths):
            if target_length <= along + length + 1e-9 or index == len(lengths) - 1:
                ratio = 0.0 if length <= 1e-12 else min(
                    1.0, max(0.0, (target_length - along) / length)
                )
                start, end = points[index], points[index + 1]
                return {
                    axis: float(start.get(axis, 0.0))
                    + ratio * (float(end.get(axis, 0.0)) - float(start.get(axis, 0.0)))
                    for axis in ("x", "y", "z")
                }
            along += length
        raise SchemaError(f"{path} lane terminal interpolation failed")

    def exact_lane(
        point: dict[str, float], expected: tuple[int, int, int] | None,
        serialized_yaw: float, label: str,
    ) -> None:
        waypoint = carla_map.get_waypoint(
            carla.Location(x=point["x"], y=point["y"], z=point["z"]),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        observed = None if waypoint is None else (
            int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id),
        )
        z_error = math.inf if waypoint is None else abs(
            point["z"] - float(waypoint.transform.location.z)
        )
        yaw_error = math.inf if waypoint is None else abs(
            (float(waypoint.transform.rotation.yaw) - float(serialized_yaw) + 180.0)
            % 360.0 - 180.0
        )
        if (waypoint is None or (expected is not None and observed != expected)
                or z_error > 0.1 + 1e-9 or yaw_error > 3.0 + 1e-9):
            raise SchemaError(
                f"{path} exact terminal lane proof failed at {label}: "
                f"{observed} != {expected}, z_error={z_error:.6f}m, "
                f"yaw_error={yaw_error:.6f}deg"
            )

    source_tuple = lane_tuple(proof["source_lane_id"])
    target_tuple = lane_tuple(proof["target_lane_id"])
    connector_pose = change_path[1]
    connector_waypoint = carla_map.get_waypoint(
        carla.Location(
            x=float(connector_pose["x"]), y=float(connector_pose["y"]),
            z=float(connector_pose.get("z", 0.0)),
        ),
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    permission = "" if connector_waypoint is None else str(
        connector_waypoint.lane_change
    ).casefold()
    connector_adjacent = None if connector_waypoint is None else getattr(
        connector_waypoint, f"get_{adjacent_side}_lane"
    )()
    connector_adjacent_tuple = None if connector_adjacent is None else (
        int(connector_adjacent.road_id), int(connector_adjacent.section_id),
        int(connector_adjacent.lane_id),
    )
    if (connector_waypoint is None or connector_waypoint.is_junction
            or not (adjacent_side in permission or "both" in permission)
            or connector_adjacent is None
            or connector_adjacent.lane_type != carla.LaneType.Driving
            or connector_adjacent_tuple != target_tuple
            or int(connector_waypoint.lane_id) * int(connector_adjacent.lane_id) <= 0):
        raise SchemaError(f"{path} connector start lacks exact local lane-change permission")
    for label, points, expected_keys in (
        ("change", change_path, (None, source_tuple, target_tuple, target_tuple)),
        ("keep", current_path, (None, source_tuple, source_tuple)),
    ):
        for pose_index, (pose, expected) in enumerate(zip(points, expected_keys)):
            exact_lane(
                {axis: float(pose.get(axis, 0.0)) for axis in ("x", "y", "z")},
                expected, float(pose["yaw_deg"]), f"{label}:raw:{pose_index}",
            )

    def segment_yaw(points: list[dict[str, Any]]) -> float:
        start, end = points[-2], points[-1]
        dx = float(end["x"]) - float(start["x"])
        dy = float(end["y"]) - float(start["y"])
        if math.hypot(dx, dy) <= 1e-9:
            raise SchemaError(f"{path} lane terminal segment has no planar direction")
        return math.degrees(math.atan2(dy, dx))

    for label, points, expected in (
        ("change", change_path, target_tuple),
        ("keep", current_path, source_tuple),
    ):
        terminal_yaw = segment_yaw(points)
        for index in range(101):
            fraction = 0.9 + index * 0.001
            exact_lane(
                fraction_point(points, fraction), expected,
                terminal_yaw, f"{label}:{fraction:.3f}",
            )

    # Replay the complete sparse paths with the same directed, exact-lane
    # semantics used by the producer.  Raw endpoint membership alone cannot
    # detect a connector chord that doubles back along an otherwise legal lane.
    for label, points in (("change", change_path), ("keep", current_path)):
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            dx = float(end["x"]) - float(start["x"])
            dy = float(end["y"]) - float(start["y"])
            dz = float(end.get("z", 0.0)) - float(start.get("z", 0.0))
            distance = math.hypot(dx, dy)
            if distance <= 0.05:
                continue
            tangent_yaw = math.degrees(math.atan2(dy, dx))
            sample_count = max(1, int(math.ceil(distance / 0.1)))
            for sample_index in range(sample_count + 1):
                ratio = sample_index / sample_count
                sample = carla.Location(
                    x=float(start["x"]) + ratio * dx,
                    y=float(start["y"]) + ratio * dy,
                    z=float(start.get("z", 0.0)) + ratio * dz,
                )
                waypoint = None
                for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
                    waypoint = carla_map.get_waypoint(
                        sample, project_to_road=False, lane_type=lane_type,
                    )
                    if waypoint is not None:
                        break
                if waypoint is None:
                    raise SchemaError(
                        f"{path} {label} directed path has no exact lane at "
                        f"segment {segment_index} sample {sample_index}"
                    )
                if not waypoint.is_junction:
                    heading_error = abs(
                        (tangent_yaw - float(waypoint.transform.rotation.yaw) + 180.0)
                        % 360.0 - 180.0
                    )
                    if heading_error > 90.0 + 1e-9:
                        raise SchemaError(
                            f"{path} {label} directed path runs backwards at "
                            f"segment {segment_index} sample {sample_index}"
                        )
    suffix = change_path[2:]
    suffix_segments = expected_counts["change_connector_to_terminal"] - 1
    suffix_yaw = segment_yaw(suffix)
    for index in range(suffix_segments + 1):
        exact_lane(
            fraction_point(suffix, index / suffix_segments),
            target_tuple, suffix_yaw,
            f"change_suffix:{index}/{suffix_segments}",
        )


@dataclass(frozen=True)
class ArmIdentity:
    pcu_id: str
    arm_id: str
    paraphrase: int
    seed: int
    track: str

    def __post_init__(self) -> None:
        _identifier(self.pcu_id, "pcu_id")
        _identifier(self.arm_id, "arm_id")
        if not isinstance(self.paraphrase, int) or isinstance(self.paraphrase, bool) or not 0 <= self.paraphrase <= 2:
            raise SchemaError("paraphrase must be one of 0, 1, or 2")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SchemaError("seed must be a non-negative integer")
        if self.track not in {"standardized", "native", "oracle"}:
            raise SchemaError("track must be standardized, native, or oracle")

    @property
    def run_id(self) -> str:
        return f"{self.track}__{self.pcu_id}__{self.arm_id}__p{self.paraphrase}__s{self.seed}"


def validate_pcu(pcu: dict[str, Any]) -> None:
    if not isinstance(pcu, dict):
        raise SchemaError("PCU must be an object")
    required = {"id", "layer", "split", "group_id", "town", "coarse_envelope", "arms"}
    missing = required - pcu.keys()
    if missing:
        raise SchemaError(f"PCU {pcu.get('id', '<unknown>')} missing {sorted(missing)}")
    identifier = _identifier(pcu["id"], "PCU.id")
    registry_town = _parking_registry_town(pcu)
    if pcu["layer"] not in {"A", "B", "C", "D"} or pcu["split"] not in {"dev", "test"}:
        raise SchemaError(f"PCU {identifier} has invalid layer or split")
    if not identifier.startswith(f"{pcu['layer']}-"):
        raise SchemaError(f"PCU {identifier} id must start with its layer and '-' ")
    _identifier(pcu["group_id"], f"PCU {identifier}.group_id")
    _nonempty_string(pcu["town"], f"PCU {identifier}.town")
    if registry_town is not None and _parking_targets(pcu.get("arms", [])):
        if pcu["town"] != registry_town["town"]:
            raise SchemaError(f"PCU {identifier}.town disagrees with parking registry")
    eligibility = pcu.get("eligibility")
    if eligibility is not None:
        if not isinstance(eligibility, dict):
            raise SchemaError(f"PCU {identifier}.eligibility must be an object")
        for field, allowed in {
            "static_map": {"passed", "failed", "pending"},
            "semantic_assets": {"existing_annotation", "confirmed", "pending_review", "not_applicable"},
            "carla_oracle": {"passed", "failed", "pending_execution"},
        }.items():
            if eligibility.get(field) not in allowed:
                raise SchemaError(f"PCU {identifier}.eligibility.{field} must be one of {sorted(allowed)}")

    envelope = pcu["coarse_envelope"]
    if not isinstance(envelope, list) or len(envelope) < 2:
        raise SchemaError(f"PCU {identifier}.coarse_envelope requires at least two points")
    for index, point in enumerate(envelope):
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            raise SchemaError(f"PCU {identifier}.coarse_envelope[{index}] requires x and y")
        _number(point["x"], f"PCU {identifier}.coarse_envelope[{index}].x")
        _number(point["y"], f"PCU {identifier}.coarse_envelope[{index}].y")
        if "z" in point:
            _number(point["z"], f"PCU {identifier}.coarse_envelope[{index}].z")

    arms = pcu["arms"]
    if not isinstance(arms, list) or len(arms) != 2:
        raise SchemaError(f"PCU {identifier} must have exactly two arms")
    arm_ids, outcomes, all_instructions = set(), set(), set()
    for arm_index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            raise SchemaError(f"PCU {identifier} arm {arm_index} must be an object")
        for field in ("id", "instructions", "contract", "opposite_contract", "outcome_key"):
            if field not in arm:
                raise SchemaError(f"PCU {identifier} arm missing {field}")
        arm_id = _identifier(arm["id"], f"PCU {identifier}.arms[{arm_index}].id")
        instructions = arm["instructions"]
        if not isinstance(instructions, list) or len(instructions) != 3:
            raise SchemaError(f"PCU {identifier} arm {arm_id} requires exactly 3 instructions")
        normalized = []
        for instruction_index, instruction in enumerate(instructions):
            text = _nonempty_string(instruction, f"PCU {identifier} arm {arm_id}.instructions[{instruction_index}]")
            normalized.append(" ".join(text.casefold().split()))
        if len(set(normalized)) != 3:
            raise SchemaError(f"PCU {identifier} arm {arm_id} instructions must be distinct")
        if all_instructions.intersection(normalized):
            raise SchemaError(f"PCU {identifier} arms cannot share an instruction")
        all_instructions.update(normalized)

        validate_contract(arm["contract"], f"PCU {identifier}.arms[{arm_index}].contract")
        validate_contract(arm["opposite_contract"], f"PCU {identifier}.arms[{arm_index}].opposite_contract")
        if arm["contract"] == arm["opposite_contract"]:
            raise SchemaError(f"PCU {identifier} arm {arm_id} own and opposite contracts must differ")
        outcome = _nonempty_string(arm["outcome_key"], f"PCU {identifier}.arms[{arm_index}].outcome_key")
        if "deadline_seconds" in arm and arm["deadline_seconds"] is not None:
            _number(arm["deadline_seconds"], f"PCU {identifier}.arms[{arm_index}].deadline_seconds", minimum=0)
        if "actor_setup" in arm and not isinstance(arm["actor_setup"], dict):
            raise SchemaError(f"PCU {identifier}.arms[{arm_index}].actor_setup must be an object")
        _validate_oracle_path(arm.get("oracle_path"), f"PCU {identifier}.arms[{arm_index}].oracle_path")
        parking_targets = _parking_targets(arm.get("target"))
        selected_parking = _selected_parking_target(arm.get("target"), arm.get("actor_setup", {}))
        for parking_index, parking_target in enumerate(parking_targets):
            _validate_parking_target(
                parking_target,
                arm.get("oracle_path") if parking_target is selected_parking else None,
                f"PCU {identifier}.arms[{arm_index}].target.parking[{parking_index}]",
                registry_town,
            )
        arm_path = f"PCU {identifier}.arms[{arm_index}]"
        if selected_parking is not None:
            if registry_town is None:
                raise SchemaError(f"{arm_path} selected parking target lacks registry")
            _parking_v5_route_proof(
                arm, selected_parking, arm_path, registry_town,
            )
        elif "parking_v5_route_proof" in arm:
            raise SchemaError(f"{arm_path} non-parking arm carries parking_v5_route_proof")
        _validate_speed_proof(arm, f"PCU {identifier}.arms[{arm_index}]", pcu["town"])
        arm_ids.add(arm_id)
        outcomes.add(outcome)
    if len(arm_ids) != 2 or len(outcomes) != 2:
        raise SchemaError(f"PCU {identifier} arms must have unique ids and mutually exclusive outcomes")
    if arms[0]["contract"] != arms[1]["opposite_contract"] or arms[1]["contract"] != arms[0]["opposite_contract"]:
        raise SchemaError(f"PCU {identifier} arm contracts must be exact counterfactual opposites")
    first_start, second_start = arms[0]["oracle_path"][0], arms[1]["oracle_path"][0]
    if math.dist(
        tuple(float(first_start[axis]) for axis in ("x", "y", "z")),
        tuple(float(second_start[axis]) for axis in ("x", "y", "z")),
    ) > 1e-3:
        raise SchemaError(f"PCU {identifier} counterfactual arms must share one start position")
    yaw_delta = (float(first_start["yaw_deg"]) - float(second_start["yaw_deg"]) + 180.0) % 360.0 - 180.0
    if abs(yaw_delta) > 1e-3:
        raise SchemaError(f"PCU {identifier} counterfactual arms must share one start yaw")
    _validate_lane_terminal_commitment(pcu, f"PCU {identifier}")
