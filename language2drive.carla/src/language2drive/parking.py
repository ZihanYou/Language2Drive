from __future__ import annotations

import json
import math
from .frozen_paths import ProvenancePath as Path
from typing import Any, Callable

from .io import sha256_file, stable_hash


BBOX_ASSET = Path("generated/maps/static_bboxes/parkinglotfish-new3.level_bboxes.v1.json")
VEHICLE_BBOX_ASSET = Path("generated/maps/vehicle_geometry/lincoln_mkz_2020.bounding_box.v1.json")
VEHICLE_DYNAMICS_ASSET = Path("generated/maps/vehicle_dynamics/lincoln_mkz_2020.dynamics.v1.json")
SLOT_COLLISION_OCCUPANCY_ASSET = Path(
    "generated/maps/parkinglotfish-new3.slot_collision_occupancy.v1.json"
)
BBOX_CATEGORIES = [
    "Buildings", "Fences", "GuardRail", "Other", "Poles", "Static",
    "TrafficLight", "TrafficSigns", "Vegetation", "Walls",
]
MAX_NORMALIZED_CONTROL_STEER = 0.8
FRONT_AXLE_X_M = 1.471838832
REAR_AXLE_X_M = -1.388632059
WHEELBASE_M = 2.860470891
VEHICLE_BLUEPRINT = "vehicle.lincoln.mkz_2020"
FOOTPRINT_LENGTH_M = 4.892382
FOOTPRINT_WIDTH_M = 1.836713
FOOTPRINT_TOLERANCE_M = 0.2
MANOEUVRE_SPACING_M = 0.1
SAMPLE_STEP_M = 0.075
STATIC_CLEARANCE_M = 0.2
VERTICAL_PADDING_M = 0.05
LANE_SURFACE_NUMERIC_TOLERANCE_M = 0.005
# Generation reserves 0.25 normalized command (31.25% of the full command)
# for runtime cross-track/heading feedback.  The corresponding curvature is
# looked up from the frozen 33-point response rather than duplicated here.
# This is a deterministic generation policy, not the contract qualification
# limit: the schema/runtime independently require the proof-derived |u|<=0.8.
CANDIDATE_SEARCH_MAX_NORMALIZED_STEER = 0.55
V5_INGRESS_HINTS: dict[str, tuple[tuple[float, float], ...]] = {
    # These are geometry parameters, never serialized paths.  Each hint still
    # goes through the complete axle/OBB/XODR/topology qualification.  The
    # stable ordering preserves the independently red-teamed blocker witnesses
    # and avoids an expensive blind prefix sweep on edge rows.
    "B101": ((2.0, 0.55),),
    "B102": ((3.0, 0.55),),
    "D402": ((2.0, 0.55),),
    "D403": ((3.0, 0.45),),
    "D404": ((10.0, 0.35),),
}
V5_SLOT_AXIS_DISTANCE_HINTS: dict[str, tuple[float, ...]] = {
    # D404 is an inactive counterfactual in the empty D-015 program.  Its
    # 2.5m slot-axis witness is fully exact-map/OBB qualified even though the
    # actor transform is just outside the 0.15m centre-screen heuristic.
    "D404": (2.5,),
}
# These two rows reproduce the already-qualified pre-live-sweep manoeuvres.
# Keeping the grid indices explicit avoids re-enumerating hundreds of
# thousands of equivalent templates whenever the suite is regenerated.  They
# are search hints only: the resulting paths still traverse every v4/v5 map,
# footprint, control, topology, and serialization proof below.
V5_SLOT_AXIS_BASE_HINTS: dict[
    str, tuple[tuple[float, int, int, float, int], ...]
] = {
    "B405": ((3.5, -30, 18, 0.35, -1),),
    "B501": ((3.5, -30, 18, 0.35, -1),),
}
_GEOMETRY_CACHE: dict[str, dict[str, Any]] = {}
_BASE_MANOEUVRE_CACHE: dict[
    tuple[str, ...],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
] = {}
_MANOEUVRE_CACHE: dict[
    tuple[str, ...],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
] = {}


class _NoParkingCandidate(RuntimeError):
    """Internal sentinel used to advance deterministic template phases."""


def source_paths(project_root: Path) -> tuple[Path, Path]:
    return (
        (project_root / BBOX_ASSET).resolve(),
        (project_root / VEHICLE_DYNAMICS_ASSET).resolve(),
    )


def _slot_collision_occupancy_registry(project_root: Path) -> dict[str, Any]:
    """Materialize the CARLA try-spawn terminal occupancy authority."""

    path = (project_root / SLOT_COLLISION_OCCUPANCY_ASSET).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_root = {
        "schema_version", "method", "town", "map_name", "vehicle_blueprint",
        "spawn_z_m", "repeat_count", "slots", "slots_sha256", "summary",
        "source_files", "runtime_authority", "listener_process_receipt_sha256",
        "scope_statement",
    }
    if not isinstance(value, dict) or set(value) != expected_root:
        raise RuntimeError("parking slot occupancy evidence has non-canonical shape")
    if (
        value["schema_version"] != "language2drive.parking_slot_collision_occupancy.v1"
        or value["method"] != "carla_try_spawn_release_ego_at_xodr_slot_terminal_v1"
        or value["town"] != "parkinglotfish-new3"
        or value["vehicle_blueprint"] != VEHICLE_BLUEPRINT
        or value["spawn_z_m"] != 0.5
        or type(value["repeat_count"]) is not int
        or value["repeat_count"] != 2
        or not isinstance(value["scope_statement"], str)
        or not value["scope_statement"]
    ):
        raise RuntimeError("parking slot occupancy evidence method differs")
    rows = value["slots"]
    if (
        not isinstance(rows, list)
        or [row.get("slot_number") for row in rows] != sorted(
            row.get("slot_number") for row in rows
        )
        or value["slots_sha256"] != stable_hash(rows)
    ):
        raise RuntimeError("parking slot occupancy rows are not canonical")
    clear = sorted(row["slot_number"] for row in rows if row.get("terminal_spawn_clear") is True)
    blocked = sorted(row["slot_number"] for row in rows if row.get("terminal_spawn_clear") is False)
    if any(
        not isinstance(row, dict)
        or set(row) != {"slot_number", "physical_space_id", "source_entity_id", "terminal_pose", "terminal_spawn_clear"}
        or type(row["terminal_spawn_clear"]) is not bool
        for row in rows
    ):
        raise RuntimeError("parking slot occupancy row shape differs")
    summary = value["summary"]
    if summary != {
        "slot_count": len(rows), "clear_count": len(clear), "blocked_count": len(blocked),
        "clear_slots_sha256": stable_hash(clear), "blocked_slots_sha256": stable_hash(blocked),
    }:
        raise RuntimeError("parking slot occupancy summary differs")
    source_files = value["source_files"]
    if not isinstance(source_files, dict) or set(source_files) != {
        "slot_registry", "runtime_manifest", "runtime_provenance",
        "available_maps_witness", "capture_tool",
    }:
        raise RuntimeError("parking slot occupancy source map differs")
    for item in source_files.values():
        if (
            not isinstance(item, dict) or set(item) != {"path", "sha256"}
            or not Path(item["path"]).is_file()
            or sha256_file(Path(item["path"])) != item["sha256"]
        ):
            raise RuntimeError("parking slot occupancy nested source is stale")
    return {
        "source_ref": str(path),
        "source_file_sha256": sha256_file(path),
        "schema_version": value["schema_version"],
        "method": value["method"],
        "town": value["town"],
        "map_name": value["map_name"],
        "vehicle_blueprint": value["vehicle_blueprint"],
        "spawn_z_m": value["spawn_z_m"],
        "repeat_count": value["repeat_count"],
        "slots_sha256": value["slots_sha256"],
        "slot_count": len(rows),
        "clear_count": len(clear),
        "blocked_count": len(blocked),
        "clear_slots": clear,
        "blocked_slots": blocked,
        "clear_slots_sha256": stable_hash(clear),
        "blocked_slots_sha256": stable_hash(blocked),
        "source_files": {
            key: {"path": item["path"], "sha256": item["sha256"]}
            for key, item in sorted(source_files.items())
        },
        "scope_statement": value["scope_statement"],
    }


def registry(project_root: Path) -> dict[str, Any]:
    """Return immutable source-level provenance for every parking v5 proof."""

    context = _geometry(project_root)
    vehicle_path = (project_root / VEHICLE_BBOX_ASSET).resolve()
    vehicle = json.loads(vehicle_path.read_text(encoding="utf-8"))
    if (
        vehicle.get("schema_version") != "language2drive.vehicle_local_bbox_evidence.v1"
        or vehicle.get("vehicle_blueprint") != VEHICLE_BLUEPRINT
        or stable_hash(vehicle.get("local_bounding_box"))
        != vehicle.get("local_bounding_box_sha256")
    ):
        raise RuntimeError("parking vehicle BBox evidence fails canonical closure")
    dynamics_path = (project_root / VEHICLE_DYNAMICS_ASSET).resolve()
    dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
    occupancy_registry = _slot_collision_occupancy_registry(project_root)
    wheel = dynamics.get("wheel_geometry", {})
    response = dynamics.get("normalized_steer_response", {})
    if (
        dynamics.get("schema_version") != "language2drive.vehicle_dynamics_evidence.v1"
        or dynamics.get("vehicle_blueprint") != VEHICLE_BLUEPRINT
        or sha256_file(dynamics_path)
        != "031643aa33e5f234ea6ba4c87a403908aee7942e322802b4b1b0e545d38dde47"
        or stable_hash(dynamics)
        != "d94392a5c4118dcb8f8660461267cfa1bb44e5f7c9fb111b14057d862d948791"
        or stable_hash(wheel) != dynamics.get("wheel_geometry_sha256")
        or stable_hash(response.get("canonical_samples"))
        != response.get("canonical_samples_sha256")
        or abs(float(wheel.get("front_axle_x_m", math.nan)) - FRONT_AXLE_X_M) > 1e-9
        or abs(float(wheel.get("rear_axle_x_m", math.nan)) - REAR_AXLE_X_M) > 1e-9
        or abs(float(wheel.get("wheelbase_m", math.nan)) - WHEELBASE_M) > 1e-9
    ):
        raise RuntimeError("parking vehicle dynamics evidence fails canonical closure")
    return {
        "schema_version": "language2drive.parking_manoeuvre_registry.v5",
        "method": "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v5",
        "towns": {
            "parkinglotfish-new3": {
                "town": "parkinglotfish-new3",
                "bbox_asset": {
                    "source_ref": str(context["asset_path"]),
                    "source_file_sha256": context["asset_file_sha256"],
                    "schema_version": "language2drive.carla_level_bboxes.v1",
                    "bboxes_sha256": context["asset"]["bboxes_sha256"],
                    "bbox_count": context["asset"]["bbox_count"],
                    "categories": context["asset"]["categories"],
                    "scope_statement": (
                        "Complete 2D ego footprints are checked against CARLA world.get_level_bbs "
                        "for the frozen non-road semantic categories only; this is not a claim about "
                        "unlabelled Unreal collision meshes. Live three-seed collision witnesses "
                        "remain authoritative."
                    ),
                    "excluded_scope": context["asset"]["excluded_scope"],
                    "quantization_decimals": context["asset"]["quantization_decimals"],
                    "canonical_sort_key": context["asset"]["canonical_sort_key"],
                    "source_api": context["asset"]["provenance"]["source_api"],
                },
                "xodr": {"source_ref": str(context["xodr_path"]), "sha256": context["xodr_sha256"]},
                "lane_graph": {"source_ref": str(context["lane_graph_path"]), "sha256": context["lane_graph_sha256"]},
                "control_model": {
                    "schema_version": "language2drive.parking_axle_control_model.v4",
                    "pose_reference": "carla_actor_transform",
                    "integration_reference": "rear_axle_center",
                    "vehicle_dynamics_source_ref": str(dynamics_path),
                    "vehicle_dynamics_source_file_sha256": sha256_file(dynamics_path),
                    "wheel_geometry_sha256": dynamics["wheel_geometry_sha256"],
                    "normalized_steer_samples_sha256": response["canonical_samples_sha256"],
                    "front_axle_x_m": FRONT_AXLE_X_M,
                    "rear_axle_x_m": REAR_AXLE_X_M,
                    "wheelbase_m": WHEELBASE_M,
                    "max_normalized_control_steer": MAX_NORMALIZED_CONTROL_STEER,
                },
                "vehicle_geometry": {
                    "vehicle_blueprint": VEHICLE_BLUEPRINT,
                    "coordinate_frame": "carla_actor_local",
                    "local_bbox": vehicle["local_bounding_box"],
                    "local_bbox_sha256": stable_hash(vehicle["local_bounding_box"]),
                    "source_method": "constant_bbox_across_authoritative_trace_v1",
                    "source_trace_ref": vehicle["source_files"]["trace.jsonl"]["path"],
                    "source_trace_sha256": vehicle["source_files"]["trace.jsonl"]["sha256"],
                    "source_trace_rows": vehicle["observation"]["sample_count"],
                    "matching_bbox_rows": vehicle["observation"]["consistent_sample_count"],
                    "actor_reference_z_min_m": vehicle["observation"]["ego_reference_z_min"],
                    "actor_reference_z_max_m": vehicle["observation"]["ego_reference_z_max"],
                },
                "slot_collision_occupancy": occupancy_registry,
            }
        },
    }


def _world_vertical_half_extent(extent: dict[str, Any], rotation: dict[str, Any]) -> float:
    pitch = math.radians(float(rotation["pitch"])); roll = math.radians(float(rotation["roll"]))
    return (
        abs(-math.sin(pitch)) * float(extent["x"])
        + abs(math.cos(pitch) * math.sin(roll)) * float(extent["y"])
        + abs(math.cos(pitch) * math.cos(roll)) * float(extent["z"])
    )


def _overlap_index_sets(context: dict[str, Any], poses: list[dict[str, Any]]) -> list[list[int]]:
    local = context["vehicle"]["local_bounding_box"]
    ego_half = _world_vertical_half_extent(local["extent"], local["rotation"])
    result: list[list[int]] = []
    for pose in poses:
        z_key = round(float(pose["z"]), 6)
        cached = context["overlap_by_z"].get(z_key)
        if cached is None:
            observation = context["vehicle"]["observation"]
            ego_min = (z_key + float(observation["ego_reference_z_min"])
                       + float(local["location"]["z"]) - ego_half)
            ego_max = (z_key + float(observation["ego_reference_z_max"])
                       + float(local["location"]["z"]) + ego_half)
            cached = [
                index for index, (static_min, static_max) in enumerate(context["static_z_intervals"])
                if (ego_max + VERTICAL_PADDING_M >= static_min - VERTICAL_PADDING_M - 1e-9
                    and static_max + VERTICAL_PADDING_M >= ego_min - VERTICAL_PADDING_M - 1e-9)
            ]
            context["overlap_by_z"][z_key] = cached
        result.append(list(cached))
    return result


def _geometry(project_root: Path) -> dict[str, Any]:
    key = str(project_root.resolve())
    if key in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[key]
    try:
        from shapely.geometry import LineString, Point, Polygon
        from shapely.ops import unary_union
        from shapely.strtree import STRtree
    except ImportError as error:
        raise RuntimeError("parking v2 generation requires Shapely >= 2.0") from error

    asset_path, _ = source_paths(project_root)
    lane_path = (project_root / "generated/maps/lane_graphs/parkinglotfish-new3.lane_graph.json").resolve()
    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    if (
        asset.get("schema_version") != "language2drive.carla_level_bboxes.v1"
        or asset.get("town") != "parkinglotfish-new3"
        or asset.get("categories") != BBOX_CATEGORIES
        or int(asset.get("bbox_count", -1)) != len(asset.get("bboxes", []))
        or stable_hash(asset.get("bboxes")) != asset.get("bboxes_sha256")
    ):
        raise RuntimeError("parking static BBox asset fails canonical closure")
    maps = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    xodr_path = (Path(maps["xodr_root"]) / "parkinglotfish-new3.xodr").resolve()
    xodr_hash = sha256_file(xodr_path)
    if asset.get("provenance", {}).get("registered_xodr_sha256") != xodr_hash:
        raise RuntimeError("parking BBox asset does not match registered XODR")

    graph = json.loads(lane_path.read_text(encoding="utf-8"))
    surfaces = []
    for lane in graph.get("lanes", []):
        samples = lane.get("samples", [])
        if len(samples) < 2:
            continue
        width = min(float(sample.get("lane_width_m", 0.0)) for sample in samples)
        if width <= 0:
            raise RuntimeError(f"parking lane {lane.get('id')} has non-positive width")
        surfaces.append(LineString([
            (float(sample["x"]), float(sample["y"])) for sample in samples
        ]).buffer(
            width / 2 + LANE_SURFACE_NUMERIC_TOLERANCE_M,
            cap_style="flat", join_style="round",
        ))
    if not surfaces:
        raise RuntimeError("parking lane graph contains no surface geometry")

    def rotation_matrix(rotation: dict[str, Any]) -> list[list[float]]:
        pitch = math.radians(float(rotation["pitch"])); yaw = math.radians(float(rotation["yaw"]))
        roll = math.radians(float(rotation["roll"]))
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        # Rz(yaw) @ Ry(pitch) @ Rx(roll)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    def bbox_polygon(value: dict[str, Any]) -> Any:
        center, extent, rotation = value["center"], value["extent"], value["rotation"]
        matrix = rotation_matrix(rotation)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local = [sx * float(extent["x"]), sy * float(extent["y"]), sz * float(extent["z"])]
                    corners.append((
                        float(center["x"]) + sum(matrix[0][axis] * local[axis] for axis in range(3)),
                        float(center["y"]) + sum(matrix[1][axis] * local[axis] for axis in range(3)),
                    ))
        return Polygon(corners).convex_hull

    obstacles = [bbox_polygon(value) for value in asset["bboxes"]]
    static_z_intervals = []
    for value in asset["bboxes"]:
        half = _world_vertical_half_extent(value["extent"], value["rotation"])
        center_z = float(value["center"]["z"])
        static_z_intervals.append((center_z - half, center_z + half))
    vehicle_path = (project_root / VEHICLE_BBOX_ASSET).resolve()
    vehicle = json.loads(vehicle_path.read_text(encoding="utf-8"))
    dynamics_path = (project_root / VEHICLE_DYNAMICS_ASSET).resolve()
    dynamics = json.loads(dynamics_path.read_text(encoding="utf-8"))
    response = dynamics.get("normalized_steer_response", {})
    if (
        sha256_file(dynamics_path)
        != "031643aa33e5f234ea6ba4c87a403908aee7942e322802b4b1b0e545d38dde47"
        or stable_hash(dynamics)
        != "d94392a5c4118dcb8f8660461267cfa1bb44e5f7c9fb111b14057d862d948791"
        or stable_hash(response.get("canonical_samples"))
        != response.get("canonical_samples_sha256")
    ):
        raise RuntimeError("parking vehicle dynamics asset is stale or invalid")
    result = {
        "asset": asset, "asset_path": asset_path,
        "asset_file_sha256": sha256_file(asset_path),
        "lane_graph_path": lane_path, "lane_graph_sha256": sha256_file(lane_path),
        "xodr_path": xodr_path, "xodr_sha256": xodr_hash,
        "vehicle_path": vehicle_path, "vehicle": vehicle,
        "dynamics_path": dynamics_path, "dynamics": dynamics,
        "steer_samples": response["canonical_samples"],
        "lane_surface": unary_union(surfaces), "obstacles": obstacles,
        "obstacle_tree": STRtree(obstacles), "Polygon": Polygon, "Point": Point,
        "static_z_intervals": static_z_intervals, "overlap_by_z": {},
        "lane_point_cache": {},
    }
    _GEOMETRY_CACHE[key] = result
    return result


def _footprint(context: dict[str, Any], x: float, y: float, yaw: float) -> Any:
    local = context["vehicle"]["local_bounding_box"]
    half_length, half_width = float(local["extent"]["x"]), float(local["extent"]["y"])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    center_x = x + cosine * float(local["location"]["x"]) - sine * float(local["location"]["y"])
    center_y = y + sine * float(local["location"]["x"]) + cosine * float(local["location"]["y"])
    return context["Polygon"]([
        (center_x + cosine * dx - sine * dy, center_y + sine * dx + cosine * dy)
        for dx, dy in (
            (half_length, half_width), (half_length, -half_width),
            (-half_length, -half_width), (-half_length, half_width),
        )
    ])


def _bezier(
    start: Any, end: Any, start_yaw: float, end_yaw: float, fraction: float,
    *, screening: bool = False,
) -> tuple[Any, Any, float]:
    import numpy as np

    distance = float(np.linalg.norm(end - start))
    first = start + np.array([math.cos(start_yaw), math.sin(start_yaw)]) * distance * fraction
    second = end - np.array([math.cos(end_yaw), math.sin(end_yaw)]) * distance * fraction
    if screening:
        # Screening only rejects clearly over-curved templates.  Eleven
        # deterministic samples are sufficient here because every accepted
        # trajectory is resampled at <=0.075 m and remeasured exactly below.
        parameters = np.linspace(0.0, 1.0, 11)
    else:
        dense = np.linspace(0.0, 1.0, 1001); complement = 1.0 - dense
        points = (complement[:, None] ** 3 * start
                  + 3 * complement[:, None] ** 2 * dense[:, None] * first
                  + 3 * complement[:, None] * dense[:, None] ** 2 * second
                  + dense[:, None] ** 3 * end)
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        count = max(1, int(math.ceil(float(cumulative[-1]) / SAMPLE_STEP_M)))
        parameters = np.interp(np.linspace(0.0, float(cumulative[-1]), count + 1), cumulative, dense)
    complement = 1.0 - parameters
    points = (complement[:, None] ** 3 * start
              + 3 * complement[:, None] ** 2 * parameters[:, None] * first
              + 3 * complement[:, None] * parameters[:, None] ** 2 * second
              + parameters[:, None] ** 3 * end)
    derivative = (3 * complement[:, None] ** 2 * (first - start)
                  + 6 * complement[:, None] * parameters[:, None] * (second - first)
                  + 3 * parameters[:, None] ** 2 * (end - second))
    second_derivative = (6 * complement[:, None] * (second - 2 * first + start)
                         + 6 * parameters[:, None] * (end - 2 * second + first))
    speeds = np.linalg.norm(derivative, axis=1)
    if float(speeds.min()) <= 1e-8:
        return points, np.arctan2(derivative[:, 1], derivative[:, 0]), math.inf
    curvature = np.abs(derivative[:, 0] * second_derivative[:, 1]
                       - derivative[:, 1] * second_derivative[:, 0]) / speeds ** 3
    return points, np.arctan2(derivative[:, 1], derivative[:, 0]), float(curvature.max())


def _circle(
    stage: Any, movement_yaw: float, goal_yaw: float, sign: int, radius: float,
    *, screening: bool = False,
) -> tuple[Any, Any]:
    import numpy as np

    delta = ((goal_yaw - movement_yaw) % (2 * math.pi)
             if sign > 0 else (movement_yaw - goal_yaw) % (2 * math.pi))
    count = (min(25, max(2, int(math.ceil(radius * delta / 0.3)))) if screening
             else max(1, int(math.ceil(radius * delta / SAMPLE_STEP_M))))
    headings = movement_yaw + np.linspace(0.0, sign * delta, count + 1)
    points = np.column_stack((
        float(stage[0]) + radius / sign * (np.sin(headings) - math.sin(movement_yaw)),
        float(stage[1]) + radius / sign * (-np.cos(headings) + math.cos(movement_yaw)),
    ))
    return points, headings - math.pi


def _actor_pose_from_rear(rear: dict[str, Any]) -> dict[str, Any]:
    """Serialize a CARLA actor transform from one serialized rear-axle pose."""

    yaw = math.radians(float(rear["yaw_deg"]))
    # rear = actor + rear_axle_x * heading; rear_axle_x is negative.
    return {
        "x": round(float(rear["x"]) - REAR_AXLE_X_M * math.cos(yaw), 4),
        "y": round(float(rear["y"]) - REAR_AXLE_X_M * math.sin(yaw), 4),
        "z": rear["z"],
        "yaw_deg": rear["yaw_deg"],
        "gear": rear["gear"],
    }


def _canonical_yaw_deg(value: float) -> float:
    """Serialize equivalent headings once in the canonical [-180, 180) range."""

    return round((math.degrees(float(value)) + 180.0) % 360.0 - 180.0, 3)


def _rear_pose_from_actor(actor: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the frozen rear reference from an actor pose."""

    yaw = math.radians(float(actor["yaw_deg"]))
    return {
        "x": round(float(actor["x"]) + REAR_AXLE_X_M * math.cos(yaw), 6),
        "y": round(float(actor["y"]) + REAR_AXLE_X_M * math.sin(yaw), 6),
        "z": actor["z"],
        "yaw_deg": actor["yaw_deg"],
        "gear": actor["gear"],
    }


def _poses(
    forward_rear_points: Any, forward_yaws: Any,
    reverse_rear_points: Any, reverse_yaws: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize rear-axle states first, then derive exact actor transforms.

    Rear x/y use six decimals and yaw uses three.  Actor x/y are Python
    ``round(..., 4)`` of those serialized rear states.  Both lists retain the
    same-pose forward-to-reverse transition.
    """

    rear = [
        {"x": round(float(point[0]), 6), "y": round(float(point[1]), 6), "z": 0.0,
         "yaw_deg": _canonical_yaw_deg(float(yaw)), "gear": "forward"}
        for point, yaw in zip(forward_rear_points, forward_yaws)
    ]
    transition = dict(rear[-1]); transition["gear"] = "reverse"; rear.append(transition)
    rear.extend(
        {"x": round(float(point[0]), 6), "y": round(float(point[1]), 6), "z": 0.0,
         "yaw_deg": _canonical_yaw_deg(float(yaw)), "gear": "reverse"}
        for point, yaw in zip(reverse_rear_points[1:], reverse_yaws[1:])
    )
    # The terminal contract is expressed in actor-transform coordinates.
    # Snap the serialized rear state through that exact four-decimal actor
    # target so rear->actor reconstruction cannot land one rounding unit away
    # from the canonical slot centroid.
    terminal_yaw = math.radians(float(rear[-1]["yaw_deg"]))
    raw_terminal_yaw = float(reverse_yaws[-1])
    terminal_actor = {
        "x": round(float(reverse_rear_points[-1][0])
                   - REAR_AXLE_X_M * math.cos(raw_terminal_yaw), 4),
        "y": round(float(reverse_rear_points[-1][1])
                   - REAR_AXLE_X_M * math.sin(raw_terminal_yaw), 4),
        "z": 0.0, "yaw_deg": rear[-1]["yaw_deg"], "gear": "reverse",
    }
    rear[-1] = _rear_pose_from_actor(terminal_actor)
    return [_actor_pose_from_rear(pose) for pose in rear], rear


def _normalized_steer_for_curvature(
    samples: list[dict[str, Any]], curvature: float,
) -> float:
    """Piecewise-invert the frozen signed rear-curvature response."""

    requested = float(curvature)
    values = [
        (float(value["ackermann_center_curvature_1pm"]),
         float(value["normalized_steer"]))
        for value in samples
    ]
    if requested < values[0][0] - 1e-12 or requested > values[-1][0] + 1e-12:
        raise ValueError(f"rear curvature {requested} is outside frozen steer response")
    for (first_k, first_u), (second_k, second_u) in zip(values, values[1:]):
        if first_k - 1e-12 <= requested <= second_k + 1e-12:
            if abs(second_k - first_k) <= 1e-15:
                raise ValueError("frozen steer response contains duplicate curvature")
            fraction = (requested - first_k) / (second_k - first_k)
            return first_u + fraction * (second_u - first_u)
    if abs(requested - values[-1][0]) <= 1e-12:
        return values[-1][1]
    raise ValueError(f"cannot invert rear curvature {requested}")


def _curvature_at_normalized_steer(
    samples: list[dict[str, Any]], normalized_steer: float,
) -> float:
    """Return one exact frozen response breakpoint, failing closed."""

    matches = [
        float(value["ackermann_center_curvature_1pm"])
        for value in samples
        if abs(float(value["normalized_steer"]) - normalized_steer) <= 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"frozen steering response lacks unique u={normalized_steer} breakpoint"
        )
    return matches[0]


def _audit(
    context: dict[str, Any], actor_poses: list[dict[str, Any]],
    rear_poses: list[dict[str, Any]], polygon: list[list[float]],
    overlap_sets: list[list[int]],
) -> tuple[float, float, float] | None:
    slot = context["Polygon"](polygon).buffer(FOOTPRINT_TOLERANCE_M)
    legal = context["lane_surface"].union(slot)
    maximum_curvature, required_max_steer = 0.0, 0.0
    minimum_clearance, transitions = math.inf, 0
    if len(actor_poses) != len(rear_poses):
        return None
    for index, (pose, rear_pose) in enumerate(zip(actor_poses, rear_poses)):
        if _actor_pose_from_rear(rear_pose) != pose:
            return None
        yaw = math.radians(float(pose["yaw_deg"])); footprint = _footprint(
            context, float(pose["x"]), float(pose["y"]), yaw,
        )
        if not legal.covers(footprint):
            return None
        clearance = min(
            (footprint.distance(context["obstacles"][item]) for item in overlap_sets[index]),
            default=math.inf,
        )
        minimum_clearance = min(minimum_clearance, float(clearance))
        if clearance < STATIC_CLEARANCE_M - 1e-9:
            return None
        if index == 0:
            continue
        previous = actor_poses[index - 1]
        previous_rear = rear_poses[index - 1]
        actor_distance = math.dist(
            tuple(float(previous[a]) for a in ("x", "y", "z")),
            tuple(float(pose[a]) for a in ("x", "y", "z")),
        )
        rear_distance = math.dist(
            tuple(float(previous_rear[a]) for a in ("x", "y", "z")),
            tuple(float(rear_pose[a]) for a in ("x", "y", "z")),
        )
        signed_yaw_step = (
            (float(rear_pose["yaw_deg"]) - float(previous_rear["yaw_deg"]) + 180)
            % 360 - 180
        )
        yaw_step = abs(signed_yaw_step)
        if previous_rear["gear"] != rear_pose["gear"]:
            transitions += 1
            if (
                previous_rear["gear"] != "forward"
                or rear_pose["gear"] != "reverse"
                or any(previous_rear[field] != rear_pose[field]
                       for field in ("x", "y", "z", "yaw_deg"))
                or any(previous[field] != pose[field]
                       for field in ("x", "y", "z", "yaw_deg"))
            ):
                return None
            continue
        if (
            not 1e-6 < actor_distance <= MANOEUVRE_SPACING_M + 1e-9
            or not 1e-6 < rear_distance <= MANOEUVRE_SPACING_M + 1e-9
            or yaw_step > 5 + 1e-9
        ):
            return None
        tangent = math.degrees(math.atan2(
            float(rear_pose["y"]) - float(previous_rear["y"]),
            float(rear_pose["x"]) - float(previous_rear["x"]),
        ))
        expected = float(previous_rear["yaw_deg"]) + (
            180 if previous_rear["gear"] == "reverse" else 0
        )
        if abs((tangent - expected + 180) % 360 - 180) > 5 + 1e-9:
            return None
        signed_distance = rear_distance * (
            -1.0 if previous_rear["gear"] == "reverse" else 1.0
        )
        signed_curvature = math.radians(signed_yaw_step) / signed_distance
        try:
            normalized = _normalized_steer_for_curvature(
                context["steer_samples"], signed_curvature,
            )
        except ValueError:
            return None
        maximum_curvature = max(maximum_curvature, abs(signed_curvature))
        required_max_steer = max(required_max_steer, abs(normalized))
    if transitions != 1:
        return None
    terminal = actor_poses[-1]
    if not slot.covers(_footprint(context, float(terminal["x"]), float(terminal["y"]),
                                  math.radians(float(terminal["yaw_deg"])))):
        return None
    if required_max_steer > MAX_NORMALIZED_CONTROL_STEER + 1e-9:
        return None
    return maximum_curvature, required_max_steer, minimum_clearance


def _quick_audit(
    context: dict[str, Any], poses: list[dict[str, Any]], legal: Any,
    overlap_sets: list[list[int]],
) -> bool:
    """Reject coarse candidates without performing hundreds of OBB distances."""

    indices = sorted(set(range(0, len(poses), 8)) | {len(poses) - 1})
    for index in indices:
        pose = poses[index]
        footprint = _footprint(
            context, float(pose["x"]), float(pose["y"]),
            math.radians(float(pose["yaw_deg"])),
        )
        if not legal.covers(footprint):
            return False
        if any(footprint.distance(context["obstacles"][item]) < STATIC_CLEARANCE_M - 1e-9
               for item in overlap_sets[index]):
            return False
    return True


def _exact_map_footprint_audit(
    context: dict[str, Any], carla_map: Any, poses: list[dict[str, Any]],
    polygon: list[list[float]],
) -> bool:
    """Recheck each accepted footprint against the authoritative offline map.

    The lane-graph surface is only a fast candidate screen.  Qualification
    samples the complete local OBB on a <=0.2 m boundary/interior lattice and
    requires every sample to be in the own-slot tolerance polygon or resolve
    without projection to an exact Driving/Parking waypoint.
    """

    import carla
    import numpy as np

    local = context["vehicle"]["local_bounding_box"]
    ex, ey = float(local["extent"]["x"]), float(local["extent"]["y"])
    lx, ly = float(local["location"]["x"]), float(local["location"]["y"])
    x_values = np.linspace(-ex, ex, max(1, int(math.ceil(2 * ex / 0.2))) + 1)
    y_values = np.linspace(-ey, ey, max(1, int(math.ceil(2 * ey / 0.2))) + 1)
    local_samples = [(lx + float(dx), ly + float(dy)) for dx in x_values for dy in y_values]
    own_slot = context["Polygon"](polygon).buffer(FOOTPRINT_TOLERANCE_M)
    lane_types = (carla.LaneType.Driving, carla.LaneType.Parking)
    cache = context["lane_point_cache"]
    for pose in poses:
        yaw = math.radians(float(pose["yaw_deg"])); cosine, sine = math.cos(yaw), math.sin(yaw)
        for local_x, local_y in local_samples:
            world_x = float(pose["x"]) + cosine * local_x - sine * local_y
            world_y = float(pose["y"]) + sine * local_x + cosine * local_y
            point = context["Point"](world_x, world_y)
            if own_slot.covers(point):
                continue
            key = (round(world_x, 4), round(world_y, 4), round(float(pose["z"]), 3))
            legal = cache.get(key)
            if legal is None:
                location = carla.Location(x=world_x, y=world_y, z=float(pose["z"]))
                legal = any(
                    carla_map.get_waypoint(location, project_to_road=False, lane_type=lane_type)
                    is not None for lane_type in lane_types
                )
                cache[key] = legal
            if not legal:
                return False
    return True


def _mine_slot_axis_manoeuvre_v4(
    project_root: Path,
    target: dict[str, Any],
    carla_map: Any,
    waypoint_lookup: Callable[[Any, dict[str, Any]], Any],
    *,
    _template_override: list[tuple[float, int, float, float, int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Mine the immutable slot-axis-to-terminal rear-axle suffix.

    This private stage intentionally retains the v4 wire shape.  ``mine``
    prepends the proof-bound v5 ingress and rewrites every versioned field
    before anything can leave the producer.
    """

    import numpy as np

    context = _geometry(project_root)
    cache_key = (
        str(target["slot_number"]), stable_hash(target["polygon"]),
        context["asset"]["bboxes_sha256"],
        stable_hash(context["vehicle"]["local_bounding_box"]),
        context["dynamics"]["normalized_steer_response"]["canonical_samples_sha256"],
        f"control-margin-u<={CANDIDATE_SEARCH_MAX_NORMALIZED_STEER:.2f}",
    )
    if cache_key in _BASE_MANOEUVRE_CACHE:
        return tuple(json.loads(json.dumps(value)) for value in _BASE_MANOEUVRE_CACHE[cache_key])  # type: ignore[return-value]

    vertices = target["polygon"][:-1] if target["polygon"][0] == target["polygon"][-1] else target["polygon"]
    center = {"x": sum(float(p[0]) for p in vertices) / len(vertices),
              "y": sum(float(p[1]) for p in vertices) / len(vertices)}
    slot_heading_deg = float(target["heading_deg"]); slot_heading = math.radians(slot_heading_deg)
    goal_movement = slot_heading + math.pi
    terminal_actor = np.array([center["x"], center["y"]])
    terminal_rear = terminal_actor + REAR_AXLE_X_M * np.array([
        math.cos(slot_heading), math.sin(slot_heading),
    ])
    max_curvature = _curvature_at_normalized_steer(
        context["steer_samples"], CANDIDATE_SEARCH_MAX_NORMALIZED_STEER,
    )
    min_radius = 1 / max_curvature
    if _template_override is None:
        # These geometry-only hints cover the recurring parking-row layouts.
        # Each is still serialized and passed through the complete proof.  A
        # failed hint raises only the private sentinel and advances to the
        # next; the expensive complete grid is materialized only if all fail.
        # A303 sits beside a low traffic-sign BBox at the shortest-path handoff.
        # Its deterministic grid witness below preserves >0.80 m pointwise
        # global/transition clearance while remaining well inside the frozen
        # u<=0.55 search policy.  It is tried only for that physical slot and
        # still passes the same full footprint, level-BBox and exact-map audits;
        # no serialized pose or proof is hard-coded.
        slot_hints = {
            "A303": [
                (4.5, -30, min_radius + 0.02 + 16 * 0.2, 0.35, -1),
            ],
        }
        hints = [
            (5.0, -30, 8.3, 0.35, -1),
            (5.0, -30, 8.4, 0.35, -1),
            (4.0, -30, 7.3, 0.35, -1),
            (4.5, -30, 7.7, 0.35, -1),
            (4.75, -30, 7.9, 0.35, -1),
            (4.75, -30, 8.0, 0.35, -1),
            (4.75, -30, 8.1, 0.35, -1),
            (5.0, -30, 8.2, 0.35, -1),
            (3.0, -38, 7.0, 0.35, -1),
            (3.0, -38, 7.1, 0.35, -1),
            (4.0, -20, 10.8, 0.35, -1),
            (4.0, -30, 7.4, 0.35, -1),
        ]
        for distance, offset, radius_index, sign in (
            (4.75, 8, 4, 1), (5.0, 22, 13, 1),
            (5.25, -2, 1, 1),
        ):
            radius = min_radius + 0.02 + radius_index * 0.2
            hints.extend(
                (distance, offset, radius, fraction, sign)
                for fraction in (0.25, 0.35, 0.45, 0.55)
            )
        for hint in [*slot_hints.get(str(target["slot_number"]), []), *hints]:
            try:
                return _mine_slot_axis_manoeuvre_v4(
                    project_root, target, carla_map, waypoint_lookup,
                    _template_override=[hint],
                )
            except _NoParkingCandidate:
                pass
    screen_legal = context["lane_surface"].union(
        context["Polygon"](target["polygon"]).buffer(FOOTPRINT_TOLERANCE_M)
    )
    candidates: list[tuple[Any, ...]] = []
    templates = _template_override if _template_override is not None else [
        (distance_index * 0.25, offset_deg,
         min_radius + 0.02 + radius_index * 0.2, fraction, sign)
        for distance_index in range(8, 49)
        for offset_deg in range(-30, 31, 2)
        for sign in (-1, 1)
        for radius_index in range(20)
        for fraction in (0.25, 0.35, 0.45, 0.55)
    ]
    waypoint_by_distance: dict[float, Any] = {}
    for distance, offset_deg, radius, fraction, sign in templates:
        approach_actor = np.array([center["x"] + distance * math.cos(slot_heading),
                                   center["y"] + distance * math.sin(slot_heading)])
        if distance not in waypoint_by_distance:
            waypoint_by_distance[distance] = waypoint_lookup(carla_map, {
                "x": float(approach_actor[0]), "y": float(approach_actor[1]), "z": 0.0,
            })
        waypoint = waypoint_by_distance[distance]
        if waypoint is None:
            continue
        lane_yaw = math.radians(float(waypoint.transform.rotation.yaw))
        lane_direction = np.array([math.cos(lane_yaw), math.sin(lane_yaw)])
        approach_rear = approach_actor + REAR_AXLE_X_M * lane_direction
        stage_yaw = lane_yaw + math.radians(offset_deg); movement = stage_yaw + math.pi
        delta = ((goal_movement - movement) % (2 * math.pi)
                 if sign > 0 else (movement - goal_movement) % (2 * math.pi))
        if delta > math.pi:
            continue
        displacement = np.array([
            radius / sign * (math.sin(goal_movement) - math.sin(movement)),
            radius / sign * (-math.cos(goal_movement) + math.cos(movement)),
        ])
        stage_rear = terminal_rear - displacement
        if float(np.dot(stage_rear - approach_rear, lane_direction)) < 0.1:
            continue
        points, _, curvature = _bezier(
            approach_rear, stage_rear, lane_yaw, stage_yaw,
            fraction, screening=True,
        )
        if curvature > max_curvature + 1e-9:
            continue
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) + radius * delta
        candidates.append((round(length, 9), distance, int(waypoint.road_id),
                           int(waypoint.section_id), int(waypoint.lane_id), lane_yaw,
                           offset_deg, radius, fraction, approach_actor,
                           approach_rear, stage_rear, stage_yaw, sign))
    slot_surface = context["Polygon"](target["polygon"]).buffer(FOOTPRINT_TOLERANCE_M)
    legal_surface = context["lane_surface"].union(slot_surface)
    # Stable shortest-first, then exact proof: this is both reproducible and
    # avoids an O(all-candidates × poses × obstacles) global ranking pass.
    candidates.sort(key=lambda item: item[:9])
    rejection_counts = {
        "centreline": 0, "quick_footprint": 0, "full_audit": 0,
        "control_headroom": 0, "exact_map": 0,
    }
    for candidate in candidates:
        (_, _, road_id, section_id, lane_id, lane_yaw, _, radius, fraction,
         approach_actor_raw, approach_rear, stage_rear, stage_yaw, sign) = candidate
        forward_points, forward_yaws, _ = _bezier(
            approach_rear, stage_rear, lane_yaw, stage_yaw, fraction,
        )
        reverse_points, reverse_yaws = _circle(
            stage_rear, stage_yaw + math.pi, goal_movement, sign, radius,
        )
        # Cheap deterministic centreline screening prevents thousands of
        # candidates from reaching the much more expensive complete-footprint
        # and static-OBB audit.  It is only an optimization: every accepted
        # candidate is still rechecked by ``_audit`` after 4-decimal
        # serialization.
        screening_points = list(forward_points[::4]) + list(reverse_points[::4])
        if not all(legal_surface.covers(context["Point"](float(point[0]), float(point[1])))
                   for point in screening_points):
            rejection_counts["centreline"] += 1
            continue
        poses, rear_poses = _poses(
            forward_points, forward_yaws, reverse_points, reverse_yaws,
        )
        overlap_sets = _overlap_index_sets(context, poses)
        if not _quick_audit(context, poses, legal_surface, overlap_sets):
            rejection_counts["quick_footprint"] += 1
            continue
        audit = _audit(
            context, poses, rear_poses, target["polygon"], overlap_sets,
        )
        if audit is None:
            rejection_counts["full_audit"] += 1
            continue
        if audit[1] > CANDIDATE_SEARCH_MAX_NORMALIZED_STEER + 1e-9:
            rejection_counts["control_headroom"] += 1
            continue
        if not _exact_map_footprint_audit(context, carla_map, poses, target["polygon"]):
            rejection_counts["exact_map"] += 1
            continue
        measured_curvature, required_steer, clearance = audit
        approach = {key: poses[0][key] for key in ("x", "y", "z", "yaw_deg")}
        if math.hypot(
            float(approach["x"]) - float(approach_actor_raw[0]),
            float(approach["y"]) - float(approach_actor_raw[1]),
        ) > 0.0001 + 1e-9:
            raise RuntimeError("rear-to-actor serialization moved the exact parking approach")
        lane_yaw_deg = _canonical_yaw_deg(lane_yaw)
        heading_error = abs((lane_yaw_deg - slot_heading_deg + 180) % 360 - 180)
        distance = math.hypot(float(approach["x"]) - center["x"], float(approach["y"]) - center["y"])
        approach_proof = {
            "method": "slot_axis_lane_intersection_v2",
            "distance_from_slot_center_m": round(distance, 6),
            "heading_error_deg": round(heading_error, 3),
            "lane_id": lane_id, "lane_yaw_deg": lane_yaw_deg,
            "road_id": road_id, "section_id": section_id,
            "reverse": {"method": "audited_reverse_manoeuvre_v4",
                        "own_slot_tolerance_m": FOOTPRINT_TOLERANCE_M,
                        "sample_spacing_max_m": MANOEUVRE_SPACING_M, "samples": len(poses)},
        }
        checked_pairs = len(poses) * int(context["asset"]["bbox_count"])
        overlapping_pairs = sum(len(indices) for indices in overlap_sets)
        vertical_proof = {
            "method": "carla_obb_world_z_interval_overlap_v1",
            "vertical_padding_m": VERTICAL_PADDING_M,
            "ego_local_bbox_sha256": stable_hash(
                json.loads(context["vehicle_path"].read_text(encoding="utf-8"))["local_bounding_box"]
            ),
            "pose_count": len(poses),
            "static_bbox_count": int(context["asset"]["bbox_count"]),
            "checked_pair_count": checked_pairs,
            "vertically_overlapping_pair_count": overlapping_pairs,
            "vertically_disjoint_pair_count": checked_pairs - overlapping_pairs,
            "overlap_index_sets_sha256": stable_hash(overlap_sets),
        }
        manoeuvre = {
            "schema_version": "language2drive.parking_reverse_manoeuvre.v4",
            "poses": poses, "rear_axle_poses": rear_poses,
            "proof": {
                "method": "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v4",
                "poses_sha256": stable_hash(poses), "pose_count": len(poses),
                "sample_spacing_max_m": MANOEUVRE_SPACING_M,
                "pose_reference": "carla_actor_transform",
                "integration_reference": "rear_axle_center",
                "rear_axle_poses_sha256": stable_hash(rear_poses),
                "rear_axle_sample_spacing_max_m": MANOEUVRE_SPACING_M,
                "required_max_rear_axle_curvature_1pm": measured_curvature,
                "required_max_abs_normalized_steer": required_steer,
                "max_normalized_control_steer": MAX_NORMALIZED_CONTROL_STEER,
                "vehicle_blueprint": VEHICLE_BLUEPRINT,
                "footprint_length_m": FOOTPRINT_LENGTH_M, "footprint_width_m": FOOTPRINT_WIDTH_M,
                "footprint_tolerance_m": FOOTPRINT_TOLERANCE_M,
                "full_footprint_legal_samples": len(poses), "static_clearance_samples": len(poses),
                "static_collision_samples": 0,
                "static_clearance_min_m": (
                    STATIC_CLEARANCE_M if math.isinf(clearance)
                    else math.floor(clearance * 1_000_000) / 1_000_000
                ),
                "static_bbox_categories": BBOX_CATEGORIES,
                "static_bbox_count": int(context["asset"]["bbox_count"]),
                "static_bboxes_sha256": context["asset"]["bboxes_sha256"],
                "town": "parkinglotfish-new3", "xodr_sha256": context["xodr_sha256"],
                "lane_graph_sha256": context["lane_graph_sha256"],
                "vertical_overlap_proof": vertical_proof,
            },
        }
        stored = (approach, approach_proof, manoeuvre)
        _BASE_MANOEUVRE_CACHE[cache_key] = tuple(json.loads(json.dumps(value)) for value in stored)  # type: ignore[assignment]
        return stored
    raise _NoParkingCandidate(
        f"slot {target['slot_number']}: no full-footprint, static-clear, axle-bounded parking v4 "
        f"manoeuvre among {len(candidates)} stable candidates; rejections={rejection_counts}"
    )


def _exact_waypoint(carla_map: Any, pose: dict[str, Any], *, project: bool = False) -> Any:
    """Return exact Driving/Parking membership with the release precedence."""

    import carla

    location = carla.Location(
        x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0)),
    )
    for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
        waypoint = carla_map.get_waypoint(
            location, project_to_road=project, lane_type=lane_type,
        )
        if waypoint is not None:
            return waypoint
    return None


def _lane_graph_index(context: dict[str, Any]) -> tuple[
    dict[tuple[int, int, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]
]:
    graph = json.loads(context["lane_graph_path"].read_text(encoding="utf-8"))
    by_key = {
        (int(row["road_id"]), int(row["section_id"]), int(row["lane_id"])): row
        for row in graph["lanes"]
    }
    by_id = {str(row["id"]): row for row in graph["lanes"]}
    directions = {}
    for identifier, row in by_id.items():
        samples = row.get("samples", [])
        if len(samples) < 2:
            raise RuntimeError(f"parking lane {identifier} has no direction samples")
        delta = float(samples[-1]["s"]) - float(samples[0]["s"])
        if abs(delta) <= 1e-9:
            raise RuntimeError(f"parking lane {identifier} has ambiguous s direction")
        directions[identifier] = 1 if delta > 0 else -1
    return by_key, by_id, directions


def _canonical_waypoint_transform(waypoint: Any, pose_index: int) -> dict[str, Any]:
    return {
        "pose_index": int(pose_index),
        "x": round(float(waypoint.transform.location.x), 4),
        "y": round(float(waypoint.transform.location.y), 4),
        "z": round(float(waypoint.transform.location.z), 4),
        "yaw_deg": _canonical_yaw_deg(math.radians(float(waypoint.transform.rotation.yaw))),
    }


def _construction_topology_proof(
    context: dict[str, Any], carla_map: Any, poses: list[dict[str, Any]],
    construction_rows: list[dict[str, Any]], *, first_pose_index: int = 0,
    pose_index_space: str = "reverse_manoeuvre.poses",
    require_raw_transform: bool = False,
) -> dict[str, Any]:
    """Freeze a minimal, replayable stored-XODR construction sequence."""

    if len(poses) != len(construction_rows) or not poses:
        raise RuntimeError("parking topology pose/construction inventory differs")
    by_key, by_id, directions = _lane_graph_index(context)
    samples: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    graph_ids: list[str] = []
    for offset, (pose, row) in enumerate(zip(poses, construction_rows)):
        pose_index = first_pose_index + offset
        road_id = int(row["road_id"]); section_id = int(row["section_id"])
        lane_id = int(row["lane_id"]); s_m = float(row["s_m"])
        waypoint = carla_map.get_waypoint_xodr(road_id, lane_id, s_m)
        if waypoint is None or int(waypoint.section_id) != section_id:
            raise _NoParkingCandidate(
                f"stored-XODR construction point cannot reconstruct: "
                f"{road_id}/{section_id}/{lane_id}/{s_m}"
            )
        key = (road_id, section_id, lane_id)
        graph = by_key.get(key)
        if graph is None:
            raise _NoParkingCandidate(f"stored-XODR lane is absent from frozen graph: {key}")
        dx = float(pose["x"]) - float(waypoint.transform.location.x)
        dy = float(pose["y"]) - float(waypoint.transform.location.y)
        yaw = math.radians(float(waypoint.transform.rotation.yaw))
        longitudinal = abs(dx * math.cos(yaw) + dy * math.sin(yaw))
        vertical = abs(float(pose.get("z", 0.0)) - float(waypoint.transform.location.z))
        if longitudinal > 0.1001 + 1e-9 or vertical > 0.1001 + 1e-9:
            raise _NoParkingCandidate(
                f"construction pose residual exceeds v5 bound: {longitudinal}/{vertical}"
            )
        transform = _canonical_waypoint_transform(waypoint, pose_index)
        public = {key: pose[key] for key in ("x", "y", "z", "yaw_deg")}
        canonical = {key: transform[key] for key in ("x", "y", "z", "yaw_deg")}
        if (require_raw_transform or (pose_index_space == "reverse_manoeuvre.poses" and offset == 0)) \
                and public != canonical:
            raise _NoParkingCandidate(
                f"construction pose {pose_index} is not raw canonical XODR transform"
            )
        samples.append({
            "pose_index": pose_index, "road_id": road_id,
            "section_id": section_id, "lane_id": lane_id, "s_m": s_m,
        })
        transforms.append(transform)
        graph_ids.append(str(graph["id"]))

    runs: list[dict[str, Any]] = []
    for sample, graph_id in zip(samples, graph_ids):
        if not runs or runs[-1]["lane_graph_id"] != graph_id:
            runs.append({
                "first_pose_index": sample["pose_index"],
                "last_pose_index": sample["pose_index"],
                "lane_graph_id": graph_id,
                "travel_s_direction": directions[graph_id],
            })
        else:
            previous = samples[sample["pose_index"] - first_pose_index - 1]
            if (float(sample["s_m"]) - float(previous["s_m"])) * directions[graph_id] <= 1e-8:
                raise _NoParkingCandidate(
                    f"stored-XODR construction reverses within {graph_id}"
                )
            runs[-1]["last_pose_index"] = sample["pose_index"]
    transitions = []
    for index, (first, second) in enumerate(zip(runs, runs[1:])):
        if second["lane_graph_id"] not in by_id[first["lane_graph_id"]].get("successors", []):
            raise _NoParkingCandidate(
                f"stored-XODR construction is not a direct successor: "
                f"{first['lane_graph_id']}->{second['lane_graph_id']}"
            )
        transitions.append({"from_run_index": index, "to_run_index": index + 1})
    return {
        "schema_version": "language2drive.parking_v5_stored_xodr_topology.v1",
        "method": "stored_xodr_construction_waypoint_sequence_v1",
        "pose_index_space": pose_index_space,
        "first_pose_index": first_pose_index,
        "last_pose_index": first_pose_index + len(poses) - 1,
        "samples": samples,
        "samples_sha256": stable_hash(samples),
        "canonical_waypoint_transforms_sha256": stable_hash(transforms),
        "runs": runs,
        "transitions": transitions,
    }


def _continuity_construction_rows(
    context: dict[str, Any], carla_map: Any, poses: list[dict[str, Any]],
    first_waypoint: Any,
) -> list[dict[str, Any]]:
    """Bind poses to one forward XODR construction chain.

    CARLA's inverse lookup is ambiguous where several lane surfaces overlap
    (D315 can jump from road38 to unrelated road49).  Starting from the frozen
    projected waypoint, each next sample is therefore selected only from the
    current lane's forward continuation and its direct lane-graph successors.
    Distance to the serialized actor pose is the primary geometry key; the
    raw XODR tuple/s supplies a deterministic tiebreaker and replay authority.
    Exact inverse lookup remains a separate membership check in the topology
    validator and full-footprint audit.
    """

    import carla

    if not poses:
        raise _NoParkingCandidate("empty construction pose sequence")
    by_key, by_id, directions = _lane_graph_index(context)

    def key(waypoint: Any) -> tuple[int, int, int]:
        return (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))

    def graph_id(waypoint: Any) -> str:
        row = by_key.get(key(waypoint))
        if row is None:
            raise _NoParkingCandidate(f"construction lane absent from graph: {key(waypoint)}")
        return str(row["id"])

    rows = [{
        "road_id": int(first_waypoint.road_id),
        "section_id": int(first_waypoint.section_id),
        "lane_id": int(first_waypoint.lane_id), "s_m": float(first_waypoint.s),
    }]
    previous = first_waypoint
    for pose_index, pose in enumerate(poses[1:], 1):
        previous_id = graph_id(previous)
        allowed_ids = {previous_id, *by_id[previous_id].get("successors", [])}
        location = carla.Location(
            x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0)),
        )
        candidates: dict[tuple[int, int, int, float], Any] = {}
        # ``next`` preserves CARLA topology through junction/overlap areas.
        # Multiple small lookaheads cover serialized 0.075m samples and the
        # occasional shorter rounding step without using a global inverse.
        for lookahead in (0.025, 0.05, 0.075, 0.1, 0.125, 0.15):
            for waypoint in previous.next(lookahead):
                try:
                    identifier = graph_id(waypoint)
                except _NoParkingCandidate:
                    continue
                if identifier not in allowed_ids:
                    continue
                candidate_key = (
                    int(waypoint.road_id), int(waypoint.section_id),
                    int(waypoint.lane_id), float(waypoint.s),
                )
                candidates[candidate_key] = waypoint
        ranked = []
        for candidate_key, waypoint in candidates.items():
            identifier = graph_id(waypoint)
            if identifier == previous_id:
                delta_s = float(waypoint.s) - float(previous.s)
                if delta_s * directions[identifier] <= 1e-8:
                    continue
            distance = waypoint.transform.location.distance(location)
            ranked.append((
                round(float(distance), 12),
                0 if identifier == previous_id else 1,
                candidate_key,
                waypoint,
            ))
        if not ranked:
            raise _NoParkingCandidate(
                f"construction chain has no forward continuation at pose {pose_index}"
            )
        _, _, _, chosen = min(ranked, key=lambda value: value[:3])
        previous = chosen
        rows.append({
            "road_id": int(chosen.road_id), "section_id": int(chosen.section_id),
            "lane_id": int(chosen.lane_id), "s_m": float(chosen.s),
        })
    return rows


def _centered_slot_axis_templates(
    context: dict[str, Any], carla_map: Any, target: dict[str, Any],
) -> list[tuple[float, int, float, float, int]]:
    """Enumerate only physical slot-axis stations within 0.15m of lane centre."""

    import carla

    vertices = target["polygon"][:-1] if target["polygon"][0] == target["polygon"][-1] else target["polygon"]
    center_x = sum(float(point[0]) for point in vertices) / len(vertices)
    center_y = sum(float(point[1]) for point in vertices) / len(vertices)
    heading = math.radians(float(target["heading_deg"]))
    eligible: list[float] = []
    for distance_index in range(8, 49):
        distance = distance_index * 0.25
        location = carla.Location(
            x=center_x + distance * math.cos(heading),
            y=center_y + distance * math.sin(heading), z=0.0,
        )
        waypoint = carla_map.get_waypoint(
            location, project_to_road=False, lane_type=carla.LaneType.Driving,
        )
        if waypoint is not None and location.distance(waypoint.transform.location) <= 0.15 + 1e-9:
            eligible.append(distance)
    if not eligible:
        eligible = []
    frozen_distance_hints = V5_SLOT_AXIS_DISTANCE_HINTS.get(
        str(target["slot_number"]), (),
    )
    # A frozen hint is a qualified search policy, not an extra member of the
    # shortest-first pool: letting later centre-grid stations compete can
    # silently replace the all-branch witness (D404 previously drifted from
    # 2.5m to 3.25m).  Every hinted station still traverses the full template,
    # OBB, exact-map and topology audits below.
    eligible = list(dict.fromkeys(frozen_distance_hints or tuple(eligible)))
    if not eligible:
        return []
    maximum_curvature = _curvature_at_normalized_steer(
        context["steer_samples"], CANDIDATE_SEARCH_MAX_NORMALIZED_STEER,
    )
    minimum_radius = 1.0 / maximum_curvature
    return [
        (distance, offset, minimum_radius + 0.02 + radius_index * 0.2, fraction, sign)
        for distance in eligible
        for offset in range(-30, 31, 2)
        for sign in (-1, 1)
        for radius_index in range(20)
        for fraction in (0.25, 0.35, 0.45, 0.55)
    ]


def _v5_ingress_candidate(
    project_root: Path, target: dict[str, Any], carla_map: Any,
    base: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    *, blend_m: float, blend_fraction: float, straight_m: float = 8.1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Prepend G1 alignment and a proof-bound >=8m lane-aligned settling run."""

    import carla
    import numpy as np

    context = _geometry(project_root)
    base_approach, _, base_manoeuvre = base
    old_actor = base_manoeuvre["poses"]
    old_rear = base_manoeuvre["rear_axle_poses"]
    target_yaw = math.radians(float(base_approach["yaw_deg"]))
    forward = np.array([math.cos(target_yaw), math.sin(target_yaw)])
    normal = np.array([-math.sin(target_yaw), math.cos(target_yaw)])
    membership = _exact_waypoint(carla_map, base_approach)
    if membership is None:
        raise _NoParkingCandidate("slot-axis approach has no exact lane membership")
    lane_point = np.array([
        float(membership.transform.location.x), float(membership.transform.location.y),
    ])
    approach_xy = np.array([float(base_approach["x"]), float(base_approach["y"])])
    lateral = float(np.dot(approach_xy - lane_point, normal))
    raw_start = approach_xy - (float(straight_m) + float(blend_m)) * forward - lateral * normal
    start_wp = carla_map.get_waypoint(
        carla.Location(x=float(raw_start[0]), y=float(raw_start[1]), z=0.0),
        project_to_road=True, lane_type=carla.LaneType.Driving,
    )
    if start_wp is None:
        raise _NoParkingCandidate("v5 prelead cannot project to a Driving waypoint")
    start_actor = np.array([
        float(start_wp.transform.location.x), float(start_wp.transform.location.y),
    ])
    start_yaw = math.radians(float(start_wp.transform.rotation.yaw))
    if abs((math.degrees(start_yaw - target_yaw) + 180.0) % 360.0 - 180.0) > 3.0 + 1e-9:
        raise _NoParkingCandidate("v5 prelead lane-yaw drift exceeds 3 degrees")
    aligned_actor = approach_xy - float(straight_m) * forward
    start_rear = start_actor + REAR_AXLE_X_M * np.array([
        math.cos(start_yaw), math.sin(start_yaw),
    ])
    aligned_rear = aligned_actor + REAR_AXLE_X_M * forward
    blend_points, blend_yaws, _ = _bezier(
        start_rear, aligned_rear, start_yaw, target_yaw, blend_fraction,
    )
    count = max(1, int(math.ceil(float(straight_m) / SAMPLE_STEP_M)))
    terminal_rear = np.array([float(old_rear[0]["x"]), float(old_rear[0]["y"])])
    straight_points = np.array([
        aligned_rear + (index / count) * (terminal_rear - aligned_rear)
        for index in range(1, count + 1)
    ])
    rear_prefix: list[dict[str, Any]] = []
    for point, yaw in zip(
        list(blend_points) + list(straight_points),
        list(blend_yaws) + [target_yaw] * len(straight_points),
    ):
        pose = {
            "x": round(float(point[0]), 6), "y": round(float(point[1]), 6),
            "z": 0.0, "yaw_deg": _canonical_yaw_deg(float(yaw)), "gear": "forward",
        }
        if not rear_prefix or pose != rear_prefix[-1]:
            rear_prefix.append(pose)
    rear_prefix[-1] = dict(old_rear[0])
    actor_prefix = [_actor_pose_from_rear(pose) for pose in rear_prefix]
    if actor_prefix[-1] != old_actor[0]:
        raise _NoParkingCandidate("v5 ingress does not meet the slot-axis suffix raw-exact")
    # The ordinary route hands off at the actor-centre prelead pose.  A rear-
    # axle G1 curve alone does not guarantee that the offset actor centre's
    # first chord is tangent when initial curvature is high.  Qualify this
    # target-local seam now so every possible common route can approach pose
    # zero along its frozen lane yaw without exceeding the 3-degree runtime
    # handoff contract.  This is a candidate gate, never a yaw rewrite.
    first_distinct = next((pose for pose in actor_prefix[1:] if math.dist(
        tuple(float(actor_prefix[0][key]) for key in ("x", "y", "z")),
        tuple(float(pose[key]) for key in ("x", "y", "z")),
    ) > 1e-6), None)
    if first_distinct is None:
        raise _NoParkingCandidate("v5 ingress has no nonzero actor-centre lead-in chord")
    first_chord_yaw = math.degrees(math.atan2(
        float(first_distinct["y"]) - float(actor_prefix[0]["y"]),
        float(first_distinct["x"]) - float(actor_prefix[0]["x"]),
    ))
    if abs((
        first_chord_yaw - float(actor_prefix[0]["yaw_deg"]) + 180.0
    ) % 360.0 - 180.0) > 3.0 + 1e-9:
        raise _NoParkingCandidate(
            "v5 actor-centre prelead chord exceeds 3-degree route seam"
        )
    rear = rear_prefix + [dict(pose) for pose in old_rear[1:]]
    poses = actor_prefix + [dict(pose) for pose in old_actor[1:]]
    aligned_serial = (round(float(aligned_rear[0]), 6), round(float(aligned_rear[1]), 6))
    try:
        aligned_index = next(
            index for index, pose in enumerate(rear_prefix)
            if (pose["x"], pose["y"]) == aligned_serial
        )
    except StopIteration as error:
        raise _NoParkingCandidate("v5 aligned-straight start disappeared during serialization") from error
    slot_axis_index = len(actor_prefix) - 1

    overlap_sets = _overlap_index_sets(context, poses)
    audit = _audit(context, poses, rear, target["polygon"], overlap_sets)
    if audit is None:
        # The sampled lane-surface union is only a fast conservative screen.
        exact_context = dict(context)
        exact_context["lane_surface"] = context["Polygon"]([
            (-1000.0, -1000.0), (1000.0, -1000.0),
            (1000.0, 1000.0), (-1000.0, 1000.0),
        ])
        audit = _audit(exact_context, poses, rear, target["polygon"], overlap_sets)
    if audit is None or audit[1] > CANDIDATE_SEARCH_MAX_NORMALIZED_STEER + 1e-9:
        raise _NoParkingCandidate("v5 ingress fails actor/rear/static/control qualification")
    if not _exact_map_footprint_audit(context, carla_map, poses, target["polygon"]):
        raise _NoParkingCandidate("v5 ingress full OBB leaves exact Driving/Parking/own-slot")

    construction_rows = _continuity_construction_rows(
        context, carla_map, poses[:slot_axis_index + 1], start_wp,
    )
    topology = _construction_topology_proof(
        context, carla_map, poses[:slot_axis_index + 1], construction_rows,
    )

    line = rear[aligned_index:slot_axis_index + 1]
    line_dx = float(line[-1]["x"]) - float(line[0]["x"])
    line_dy = float(line[-1]["y"]) - float(line[0]["y"])
    baseline = math.hypot(line_dx, line_dy)
    accumulated = sum(math.dist(
        tuple(float(first[key]) for key in ("x", "y", "z")),
        tuple(float(second[key]) for key in ("x", "y", "z")),
    ) for first, second in zip(line, line[1:]))
    if baseline < 8.0 - 1e-9 or accumulated < 8.0 - 1e-9:
        raise _NoParkingCandidate("v5 aligned settling segment is shorter than 8m")
    line_yaw = math.degrees(math.atan2(line_dy, line_dx))
    ux, uy = line_dx / baseline, line_dy / baseline
    previous_progress = -math.inf
    for relative, pose in enumerate(line):
        dx = float(pose["x"]) - float(line[0]["x"])
        dy = float(pose["y"]) - float(line[0]["y"])
        progress = dx * ux + dy * uy
        lateral_error = abs(dx * (-uy) + dy * ux)
        if relative and progress <= previous_progress + 1e-9:
            raise _NoParkingCandidate("v5 aligned settling segment is not strictly progressive")
        previous_progress = progress
        if lateral_error > 0.001 + 1e-9 or abs(
            (float(pose["yaw_deg"]) - line_yaw + 180.0) % 360.0 - 180.0
        ) > 1.0 + 1e-9:
            raise _NoParkingCandidate("v5 aligned settling segment is not straight/yaw aligned")
    for sample in topology["samples"][aligned_index:slot_axis_index + 1]:
        waypoint = carla_map.get_waypoint_xodr(
            int(sample["road_id"]), int(sample["lane_id"]), float(sample["s_m"]),
        )
        if waypoint is None or abs(
            (float(waypoint.transform.rotation.yaw) - line_yaw + 180.0) % 360.0 - 180.0
        ) > 1.0 + 1e-9:
            raise _NoParkingCandidate("v5 aligned settling lane yaw exceeds 1 degree")

    slot_sample = topology["samples"][slot_axis_index]
    slot_wp = carla_map.get_waypoint_xodr(
        int(slot_sample["road_id"]), int(slot_sample["lane_id"]), float(slot_sample["s_m"]),
    )
    if slot_wp is None:
        raise _NoParkingCandidate("v5 slot-axis construction waypoint cannot reconstruct")
    approach = {
        key: poses[slot_axis_index][key] for key in ("x", "y", "z", "yaw_deg")
    }
    vertices = target["polygon"][:-1] if target["polygon"][0] == target["polygon"][-1] else target["polygon"]
    center_x = sum(float(point[0]) for point in vertices) / len(vertices)
    center_y = sum(float(point[1]) for point in vertices) / len(vertices)
    heading_deg = float(target["heading_deg"])
    heading = math.radians(heading_deg)
    dx = float(approach["x"]) - center_x; dy = float(approach["y"]) - center_y
    along = dx * math.cos(heading) + dy * math.sin(heading)
    cross = dx * (-math.sin(heading)) + dy * math.cos(heading)
    if along <= 0.0 or abs(cross) > 0.001 + 1e-9:
        raise _NoParkingCandidate("v5 slot-axis approach is not on the positive target ray")
    lane_yaw = _canonical_yaw_deg(math.radians(float(slot_wp.transform.rotation.yaw)))
    approach_proof = {
        "method": "slot_axis_lane_intersection_v5",
        "distance_from_slot_center_m": round(along, 6),
        "heading_error_deg": round(abs((
            float(slot_wp.transform.rotation.yaw) - heading_deg + 180.0
        ) % 360.0 - 180.0), 3),
        "lane_id": int(slot_sample["lane_id"]), "lane_yaw_deg": lane_yaw,
        "road_id": int(slot_sample["road_id"]), "section_id": int(slot_sample["section_id"]),
        "reverse": {
            "method": "audited_slot_axis_to_reverse_manoeuvre_v5",
            "own_slot_tolerance_m": FOOTPRINT_TOLERANCE_M,
            "sample_spacing_max_m": MANOEUVRE_SPACING_M,
            "samples": len(poses) - slot_axis_index,
        },
    }
    checked_pairs = len(poses) * int(context["asset"]["bbox_count"])
    overlapping_pairs = sum(len(indices) for indices in overlap_sets)
    proof = dict(base_manoeuvre["proof"])
    proof.update({
        "method": "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v5",
        "poses_sha256": stable_hash(poses), "pose_count": len(poses),
        "rear_axle_poses_sha256": stable_hash(rear),
        "required_max_rear_axle_curvature_1pm": audit[0],
        "required_max_abs_normalized_steer": audit[1],
        "full_footprint_legal_samples": len(poses),
        "static_clearance_samples": len(poses),
        "static_clearance_min_m": (
            STATIC_CLEARANCE_M if math.isinf(audit[2])
            else math.floor(float(audit[2]) * 1_000_000.0) / 1_000_000.0
        ),
        "vertical_overlap_proof": {
            **base_manoeuvre["proof"]["vertical_overlap_proof"],
            "pose_count": len(poses),
            "static_bbox_count": int(context["asset"]["bbox_count"]),
            "checked_pair_count": checked_pairs,
            "vertically_overlapping_pair_count": overlapping_pairs,
            "vertically_disjoint_pair_count": checked_pairs - overlapping_pairs,
            "overlap_index_sets_sha256": stable_hash(overlap_sets),
        },
        "aligned_lead_in_start_index": aligned_index,
        "slot_axis_approach_index": slot_axis_index,
        "construction_topology_proof": topology,
    })
    manoeuvre = {
        "schema_version": "language2drive.parking_reverse_manoeuvre.v5",
        "poses": poses, "rear_axle_poses": rear, "proof": proof,
    }
    return approach, approach_proof, manoeuvre


def mine(
    project_root: Path,
    target: dict[str, Any],
    carla_map: Any,
    waypoint_lookup: Callable[[Any, dict[str, Any]], Any],
    *,
    _template_override: list[tuple[float, int, float, float, int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return one deterministic, full-footprint parking-v5 manoeuvre."""

    context = _geometry(project_root)
    slot_number = str(target["slot_number"])
    cache_key = (
        slot_number, stable_hash(target["polygon"]),
        context["asset"]["bboxes_sha256"],
        stable_hash(context["vehicle"]["local_bounding_box"]),
        context["dynamics"]["normalized_steer_response"]["canonical_samples_sha256"],
        "parking-v5-g1-aligned8.1-stored-xodr-actor-seam3",
    )
    if _template_override is None and cache_key in _MANOEUVRE_CACHE:
        return tuple(
            json.loads(json.dumps(value)) for value in _MANOEUVRE_CACHE[cache_key]
        )  # type: ignore[return-value]

    if _template_override is not None:
        base = _mine_slot_axis_manoeuvre_v4(
            project_root, target, carla_map, waypoint_lookup,
            _template_override=_template_override,
        )
    else:
        minimum_radius = 1.0 / _curvature_at_normalized_steer(
            context["steer_samples"], CANDIDATE_SEARCH_MAX_NORMALIZED_STEER,
        )
        frozen_base_hints: dict[
            str, list[tuple[float, int, float, float, int]]
        ] = {
            "A303": [(4.5, -30, minimum_radius + 0.02 + 16 * 0.2, 0.35, -1)],
            "A705": [(5.0, -30, 8.3, 0.35, -1)],
            "A706": [(5.0, -30, 8.4, 0.35, -1)],
            **{
                identifier: [
                    (
                        distance, offset,
                        minimum_radius + 0.02 + radius_index * 0.2,
                        fraction, sign,
                    )
                    for distance, offset, radius_index, fraction, sign in values
                ]
                for identifier, values in V5_SLOT_AXIS_BASE_HINTS.items()
            },
        }
        centered = frozen_base_hints.get(slot_number) or _centered_slot_axis_templates(
            context, carla_map, target,
        )
        base = _mine_slot_axis_manoeuvre_v4(
            project_root, target, carla_map, waypoint_lookup,
            _template_override=centered if centered else None,
        )
    blends = (
        (8.0, 10.0, 12.0, 14.0)
        if str(target["slot_number"]) in {"A705", "A706"}
        else (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
    )
    failures = []
    generic = tuple(
        (blend_m, blend_fraction)
        for blend_m in blends for blend_fraction in (0.25, 0.35, 0.45, 0.55)
    )
    ordered = tuple(V5_INGRESS_HINTS.get(str(target["slot_number"]), ())) + generic
    seen: set[tuple[float, float]] = set()
    for blend_m, blend_fraction in ordered:
        if (blend_m, blend_fraction) in seen:
            continue
        seen.add((blend_m, blend_fraction))
        try:
            result = _v5_ingress_candidate(
                project_root, target, carla_map, base,
                blend_m=blend_m, blend_fraction=blend_fraction,
            )
            if _template_override is None:
                _MANOEUVRE_CACHE[cache_key] = tuple(
                    json.loads(json.dumps(value)) for value in result
                )  # type: ignore[assignment]
            return result
        except _NoParkingCandidate as error:
            failures.append(f"b{blend_m}:f{blend_fraction}:{error}")
    raise _NoParkingCandidate(
        f"slot {target['slot_number']}: no parking-v5 ingress for frozen slot-axis suffix; "
        f"attempts={len(failures)} last={failures[-3:]}"
    )


def manoeuvre_path(target: dict[str, Any]) -> list[dict[str, Any]]:
    raw = [
        {key: pose[key] for key in ("x", "y", "z", "yaw_deg")}
        for pose in target["reverse_manoeuvre"]["poses"]
    ]
    # ``poses`` preserves the exact same-pose forward→reverse gear transition.
    # ``oracle_path`` is a geometric polyline and therefore collapses that one
    # intentional duplicate; gear remains authoritative in target.poses.
    result: list[dict[str, Any]] = []
    for pose in raw:
        if result and pose == result[-1]:
            continue
        result.append(pose)
    return result
