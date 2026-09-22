from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from team_code.l2d_oracle_agent import (
    Language2DriveOracleAgent,
    OracleProgramError,
    get_entry_point,
)
from team_code.language2drive_oracle_audit import (
    audit_private_sidecar,
    validate_sidecar_source_files,
)
from team_code.language2drive_signal_free_proof import stable_hash
from team_code.language2drive_parking_manoeuvre import (
    PARKING_BBOX_CANONICAL_SORT_KEY,
    PARKING_BBOX_EXCLUDED_SCOPE,
    PARKING_BBOX_SCOPE_STATEMENT,
    PARKING_STATIC_BBOX_CATEGORIES,
    PARKING_DYNAMICS_SOURCE_FILE_SHA256,
    PARKING_DYNAMICS_SOURCE_REF,
    PARKING_FRONT_AXLE_X_M,
    PARKING_NORMALIZED_STEER_SAMPLES_SHA256,
    PARKING_RAW_STEER_SWEEPS_SHA256,
    PARKING_REAR_AXLE_X_M,
    PARKING_WHEELBASE_M,
    PARKING_WHEEL_GEOMETRY_SHA256,
    _static_horizontal_clearance,
    _vertical_overlap_index_sets,
    normalized_steer_for_curvature,
    validate_parking_manoeuvre,
)
from team_code.language2drive_trace import (
    PrivateSidecarError,
    _validate_exact_parking_manoeuvre,
)


class _Hero:
    attributes = {"role_name": "hero"}
    speed_limit_kmh = 36.0

    @classmethod
    def get_speed_limit(cls):
        return cls.speed_limit_kmh


class _World:
    @staticmethod
    def get_actors():
        return [_Hero()]


def _contains_parking(value):
    if isinstance(value, dict):
        return value.get("type") == "parking_slot" or any(
            _contains_parking(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_parking(child) for child in value)
    return False


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parking_registry(bboxes=None):
    root = Path(
        "/root/autodl-tmp/lwq/language2drive.v1/.tmp/"
        "simlingo_parking_v4_fixture"
    )
    root.mkdir(parents=True, exist_ok=True)
    local_bbox = {
        "location": {"x": -0.0063, "y": 0.0, "z": 0.7488},
        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "extent": {"x": 2.4462, "y": 0.9184, "z": 0.7451},
    }
    bboxes = bboxes or [{
        "category": "Static",
        "center": {"x": 0.0, "y": 0.0, "z": 20.0},
        "extent": {"x": 2.0, "y": 2.0, "z": 0.5},
        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    }]
    bbox_asset = {
        "schema_version": "language2drive.carla_level_bboxes.v1",
        "town": "parkinglotfish-new3",
        "categories": PARKING_STATIC_BBOX_CATEGORIES,
        "bbox_count": len(bboxes),
        "bboxes": bboxes,
        "bboxes_sha256": stable_hash(bboxes),
        "scope_statement": PARKING_BBOX_SCOPE_STATEMENT,
        "excluded_scope": PARKING_BBOX_EXCLUDED_SCOPE,
        "quantization_decimals": 6,
        "canonical_sort_key": PARKING_BBOX_CANONICAL_SORT_KEY,
        "provenance": {
            "source_api": "carla.World.get_level_bbs(CityObjectLabel)"
        },
    }
    bbox_path = root / "bboxes.json"
    bbox_path.write_text(
        json.dumps(bbox_asset, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    xodr_path = root / "parkinglotfish-new3.xodr"
    xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    lane_path = root / "parkinglotfish-new3.lane_graph.json"
    lane_path.write_text('{"lanes":[]}', encoding="utf-8")
    trace_path = root / "trace.jsonl"
    trace_row = {
        "ego": {
            "transform": {"location": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "bounding_box": local_bbox,
        }
    }
    trace_path.write_text(
        "\n".join(json.dumps(trace_row, sort_keys=True) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "language2drive.parking_manoeuvre_registry.v4",
        "method": "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v4",
        "towns": {
            "parkinglotfish-new3": {
                "town": "parkinglotfish-new3",
                "bbox_asset": {
                    "source_ref": str(bbox_path),
                    "source_file_sha256": _sha256(bbox_path),
                    "schema_version": bbox_asset["schema_version"],
                    "bboxes_sha256": bbox_asset["bboxes_sha256"],
                    "bbox_count": len(bboxes),
                    "categories": PARKING_STATIC_BBOX_CATEGORIES,
                    "scope_statement": PARKING_BBOX_SCOPE_STATEMENT,
                    "excluded_scope": PARKING_BBOX_EXCLUDED_SCOPE,
                    "quantization_decimals": 6,
                    "canonical_sort_key": PARKING_BBOX_CANONICAL_SORT_KEY,
                    "source_api": "carla.World.get_level_bbs(CityObjectLabel)",
                },
                "xodr": {"source_ref": str(xodr_path), "sha256": _sha256(xodr_path)},
                "lane_graph": {"source_ref": str(lane_path), "sha256": _sha256(lane_path)},
                "control_model": {
                    "schema_version": "language2drive.parking_axle_control_model.v4",
                    "pose_reference": "carla_actor_transform",
                    "integration_reference": "rear_axle_center",
                    "vehicle_dynamics_source_ref": PARKING_DYNAMICS_SOURCE_REF,
                    "vehicle_dynamics_source_file_sha256": PARKING_DYNAMICS_SOURCE_FILE_SHA256,
                    "wheel_geometry_sha256": PARKING_WHEEL_GEOMETRY_SHA256,
                    "normalized_steer_samples_sha256": PARKING_NORMALIZED_STEER_SAMPLES_SHA256,
                    "front_axle_x_m": PARKING_FRONT_AXLE_X_M,
                    "rear_axle_x_m": PARKING_REAR_AXLE_X_M,
                    "wheelbase_m": PARKING_WHEELBASE_M,
                    "max_normalized_control_steer": 0.8,
                },
                "vehicle_geometry": {
                    "vehicle_blueprint": "vehicle.lincoln.mkz_2020",
                    "coordinate_frame": "carla_actor_local",
                    "local_bbox": local_bbox,
                    "local_bbox_sha256": stable_hash(local_bbox),
                    "source_method": "constant_bbox_across_authoritative_trace_v1",
                    "source_trace_ref": str(trace_path),
                    "source_trace_sha256": _sha256(trace_path),
                    "source_trace_rows": 2,
                    "matching_bbox_rows": 2,
                    "actor_reference_z_min_m": 0.0,
                    "actor_reference_z_max_m": 0.0,
                },
            }
        },
    }


def _parking_source_files(registry):
    entry = registry["towns"]["parkinglotfish-new3"]
    dynamics = json.loads(Path(PARKING_DYNAMICS_SOURCE_REF).read_text(encoding="utf-8"))
    return {
        entry["bbox_asset"]["source_ref"]: entry["bbox_asset"]["source_file_sha256"],
        entry["xodr"]["source_ref"]: entry["xodr"]["sha256"],
        entry["lane_graph"]["source_ref"]: entry["lane_graph"]["sha256"],
        entry["control_model"]["vehicle_dynamics_source_ref"]: entry["control_model"]["vehicle_dynamics_source_file_sha256"],
        entry["vehicle_geometry"]["source_trace_ref"]: entry["vehicle_geometry"]["source_trace_sha256"],
        **dynamics["source_files"],
    }


def _private(target, contract, *, actor_setup=None, path=None):
    route_path = path or [
        {"x": float(index), "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
        for index in range(61)
    ]
    town = "parkinglotfish-new3" if _contains_parking(target) else "Town03"
    route = {
        "route_entry_id": 7,
        "pcu_id": "T-001",
        "arm_id": "own",
        "paraphrase": 0,
        "split": "dev",
        "layer": "A",
        "town": town,
        "instruction": "Execute the private witness.",
        "target": target,
        "contract": contract,
        "opposite_contract": {"eventually": {"progress": {"min": 2.0}}},
        "actor_setup": actor_setup or {},
        "deadline_seconds": 40,
        "legal_speed_mps": 10.0,
        "legal_speed_proof": {
            "method": "oracle_path_exact_waypoint_min_v1",
            "samples": 101,
            "lane_types": ["Driving", "Parking"],
            "minimum_positive_mps": 10.0,
            "sample_spacing_max_m": 0.1,
            "xodr_sha256": "a" * 64,
            "town": town,
        },
        "speed_feasibility_cap_proof": {
            "method": "carla_default_limit_conservative_cap_v1",
            "cap_mps": 8.333333,
            "control_margin_mps": 0.35,
        },
        "oracle_path": route_path,
    }
    if '"speed_band"' in json.dumps(contract, sort_keys=True):
        length = sum(
            ((second["x"] - first["x"]) ** 2
             + (second["y"] - first["y"]) ** 2) ** 0.5
            for first, second in zip(route_path, route_path[1:])
        )
        route["signal_free_corridor_proof"] = {
            "method": "exact_signal_stop_free_lane_slice_v2",
            "lane_id": "Town03:7:0:-1",
            "path_start_index_inclusive": 20,
            "path_end_index_inclusive": 20 + len(route_path) - 1,
            "buffer_start_index_inclusive": 0,
            "buffer_end_index_inclusive": 40 + len(route_path) - 1,
            "road_s_min": 20.0,
            "road_s_max": 20.0 + length,
            "corridor_length_m": length,
            "before_buffer_length_m": 20.0,
            "after_buffer_length_m": 20.0,
            "sample_spacing_max_m": 0.1,
            "samples_checked": 1200,
            "junction_sample_count": 0,
            "road_signal_count": 0,
            "road_signal_reference_count": 0,
            "road_stop_control_count": 0,
            "map_road_signal_count": 0,
            "other_road_signal_clearance_m": None,
            "execution_path_sha256": stable_hash(route_path),
            "safety_buffer_path_sha256": "b" * 64,
            "lane_graph_sha256": "c" * 64,
            "xodr_sha256": "a" * 64,
            "audit_sha256": "d" * 64,
        }
    result = {
        "route": {
            **route,
        }
    }
    if _contains_parking(target):
        registry = _parking_registry()
        result["parking_manoeuvre_registry"] = registry
        result["parking_source_files"] = _parking_source_files(registry)
    return result


def _parking_v3(*, radius=5.0):
    """Return one deterministic rear-axle-closed v4 parking fixture."""

    rear_poses = []
    forward_steps = int(math.ceil(radius / 0.075))
    for index in range(forward_steps + 1):
        distance = radius * index / forward_steps
        rear_poses.append({
            "x": distance, "y": 0.0, "z": 0.0,
            "yaw_deg": 0.0, "gear": "forward",
        })
    # Exact repeated staging transform makes the gear change unambiguous.
    rear_poses.append({**rear_poses[-1], "gear": "reverse"})
    arc_length = 0.5 * math.pi * radius
    reverse_steps = int(math.ceil(arc_length / 0.075))
    for index in range(1, reverse_steps + 1):
        travelled = arc_length * index / reverse_steps
        angle = travelled / radius
        rear_poses.append({
            "x": radius - radius * math.sin(angle),
            "y": radius * (1.0 - math.cos(angle)),
            "z": 0.0,
            "yaw_deg": -math.degrees(angle),
            "gear": "reverse",
        })
    rear_poses = [{
        "x": round(float(pose["x"]), 6),
        "y": round(float(pose["y"]), 6),
        "z": 0.0,
        "yaw_deg": round(float(pose["yaw_deg"]), 3),
        "gear": pose["gear"],
    } for pose in rear_poses]
    poses = []
    for pose in rear_poses:
        yaw = math.radians(float(pose["yaw_deg"]))
        poses.append({
            "x": round(float(pose["x"]) + (-PARKING_REAR_AXLE_X_M) * math.cos(yaw), 4),
            "y": round(float(pose["y"]) + (-PARKING_REAR_AXLE_X_M) * math.sin(yaw), 4),
            "z": pose["z"],
            "yaw_deg": pose["yaw_deg"],
            "gear": pose["gear"],
        })
    approach = {key: poses[0][key] for key in ("x", "y", "z", "yaw_deg")}
    maximum_curvature = 0.0
    required_max_steer = 0.0
    samples = json.loads(Path(PARKING_DYNAMICS_SOURCE_REF).read_text(encoding="utf-8"))[
        "normalized_steer_response"
    ]["canonical_samples"]
    for first, second in zip(rear_poses, rear_poses[1:]):
        distance = math.hypot(
            second["x"] - first["x"], second["y"] - first["y"]
        )
        if distance <= 1e-9:
            continue
        signed_yaw_step = math.radians(
            (second["yaw_deg"] - first["yaw_deg"] + 180.0) % 360.0 - 180.0
        )
        signed_distance = distance * (-1.0 if first["gear"] == "reverse" else 1.0)
        curvature = signed_yaw_step / signed_distance
        maximum_curvature = max(maximum_curvature, abs(curvature))
        required_max_steer = max(
            required_max_steer,
            abs(normalized_steer_for_curvature(curvature, samples)),
        )
    terminal = poses[-1]
    polygon = [
        [terminal["x"] - 1.6, terminal["y"] - 2.7],
        [terminal["x"] + 1.6, terminal["y"] - 2.7],
        [terminal["x"] + 1.6, terminal["y"] + 2.7],
        [terminal["x"] - 1.6, terminal["y"] + 2.7],
        [terminal["x"] - 1.6, terminal["y"] - 2.7],
    ]
    manoeuvre = {
        "schema_version": "language2drive.parking_reverse_manoeuvre.v4",
        "poses": poses,
        "rear_axle_poses": rear_poses,
        "proof": {
            "method": "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v4",
            "pose_reference": "carla_actor_transform",
            "integration_reference": "rear_axle_center",
            "poses_sha256": stable_hash(poses),
            "rear_axle_poses_sha256": stable_hash(rear_poses),
            "pose_count": len(poses),
            "sample_spacing_max_m": 0.1,
            "rear_axle_sample_spacing_max_m": 0.1,
            "required_max_rear_axle_curvature_1pm": maximum_curvature,
            "required_max_abs_normalized_steer": required_max_steer,
            "max_normalized_control_steer": 0.8,
            "vehicle_blueprint": "vehicle.lincoln.mkz_2020",
            "footprint_length_m": 4.892382,
            "footprint_width_m": 1.836713,
            "footprint_tolerance_m": 0.2,
            "full_footprint_legal_samples": len(poses),
            "static_clearance_samples": len(poses),
            "static_collision_samples": 0,
            "static_clearance_min_m": 0.2,
            "static_bbox_categories": PARKING_STATIC_BBOX_CATEGORIES,
            "static_bbox_count": 1,
            "static_bboxes_sha256": _parking_registry()["towns"]["parkinglotfish-new3"]["bbox_asset"]["bboxes_sha256"],
            "town": "parkinglotfish-new3",
            "xodr_sha256": _parking_registry()["towns"]["parkinglotfish-new3"]["xodr"]["sha256"],
            "lane_graph_sha256": _parking_registry()["towns"]["parkinglotfish-new3"]["lane_graph"]["sha256"],
            "vertical_overlap_proof": {
                "method": "carla_obb_world_z_interval_overlap_v1",
                "vertical_padding_m": 0.05,
                "ego_local_bbox_sha256": _parking_registry()["towns"]["parkinglotfish-new3"]["vehicle_geometry"]["local_bbox_sha256"],
                "pose_count": len(poses),
                "static_bbox_count": 1,
                "checked_pair_count": len(poses),
                "vertically_overlapping_pair_count": 0,
                "vertically_disjoint_pair_count": len(poses),
                "overlap_index_sets_sha256": stable_hash([[] for _ in poses]),
            },
        },
    }
    target = {
        "type": "parking_slot",
        "id": "slot:A2",
        "slot_number": "A2",
        "polygon": polygon,
        "heading_deg": -90.0,
        "dwell_seconds": 3.0,
        "approach_pose": approach,
        "approach_pose_proof": {
            "method": "slot_axis_lane_intersection_v2",
            "distance_from_slot_center_m": math.hypot(
                approach["x"] - terminal["x"], approach["y"] - terminal["y"]
            ),
            "heading_error_deg": 90.0,
            "lane_id": -1,
            "lane_yaw_deg": 0.0,
            "road_id": 7,
            "section_id": 0,
            "reverse": {
                "method": "audited_reverse_manoeuvre_v4",
                "own_slot_tolerance_m": 0.2,
                "sample_spacing_max_m": 0.1,
                "samples": len(poses),
            },
        },
        "reverse_manoeuvre": manoeuvre,
    }
    # The target manoeuvre retains both gear-labelled copies of the staging
    # pose; the CARLA route suffix collapses the reverse-labelled duplicate.
    transition = next(
        index for index, (first, second) in enumerate(zip(poses, poses[1:]))
        if first["gear"] != second["gear"]
    )
    executable_poses = poses[:transition + 1] + poses[transition + 2:]
    path = [
        {"x": -1.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
        *[
            {key: pose[key] for key in ("x", "y", "z", "yaw_deg")}
            for pose in executable_poses
        ],
    ]
    return target, path


def _attach_real_v5_slot_occupancy(registry):
    """Upgrade the synthetic v4 fixture with the current L2D v5 authority."""

    path = Path(
        "/root/autodl-tmp/lwq/language2drive.v1/generated/maps/"
        "parkinglotfish-new3.slot_collision_occupancy.v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["slots"]
    clear = sorted(row["slot_number"] for row in rows if row["terminal_spawn_clear"])
    blocked = sorted(row["slot_number"] for row in rows if not row["terminal_spawn_clear"])
    registry["towns"]["parkinglotfish-new3"]["slot_collision_occupancy"] = {
        "source_ref": str(path),
        "source_file_sha256": _sha256(path),
        "schema_version": payload["schema_version"],
        "method": payload["method"],
        "town": payload["town"],
        "map_name": payload["map_name"],
        "vehicle_blueprint": payload["vehicle_blueprint"],
        "spawn_z_m": payload["spawn_z_m"],
        "repeat_count": payload["repeat_count"],
        "slots_sha256": payload["slots_sha256"],
        "slot_count": len(rows),
        "clear_count": len(clear),
        "blocked_count": len(blocked),
        "clear_slots": clear,
        "blocked_slots": blocked,
        "clear_slots_sha256": stable_hash(clear),
        "blocked_slots_sha256": stable_hash(blocked),
        "source_files": payload["source_files"],
        "scope_statement": payload["scope_statement"],
    }
    return registry


def _bind_parking_fixture_to_registry(target, registry, overlap_index_sets):
    """Update one synthetic target with exact registry-bound vertical proof."""

    entry = registry["towns"]["parkinglotfish-new3"]
    proof = target["reverse_manoeuvre"]["proof"]
    poses = target["reverse_manoeuvre"]["poses"]
    overlap_count = sum(len(indices) for indices in overlap_index_sets)
    checked = len(poses) * entry["bbox_asset"]["bbox_count"]
    proof.update({
        "static_bbox_count": entry["bbox_asset"]["bbox_count"],
        "static_bboxes_sha256": entry["bbox_asset"]["bboxes_sha256"],
        "xodr_sha256": entry["xodr"]["sha256"],
        "lane_graph_sha256": entry["lane_graph"]["sha256"],
        "vertical_overlap_proof": {
            "method": "carla_obb_world_z_interval_overlap_v1",
            "vertical_padding_m": 0.05,
            "ego_local_bbox_sha256": entry["vehicle_geometry"]["local_bbox_sha256"],
            "pose_count": len(poses),
            "static_bbox_count": entry["bbox_asset"]["bbox_count"],
            "checked_pair_count": checked,
            "vertically_overlapping_pair_count": overlap_count,
            "vertically_disjoint_pair_count": checked - overlap_count,
            "overlap_index_sets_sha256": stable_hash(overlap_index_sets),
        },
    })


class OracleAgentProgramTest(unittest.TestCase):
    def setUp(self):
        self.original_world = CarlaDataProvider._world
        CarlaDataProvider._world = _World()
        self.original_speed_limit = _Hero.speed_limit_kmh
        _Hero.speed_limit_kmh = 36.0
        self.record_dir = os.environ.pop("L2D_RECORD_DIR", None)

    def tearDown(self):
        CarlaDataProvider._world = self.original_world
        _Hero.speed_limit_kmh = self.original_speed_limit
        if self.record_dir is not None:
            os.environ["L2D_RECORD_DIR"] = self.record_dir

    def _agent(self, private):
        agent = Language2DriveOracleAgent("127.0.0.1", 2000, False)
        agent.set_language2drive_context({
            "schema_version": "language2drive.public_context.v1",
            "track": "native",
            "instruction": "Execute the private witness.",
            "instruction_version": 0,
            "route_name": "RouteScenario_7_rep0",
            "seed": 1,
        })
        agent.set_language2drive_private_context(private)
        agent.set_global_plan([], [(object(), object())])
        agent.setup("ignored.json+RouteScenario_7")
        return agent

    def test_entrypoint_and_sensor_track_are_bench2drive_compatible(self):
        agent = self._agent(
            _private(
                {"type": "gate", "id": "g"},
                {"eventually": {"enter_gate": {"id": "g"}}},
            )
        )
        self.assertEqual("Language2DriveOracleAgent", get_entry_point())
        self.assertEqual("SENSORS", agent.track.value)
        self.assertEqual([{
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed",
        }], agent.sensors())
        self.assertEqual(1, agent._native_plan_length)
        agent.destroy()
        agent.destroy()

    def test_speed_wait_and_conditional_parking_compile_from_private_program(self):
        speed = self._agent(
            _private(
                {"type": "delayed_progress", "min": 0.15, "wait_seconds": 10},
                {"sequence": [
                    {"dwell": {"condition": {"speed_band": {
                        "min_mps": 0.0, "max_mps": 0.2,
                    }}, "seconds": 10}},
                    {"progress": {"min": 0.15}},
                ]},
            )
        )
        self.assertEqual(10.0, speed._wait_seconds)
        self.assertIsNone(speed._speed_band)
        speed.destroy()

        missing_proof = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        missing_proof["route"].pop("legal_speed_mps")
        with self.assertRaisesRegex(OracleProgramError, "legal_speed_mps proof"):
            self._agent(missing_proof)

        insufficient_margin = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        insufficient_margin["route"]["legal_speed_mps"] = 5.2
        insufficient_margin["route"]["legal_speed_proof"][
            "minimum_positive_mps"
        ] = 5.2
        with self.assertRaisesRegex(OracleProgramError, "lacks 0.35m/s margin"):
            self._agent(insufficient_margin)

        missing_cap_proof = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        missing_cap_proof["route"].pop("speed_feasibility_cap_proof")
        with self.assertRaisesRegex(
            OracleProgramError, "requires speed_feasibility_cap_proof"
        ):
            self._agent(missing_cap_proof)

        insufficient_cap = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        insufficient_cap["route"]["speed_feasibility_cap_proof"][
            "cap_mps"
        ] = 5.2
        with self.assertRaisesRegex(
            OracleProgramError, "requires cap_mps=8.333333"
        ):
            self._agent(insufficient_cap)

        overstated_cap = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        overstated_cap["route"]["speed_feasibility_cap_proof"][
            "cap_mps"
        ] = 100.0
        with self.assertRaisesRegex(
            OracleProgramError, "requires cap_mps=8.333333"
        ):
            self._agent(overstated_cap)

        malformed_cap = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        malformed_cap["route"]["speed_feasibility_cap_proof"][
            "unversioned"
        ] = True
        with self.assertRaisesRegex(OracleProgramError, "keys must be exactly"):
            self._agent(malformed_cap)

        safety_cap_exception = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        safety_cap_exception["route"]["layer"] = "Safety"
        safety_cap_exception["route"].pop("speed_feasibility_cap_proof")
        safety = self._agent(safety_cap_exception)
        self.assertIsNone(safety._speed_feasibility_cap_mps)
        safety.destroy()

        xodr_proof_above_live_limit = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        xodr_proof_above_live_limit["route"]["legal_speed_mps"] = 9.0
        xodr_proof_above_live_limit["route"]["legal_speed_proof"][
            "minimum_positive_mps"
        ] = 9.0
        _Hero.speed_limit_kmh = 28.8  # 8m/s at the actual CARLA start pose.
        conservative_live = self._agent(xodr_proof_above_live_limit)
        self.assertEqual(8.0, conservative_live._setup_speed_limit_mps)
        conservative_live.destroy()

        _Hero.speed_limit_kmh = 18.72  # 5.2m/s cannot retain the 0.35 margin.
        with self.assertRaisesRegex(
            OracleProgramError,
            r"speed band max 5\.000m/s lacks 0\.35m/s margin below live hero",
        ):
            self._agent(xodr_proof_above_live_limit)
        _Hero.speed_limit_kmh = 36.0

        live = self._agent(
            _private(
                {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
                {"eventually": {"speed_band": {
                    "min_mps": 3.0, "max_mps": 5.0,
                }}},
            )
        )
        runtime_hero = SimpleNamespace(get_speed_limit=lambda: 28.8)
        self.assertEqual(4.0, live._map_legal_speed(runtime_hero, 4.0))
        runtime_hero.get_speed_limit = lambda: 18.72
        with self.assertRaisesRegex(
            OracleProgramError,
            r"speed band max 5\.000m/s lacks 0\.35m/s margin below live map",
        ):
            live._map_legal_speed(runtime_hero, 4.0)
        runtime_hero.get_speed_limit = lambda: float("nan")
        with self.assertRaisesRegex(OracleProgramError, "non-finite"):
            live._map_legal_speed(runtime_hero, 4.0)
        live.destroy()

    def test_signal_free_corridor_v2_binds_exact_private_execution_path(self):
        base = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        compiled = self._agent(copy.deepcopy(base))
        self.assertEqual(
            "exact_signal_stop_free_lane_slice_v2",
            compiled._signal_free_corridor_proof["method"],
        )
        compiled.destroy()

        mutations = (
            (
                lambda route: route.pop("signal_free_corridor_proof"),
                "requires signal_free_corridor_proof",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"unversioned": True}
                ),
                "keys must be exactly",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"execution_path_sha256": "e" * 64}
                ),
                "does not bind serialized oracle_path",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"corridor_length_m": 59.0}
                ),
                "length does not match serialized oracle_path",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"path_end_index_inclusive": 81}
                ),
                "indices do not close oracle_path point count",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"map_road_signal_count": 1}
                ),
                "null clearance requires zero map signals",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update({
                    "map_road_signal_count": 1,
                    "other_road_signal_clearance_m": 39.9,
                }),
                "clearance must be >=40m",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"sample_spacing_max_m": 0.2}
                ),
                "must equal 0.1",
            ),
            (
                lambda route: route["signal_free_corridor_proof"].update(
                    {"lane_id": "Town02:7:0:-1"}
                ),
                "lane_id town mismatch",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                broken = copy.deepcopy(base)
                mutation(broken["route"])
                with self.assertRaisesRegex(OracleProgramError, message):
                    self._agent(broken)

        safety = copy.deepcopy(base)
        safety["route"]["layer"] = "Safety"
        safety["route"].pop("signal_free_corridor_proof")
        safety_agent = self._agent(safety)
        self.assertIsNone(safety_agent._signal_free_corridor_proof)
        safety_agent.destroy()

        malformed_provenance = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        malformed_provenance["route"]["legal_speed_proof"][
            "lane_types"
        ] = ["Parking", "Driving"]
        with self.assertRaisesRegex(OracleProgramError, "Driving->Parking"):
            self._agent(malformed_provenance)

        fractional_samples = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        fractional_samples["route"]["legal_speed_proof"]["samples"] = 1.5
        with self.assertRaisesRegex(OracleProgramError, "numeric fields"):
            self._agent(fractional_samples)

        unknown_proof_field = _private(
            {"type": "speed_band", "min_mps": 3.0, "max_mps": 5.0},
            {"eventually": {"speed_band": {
                "min_mps": 3.0, "max_mps": 5.0,
            }}},
        )
        unknown_proof_field["route"]["legal_speed_proof"]["unversioned"] = True
        with self.assertRaisesRegex(OracleProgramError, "keys must be exactly"):
            self._agent(unknown_proof_field)

        selected_parking, selected_path = _parking_v3()
        selected_polygon = selected_parking["polygon"]
        parking_target = {
            "type": "if_else",
            "condition": "occupied:A1",
            "then": selected_parking,
            "else": {
                "type": "parking_slot", "id": "slot:A1",
                "slot_number": "A1",
                "polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]],
                "heading_deg": 0.0, "dwell_seconds": 3.0,
            },
        }
        parking = self._agent(
            _private(
                parking_target,
                {"if_else": {
                    "condition": {"actor_order": {"id": "occupied:A1"}},
                    "then": {"vehicle_footprint_inside_slot": {
                        "polygon": selected_polygon, "heading_deg": 0.0,
                    }},
                    "else": {"progress": {"min": 2.0}},
                }},
                actor_setup={
                    "actors": [{"id": "occupant:A1", "slot": "A1"}],
                    "slot_occupancy": {"A1": True},
                },
                path=selected_path,
            )
        )
        self.assertEqual("then", parking._conditional_branch)
        self.assertEqual("parking_slot", parking._target["type"])
        self.assertEqual(
            tuple(
                selected_parking["reverse_manoeuvre"]["poses"][-1][axis]
                for axis in ("x", "y")
            ),
            parking._parking["center"],
        )
        self.assertEqual("approach", parking._parking_state)
        self.assertEqual(
            "target.approach_pose:v4_physical_lane",
            parking._parking["approach_pose_source"],
        )
        self.assertGreater(len(parking._parking["reverse_poses"]), 2)
        parking.destroy()

    def test_parking_rejects_unaudited_or_unbound_approach_geometry(self):
        target, path = _parking_v3()
        polygon = target["polygon"]
        contract = {"eventually": {"vehicle_footprint_inside_slot": {
            "polygon": polygon, "heading_deg": -90.0,
        }}}

        without_approach = dict(target)
        without_approach.pop("approach_pose")
        with self.assertRaisesRegex(OracleProgramError, "approach_pose keys"):
            self._agent(_private(without_approach, contract, path=path))

        incomplete_approach = copy.deepcopy(target)
        incomplete_approach["approach_pose"].pop("z")
        with self.assertRaisesRegex(OracleProgramError, "keys must be exactly"):
            self._agent(_private(incomplete_approach, contract, path=path))

        sub_tolerance_approach = copy.deepcopy(target)
        sub_tolerance_approach["approach_pose"]["x"] += 5e-7
        with self.assertRaisesRegex(
            OracleProgramError, "first pose disagrees exactly"
        ):
            self._agent(
                _private(sub_tolerance_approach, contract, path=path)
            )

        sub_tolerance_transition = copy.deepcopy(target)
        transition_poses = sub_tolerance_transition[
            "reverse_manoeuvre"
        ]["poses"]
        transition = next(
            index
            for index, (first, second) in enumerate(
                zip(transition_poses, transition_poses[1:])
            )
            if first["gear"] != second["gear"]
        )
        transition_poses[transition + 1]["x"] += 5e-7
        sub_tolerance_transition["reverse_manoeuvre"]["proof"][
            "poses_sha256"
        ] = stable_hash(transition_poses)
        with self.assertRaisesRegex(
            OracleProgramError, "raw-exact|rear-to-actor"
        ):
            self._agent(_private(sub_tolerance_transition, contract, path=path))

        displaced_path = [dict(point) for point in path]
        displaced_path[-2]["x"] += 0.02
        with self.assertRaisesRegex(OracleProgramError, "suffix disagrees"):
            self._agent(_private(target, contract, path=displaced_path))

        sub_tolerance_path = [dict(point) for point in path]
        sub_tolerance_path[-2]["x"] += 5e-7
        with self.assertRaisesRegex(
            OracleProgramError, "suffix disagrees exactly"
        ):
            self._agent(_private(target, contract, path=sub_tolerance_path))

        missing_yaw_path = [dict(point) for point in path]
        missing_yaw_path[-2].pop("yaw_deg")
        with self.assertRaisesRegex(OracleProgramError, "x/y/z/yaw_deg"):
            self._agent(_private(target, contract, path=missing_yaw_path))

        wrong_terminal_yaw = [dict(point) for point in path]
        wrong_terminal_yaw[-1]["yaw_deg"] = 90.0
        with self.assertRaisesRegex(OracleProgramError, "suffix disagrees"):
            self._agent(_private(target, contract, path=wrong_terminal_yaw))

        wrong_approach_yaw = copy.deepcopy(target)
        wrong_approach_yaw["approach_pose"]["yaw_deg"] = -90.0
        with self.assertRaisesRegex(OracleProgramError, "physical lane_yaw"):
            self._agent(_private(wrong_approach_yaw, contract, path=path))

        tampered_hash = copy.deepcopy(target)
        tampered_hash["reverse_manoeuvre"]["proof"]["poses_sha256"] = "0" * 64
        with self.assertRaisesRegex(OracleProgramError, "poses_sha256 mismatch"):
            self._agent(_private(tampered_hash, contract, path=path))

    def test_parking_v3_vertical_overlap_and_source_hashes_fail_closed(self):
        target, path = _parking_v3()
        registry = _parking_registry()
        entry = registry["towns"]["parkinglotfish-new3"]
        poses = target["reverse_manoeuvre"]["poses"]
        local_bbox = entry["vehicle_geometry"]["local_bbox"]

        # A high decorative bbox is conservatively present in the frozen
        # asset but is not a 2D collision candidate for the Lincoln.
        overhead = json.loads(Path(
            entry["bbox_asset"]["source_ref"]
        ).read_text(encoding="utf-8"))["bboxes"]
        overlap_sets = _vertical_overlap_index_sets(
            poses,
            overhead,
            local_bbox,
            actor_reference_z_min_m=0.0,
            actor_reference_z_max_m=0.0,
        )
        self.assertEqual([[] for _ in poses], overlap_sets)
        validate_parking_manoeuvre(
            {"layer": "A", "town": "parkinglotfish-new3", "target": target,
             "oracle_path": path},
            registry=registry,
        )

        tampered = copy.deepcopy(target)
        tampered["reverse_manoeuvre"]["proof"][
            "vertical_overlap_proof"
        ]["overlap_index_sets_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            Exception, "index hash/counts disagree"
        ):
            validate_parking_manoeuvre(
                {"layer": "A", "town": "parkinglotfish-new3", "target": tampered,
                 "oracle_path": path},
                registry=registry,
            )

        # Same-height geometry cannot be discarded by the vertical filter;
        # an overlapping XY OBB has zero clearance and the complete proof is
        # rejected without weakening the 0.2m safety threshold.
        collision_bbox = [{
            "category": "Static",
            "center": {"x": 0.0, "y": 0.0, "z": 0.75},
            "extent": {"x": 2.0, "y": 2.0, "z": 0.5},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        }]
        collision_registry = _parking_registry(collision_bbox)
        collision_entry = collision_registry["towns"]["parkinglotfish-new3"]
        collision_sets = _vertical_overlap_index_sets(
            poses,
            collision_bbox,
            collision_entry["vehicle_geometry"]["local_bbox"],
            actor_reference_z_min_m=0.0,
            actor_reference_z_max_m=0.0,
        )
        self.assertTrue(all(indices == [0] for indices in collision_sets))
        self.assertEqual(
            0.0,
            _static_horizontal_clearance(
                poses,
                collision_bbox,
                collision_sets,
                local_bbox=collision_entry["vehicle_geometry"]["local_bbox"],
            ),
        )
        collision_target = copy.deepcopy(target)
        _bind_parking_fixture_to_registry(
            collision_target, collision_registry, collision_sets
        )
        with self.assertRaisesRegex(
            Exception, "horizontal clearance disagrees"
        ):
            validate_parking_manoeuvre(
                {"layer": "A", "town": "parkinglotfish-new3",
                 "target": collision_target, "oracle_path": path},
                registry=collision_registry,
            )

    def test_parking_v3_vertical_padding_boundary_is_overlap(self):
        target, _ = _parking_v3()
        poses = target["reverse_manoeuvre"]["poses"][:1]
        registry = _parking_registry()
        local_bbox = registry["towns"]["parkinglotfish-new3"][
            "vehicle_geometry"
        ]["local_bbox"]
        ego_max = 0.7488 + 0.7451
        # Each interval is expanded by 0.05m: an original 0.10m gap is still
        # overlapping, while a gap larger by 2e-8 is disjoint after epsilon.
        def indices(gap):
            bbox = [{
                "category": "Static",
                "center": {"x": 100.0, "y": 100.0,
                           "z": ego_max + gap + 0.5},
                "extent": {"x": 1.0, "y": 1.0, "z": 0.5},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            }]
            return _vertical_overlap_index_sets(
                poses, bbox, local_bbox,
                actor_reference_z_min_m=0.0,
                actor_reference_z_max_m=0.0,
            )[0]
        self.assertEqual([0], indices(0.1))
        self.assertEqual([], indices(0.10000002))

    def test_parking_v3_inactive_conditional_branch_is_validated(self):
        selected, path = _parking_v3()
        inactive = copy.deepcopy(selected)
        inactive["id"] = "slot:inactive"
        inactive.pop("reverse_manoeuvre")
        route = {
            "layer": "A",
            "town": "parkinglotfish-new3",
            "target": {
                "type": "if_else",
                "condition": "occupied:A2",
                "then": selected,
                "else": inactive,
            },
            "actor_setup": {"slot_occupancy": {"A2": True}},
            "oracle_path": path,
        }
        with self.assertRaisesRegex(
            PrivateSidecarError, "inactive|requires reverse_manoeuvre v4"
        ):
            registry = _parking_registry()
            _validate_exact_parking_manoeuvre(
                route, registry, _parking_source_files(registry)
            )

    def test_parking_v3_stops_then_tracks_only_frozen_reverse_curve(self):
        target, path = _parking_v3()
        self.assertTrue(all(
            math.hypot(second["x"] - first["x"], second["y"] - first["y"])
            > 1e-6
            for first, second in zip(path, path[1:])
        ))
        poses = target["reverse_manoeuvre"]["poses"]
        transitions = [
            index for index, (first, second) in enumerate(zip(poses, poses[1:]))
            if first["gear"] != second["gear"]
        ]
        self.assertEqual(1, len(transitions))
        transition = transitions[0]
        self.assertEqual(
            {
                key: poses[transition][key]
                for key in ("x", "y", "z", "yaw_deg")
            },
            {
                key: poses[transition + 1][key]
                for key in ("x", "y", "z", "yaw_deg")
            },
        )
        agent = self._agent(_private(
            target,
            {"sequence": [
                {"vehicle_footprint_inside_slot": {
                    "polygon": target["polygon"],
                    "heading_deg": target["heading_deg"],
                    "heading_tolerance_deg": 15.0,
                    "tolerance_m": 0.2,
                }},
                {"dwell": {
                    "key": target["id"], "seconds": 3.0,
                    "max_speed_mps": 0.2,
                }},
            ]},
            path=path,
        ))

        class Ego:
            def __init__(self, x, y, yaw, speed=0.0):
                self.set(x, y, yaw, speed)

            def set(self, x, y, yaw, speed=0.0):
                self.transform = SimpleNamespace(
                    location=SimpleNamespace(x=x, y=y, z=0.0),
                    rotation=SimpleNamespace(yaw=yaw),
                )
                self.velocity = SimpleNamespace(x=speed, y=0.0, z=0.0)

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return self.velocity

        stage = agent._parking["forward_poses"][-1]
        ego = Ego(stage["x"], stage["y"], stage["yaw_deg"], 0.0)

        # A parking staging curve is shorter than the generic follower's
        # minimum lookahead.  Its dedicated controller must therefore use the
        # serialized pose yaw instead of cutting across the endpoint chord.
        original_forward_rear_poses = agent._parking["forward_rear_poses"]
        curved_forward_rear_poses = [
            {"x": 0.0, "y": 0.0, "z": 0.0,
             "yaw_deg": 0.0, "gear": "forward"},
            {"x": 0.7, "y": -0.05, "z": 0.0,
             "yaw_deg": -8.0, "gear": "forward"},
            {"x": 1.3, "y": -0.2, "z": 0.0,
             "yaw_deg": -18.0, "gear": "forward"},
            {"x": 1.8, "y": -0.5, "z": 0.0,
             "yaw_deg": -28.0, "gear": "forward"},
        ]
        agent._parking["forward_rear_poses"] = curved_forward_rear_poses
        agent._parking["proof"]["required_max_abs_normalized_steer"] = 0.8
        agent._set_path([
            (pose["x"], pose["y"], pose["z"])
            for pose in curved_forward_rear_poses
        ])
        curve_ego = Ego(-PARKING_REAR_AXLE_X_M, 0.0, 0.0, 0.5)
        curve_control, _, _ = agent._forward_parking_path_control(curve_ego)
        self.assertLess(float(curve_control.steer), -0.5)
        self.assertEqual(0.5, agent._effective_lookahead_m)
        self.assertFalse(curve_control.reverse)

        # Near the endpoint, lateral pure-pursuit error must not reverse the
        # frozen curve's steering sign.  The live route-528 failure reached
        # this pose with roughly -17deg yaw and used to steer back to +0.15.
        endpoint_yaw = -17.0
        endpoint_yaw_rad = math.radians(endpoint_yaw)
        curve_ego.set(
            1.72 - PARKING_REAR_AXLE_X_M * math.cos(endpoint_yaw_rad),
            -0.47 - PARKING_REAR_AXLE_X_M * math.sin(endpoint_yaw_rad),
            endpoint_yaw,
            0.2,
        )
        endpoint_control, endpoint_remaining, _ = (
            agent._forward_parking_path_control(curve_ego)
        )
        self.assertLess(endpoint_remaining, 0.2)
        self.assertLess(float(endpoint_control.steer), -0.3)
        agent._parking["forward_rear_poses"] = original_forward_rear_poses
        agent._set_path([
            (pose["x"], pose["y"], pose["z"])
            for pose in original_forward_rear_poses
        ])

        # The dedicated staging controller and its gear transition must share
        # the same 0.10m boundary.  This both prevents the original threshold
        # deadlock and keeps the forward-to-reverse handoff on the raw-exact
        # audited staging transform instead of 0.35m before it.
        original_forward_control = agent._forward_parking_path_control
        agent._parking_state = "staging_forward"
        agent._forward_parking_path_control = lambda _ego: (
            SimpleNamespace(), 0.1001,
            (stage["x"], stage["y"], stage["z"]),
        )
        agent._parking_control(ego)
        self.assertEqual("staging_forward", agent._parking_state)

        # Along-path remaining is not enough to authorize a gear change.  A
        # stopped vehicle laterally displaced from the explicit rear-axle
        # staging pose must fail immediately instead of waiting for deadline.
        ego.set(stage["x"], stage["y"] + 0.1001, stage["yaw_deg"], 0.0)
        agent._forward_parking_path_control = lambda _ego: (
            SimpleNamespace(), 0.0999,
            (stage["x"], stage["y"], stage["z"]),
        )
        with self.assertRaisesRegex(
            OracleProgramError, "rear axle misses frozen handoff"
        ):
            agent._parking_control(ego)
        self.assertEqual("staging_forward", agent._parking_state)

        ego.set(stage["x"], stage["y"] + 0.0999, stage["yaw_deg"], 0.0)
        agent._forward_parking_path_control = lambda _ego: (
            SimpleNamespace(), 0.0999,
            (stage["x"], stage["y"], stage["z"]),
        )
        _, simulated_remaining, _ = agent._parking_control(ego)
        self.assertGreaterEqual(simulated_remaining, 0.0)
        self.assertLess(
            simulated_remaining, agent.PARKING_STAGING_ENDPOINT_STOP_M
        )
        self.assertEqual("staging_stop", agent._parking_state)
        self.assertEqual(
            "privileged_oracle:parking_staging_forward",
            agent.controller_source,
        )

        # The handoff yaw is independently fail-closed at 3 degrees.  This
        # capture gate is deliberately stricter than the 5-degree adjacent
        # proof discretization; it is not a swept-footprint clearance claim.
        agent._parking_state = "staging_forward"
        ego.set(
            stage["x"], stage["y"], stage["yaw_deg"] + 3.0001, 0.0
        )
        with self.assertRaisesRegex(
            OracleProgramError, "parking staging heading error"
        ):
            agent._parking_control(ego)
        self.assertEqual("staging_forward", agent._parking_state)

        ego.set(
            stage["x"], stage["y"], stage["yaw_deg"] + 2.9999, 0.0
        )
        _, yaw_boundary_remaining, _ = agent._parking_control(ego)
        self.assertLess(
            yaw_boundary_remaining, agent.PARKING_STAGING_ENDPOINT_STOP_M
        )
        self.assertEqual("staging_stop", agent._parking_state)
        agent._forward_parking_path_control = original_forward_control

        stop_control, _, _ = agent._parking_control(ego)
        self.assertFalse(stop_control.reverse)
        self.assertEqual("reverse", agent._parking_state)

        observed_steer = []
        reverse_poses = agent._parking["reverse_poses"]
        for index, pose in enumerate(reverse_poses):
            ego.set(
                pose["x"], pose["y"], pose["yaw_deg"],
                0.0 if index == len(reverse_poses) - 1 else 0.5,
            )
            control, _, _ = agent._parking_control(ego)
            observed_steer.append(abs(float(control.steer)))
            if index < len(reverse_poses) - 1:
                self.assertTrue(control.reverse)
                self.assertEqual(-1, control.gear)
            if index == 0:
                self.assertGreater(float(control.steer), 0.45)
        self.assertLessEqual(max(observed_steer), 0.8 + 1e-6)
        self.assertEqual("dwell", agent._parking_state)
        dwell_control, _, _ = agent._parking_control(ego)
        self.assertFalse(dwell_control.reverse)
        self.assertEqual("privileged_oracle:parking_dwell", agent.controller_source)
        agent.destroy()

    def test_parking_v4_rejects_actor_origin_only_and_axle_hash_substitution(self):
        target, path = _parking_v3()
        registry = _parking_registry()
        route = {
            "layer": "A", "town": "parkinglotfish-new3", "target": target,
            "oracle_path": path,
        }
        validate_parking_manoeuvre(route, registry=registry)

        old_schema = copy.deepcopy(target)
        old_schema["reverse_manoeuvre"].pop("rear_axle_poses")
        old_schema["reverse_manoeuvre"]["schema_version"] = (
            "language2drive.parking_reverse_manoeuvre.v3"
        )
        with self.assertRaisesRegex(Exception, "keys must be exactly"):
            validate_parking_manoeuvre(
                {**route, "target": old_schema}, registry=registry
            )

        for sequence_name in ("poses", "rear_axle_poses"):
            with self.subTest(noncanonical_yaw_sequence=sequence_name):
                noncanonical_yaw = copy.deepcopy(target)
                noncanonical_yaw["reverse_manoeuvre"][sequence_name][7][
                    "yaw_deg"
                ] = -363.0
                with self.assertRaisesRegex(
                    Exception, "yaw_deg is not canonical"
                ):
                    validate_parking_manoeuvre(
                        {**route, "target": noncanonical_yaw},
                        registry=registry,
                    )

        off_center = copy.deepcopy(target)
        for point in off_center["polygon"]:
            point[0] += 0.0001
        with self.assertRaisesRegex(Exception, "exact slot pose"):
            validate_parking_manoeuvre(
                {**route, "target": off_center}, registry=registry
            )

        wrong_sign = copy.deepcopy(target)
        rear = wrong_sign["reverse_manoeuvre"]["rear_axle_poses"]
        rear[10]["x"] = round(rear[10]["x"] + 0.01, 6)
        wrong_sign["reverse_manoeuvre"]["proof"][
            "rear_axle_poses_sha256"
        ] = stable_hash(rear)
        with self.assertRaisesRegex(Exception, "rear-to-actor"):
            validate_parking_manoeuvre(
                {**route, "target": wrong_sign}, registry=registry
            )

        substituted = copy.deepcopy(target)
        substituted["reverse_manoeuvre"]["proof"][
            "rear_axle_poses_sha256"
        ] = substituted["reverse_manoeuvre"]["proof"]["poses_sha256"]
        with self.assertRaisesRegex(Exception, "rear_axle_poses_sha256"):
            validate_parking_manoeuvre(
                {**route, "target": substituted}, registry=registry
            )

    def test_parking_v4_piecewise_inverse_and_proof_boundaries_fail_closed(self):
        dynamics = json.loads(
            Path(PARKING_DYNAMICS_SOURCE_REF).read_text(encoding="utf-8")
        )
        samples = dynamics["normalized_steer_response"]["canonical_samples"]
        for sample in samples:
            self.assertAlmostEqual(
                float(sample["normalized_steer"]),
                normalized_steer_for_curvature(
                    float(sample["ackermann_center_curvature_1pm"]), samples
                ),
                places=12,
            )
        midpoint_curvature = 0.5 * (
            float(samples[20]["ackermann_center_curvature_1pm"])
            + float(samples[21]["ackermann_center_curvature_1pm"])
        )
        self.assertAlmostEqual(
            0.225,
            normalized_steer_for_curvature(midpoint_curvature, samples),
            places=12,
        )
        with self.assertRaisesRegex(Exception, "qualification envelope"):
            normalized_steer_for_curvature(
                float(samples[-1]["ackermann_center_curvature_1pm"]) + 1e-6,
                samples,
            )

        target, path = _parking_v3()
        registry = _parking_registry()
        route = {
            "layer": "A", "town": "parkinglotfish-new3", "target": target,
            "oracle_path": path,
        }
        tampered = copy.deepcopy(target)
        tampered["reverse_manoeuvre"]["proof"][
            "required_max_abs_normalized_steer"
        ] += 1e-8
        with self.assertRaisesRegex(Exception, "piecewise map inverse"):
            validate_parking_manoeuvre(
                {**route, "target": tampered}, registry=registry
            )

        tampered = copy.deepcopy(target)
        tampered["reverse_manoeuvre"]["proof"][
            "required_max_rear_axle_curvature_1pm"
        ] += 2e-9
        with self.assertRaisesRegex(Exception, "disagrees with rear poses"):
            validate_parking_manoeuvre(
                {**route, "target": tampered}, registry=registry
            )

    def test_parking_v4_registry_asset_and_axle_closure_fail_closed(self):
        target, path = _parking_v3()
        route = {
            "layer": "A", "town": "parkinglotfish-new3", "target": target,
            "oracle_path": path,
        }
        wrong_axle = _parking_registry()
        wrong_axle["towns"]["parkinglotfish-new3"]["control_model"][
            "rear_axle_x_m"
        ] = abs(PARKING_REAR_AXLE_X_M)
        with self.assertRaisesRegex(Exception, "axle/map contract"):
            validate_parking_manoeuvre(route, registry=wrong_axle)

        wrong_hash = _parking_registry()
        wrong_hash["towns"]["parkinglotfish-new3"]["control_model"][
            "normalized_steer_samples_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(Exception, "axle/map contract"):
            validate_parking_manoeuvre(route, registry=wrong_hash)

    def test_parking_v4_runtime_projects_live_rear_axle_not_actor_origin(self):
        target, path = _parking_v3()
        agent = self._agent(_private(
            target,
            {"eventually": {"vehicle_footprint_inside_slot": {
                "polygon": target["polygon"],
                "heading_deg": target["heading_deg"],
            }}},
            path=path,
        ))

        class Ego:
            def __init__(self, actor_pose):
                self.transform = SimpleNamespace(
                    location=SimpleNamespace(
                        x=actor_pose["x"], y=actor_pose["y"], z=actor_pose["z"]
                    ),
                    rotation=SimpleNamespace(yaw=actor_pose["yaw_deg"]),
                )
                self.velocity = SimpleNamespace(x=0.2, y=0.0, z=0.0)

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return self.velocity

        actor_pose = agent._parking["forward_poses"][10]
        rear_pose = agent._parking["forward_rear_poses"][10]
        required = agent._parking["proof"][
            "required_max_abs_normalized_steer"
        ]
        for reverse, poses in (
            (False, agent._parking["forward_rear_poses"]),
            (True, agent._parking["reverse_rear_poses"]),
        ):
            for index in range(len(poses)):
                curvature = agent._parking_path_curvature(
                    poses, index, reverse
                )
                self.assertLessEqual(
                    abs(agent._parking_steer_for_curvature(curvature)),
                    required + 1e-9,
                )
        agent._set_path([
            (pose["x"], pose["y"], pose["z"])
            for pose in agent._parking["forward_rear_poses"]
        ], minimum_spacing_m=1e-6)
        control, _, _ = agent._forward_parking_path_control(Ego(actor_pose))
        self.assertAlmostEqual(float(rear_pose["y"]), 0.0, places=9)
        self.assertLess(abs(float(control.steer)), 0.05)
        agent.destroy()

    def test_parking_v5_captures_stored_prelead_before_explicit_path(self):
        """The v5 handoff is poses[0]/rear[0], never slot-axis metadata."""

        target, path = _parking_v3()
        agent = self._agent(_private(
            target,
            {"eventually": {"vehicle_footprint_inside_slot": {
                "polygon": target["polygon"],
                "heading_deg": target["heading_deg"],
            }}},
            path=path,
        ))
        manoeuvre = copy.deepcopy(agent._parking_manoeuvre)
        poses = manoeuvre["poses"]
        rear_poses = manoeuvre["rear_axle_poses"]
        # A compiler-level v5 stub is sufficient here: the shared validator's
        # stored-XODR/root29/route9 replay has separate qualification tests.
        # Make the slot-axis metadata observably different from the prelead.
        slot_axis_index = min(10, int(manoeuvre["transition_index"]) - 1)
        manoeuvre["schema_version"] = (
            "language2drive.parking_reverse_manoeuvre.v5"
        )
        manoeuvre["is_v5"] = True
        manoeuvre["approach_pose"] = {
            key: poses[slot_axis_index][key]
            for key in ("x", "y", "z", "yaw_deg")
        }
        manoeuvre["v5_ingress"] = {
            "prelead_pose": {
                key: poses[0][key] for key in ("x", "y", "z", "yaw_deg")
            },
            "prelead_rear_pose": dict(rear_poses[0]),
        }
        points = [
            (point["x"], point["y"], point["z"])
            for point in path
        ]
        agent._compile_parking(points, target, path, manoeuvre)
        self.assertEqual(
            tuple(poses[0][axis] for axis in ("x", "y", "z")),
            agent._points[-1],
        )
        self.assertEqual(
            tuple(poses[0][axis] for axis in ("x", "y", "z")),
            agent._parking["approach"],
        )
        self.assertNotEqual(
            tuple(manoeuvre["approach_pose"][axis] for axis in ("x", "y", "z")),
            agent._points[-1],
        )
        self.assertEqual(
            rear_poses[0], agent._parking["forward_rear_poses"][0]
        )
        self.assertEqual(
            "reverse_manoeuvre.poses[0]:v5_stored_xodr_prelead",
            agent._parking["approach_pose_source"],
        )

        observed_endpoint = []
        original_endpoint_control = agent._path_control_to_endpoint
        agent._path_control_to_endpoint = lambda _ego, speed, endpoint: (
            observed_endpoint.append((speed, endpoint))
            or (SimpleNamespace(throttle=0.2, brake=0.0), 0.1001, (0, 0, 0))
        )
        agent._parking_prelead_path_control(None)
        self.assertEqual([(2.0, 0.10)], observed_endpoint)
        agent._path_control_to_endpoint = original_endpoint_control

        class Ego:
            def __init__(self, rear_offset, yaw_error, speed):
                rear = agent._parking["prelead_rear"]
                yaw = agent._parking["approach_yaw_deg"] + yaw_error
                yaw_rad = math.radians(yaw)
                rear_x = rear[0] - math.sin(yaw_rad) * rear_offset
                rear_y = rear[1] + math.cos(yaw_rad) * rear_offset
                axle = agent._parking["control_model"]["rear_axle_x_m"]
                self.transform = SimpleNamespace(
                    location=SimpleNamespace(
                        x=rear_x - axle * math.cos(yaw_rad),
                        y=rear_y - axle * math.sin(yaw_rad),
                        z=rear[2],
                    ),
                    rotation=SimpleNamespace(yaw=yaw),
                )
                self.velocity = SimpleNamespace(x=speed, y=0.0, z=0.0)

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return self.velocity

        def capture_control(remaining):
            return lambda _ego: (
                SimpleNamespace(throttle=0.2, brake=0.0),
                remaining,
                agent._parking["approach"],
            )

        # 10.01cm remains live forward capture; generic 35cm stopping is not
        # allowed to strand the rear axle outside the 10cm proof envelope.
        agent._parking_prelead_path_control = capture_control(0.1001)
        control, _, _ = agent._parking_control(Ego(0.0, 0.0, 0.0))
        self.assertGreater(control.throttle, 0.0)
        self.assertEqual("approach", agent._parking_state)

        agent._parking_prelead_path_control = capture_control(0.0999)
        with self.assertRaisesRegex(
            OracleProgramError, "prelead rear axle misses frozen handoff"
        ):
            agent._parking_control(Ego(0.1001, 0.0, 0.0))
        self.assertEqual("approach", agent._parking_state)

        with self.assertRaisesRegex(
            OracleProgramError, "prelead heading error"
        ):
            agent._parking_control(Ego(0.0999, 3.0001, 0.0))
        self.assertEqual("approach", agent._parking_state)

        agent._parking_control(Ego(0.0, 0.0, 0.15))
        self.assertEqual("approach", agent._parking_state)
        self.assertFalse(agent._parking_prelead_gate_passed)

        agent._parking_control(Ego(0.0999, 2.9999, 0.0))
        self.assertEqual("staging_forward", agent._parking_state)
        self.assertTrue(agent._parking_prelead_gate_passed)
        self.assertEqual(agent._tick, agent._parking_prelead_handoff_tick)
        self.assertEqual(
            "privileged_oracle:parking_v5_prelead_capture",
            agent.controller_source,
        )
        agent._plans = io.StringIO()
        agent._tick = 20
        agent._write_plan(1.0, 0.0, 0.0, agent._parking["approach"])
        trace = json.loads(agent._plans.getvalue())
        self.assertTrue(trace["parking_prelead_gate_passed"])
        self.assertEqual(
            "language2drive.parking_reverse_manoeuvre.v5",
            trace["parking_schema_version"],
        )
        self.assertEqual(
            "reverse_manoeuvre.poses[0]:v5_stored_xodr_prelead",
            trace["parking_approach_pose_source"],
        )
        self.assertAlmostEqual(0.0999, trace["parking_prelead_rear_distance_m"], places=4)
        self.assertAlmostEqual(2.9999, trace["parking_prelead_yaw_error_deg"], places=4)
        agent.destroy()

    def test_parking_v5_rejects_numeric_strings_before_map_replay(self):
        target, path = _parking_v3()
        target = copy.deepcopy(target)
        registry = _parking_registry()
        registry = _attach_real_v5_slot_occupancy(registry)
        registry["schema_version"] = (
            "language2drive.parking_manoeuvre_registry.v5"
        )
        registry["method"] = (
            "carla_level_bbox_vertical_overlap_rear_axle_manoeuvre_v5"
        )
        target["reverse_manoeuvre"]["schema_version"] = (
            "language2drive.parking_reverse_manoeuvre.v5"
        )
        target["slot_number"] = "D402"
        target["approach_pose"]["x"] = str(target["approach_pose"]["x"])
        route = {
            "layer": "A", "town": "parkinglotfish-new3",
            "target": target, "actor_setup": {"slot_occupancy": {}},
            "oracle_path": path,
        }
        with self.assertRaisesRegex(
            ValueError, "finite JSON number"
        ):
            validate_parking_manoeuvre(route, registry=registry)

    def test_old_slot_heading_approach_is_fail_closed_not_forward_aligned(self):
        target, path = _parking_v3()
        target.pop("reverse_manoeuvre")
        target["approach_pose_proof"]["method"] = (
            "slot_heading_ray_exact_lane_v1"
        )
        target["approach_pose"]["yaw_deg"] = target["heading_deg"]
        with self.assertRaisesRegex(
            OracleProgramError, "requires reverse_manoeuvre v4"
        ):
            self._agent(_private(
                target,
                {"eventually": {"vehicle_footprint_inside_slot": {
                    "polygon": target["polygon"],
                    "heading_deg": target["heading_deg"],
                }}},
                path=path,
            ))

    def test_region_requires_exact_path_bound_terminal_pose(self):
        polygon = [[8, -2], [12, -2], [12, 2], [8, 2], [8, -2]]
        region = self._agent(
            _private(
                {
                    "type": "region",
                    "id": "A",
                    "polygon": polygon,
                    "terminal_pose": {
                        "x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0,
                    },
                },
                {"eventually": {"dwell": {
                    "condition": {"inside_region": {"polygon": polygon}},
                    "seconds": 5.0,
                }}},
                path=[
                    {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                    {"x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                ],
            )
        )
        self.assertEqual("approach", region._terminal_state)
        self.assertEqual(10.0, region._terminal_pose["x"])
        region.destroy()

        with self.assertRaisesRegex(OracleProgramError, "terminal_pose"):
            self._agent(
                _private(
                    {"type": "region", "id": "A", "polygon": polygon},
                    {"eventually": {"inside_region": {"polygon": polygon}}},
                    path=[
                        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                        {"x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                    ],
                )
            )

        incomplete_pose = {
            "type": "region",
            "id": "A",
            "polygon": polygon,
            "terminal_pose": {
                "x": 10.0, "y": 0.0, "yaw_deg": 0.0,
            },
        }
        with self.assertRaisesRegex(OracleProgramError, "exactly x/y/z/yaw_deg"):
            self._agent(_private(
                incomplete_pose,
                {"eventually": {"inside_region": {"polygon": polygon}}},
                path=[
                    {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                    {"x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                ],
            ))

        wrong_yaw_target = {
            "type": "region",
            "id": "A",
            "polygon": polygon,
            "terminal_pose": {
                "x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0,
            },
        }
        with self.assertRaisesRegex(OracleProgramError, "terminal yaw"):
            self._agent(_private(
                wrong_yaw_target,
                {"eventually": {"inside_region": {"polygon": polygon}}},
                path=[
                    {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                    {"x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 45.0},
                ],
            ))

    def test_standardized_and_unknown_program_fail_closed(self):
        agent = Language2DriveOracleAgent("127.0.0.1", 2000, False)
        with self.assertRaisesRegex(OracleProgramError, "Native-System only"):
            agent.set_language2drive_context({
                "schema_version": "language2drive.public_context.v1",
                "track": "standardized",
                "instruction": "Do not expose private state.",
                "instruction_version": 0,
            })
        bad = self._agent
        with self.assertRaisesRegex(OracleProgramError, "unsupported oracle target"):
            bad(_private({"type": "teleport"}, {"progress": {"min": 0.1}}))

        malformed_condition = _private(
            {
                "type": "if_else",
                "condition": "occupied:A1",
                "then": {"type": "not_supported"},
                "else": {"type": "gate", "id": "g"},
            },
            {"eventually": {"progress": {"min": 0.1}}},
            actor_setup={
                "actors": [{"id": "occupant:A1", "slot": "A1"}],
                "slot_occupancy": {"A1": True},
            },
        )
        with self.assertRaisesRegex(OracleProgramError, "resolved oracle"):
            bad(malformed_condition)

        incomplete_path = _private(
            {"type": "gate", "id": "g"},
            {"eventually": {"enter_gate": {"id": "g"}}},
        )
        incomplete_path["route"]["oracle_path"][0].pop("z")
        with self.assertRaisesRegex(OracleProgramError, "x/y/z/yaw_deg"):
            bad(incomplete_path)

        reverse_spawn = _private(
            {"type": "gate", "id": "g"},
            {"eventually": {"enter_gate": {"id": "g"}}},
        )
        reverse_spawn["route"]["oracle_path"][0]["yaw_deg"] = 180.0
        with self.assertRaisesRegex(OracleProgramError, "spawn yaw opposes"):
            bad(reverse_spawn)

        reverse_middle = _private(
            {"type": "gate", "id": "g"},
            {"eventually": {"enter_gate": {"id": "g"}}},
            path=[
                {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                {"x": 5.0, "y": 0.0, "z": 0.0, "yaw_deg": 180.0},
                {"x": 10.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            ],
        )
        with self.assertRaisesRegex(
            OracleProgramError, "yaw opposes directed segment 1"
        ):
            bad(reverse_middle)

        malformed_speed_dsl = _private(
            {"type": "progress", "min": 0.1},
            {"eventually": {"speed_band": []}},
        )
        with self.assertRaisesRegex(
            OracleProgramError, "speed_band must be an object"
        ):
            bad(malformed_speed_dsl)

    def test_curvature_preview_slows_sharp_turns_but_not_straights(self):
        agent = Language2DriveOracleAgent.__new__(Language2DriveOracleAgent)
        agent._speed_band = None
        agent._set_path([
            (0.0, 0.0, 0.0),
            (50.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
        ])
        self.assertEqual([], agent._curve_vertices)
        self.assertEqual(
            (7.0, 6.0), agent._curve_control_limits(20.0, 7.0, 6.0)
        )

        agent._set_path([
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
        ])
        approach_speed, approach_lookahead = agent._curve_control_limits(
            0.0, 7.0, 6.0
        )
        vertex_speed, vertex_lookahead = agent._curve_control_limits(
            9.0, 7.0, 6.0
        )
        self.assertLess(approach_speed, 6.3)
        self.assertLessEqual(approach_lookahead, 3.0)
        self.assertLess(vertex_speed, 3.6)
        self.assertLessEqual(vertex_lookahead, 2.1)

        agent._speed_band = (5.9, 7.9)
        with self.assertRaisesRegex(
            OracleProgramError, "conflicts with private-path curvature cap"
        ):
            agent._set_path([
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (10.0, 10.0, 0.0),
            ])

    def test_until_speed_band_does_not_brake_before_gate_entry(self):
        agent = Language2DriveOracleAgent.__new__(Language2DriveOracleAgent)
        agent._speed_band = (5.9, 7.9)
        agent._curve_vertices = []
        agent._target = {
            "type": "until",
            "end": {
                "type": "gate", "id": "g",
                "center": {"x": 20.0, "y": 0.0}, "radius_m": 4.0,
            },
        }
        agent._until_end = copy.deepcopy(agent._target["end"])
        agent._until_end_entered = False
        agent._until_progress_min = None
        agent._until_progress_entered = False
        agent._points = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        agent._cumulative = [0.0, 20.0]
        agent._segment = 0
        agent._along = 0.0
        agent._tick = 1
        agent._project = lambda location: (float(location.x), 0.0)
        agent._adjacent_pass_target = lambda ego, target: target
        commanded = []
        agent._longitudinal_control = lambda speed, target: (
            commanded.append(target) or SimpleNamespace(steer=0.0)
        )

        class Ego:
            def __init__(self, x):
                self.x = x

            def get_transform(self):
                return SimpleNamespace(
                    location=SimpleNamespace(x=self.x, y=0.0, z=0.0),
                    rotation=SimpleNamespace(yaw=0.0),
                )

            @staticmethod
            def get_velocity():
                return SimpleNamespace(x=7.0, y=0.0, z=0.0)

        agent._path_control(Ego(15.0), 7.6)
        self.assertAlmostEqual(7.6, commanded[-1])
        self.assertFalse(agent._until_end_entered)
        agent._path_control(Ego(16.1), 7.6)
        self.assertTrue(agent._until_end_entered)
        self.assertAlmostEqual(6.1, commanded[-1])

    def test_speed_band_progress_until_does_not_brake_before_progress_edge(self):
        contract = {"sequence": [
            {"dwell": {"condition": {"speed_band": {
                "min_mps": 5.9, "max_mps": 7.9,
            }}, "seconds": 2.0}},
            {"until": {
                "invariant": {"speed_band": {
                    "min_mps": 5.9, "max_mps": 7.9,
                }},
                "condition": {"progress": {"min": 0.9}},
            }},
        ]}
        agent = self._agent(_private(
            {"type": "speed_band", "min_mps": 5.9, "max_mps": 7.9},
            contract,
        ))
        self.assertEqual(0.9, agent._until_progress_min)
        self.assertFalse(agent._until_progress_entered)
        agent._project = lambda location: (float(location.x), 0.0)
        agent._adjacent_pass_target = lambda ego, target: target
        commanded = []
        agent._longitudinal_control = lambda speed, target: (
            commanded.append(target) or SimpleNamespace(steer=0.0)
        )

        class Ego:
            def __init__(self, x):
                self.x = x

            def get_transform(self):
                return SimpleNamespace(
                    location=SimpleNamespace(x=self.x, y=0.0, z=0.0),
                    rotation=SimpleNamespace(yaw=0.0),
                )

            @staticmethod
            def get_velocity():
                return SimpleNamespace(x=7.0, y=0.0, z=0.0)

        # The exact frozen test path is 60m.  At 49m terminal braking would
        # normally be active (<12m remaining), but the 0.9 invariant edge is
        # still pending and the controller must stay inside the high band.
        agent._path_control(Ego(49.0), 7.6)
        self.assertAlmostEqual(7.6, commanded[-1])
        self.assertFalse(agent._until_progress_entered)
        # At 54.1/60 progress exceeds 0.9, permanently latching the condition;
        # controlled braking may then start without invalidating the suffix.
        agent._path_control(Ego(54.1), 7.6)
        self.assertTrue(agent._until_progress_entered)
        self.assertLess(commanded[-1], 7.6)
        self.assertGreaterEqual(commanded[-1], 6.1)
        agent.destroy()

    def test_static_sidecar_audit_compiles_every_expression_and_arm(self):
        private = _private(
            {"type": "gate", "id": "g"},
            {"eventually": {"enter_gate": {"id": "g"}}},
        )["route"]
        routes = []
        for paraphrase, instruction in enumerate((
            "Take the first gate.",
            "Drive through gate one.",
            "Use the initial marked exit.",
        )):
            route = copy.deepcopy(private)
            route.update({
                "route_entry_id": paraphrase,
                "pcu_id": "A-001",
                "arm_id": "left",
                "paraphrase": paraphrase,
                "instruction": instruction,
            })
            routes.append(route)
        tmp_root = os.environ.get("TMPDIR")
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            frozen_source = Path(directory) / "gate-registry.json"
            frozen_source.write_text('{"gate":"g"}\n', encoding="utf-8")
            sidecar = Path(directory) / "sidecar.private.json"
            sidecar.write_text(json.dumps({
                "schema_version": "language2drive.private_sidecar.v1",
                "source_files": {
                    str(frozen_source.resolve()): _sha256(frozen_source),
                },
                "routes": routes,
                "pcus": [{
                    "id": "A-001", "town": "Town03", "split": "dev",
                    "layer": "A", "source_refs": ["test://gate-registry"],
                }],
                "geometry_registry": {
                    "g": {
                        "type": "gate", "id": "g",
                        "polygon": [[4, -1], [6, -1], [6, 1], [4, 1], [4, -1]],
                        "geometry_version": "test-v1",
                        "source_ref": "test://gate-registry",
                        "source_sha256": "a" * 64,
                    },
                },
            }), encoding="utf-8")
            xml_routes = ["<routes>"]
            for route in routes:
                xml_routes.append(
                    '<route id="{route_entry_id}" town="Town03">'
                    '<simlingo_metadata benchmark_type="language2drive_v1" '
                    'track="standardized" instruction="{instruction}" />'
                    '<instruction>{instruction}</instruction>'
                    '<waypoints><position x="0" y="0" z="0" />'
                    '<position x="1" y="0" z="0" /></waypoints>'
                    '</route>'.format(**route)
                )
            xml_routes.append("</routes>")
            (Path(directory) / "routes_standardized.xml").write_text(
                "".join(xml_routes), encoding="utf-8"
            )
            report = audit_private_sidecar(
                sidecar,
                expected_routes=3,
                expected_pcus=1,
                expected_arms=1,
                expected_expressions_per_arm=3,
                required_target_types={"gate"},
            )
            frozen_source.write_text('{"gate":"tampered"}\n', encoding="utf-8")
            tampered_report = audit_private_sidecar(
                sidecar,
                expected_routes=3,
                expected_pcus=1,
                expected_arms=1,
                expected_expressions_per_arm=3,
                required_target_types={"gate"},
            )
        self.assertTrue(report["success"], report)
        self.assertEqual(3, report["compiled_routes"])
        self.assertEqual(1, report["verified_source_file_count"])
        self.assertEqual([], report["source_file_errors"])
        self.assertEqual({"gate": 1}, report["declared_target_type_arm_counts"])
        self.assertTrue(report["public_standardized_xml_checked"])
        self.assertEqual([], report["public_standardized_xml_errors"])
        self.assertFalse(tampered_report["success"])
        self.assertEqual(1, len(tampered_report["source_file_errors"]))

    def test_release_source_file_map_requires_canonical_absolute_sha_entries(self):
        tmp_root = os.environ.get("TMPDIR")
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            backing = Path(directory) / "backing.json"
            backing.write_text("{}\n", encoding="utf-8")
            source = Path(directory) / "source.json"
            source.symlink_to(backing)
            source.write_text("{}\n", encoding="utf-8")
            canonical = os.path.abspath(str(source))
            self.assertNotEqual(canonical, str(source.resolve()))
            result = validate_sidecar_source_files(
                {canonical: _sha256(source)}, required=True
            )
            self.assertEqual(1, result["verified"])
            self.assertEqual(
                0,
                validate_sidecar_source_files({}, required=False)["verified"],
            )
            with self.assertRaisesRegex(ValueError, "non-empty source_files"):
                validate_sidecar_source_files({}, required=True)
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                validate_sidecar_source_files(
                    {"source.json": _sha256(source)}, required=True
                )
            parent_spelling = os.path.join(
                directory, "unused", "..", source.name
            )
            with self.assertRaisesRegex(ValueError, "must be canonical"):
                validate_sidecar_source_files(
                    {parent_spelling: _sha256(source)}, required=True
                )
            with self.assertRaisesRegex(
                ValueError, "normalizes duplicate path"
            ):
                validate_sidecar_source_files(
                    {
                        canonical: _sha256(source),
                        parent_spelling: _sha256(source),
                    },
                    required=True,
                )
            with self.assertRaisesRegex(ValueError, "lowercase SHA256"):
                validate_sidecar_source_files(
                    {canonical: _sha256(source).upper()}, required=True
                )

    def test_static_audit_rejects_counterfactual_scene_side_channels(self):
        base = _private(
            {"type": "gate", "id": "g"},
            {"eventually": {"enter_gate": {"id": "g"}}},
            actor_setup={"actors": [{
                "id": "lead", "blueprint": "vehicle.test",
                "program": "constant_speed", "relative_spawn_m": 5.0,
            }]},
        )["route"]
        routes = []
        for arm_index, arm_id in enumerate(("left", "right")):
            for paraphrase in range(3):
                route = copy.deepcopy(base)
                route.update({
                    "route_entry_id": len(routes),
                    "pcu_id": "A-001",
                    "arm_id": arm_id,
                    "paraphrase": paraphrase,
                    "instruction": "{} expression {}".format(
                        arm_id, paraphrase
                    ),
                })
                if arm_index:
                    route["actor_setup"] = {
                        "actors": [
                            *copy.deepcopy(base["actor_setup"]["actors"]),
                            {"id": "visible-arm-marker"},
                        ]
                    }
                    route["oracle_path"][0]["x"] += 1.0
                routes.append(route)
        tmp_root = os.environ.get("TMPDIR")
        with tempfile.TemporaryDirectory(dir=tmp_root) as directory:
            sidecar = Path(directory) / "sidecar.private.json"
            sidecar.write_text(json.dumps({
                "schema_version": "language2drive.private_sidecar.v1",
                "routes": routes,
                "pcus": [{
                    "id": "A-001", "town": "Town03", "split": "dev",
                    "layer": "A", "source_refs": ["test://gate-registry"],
                }],
                "geometry_registry": {
                    "g": {
                        "type": "gate", "id": "g",
                        "polygon": [[4, -1], [6, -1], [6, 1], [4, 1], [4, -1]],
                        "geometry_version": "test-v1",
                        "source_ref": "test://gate-registry",
                        "source_sha256": "a" * 64,
                    },
                },
            }), encoding="utf-8")
            report = audit_private_sidecar(
                sidecar,
                expected_routes=6,
                expected_pcus=1,
                expected_arms=2,
                expected_expressions_per_arm=3,
                required_target_types={"gate"},
            )
        self.assertFalse(report["success"])
        self.assertIn(
            "pcu_A-001:actor_setup_differs_across_arms", report["errors"]
        )
        self.assertIn(
            "pcu_A-001:counterfactual_spawn_differs", report["errors"]
        )
        self.assertIn(
            "pcu_A-001:actor_lead_spawn_differs_across_arms",
            report["errors"],
        )

    def test_pass_lane_is_permission_checked_and_locked_after_first_choice(self):
        class Waypoint:
            def __init__(self, lane_id, x, *, lane_change="None"):
                self.road_id = 7
                self.section_id = 0
                self.lane_id = lane_id
                self.lane_type = "Driving"
                self.lane_change = lane_change
                self.transform = SimpleNamespace(
                    location=SimpleNamespace(x=x, y=float(lane_id), z=0.0),
                    rotation=SimpleNamespace(yaw=0.0),
                )
                self.left = None
                self.right = None
                self.forward = None

            def get_left_lane(self):
                return self.left

            def get_right_lane(self):
                return self.right

            def next(self, distance):
                self.asserted_distance = distance
                return [] if self.forward is None else [self.forward]

        source = Waypoint(-1, 0.0, lane_change="Left")
        selected = Waypoint(-2, 0.0, lane_change="Left")
        selected_forward = Waypoint(-2, 5.0, lane_change="Left")
        drifting_lane = Waypoint(-3, 0.0, lane_change="Left")
        drifting_forward = Waypoint(-3, 5.0, lane_change="Left")
        source.left = selected
        selected.left = drifting_lane
        selected.forward = selected_forward
        selected_forward.forward = selected_forward
        drifting_lane.forward = drifting_forward

        world_map = SimpleNamespace(current=source)
        world_map.get_waypoint = lambda location: world_map.current
        ego = SimpleNamespace(get_location=lambda: object())
        agent = Language2DriveOracleAgent.__new__(Language2DriveOracleAgent)
        agent._target = {
            "type": "actor_order",
            "order": "ego_ahead",
            "pass_lane_id": -2,
        }
        agent._passing_complete = False
        agent._passing_lane_key = None
        agent._world_map = world_map
        fallback = (99.0, 99.0, 0.0)

        self.assertEqual(
            (5.0, -2.0, 0.0), agent._adjacent_pass_target(ego, fallback)
        )
        self.assertEqual((7, 0, -2), agent._passing_lane_key)

        # Once the ego has entered lane -2, its legal left neighbor is lane -3.
        # The locked controller must advance on -2 rather than selecting -3.
        world_map.current = selected
        self.assertEqual(
            (5.0, -2.0, 0.0), agent._adjacent_pass_target(ego, fallback)
        )
        self.assertEqual((7, 0, -2), agent._passing_lane_key)

        agent._passing_complete = True
        self.assertEqual(fallback, agent._adjacent_pass_target(ego, fallback))

        disallowed = Waypoint(-1, 0.0, lane_change="None")
        disallowed.left = selected
        world_map.current = disallowed
        agent._passing_complete = False
        agent._passing_lane_key = None
        agent._target.pop("pass_lane_id")
        with self.assertRaisesRegex(OracleProgramError, "lane-change-permitted"):
            agent._adjacent_pass_target(ego, fallback)

    def test_pass_target_does_not_deadlock_generic_obstacle_layer(self):
        location = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        forward = SimpleNamespace(x=1.0, y=0.0, z=0.0)
        transform = SimpleNamespace(
            location=location,
            get_forward_vector=lambda: forward,
        )
        zero_velocity = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        ego = SimpleNamespace(
            id=1,
            get_transform=lambda: transform,
            get_velocity=lambda: zero_velocity,
        )

        def actor(identifier, role_name):
            return SimpleNamespace(
                id=identifier,
                is_alive=True,
                type_id="vehicle.test",
                attributes={"role_name": role_name},
                get_location=lambda: SimpleNamespace(x=2.5, y=0.0, z=0.0),
            )

        lead = actor(2, "l2d:lead")
        unrelated = actor(3, "background")
        agent = Language2DriveOracleAgent.__new__(Language2DriveOracleAgent)
        agent._world = SimpleNamespace(get_actors=lambda: [ego, lead])
        agent._target = {
            "type": "actor_order",
            "order": "ego_ahead",
            "actor_id": "lead",
        }
        agent._passing_complete = False
        agent._passing_lane_key = None
        self.assertEqual("vehicle", agent._obstacle_hazard(ego))

        # The exact actor is exempt only after a permission-checked adjacent
        # pass lane has been frozen by the steering controller.
        agent._passing_lane_key = (7, 0, -2)
        self.assertIsNone(agent._obstacle_hazard(ego))

        agent._world = SimpleNamespace(
            get_actors=lambda: [ego, lead, unrelated]
        )
        self.assertEqual("vehicle", agent._obstacle_hazard(ego))

        agent._world = SimpleNamespace(get_actors=lambda: [ego, lead])
        agent._target["order"] = "ego_behind"
        self.assertEqual("vehicle", agent._obstacle_hazard(ego))

        agent._target["order"] = "ego_ahead"
        agent._passing_complete = True
        self.assertEqual("vehicle", agent._obstacle_hazard(ego))

    def test_hold_program_is_compiled_as_stationary_contract(self):
        hold = self._agent(
            _private(
                {"type": "hold", "seconds": 10.0},
                {"dwell": {"condition": {"speed_band": {
                    "min_mps": 0.0, "max_mps": 0.2,
                }}, "seconds": 10.0}},
            )
        )
        self.assertEqual("hold", hold._target["type"])
        hold.destroy()

        invalid_hold = _private(
            {"type": "hold", "seconds": 0.0},
            {"always": {"speed_band": {
                "min_mps": 0.0, "max_mps": 0.2,
            }}},
        )
        with self.assertRaisesRegex(OracleProgramError, "positive seconds"):
            self._agent(invalid_hold)

        invalid_deadline = _private(
            {"type": "gate", "id": "g"},
            {"eventually": {"enter_gate": {"id": "g"}}},
        )
        invalid_deadline["route"]["deadline_seconds"] = True
        with self.assertRaisesRegex(OracleProgramError, "deadline_seconds"):
            self._agent(invalid_deadline)


if __name__ == "__main__":
    unittest.main()
