from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import leaderboard.utils.closed_loop_logger as logger_module
from leaderboard.utils.closed_loop_logger import ClosedLoopLogger
from team_code.language2drive_trace import (
    Language2DriveTracePostprocessor,
    PrivateSidecarError,
    _validate_exact_parking_manoeuvre,
    contract_capture_requirements,
    expected_run_id,
    load_private_route_context,
    private_public_start_consistency_errors,
    public_private_consistency_errors,
)
from team_code.language2drive_runtime import Language2DriveEvaluatorRuntime
from team_code.language2drive_signal_free_proof import stable_hash


L2D_V1_ROOT = Path(__file__).resolve().parents[1]


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Rotation:
    def __init__(self, yaw=0.0):
        self.pitch, self.yaw, self.roll = 0.0, yaw, 0.0


class Transform:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.location = Vector(x, y, 0.0)
        self.rotation = Rotation(yaw)

    def get_forward_vector(self):
        return Vector(1.0, 0.0, 0.0)

    def get_right_vector(self):
        return Vector(0.0, 1.0, 0.0)


class BoundingBox:
    def __init__(self):
        self.location = Vector()
        self.rotation = Rotation()
        self.extent = Vector(2.0, 1.0, 0.8)


class Actor:
    def __init__(self, instance_id, role_name, x, *, type_id="vehicle.test", color="1,2,3"):
        self.id = instance_id
        self.type_id = type_id
        self.attributes = {"role_name": role_name, "color": color}
        self._transform = Transform(x, 0.0)
        self.bounding_box = BoundingBox()

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return Vector(2.0, 0.0, 0.0)

    def get_acceleration(self):
        return Vector()

    def get_angular_velocity(self):
        return Vector()

    def get_light_state(self):
        return "NONE"

    def is_at_traffic_light(self):
        return False

    def get_traffic_light(self):
        return None


class Waypoint:
    road_id = 7
    section_id = 0
    lane_id = -1
    s = 12.5
    is_junction = False


class World:
    def __init__(self, actors):
        self._actors = actors

    def get_actors(self):
        return self._actors


class Map:
    @staticmethod
    def get_waypoint(_location):
        return Waypoint()


class Provider:
    world = None

    @classmethod
    def get_world(cls):
        return cls.world

    @staticmethod
    def get_map():
        return Map()


class Timestamp:
    def __init__(self, elapsed_seconds):
        self.elapsed_seconds = elapsed_seconds


class Action:
    throttle = 0.2
    steer = 0.0
    brake = 0.0
    reverse = False
    hand_brake = False
    gear = 1


class EventType:
    def __init__(self, name):
        self.name = name


class TrafficEvent:
    def __init__(self, name, frame=0):
        self._type = EventType(name)
        self._frame = frame

    def get_type(self):
        return self._type

    def get_frame(self):
        return self._frame

    def get_message(self):
        return "observed"

    def get_dict(self):
        return {"authoritative": True}


class CollisionTest:
    def __init__(self, events=None):
        self.events = list(events or [])


class OutsideRouteLanesTest:
    events = []


class RunningRedLightTest:
    events = []


class Scenario:
    def __init__(self, collision_events=None):
        self.criteria = [
            CollisionTest(collision_events),
            OutsideRouteLanesTest(),
            RunningRedLightTest(),
        ]

    def get_criteria(self):
        return self.criteria


def private_context():
    route = {
        "route_entry_id": 0,
        "pcu_id": "A-001",
        "arm_id": "left",
        "contract": {"eventually": {"enter_gate": {"id": "gate:left"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "gate:right"}}},
        "deadline_seconds": 10.0,
        "oracle_path": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
        "actor_setup": {
            "actors": [
                {"id": "lead", "blueprint": "vehicle.test", "color": "1,2,3"}
            ]
        },
    }
    context = {
        "route": route,
        "pcu": {"id": "A-001", "layer": "A", "town": "Town01"},
        "capture_targets": [
            {
                "type": "gate",
                "id": "gate:left",
                "polygon": [[4, -2], [8, -2], [8, 2], [4, 2], [4, -2]],
            },
            {
                "type": "gate",
                "id": "gate:right",
                "polygon": [[20, -2], [24, -2], [24, 2], [20, 2], [20, -2]],
            },
        ],
        "sidecar_sha256": "abc",
    }
    context["requirements"] = contract_capture_requirements(context)
    # Exercise actor filtering even though the simple gate contract has no actor predicate.
    context["requirements"]["actor_ids"] = ["lead"]
    return context


class PrivateSidecarTest(unittest.TestCase):
    def test_v5_inactive_conditional_target_is_always_validated(self):
        def parking(slot):
            return {
                "type": "parking_slot", "id": f"slot:{slot}",
                "slot_number": slot,
                "reverse_manoeuvre": {
                    "schema_version": (
                        "language2drive.parking_reverse_manoeuvre.v5"
                    )
                },
            }

        inactive = parking("D404")
        selected = parking("D403")
        route = {
            "layer": "D", "town": "TestParking",
            "target": {
                "type": "if_else", "condition": "occupied:D402",
                "then": inactive, "else": selected,
            },
            "actor_setup": {"slot_occupancy": {"D402": False}},
        }
        with patch(
            "team_code.language2drive_trace."
            "validate_parking_registry_source_files"
        ) as registry_check, patch(
            "team_code.language2drive_trace.validate_parking_manoeuvre"
        ) as validator:
            _validate_exact_parking_manoeuvre(route, {"v5": True}, {})
        registry_check.assert_called_once()
        self.assertEqual(2, validator.call_count)
        by_slot = {
            call.args[1]["slot_number"]: call.kwargs["require_oracle_suffix"]
            for call in validator.call_args_list
        }
        self.assertEqual({"D404": False, "D403": True}, by_slot)

    def test_safety_parking_without_privileged_witness_does_not_require_registry(self):
        parking = {
            "type": "parking_slot",
            "id": "slot:A1",
            "slot_number": "A1",
            "polygon": [[0, 0], [3, 0], [3, 6], [0, 6], [0, 0]],
            "heading_deg": 0.0,
        }
        route = {
            "layer": "Safety",
            "town": "TestParking",
            "target": {
                "type": "if_else",
                "condition": "occupied:A1",
                "then": dict(parking),
                "else": {**parking, "id": "slot:A2", "slot_number": "A2"},
            },
            "actor_setup": {"slot_occupancy": {"A1": True}},
        }
        _validate_exact_parking_manoeuvre(route, None, None)

        route["target"]["then"]["reverse_manoeuvre"] = {}
        with self.assertRaisesRegex(
            PrivateSidecarError, "parking_manoeuvre_registry"
        ):
            _validate_exact_parking_manoeuvre(route, None, None)

    def test_core_speed_feasibility_cap_is_strictly_loaded(self):
        oracle_path = [
            {"x": float(index), "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
            for index in range(61)
        ]
        route = {
            "route_entry_id": 5,
            "pcu_id": "B-001",
            "arm_id": "low", "paraphrase": 0,
            "instruction": "Keep the low speed band.",
            "town": "Town03", "split": "dev", "layer": "B",
            "target": {"type": "speed_band", "min_mps": 3, "max_mps": 5},
            "contract": {"eventually": {"speed_band": {
                "min_mps": 3, "max_mps": 5,
            }}},
            "opposite_contract": {}, "actor_setup": {},
            "deadline_seconds": 20,
            "oracle_path": oracle_path,
            "legal_speed_mps": 10,
            "speed_feasibility_cap_proof": {
                "method": "carla_default_limit_conservative_cap_v1",
                "cap_mps": 8.333333,
                "control_margin_mps": 0.35,
            },
            "signal_free_corridor_proof": {
                "method": "exact_signal_stop_free_lane_slice_v2",
                "lane_id": "Town03:7:0:-1",
                "path_start_index_inclusive": 20,
                "path_end_index_inclusive": 80,
                "buffer_start_index_inclusive": 0,
                "buffer_end_index_inclusive": 100,
                "road_s_min": 20.0,
                "road_s_max": 80.0,
                "corridor_length_m": 60.0,
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
                "execution_path_sha256": stable_hash(oracle_path),
                "safety_buffer_path_sha256": "b" * 64,
                "lane_graph_sha256": "c" * 64,
                "xodr_sha256": "d" * 64,
                "audit_sha256": "e" * 64,
            },
        }
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [route],
            "pcus": [{
                "id": "B-001", "layer": "B", "town": "Town03",
                "split": "dev", "arms": [],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "sidecar.private.json"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            context = load_private_route_context(str(sidecar), 5)
            self.assertEqual(
                8.333333,
                context["route"]["speed_feasibility_cap_proof"]["cap_mps"],
            )
            for mutation, message in (
                (lambda item: item.pop("speed_feasibility_cap_proof"),
                 "requires speed_feasibility_cap_proof"),
                (lambda item: item["speed_feasibility_cap_proof"].update(
                    {"unknown": True}), "keys must be exactly"),
                (lambda item: item["speed_feasibility_cap_proof"].update(
                    {"cap_mps": 5.2}), "requires cap_mps=8.333333"),
                (lambda item: item["speed_feasibility_cap_proof"].update(
                    {"cap_mps": 100}), "requires cap_mps=8.333333"),
            ):
                broken = json.loads(json.dumps(payload))
                mutation(broken["routes"][0])
                sidecar.write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaisesRegex(PrivateSidecarError, message):
                    load_private_route_context(str(sidecar), 5)

            for mutation, message in (
                (lambda item: item.pop("signal_free_corridor_proof"),
                 "requires signal_free_corridor_proof"),
                (lambda item: item["signal_free_corridor_proof"].update(
                    {"unversioned": True}), "keys must be exactly"),
                (lambda item: item["signal_free_corridor_proof"].update(
                    {"execution_path_sha256": "f" * 64}),
                 "does not bind serialized oracle_path"),
                (lambda item: item["signal_free_corridor_proof"].update(
                    {"path_end_index_inclusive": 81}),
                 "indices do not close oracle_path point count"),
                (lambda item: item["signal_free_corridor_proof"].update(
                    {"map_road_signal_count": 1}),
                 "null clearance requires zero map signals"),
            ):
                broken = json.loads(json.dumps(payload))
                mutation(broken["routes"][0])
                sidecar.write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaisesRegex(PrivateSidecarError, message):
                    load_private_route_context(str(sidecar), 5)

        safety = json.loads(json.dumps(payload))
        safety["routes"][0]["layer"] = "Safety"
        safety["pcus"][0]["layer"] = "Safety"
        safety["routes"][0].pop("speed_feasibility_cap_proof")
        safety["routes"][0].pop("signal_free_corridor_proof")
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "safety.private.json"
            sidecar.write_text(json.dumps(safety), encoding="utf-8")
            load_private_route_context(str(sidecar), 5)

        malformed = json.loads(json.dumps(payload))
        malformed["routes"][0]["contract"] = {
            "eventually": {"speed_band": []}
        }
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "malformed.private.json"
            sidecar.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(
                PrivateSidecarError, "speed_band must be an object"
            ):
                load_private_route_context(str(sidecar), 5)

    def test_private_spawn_must_match_public_xml_start(self):
        context = {
            "route": {
                "oracle_path": [
                    {"x": 1.10, "y": 2.05, "z": 0.0, "yaw_deg": 0.0},
                    {"x": 5.0, "y": 2.0, "z": 0.0, "yaw_deg": 0.0},
                ]
            }
        }
        public = [{"x": 1.0, "y": 2.0, "z": 0.0}]
        self.assertEqual(
            [], private_public_start_consistency_errors(context, public)
        )
        context["route"]["oracle_path"][0]["x"] = 1.26
        errors = private_public_start_consistency_errors(context, public)
        self.assertEqual(1, len(errors))
        self.assertIn("spawn horizontal mismatch", errors[0])
        self.assertTrue(
            private_public_start_consistency_errors(context, [])
        )

    def test_private_sidecar_schema_is_required_and_versioned(self):
        for schema_version in (None, "1.0", "language2drive.private_sidecar.v2"):
            payload = {"routes": [], "pcus": []}
            if schema_version is not None:
                payload["schema_version"] = schema_version
            with self.subTest(schema_version=schema_version):
                with tempfile.TemporaryDirectory() as temporary:
                    sidecar = Path(temporary) / "sidecar.private.json"
                    sidecar.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        PrivateSidecarError, "unsupported private sidecar schema"
                    ):
                        load_private_route_context(str(sidecar), 0)

    def test_region_pose_never_becomes_map_legality_allowlist(self):
        own_polygon = [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]
        opposite_polygon = [[10, 0], [14, 0], [14, 4], [10, 4], [10, 0]]
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [{
                "route_entry_id": 3,
                "pcu_id": "C-001",
                "arm_id": "own", "paraphrase": 0,
                "instruction": "Enter the own region.",
                "town": "semantic", "split": "dev", "layer": "C",
                "target": {
                    "type": "region",
                    "id": "own",
                    "polygon": own_polygon,
                    "terminal_pose": {
                        "x": 2, "y": 2, "z": 0, "yaw_deg": 90,
                    },
                },
                "contract": {"eventually": {"inside_region": {
                    "polygon": own_polygon, "tolerance_m": 9.0,
                }}},
                "opposite_contract": {"eventually": {"inside_region": {
                    "polygon": opposite_polygon,
                }}},
                "actor_setup": {},
                "deadline_seconds": 10,
                "oracle_path": [{"x": 0, "y": 0}, {"x": 2, "y": 2}],
            }],
            "pcus": [{"id": "C-001", "layer": "C", "town": "semantic",
                       "split": "dev", "arms": []}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "sidecar.json"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            context = load_private_route_context(str(sidecar), 3)
        self.assertNotIn("semantic_terminal_definitions", context)

    def test_region_without_explicit_terminal_pose_cannot_authorize_offroad(self):
        polygon = [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [{
                "route_entry_id": 4,
                "pcu_id": "C-002",
                "arm_id": "own", "paraphrase": 0,
                "instruction": "Enter the region.",
                "town": "semantic", "split": "dev", "layer": "C",
                "target": {"type": "region", "polygon": polygon},
                "contract": {"eventually": {"inside_region": {
                    "polygon": polygon,
                }}},
                "opposite_contract": {},
                "actor_setup": {}, "deadline_seconds": 10,
                "oracle_path": [{"x": 0, "y": 0}, {"x": 10, "y": 10}],
            }],
            "pcus": [{"id": "C-002", "layer": "C", "town": "semantic",
                       "split": "dev", "arms": []}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "sidecar.json"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            context = load_private_route_context(str(sidecar), 4)
        self.assertNotIn("semantic_terminal_definitions", context)

    def test_route_lookup_hash_target_index_and_identity_check(self):
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [
                {
                    "route_entry_id": 9,
                    "pcu_id": "B-001",
                    "arm_id": "fast",
                    "paraphrase": 2,
                    "instruction": "Take the fast corridor.",
                    "town": "Town01", "split": "dev", "layer": "B",
                    "target": {"type": "gate", "id": "g",
                               "polygon": [[0, 0], [1, 0], [0, 1]]},
                    "contract": {"eventually": {"enter_gate": {"id": "g"}}},
                    "opposite_contract": {"eventually": {"enter_gate": {"id": "h"}}},
                    "actor_setup": {},
                    "deadline_seconds": 5,
                    "oracle_path": [],
                }
            ],
            "pcus": [
                {
                    "id": "B-001",
                    "layer": "B",
                    "town": "Town01", "split": "dev",
                    "arms": [
                        {"target": {"type": "gate", "id": "g", "polygon": [[0, 0], [1, 0], [0, 1]]}}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "sidecar.private.json"
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            context = load_private_route_context(str(sidecar), 9)
        self.assertEqual(64, len(context["sidecar_sha256"]))
        self.assertEqual("g", context["capture_targets"][0]["id"])
        self.assertEqual(
            [],
            public_private_consistency_errors(
                context, {"pcu_id": "B-001", "arm_id": "fast", "paraphrase": "2"}
            ),
        )
        context["route"].update(
            {"town": "Town01", "instruction": "Take the fast corridor."}
        )
        errors = public_private_consistency_errors(
            context,
            {
                "route_entry_id": 9,
                "town": "Town02",
                "instruction": "Take the slow corridor.",
            },
        )
        self.assertTrue(any("town mismatch" in error for error in errors))
        self.assertTrue(any("instruction mismatch" in error for error in errors))

    def test_real_sidecar_resolves_circle_memory_occupancy_and_parking(self):
        sidecar = L2D_V1_ROOT / "build/suite/sidecar.private.json"

        until_context = load_private_route_context(str(sidecar), 672)
        until_ids = set(until_context["requirements"]["gate_ids"])
        self.assertTrue(until_ids)
        self.assertEqual(
            until_ids,
            {target["id"] for target in until_context["capture_targets"]},
        )
        self.assertTrue(
            all(
                target.get("polygon")
                or (target.get("center") and target.get("radius_m") is not None)
                for target in until_context["capture_targets"]
            )
        )

        memory_context = load_private_route_context(str(sidecar), 720)
        self.assertEqual(2, len(memory_context["requirements"]["gate_ids"]))
        self.assertEqual(
            set(memory_context["requirements"]["gate_ids"]),
            {target["id"] for target in memory_context["capture_targets"]},
        )

        # During the atomic v3 source/suite migration, the pre-v3 build must
        # fail closed. Once the fresh registry-backed suite lands this same
        # test continues into the real occupancy/parking assertions below.
        try:
            occupancy_context = load_private_route_context(str(sidecar), 624)
        except PrivateSidecarError as exc:
            expected_pre_v3_failure = (
                "reverse_manoeuvre v3" in str(exc)
                or "parking_manoeuvre_registry" in str(exc)
            )
            if not expected_pre_v3_failure:
                raise
            with self.assertRaisesRegex(
                PrivateSidecarError,
                "parking_manoeuvre_registry|reverse_manoeuvre v3",
            ):
                load_private_route_context(str(sidecar), 528)
            return
        slots = occupancy_context["requirements"]["occupancy_slots"]
        actors = occupancy_context["requirements"]["actor_ids"]
        self.assertEqual(1, len(slots))
        self.assertEqual(1, len(actors))
        self.assertIn(slots[0], occupancy_context["actor_spawn_targets"])
        setup_actors = occupancy_context["route"]["actor_setup"]["actors"]
        self.assertEqual(slots[0], setup_actors[0]["slot"])
        self.assertEqual(actors[0], setup_actors[0]["id"])

        parking_context = load_private_route_context(str(sidecar), 528)
        self.assertEqual(2, len(parking_context["requirements"]["dwell_ids"]))
        self.assertEqual(2, len(parking_context["capture_targets"]))

    def test_geometry_registry_resolves_cross_pcu_stage_gates(self):
        sidecar = L2D_V1_ROOT / "build/suite/sidecar.private.json"
        context = load_private_route_context(str(sidecar), 576)
        gate_ids = set(context["requirements"]["gate_ids"])
        self.assertTrue(gate_ids)
        self.assertEqual(
            gate_ids, {target["id"] for target in context["capture_targets"]}
        )
        self.assertEqual([], context["unresolved_capture_target_ids"])
        self.assertEqual(
            "standardized__B-001__fast__p2__s3",
            expected_run_id(
                "standardized",
                {"pcu_id": "B-001", "arm_id": "fast", "paraphrase": 2},
                3,
            ),
        )

    def test_geometry_registry_conflict_with_embedded_geometry_is_rejected(self):
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [
                {
                    "route_entry_id": 1,
                    "pcu_id": "D-001",
                    "arm_id": "a",
                    "paraphrase": 0, "instruction": "Use gate g.",
                    "town": "Town01", "split": "dev", "layer": "D",
                    "target": {"type": "gate", "id": "g",
                               "polygon": [[0, 0], [1, 0], [0, 1]]},
                    "contract": {"eventually": {"enter_gate": {"id": "g"}}},
                    "opposite_contract": {
                        "eventually": {"enter_gate": {"id": "h"}}
                    },
                    "actor_setup": {},
                    "deadline_seconds": 5,
                    "oracle_path": [],
                }
            ],
            "pcus": [
                {
                    "id": "D-001",
                    "town": "Town01", "split": "dev", "layer": "D",
                    "source_refs": ["test://gate-registry"],
                    "arms": [
                        {
                            "target": {
                                "type": "gate",
                                "id": "g",
                                "polygon": [[0, 0], [1, 0], [0, 1]],
                            }
                        },
                        {
                            "target": {
                                "type": "gate",
                                "id": "h",
                                "polygon": [[2, 0], [3, 0], [2, 1]],
                            }
                        },
                    ],
                }
            ],
            "geometry_registry": {
                "g": {
                    "type": "gate",
                    "id": "g",
                    "polygon": [[10, 0], [11, 0], [10, 1]],
                    "geometry_version": "test-v1",
                    "source_ref": "test://gate-registry",
                    "source_sha256": "a" * 64,
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                PrivateSidecarError, "conflicting target geometry"
            ):
                load_private_route_context(str(path), 1)

    def test_parking_capture_tolerances_cannot_expand_safe_legality(self):
        target = {
            "type": "parking_slot", "id": "slot:A1", "slot_number": "A1",
            "polygon": [[0, 0], [3, 0], [3, 6], [0, 6], [0, 0]],
            "heading_deg": 0.0, "heading_tolerance_deg": 15.0,
            "speed_max_mps": 0.2, "tolerance_m": 0.21,
        }
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [{
                "route_entry_id": 1, "pcu_id": "C-001", "arm_id": "a",
                "paraphrase": 0, "instruction": "Park in A1.",
                "town": "parking", "split": "dev", "layer": "C",
                "target": target,
                "contract": {"eventually": {
                    "vehicle_footprint_inside_slot": {"id": "slot:A1"}
                }},
                "opposite_contract": {}, "actor_setup": {},
                "deadline_seconds": 10, "oracle_path": [],
            }],
            "pcus": [{
                "id": "C-001", "town": "parking", "split": "dev",
                "layer": "C", "arms": [{"target": target}],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            # Capture tolerances are validated before executable manoeuvre
            # qualification, so an oversized allowance remains the explicit
            # rejection reason even if the synthetic target lacks v2 fields.
            with self.assertRaisesRegex(
                PrivateSidecarError, "tolerance_m=.*outside"
            ):
                load_private_route_context(str(path), 1)

    def test_conflicting_legacy_geometry_without_registry_is_rejected(self):
        payload = {
            "schema_version": "language2drive.private_sidecar.v1",
            "routes": [
                {
                    "route_entry_id": 1,
                    "pcu_id": "D-001",
                    "arm_id": "a",
                    "paraphrase": 0, "instruction": "Use gate g.",
                    "town": "Town01", "split": "dev", "layer": "D",
                    "target": {"type": "gate", "id": "g",
                               "polygon": [[0, 0], [1, 0], [0, 1]]},
                    "contract": {"eventually": {"enter_gate": {"id": "g"}}},
                    "opposite_contract": {
                        "eventually": {"enter_gate": {"id": "h"}}
                    },
                    "actor_setup": {},
                    "deadline_seconds": 5,
                    "oracle_path": [],
                }
            ],
            "pcus": [
                {
                    "id": "D-001",
                    "town": "Town01", "split": "dev", "layer": "D",
                    "arms": [
                        {
                            "target": {
                                "type": "gate", "id": "g",
                                "polygon": [[0, 0], [1, 0], [0, 1]],
                            }
                        },
                        {
                            "target": {
                                "type": "gate", "id": "h",
                                "polygon": [[2, 0], [3, 0], [2, 1]],
                            }
                        },
                    ],
                },
                {
                    "id": "D-002",
                    "town": "Town01", "split": "dev", "layer": "D",
                    "arms": [
                        {
                            "target": {
                                "type": "gate", "id": "g",
                                "polygon": [[10, 0], [11, 0], [10, 1]],
                            }
                        }
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                PrivateSidecarError, "conflicting target geometry"
            ):
                load_private_route_context(str(path), 1)


class TraceCaptureTest(unittest.TestCase):
    def setUp(self):
        self.original_available = logger_module._CARLA_AVAILABLE
        self.original_provider = getattr(logger_module, "CarlaDataProvider", None)
        logger_module._CARLA_AVAILABLE = True
        logger_module.CarlaDataProvider = Provider

    def tearDown(self):
        logger_module._CARLA_AVAILABLE = self.original_available
        if self.original_provider is None:
            delattr(logger_module, "CarlaDataProvider")
        else:
            logger_module.CarlaDataProvider = self.original_provider

    def _parking_legality_events(
        self,
        *,
        x: float = 52.5741,
        y: float = -57.6179,
        yaw: float = -29.7282,
        road_id: int = 12,
        lane_id: int = -1,
        frame: int = 29937,
        sim_time: float = 68.65,
        controller_source: str,
        has_lane: bool = True,
        has_parking_definition: bool = True,
        through_collect_frame: bool = False,
    ):
        class LaneTypeStub:
            Driving = "Driving"
            Parking = "Parking"

        class CarlaStub:
            LaneType = LaneTypeStub

        class SnapshotWaypoint:
            section_id = 0
            s = 30.2368
            is_junction = False
            lane_type = "Driving"

            def __init__(self):
                self.road_id = road_id
                self.lane_id = lane_id
                self.transform = Transform(yaw=yaw + 180.0)

        waypoint = SnapshotWaypoint()

        class SnapshotMap:
            @staticmethod
            def get_waypoint(_location, **_kwargs):
                return waypoint if has_lane else None

        context = {
            "route": {"actor_setup": {"actors": []}},
            "requirements": {"actor_ids": []},
            "capture_targets": [],
        }
        if has_parking_definition:
            context["capture_targets"].append({
                "type": "parking_slot",
                "id": "parkinglotfish-new3:xodr-parking:131",
                "polygon": [
                    [100.0, 100.0], [106.0, 100.0], [106.0, 103.0],
                    [100.0, 103.0], [100.0, 100.0],
                ],
                "tolerance_m": 0.2,
            })
        ego = Actor(1, "hero", x, type_id="vehicle.ego")
        ego._transform = Transform(x, y, yaw)
        Provider.world = World([ego])
        with patch.object(logger_module, "carla", CarlaStub), patch.object(
            Provider, "get_map", return_value=SnapshotMap()
        ), tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(
                temporary,
                {"town_name": "parkinglotfish-new3"},
                private_context=context,
            )
            if through_collect_frame:
                row = logger._collect_frame(
                    frame,
                    Timestamp(sim_time),
                    ego,
                    Action(),
                    controller_source=controller_source,
                )
                return row["events"]
            return logger._map_legality_event(
                ego,
                ego.get_transform(),
                sim_time,
                frame,
                controller_source=controller_source,
            )

    def test_route_156_624_630_snapshots_exempt_explicit_parking_stage(self):
        snapshots = (
            # Fresh failures before the controller-aware legality fix.
            (156, 29937, 68.65, 52.5741, -57.6179, -29.7282, 12, -1),
            (624, 53471, 95.30, -27.3169, -53.4068, -31.4156, 13, -1),
            (630, 55531, 88.15, -38.4835, -52.8084, -31.4883, 13, -1),
        )
        for route_id, frame, sim_time, x, y, yaw, road_id, lane_id in snapshots:
            with self.subTest(route_entry_id=route_id):
                events = self._parking_legality_events(
                    x=x,
                    y=y,
                    yaw=yaw,
                    road_id=road_id,
                    lane_id=lane_id,
                    frame=frame,
                    sim_time=sim_time,
                    controller_source=(
                        "privileged_oracle:parking_staging_forward"
                    ),
                    through_collect_frame=True,
                )
                self.assertEqual([], events)

    def test_all_audited_explicit_parking_stages_exempt_wrong_direction(self):
        for source in (
            "privileged_oracle:parking_staging_forward",
            "privileged_oracle:parking_staging_stop",
            "privileged_oracle:parking_reverse_path",
            "privileged_oracle:parking_dwell",
        ):
            with self.subTest(controller_source=source):
                self.assertEqual(
                    [], self._parking_legality_events(controller_source=source)
                )

    def test_parking_prelead_and_ordinary_sources_still_reject_wrong_direction(self):
        for source in (
            "privileged_oracle:parking_v5_prelead_capture",
            "privileged_oracle:path",
        ):
            with self.subTest(controller_source=source):
                events = self._parking_legality_events(controller_source=source)
                self.assertEqual(1, len(events))
                self.assertEqual("wrong_direction", events[0]["reason"])

    def test_explicit_parking_stage_outside_lane_still_rejects(self):
        events = self._parking_legality_events(
            controller_source="privileged_oracle:parking_reverse_path",
            has_lane=False,
        )
        self.assertEqual(1, len(events))
        self.assertEqual(
            "outside_driving_and_parking_lanes", events[0]["reason"]
        )

    def test_explicit_stage_without_current_parking_definition_still_rejects(self):
        events = self._parking_legality_events(
            controller_source="privileged_oracle:parking_reverse_path",
            has_parking_definition=False,
        )
        self.assertEqual(1, len(events))
        self.assertEqual("wrong_direction", events[0]["reason"])

    def test_contract_referenced_world_capture_and_offline_result(self):
        ego = Actor(1, "hero", 5.0, type_id="vehicle.ego")
        lead = Actor(2, "lead", 10.0)
        unrelated = Actor(3, "bystander", 8.0)
        Provider.world = World([ego, lead, unrelated])
        context = private_context()
        postprocessor = Language2DriveTracePostprocessor(
            context,
            str(L2D_V1_ROOT),
            stability_seconds=2.0,
            progress_minimum=0.05,
        )
        route_meta = {
            "town_name": "Town01",
            "language2drive_public_context": {
                "instruction": "Take the left gate.",
                "instruction_version": 0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(
                temporary,
                route_meta,
                private_context=context,
                postprocessor=postprocessor,
                capture_provenance={"route_entry_id": 0, "private_sidecar_sha256": "abc"},
            )
            logger.start_route()
            scenario = Scenario()
            for second, x in enumerate((5.0, 6.0, 7.0)):
                ego._transform = Transform(x, 0.0)
                logger.log_frame(
                    second * 20,
                    Timestamp(float(second)),
                    ego,
                    Action(),
                    scenario=scenario,
                    controller_source="model_plan_tracker",
                )
            logger.close_route({"termination": "test"})
            root = Path(temporary)
            rows = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            result = json.loads((root / "contract_result.json").read_text())
            capabilities = json.loads((root / "capture_capabilities.json").read_text())
            run_meta = json.loads((root / "run_meta.json").read_text())

        self.assertEqual("Town01:7:0:-1", rows[0]["ego"]["lane_relation"])
        self.assertEqual(12.5, rows[0]["ego"]["s"])
        self.assertEqual(2.0, rows[0]["ego"]["speed_mps"])
        self.assertEqual("model_plan_tracker", rows[0]["controller_source"])
        self.assertEqual(["lead"], [actor["semantic_id"] for actor in rows[0]["actors"]])
        self.assertEqual(2, rows[0]["actors"][0]["instance_id"])
        self.assertTrue(rows[0]["actors"][0]["bounding_box"])
        self.assertIn("lights", rows[0]["actors"][0])
        self.assertEqual(0.5, rows[0]["progress"])
        self.assertEqual(1, sum(event["type"] == "gate" for event in events))
        # Actors present in the first world snapshot are initial observations,
        # not fabricated mid-rollout spawn events.  The evaluator actor runtime
        # records its own authoritative pre-trace spawn event separately.
        self.assertEqual(0, sum(event["type"] == "actor_spawn" for event in events))
        self.assertTrue(result["success"])
        self.assertEqual(70.0, result["driving_score"])
        self.assertTrue(result["capture_complete"])
        self.assertEqual([], result["unsupported_required_fields"])
        self.assertEqual(["lead"], capabilities["resolved_actor_ids"])
        self.assertEqual("abc", run_meta["private_capture_provenance"]["private_sidecar_sha256"])
        self.assertNotIn("route", run_meta)

    def test_authoritative_collision_event_is_normalized_and_deduplicated(self):
        ego = Actor(1, "hero", 0.0, type_id="vehicle.ego")
        Provider.world = World([ego])
        event = TrafficEvent("COLLISION_VEHICLE", frame=4)
        scenario = Scenario([event])
        with tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(temporary, {"town_name": "Town01"})
            logger.start_route()
            logger.log_frame(4, Timestamp(0.2), ego, Action(), scenario=scenario)
            logger.log_frame(5, Timestamp(0.25), ego, Action(), scenario=scenario)
            logger.close_route()
            events = [json.loads(line) for line in (Path(temporary) / "events.jsonl").read_text().splitlines()]
        collisions = [value for value in events if value["type"] == "collision"]
        self.assertEqual(1, len(collisions))
        self.assertEqual("scenario_criterion", collisions[0]["source"])
        self.assertEqual("COLLISION_VEHICLE", collisions[0]["source_event_type"])

    def test_stop_infraction_is_normalized_but_not_in_default_safe_scope(self):
        ego = Actor(1, "hero", 0.0, type_id="vehicle.ego")
        Provider.world = World([ego])
        event = TrafficEvent("STOP_INFRACTION", frame=4)
        criterion = type("RunningStopTest", (), {"events": [event]})()
        scenario = Scenario([])
        scenario.get_criteria = lambda: [criterion]
        with tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(temporary, {"town_name": "Town01"})
            logger.start_route()
            logger.log_frame(4, Timestamp(0.2), ego, Action(), scenario=scenario)
            logger.close_route()
            events = [
                json.loads(line)
                for line in (Path(temporary) / "events.jsonl").read_text().splitlines()
            ]
            capabilities = json.loads(
                (Path(temporary) / "capture_capabilities.json").read_text()
            )
        self.assertEqual("stop_sign", events[0]["type"])
        self.assertEqual("STOP_INFRACTION", events[0]["source_event_type"])
        self.assertTrue(capabilities["criterion_event_support"]["stop_sign"])
        # The default contract engine scope is intentionally unchanged.
        from language2drive.contracts import ContractEngine
        self.assertTrue(ContractEngine([{
            "sim_time": 0.2,
            "events": [{"type": "stop_sign"}],
        }]).evaluate({"safe_legal": {}}).passed)
        self.assertFalse(ContractEngine([{
            "sim_time": 0.2,
            "events": [{"type": "stop_sign"}],
        }]).evaluate({"safe_legal": {
            "forbidden_events": ["stop_sign"],
        }}).passed)

    def test_map_legality_rejects_region_polygon_without_real_lane(self):
        class LaneTypeStub:
            Driving = "Driving"
            Parking = "Parking"

        class CarlaStub:
            LaneType = LaneTypeStub

        class NoLaneMap:
            @staticmethod
            def get_waypoint(_location, **_kwargs):
                return None

        original_carla = logger_module.carla
        original_get_map = Provider.__dict__["get_map"]
        logger_module.carla = CarlaStub
        Provider.get_map = staticmethod(lambda: NoLaneMap())
        context = {
            "semantic_terminal_definitions": [{
                "polygon": [[1.65, 1.65], [2.35, 1.65], [2.35, 2.35],
                            [1.65, 2.35], [1.65, 1.65]],
                "tolerance_m": 0.0,
                "source": "own_contract.inline_inside_region.terminal_pose",
                "contract_sha256": "a" * 64,
            }]
        }
        try:
            with tempfile.TemporaryDirectory() as temporary:
                logger = ClosedLoopLogger(
                    temporary, {"town_name": "semantic"}, private_context=context
                )
                ego = Actor(1, "hero", 2.0, type_id="vehicle.ego")
                inside = logger._map_legality_event(
                    ego, Transform(2.0, 2.0), 0.0, 0
                )
                logger_outside = ClosedLoopLogger(
                    temporary, {"town_name": "semantic"}, private_context=context
                )
                outside = logger_outside._map_legality_event(
                    ego, Transform(8.0, 8.0), 0.05, 1
                )
            self.assertEqual(1, len(inside))
            self.assertEqual("off_road", inside[0]["type"])
            self.assertEqual(1, len(outside))
            self.assertEqual("off_road", outside[0]["type"])
        finally:
            logger_module.carla = original_carla
            Provider.get_map = original_get_map

    def test_missing_authoritative_gate_fails_closed_with_audit_reason(self):
        context = private_context()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {
                "sim_time": 0.0,
                "ego": {"location": {"x": 5.0, "y": 0.0}, "speed_mps": 2.0},
                "events": [{"type": "gate", "id": "gate:left"}],
            }
            (root / "trace.jsonl").write_text(json.dumps(row) + "\n")
            postprocessor = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=0.0
            )
            result = postprocessor(
                temporary,
                {
                    "resolved_gate_ids": [],
                    "resolved_actor_ids": ["lead"],
                    "resolved_dwell_ids": [],
                    "criterion_event_support": {
                        "collision": True,
                        "off_road": True,
                        "red_light": True,
                    },
                    "lane_samples": 1,
                },
            )
        self.assertFalse(result["success"])
        self.assertFalse(result["capture_complete"])
        self.assertIn(
            "gate:gate:left:geometry_or_capture_missing",
            result["unsupported_required_fields"],
        )

    def test_contract_result_carries_private_termination_and_cleanup(self):
        context = private_context()
        context["runtime_required"] = True
        rows = [
            {
                "frame": frame,
                "sim_time": round(frame * 0.05, 4),
                "ego": {
                    "location": {"x": float(5 + frame * 0.05), "y": 0.0},
                    "speed_mps": 2.0,
                },
                "events": (
                    [{"type": "gate", "id": "gate:left"}]
                    if frame == 0
                    else []
                ),
            }
            for frame in range(41)
        ]
        capture_report = {
            "resolved_gate_ids": ["gate:left", "gate:right"],
            "resolved_actor_ids": ["lead"],
            "resolved_dwell_ids": [],
            "criterion_event_support": {
                "collision": True,
                "off_road": True,
                "red_light": True,
            },
            "lane_samples": 41,
            "traffic_light_samples": 41,
            "map_legality_samples": 41,
            "controller_source_samples": 41,
            "unknown_controller_source_samples": 0,
            "logger_error_count": 0,
            "evaluator_runtime": {
                "setup_complete": True,
                "cleanup_complete": True,
                "failure_reason": None,
                "termination_reason": "contract_complete",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=2.0
            )(temporary, capture_report)
        self.assertTrue(result["success"])
        self.assertEqual("contract_complete", result["termination_reason"])
        self.assertIsNone(result["runtime_failure_reason"])

    def test_out_of_world_termination_is_complete_failed_capture(self):
        context = private_context()
        context["runtime_required"] = True
        rows = [
            {
                "frame": 0,
                "sim_time": 0.0,
                "ego": {
                    "location": {"x": 10.0, "y": 0.0, "z": 0.0},
                    "speed_mps": 20.0,
                },
                "events": [],
            },
            {
                "frame": 1,
                "sim_time": 0.05,
                "ego": {
                    "location": {"x": 20.0, "y": 0.0, "z": -51.0},
                    "speed_mps": 20.0,
                },
                "events": [{
                    "type": "off_road",
                    "reason": "ego_out_of_world",
                    "source": "evaluator_terminal_guard",
                }],
            },
        ]
        capture_report = {
            "resolved_gate_ids": ["gate:left", "gate:right"],
            "resolved_actor_ids": ["lead"],
            "resolved_dwell_ids": [],
            "criterion_event_support": {
                "collision": True,
                "off_road": True,
                "red_light": True,
            },
            "lane_samples": 2,
            "traffic_light_samples": 2,
            "map_legality_samples": 2,
            "controller_source_samples": 2,
            "unknown_controller_source_samples": 0,
            "logger_error_count": 0,
            "evaluator_runtime": {
                "setup_complete": True,
                "cleanup_complete": True,
                "failure_reason": None,
                "termination_reason": "ego_out_of_world",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=0.0
            )(temporary, capture_report)

        self.assertEqual("complete", result["status"])
        self.assertTrue(result["capture_complete"])
        self.assertFalse(result["success"])
        self.assertFalse(result["safe_legal"])
        self.assertEqual("ego_out_of_world", result["termination_reason"])

    def test_partial_trace_with_missing_private_termination_is_not_complete(self):
        context = private_context()
        context["runtime_required"] = True
        rows = [
            {
                "frame": frame,
                "sim_time": round(frame * 0.05, 4),
                "ego": {
                    "location": {"x": 0.1 * frame, "y": 0.0},
                    "speed_mps": 2.0,
                },
                "events": [],
            }
            for frame in range(115)
        ]
        report = {
            "resolved_gate_ids": ["gate:left", "gate:right"],
            "resolved_actor_ids": ["lead"],
            "resolved_dwell_ids": [],
            "criterion_event_support": {
                "collision": True, "off_road": True, "red_light": True,
            },
            "lane_samples": len(rows),
            "traffic_light_samples": len(rows),
            "map_legality_samples": len(rows),
            "controller_source_samples": len(rows),
            "unknown_controller_source_samples": 0,
            "logger_error_count": 0,
            "evaluator_runtime": {
                "setup_complete": True,
                "cleanup_complete": True,
                "failure_reason": None,
                "termination_reason": None,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=0.0
            )(temporary, report)
        self.assertFalse(result["success"])
        self.assertFalse(result["capture_complete"])
        self.assertIsNone(result["termination_reason"])
        self.assertIn(
            "evaluator_runtime:termination_missing",
            result["unsupported_required_fields"],
        )

    def test_offline_success_matches_online_candidate_stability_and_termination(self):
        context = private_context()
        context["runtime_required"] = True
        base_report = {
            "resolved_gate_ids": ["gate:left", "gate:right"],
            "resolved_actor_ids": ["lead"],
            "resolved_dwell_ids": [],
            "criterion_event_support": {
                "collision": True, "off_road": True, "red_light": True,
            },
            "lane_samples": 3,
            "traffic_light_samples": 3,
            "map_legality_samples": 3,
            "controller_source_samples": 3,
            "unknown_controller_source_samples": 0,
            "logger_error_count": 0,
            "evaluator_runtime": {
                "setup_complete": True,
                "cleanup_complete": True,
                "failure_reason": None,
                "termination_reason": "contract_complete",
            },
        }
        rows = [{
            "frame": frame,
            "sim_time": round(frame * 0.05, 4),
            "ego": {"location": {
                "x": 0.0 if frame < 60 else 1.0, "y": 0.0,
            }},
            "events": (
                [{"type": "gate", "id": "gate:left"}]
                if frame == 0 else []
            ),
        } for frame in range(61)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            too_early = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=2.0
            )(temporary, base_report)
        self.assertFalse(too_early["success"])
        self.assertFalse(too_early["stability"])

        rows.extend({
            "frame": frame,
            "sim_time": round(frame * 0.05, 4),
            "ego": {"location": {"x": 2.0, "y": 0.0}},
            "events": [],
        } for frame in range(61, 101))
        crash_report = json.loads(json.dumps(base_report))
        crash_report["evaluator_runtime"]["termination_reason"] = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            forced_close = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=2.0
            )(temporary, crash_report)
        self.assertTrue(forced_close["stability"])
        self.assertFalse(forced_close["success"])
        self.assertIsNone(forced_close["termination_reason"])

    def test_runtime_trace_requires_consecutive_twenty_hz_frames(self):
        context = private_context()
        context["runtime_required"] = True
        rows = [
            {"frame": 10, "sim_time": 0.0,
             "ego": {"location": {"x": 5.0, "y": 0.0}},
             "events": [{"type": "gate", "id": "gate:left"}]},
            {"frame": 12, "sim_time": 0.2,
             "ego": {"location": {"x": 7.0, "y": 0.0}}, "events": []},
        ]
        report = {
            "resolved_gate_ids": ["gate:left", "gate:right"],
            "resolved_actor_ids": ["lead"], "resolved_dwell_ids": [],
            "criterion_event_support": {
                "collision": True, "off_road": True, "red_light": True,
            },
            "lane_samples": 2, "traffic_light_samples": 2,
            "map_legality_samples": 2, "controller_source_samples": 2,
            "unknown_controller_source_samples": 0, "logger_error_count": 0,
            "evaluator_runtime": {
                "setup_complete": True, "cleanup_complete": True,
                "failure_reason": None,
                "termination_reason": "contract_complete",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = Language2DriveTracePostprocessor(
                context, str(L2D_V1_ROOT), stability_seconds=0.0
            )(temporary, report)
            capabilities = json.loads(
                (root / "capture_capabilities.json").read_text()
            )
        self.assertFalse(result["success"])
        self.assertFalse(result["capture_complete"])
        self.assertIn(
            "trace:non_consecutive_20hz_cadence",
            result["unsupported_required_fields"],
        )
        self.assertFalse(capabilities["trace_cadence_complete"])
        self.assertTrue(capabilities["trace_cadence_errors"])


class RuntimeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class RuntimeRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class RuntimeTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or RuntimeLocation()
        self.rotation = rotation or RuntimeRotation()


class RuntimeLaneType:
    Driving = "Driving"


class RuntimeCarla:
    Location = RuntimeLocation
    Rotation = RuntimeRotation
    Transform = RuntimeTransform
    LaneType = RuntimeLaneType


class RuntimeWaypoint:
    def __init__(self, location):
        self.transform = RuntimeTransform(
            RuntimeLocation(location.x, location.y, 0.3), RuntimeRotation(yaw=5.0)
        )


class RuntimeMap:
    @staticmethod
    def get_waypoint(location, **_kwargs):
        return RuntimeWaypoint(location)


class RuntimeActor:
    def __init__(self, instance_id, blueprint, role_name):
        self.id = instance_id
        self.type_id = blueprint
        self.attributes = {"role_name": role_name}
        self.is_alive = True
        self.autopilot_calls = []
        self.physics_calls = []
        self.destroyed = False

    def set_autopilot(self, enabled, port):
        self.autopilot_calls.append((enabled, port))

    def set_simulate_physics(self, enabled):
        self.physics_calls.append(enabled)

    def destroy(self):
        self.destroyed = True
        self.is_alive = False


class RuntimeProvider:
    actors = {}
    spawn_calls = []

    @classmethod
    def reset(cls):
        cls.actors = {}
        cls.spawn_calls = []

    @staticmethod
    def get_map():
        return RuntimeMap()

    @classmethod
    def request_new_actor(
        cls, blueprint, transform, *, rolename, autopilot, color, tick
    ):
        actor = RuntimeActor(len(cls.actors) + 10, blueprint, rolename)
        cls.actors[actor.id] = actor
        cls.spawn_calls.append(
            {
                "blueprint": blueprint,
                "transform": transform,
                "rolename": rolename,
                "autopilot": autopilot,
                "color": color,
                "tick": tick,
            }
        )
        return actor

    @classmethod
    def actor_id_exists(cls, actor_id):
        return actor_id in cls.actors

    @classmethod
    def remove_actor_by_id(cls, actor_id):
        actor = cls.actors.pop(actor_id)
        actor.destroy()


class RuntimeTrafficManager:
    def __init__(self):
        self.lane_change = []
        self.desired_speed = []

    def auto_lane_change(self, actor, enabled):
        self.lane_change.append((actor.id, enabled))

    def set_desired_speed(self, actor, speed_kmh):
        self.desired_speed.append((actor.id, speed_kmh))


class RuntimeScenario:
    def __init__(self):
        self.occupied_parking_locations = []


def runtime_context(actor_setup=None):
    route = {
        "route_entry_id": 0,
        "pcu_id": "A-001",
        "arm_id": "left",
        "contract": {"eventually": {"enter_gate": {"id": "left"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "right"}}},
        "deadline_seconds": 5.0,
        "oracle_path": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "actor_setup": actor_setup or {"actors": []},
        "target": {"type": "gate", "id": "left"},
    }
    return {
        "route": route,
        "requirements": {
            "predicates": ["enter_gate"],
            "actor_ids": [],
            "gate_ids": ["left", "right"],
            "dwell_ids": [],
            "occupancy_slots": [],
        },
        "capture_targets": [],
        "actor_spawn_targets": {
            "B102": {
                "type": "parking_slot",
                "slot_number": "B102",
                "polygon": [[20, 20], [24, 20], [24, 24], [20, 24], [20, 20]],
                "heading_deg": 90.0,
            }
        },
        "runtime_required": True,
    }


class EvaluatorRuntimeTest(unittest.TestCase):
    def setUp(self):
        RuntimeProvider.reset()

    def make_runtime(self, context, traffic_manager=None):
        return Language2DriveEvaluatorRuntime(
            context,
            str(L2D_V1_ROOT),
            provider=RuntimeProvider,
            carla_module=RuntimeCarla,
            traffic_manager=traffic_manager,
            traffic_manager_port=8111 if traffic_manager is not None else None,
            stability_seconds=2.0,
            progress_minimum=0.05,
        )

    def test_deterministic_moving_and_slot_actor_setup_and_cleanup(self):
        context = runtime_context(
            {
                "actors": [
                    {
                        "id": "lead",
                        "blueprint": "vehicle.lead",
                        "color": "1,2,3",
                        "program": "constant_speed",
                        "relative_spawn_m": 25,
                    },
                    {
                        "id": "occupant:B102",
                        "blueprint": "vehicle.parked",
                        "slot": "B102",
                    },
                ]
            }
        )
        manager = RuntimeTrafficManager()
        scenario = RuntimeScenario()
        runtime = self.make_runtime(context, manager)
        runtime.setup(object(), scenario)

        self.assertEqual(["l2d:lead", "l2d:occupant:B102"], [
            call["rolename"] for call in RuntimeProvider.spawn_calls
        ])
        self.assertEqual(25.0, RuntimeProvider.spawn_calls[0]["transform"].location.x)
        self.assertEqual(22.0, RuntimeProvider.spawn_calls[1]["transform"].location.x)
        self.assertEqual(22.0, RuntimeProvider.spawn_calls[1]["transform"].location.y)
        self.assertEqual([(10, 18.0)], manager.desired_speed)
        parked_actor = RuntimeProvider.actors[11]
        self.assertEqual([False], parked_actor.physics_calls)
        self.assertEqual(1, len(scenario.occupied_parking_locations))
        self.assertEqual(
            ["lead", "occupant:B102"], runtime.report()["spawned_actor_ids"]
        )

        runtime.cleanup()
        self.assertEqual({}, RuntimeProvider.actors)
        report = runtime.report()
        self.assertTrue(report["cleanup_complete"])
        self.assertEqual(
            ["lead", "occupant:B102"], report["destroyed_actor_ids"]
        )
        events = runtime.drain_final_events()
        self.assertEqual(2, len(events))
        self.assertTrue(
            all(event["source"] == "evaluator_actor_runtime_cleanup" for event in events)
        )
        self.assertEqual([], runtime.drain_final_events())
        runtime.cleanup()  # idempotent

    def test_contract_monitor_owns_completion_and_deadline(self):
        runtime = self.make_runtime(runtime_context())
        runtime.setup(object(), RuntimeScenario())
        for second in (0.0, 1.0, 2.0):
            runtime.observe_frame(
                {
                    "frame": int(second * 20),
                    "sim_time": second,
                    "ego": {"location": {"x": 10.0, "y": 0.0}},
                    "events": ([{"type": "gate", "id": "left"}] if second == 0 else []),
                }
            )
        self.assertTrue(runtime.should_terminate)
        self.assertEqual("contract_complete", runtime.termination_reason)

        deadline_runtime = self.make_runtime(runtime_context())
        deadline_runtime.setup(object(), RuntimeScenario())
        for second in (0.0, 5.0):
            deadline_runtime.observe_frame(
                {
                    "frame": int(second * 20),
                    "sim_time": second,
                    "ego": {"location": {"x": 10.0, "y": 0.0}},
                    "events": [],
                }
            )
        self.assertEqual("deadline", deadline_runtime.termination_reason)

    def test_irrecoverable_below_route_state_terminates_with_failed_evidence(self):
        runtime = self.make_runtime(runtime_context())
        runtime.setup(object(), RuntimeScenario())
        runtime.observe_frame(
            {
                "frame": 20,
                "sim_time": 1.0,
                "ego": {
                    "location": {"x": 20.0, "y": 0.0, "z": -51.0},
                },
                "events": [],
            }
        )

        self.assertTrue(runtime.should_terminate)
        self.assertEqual("ego_out_of_world", runtime.termination_reason)
        report = runtime.report()
        self.assertEqual(
            50.0,
            report["terminal_guards"]["ego_below_route_margin_m"],
        )
        self.assertFalse(report["latest_monitor"]["safe_legal"])
        self.assertEqual(
            51.0,
            report["latest_monitor"]["ego_out_of_world"]["below_route_m"],
        )

    def test_normal_route_elevation_does_not_trigger_below_map_guard(self):
        runtime = self.make_runtime(runtime_context())
        runtime.setup(object(), RuntimeScenario())
        runtime.observe_frame(
            {
                "frame": 20,
                "sim_time": 1.0,
                "ego": {
                    "location": {"x": 20.0, "y": 0.0, "z": -49.9},
                },
                "events": [],
            }
        )

        self.assertFalse(runtime.should_terminate)
        self.assertIsNone(runtime.report()["latest_monitor"]["ego_out_of_world"])

    def test_safety_failure_does_not_hide_later_semantic_completion(self):
        runtime = self.make_runtime(runtime_context())
        runtime.setup(object(), RuntimeScenario())
        runtime.observe_frame(
            {
                "frame": 0,
                "sim_time": 0.0,
                "ego": {"location": {"x": 1.0, "y": 0.0}},
                "events": [{"type": "off_road", "reason": "wrong_direction"}],
            }
        )
        self.assertFalse(runtime.should_terminate)
        self.assertFalse(runtime.report()["latest_monitor"]["safe_legal"])

        runtime.observe_frame(
            {
                "frame": 20,
                "sim_time": 1.0,
                "ego": {"location": {"x": 10.0, "y": 0.0}},
                "events": [{"type": "gate", "id": "left"}],
            }
        )
        self.assertTrue(runtime.should_terminate)
        self.assertEqual(
            "semantic_contract_complete", runtime.termination_reason
        )
        self.assertTrue(runtime.report()["latest_monitor"]["semantic_success"])

    def test_actor_program_failure_is_fail_closed(self):
        context = runtime_context(
            {
                "actors": [
                    {
                        "id": "bad",
                        "blueprint": "vehicle.bad",
                        "program": "unversioned_program",
                    }
                ]
            }
        )
        runtime = self.make_runtime(context, RuntimeTrafficManager())
        with self.assertRaisesRegex(Exception, "unsupported program"):
            runtime.setup(object(), RuntimeScenario())
        self.assertIn("actor_setup_failed", runtime.failure_reason)
        self.assertEqual({}, RuntimeProvider.actors)

    def test_capture_failure_terminates_runtime_fail_closed(self):
        runtime = self.make_runtime(runtime_context())
        runtime.setup(object(), RuntimeScenario())
        runtime.record_capture_failure("log_frame", ValueError("bad trace"))
        self.assertTrue(runtime.should_terminate)
        self.assertIn("capture_failed:log_frame:ValueError", runtime.failure_reason)

    def test_evaluator_abort_is_explicit_and_does_not_override_terminal_verdict(self):
        runtime = self.make_runtime(runtime_context())
        runtime.setup(object(), RuntimeScenario())
        runtime.record_evaluator_failure(
            "agent_runtime", RuntimeError("ego diverged")
        )
        self.assertTrue(runtime.should_terminate)
        self.assertIn(
            "evaluator_failed:agent_runtime:RuntimeError:ego diverged",
            runtime.failure_reason,
        )

        completed = self.make_runtime(runtime_context())
        completed.setup(object(), RuntimeScenario())
        completed.termination_reason = "contract_complete"
        completed.record_evaluator_failure(
            "scenario_stopped_without_private_termination"
        )
        self.assertEqual("contract_complete", completed.termination_reason)
        self.assertIsNone(completed.failure_reason)


if __name__ == "__main__":
    unittest.main()
