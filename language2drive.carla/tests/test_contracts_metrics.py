import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from language2drive.contracts import (
    ContractEngine,
    Verdict,
    score_rollout,
    semantic_instruction_verdict,
)
from language2drive.generator import LANE_TERMINAL_COMMITMENT_PROGRESS
from language2drive.io import atomic_json, atomic_jsonl, read_jsonl, stable_hash
from language2drive.metrics import aggregate
from language2drive.schema import ArmIdentity, SchemaError, validate_contract, validate_pcu


def trace_row(timestamp, *, events=None, speed=1.0, progress=0.0, **extra):
    value = {
        "sim_time": timestamp,
        "events": [] if events is None else events,
        "ego": {"speed_mps": speed},
        "progress": progress,
    }
    value.update(extra)
    return value


def sample_pcu():
    left = {"eventually": {"enter_gate": {"id": "left"}}}
    right = {"eventually": {"enter_gate": {"id": "right"}}}
    return {
        "id": "A-001",
        "layer": "A",
        "split": "dev",
        "group_id": "topology-1",
        "town": "Town01",
        "coarse_envelope": [{"x": 0, "y": 0}, {"x": 20, "y": 0}],
        "arms": [
            {
                "id": "left", "instructions": ["Turn left.", "Take the left turn.", "Go to the left."],
                "contract": left, "opposite_contract": right, "outcome_key": "left",
                "oracle_path": [{"x": 0, "y": 0, "z": 0, "yaw_deg": 0},
                                {"x": 20, "y": 1, "z": 0, "yaw_deg": 3}],
            },
            {
                "id": "right", "instructions": ["Turn right.", "Take the right turn.", "Go to the right."],
                "contract": right, "opposite_contract": left, "outcome_key": "right",
                "oracle_path": [{"x": 0, "y": 0, "z": 0, "yaw_deg": 0},
                                {"x": 20, "y": -1, "z": 0, "yaw_deg": -3}],
            },
        ],
    }


def signal_free_proof(arm, *, lane_id="Town01:1:0:-1"):
    # Speed-proof fixtures use the same 60 m frozen geometry as release arms.
    arm["oracle_path"] = [
        {"x": float(index), "y": 0.0,
         "z": 0.0, "yaw_deg": 0.0}
        for index in range(61)
    ]
    return {
        "method": "exact_signal_stop_free_lane_slice_v2",
        "lane_id": lane_id,
        "path_start_index_inclusive": 20,
        "path_end_index_inclusive": 80,
        "buffer_start_index_inclusive": 0,
        "buffer_end_index_inclusive": 100,
        "road_s_min": 0.0,
        "road_s_max": 65.0,
        "corridor_length_m": round(sum(
            math.hypot(float(b["x"]) - float(a["x"]),
                       float(b["y"]) - float(a["y"]))
            for a, b in zip(arm["oracle_path"], arm["oracle_path"][1:])
        ), 6),
        "before_buffer_length_m": 20.0,
        "after_buffer_length_m": 20.0,
        "sample_spacing_max_m": 0.1,
        "samples_checked": 100,
        "junction_sample_count": 0,
        "road_signal_count": 0,
        "road_signal_reference_count": 0,
        "road_stop_control_count": 0,
        "map_road_signal_count": 1,
        "other_road_signal_clearance_m": 50.0,
        "execution_path_sha256": stable_hash(arm["oracle_path"]),
        "safety_buffer_path_sha256": "b" * 64,
        "lane_graph_sha256": "c" * 64,
        "xodr_sha256": "d" * 64,
        "audit_sha256": "e" * 64,
    }


def valid_speed_pcu():
    value = sample_pcu()
    low = {"eventually": {"speed_band": {"min_mps": 1, "max_mps": 2}}}
    high = {"eventually": {"speed_band": {"min_mps": 4, "max_mps": 5}}}
    value["arms"][0]["contract"], value["arms"][0]["opposite_contract"] = low, high
    value["arms"][1]["contract"], value["arms"][1]["opposite_contract"] = high, low
    for arm in value["arms"]:
        arm["legal_speed_mps"] = 10.0
        arm["legal_speed_proof"] = {
            "method": "oracle_path_exact_waypoint_min_v1", "samples": 601,
            "lane_types": ["Driving", "Parking"], "sample_spacing_max_m": 0.1,
            "minimum_positive_mps": 10.0, "town": "Town01",
            "xodr_sha256": "a" * 64,
        }
        arm["speed_feasibility_cap_proof"] = {
            "method": "carla_default_limit_conservative_cap_v1",
            "cap_mps": 8.333333, "control_margin_mps": 0.35,
        }
        arm["signal_free_corridor_proof"] = signal_free_proof(arm)
    return value


class TemporalContractTests(unittest.TestCase):
    def test_shared_predicate_cache_matches_every_growing_prefix(self):
        rows = [
            trace_row(0.0, speed=8.0, progress=0.0, scopes=[]),
            trace_row(
                1.0, events=[{"type": "gate", "id": "first"}],
                speed=3.0, progress=0.1, scopes=["corridor"],
            ),
            trace_row(2.0, speed=9.0, progress=0.2, scopes=[]),
            trace_row(
                3.0, events=[{"type": "gate", "id": "second"}],
                speed=3.0, progress=0.8, scopes=["corridor"],
            ),
            trace_row(4.0, events=["collision"], speed=3.0, progress=1.0),
        ]
        expressions = [
            {"sequence": [
                {"enter_gate": {"id": "first"}},
                {"enter_gate": {"id": "second"}},
            ]},
            {"always": {"speed_band": {
                "min_mps": 2.0, "max_mps": 4.0, "scope": "corridor",
            }}},
            {"safe_legal": {}},
            {"eventually": {"progress": {"min": 0.5}}},
        ]
        cache = {}
        for length in range(1, len(rows) + 1):
            cached = ContractEngine(
                rows[:length], predicate_cache=cache,
            )
            plain = ContractEngine(rows[:length])
            self.assertEqual(
                [cached.evaluate(expression) for expression in expressions],
                [plain.evaluate(expression) for expression in expressions],
            )

    def test_within_uses_trace_time_and_separates_immediate_from_delayed_progress(self):
        immediate = {"within": {
            "condition": {"progress": {"min": 0.15}},
            "seconds": 8,
        }}
        delayed = {"sequence": [
            {"dwell": {
                "condition": {"speed_band": {"min_mps": 0.0, "max_mps": 0.2}},
                "seconds": 10,
            }},
            {"progress": {"min": 0.15}},
        ]}
        start_trace = [
            trace_row(20.0, speed=0.0, progress=0.0),
            trace_row(27.9, speed=4.0, progress=0.15),
            trace_row(31.0, speed=4.0, progress=0.3),
        ]
        wait_trace = [
            trace_row(20.0, speed=0.0, progress=0.0),
            trace_row(30.0, speed=0.0, progress=0.0),
            trace_row(31.0, speed=2.0, progress=0.15),
        ]
        self.assertTrue(ContractEngine(start_trace).evaluate(immediate).passed)
        self.assertFalse(ContractEngine(start_trace).evaluate(delayed).passed)
        self.assertTrue(ContractEngine(wait_trace).evaluate(delayed).passed)
        self.assertFalse(ContractEngine(wait_trace).evaluate(immediate).passed)
        validate_contract(immediate)
        with self.assertRaisesRegex(SchemaError, "condition and seconds"):
            validate_contract({"within": {"seconds": 8}})

    def test_dwell_can_be_a_strict_sequence_stage(self):
        trace = [
            {"sim_time": 0.0, "ego": {"speed_mps": 0.0}, "progress": 0.0},
            {"sim_time": 1.0, "ego": {"speed_mps": 0.0}, "progress": 0.0},
            {"sim_time": 2.0, "ego": {"speed_mps": 0.0}, "progress": 0.0},
            {"sim_time": 3.0, "ego": {"speed_mps": 2.0}, "progress": 0.2},
            {"sim_time": 4.0, "ego": {"speed_mps": 2.0}, "progress": 0.4},
        ]
        expression = {"sequence": [
            {"dwell": {"condition": {"speed_band": {"min_mps": 0.0, "max_mps": 0.2}}, "seconds": 2.0}},
            {"progress": {"min": 0.15}},
        ]}
        self.assertTrue(ContractEngine(trace).evaluate(expression).passed)

    def test_empty_and_malformed_trace_fail_closed(self):
        self.assertFalse(ContractEngine([]).evaluate({"always": {"safe_legal": {}}}).passed)
        with self.assertRaisesRegex(ValueError, "sim_time"):
            ContractEngine([{"events": []}])
        with self.assertRaisesRegex(ValueError, "monotonic"):
            ContractEngine([trace_row(2), trace_row(1)])

    def test_always_and_bare_safety_cover_the_complete_rollout(self):
        trace = [trace_row(0), trace_row(1, events=["left"]), trace_row(2, events=["collision"])]
        engine = ContractEngine(trace)
        self.assertFalse(engine.evaluate({"always": {"safe_legal": {}}}).passed)
        self.assertFalse(engine.evaluate({"safe_legal": {}}).passed)
        self.assertEqual(engine._mask({"always": {"safe_legal": {}}}), [True, True, False])

        clean = ContractEngine(trace[:-1]).evaluate({"always": {"safe_legal": {}}})
        self.assertTrue(clean.passed)
        self.assertEqual(clean.first_satisfied_at, 1.0)

    def test_strong_until_checks_every_sample_before_trigger(self):
        expression = {
            "until": {
                "invariant": {"speed_band": {"min_mps": 0, "max_mps": 5}},
                "condition": {"enter_gate": {"id": "exit"}},
            }
        }
        good = [trace_row(0, speed=3), trace_row(1, speed=4), trace_row(2, speed=8, events=["exit"])]
        verdict = ContractEngine(good).evaluate(expression)
        self.assertTrue(verdict.passed)  # invariant is required strictly before the terminal event
        self.assertEqual(verdict.first_satisfied_at, 2.0)

        violated = [trace_row(0, speed=3), trace_row(1, speed=8), trace_row(2, speed=3, events=["exit"])]
        self.assertFalse(ContractEngine(violated).evaluate(expression).passed)
        never_triggered = [trace_row(0, speed=3), trace_row(1, speed=3)]
        self.assertFalse(ContractEngine(never_triggered).evaluate(expression).passed)

    def test_sequence_activates_nested_until_only_after_prior_stage(self):
        expression = {"sequence": [
            {"dwell": {
                "condition": {"speed_band": {"min_mps": 3, "max_mps": 5}},
                "seconds": 2,
            }},
            {"until": {
                "invariant": {"speed_band": {"min_mps": 3, "max_mps": 5}},
                "condition": {"enter_gate": {"id": "end"}},
            }},
        ]}
        # Initial acceleration is outside the band.  It precedes activation of
        # the until stage and therefore must not poison the invariant.
        good = [
            trace_row(0, speed=0), trace_row(1, speed=2),
            trace_row(2, speed=3), trace_row(4, speed=4),
            trace_row(5, speed=4), trace_row(6, speed=4, events=["end"]),
        ]
        self.assertTrue(ContractEngine(good).evaluate(expression).passed)
        violated_after_activation = good[:-2] + [
            trace_row(5, speed=7), trace_row(6, speed=4, events=["end"]),
        ]
        self.assertFalse(ContractEngine(violated_after_activation).evaluate(expression).passed)

    def test_progress_commitment_delays_lane_counterfactual_until_shared_prefix_ends(self):
        left = {"sequence": [
            {"progress": {"min": LANE_TERMINAL_COMMITMENT_PROGRESS}},
            {"dwell": {
                "condition": {"lane_relation": {"value": "left"}},
                "seconds": 2,
            }},
        ]}
        current = {"sequence": [
            {"progress": {"min": LANE_TERMINAL_COMMITMENT_PROGRESS}},
            {"dwell": {
                "condition": {"lane_relation": {"value": "current"}},
                "seconds": 2,
            }},
        ]}
        # The correct change is intentionally slow: it remains in the source
        # lane for more than two seconds after progress .5, then finishes the
        # connector before the terminal .9 commitment window begins.
        changed = [
            trace_row(0, progress=0, ego={"lane_relation": "current"}),
            trace_row(3, progress=.49, ego={"lane_relation": "current"}),
            trace_row(5, progress=.6, ego={"lane_relation": "current"}),
            trace_row(7.5, progress=.8, ego={"lane_relation": "left"}),
            trace_row(8, progress=.9, ego={"lane_relation": "left"}),
            trace_row(8.05, progress=.91, ego={"lane_relation": "left"}),
            trace_row(10.05, progress=.99, ego={"lane_relation": "left"}),
        ]
        held = [
            trace_row(0, progress=0, ego={"lane_relation": "current"}),
            trace_row(4, progress=.5, ego={"lane_relation": "current"}),
            trace_row(8, progress=.9, ego={"lane_relation": "current"}),
            trace_row(8.05, progress=.91, ego={"lane_relation": "current"}),
            trace_row(10.05, progress=.99, ego={"lane_relation": "current"}),
        ]
        self.assertTrue(ContractEngine(changed).evaluate(left).passed)
        self.assertFalse(ContractEngine(changed).evaluate(current).passed)
        self.assertTrue(ContractEngine(held).evaluate(current).passed)
        self.assertFalse(ContractEngine(held).evaluate(left).passed)

    def test_sequence_is_strict_about_early_duplicate_and_simultaneous_events(self):
        expression = {"sequence": [{"enter_gate": {"id": "pickup"}}, {"enter_gate": {"id": "dropoff"}}]}
        good = [trace_row(0), trace_row(1, events=["pickup"]), trace_row(2), trace_row(3, events=["dropoff"])]
        self.assertTrue(ContractEngine(good).evaluate(expression).passed)

        early = [trace_row(0, events=["dropoff"]), trace_row(1, events=["pickup"]), trace_row(2, events=["dropoff"])]
        self.assertFalse(ContractEngine(early).evaluate(expression).passed)
        duplicate = [trace_row(0), trace_row(1, events=["pickup"]), trace_row(2, events=["pickup"]),
                     trace_row(3, events=["dropoff"])]
        self.assertFalse(ContractEngine(duplicate).evaluate(expression).passed)
        simultaneous = [trace_row(0), trace_row(1, events=["pickup", "dropoff"])]
        self.assertFalse(ContractEngine(simultaneous).evaluate(expression).passed)

    def test_if_else_activates_exactly_one_branch(self):
        expression = {
            "if_else": {
                "condition": {"enter_gate": {"id": "closed"}},
                "then": {"enter_gate": {"id": "second_entrance"}},
                "else": {"enter_gate": {"id": "first_entrance"}},
            }
        }
        closed = [trace_row(0, events=["closed"]), trace_row(1, events=["second_entrance"])]
        verdict = ContractEngine(closed).evaluate(expression)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.details["selected_branch"], "then")
        open_trace = [trace_row(0), trace_row(1, events=["first_entrance"])]
        self.assertTrue(ContractEngine(open_trace).evaluate(expression).passed)
        wrong = [trace_row(0, events=["closed"]), trace_row(1, events=["first_entrance"])]
        self.assertFalse(ContractEngine(wrong).evaluate(expression).passed)

        occupied = {
            "if_else": {
                "condition": {"actor_order": {"id": "occupied:A811"}},
                "then": {"enter_gate": {"id": "alternate"}},
                "else": {"enter_gate": {"id": "preferred"}},
            }
        }
        occupancy_trace = [trace_row(0, slot_occupancy={"A811": True}),
                           trace_row(1, events=["alternate"])]
        self.assertTrue(ContractEngine(occupancy_trace).evaluate(occupied).passed)
        no_retroactive_switch = [trace_row(0), trace_row(1, events=["first_entrance"]),
                                 trace_row(2, events=["closed"])]
        self.assertTrue(ContractEngine(no_retroactive_switch).evaluate(expression).passed)

    def test_reference_binding_freezes_actor_identity(self):
        expression = {
            "bind_reference": {
                "reference": "lead_vehicle",
                "body": {"actor_order": {"id": "$lead_vehicle"}},
            }
        }
        good = [trace_row(0, references={"lead_vehicle": "actor-7"}),
                trace_row(1, events=[{"type": "actor_order", "actor_id": "actor-7"}])]
        self.assertTrue(ContractEngine(good).evaluate(expression).passed)
        wrong_actor = [trace_row(0, references={"lead_vehicle": "actor-7"}),
                       trace_row(1, events=[{"type": "actor_order", "actor_id": "actor-8"}])]
        self.assertFalse(ContractEngine(wrong_actor).evaluate(expression).passed)

        generated_memory_shape = {
            "bind_reference": {
                "reference": "remembered_place",
                "value": "port:v5",
                "body": {"sequence": [
                    {"enter_gate": {"id": "port:v6"}},
                    {"enter_gate": {"id": {"$ref": "remembered_place"}}},
                ]},
            }
        }
        memory_trace = [trace_row(0), trace_row(1, events=["port:v6"]), trace_row(2, events=["port:v5"])]
        self.assertTrue(ContractEngine(memory_trace).evaluate(generated_memory_shape).passed)

    def test_dwell_uses_real_sim_time_and_resets_on_interruption(self):
        expression = {
            "dwell": {
                "condition": {"speed_band": {"min_mps": 0, "max_mps": 0.2}},
                "min_seconds": 3.0,
            }
        }
        trace = [trace_row(10.0, speed=0.1), trace_row(11.7, speed=0.1), trace_row(13.0, speed=0.1)]
        verdict = ContractEngine(trace).evaluate(expression)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.first_satisfied_at, 13.0)

        interrupted = [trace_row(10.0, speed=0.1), trace_row(12.9, speed=0.3),
                       trace_row(13.0, speed=0.1), trace_row(15.9, speed=0.1)]
        self.assertFalse(ContractEngine(interrupted).evaluate(expression).passed)

        keyed = {"dwell": {"key": "slot-A101", "seconds": 3.0, "max_speed_mps": 0.2}}
        keyed_trace = [trace_row(0, speed=0.1, dwell={"slot-A101": True}),
                       trace_row(2, speed=0.3, dwell={"slot-A101": True}),
                       trace_row(4, speed=0.1, dwell={"slot-A101": True})]
        self.assertFalse(ContractEngine(keyed_trace).evaluate(keyed).passed)

    def test_stable_speed_and_actor_specific_headway_are_not_instantaneous_or_global(self):
        stable_speed = {"speed_band": {"min_mps": 2, "max_mps": 4, "stable_seconds": 2}}
        too_short = [trace_row(0, speed=3), trace_row(1.9, speed=3)]
        self.assertFalse(ContractEngine(too_short).evaluate(stable_speed).passed)
        long_enough = too_short + [trace_row(2.0, speed=3)]
        self.assertTrue(ContractEngine(long_enough).evaluate(stable_speed).passed)

        headway = {"headway_band": {"actor_id": "lead-17", "min_seconds": 1.5, "max_seconds": 2.5}}
        wrong_actor_only = [trace_row(0, headway_seconds=2.0, headways={"other": 2.0})]
        self.assertFalse(ContractEngine(wrong_actor_only).evaluate(headway).passed)
        correct_actor = [trace_row(0, headways={"lead-17": 2.0})]
        self.assertTrue(ContractEngine(correct_actor).evaluate(headway).passed)

        # The connected logger currently calls this field ``speed``; accept it
        # as a normalized alias without silently treating a moving car as 0.
        logger_shape = [{"sim_time": 0, "events": [], "ego": {"speed": 3.0}}]
        self.assertTrue(ContractEngine(logger_shape).evaluate(
            {"speed_band": {"min_mps": 2, "max_mps": 4}}
        ).passed)

        scoped = {"always": {"speed_band": {"min_mps": 2, "max_mps": 4, "scope": "corridor"}}}
        scoped_trace = [trace_row(0, speed=8, scopes=[]), trace_row(1, speed=3, scopes=["corridor"]),
                        trace_row(2, speed=9, scopes=[])]
        self.assertTrue(ContractEngine(scoped_trace).evaluate(scoped).passed)
        self.assertFalse(ContractEngine([trace_row(0, speed=3)]).evaluate(scoped).passed)

    def test_parking_checks_whole_footprint_speed_heading_and_tolerance(self):
        polygon = [[-2, -1], [2, -1], [2, 1], [-2, 1]]
        expression = {
            "vehicle_footprint_inside_slot": {
                "polygon": polygon,
                "tolerance_m": 0.2,
                "heading_deg": 1,
                "heading_tolerance_deg": 15,
                "max_speed_mps": 0.2,
            }
        }
        ego = {"speed_mps": 0.19, "heading_deg": 359,
               "footprint": [[-1.9, -0.9], [1.9, -0.9], [1.9, 1.15], [-1.9, 1.15]]}
        self.assertTrue(ContractEngine([trace_row(0, ego=ego)]).evaluate(expression).passed)

        too_fast = {**ego, "speed_mps": 0.2}
        self.assertFalse(ContractEngine([trace_row(0, ego=too_fast)]).evaluate(expression).passed)
        wrong_heading = {**ego, "heading_deg": 30}
        self.assertFalse(ContractEngine([trace_row(0, ego=wrong_heading)]).evaluate(expression).passed)
        outside = {**ego, "footprint": [[-1.9, -0.9], [1.9, -0.9], [1.9, 1.21], [-1.9, 1.21]]}
        self.assertFalse(ContractEngine([trace_row(0, ego=outside)]).evaluate(expression).passed)

    def test_deadline_is_based_on_completion_not_logger_shutdown(self):
        own = Verdict(True, 4.0)
        progress = Verdict(True, 5.0)
        stable = Verdict(True, 6.0)
        result = score_rollout(own, Verdict(False), Verdict(True), progress, 20.0, 6.0, stable)
        self.assertTrue(result["success"])
        self.assertTrue(result["semantic_success"])
        self.assertEqual(result["completion_time"], 6.0)
        self.assertFalse(score_rollout(own, Verdict(False), Verdict(True), progress, 20.0, 5.9, stable)["success"])
        self.assertFalse(score_rollout(own, Verdict(True), Verdict(True), progress, 20.0, 6.0, stable)["success"])
        unsafe = score_rollout(
            own, Verdict(False), Verdict(False), progress, 20.0, 6.0, stable
        )
        self.assertFalse(unsafe["success"])
        self.assertTrue(unsafe["semantic_success"])
        legacy_unknown = score_rollout(Verdict(True), Verdict(False), Verdict(True), progress, 20.0, 10.0, stable)
        self.assertFalse(legacy_unknown["deadline"])  # unknown own completion falls back to rollout end

        semantic_only = score_rollout(
            Verdict(False), Verdict(False), Verdict(False), Verdict(False),
            20.0, 10.0, Verdict(False), semantic=Verdict(True, 4.0),
        )
        self.assertFalse(semantic_only["success"])
        self.assertTrue(semantic_only["semantic_success"])
        self.assertEqual(semantic_only["semantic_completion_time"], 4.0)

    def test_loose_junction_semantics_accept_heading_intent_without_gate(self):
        rows = [
            trace_row(
                0.0, progress=0.0,
                ego={"location": {"x": 0.0, "y": 0.0, "z": 0.0},
                     "rotation": {"yaw": 0.0}, "speed_mps": 0.0},
            ),
            trace_row(
                4.0, progress=0.3,
                ego={"location": {"x": 8.0, "y": 4.0, "z": 0.0},
                     "rotation": {"yaw": 52.0}, "speed_mps": 4.0},
            ),
        ]
        target = {"type": "gate", "id": "right", "maneuver": "right"}
        path = [
            {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
            {"x": 10.0, "y": 10.0, "yaw_deg": 90.0},
        ]
        verdict = semantic_instruction_verdict(
            rows, target=target, oracle_path=path, own=Verdict(False), layer="A",
        )
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.details["method"], "loose_endpoint_heading")

        wrong_path = [path[0], {"x": 10.0, "y": -10.0, "yaw_deg": -90.0}]
        self.assertFalse(semantic_instruction_verdict(
            rows, target={**target, "maneuver": "left"},
            oracle_path=wrong_path, own=Verdict(False), layer="A",
        ).passed)

    def test_loose_semantic_proxies_cover_b_c_and_d_layers(self):
        b_rows = [
            trace_row(
                0.0, speed=4.5, progress=0.0,
                ego={"location": {"x": 0.0, "y": 0.0, "z": 0.0},
                     "rotation": {"yaw": 0.0}, "speed_mps": 4.5},
            ),
            trace_row(
                4.0, speed=4.5, progress=0.7,
                ego={"location": {"x": 20.0, "y": 0.0, "z": 0.0},
                     "rotation": {"yaw": 0.0}, "speed_mps": 4.5},
            ),
        ]
        b_target = {
            "type": "until",
            "constraint": {"min_mps": 2.0, "max_mps": 4.0},
            "end": {"type": "gate", "center": {"x": 20.0, "y": 0.0},
                    "radius_m": 1.0},
        }
        self.assertTrue(semantic_instruction_verdict(
            b_rows, target=b_target,
            oracle_path=[{"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                         {"x": 20.0, "y": 0.0, "yaw_deg": 0.0}],
            own=Verdict(False), layer="B",
        ).passed)

        c_rows = [trace_row(
            2.0,
            ego={"location": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "rotation": {"yaw": 35.0}, "speed_mps": 3.0},
        )]
        c_target = {
            "type": "parking_slot",
            "polygon": [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
            "heading_deg": 0.0,
        }
        self.assertTrue(semantic_instruction_verdict(
            c_rows, target=c_target, oracle_path=[],
            own=Verdict(False), layer="C",
        ).passed)

        d_rows = [
            trace_row(1.0, events=[{"type": "enter_gate", "id": "parking-stage:B208"}]),
            trace_row(2.0, events=[{"type": "enter_gate", "id": "parking-stage:F206"}]),
        ]
        self.assertTrue(semantic_instruction_verdict(
            d_rows,
            target={"type": "sequence", "stages": ["B208", "F206"]},
            oracle_path=[], own=Verdict(False), layer="D",
        ).passed)
        self.assertFalse(semantic_instruction_verdict(
            d_rows,
            target={"type": "sequence", "stages": ["F206", "B208"]},
            oracle_path=[], own=Verdict(False), layer="D",
        ).passed)


class SchemaContractTests(unittest.TestCase):
    def test_full_pcu_contract_is_valid(self):
        validate_pcu(sample_pcu())

    def test_schema_rejects_substring_layer_duplicate_language_and_non_mirrored_pair(self):
        value = sample_pcu()
        value["layer"] = "AB"
        with self.assertRaises(SchemaError):
            validate_pcu(value)

        value = sample_pcu()
        value["arms"][0]["instructions"][2] = value["arms"][0]["instructions"][0]
        with self.assertRaisesRegex(SchemaError, "distinct"):
            validate_pcu(value)

        value = sample_pcu()
        value["arms"][1]["opposite_contract"] = {"eventually": {"enter_gate": {"id": "other"}}}
        with self.assertRaisesRegex(SchemaError, "counterfactual opposites"):
            validate_pcu(value)

    def test_schema_checks_numeric_and_polygon_arguments(self):
        with self.assertRaises(SchemaError):
            validate_contract({"speed_band": {"min_mps": 5, "max_mps": 2}})
        with self.assertRaises(SchemaError):
            validate_contract({"inside_region": {"polygon": [[0, 0], [1, 1], [0, 0]]}})
        with self.assertRaises(SchemaError):
            validate_contract({"dwell": {"key": "parked", "condition": {"safe_legal": {}}}})
        with self.assertRaisesRegex(SchemaError, "unknown arguments"):
            validate_contract({"speed_band": {"min_mps": 1, "max_mps": 2, "max_mpss": 3}})

    def test_speed_program_requires_versioned_path_wide_legal_proof(self):
        value = sample_pcu()
        low = {"eventually": {"speed_band": {"min_mps": 1, "max_mps": 2}}}
        high = {"eventually": {"speed_band": {"min_mps": 4, "max_mps": 5}}}
        value["arms"][0]["contract"], value["arms"][0]["opposite_contract"] = low, high
        value["arms"][1]["contract"], value["arms"][1]["opposite_contract"] = high, low
        with self.assertRaisesRegex(SchemaError, "legal_speed"):
            validate_pcu(value)
        for arm in value["arms"]:
            arm["legal_speed_mps"] = 10.0
            arm["legal_speed_proof"] = {
                "method": "oracle_path_exact_waypoint_min_v1", "samples": 101,
                "lane_types": ["Driving", "Parking"], "sample_spacing_max_m": 0.1,
                "minimum_positive_mps": 10.0, "town": "Town01",
                "xodr_sha256": "a" * 64,
            }
            arm["speed_feasibility_cap_proof"] = {
                "method": "carla_default_limit_conservative_cap_v1",
                "cap_mps": 8.333333,
                "control_margin_mps": 0.35,
            }
            arm["signal_free_corridor_proof"] = signal_free_proof(arm)
        validate_pcu(value)
        value["arms"][0]["legal_speed_proof"]["lane_types"] = ["Parking", "Driving"]
        with self.assertRaisesRegex(SchemaError, "precedence"):
            validate_pcu(value)

        value = sample_pcu()
        value["arms"][0]["contract"], value["arms"][0]["opposite_contract"] = high, low
        value["arms"][1]["contract"], value["arms"][1]["opposite_contract"] = low, high
        for arm in value["arms"]:
            arm["legal_speed_mps"] = 10.0
            arm["legal_speed_proof"] = {
                "method": "oracle_path_exact_waypoint_min_v1", "samples": 101,
                "lane_types": ["Driving", "Parking"], "sample_spacing_max_m": 0.1,
                "minimum_positive_mps": 10.0, "town": "Town01",
                "xodr_sha256": "a" * 64,
            }
            arm["speed_feasibility_cap_proof"] = {
                "method": "carla_default_limit_conservative_cap_v1",
                "cap_mps": 5.2,
                "control_margin_mps": 0.35,
            }
            arm["signal_free_corridor_proof"] = signal_free_proof(arm)
        with self.assertRaisesRegex(SchemaError, "frozen 8.333333"):
            validate_pcu(value)

        value = sample_pcu()
        too_fast = {"eventually": {"speed_band": {"min_mps": 6, "max_mps": 8.1}}}
        value["arms"][0]["contract"], value["arms"][0]["opposite_contract"] = too_fast, low
        value["arms"][1]["contract"], value["arms"][1]["opposite_contract"] = low, too_fast
        for arm in value["arms"]:
            arm["legal_speed_mps"] = 10.0
            arm["legal_speed_proof"] = {
                "method": "oracle_path_exact_waypoint_min_v1", "samples": 101,
                "lane_types": ["Driving", "Parking"], "sample_spacing_max_m": 0.1,
                "minimum_positive_mps": 10.0, "town": "Town01",
                "xodr_sha256": "a" * 64,
            }
            arm["speed_feasibility_cap_proof"] = {
                "method": "carla_default_limit_conservative_cap_v1",
                "cap_mps": 8.333333,
                "control_margin_mps": 0.35,
            }
            arm["signal_free_corridor_proof"] = signal_free_proof(arm)
        with self.assertRaisesRegex(SchemaError, "effective XODR/CARLA cap"):
            validate_pcu(value)

    def test_signal_free_v2_proof_is_exactly_bound_and_fail_closed(self):
        value = valid_speed_pcu()
        validate_pcu(value)

        mutations = (
            ("path hash", lambda proof, arm: arm["oracle_path"][30].__setitem__("y", 0.2),
             "does not bind oracle_path"),
            ("short buffer", lambda proof, arm: proof.__setitem__("before_buffer_length_m", 19.999),
             "must be >= 20"),
            ("signal", lambda proof, arm: proof.__setitem__("road_signal_count", 1),
             "must equal integer zero"),
            ("signal reference", lambda proof, arm: proof.__setitem__("road_signal_reference_count", 1),
             "must equal integer zero"),
            ("stop", lambda proof, arm: proof.__setitem__("road_stop_control_count", 1),
             "must equal integer zero"),
            ("junction", lambda proof, arm: proof.__setitem__("junction_sample_count", 1),
             "must equal integer zero"),
            ("bad clearance", lambda proof, arm: proof.__setitem__("other_road_signal_clearance_m", 39.9),
             "must be >= 40"),
            ("null with signals", lambda proof, arm: (
                proof.__setitem__("other_road_signal_clearance_m", None),
                proof.__setitem__("map_road_signal_count", 1),
            ), "null clearance requires zero"),
            ("malformed lane", lambda proof, arm: proof.__setitem__("lane_id", "Town01:road"),
             "town mismatch"),
            ("slice mismatch", lambda proof, arm: proof.__setitem__("path_end_index_inclusive", 79),
             "slice size must equal oracle_path"),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                candidate = copy.deepcopy(value)
                arm = candidate["arms"][0]
                mutate(arm["signal_free_corridor_proof"], arm)
                with self.assertRaisesRegex(SchemaError, message):
                    validate_pcu(candidate)


class MetricsTests(unittest.TestCase):
    def _ledger_run(self, pcu_id, arm, paraphrase, seed=1, track="standardized"):
        identity = ArmIdentity(pcu_id, arm, paraphrase, seed, track)
        return {
            **identity.__dict__, "run_id": identity.run_id,
            "route_entry_id": 1, "status": "expected",
        }

    def _bundle(self, track="standardized"):
        digest = "a" * 64
        value = {
            "schema_version": "language2drive.execution_bundle.v2",
            **{key: digest for key in (
                "route_xml_sha256", "agent_sha256", "agent_config_sha256",
                "evaluator_sha256", "dependency_manifest_sha256",
                "execution_environment_sha256", "storage_policy_sha256",
                "wall_timeout_policy_sha256", "carla_runtime_asset_manifest_sha256",
                "carla_runtime_asset_inventory_sha256",
                "carla_runtime_asset_files_sha256",
                "carla_runtime_shipping_binary_sha256",
                "carla_runtime_asset_provenance_sha256",
                "carla_runtime_available_maps_sha256",
                "carla_runtime_server_readonly_permissions_sha256",
                "carla_runtime_asset_validator_sha256",
                "carla_server_process_receipt_sha256", "private_sidecar_sha256",
            )},
            "workload": "core", "benchmark_track": track,
            "evaluator_track": track, "dependency_profile": "test.v1",
            "dependency_hashes": {"test.py": digest},
            "carla_runtime_root": "/frozen/carla",
            "carla_runtime_asset_file_count": 1,
            "carla_runtime_asset_total_size_bytes": 1,
            "carla_runtime_available_map_count": 1,
            "carla_runtime_client_version": "0.9.14",
            "carla_runtime_server_version": "0.9.14",
            "carla_runtime_boot_id": "boot",
            "carla_runtime_freeze_monotonic_ticks": 1,
            "carla_runtime_clock_ticks_per_second": 100,
            "carla_runtime_server_readonly_permission_entry_count": 1,
            "carla_runtime_server_uid": 1000,
            "carla_runtime_server_gids": [1000],
            "carla_server_process_receipt": {"pid": 1},
        }
        value["dependency_manifest_sha256"] = stable_hash(value["dependency_hashes"])
        value["carla_server_process_receipt_sha256"] = stable_hash(
            value["carla_server_process_receipt"]
        )
        return value

    def _write_result(self, results, run, bundle, *, success=True, status="complete", trace=True, **overrides):
        directory = results / run["run_id"]
        directory.mkdir(parents=True)
        terminal = status in {"crash", "timeout", "deadlock"}
        result = {
            "schema_version": "language2drive.contract_result.v1",
            "status": status,
            "success": success,
            "semantic_success": success if not terminal else False,
            "capture_complete": not terminal,
            "own_contract": success if not terminal else False,
            "opposite_contract": False,
            "safe_legal": success if not terminal else False,
            "progress": success if not terminal else False,
            "deadline": success if not terminal else False,
            "stability": success if not terminal else False,
            "scorer": "language2drive.v1.offline_contract_engine",
            **{key: run[key] for key in (
                "run_id", "pcu_id", "arm_id", "paraphrase", "seed", "track",
                "route_entry_id",
            )},
        }
        result.update(overrides)
        atomic_json(directory / "contract_result.json", result)
        atomic_json(directory / "runner_command.json", {"run": run, "input_hashes": bundle})
        if trace:
            atomic_jsonl(directory / "trace.jsonl", [{"sim_time": 0.0}])

    def test_cfps_pairing_and_unit_level_worst_paraphrase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            runs = [self._ledger_run(pcu, arm, para) for pcu in ("A-001", "A-002")
                    for para in range(3) for arm in ("plus", "minus")]
            atomic_json(root / "ledger.json", {"runs": runs})
            bundle = self._bundle()
            for run in runs:
                failed = run["pcu_id"] == "A-001" and run["paraphrase"] == 2 and run["arm_id"] == "minus"
                self._write_result(results, run, bundle, success=not failed)

            value = aggregate(root / "ledger.json", results, expected_input_hashes=bundle)
            self.assertEqual(value["expected_runs"], 12)
            self.assertAlmostEqual(value["CFPS"]["A"], 5 / 6)
            self.assertEqual(value["CFPS_by_paraphrase"], {"0": 1.0, "1": 1.0, "2": 0.5})
            self.assertEqual(value["Worst-paraphrase"], 0.5)
            self.assertAlmostEqual(value["SISR"], 11 / 12)
            self.assertAlmostEqual(value["SemISR"], 11 / 12)

    def test_semisr_accepts_loose_semantics_independently_of_own_and_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            runs = [
                self._ledger_run("A-001", arm, 0)
                for arm in ("plus", "minus")
            ]
            atomic_json(root / "ledger.json", {"runs": runs})
            bundle = self._bundle()
            self._write_result(
                results,
                runs[0],
                bundle,
                success=False,
                own_contract=False,
                semantic_success=True,
                safe_legal=False,
                progress=True,
                deadline=True,
                stability=False,
            )
            self._write_result(results, runs[1], bundle, success=False)

            value = aggregate(
                root / "ledger.json", results, expected_input_hashes=bundle
            )
            self.assertEqual(value["SISR"], 0.0)
            self.assertEqual(value["SemISR"], 0.5)
            self.assertEqual(value["SemISR_by_layer"]["A"], 0.5)
            semantic_row = next(
                row for row in value["rows"] if row["arm_id"] == "plus"
            )
            self.assertTrue(semantic_row["semantic_success"])
            self.assertFalse(semantic_row["safe_legal"])

    def test_missing_crash_timeout_no_trace_and_inconsistent_results_all_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            runs = [self._ledger_run("B-001", arm, para) for para in range(3) for arm in ("plus", "minus")]
            atomic_json(root / "ledger.json", {"runs": runs})
            bundle = self._bundle()
            # run 0 is absent (missing)
            self._write_result(results, runs[1], bundle, success=False, status="crash", trace=False)
            self._write_result(results, runs[2], bundle, success=True, trace=False)
            self._write_result(results, runs[3], bundle, success=False, status="timeout", trace=True)
            self._write_result(results, runs[4], bundle, success=True, trace=True, opposite_contract=True)
            self._write_result(results, runs[5], bundle, success=False, trace=True)

            value = aggregate(root / "ledger.json", results, expected_input_hashes=bundle)
            self.assertEqual(value["expected_runs"], 6)
            self.assertEqual(value["observed_runs"], 5)
            self.assertEqual(value["missing_runs"], 1)
            self.assertEqual(value["crashed_runs"], 1)
            self.assertEqual(value["timeout_runs"], 1)
            self.assertEqual(value["no_trace_runs"], 1)
            self.assertEqual(value["invalid_runs"], 1)
            self.assertEqual(value["CFPS"]["B"], 0.0)
            self.assertEqual(value["SISR"], 0.0)
            self.assertEqual(value["FCS-2x2"], {"value": None, "reason": "fixed_layout_only"})

    def test_ledger_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._ledger_run("A-001", "plus", 0)
            run["run_id"] = "wrong"
            atomic_json(root / "ledger.json", {"runs": [run]})
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                aggregate(root / "ledger.json", root / "results")

    def test_native_track_never_contaminates_standardized_leaderboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            runs = []
            for track in ("standardized", "native"):
                for paraphrase in range(3):
                    for arm in ("plus", "minus"):
                        runs.append(self._ledger_run("A-001", arm, paraphrase, track=track))
            atomic_json(root / "ledger.json", {"runs": runs})
            bundles = {track: self._bundle(track) for track in ("standardized", "native")}
            for run in runs:
                self._write_result(
                    results, run, bundles[run["track"]],
                    success=run["track"] == "native",
                )
            value = aggregate(root / "ledger.json", results, expected_input_hashes=bundles)
            self.assertEqual(value["leaderboard_track"], "standardized")
            self.assertEqual(value["CFPS"]["A"], 0.0)
            self.assertEqual(value["metrics_by_track"]["native"]["CFPS"]["A"], 1.0)

    def test_oracle_witness_ledger_uses_the_same_recoverable_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = ArmIdentity("A-001", "plus", 0, 1, "oracle")
            run = {**identity.__dict__, "run_id": identity.run_id, "status": "expected"}
            atomic_json(root / "ledger.json", {"runs": [run]})
            value = aggregate(root / "ledger.json", root / "results")
            self.assertEqual(value["leaderboard_track"], "oracle")
            self.assertEqual(value["expected_runs"], 1)
            self.assertEqual(value["missing_runs"], 1)


class IoTests(unittest.TestCase):
    def test_canonical_hash_and_jsonl_object_validation(self):
        self.assertEqual(stable_hash({"b": 2, "a": 1}), stable_hash({"a": 1, "b": 2}))
        with self.assertRaises(ValueError):
            stable_hash({"bad": math.nan})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                list(read_jsonl(path))

    def test_atomic_json_is_publishable_and_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            atomic_json(path, {"old": True})
            atomic_json(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"new": True})
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o644)
            private = Path(temporary) / "sidecar.private.json"
            atomic_json(private, {"target_gate": [1, 2]})
            self.assertEqual(os.stat(private).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
