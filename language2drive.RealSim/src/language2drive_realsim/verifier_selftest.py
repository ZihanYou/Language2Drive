"""Executable adversarial verifier suite used as release evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

from .constants import FAMILY_TARGETS
from .verifier import verify_trace


def _contract(kind: str, identity: str, opposing: str) -> Dict[str, Any]:
    common = {
        "id": "contract-%s" % identity,
        "kind": kind,
        "signature_set": ["signature-%s" % identity],
        "deadline_s": 6.0,
    }
    if kind == "gate":
        common.update(target_gate="gate-%s" % identity, opposing_gate="gate-%s" % opposing)
    elif kind in {"lane", "scoped_lane"}:
        common.update(target_lane="lane-%s" % identity, scope="scope-main", stable_duration_s=2.0)
        if kind == "scoped_lane":
            common.update(entry_gate="gate-entry", exit_gate="gate-exit")
    elif kind in {"speed_band", "triggered_speed_band"}:
        common.update(entry_gate="gate-entry", exit_gate="gate-exit")
        if identity == "0000":
            common.update(speed_min_mps=6.0, speed_max_mps=8.0, scope="scope-main")
        else:
            common.update(speed_min_mps=10.0, speed_max_mps=12.0, scope="scope-main")
        if kind == "triggered_speed_band":
            common["trigger_id"] = "trigger-main"
    elif kind == "stop_region":
        common.update(region="region-%s" % identity, scope="scope-main", dwell_s=2.0, max_speed_mps=0.3)
    elif kind == "continue_region":
        common.update(region="region-%s" % opposing, scope="scope-main", target_gate="gate-%s" % identity)
    elif kind == "pass_side":
        common.update(actor_id="actor-0000", side="LEFT" if identity == "0000" else "RIGHT", min_clearance_m=1.0)
    return common


def _pcu(kind: str, layer: str = "A") -> Dict[str, Any]:
    if kind == "stop_region":
        second_kind = "continue_region"
    elif kind == "continue_region":
        second_kind = "stop_region"
    else:
        second_kind = kind
    first = _contract(kind, "0000", "0001")
    second = _contract(second_kind, "0001", "0000")
    return {
        "primary_layer": layer,
        "decision_window": {
            "start_s": 1.0,
            "deadline_s": 6.0,
            "commitment_gates": ["gate-0000", "gate-0001"],
        },
        "hard_constraints": {"minimum_progress_m": 5.0},
        "arms": {
            "R0": {"semantic_contract": first, "opposing_contract": second},
            "R1": {"semantic_contract": second, "opposing_contract": first},
        },
    }


def _rows() -> List[Dict[str, Any]]:
    return [
        {
            "time_s": float(time),
            "position": [float(time), 0.0],
            "speed_mps": 7.0,
            "progress_m": float(time),
            "lane_ids": [],
            "regions": [],
            "scopes": ["scope-main"] if 1 <= time <= 8 else [],
            "triggers": [],
            "gates_crossed": [],
            "relative_side": {},
            "relative_longitudinal_m": {},
            "clearance_m": {},
            "collision": False,
            "offroad": False,
            "image_black": False,
            "render_valid": True,
            "reconstruction_support_valid": True,
        }
        for time in range(10)
    ]


def _semantic_trace(kind: str, mode: str) -> List[Dict[str, Any]]:
    rows = _rows()
    if kind == "gate":
        if mode == "positive":
            rows[4]["gates_crossed"] = ["gate-0000"]
        elif mode in {"opposing", "wrong_first"}:
            rows[3]["gates_crossed"] = ["gate-0001"]
            if mode == "wrong_first":
                rows[4]["gates_crossed"] = ["gate-0000"]
        elif mode == "late":
            rows[7]["gates_crossed"] = ["gate-0000"]
        elif mode == "edge_pass":
            rows[6]["gates_crossed"] = ["gate-0000"]
        elif mode == "edge_fail":
            rows[7]["gates_crossed"] = ["gate-0000"]
    elif kind == "lane":
        if mode in {"positive", "edge_pass"}:
            for index in range(3, 6):
                rows[index]["lane_ids"] = ["lane-0000"]
        elif mode in {"opposing", "wrong_first"}:
            for index in range(2, 5):
                rows[index]["lane_ids"] = ["lane-0001"]
            if mode == "wrong_first":
                for index in range(5, 8):
                    rows[index]["lane_ids"] = ["lane-0000"]
        elif mode in {"partial", "oscillation"}:
            for index in (2, 4, 6):
                rows[index]["lane_ids"] = ["lane-0000"]
        elif mode in {"late", "edge_fail"}:
            for index in range(5, 8):
                rows[index]["lane_ids"] = ["lane-0000"]
    elif kind == "scoped_lane":
        for row in rows:
            row["scopes"] = ["scope-main"] if 2 <= row["time_s"] < 6 else []
            row["lane_ids"] = ["lane-0000"] if 2 <= row["time_s"] < 6 else []
        rows[2]["gates_crossed"] = ["gate-entry"]
        rows[6]["gates_crossed"] = ["gate-exit"]
        if mode in {"opposing", "wrong_first"}:
            rows[2]["lane_ids"] = ["lane-0001"]
        elif mode in {"partial", "oscillation", "post_dwell_departure"}:
            rows[5]["lane_ids"] = ["lane-0001"]
        elif mode in {"late", "edge_fail", "late_exit"}:
            rows[6]["gates_crossed"] = []
            rows[7]["gates_crossed"] = ["gate-exit"]
        elif mode in {"no_commit", "incomplete_scope"}:
            rows[6]["gates_crossed"] = []
        elif mode == "wrong_scope_order":
            rows[2]["gates_crossed"], rows[6]["gates_crossed"] = ["gate-exit"], ["gate-entry"]
        elif mode == "scope_escape":
            rows[3]["scopes"] = []
        elif mode == "ambiguous_lane":
            rows[5]["lane_ids"] = ["lane-0000", "lane-0001"]
    elif kind in {"speed_band", "triggered_speed_band"}:
        for row in rows:
            row["scopes"] = ["scope-main"] if 1 <= row["time_s"] <= 5 else []
        rows[1]["gates_crossed"] = ["gate-entry"]
        rows[6]["gates_crossed"] = ["gate-exit"]
        if kind == "triggered_speed_band":
            for row in rows[1:9]:
                row["triggers"] = ["trigger-main"]
        if mode in {"opposing", "wrong_first"}:
            for row in rows[1:9]:
                row["speed_mps"] = 11.0
        elif mode == "partial":
            rows[4]["speed_mps"] = 9.0
        elif mode == "oscillation":
            for index in range(1, 9, 2):
                rows[index]["speed_mps"] = 9.0
        elif mode == "edge_pass":
            for row in rows[1:9]:
                row["speed_mps"] = 8.0000005
        elif mode == "edge_fail":
            for row in rows[1:9]:
                row["speed_mps"] = 8.001
        elif mode == "late":
            for row in rows:
                row["scopes"] = ["scope-main"] if row["time_s"] >= 7 else []
                row["gates_crossed"] = []
            rows[7]["gates_crossed"] = ["gate-entry"]
            rows[9]["gates_crossed"] = ["gate-exit"]
        elif mode == "incomplete_scope":
            rows[6]["gates_crossed"] = []
        elif mode == "late_exit":
            rows[6]["scopes"] = ["scope-main"]
            rows[6]["gates_crossed"] = []
            rows[7]["gates_crossed"] = ["gate-exit"]
        elif mode == "wrong_scope_order":
            rows[1]["gates_crossed"] = ["gate-exit"]
            rows[6]["gates_crossed"] = ["gate-entry"]
        elif mode == "scope_escape":
            rows[3]["scopes"] = []
        elif mode == "trigger_avoidance" and kind == "triggered_speed_band":
            for row in rows:
                row["triggers"] = []
        elif mode in {"trigger_latched", "trigger_exit_speed_spike", "late_trigger", "preinstruction_trigger"} and kind == "triggered_speed_band":
            for row in rows:
                row["triggers"] = []
            trigger_index = 3 if mode == "late_trigger" else 0 if mode == "preinstruction_trigger" else 1
            rows[trigger_index]["triggers"] = ["trigger-main"]
            if mode == "trigger_exit_speed_spike":
                rows[3]["speed_mps"] = 20.0
    elif kind == "stop_region":
        if mode in {"positive", "edge_pass"}:
            for index in range(3, 6):
                rows[index]["regions"] = ["region-0000"]
                rows[index]["speed_mps"] = 0.0
        elif mode in {"opposing", "wrong_first"}:
            rows[3]["gates_crossed"] = ["gate-0001"]
            if mode == "wrong_first":
                for index in range(4, 7):
                    rows[index]["regions"] = ["region-0000"]
                    rows[index]["speed_mps"] = 0.0
        elif mode in {"partial", "oscillation"}:
            for index in (3, 5):
                rows[index]["regions"] = ["region-0000"]
                rows[index]["speed_mps"] = 0.0
        elif mode in {"late", "edge_fail"}:
            for index in range(6, 9):
                rows[index]["regions"] = ["region-0000"]
                rows[index]["speed_mps"] = 0.0
    elif kind == "continue_region":
        if mode in {"positive", "edge_pass"}:
            rows[4]["gates_crossed"] = ["gate-0000"]
        elif mode in {"opposing", "wrong_first"}:
            for index in range(2, 5):
                rows[index]["regions"] = ["region-0001"]
                rows[index]["speed_mps"] = 0.0
            if mode == "wrong_first":
                rows[5]["gates_crossed"] = ["gate-0000"]
        elif mode in {"late", "edge_fail"}:
            rows[7]["gates_crossed"] = ["gate-0000"]
    elif kind == "pass_side":
        crossing_index = 4
        side = "LEFT"
        clearance = 1.5
        if mode in {"opposing", "wrong_first"}:
            side = "RIGHT"
        elif mode == "partial":
            clearance = 0.5
        elif mode in {"late", "edge_fail"}:
            crossing_index = 7
        elif mode == "edge_pass":
            crossing_index = 6
        elif mode == "oscillation":
            for index, longitudinal in ((2, 2.0), (3, 0.5), (4, 1.5)):
                rows[index]["relative_longitudinal_m"] = {
                    "actor-0000": longitudinal
                }
                rows[index]["relative_side"] = {"actor-0000": "LEFT"}
                rows[index]["clearance_m"] = {"actor-0000": 1.5}
            return rows
        if mode not in {"no_commit", "trigger_avoidance"}:
            for index, longitudinal in (
                (crossing_index - 1, 2.0),
                (crossing_index, 0.0),
                (crossing_index + 1, -2.0),
            ):
                rows[index]["relative_longitudinal_m"] = {
                    "actor-0000": longitudinal
                }
                rows[index]["relative_side"] = {"actor-0000": side}
                rows[index]["clearance_m"] = {"actor-0000": clearance}
    return rows


def _case(kind: str, mode: str, expected_pass: bool, expected_failure: str = None) -> Dict[str, Any]:
    pcu = _pcu(kind)
    rows = _semantic_trace(kind, mode)
    if mode == "always_stop":
        for row in rows:
            row["progress_m"] = 0.0
            row["speed_mps"] = 0.0
    elif mode == "slow_creep":
        for index, row in enumerate(rows):
            row["progress_m"] = index * 0.1
            row["speed_mps"] = 0.1
    elif mode == "trigger_avoidance":
        for row in rows:
            row["scopes"] = []
            row["triggers"] = []
    elif mode == "no_commit" and kind in {"speed_band", "triggered_speed_band"}:
        for row in rows:
            row["scopes"] = []
            row["triggers"] = []
    elif mode == "collision":
        rows[4]["collision"] = True
    elif mode == "offroad":
        rows[4]["offroad"] = True
    elif mode == "unsafe":
        rows[4]["collision"] = True
    elif mode == "preintervention_collision":
        rows[0]["collision"] = True
    elif mode == "invalid_render":
        rows[4]["image_black"] = True
        rows[4]["render_valid"] = False
    elif mode == "outside_reconstruction_support":
        rows[4]["reconstruction_support_valid"] = False
        rows[4]["reconstruction_deviation_m"] = 0.5
        rows[4]["render_valid"] = False
    elif mode == "infrastructure_failure":
        rows[4]["infrastructure_failure"] = "renderer_unavailable"
    result = verify_trace(pcu, "R0", rows, threshold_profile={"speed_epsilon_mps": 1e-6})
    passed = result.passed is expected_pass and (
        expected_failure is None or result.failure == expected_failure
    )
    return {
        "name": mode,
        "pass": passed,
        "expected_pass": expected_pass,
        "expected_failure": expected_failure,
        "actual": result.to_dict(),
    }


def run_verifier_selftests() -> Dict[str, Any]:
    expected_by_kind = {
        "gate": {
            "positive": (True, None), "opposing": (False, "F2_WRONG_SEMANTIC"),
            "wrong_first": (False, "F2_WRONG_SEMANTIC"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F1_NO_COMMIT"), "oscillation": (False, "F1_NO_COMMIT"),
            "edge_pass": (True, None), "edge_fail": (False, "F6_ORDER_SCOPE"),
        },
        "lane": {
            "positive": (True, None), "opposing": (False, "F2_WRONG_SEMANTIC"),
            "wrong_first": (False, "F2_WRONG_SEMANTIC"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F1_NO_COMMIT"), "oscillation": (False, "F1_NO_COMMIT"),
            "edge_pass": (True, None), "edge_fail": (False, "F6_ORDER_SCOPE"),
        },
        "scoped_lane": {
            "positive": (True, None), "opposing": (False, "F2_WRONG_SEMANTIC"),
            "wrong_first": (False, "F2_WRONG_SEMANTIC"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F2_WRONG_SEMANTIC"), "oscillation": (False, "F2_WRONG_SEMANTIC"),
            "edge_pass": (True, None), "edge_fail": (False, "F6_ORDER_SCOPE"),
            "post_dwell_departure": (False, "F2_WRONG_SEMANTIC"), "ambiguous_lane": (False, "F2_WRONG_SEMANTIC"),
        },
        "speed_band": {
            "positive": (True, None), "opposing": (False, "F5_QUANTITATIVE"),
            "wrong_first": (False, "F5_QUANTITATIVE"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F5_QUANTITATIVE"), "oscillation": (False, "F5_QUANTITATIVE"),
            "edge_pass": (True, None), "edge_fail": (False, "F5_QUANTITATIVE"),
        },
        "triggered_speed_band": {
            "positive": (True, None), "opposing": (False, "F5_QUANTITATIVE"),
            "wrong_first": (False, "F5_QUANTITATIVE"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F5_QUANTITATIVE"), "oscillation": (False, "F5_QUANTITATIVE"),
            "edge_pass": (True, None), "edge_fail": (False, "F5_QUANTITATIVE"),
        },
        "stop_region": {
            "positive": (True, None), "opposing": (False, "F2_WRONG_SEMANTIC"),
            "wrong_first": (False, "F2_WRONG_SEMANTIC"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F1_NO_COMMIT"), "oscillation": (False, "F1_NO_COMMIT"),
            "edge_pass": (True, None), "edge_fail": (False, "F6_ORDER_SCOPE"),
        },
        "continue_region": {
            "positive": (True, None), "opposing": (False, "F2_WRONG_SEMANTIC"),
            "wrong_first": (False, "F2_WRONG_SEMANTIC"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F1_NO_COMMIT"), "oscillation": (False, "F1_NO_COMMIT"),
            "edge_pass": (True, None), "edge_fail": (False, "F6_ORDER_SCOPE"),
        },
        "pass_side": {
            "positive": (True, None), "opposing": (False, "F2_WRONG_SEMANTIC"),
            "wrong_first": (False, "F2_WRONG_SEMANTIC"), "late": (False, "F6_ORDER_SCOPE"),
            "partial": (False, "F3_UNSAFE_FOLLOW"), "oscillation": (False, "F1_NO_COMMIT"),
            "edge_pass": (True, None), "edge_fail": (False, "F6_ORDER_SCOPE"),
        },
    }
    implementations = {}
    generic = {
        "no_commit": (False, None),
        "always_stop": (False, "F10_ANTI_PROGRESS_PATTERN"),
        "slow_creep": (False, "F10_ANTI_PROGRESS_PATTERN"),
        "trigger_avoidance": (False, None),
        "collision": (False, "F3_UNSAFE_FOLLOW"),
        "offroad": (False, "F3_UNSAFE_FOLLOW"),
        "unsafe": (False, "F3_UNSAFE_FOLLOW"),
        "preintervention_collision": (False, "F0_PREINTERVENTION"),
        "invalid_render": (False, "INVALID_SIM"),
        "outside_reconstruction_support": (False, "F3_UNSAFE_FOLLOW"),
        "infrastructure_failure": (False, "INVALID_SIM"),
    }
    for kind, cases in expected_by_kind.items():
        if kind in {"speed_band", "triggered_speed_band", "scoped_lane"}:
            cases.update({name: (False, "F6_ORDER_SCOPE") for name in (
                "incomplete_scope", "late_exit", "wrong_scope_order", "scope_escape")})
        if kind == "triggered_speed_band":
            cases.update(trigger_latched=(True, None), trigger_exit_speed_spike=(False, "F5_QUANTITATIVE"),
                         late_trigger=(False, "F6_ORDER_SCOPE"), preinstruction_trigger=(False, "F6_ORDER_SCOPE"))
        rows = []
        for name, (expected_pass, expected_failure) in dict(cases, **generic).items():
            rows.append(_case(kind, name, expected_pass, expected_failure))
        implementations[kind] = {"pass": all(row["pass"] for row in rows), "cases": rows}
    family_kind = {
        "A1_intersection_left_right": "gate", "A2_straight_turn": "gate",
        "A3_fork_left_right": "gate", "A4_lane_change_left_right": "lane",
        "A5_lane_change_keep": "lane", "A6_pull_over_continue": "stop_region",
        "B1_segment_speed_band": "speed_band", "B2_scoped_lane_preference": "scoped_lane",
        "B3_triggered_speed_constraint": "triggered_speed_band", "B4_object_relative_passing": "pass_side",
        "B5_relative_stopping_region": "stop_region", "C1_direction_sign_grounding": "gate",
        "C2_road_exit_text_grounding": "gate", "C3_landmark_adjacent_branch": "gate",
        "C4_attribute_landmark_target": "gate",
    }
    families = {
        family: {"implementation": kind, "pass": implementations[kind]["pass"]}
        for family, kind in family_kind.items()
    }
    layer_c_result = verify_trace(
        _pcu("gate", layer="C"),
        "R0",
        _semantic_trace("gate", "opposing"),
        threshold_profile={"speed_epsilon_mps": 1e-6},
    )
    layer_c_wrong_referent = {
        "pass": (
            layer_c_result.passed is False
            and layer_c_result.failure == "F4_WRONG_REFERENT_OUTCOME"
        ),
        "actual": layer_c_result.to_dict(),
    }
    return {
        "schema_version": "l2d.realsim.verifier_tests.v0.1",
        "pass": (
            all(row["pass"] for row in implementations.values())
            and set(families) == set(FAMILY_TARGETS)
            and layer_c_wrong_referent["pass"]
        ),
        "implementations": implementations,
        "families": families,
        "layer_c_wrong_referent": layer_c_wrong_referent,
    }
