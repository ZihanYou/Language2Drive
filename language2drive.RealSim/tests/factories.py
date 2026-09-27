from copy import deepcopy

from language2drive_realsim.constants import (
    COUPLED_SEEDS,
    FAMILY_TARGETS,
    ORACLE_CONTROLLER_ID,
    SCHEMA_VERSION,
)


def contract(kind, serial, arm, family):
    other = 1 - arm
    value = {
        "id": "contract-%04x-%04x" % (serial, arm),
        "kind": kind,
        "signature_set": ["signature-%04x-%04x" % (serial, arm)],
        "deadline_s": 8.0,
    }
    if kind == "gate":
        value.update(target_gate="gate-%04x-%d" % (serial, arm), opposing_gate="gate-%04x-%d" % (serial, other))
    elif kind in {"lane", "scoped_lane"}:
        value.update(target_lane="lane-%04x-%d" % (serial, arm), scope="scope-%04x" % serial, stable_duration_s=1.0)
        if kind == "scoped_lane":
            value.update(entry_gate="gate-%04x-0" % serial, exit_gate="gate-%04x-1" % serial)
    elif kind in {"speed_band", "triggered_speed_band"}:
        value.update(entry_gate="gate-%04x-0" % serial, exit_gate="gate-%04x-1" % serial)
        lower = 5.0 if arm == 0 else 10.0
        value.update(speed_min_mps=lower, speed_max_mps=lower + 2.0, scope="scope-%04x" % serial)
        if kind == "triggered_speed_band":
            value["trigger_id"] = "trigger-%04x" % serial
    elif kind == "pass_side":
        value.update(actor_id="actor-%04x" % serial, side="LEFT" if arm == 0 else "RIGHT", min_clearance_m=1.0)
    elif kind == "stop_region":
        value.update(region="region-%04x-%d" % (serial, arm), scope="scope-%04x" % serial, dwell_s=1.0, max_speed_mps=0.3)
    elif kind == "continue_region":
        value.update(region="region-%04x-%d" % (serial, other), scope="scope-%04x" % serial, target_gate="gate-%04x-%d" % (serial, arm))
    return value


def family_kind(family, arm):
    if family.startswith(("A1", "A2", "A3", "C")):
        return "gate"
    if family.startswith("B2"):
        return "scoped_lane"
    if family.startswith(("A4", "A5")):
        return "lane"
    if family.startswith("B1"):
        return "speed_band"
    if family.startswith("B3"):
        return "triggered_speed_band"
    if family.startswith("B4"):
        return "pass_side"
    if family.startswith("A6"):
        return "stop_region" if arm == 0 else "continue_region"
    return "stop_region"


def make_pcu(serial, family):
    layer = family[0]
    contracts = [contract(family_kind(family, arm), serial, arm, family) for arm in (0, 1)]
    gate_ids = []
    for item in contracts:
        if item.get("target_gate"):
            gate_ids.append(item["target_gate"])
        if item.get("opposing_gate"):
            gate_ids.append(item["opposing_gate"])
    gate_ids = sorted(set(gate_ids)) or ["gate-%04x-0" % serial, "gate-%04x-1" % serial]
    artifact_uuid = "00000000-0000-4000-8000-%012x" % serial
    scene_uuid = "10000000-0000-4000-8000-%012x" % serial
    gates = {
        gate_id: (
            [[3.0, -5.0], [3.0, -0.1]]
            if gate_id.endswith("-0")
            else [[3.0, 0.1], [3.0, 5.0]]
        )
        for gate_id in gate_ids
    }
    lanes = {}
    regions = {}
    scopes = {"scope-%04x" % serial: [[0.0, -10.0], [20.0, -10.0], [20.0, 10.0], [0.0, 10.0]]}
    triggers = {
        "trigger-%04x" % serial: [
            [0.0, -4.0],
            [2.0, -4.0],
            [2.0, 4.0],
            [0.0, 4.0],
        ]
    }
    actor_ids = []
    for item in contracts:
        if item.get("target_lane"):
            lower, upper = (-4.0, -0.1) if item["target_lane"].endswith("-0") else (0.1, 4.0)
            lanes[item["target_lane"]] = [[0.0, lower], [20.0, lower], [20.0, upper], [0.0, upper]]
        if item.get("region"):
            lower, upper = (-2.0, -0.1) if item["region"].endswith("-0") else (0.1, 2.0)
            regions[item["region"]] = [[3.0, lower], [6.0, lower], [6.0, upper], [3.0, upper]]
        if item.get("trigger_id"):
            triggers[item["trigger_id"]] = [[2.0, -4.0], [8.0, -4.0], [8.0, 4.0], [2.0, 4.0]]
        if item.get("actor_id"):
            actor_ids.append(item["actor_id"])
    annotations = {
        "gates": gates,
        "regions": regions,
        "lanes": lanes,
        "scopes": scopes,
        "triggers": triggers,
        "actor_ids": sorted(set(actor_ids)),
        "reconstruction_support_polygon": [[-10.0, -20.0], [40.0, -20.0], [40.0, 20.0], [-10.0, 20.0]],
        "legality_reviewed": True,
        "balance_reviewed": True,
        "render_reviewed": True,
    }
    if family == "B3_triggered_speed_constraint":
        annotations["trigger_referent"] = {
            "referent_id": "referent-%04x" % serial,
            "referent_kind": "landmark",
            "trigger_id": contracts[0]["trigger_id"],
            "private_description": "Synthetic fixture landmark; not an actual scene annotation",
            "model_input_resolution": [320, 576],
            "frame_refs": ["synthetic-frame-%04x" % serial],
            "identity_reviewed": True,
            "visibility_reviewed": True,
            "trigger_geometry_reviewed": True,
        }
    if layer == "C":
        annotations["visibility_review"] = {
            "model_input_resolution": [320, 576],
            "legible": True,
            "reviewed_frame_refs": ["frame-%04x" % serial],
        }
        cue_type = {
            "C1_direction_sign_grounding": "direction_sign",
            "C2_road_exit_text_grounding": "road_exit_text",
            "C3_landmark_adjacent_branch": "landmark",
            "C4_attribute_landmark_target": "attribute_landmark",
        }[family]
        annotations["semantic_referents"] = {
            arm_id: {
                "referent_id": "referent-%04x-%04x" % (serial, arm),
                "cue_type": cue_type,
                "target_gate": contracts[arm]["target_gate"],
                "private_description": "synthetic fixture referent %d" % arm,
                "reviewed_frame_refs": ["frame-%04x" % serial],
            }
            for arm, arm_id in enumerate(("R0", "R1"))
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "pcu_id": "pcu-%04x" % serial,
        "meta_group_id": "group-%04x" % serial,
        "backend": {
            "type": "alpasim",
            "alpasim_commit": "affc2eab209fa43bdfa2f26c0f8d437922d78a68",
            "renderer_type": "nurec",
            "dataset_version": "26.01",
        },
        "primary_layer": layer,
        "task_family": family,
        "scene": {
            "artifact_id": artifact_uuid,
            "scene_id_private": "clipgt-" + scene_uuid,
            "recording_id_private": "recording-%04x" % serial,
            "recording_start_timestamp_us": 0,
            "start_timestamp_us": 1000000,
            "start_time_offset_us": 1000000,
            "initial_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_rad": 0.0},
            "camera_configuration": {"logical_ids": ["camera_front_wide_120fov"], "height": 320, "width": 576},
            "reconstruction_support_id": "support-%04x" % serial,
        },
        "shared_non_language_input": {
            "route_envelope": [[0.0, 0.0], [25.0, 0.0]],
            "route_envelope_kind": "COMMON_SUPPORT",
            "route_semantic_target": None,
            "route_neutrality_reviewed": True,
            "command_policy": "UNKNOWN_NEUTRAL",
            "coupled_exogenous_seeds": list(COUPLED_SEEDS),
            "preintervention_history_id": "prefix-%04x" % serial,
            "simulator_configuration": {"renderer": {"kind": "sensorsim", "nurec": "26.1"}},
        },
        "decision_window": {
            "trigger": "trigger-%04x" % serial,
            "start_s": 1.0,
            "deadline_s": 8.0,
            "commitment_gates": gate_ids,
        },
        "arms": {
            "R0": {
                "instruction_schedule": [{"issue_at_us": 1000000, "utterances": [
                    "Apply option alpha %04x." % serial,
                    "Use the first described choice %04x." % serial,
                    "Follow instruction alpha %04x." % serial,
                ]}],
                "semantic_contract": contracts[0],
                "opposing_contract": deepcopy(contracts[1]),
            },
            "R1": {
                "instruction_schedule": [{"issue_at_us": 1000000, "utterances": [
                    "Apply option beta %04x." % serial,
                    "Use the second described choice %04x." % serial,
                    "Follow instruction beta %04x." % serial,
                ]}],
                "semantic_contract": contracts[1],
                "opposing_contract": deepcopy(contracts[0]),
            },
        },
        "hard_constraints": {
            "collision": False,
            "offroad": False,
            "minimum_progress_m": 5.0,
            "legality": True,
            "deadline_s": 8.0,
        },
        "oracle_plans_by_seed": {
            str(seed): {
                arm_id: {
                    "controller": ORACLE_CONTROLLER_ID,
                    "coordinate_frame": "nre_local",
                    "waypoints": [
                        {"time_s": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw_rad": 0.0},
                        {
                            "time_s": 10.0,
                            "x": 10.0,
                            "y": -2.0 if arm_id == "R0" else 2.0,
                            "z": 0.0,
                            "yaw_rad": 0.0,
                        },
                    ],
                }
                for arm_id in ("R0", "R1")
            }
            for seed in COUPLED_SEEDS
        },
        "private_annotations": annotations,
    }


def make_release_source():
    pcus = []
    serial = 1
    for family, count in FAMILY_TARGETS.items():
        for _ in range(count):
            pcus.append(make_pcu(serial, family))
            serial += 1
    return {"schema_version": SCHEMA_VERSION, "pcus": pcus}


def basic_trace():
    return [
        {
            "time_s": float(index),
            "position": [float(index), 0.0],
            "speed_mps": 6.0,
            "progress_m": float(index),
            "lane_ids": [],
            "regions": [],
            "scopes": [],
            "triggers": [],
            "gates_crossed": [],
            "relative_side": {},
            "clearance_m": {},
            "collision": False,
            "offroad": False,
            "render_valid": True,
        }
        for index in range(10)
    ]
