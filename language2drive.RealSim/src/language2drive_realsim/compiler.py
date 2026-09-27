"""Compile trusted PCUs into public AlpaSim workloads and a private sidecar."""

import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from .constants import ARMS, NUREC_VERSION_STRING
from .io import atomic_json, stable_hash
from .pairing import assert_paired_inputs, build_leakage_audit
from .schema import validate_source


SESSION_NAMESPACE = uuid.UUID("f1779319-c80f-4a51-a4c5-a26753e50bca")


def _session_id(pcu_id: str, arm_id: str, seed: int, source_hash: str) -> str:
    # UUID text is opaque and contains no task/family/answer vocabulary.
    return str(uuid.uuid5(SESSION_NAMESPACE, "%s:%s:%s:%s" % (source_hash, pcu_id, arm_id, seed)))


def _non_language_run(pcu: Dict[str, Any], seed: int) -> Dict[str, Any]:
    scene = pcu["scene"]
    shared = pcu["shared_non_language_input"]
    return {
        "initial_state": scene["initial_pose"],
        "route": shared["route_envelope"],
        "command": "UNKNOWN",
        "camera_configuration": scene["camera_configuration"],
        "sensor_history_before_intervention": shared.get(
            "preintervention_history_id", "shared-recorded-prefix"
        ),
        "ego_state": scene["initial_pose"],
        "renderer_configuration": shared["simulator_configuration"].get("renderer", {}),
        "scene_reference": {
            "artifact_id": scene["artifact_id"],
            "recording_start_timestamp_us": scene["recording_start_timestamp_us"],
            "start_timestamp_us": scene["start_timestamp_us"],
            "start_time_offset_us": scene["start_time_offset_us"],
        },
        "simulation_seed": int(seed),
    }


def compile_source(source: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    validate_source(source)
    source_hash = stable_hash(source)
    public_runs = []
    private_sessions = {}
    audit_runs = []
    for pcu in source["pcus"]:
        seeds = pcu["shared_non_language_input"]["coupled_exogenous_seeds"]
        for seed_index, seed in enumerate(seeds):
            paired = {}
            for arm_id in ARMS:
                session_uuid = _session_id(pcu["pcu_id"], arm_id, seed, source_hash)
                non_language = _non_language_run(pcu, seed)
                public_run = dict(non_language)
                public_run.update(
                    {
                        "session_uuid": session_uuid,
                        "output_name": session_uuid + ".asl",
                        "route_use_waypoint_commands": False,
                    }
                )
                public_runs.append(public_run)
                audit_run = dict(public_run)
                audit_run["_private_identity"] = {"pcu_id": pcu["pcu_id"], "arm_id": arm_id}
                audit_runs.append(audit_run)
                paired[arm_id] = audit_run
                issuance = pcu["arms"][arm_id]["instruction_schedule"][0]
                instruction = issuance["utterances"][seed_index % len(issuance["utterances"])]
                private_sessions[session_uuid] = {
                    "pcu_id": pcu["pcu_id"],
                    "arm_id": arm_id,
                    "seed": seed,
                    "scene_id_private": pcu["scene"]["scene_id_private"],
                    "artifact_id": pcu["scene"]["artifact_id"],
                    "nre_version_string": NUREC_VERSION_STRING,
                    "recording_start_timestamp_us": pcu["scene"][
                        "recording_start_timestamp_us"
                    ],
                    "start_timestamp_us": pcu["scene"]["start_timestamp_us"],
                    "start_time_offset_us": pcu["scene"]["start_time_offset_us"],
                    "initial_pose": pcu["scene"]["initial_pose"],
                    "camera_configuration": pcu["scene"]["camera_configuration"],
                    "instruction_schedule": [
                        {
                            "issued_at_us": int(issuance["issue_at_us"]),
                            "version": 1,
                            "text": instruction,
                        }
                    ],
                    "semantic_contract": pcu["arms"][arm_id]["semantic_contract"],
                    "opposing_contract": pcu["arms"][arm_id]["opposing_contract"],
                    "decision_window": pcu["decision_window"],
                    "hard_constraints": pcu["hard_constraints"],
                    "annotations": pcu.get("private_annotations", {}),
                    "oracle_plan": pcu.get("oracle_plans_by_seed", {})
                    .get(str(seed), {})
                    .get(arm_id),
                }
            assert_paired_inputs(paired["R0"], paired["R1"])
    public = {
        "schema_version": "l2d.realsim.public_workload.v0.1",
        "source_binding": source_hash,
        "model_boundary": "camera+ego+UNKNOWN_command+free_form_instruction; route withheld by driver adapter",
        "runs": public_runs,
    }
    sidecar = {
        "schema_version": "l2d.realsim.sidecar.v0.1",
        "source_binding": source_hash,
        "sessions": private_sessions,
    }
    audit = build_leakage_audit({"runs": audit_runs})
    audit["source_binding"] = source_hash
    return public, sidecar, audit


def compile_to_directory(source: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    public, sidecar, audit = compile_source(source)
    output_dir = Path(output_dir)
    paths = {
        "public": output_dir / "public_workload.json",
        "sidecar": output_dir / "sidecar.private.json",
        "leakage": output_dir / "leakage_audit.json",
    }
    atomic_json(paths["public"], public)
    atomic_json(paths["sidecar"], sidecar)
    atomic_json(paths["leakage"], audit)
    return paths
