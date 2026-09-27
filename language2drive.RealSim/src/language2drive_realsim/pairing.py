"""Paired-input controls and route-command leakage checks."""

from pathlib import Path
from typing import Any, Dict, List

from .constants import LEAKAGE_TOKENS
from .io import stable_hash


PAIR_FIELDS = (
    "initial_state",
    "route",
    "command",
    "camera_configuration",
    "sensor_history_before_intervention",
    "ego_state",
    "renderer_configuration",
    "scene_reference",
    "simulation_seed",
    "route_use_waypoint_commands",
)


class LeakageError(ValueError):
    pass


def non_language_projection(run: Dict[str, Any]) -> Dict[str, Any]:
    missing = [field for field in PAIR_FIELDS if field not in run]
    if missing:
        raise LeakageError("non-language input missing fields: %s" % missing)
    return {field: run[field] for field in PAIR_FIELDS}


def non_language_hash(run: Dict[str, Any]) -> str:
    return stable_hash(non_language_projection(run))


def assert_paired_inputs(first: Dict[str, Any], second: Dict[str, Any]) -> str:
    left = non_language_projection(first)
    right = non_language_projection(second)
    if left != right:
        mismatches = [field for field in PAIR_FIELDS if left[field] != right[field]]
        raise LeakageError("arm-dependent non-language inputs: %s" % mismatches)
    if left["command"] not in ("UNKNOWN", "NEUTRAL"):
        raise LeakageError("command must be UNKNOWN/NEUTRAL, got %r" % left["command"])
    if left["route_use_waypoint_commands"] is not False:
        raise LeakageError("target-derived waypoint commands must be disabled")
    return stable_hash(left)


def audit_identifier(value: str, path: str) -> List[str]:
    normalized = Path(value).name.casefold()
    return sorted(token for token in LEAKAGE_TOKENS if token in normalized)


def build_leakage_audit(compiled: Dict[str, Any]) -> Dict[str, Any]:
    by_pair = {}
    for run in compiled.get("runs", []):
        private = run.get("_private_identity", {})
        key = (private.get("pcu_id"), run.get("simulation_seed"))
        by_pair.setdefault(key, {})[private.get("arm_id")] = run
    rows = []
    for (pcu_id, seed), arms in sorted(by_pair.items()):
        errors = []
        fingerprint = None
        if set(arms) != {"R0", "R1"}:
            errors.append("missing_arm")
        else:
            try:
                fingerprint = assert_paired_inputs(arms["R0"], arms["R1"])
            except LeakageError as error:
                errors.append(str(error))
        for arm, run in arms.items():
            leaked = audit_identifier(run.get("session_uuid", ""), "session_uuid")
            if leaked:
                errors.append("%s session identifier leaks %s" % (arm, leaked))
            if "pcu_id" in run or "arm_id" in run or "semantic_contract" in run:
                errors.append("%s public run contains private identity/contract" % arm)
        rows.append(
            {
                "pcu_id": pcu_id,
                "seed": seed,
                "paired_input_hash": fingerprint,
                "pass": not errors,
                "errors": errors,
            }
        )
    return {
        "schema_version": "l2d.realsim.leakage_audit.v0.1",
        "pass": bool(rows) and all(row["pass"] for row in rows),
        "pairs": rows,
    }
