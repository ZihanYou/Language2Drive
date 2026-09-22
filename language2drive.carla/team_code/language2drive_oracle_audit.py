"""Static, fail-closed audit for the Language2Drive privileged oracle.

The release suite contains three language expressions for each of 256
counterfactual arms.  This module compiles every private route with the exact
oracle program compiler, verifies that the three expressions share one hidden
program, and reports target-type coverage.  It never connects to CARLA and
does not write unless ``--output`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from xml.etree import ElementTree as ET

from team_code.l2d_oracle_agent import (
    Language2DriveOracleAgent,
    OracleProgramError,
)
from team_code.language2drive_trace import (
    load_private_route_context,
    private_public_start_consistency_errors,
    public_private_consistency_errors,
)
from team_code.language2drive_runtime import (
    _path_point,
    validate_actor_setup_program,
)


CORE_TARGET_TYPES = frozenset({
    "actor_order",
    "delayed_progress",
    "gate",
    "headway",
    "if_else",
    "lane",
    "memory",
    "parking_slot",
    "progress",
    "region",
    "sequence",
    "speed_band",
    "until",
})

CANONICAL_CORE_SIGNAL_FREE_PROOF_ROUTES = 132
CANONICAL_CORE_SIGNAL_FREE_PROOF_ARMS = 44

_EXPRESSION_IDENTITY_KEYS = frozenset({
    "route_entry_id",
    "instruction",
    "paraphrase",
})

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _program_signature(route: Mapping[str, Any]) -> str:
    # Compare every current and future private field, excluding only the three
    # values that intentionally identify a language expression.  In
    # particular this includes legal-speed proof/provenance and any future D
    # FSM metadata, so a newly added field cannot silently differ among the
    # canonical instruction and its two controlled paraphrases.
    return _canonical({
        key: value
        for key, value in route.items()
        if key not in _EXPRESSION_IDENTITY_KEYS
    })


def validate_sidecar_source_files(
    source_files: Any,
    *,
    required: bool,
) -> Dict[str, Any]:
    """Verify every top-level release source by canonical path and SHA256.

    Parking's semantic registry closes the meaning of its five direct inputs;
    this check has the separate, broader responsibility of proving that every
    file claimed by ``sidecar.private.json/source_files`` still exists with
    the frozen bytes.  The parking loader additionally requires its bbox,
    XODR, lane graph, controller config and vehicle-geometry trace to appear in
    this map with the registry's exact digest.
    """

    if source_files is None:
        if required:
            raise ValueError("release sidecar requires source_files")
        return {"present": False, "count": 0, "verified": 0}
    if not isinstance(source_files, Mapping):
        raise ValueError("source_files must be an object")
    if not source_files:
        if required:
            raise ValueError("release sidecar requires non-empty source_files")
        return {"present": True, "count": 0, "verified": 0}

    normalized_entries = []
    for raw_path, raw_digest in source_files.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("source_files keys must be non-empty paths")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("source_files path must be absolute: {}".format(raw_path))
        # Canonical release identity is the normalized absolute spelling, not
        # the backing inode spelling.  Shared workspaces may expose the same
        # file through a deliberate lexical alias (for example ``lwq`` backed
        # by ``simlingo_fzh``); following that symlink here would disagree with
        # the producer and parking registry while proving no additional bytes.
        canonical = Path(os.path.abspath(str(candidate)))
        normalized_entries.append((raw_path, raw_digest, canonical))

    normalized_paths: Dict[str, str] = {}
    for raw_path, _, canonical in normalized_entries:
        previous = normalized_paths.get(str(canonical))
        if previous is not None:
            raise ValueError(
                "source_files normalizes duplicate path: {} and {}".format(
                    previous, raw_path
                )
            )
        normalized_paths[str(canonical)] = raw_path

    for raw_path, raw_digest, canonical in normalized_entries:
        if raw_path != str(canonical):
            raise ValueError(
                "source_files path must be canonical: {}".format(raw_path)
            )
        if not isinstance(raw_digest, str) or _SHA256.fullmatch(raw_digest) is None:
            raise ValueError(
                "source_files digest must be lowercase SHA256: {}".format(
                    canonical
                )
            )
        if not canonical.is_file():
            raise ValueError("source_files path is not a file: {}".format(canonical))
        digest = hashlib.sha256()
        with canonical.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != raw_digest:
            raise ValueError(
                "source_files SHA256 mismatch: {}".format(canonical)
            )
    return {
        "present": True,
        "count": len(source_files),
        "verified": len(source_files),
    }


def compile_private_route_static(
    route: Mapping[str, Any], *, parking_manoeuvre_registry: Any = None,
) -> Dict[str, Any]:
    """Compile one sidecar route without touching a CARLA world.

    Live map-limit verification remains an agent-setup check.  Everything
    else -- target dispatch, condition selection, speed proof, private path,
    parking approach and region terminal-pose binding -- is exercised here by
    the same implementation used at runtime.
    """

    if not isinstance(route, Mapping):
        raise OracleProgramError("private route is not an object")
    raw_path = route.get("oracle_path")
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise OracleProgramError("release oracle_path requires at least two points")
    previous = None
    for index, point in enumerate(raw_path):
        if not isinstance(point, Mapping):
            raise OracleProgramError(
                "release oracle_path point {} is not an object".format(index)
            )
        try:
            if not {"x", "y", "z", "yaw_deg"}.issubset(point):
                raise KeyError("x/y/z/yaw_deg")
            values = (
                float(point["x"]), float(point["y"]),
                float(point["z"]), float(point["yaw_deg"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleProgramError(
                "release oracle_path point {} requires finite xyz/yaw_deg".format(
                    index
                )
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise OracleProgramError(
                "release oracle_path point {} has non-finite xyz/yaw_deg".format(
                    index
                )
            )
        if previous is not None and math.hypot(
            values[0] - previous[0], values[1] - previous[1]
        ) <= 1e-6:
            raise OracleProgramError(
                "release oracle_path has duplicate consecutive point {}"
                .format(index)
            )
        previous = values

    # The Bench2Drive base constructor looks up a live hero actor.  A static
    # release audit must not depend on or silently connect to CARLA, so create
    # only the compiler state consumed by ``_compile_program``.
    agent = Language2DriveOracleAgent.__new__(Language2DriveOracleAgent)
    agent._route = copy.deepcopy(dict(route))  # trusted release audit API
    agent._target = None
    agent._conditional_branch = None
    agent._points = []
    agent._cumulative = []
    agent._curve_vertices = []
    agent._segment = 0
    agent._along = 0.0
    agent._contract_speed_band = None
    agent._speed_band = None
    agent._speed_limit_proof_mps = None
    agent._speed_feasibility_cap_mps = None
    agent._signal_free_corridor_proof = None
    agent._until_end = None
    agent._until_end_entered = False
    agent._until_progress_min = None
    agent._until_progress_entered = False
    agent._wait_seconds = 0.0
    agent._parking = None
    agent._parking_manoeuvre = None
    agent._parking_manoeuvre_registry = parking_manoeuvre_registry
    agent._parking_state = "none"
    agent._terminal_pose = None
    agent._terminal_state = "none"
    agent._compile_program()
    return {
        "declared_target_type": str(route.get("target", {}).get("type", "")),
        "resolved_target_type": str((agent._target or {}).get("type", "")),
        "path_points": len(agent._points),
        "speed_band": list(agent._speed_band) if agent._speed_band else None,
        "speed_limit_proof_mps": agent._speed_limit_proof_mps,
        "speed_feasibility_cap_mps": agent._speed_feasibility_cap_mps,
        "signal_free_corridor_method": (
            agent._signal_free_corridor_proof.get("method")
            if agent._signal_free_corridor_proof is not None
            else None
        ),
        "signal_free_corridor_lane_id": (
            agent._signal_free_corridor_proof.get("lane_id")
            if agent._signal_free_corridor_proof is not None
            else None
        ),
        "until_progress_min": agent._until_progress_min,
        "conditional_branch": agent._conditional_branch,
        "parking_approach_source": (
            agent._parking.get("approach_pose_source")
            if agent._parking is not None
            else None
        ),
        "parking_schema_version": (
            agent._parking.get("schema_version")
            if agent._parking is not None
            else None
        ),
        "parking_handoff_pose": (
            list(agent._parking.get("approach", ()))
            if agent._parking is not None
            else None
        ),
        "parking_compiled_path_endpoint": (
            list(agent._points[-1]) if agent._parking is not None else None
        ),
        "parking_slot_axis_approach_pose": (
            copy.deepcopy(agent._parking.get("slot_axis_approach_pose"))
            if agent._parking is not None
            else None
        ),
        "parking_v5_capture_contract": (
            {
                "endpoint_stop_m": (
                    Language2DriveOracleAgent.PARKING_V5_PRELEAD_ENDPOINT_STOP_M
                ),
                "rear_distance_max_m": (
                    Language2DriveOracleAgent.PARKING_V5_PRELEAD_REAR_DISTANCE_MAX_M
                ),
                "yaw_error_max_deg": (
                    Language2DriveOracleAgent.PARKING_V5_PRELEAD_YAW_ERROR_MAX_DEG
                ),
                "speed_below_mps": 0.15,
            }
            if agent._parking is not None and agent._parking.get("is_v5")
            else None
        ),
        "region_terminal_pose": agent._terminal_pose is not None,
    }


def audit_private_sidecar(
    sidecar_path: str | Path,
    *,
    expected_routes: int = 768,
    expected_pcus: int = 128,
    expected_arms: int = 256,
    expected_expressions_per_arm: int = 3,
    required_target_types: Optional[Iterable[str]] = CORE_TARGET_TYPES,
) -> Dict[str, Any]:
    """Return a deterministic release audit report for one private sidecar."""

    path = Path(sidecar_path).expanduser().resolve()
    errors = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "language2drive.oracle_static_audit.v1",
            "success": False,
            "sidecar_path": str(path),
            "errors": ["sidecar_read_failed:{}:{}".format(type(exc).__name__, exc)],
        }
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("routes"), list)
        or not isinstance(payload.get("pcus"), list)
    ):
        return {
            "schema_version": "language2drive.oracle_static_audit.v1",
            "success": False,
            "sidecar_path": str(path),
            "errors": ["sidecar_requires_routes_and_pcus_lists"],
        }
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "language2drive.private_sidecar.v1":
        errors.append(
            "unsupported_private_sidecar_schema:{!r}".format(schema_version)
        )

    routes = payload["routes"]
    pcus = payload["pcus"]
    sidecar_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical_core_counts = (
        int(expected_routes) == 768
        and int(expected_pcus) == 128
        and int(expected_arms) == 256
        and int(expected_expressions_per_arm) == 3
    )
    source_files_result = {"present": False, "count": 0, "verified": 0}
    source_file_errors = []
    try:
        source_files_result = validate_sidecar_source_files(
            payload.get("source_files"),
            required=(
                canonical_core_counts
                or payload.get("parking_manoeuvre_registry") is not None
            ),
        )
    except (OSError, ValueError) as exc:
        source_file_errors.append("{}:{}".format(type(exc).__name__, exc))
        errors.append("source_file_errors:1")
    if len(routes) != int(expected_routes):
        errors.append(
            "route_count:{}!=expected:{}".format(len(routes), expected_routes)
        )

    route_ids = []
    numeric_route_ids = []
    arm_routes: Dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    target_counts: Counter[str] = Counter()
    resolved_counts: Counter[str] = Counter()
    signal_free_proof_methods: Counter[str] = Counter()
    signal_free_proof_lanes: Counter[str] = Counter()
    compile_failures = []
    compiled = 0
    routes_by_entry_id: Dict[int, Mapping[str, Any]] = {}
    for index, route in enumerate(routes):
        if not isinstance(route, Mapping):
            compile_failures.append({
                "index": index,
                "route_entry_id": None,
                "error": "route_not_an_object",
            })
            continue
        route_id = route.get("route_entry_id")
        route_ids.append(str(route_id))
        try:
            numeric_route_id = int(route_id)
            numeric_route_ids.append(numeric_route_id)
            routes_by_entry_id[numeric_route_id] = route
        except (TypeError, ValueError):
            errors.append("route_index_{}:invalid_route_entry_id".format(index))
        identity = (str(route.get("pcu_id", "")), str(route.get("arm_id", "")))
        if not all(identity):
            errors.append("route_{}:missing_pcu_or_arm".format(route_id))
        arm_routes[identity].append(route)
        target_type = str(route.get("target", {}).get("type", ""))
        target_counts[target_type] += 1
        try:
            runtime_context = load_private_route_context(
                str(path),
                int(route_id),
                preloaded_payload=payload,
                precomputed_sha256=sidecar_sha256,
            )
            unresolved_targets = runtime_context.get(
                "unresolved_capture_target_ids", []
            )
            unresolved_slots = runtime_context.get(
                "unresolved_actor_spawn_slots", []
            )
            if unresolved_targets:
                raise OracleProgramError(
                    "runtime capture geometry unresolved for {}".format(
                        ",".join(map(str, unresolved_targets))
                    )
                )
            if unresolved_slots:
                raise OracleProgramError(
                    "runtime actor spawn geometry unresolved for {}".format(
                        ",".join(map(str, unresolved_slots))
                    )
                )
            validate_actor_setup_program(route, runtime_context)
            result = compile_private_route_static(
                route,
                parking_manoeuvre_registry=payload.get(
                    "parking_manoeuvre_registry"
                ),
            )
            compiled += 1
            resolved_counts[result["resolved_target_type"]] += 1
            if result["signal_free_corridor_method"] is not None:
                signal_free_proof_methods[
                    result["signal_free_corridor_method"]
                ] += 1
                signal_free_proof_lanes[
                    result["signal_free_corridor_lane_id"]
                ] += 1
        except Exception as exc:  # Every compiler failure belongs in the report.
            compile_failures.append({
                "index": index,
                "route_entry_id": route_id,
                "pcu_id": route.get("pcu_id"),
                "arm_id": route.get("arm_id"),
                "target_type": target_type,
                "error": "{}:{}".format(type(exc).__name__, exc),
            })

    duplicate_route_ids = sorted(
        route_id for route_id, count in Counter(route_ids).items() if count != 1
    )
    if duplicate_route_ids:
        errors.append(
            "duplicate_route_entry_ids:{}".format(",".join(duplicate_route_ids))
        )
    if sorted(numeric_route_ids) != list(range(int(expected_routes))):
        errors.append("route_entry_ids_are_not_contiguous_zero_based")
    pcu_ids = {identity[0] for identity in arm_routes if identity[0]}
    if len(pcu_ids) != int(expected_pcus):
        errors.append(
            "pcu_count:{}!=expected:{}".format(len(pcu_ids), expected_pcus)
        )
    declared_pcu_ids = [
        str(pcu.get("id", ""))
        for pcu in pcus if isinstance(pcu, Mapping)
    ]
    if len(pcus) != int(expected_pcus):
        errors.append(
            "sidecar_pcu_count:{}!=expected:{}".format(
                len(pcus), expected_pcus
            )
        )
    if (
        len(set(declared_pcu_ids)) != len(declared_pcu_ids)
        or "" in declared_pcu_ids
    ):
        errors.append("sidecar_pcu_ids_are_missing_or_not_unique")
    if set(declared_pcu_ids) != pcu_ids:
        errors.append("sidecar_pcu_ids_disagree_with_routes")
    if expected_pcus > 0 and expected_arms % expected_pcus == 0:
        expected_arms_per_pcu = expected_arms // expected_pcus
        arms_per_pcu = Counter(identity[0] for identity in arm_routes)
        malformed_pcus = sorted(
            pcu for pcu, count in arms_per_pcu.items()
            if pcu and count != expected_arms_per_pcu
        )
        if malformed_pcus:
            errors.append(
                "pcus_with_wrong_arm_count:{}".format(",".join(malformed_pcus))
            )

        # Both counterfactual arms must start in the same observable scene.
        # The evaluator intentionally uses arm-specific hidden paths after the
        # common spawn, but actor_setup itself may not reveal which arm was
        # selected to a Standardized camera agent.
        representatives: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for (pcu_id, _arm_id), members in arm_routes.items():
            if members:
                representatives[pcu_id].append(members[0])
        for pcu_id, members in sorted(representatives.items()):
            if len(members) <= 1:
                continue
            actor_programs = {
                _canonical(member.get("actor_setup", {})) for member in members
            }
            if len(actor_programs) != 1:
                errors.append(
                    "pcu_{}:actor_setup_differs_across_arms".format(pcu_id)
                )
            # A moving actor's program is expressed as a distance along the
            # private path. Equal JSON alone is insufficient: if the arms
            # diverge before that distance, CARLA would spawn the same actor at
            # two camera-visible positions and disclose the hidden arm.
            reference_actors = members[0].get("actor_setup", {}).get(
                "actors", []
            )
            for spec in reference_actors:
                if not isinstance(spec, Mapping) or spec.get(
                    "relative_spawn_m"
                ) is None:
                    continue
                identifier = str(spec.get("id", "unknown"))
                try:
                    spawn_points = [
                        _path_point(
                            member.get("oracle_path", []),
                            float(spec["relative_spawn_m"]),
                        )
                        for member in members
                    ]
                    if any(point is None for point in spawn_points):
                        raise ValueError("unresolved path point")
                    origin = spawn_points[0]
                    if any(
                        math.sqrt(sum(
                            (float(point[axis]) - float(origin[axis])) ** 2
                            for axis in range(3)
                        )) > 1e-3
                        for point in spawn_points[1:]
                    ):
                        errors.append(
                            "pcu_{}:actor_{}_spawn_differs_across_arms".format(
                                pcu_id, identifier
                            )
                        )
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        "pcu_{}:actor_{}_spawn_invalid".format(
                            pcu_id, identifier
                        )
                    )
            try:
                starts = [member["oracle_path"][0] for member in members]
                origin = starts[0]
                for start in starts[1:]:
                    position_error = math.sqrt(sum(
                        (float(start[axis]) - float(origin[axis])) ** 2
                        for axis in ("x", "y", "z")
                    ))
                    yaw_error = abs(
                        (float(start["yaw_deg"]) - float(origin["yaw_deg"])
                         + 180.0) % 360.0 - 180.0
                    )
                    if position_error > 1e-3 or yaw_error > 1e-3:
                        errors.append(
                            "pcu_{}:counterfactual_spawn_differs".format(pcu_id)
                        )
                        break
            except (KeyError, IndexError, TypeError, ValueError):
                errors.append(
                    "pcu_{}:counterfactual_spawn_invalid".format(pcu_id)
                )
    if len(arm_routes) != int(expected_arms):
        errors.append(
            "arm_count:{}!=expected:{}".format(len(arm_routes), expected_arms)
        )

    arm_target_counts: Counter[str] = Counter()
    signal_free_proof_arms = 0
    for identity, members in sorted(arm_routes.items()):
        if len(members) != int(expected_expressions_per_arm):
            errors.append(
                "arm_{}__{}:expression_count:{}!=expected:{}".format(
                    identity[0], identity[1], len(members),
                    expected_expressions_per_arm,
                )
            )
        paraphrases = []
        instructions = []
        signatures = set()
        for route in members:
            try:
                paraphrases.append(int(route["paraphrase"]))
            except (KeyError, TypeError, ValueError):
                errors.append(
                    "arm_{}__{}:invalid_paraphrase".format(*identity)
                )
            instructions.append(str(route.get("instruction", "")).strip())
            signatures.add(_program_signature(route))
        if sorted(paraphrases) != list(range(expected_expressions_per_arm)):
            errors.append(
                "arm_{}__{}:paraphrases:{}".format(
                    identity[0], identity[1], sorted(paraphrases)
                )
            )
        if any(not instruction for instruction in instructions):
            errors.append("arm_{}__{}:empty_instruction".format(*identity))
        if len(set(instructions)) != expected_expressions_per_arm:
            errors.append(
                "arm_{}__{}:instructions_not_distinct".format(*identity)
            )
        if len(signatures) != 1:
            errors.append(
                "arm_{}__{}:hidden_program_differs_across_paraphrases".format(
                    *identity
                )
            )
        proof_presence = [
            isinstance(route.get("signal_free_corridor_proof"), Mapping)
            for route in members
        ]
        if any(proof_presence) and not all(proof_presence):
            errors.append(
                "arm_{}__{}:signal_free_proof_differs_across_paraphrases"
                .format(*identity)
            )
        if proof_presence and all(proof_presence):
            signal_free_proof_arms += 1
        arm_target_counts[str(members[0].get("target", {}).get("type", ""))] += 1

    if canonical_core_counts:
        compiled_signal_free_routes = sum(signal_free_proof_methods.values())
        if compiled_signal_free_routes != CANONICAL_CORE_SIGNAL_FREE_PROOF_ROUTES:
            errors.append(
                "signal_free_proof_route_count:{}!=expected:{}".format(
                    compiled_signal_free_routes,
                    CANONICAL_CORE_SIGNAL_FREE_PROOF_ROUTES,
                )
            )
        if signal_free_proof_arms != CANONICAL_CORE_SIGNAL_FREE_PROOF_ARMS:
            errors.append(
                "signal_free_proof_arm_count:{}!=expected:{}".format(
                    signal_free_proof_arms,
                    CANONICAL_CORE_SIGNAL_FREE_PROOF_ARMS,
                )
            )

    required = set(required_target_types or [])
    observed = set(arm_target_counts)
    missing_types = sorted(required - observed)
    if missing_types:
        errors.append("missing_target_types:{}".format(",".join(missing_types)))
    if compile_failures:
        errors.append("oracle_compile_failures:{}".format(len(compile_failures)))

    # When auditing the canonical build directory, also pass the public XML
    # through the same identity/spawn checks used by the live evaluator.  A
    # copied standalone sidecar remains auditable, but the report makes that
    # missing optional companion explicit rather than claiming it was checked.
    public_xml_path = path.with_name("routes_standardized.xml")
    public_xml_checked = False
    public_xml_errors = []
    if public_xml_path.is_file():
        public_xml_checked = True
        try:
            xml_routes = ET.parse(public_xml_path).getroot().findall("route")
            if len(xml_routes) != len(routes):
                public_xml_errors.append(
                    "route_count:{}!=sidecar:{}".format(
                        len(xml_routes), len(routes)
                    )
                )
            seen_xml_ids = set()
            for xml_route in xml_routes:
                try:
                    route_id = int(xml_route.attrib["id"])
                except (KeyError, TypeError, ValueError):
                    public_xml_errors.append("route_with_invalid_id")
                    continue
                if route_id in seen_xml_ids:
                    public_xml_errors.append(
                        "route_{}:duplicate_id".format(route_id)
                    )
                seen_xml_ids.add(route_id)
                private_route = routes_by_entry_id.get(route_id)
                if private_route is None:
                    public_xml_errors.append(
                        "route_{}:missing_private_entry".format(route_id)
                    )
                    continue
                metadata_element = xml_route.find("simlingo_metadata")
                metadata = (
                    dict(metadata_element.attrib)
                    if metadata_element is not None
                    else {}
                )
                forbidden_identity = sorted(
                    set(metadata).intersection({
                        "pcu_id", "arm_id", "paraphrase", "route_entry_id",
                        "seed", "l2d_ref",
                    })
                )
                if forbidden_identity:
                    public_xml_errors.append(
                        "route_{}:public_metadata_exposes:{}".format(
                            route_id, ",".join(forbidden_identity)
                        )
                    )
                if not str(metadata.get("benchmark_type", "")).startswith(
                    "language2drive_v1"
                ):
                    public_xml_errors.append(
                        "route_{}:wrong_benchmark_type".format(route_id)
                    )
                if metadata.get("track") != "standardized":
                    public_xml_errors.append(
                        "route_{}:wrong_track".format(route_id)
                    )
                instruction_element = xml_route.find("instruction")
                instruction = (
                    (instruction_element.text or "").strip()
                    if instruction_element is not None
                    else ""
                )
                if metadata.get("instruction") != instruction:
                    public_xml_errors.append(
                        "route_{}:instruction_sources_disagree".format(route_id)
                    )
                identity_errors = public_private_consistency_errors(
                    {"route": private_route},
                    {
                        "route_entry_id": route_id,
                        "town": xml_route.attrib.get("town"),
                        "instruction": instruction,
                    },
                )
                public_xml_errors.extend(
                    "route_{}:{}".format(route_id, error)
                    for error in identity_errors
                )
                positions = [
                    {
                        "x": position.attrib.get("x"),
                        "y": position.attrib.get("y"),
                        "z": position.attrib.get("z"),
                    }
                    for position in xml_route.findall("./waypoints/position")
                ]
                if len(positions) < 2:
                    public_xml_errors.append(
                        "route_{}:coarse_envelope_has_fewer_than_two_points".format(
                            route_id
                        )
                    )
                public_xml_errors.extend(
                    "route_{}:{}".format(route_id, error)
                    for error in private_public_start_consistency_errors(
                        {"route": private_route}, positions
                    )
                )
        except (OSError, ET.ParseError) as exc:
            public_xml_errors.append(
                "parse_failed:{}:{}".format(type(exc).__name__, exc)
            )
    if public_xml_errors:
        errors.append(
            "public_standardized_xml_errors:{}".format(len(public_xml_errors))
        )

    return {
        "schema_version": "language2drive.oracle_static_audit.v1",
        "success": not errors,
        "private_sidecar_schema_version": schema_version,
        "sidecar_path": str(path),
        "sidecar_sha256": sidecar_sha256,
        "source_files_present": source_files_result["present"],
        "source_file_count": source_files_result["count"],
        "verified_source_file_count": source_files_result["verified"],
        "source_file_errors": source_file_errors,
        "public_standardized_xml_path": (
            str(public_xml_path) if public_xml_checked else None
        ),
        "public_standardized_xml_checked": public_xml_checked,
        "public_standardized_xml_errors": public_xml_errors,
        "observed_routes": len(routes),
        "expected_routes": int(expected_routes),
        "compiled_routes": compiled,
        "observed_arms": len(arm_routes),
        "observed_pcus": len(pcu_ids),
        "expected_pcus": int(expected_pcus),
        "expected_arms": int(expected_arms),
        "expected_expressions_per_arm": int(expected_expressions_per_arm),
        "declared_target_type_route_counts": dict(sorted(target_counts.items())),
        "declared_target_type_arm_counts": dict(sorted(arm_target_counts.items())),
        "resolved_target_type_route_counts": dict(sorted(resolved_counts.items())),
        "signal_free_corridor_proof_route_counts": dict(
            sorted(signal_free_proof_methods.items())
        ),
        "signal_free_corridor_lane_route_counts": dict(
            sorted(signal_free_proof_lanes.items())
        ),
        "signal_free_corridor_proof_arms": signal_free_proof_arms,
        "expected_signal_free_corridor_proof_routes": (
            CANONICAL_CORE_SIGNAL_FREE_PROOF_ROUTES
            if canonical_core_counts else None
        ),
        "expected_signal_free_corridor_proof_arms": (
            CANONICAL_CORE_SIGNAL_FREE_PROOF_ARMS
            if canonical_core_counts else None
        ),
        "required_target_types": sorted(required),
        "missing_target_types": missing_types,
        "compile_failures": compile_failures,
        "errors": errors,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile and audit all Language2Drive private oracle routes"
    )
    parser.add_argument("sidecar", help="Path to sidecar.private.json")
    parser.add_argument("--expected-routes", type=int, default=768)
    parser.add_argument("--expected-pcus", type=int, default=128)
    parser.add_argument("--expected-arms", type=int, default=256)
    parser.add_argument("--expressions-per-arm", type=int, default=3)
    parser.add_argument(
        "--required-target-types",
        default=",".join(sorted(CORE_TARGET_TYPES)),
        help=(
            "Comma-separated target types that must be covered; use an empty "
            "value for count/schema/compiler-only supplementary audits"
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = audit_private_sidecar(
        args.sidecar,
        expected_routes=args.expected_routes,
        expected_pcus=args.expected_pcus,
        expected_arms=args.expected_arms,
        expected_expressions_per_arm=args.expressions_per_arm,
        required_target_types={
            value.strip()
            for value in args.required_target_types.split(",")
            if value.strip()
        },
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(".{}.tmp".format(output.name))
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
