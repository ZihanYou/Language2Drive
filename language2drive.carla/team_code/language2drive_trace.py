"""Evaluator-only private-sidecar loading and auditable trace postprocessing.

Nothing in this module is imported by the model.  The leaderboard owns the
private sidecar, passes only a minimal capture specification to its logger, and
scores the completed trace after the agent has stopped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from team_code.language2drive_signal_free_proof import (
    SignalFreeCorridorProofError,
    validate_signal_free_corridor_proof,
)
from team_code.language2drive_parking_manoeuvre import (
    ParkingManoeuvreProofError,
    resolve_parking_target,
    validate_parking_manoeuvre,
    validate_parking_registry_source_files,
)


ATOMIC_PREDICATES = {
    "enter_gate",
    "inside_region",
    "vehicle_footprint_inside_slot",
    "lane_relation",
    "speed_band",
    "headway_band",
    "passing_side",
    "actor_order",
    "progress",
    "dwell",
    "safe_legal",
}

PRIVATE_SIDECAR_SCHEMA = "language2drive.private_sidecar.v1"


class PrivateSidecarError(ValueError):
    """The evaluator-only sidecar is absent, ambiguous, or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_signature(value: Any) -> str:
    """Canonicalize numeric spelling before comparing geometry provenance."""

    def normalize(item: Any) -> Any:
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, dict):
            return {str(key): normalize(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [normalize(nested) for nested in item]
        return item

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":")
    )


def _validate_parking_capture_target(target: Mapping[str, Any]) -> None:
    """Enforce the frozen v1 exact-parking capture tolerances."""

    bounds = (
        ("tolerance_m", 0.2, 0.0, 0.2),
        ("heading_tolerance_deg", 15.0, 0.0, 15.0),
        ("speed_max_mps", 0.2, 0.0, 0.2),
    )
    for field, default, minimum, maximum in bounds:
        raw = target.get(field, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise PrivateSidecarError(
                "parking target {} {} must be numeric".format(
                    target.get("id"), field
                )
            ) from exc
        if (
            isinstance(raw, bool)
            or not math.isfinite(value)
            or value < minimum
            or value > maximum
        ):
            raise PrivateSidecarError(
                "parking target {} {}={} is outside [{}, {}]".format(
                    target.get("id"), field, raw, minimum, maximum
                )
            )


def _walk_targets(value: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("type"), str):
            yield value
        for nested in value.values():
            yield from _walk_targets(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_targets(nested)


def _walk_predicates(expression: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if not isinstance(expression, dict):
        return
    for name, body in expression.items():
        if name in ATOMIC_PREDICATES:
            if not isinstance(body, dict):
                raise PrivateSidecarError(
                    "contract predicate {} must be an object".format(name)
                )
            yield name, body
        if isinstance(body, (dict, list)):
            if isinstance(body, dict):
                yield from _walk_predicates(body)
            else:
                for item in body:
                    yield from _walk_predicates(item)


def _validate_speed_feasibility_cap(route: Mapping[str, Any]) -> None:
    """Validate the private conservative execution cap for core speed arms.

    The cap is not a legal-speed assertion.  It records the generator's fixed
    CARLA-default feasibility bound; the runtime agent still checks the live
    vehicle limit authoritatively at setup and while the band is active.
    """

    bands = [
        body
        for name, body in _walk_predicates(route.get("contract", {}))
        if name == "speed_band"
    ]
    if not bands:
        return
    layer = str(route.get("layer", ""))
    proof = route.get("speed_feasibility_cap_proof")
    if layer in {"A", "B", "C", "D"} and not isinstance(proof, Mapping):
        raise PrivateSidecarError(
            "core speed route requires speed_feasibility_cap_proof"
        )
    if proof is None and layer == "Safety":
        return
    if not isinstance(proof, Mapping):
        raise PrivateSidecarError(
            "speed_feasibility_cap_proof must be an object"
        )
    expected = {"method", "cap_mps", "control_margin_mps"}
    if set(proof) != expected:
        raise PrivateSidecarError(
            "speed_feasibility_cap_proof keys must be exactly {}".format(
                sorted(expected)
            )
        )
    if proof.get("method") != "carla_default_limit_conservative_cap_v1":
        raise PrivateSidecarError(
            "speed_feasibility_cap_proof has unsupported method"
        )
    try:
        raw_cap = proof["cap_mps"]
        raw_margin = proof["control_margin_mps"]
        raw_legal = route["legal_speed_mps"]
        if any(isinstance(value, bool) for value in (
            raw_cap, raw_margin, raw_legal
        )):
            raise TypeError("boolean speed feasibility value")
        cap = float(raw_cap)
        margin = float(raw_margin)
        legal = float(raw_legal)
        maxima = []
        for body in bands:
            raw_maximum = body["max_mps"]
            if isinstance(raw_maximum, bool):
                raise TypeError("boolean speed-band maximum")
            maxima.append(float(raw_maximum))
    except (KeyError, TypeError, ValueError) as exc:
        raise PrivateSidecarError(
            "speed feasibility proof/band numeric fields are invalid"
        ) from exc
    if not all(math.isfinite(value) for value in (
        cap, margin, legal, *maxima
    )):
        raise PrivateSidecarError(
            "speed feasibility proof/band values must be finite"
        )
    if (
        abs(cap - 8.333333) > 1e-9
        or legal <= 0.0
        or abs(margin - 0.35) > 1e-9
    ):
        raise PrivateSidecarError(
            "speed feasibility proof requires cap_mps=8.333333, positive "
            "legal limit, and control_margin_mps=0.35"
        )
    ceiling = min(cap, legal)
    if any(maximum + margin > ceiling + 1e-9 for maximum in maxima):
        raise PrivateSidecarError(
            "speed band plus control margin exceeds XODR/conservative cap"
        )


def _validate_signal_free_speed_corridor(route: Mapping[str, Any]) -> None:
    """Run the same exact-path proof validator used by the oracle compiler."""

    try:
        validate_signal_free_corridor_proof(route)
    except SignalFreeCorridorProofError as exc:
        raise PrivateSidecarError(str(exc)) from exc


def _validate_exact_parking_manoeuvre(
    route: Mapping[str, Any], registry: Any, source_files: Any,
) -> None:
    """Validate every v4/v5 branch; only the selected branch binds its route."""

    try:
        selected = resolve_parking_target(route)
        parking_targets = [
            candidate
            for candidate in _walk_targets(route.get("target"))
            if candidate.get("type") == "parking_slot"
        ]
        layer = str(route.get("layer", ""))
        # The independent Safety engineering set may intentionally carry only
        # a refusal/fallback geometry and no privileged manoeuvre witness.  If
        # it does carry any parking witness, however, registry/source closure is
        # mandatory exactly as it is for every A--D branch.
        requires_registry = any(
            layer in {"A", "B", "C", "D"}
            or candidate.get("reverse_manoeuvre") is not None
            for candidate in parking_targets
        )
        if requires_registry:
            validate_parking_registry_source_files(
                registry, source_files, str(route.get("town", ""))
            )
        for candidate in parking_targets:
            validate_parking_manoeuvre(
                route,
                candidate,
                registry=registry,
                require_oracle_suffix=(candidate is selected),
                verify_sources=True,
            )
    except ParkingManoeuvreProofError as exc:
        raise PrivateSidecarError(str(exc)) from exc


def _static_reference_values(expression: Any) -> Dict[str, str]:
    """Return evaluator-known bind_reference constants from a contract.

    Fixed-layout v1 uses ``bind_reference.value`` for delayed references.  The
    value is private evaluator data, so resolving it here only influences the
    capture allow-list; it is never copied into the public agent context.
    """

    values: Dict[str, str] = {}
    if isinstance(expression, dict):
        binding = expression.get("bind_reference")
        if isinstance(binding, dict):
            reference = binding.get("reference")
            value = binding.get("value")
            if isinstance(reference, str) and isinstance(value, str):
                values[reference] = value
        for nested in expression.values():
            values.update(_static_reference_values(nested))
    elif isinstance(expression, list):
        for nested in expression:
            values.update(_static_reference_values(nested))
    return values


def _resolved_identifier(value: Any, references: Mapping[str, str]) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and set(value) == {"$ref"}:
        reference = value.get("$ref")
        if isinstance(reference, str):
            return references.get(reference)
    return None


def contract_capture_requirements(private_context: Mapping[str, Any]) -> Dict[str, Any]:
    route = private_context.get("route", {})
    expressions = [route.get("contract", {}), route.get("opposite_contract", {})]
    predicates = [
        item
        for expression in expressions
        for item in _walk_predicates(expression)
    ]
    references: Dict[str, str] = {}
    for expression in expressions:
        references.update(_static_reference_values(expression))
    names = sorted({name for name, _ in predicates})
    actor_ids = sorted(
        {
            str(body["actor_id"])
            for _, body in predicates
            if body.get("actor_id") is not None
        }
    )
    occupancy_slots = set()
    # actor_order encodes the semantic actor or occupancy slot after a colon.
    for name, body in predicates:
        if name == "actor_order" and isinstance(body.get("id"), str):
            identifier = body["id"]
            if identifier.startswith("occupied:"):
                occupancy_slots.add(identifier.split(":", 1)[1])
            elif ":" in identifier:
                actor_ids.append(identifier.split(":", 1)[1])
    setup = route.get("actor_setup", {}) if isinstance(route, dict) else {}
    for actor in setup.get("actors", []) if isinstance(setup, dict) else []:
        if (
            isinstance(actor, dict)
            and actor.get("id") is not None
            and str(actor.get("slot")) in occupancy_slots
        ):
            actor_ids.append(str(actor["id"]))
    actor_ids = sorted(set(actor_ids))
    gate_ids = sorted(
        {
            identifier
            for name, body in predicates
            if name == "enter_gate"
            for identifier in [_resolved_identifier(body.get("id"), references)]
            if identifier is not None
        }
    )
    dwell_ids = sorted(
        {
            str(body["key"])
            for name, body in predicates
            if name == "dwell" and body.get("key") is not None
        }
    )
    return {
        "predicates": names,
        "actor_ids": actor_ids,
        "gate_ids": gate_ids,
        "dwell_ids": dwell_ids,
        "occupancy_slots": sorted(occupancy_slots),
    }


def load_private_route_context(
    sidecar_path: str,
    route_entry_id: int,
    *,
    preloaded_payload: Optional[Mapping[str, Any]] = None,
    precomputed_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one private route without copying private values into public context."""

    path = Path(sidecar_path).expanduser().resolve()
    if preloaded_payload is None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateSidecarError(
                "cannot read private sidecar {}: {}".format(path, exc)
            ) from exc
    else:
        # The release compiler uses the exact runtime loader for all 768
        # entries without reparsing a multi-megabyte JSON document each time.
        payload = preloaded_payload
    if not isinstance(payload, dict):
        raise PrivateSidecarError("private sidecar must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != PRIVATE_SIDECAR_SCHEMA:
        raise PrivateSidecarError(
            "unsupported private sidecar schema {!r}; expected {!r}".format(
                schema_version, PRIVATE_SIDECAR_SCHEMA
            )
        )
    routes = payload.get("routes")
    pcus = payload.get("pcus")
    if not isinstance(routes, list) or not isinstance(pcus, list):
        raise PrivateSidecarError("private sidecar requires routes and pcus lists")
    matches = [
        route
        for route in routes
        if isinstance(route, dict)
        and str(route.get("route_entry_id")) == str(int(route_entry_id))
    ]
    if len(matches) != 1:
        raise PrivateSidecarError(
            "route_entry_id {} matched {} private routes".format(
                route_entry_id, len(matches)
            )
        )
    route = matches[0]
    required_route_fields = {
        "route_entry_id", "pcu_id", "arm_id", "paraphrase", "instruction",
        "town", "split", "layer", "contract", "opposite_contract",
        "target", "actor_setup", "deadline_seconds", "oracle_path",
    }
    missing_route_fields = sorted(required_route_fields - set(route))
    if missing_route_fields:
        raise PrivateSidecarError(
            "private route {} is missing {}".format(
                route_entry_id, ", ".join(missing_route_fields)
            )
        )
    # Cheap capture-bound validation precedes the larger executable proof so
    # a target cannot hide an expanded scoring/legality tolerance behind a
    # second, unrelated schema failure.
    for candidate in _walk_targets(route.get("target")):
        if candidate.get("type") == "parking_slot":
            _validate_parking_capture_target(candidate)
    _validate_speed_feasibility_cap(route)
    _validate_signal_free_speed_corridor(route)
    parking_registry = payload.get("parking_manoeuvre_registry")
    source_files = payload.get("source_files")
    _validate_exact_parking_manoeuvre(route, parking_registry, source_files)
    pcu_matches = [
        pcu
        for pcu in pcus
        if isinstance(pcu, dict) and pcu.get("id") == route.get("pcu_id")
    ]
    if len(pcu_matches) != 1:
        raise PrivateSidecarError(
            "private route {} has no unique PCU".format(route_entry_id)
        )
    pcu = pcu_matches[0]
    for identity_field in ("id", "town", "split", "layer"):
        route_field = "pcu_id" if identity_field == "id" else identity_field
        if str(pcu.get(identity_field)) != str(route.get(route_field)):
            raise PrivateSidecarError(
                "private route {} {} disagrees with PCU".format(
                    route_entry_id, route_field
                )
            )

    context: Dict[str, Any] = {
        "route": route,
        "pcu": {
            key: pcu.get(key)
            for key in ("id", "layer", "category", "split", "town", "map_family")
        },
        "sidecar_path": str(path),
        "sidecar_sha256": (
            str(precomputed_sha256)
            if precomputed_sha256 is not None
            else _sha256(path)
        ),
        "runtime_required": True,
        "parking_manoeuvre_registry": parking_registry,
        "parking_source_files": source_files,
    }
    context["requirements"] = contract_capture_requirements(context)

    # Build a sidecar-wide geometry index, then retain only targets referenced
    # by this route's own/opposite contracts.  This lets Layer D reuse already
    # annotated Layer C gates without exposing unrelated hidden targets to the
    # route logger.
    capture_targets = []
    global_targets = []
    geometry_registry = payload.get("geometry_registry", {})
    if geometry_registry is not None and not isinstance(geometry_registry, dict):
        raise PrivateSidecarError("geometry_registry must be an object")
    for registry_id, target in (geometry_registry or {}).items():
        if not isinstance(target, dict):
            raise PrivateSidecarError(
                "geometry_registry entry {} must be an object".format(registry_id)
            )
        if str(target.get("id")) != str(registry_id):
            raise PrivateSidecarError(
                "geometry_registry key {} disagrees with target id {!r}".format(
                    registry_id, target.get("id")
                )
            )
        target_type = target.get("type")
        if target_type == "gate" and not (
            isinstance(target.get("polygon"), list)
            or (
                isinstance(target.get("center"), dict)
                and target.get("radius_m") is not None
            )
        ):
            raise PrivateSidecarError(
                "geometry_registry entry {} has no gate geometry".format(
                    registry_id
                )
            )
        if target_type == "parking_slot" and not isinstance(
            target.get("polygon"), list
        ):
            raise PrivateSidecarError(
                "geometry_registry entry {} has no parking polygon".format(
                    registry_id
                )
            )
        capture_targets.append(target)
        global_targets.append(target)
    for arm in pcu.get("arms", []):
        if isinstance(arm, dict):
            capture_targets.extend(_walk_targets(arm.get("target")))
    capture_targets.extend(_walk_targets(route.get("target")))
    for candidate_pcu in pcus:
        if not isinstance(candidate_pcu, dict):
            continue
        for arm in candidate_pcu.get("arms", []):
            if isinstance(arm, dict):
                global_targets.extend(_walk_targets(arm.get("target")))
    global_targets.extend(capture_targets)

    required_target_ids = set(context["requirements"]["gate_ids"])
    required_target_ids.update(context["requirements"]["dwell_ids"])
    pcu_source_refs = {
        str(value) for value in pcu.get("source_refs", [])
        if isinstance(value, str) and value
    }
    for identifier in sorted(required_target_ids):
        registry_target = (geometry_registry or {}).get(identifier)
        if registry_target is None:
            continue
        source_ref = registry_target.get("source_ref")
        source_sha256 = str(registry_target.get("source_sha256", ""))
        geometry_version = registry_target.get("geometry_version")
        if (
            not isinstance(source_ref, str)
            or not source_ref
            or source_ref not in pcu_source_refs
        ):
            raise PrivateSidecarError(
                "geometry_registry entry {} is not provenance-bound to PCU {}".format(
                    identifier, pcu.get("id")
                )
            )
        if (
            len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise PrivateSidecarError(
                "geometry_registry entry {} has invalid source_sha256".format(
                    identifier
                )
            )
        if not isinstance(geometry_version, str) or not geometry_version:
            raise PrivateSidecarError(
                "geometry_registry entry {} has no geometry_version".format(
                    identifier
                )
            )
    geometries: Dict[Tuple[str, str], Dict[str, Any]] = {}
    conflicts: Dict[str, set[str]] = {}
    for target in global_targets:
        identifier = target.get("id")
        target_type = target.get("type")
        if not isinstance(identifier, str) or identifier not in required_target_ids:
            continue
        if target_type == "gate":
            if not (
                isinstance(target.get("polygon"), list)
                or (
                    isinstance(target.get("center"), dict)
                    and target.get("radius_m") is not None
                )
            ):
                continue
            geometry = {
                key: target.get(key)
                for key in ("polygon", "center", "radius_m")
                if target.get(key) is not None
            }
        elif target_type == "parking_slot":
            if not isinstance(target.get("polygon"), list):
                continue
            _validate_parking_capture_target(target)
            geometry = {
                key: target.get(key)
                for key in (
                    "polygon", "heading_deg", "heading_tolerance_deg",
                    "speed_max_mps",
                )
                if target.get(key) is not None
            }
        else:
            continue
        signature = _geometry_signature(geometry)
        conflicts.setdefault(identifier, set()).add(signature)
    for target in capture_targets:
        identifier = target.get("id")
        target_type = target.get("type")
        if not isinstance(identifier, str) or identifier not in required_target_ids:
            continue
        if target_type == "gate":
            if not (
                isinstance(target.get("polygon"), list)
                or (
                    isinstance(target.get("center"), dict)
                    and target.get("radius_m") is not None
                )
            ):
                continue
            geometry = {
                key: target.get(key)
                for key in ("polygon", "center", "radius_m")
                if target.get(key) is not None
            }
        elif target_type == "parking_slot":
            if not isinstance(target.get("polygon"), list):
                continue
            _validate_parking_capture_target(target)
            geometry = {
                key: target.get(key)
                for key in (
                    "polygon", "heading_deg", "heading_tolerance_deg",
                    "speed_max_mps",
                )
                if target.get(key) is not None
            }
        else:
            continue
        geometries[(identifier, _geometry_signature(geometry))] = target
    ambiguous = sorted(
        identifier for identifier, values in conflicts.items() if len(values) > 1
    )
    if ambiguous:
        raise PrivateSidecarError(
            "conflicting target geometry for {}".format(", ".join(ambiguous))
        )
    context["capture_targets"] = [
        target
        for (identifier, _), target in geometries.items()
        if identifier in required_target_ids
    ]
    captured_target_ids = {
        str(target["id"])
        for target in context["capture_targets"]
        if target.get("id") is not None
    }
    context["unresolved_capture_target_ids"] = sorted(
        required_target_ids - captured_target_ids
    )

    slot_geometries: Dict[str, Dict[str, Any]] = {}
    slot_conflicts: Dict[str, set[str]] = {}
    for target in global_targets:
        if target.get("type") != "parking_slot":
            continue
        slot_number = target.get("slot_number")
        polygon = target.get("polygon")
        if not isinstance(slot_number, str) or not isinstance(polygon, list):
            continue
        geometry = {
            key: target.get(key)
            for key in ("polygon", "heading_deg", "heading_tolerance_deg")
            if target.get(key) is not None
        }
        signature = _geometry_signature(geometry)
        slot_conflicts.setdefault(slot_number, set()).add(signature)
    for target in capture_targets:
        if target.get("type") != "parking_slot":
            continue
        slot_number = target.get("slot_number")
        polygon = target.get("polygon")
        if isinstance(slot_number, str) and isinstance(polygon, list):
            _validate_parking_capture_target(target)
            slot_geometries[slot_number] = target
    required_slots = {
        str(actor["slot"])
        for actor in route.get("actor_setup", {}).get("actors", [])
        if isinstance(actor, dict) and actor.get("slot") is not None
    }
    ambiguous_slots = sorted(
        slot for slot in required_slots if len(slot_conflicts.get(slot, set())) > 1
    )
    if ambiguous_slots:
        raise PrivateSidecarError(
            "conflicting parking geometry for {}".format(
                ", ".join(ambiguous_slots)
            )
        )
    context["actor_spawn_targets"] = {
        slot: slot_geometries[slot]
        for slot in sorted(required_slots)
        if slot in slot_geometries
    }
    context["unresolved_actor_spawn_slots"] = sorted(
        required_slots - set(context["actor_spawn_targets"])
    )
    return context


def public_private_consistency_errors(
    private_context: Mapping[str, Any], route_metadata: Mapping[str, Any]
) -> list[str]:
    """Compare XML-visible identity to the private lookup without exposing it."""

    route = private_context.get("route", {})
    errors = []
    mapping = {
        "route_entry_id": route_metadata.get("route_entry_id"),
        "pcu_id": route_metadata.get("pcu_id"),
        "arm_id": route_metadata.get("arm_id"),
        "paraphrase": route_metadata.get("paraphrase"),
        "town": route_metadata.get("town"),
        "instruction": route_metadata.get("instruction"),
    }
    for key, public_value in mapping.items():
        private_value = route.get(key)
        if public_value is not None and private_value is None:
            errors.append(
                "{} missing from private route".format(key)
            )
        elif (
            public_value is not None
            and str(public_value) != str(private_value)
        ):
            errors.append(
                "{} mismatch: XML={!r}, sidecar={!r}".format(
                    key, public_value, private_value
                )
            )
    return errors


def private_public_start_consistency_errors(
    private_context: Mapping[str, Any],
    xml_waypoints: Iterable[Mapping[str, Any]],
    *,
    horizontal_tolerance_m: float = 0.25,
    vertical_tolerance_m: float = 0.25,
) -> list[str]:
    """Require the hidden witness to start at the public XML spawn.

    The Standardized XML deliberately ends at a shared coarse envelope rather
    than the arm-specific goal, so only the first point is comparable.  This
    check prevents a mismatched private sidecar from silently changing the
    initial camera observation or ego pose used to construct the scenario.
    Small projection/serialization differences are tolerated; the current v1
    build pipeline stays below 0.25 m in both axes.
    """

    errors: list[str] = []
    route = private_context.get("route", {})
    oracle_path = route.get("oracle_path") if isinstance(route, Mapping) else None
    public_path = list(xml_waypoints or [])
    if not public_path:
        return ["public XML route has no spawn waypoint"]
    if not isinstance(oracle_path, list) or not oracle_path:
        return ["private oracle_path has no spawn waypoint"]

    public_start = public_path[0]
    private_start = oracle_path[0]
    if not isinstance(public_start, Mapping) or not isinstance(
        private_start, Mapping
    ):
        return ["public/private spawn waypoint must be an object"]
    try:
        public_xyz = tuple(float(public_start[axis]) for axis in ("x", "y", "z"))
        private_xyz = tuple(float(private_start[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError):
        return ["public/private spawn waypoint requires finite x/y/z"]
    if not all(math.isfinite(value) for value in public_xyz + private_xyz):
        return ["public/private spawn waypoint requires finite x/y/z"]

    horizontal_error = math.hypot(
        public_xyz[0] - private_xyz[0], public_xyz[1] - private_xyz[1]
    )
    vertical_error = abs(public_xyz[2] - private_xyz[2])
    if horizontal_error > float(horizontal_tolerance_m):
        errors.append(
            "spawn horizontal mismatch: XML/private={:.6f}m>{:.6f}m".format(
                horizontal_error, float(horizontal_tolerance_m)
            )
        )
    if vertical_error > float(vertical_tolerance_m):
        errors.append(
            "spawn vertical mismatch: XML/private={:.6f}m>{:.6f}m".format(
                vertical_error, float(vertical_tolerance_m)
            )
        )
    return errors


def expected_run_id(
    track: str, route_metadata: Mapping[str, Any], seed: int
) -> Optional[str]:
    try:
        pcu = str(route_metadata["pcu_id"])
        arm = str(route_metadata["arm_id"])
        paraphrase = int(route_metadata["paraphrase"])
    except (KeyError, TypeError, ValueError):
        return None
    if track not in {"standardized", "native"}:
        return None
    return "{}__{}__{}__p{}__s{}".format(
        track, pcu, arm, paraphrase, int(seed)
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    "{} row {} is not an object".format(path.name, line_number)
                )
            rows.append(value)
    return rows


def _trace_cadence_errors(
    rows: list[Mapping[str, Any]], *, expected_dt: float = 0.05
) -> list[str]:
    """Prove that a private rollout is the required consecutive 20 Hz trace."""

    if len(rows) < 2:
        return ["trace_has_fewer_than_two_frames"]
    errors = []
    previous_frame = None
    previous_time = None
    for index, row in enumerate(rows):
        frame = row.get("frame")
        sim_time = row.get("sim_time")
        if not isinstance(frame, int) or isinstance(frame, bool):
            errors.append("row_{}:frame_not_integer".format(index))
            continue
        if (
            not isinstance(sim_time, (int, float))
            or isinstance(sim_time, bool)
            or not math.isfinite(float(sim_time))
        ):
            errors.append("row_{}:sim_time_not_finite".format(index))
            continue
        if previous_frame is not None and frame != previous_frame + 1:
            errors.append(
                "row_{}:frame_gap:{}->{}".format(
                    index, previous_frame, frame
                )
            )
        if previous_time is not None:
            delta = float(sim_time) - previous_time
            if abs(delta - float(expected_dt)) > 1e-3:
                errors.append(
                    "row_{}:sim_time_delta:{:.6f}".format(index, delta)
                )
        previous_frame = frame
        previous_time = float(sim_time)
        if len(errors) >= 20:
            errors.append("additional_cadence_errors_truncated")
            break
    return errors


def _position(row: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    ego = row.get("ego", {})
    if not isinstance(ego, dict):
        return None
    location = ego.get("location")
    transform = ego.get("transform", {})
    if not isinstance(location, dict) and isinstance(transform, dict):
        location = transform.get("location")
    if not isinstance(location, dict):
        return None
    try:
        return float(location["x"]), float(location["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _project_progress(
    point: Tuple[float, float], path: list[Mapping[str, Any]]
) -> Optional[float]:
    if len(path) < 2:
        return None
    points = []
    try:
        points = [(float(item["x"]), float(item["y"])) for item in path]
    except (KeyError, TypeError, ValueError):
        return None
    segment_lengths = [
        math.hypot(bx - ax, by - ay)
        for (ax, ay), (bx, by) in zip(points, points[1:])
    ]
    total = sum(segment_lengths)
    if total <= 1e-6:
        return None
    best_distance, best_along = math.inf, 0.0
    prefix = 0.0
    px, py = point
    for index, ((ax, ay), (bx, by)) in enumerate(zip(points, points[1:])):
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        ratio = 0.0 if denominator == 0 else (
            (px - ax) * dx + (py - ay) * dy
        ) / denominator
        ratio = max(0.0, min(1.0, ratio))
        qx, qy = ax + ratio * dx, ay + ratio * dy
        distance = math.hypot(px - qx, py - qy)
        if distance < best_distance:
            best_distance = distance
            best_along = prefix + ratio * segment_lengths[index]
        prefix += segment_lengths[index]
    return max(0.0, min(1.0, best_along / total))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(".{}.tmp".format(path.name))
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(".{}.tmp".format(path.name))
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class Language2DriveTracePostprocessor:
    """Enrich a completed trace with private progress and issue a fail-closed verdict."""

    def __init__(
        self,
        private_context: Mapping[str, Any],
        language2drive_root: str,
        *,
        stability_seconds: float = 2.0,
        progress_minimum: float = 0.05,
    ) -> None:
        self.private_context = dict(private_context)
        self.language2drive_root = Path(language2drive_root).expanduser().resolve()
        self.stability_seconds = float(stability_seconds)
        self.progress_minimum = float(progress_minimum)
        if self.stability_seconds < 0 or self.progress_minimum < 0:
            raise ValueError(
                "stability_seconds and progress_minimum must be non-negative"
            )

    def _unsupported_requirements(
        self, report: Mapping[str, Any]
    ) -> list[str]:
        requirements = self.private_context.get("requirements", {})
        predicates = set(requirements.get("predicates", []))
        unsupported = []
        resolved_gates = set(report.get("resolved_gate_ids", []))
        for identifier in requirements.get("gate_ids", []):
            if identifier not in resolved_gates:
                unsupported.append("gate:{}:geometry_or_capture_missing".format(identifier))
        resolved_actors = set(report.get("resolved_actor_ids", []))
        for identifier in requirements.get("actor_ids", []):
            if identifier not in resolved_actors:
                unsupported.append("actor:{}:not_uniquely_observed".format(identifier))
        resolved_dwells = set(report.get("resolved_dwell_ids", []))
        for identifier in requirements.get("dwell_ids", []):
            if identifier not in resolved_dwells:
                unsupported.append("dwell:{}:slot_geometry_missing".format(identifier))
        if "lane_relation" in predicates and not report.get("lane_samples", 0):
            unsupported.append("lane_relation:carla_waypoint_unavailable")
        if "passing_side" in predicates and not report.get("passing_side_supported", False):
            unsupported.append("passing_side:no_authoritative_hook")
        # safe_legal is part of the global success conjunction even when the
        # arm-specific expression does not spell it out.
        support = report.get("criterion_event_support", {})
        for event_name in ("collision", "off_road", "red_light"):
            if not support.get(event_name, False):
                unsupported.append(
                    "safe_legal:{}:criterion_missing".format(event_name)
                )
        if (
            self.private_context.get("runtime_required")
            and not report.get("map_legality_samples", 0)
        ):
            unsupported.append("safe_legal:map_lane_legality_unavailable")
        if not report.get("progress_supported", False):
            unsupported.append("progress:oracle_path_projection_unavailable")
        if report.get("logger_error_count", 0):
            unsupported.append("logger:frame_or_io_error")
        if not report.get("lane_samples", 0):
            unsupported.append("ego:road_lane_s_unavailable")
        if not report.get("traffic_light_samples", 0):
            unsupported.append("ego:traffic_light_state_unavailable")
        if not report.get("controller_source_samples", 0):
            unsupported.append("controller:source_unavailable")
        elif report.get("unknown_controller_source_samples", 0):
            unsupported.append("controller:unknown_source_observed")
        if self.private_context.get("runtime_required"):
            runtime = report.get("evaluator_runtime", {})
            if not runtime.get("setup_complete", False):
                unsupported.append("actor_runtime:setup_incomplete")
            if not runtime.get("cleanup_complete", False):
                unsupported.append("actor_runtime:cleanup_incomplete")
            if runtime.get("failure_reason"):
                unsupported.append("actor_runtime:failure")
            if not report.get("trace_cadence_complete", False):
                unsupported.append("trace:non_consecutive_20hz_cadence")
        return sorted(set(unsupported))

    def __call__(
        self, save_dir: str, capture_report: Mapping[str, Any]
    ) -> Dict[str, Any]:
        root = Path(save_dir)
        trace_path = root / "trace.jsonl"
        route = self.private_context["route"]
        result: Dict[str, Any]
        try:
            trace = _read_jsonl(trace_path)
            oracle_path = route.get("oracle_path", [])
            progress_supported = False
            start_time = float(trace[0]["sim_time"]) if trace else 0.0
            target = route.get("target", {})
            for row in trace:
                if isinstance(target, dict) and target.get("type") == "hold":
                    seconds = float(target.get("seconds", 0.0))
                    progress = (
                        None
                        if seconds <= 0
                        else max(
                            0.0,
                            min(
                                1.0,
                                (float(row["sim_time"]) - start_time) / seconds,
                            ),
                        )
                    )
                else:
                    position = _position(row)
                    progress = (
                        None
                        if position is None
                        else _project_progress(position, oracle_path)
                    )
                if progress is not None:
                    row["progress"] = progress
                    progress_supported = True
            report = dict(capture_report)
            report["progress_supported"] = progress_supported
            cadence_errors = (
                _trace_cadence_errors(trace)
                if self.private_context.get("runtime_required")
                else []
            )
            report["trace_cadence_expected_hz"] = 20.0
            report["trace_cadence_complete"] = not cadence_errors
            report["trace_cadence_errors"] = cadence_errors
            _atomic_jsonl(trace_path, trace)
            # Keep the v0 alias byte-for-byte aligned with the canonical trace.
            _atomic_jsonl(root / "frames.jsonl", trace)

            src = self.language2drive_root / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from language2drive.contracts import (  # pylint: disable=import-outside-toplevel
                ContractEngine,
                Verdict,
                score_rollout,
                semantic_instruction_verdict,
            )

            engine = ContractEngine(trace)
            own = engine.evaluate(route["contract"])
            opposite = engine.evaluate(route["opposite_contract"])
            safe = engine.evaluate({"safe_legal": {}})
            progress = engine.evaluate(
                {"eventually": {"progress": {"min": self.progress_minimum}}}
            )
            final_time = float(trace[-1]["sim_time"]) if trace else 0.0
            root_operator = next(iter(route["contract"]), "")
            if root_operator == "always":
                if isinstance(target, dict) and target.get("type") == "hold":
                    horizon_reached = (
                        bool(trace)
                        and float(target.get("seconds", 0.0)) > 0.0
                        and final_time - start_time
                        >= float(target["seconds"]) - 1e-9
                    )
                else:
                    final_progress = trace[-1].get("progress") if trace else None
                    horizon_reached = (
                        isinstance(final_progress, (int, float))
                        and float(final_progress) >= 0.95
                    )
                stable_passed = own.passed and horizon_reached
                stable_details = {
                    "required_seconds": 0.0,
                    "always_horizon_reached": horizon_reached,
                }
            else:
                candidate_times = (
                    own.first_satisfied_at, progress.first_satisfied_at
                )
                candidate_started_at = (
                    max(float(value) for value in candidate_times)
                    if all(value is not None for value in candidate_times)
                    else None
                )
                stable_passed = (
                    candidate_started_at is not None
                    and final_time - candidate_started_at
                    >= self.stability_seconds - 1e-9
                )
                stable_details = {
                    "required_seconds": self.stability_seconds,
                    "candidate_started_at": candidate_started_at,
                    "candidate_requires": [
                        "own_contract", "not_opposite_contract",
                        "safe_legal", "progress",
                    ],
                }
            stable = Verdict(
                stable_passed,
                final_time if stable_passed else None,
                stable_details,
            )
            semantic = semantic_instruction_verdict(
                trace,
                target=target if isinstance(target, dict) else {},
                oracle_path=route.get("oracle_path", []),
                own=own,
                layer=route.get("layer"),
            )
            result = score_rollout(
                own,
                opposite,
                safe,
                progress,
                final_time,
                float(route["deadline_seconds"]),
                stable,
                start_time=float(trace[0]["sim_time"]) if trace else 0.0,
                semantic=semantic,
            )
            from language2drive.core_driving_score import driving_score_percent
            result["driving_score"] = driving_score_percent(
                trace, hold_target=isinstance(target, dict) and target.get("type") == "hold"
            )
            unsupported = self._unsupported_requirements(report)
            runtime_report = report.get("evaluator_runtime", {})
            runtime_termination = runtime_report.get("termination_reason")
            runtime_failure = runtime_report.get("failure_reason")
            runtime_terminal = runtime_termination in {
                "contract_complete", "semantic_contract_complete", "deadline",
                "ego_out_of_world",
            }
            if (
                self.private_context.get("runtime_required")
                and not runtime_terminal
            ):
                unsupported.append(
                    "evaluator_runtime:{}".format(
                        "failure"
                        if runtime_failure
                        else "termination_missing"
                    )
                )
                unsupported = sorted(set(unsupported))
            if unsupported:
                result["success"] = False
                result["driving_score"] = None
            if (
                self.private_context.get("runtime_required")
                and runtime_termination != "contract_complete"
            ):
                # A crash/forced manager close can leave a trace whose final
                # frame happens to satisfy the predicates.  It is not a valid
                # success unless the online private monitor observed the full
                # candidate/stability window and owned termination.
                result["success"] = False
            result.update(
                {
                    # An unsupported capture is a completed, scored rollout
                    # whose declared success is forced false.  "failed" is a
                    # valid terminal status in the v1 expected-run ledger.
                    "status": "complete" if not unsupported else "failed",
                    "capture_complete": not unsupported,
                    "unsupported_required_fields": unsupported,
                    "route_entry_id": route.get("route_entry_id"),
                    "pcu_id": route.get("pcu_id"),
                    "arm_id": route.get("arm_id"),
                    "trace_rows": len(trace),
                    "scorer": "language2drive.v1.contracts",
                    "termination_reason": runtime_termination,
                    "runtime_failure_reason": runtime_failure,
                    "semantic_details": semantic.details,
                }
            )
            _atomic_json(root / "capture_capabilities.json", report)
        except Exception as exc:  # fail closed, but always leave a ledger-consumable result
            result = {
                "status": "failed",
                "success": False,
                "semantic_success": False,
                "own_contract": False,
                "opposite_contract": False,
                "safe_legal": False,
                "progress": False,
                "driving_score": None,
                "deadline": False,
                "stability": False,
                "capture_complete": False,
                "postprocess_error": "{}: {}".format(type(exc).__name__, exc),
                "route_entry_id": route.get("route_entry_id"),
                "pcu_id": route.get("pcu_id"),
                "arm_id": route.get("arm_id"),
            }
        _atomic_json(root / "contract_result.json", result)
        return result
