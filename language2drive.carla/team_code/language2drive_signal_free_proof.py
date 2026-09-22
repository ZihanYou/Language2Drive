"""Shared fail-closed validator for v1 signal-free speed corridors.

The generator freezes an exact lane-graph slice and binds the serialized
private ``oracle_path`` with a canonical SHA256.  Both the evaluator's private
sidecar loader and the privileged oracle compiler use this module so a route
cannot pass the release audit and then be interpreted more permissively at
runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, Mapping, Optional


SIGNAL_FREE_CORRIDOR_METHOD = "exact_signal_stop_free_lane_slice_v2"
SIGNAL_FREE_SAMPLE_SPACING_M = 0.1
SIGNAL_FREE_MIN_CORRIDOR_M = 55.0
SIGNAL_FREE_MAX_CORRIDOR_M = 65.0
SIGNAL_FREE_MIN_BUFFER_M = 20.0
SIGNAL_FREE_MIN_OTHER_SIGNAL_CLEARANCE_M = 40.0

SIGNAL_FREE_CORRIDOR_KEYS = frozenset({
    "method",
    "lane_id",
    "path_start_index_inclusive",
    "path_end_index_inclusive",
    "buffer_start_index_inclusive",
    "buffer_end_index_inclusive",
    "road_s_min",
    "road_s_max",
    "corridor_length_m",
    "before_buffer_length_m",
    "after_buffer_length_m",
    "sample_spacing_max_m",
    "samples_checked",
    "junction_sample_count",
    "road_signal_count",
    "road_signal_reference_count",
    "road_stop_control_count",
    "map_road_signal_count",
    "other_road_signal_clearance_m",
    "execution_path_sha256",
    "safety_buffer_path_sha256",
    "lane_graph_sha256",
    "xodr_sha256",
    "audit_sha256",
})

_SHA256 = re.compile(r"[0-9a-f]{64}")


class SignalFreeCorridorProofError(ValueError):
    """A private route's signal-free proof is absent or not self-consistent."""


def stable_hash(value: Any) -> str:
    """Return the generator's exact canonical-JSON hash.

    ``allow_nan=False`` is intentional.  Non-finite waypoint coordinates must
    fail before hashing rather than acquire a stable-looking digest.
    """

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SignalFreeCorridorProofError(
            "oracle_path is not canonical finite JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _speed_bands(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "speed_band" in value:
            body = value["speed_band"]
            if not isinstance(body, Mapping):
                raise SignalFreeCorridorProofError(
                    "contract predicate speed_band must be an object"
                )
            yield body
        for child in value.values():
            yield from _speed_bands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _speed_bands(child)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.{} must be finite numeric".format(field)
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.{} must be finite numeric".format(field)
        ) from exc
    if not math.isfinite(result):
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.{} must be finite numeric".format(field)
        )
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.{} must be an integer >= {}".format(
                field, minimum
            )
        )
    return value


def validate_signal_free_corridor_proof(
    route: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate and return a route's v2 corridor proof.

    A--D routes containing an own-contract ``speed_band`` must carry the
    proof.  A paired non-speed arm may also declare the same frozen corridor;
    such a declaration is fully validated and counted by the release audit.
    The separate Safety engineering artifact may omit it; if Safety supplies
    a proof, however, the same exact schema is enforced.
    """

    bands = list(_speed_bands(route.get("contract", {})))
    layer = str(route.get("layer", ""))
    raw_proof = route.get("signal_free_corridor_proof")
    if not bands and raw_proof is None:
        return None
    if raw_proof is None and layer == "Safety":
        return None
    if raw_proof is None and layer in {"A", "B", "C", "D"}:
        raise SignalFreeCorridorProofError(
            "core speed route requires signal_free_corridor_proof"
        )
    if not isinstance(raw_proof, Mapping):
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof must be an object"
        )
    proof = dict(raw_proof)
    if set(proof) != SIGNAL_FREE_CORRIDOR_KEYS:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof keys must be exactly {}".format(
                sorted(SIGNAL_FREE_CORRIDOR_KEYS)
            )
        )
    if proof["method"] != SIGNAL_FREE_CORRIDOR_METHOD:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof has unsupported method"
        )

    town = str(route.get("town", ""))
    lane_identity = proof["lane_id"]
    if not isinstance(lane_identity, str) or not lane_identity:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.lane_id must be nonempty"
        )
    try:
        lane_town, road_id, section_id, lane_id = lane_identity.rsplit(":", 3)
        parsed_lane = (int(road_id), int(section_id), int(lane_id))
    except (TypeError, ValueError) as exc:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.lane_id must be town:road:section:lane"
        ) from exc
    if lane_town != town:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.lane_id town mismatch"
        )
    if parsed_lane[2] == 0:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.lane_id cannot identify lane zero"
        )

    indices = {
        field: _integer(proof[field], field)
        for field in (
            "path_start_index_inclusive",
            "path_end_index_inclusive",
            "buffer_start_index_inclusive",
            "buffer_end_index_inclusive",
        )
    }
    buffer_start = indices["buffer_start_index_inclusive"]
    path_start = indices["path_start_index_inclusive"]
    path_end = indices["path_end_index_inclusive"]
    buffer_end = indices["buffer_end_index_inclusive"]
    if not buffer_start < path_start < path_end < buffer_end:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof slice indices are not strictly nested"
        )

    oracle_path = route.get("oracle_path")
    if not isinstance(oracle_path, list) or len(oracle_path) < 2:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof requires a serialized oracle_path"
        )
    if path_end - path_start + 1 != len(oracle_path):
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof path indices do not close oracle_path point count"
        )

    zero_fields = (
        "junction_sample_count",
        "road_signal_count",
        "road_signal_reference_count",
        "road_stop_control_count",
    )
    for field in zero_fields:
        if _integer(proof[field], field) != 0:
            raise SignalFreeCorridorProofError(
                "signal_free_corridor_proof.{} must equal integer zero".format(
                    field
                )
            )

    road_s_min = _finite_number(proof["road_s_min"], "road_s_min")
    road_s_max = _finite_number(proof["road_s_max"], "road_s_max")
    if road_s_min > road_s_max:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof road s bounds are inverted"
        )
    corridor_length = _finite_number(
        proof["corridor_length_m"], "corridor_length_m"
    )
    if not SIGNAL_FREE_MIN_CORRIDOR_M <= corridor_length <= SIGNAL_FREE_MAX_CORRIDOR_M:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof corridor_length_m must be in [{}, {}]".format(
                SIGNAL_FREE_MIN_CORRIDOR_M, SIGNAL_FREE_MAX_CORRIDOR_M
            )
        )
    for field in ("before_buffer_length_m", "after_buffer_length_m"):
        if _finite_number(proof[field], field) < SIGNAL_FREE_MIN_BUFFER_M:
            raise SignalFreeCorridorProofError(
                "signal_free_corridor_proof.{} must be at least {}".format(
                    field, SIGNAL_FREE_MIN_BUFFER_M
                )
            )
    spacing = _finite_number(
        proof["sample_spacing_max_m"], "sample_spacing_max_m"
    )
    if abs(spacing - SIGNAL_FREE_SAMPLE_SPACING_M) > 1e-9:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof.sample_spacing_max_m must equal 0.1"
        )
    _integer(proof["samples_checked"], "samples_checked", minimum=1)

    map_signal_count = _integer(
        proof["map_road_signal_count"], "map_road_signal_count"
    )
    clearance = proof["other_road_signal_clearance_m"]
    if clearance is None:
        if map_signal_count != 0:
            raise SignalFreeCorridorProofError(
                "signal_free_corridor_proof null clearance requires zero map signals"
            )
    else:
        clearance_m = _finite_number(
            clearance, "other_road_signal_clearance_m"
        )
        if map_signal_count <= 0:
            raise SignalFreeCorridorProofError(
                "signal_free_corridor_proof clearance requires map signals"
            )
        if clearance_m < SIGNAL_FREE_MIN_OTHER_SIGNAL_CLEARANCE_M:
            raise SignalFreeCorridorProofError(
                "signal_free_corridor_proof other-road signal clearance must be >=40m"
            )

    for field in (
        "execution_path_sha256",
        "safety_buffer_path_sha256",
        "lane_graph_sha256",
        "xodr_sha256",
        "audit_sha256",
    ):
        digest = proof[field]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise SignalFreeCorridorProofError(
                "signal_free_corridor_proof.{} must be lowercase SHA256".format(
                    field
                )
            )
    if proof["execution_path_sha256"] != stable_hash(oracle_path):
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof does not bind serialized oracle_path"
        )

    actual_length = 0.0
    try:
        for first, second in zip(oracle_path, oracle_path[1:]):
            if not isinstance(first, Mapping) or not isinstance(second, Mapping):
                raise TypeError("waypoint is not an object")
            first_x = _finite_number(first["x"], "oracle_path.x")
            first_y = _finite_number(first["y"], "oracle_path.y")
            second_x = _finite_number(second["x"], "oracle_path.x")
            second_y = _finite_number(second["y"], "oracle_path.y")
            segment = math.hypot(second_x - first_x, second_y - first_y)
            if segment <= 1e-6:
                raise SignalFreeCorridorProofError(
                    "signal-free oracle_path has duplicate consecutive points"
                )
            actual_length += segment
    except (KeyError, TypeError) as exc:
        raise SignalFreeCorridorProofError(
            "signal-free oracle_path points require finite x/y"
        ) from exc
    if abs(actual_length - corridor_length) > 0.01:
        raise SignalFreeCorridorProofError(
            "signal_free_corridor_proof length does not match serialized oracle_path"
        )

    return proof
