"""Fail-closed schema validation for trusted RealSim PCU sources."""

import math
import re
from fractions import Fraction
from typing import Any, Dict, Iterable, List

from .events import polygon_parts, segments_intersect, point_in_polygon
from .render_validity import native_render_protocol
from .family_allocation import family_targets

from .constants import (
    ARMS,
    ALPASIM_COMMIT,
    COUPLED_SEEDS,
    FAMILY_TARGETS,
    LAYER_TARGETS,
    NUREC_DATASET_REVISION,
    ORACLE_CONTROLLER_ID,
    SCHEMA_VERSION,
)


class SchemaError(ValueError):
    pass


_OPAQUE_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[0-9a-f]{4,})+$")
_CONTRACT_KINDS = {
    "gate",
    "lane",
    "scoped_lane",
    "speed_band",
    "triggered_speed_band",
    "stop_region",
    "continue_region",
    "pass_side",
}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_FAMILY_CONTRACT_KINDS = {
    "A1_intersection_left_right": ("gate", "gate"),
    "A2_straight_turn": ("gate", "gate"),
    "A3_fork_left_right": ("gate", "gate"),
    "A4_lane_change_left_right": ("lane", "lane"),
    "A5_lane_change_keep": ("lane", "lane"),
    "A6_pull_over_continue": ("stop_region", "continue_region"),
    "B1_segment_speed_band": ("speed_band", "speed_band"),
    "B2_scoped_lane_preference": ("scoped_lane", "scoped_lane"),
    "B3_triggered_speed_constraint": (
        "triggered_speed_band",
        "triggered_speed_band",
    ),
    "B4_object_relative_passing": ("pass_side", "pass_side"),
    "B5_relative_stopping_region": ("stop_region", "stop_region"),
    "C1_direction_sign_grounding": ("gate", "gate"),
    "C2_road_exit_text_grounding": ("gate", "gate"),
    "C3_landmark_adjacent_branch": ("gate", "gate"),
    "C4_attribute_landmark_target": ("gate", "gate"),
}


def _require_object(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError("%s must be an object" % path)
    return value


def _require_list(value: Any, path: str) -> List[Any]:
    if not isinstance(value, list):
        raise SchemaError("%s must be a list" % path)
    return value


def _required(value: Dict[str, Any], fields: Iterable[str], path: str) -> None:
    missing = sorted(set(fields) - set(value))
    if missing:
        raise SchemaError("%s missing %s" % (path, missing))


def _finite(value: Any, path: str, minimum: float = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SchemaError("%s must be finite" % path)
    result = float(value)
    if minimum is not None and result < minimum:
        raise SchemaError("%s must be >= %s" % (path, minimum))
    return result


def _ring_edges_intersect(a, b, c, d):
    """Exact topology of stored coordinates, not the runtime gate epsilon.

    A short but nonzero map edge must not merge its two neighbouring vertices.
    Bounding boxes discard separated edges before rational arithmetic. This
    also detects small true crossings that an absolute determinant epsilon
    can miss; no coordinates are snapped, buffered or otherwise modified.
    """
    if any(max(a[i], b[i]) < min(c[i], d[i]) or max(c[i], d[i]) < min(a[i], b[i]) for i in (0, 1)):
        return False
    a, b, c, d = [tuple(Fraction(v) for v in p[:2]) for p in (a, b, c, d)]
    def orientation(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def on_segment(p, q, r):
        return all(min(p[i],q[i]) <= r[i] <= max(p[i],q[i]) for i in (0,1))
    o1, o2, o3, o4 = orientation(a,b,c), orientation(a,b,d), orientation(c,d,a), orientation(c,d,b)
    return ((o1*o2 < 0 and o3*o4 < 0)
            or (o1 == 0 and on_segment(a,b,c)) or (o2 == 0 and on_segment(a,b,d))
            or (o3 == 0 and on_segment(c,d,a)) or (o4 == 0 and on_segment(c,d,b)))


def _validate_polygon(value: Any, path: str) -> None:
    if isinstance(value, dict):
        try:
            parts = polygon_parts(value)
        except (ValueError, TypeError) as error:
            raise SchemaError("%s: %s" % (path, error)) from error
        for part_index, part in enumerate(parts):
            for ring_index, ring in enumerate(part):
                ring_path = "%s.part[%d].ring[%d]" % (path, part_index, ring_index)
                _require_list(ring, ring_path)
                _validate_polygon(ring, ring_path)
                if len(ring) < 4 or ring[0] != ring[-1] or any(len(p) != 2 for p in ring):
                    raise SchemaError("%s requires a closed XY ring" % ring_path)
                edges = list(zip(ring[:-1], ring[1:]))
                if any(a == b for a, b in edges):
                    raise SchemaError("%s has duplicate consecutive vertices" % ring_path)
                area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)
                if area2 == 0:
                    raise SchemaError("%s has zero area" % ring_path)
                for i, (a, b) in enumerate(edges):
                    for j in range(i + 2, len(edges)):
                        if i == 0 and j == len(edges) - 1:
                            continue
                        if _ring_edges_intersect(a, b, *edges[j]):
                            raise SchemaError("%s self-intersects" % ring_path)
            for hole_index, hole in enumerate(part[1:], 1):
                outer = part[0]
                if (not point_in_polygon(hole[0], outer)
                    or any(segments_intersect(a, b, c, d)
                           for a, b in zip(hole[:-1], hole[1:])
                           for c, d in zip(outer[:-1], outer[1:]))):
                    raise SchemaError("%s hole must lie strictly inside its exterior" % path)
                for previous in part[1:hole_index]:
                    if (point_in_polygon(hole[0], previous) or point_in_polygon(previous[0], hole)
                        or any(segments_intersect(a, b, c, d)
                               for a, b in zip(hole[:-1], hole[1:])
                               for c, d in zip(previous[:-1], previous[1:]))):
                        raise SchemaError("%s holes must be disjoint" % path)
        return
    points = _require_list(value, path)
    if len(points) < 3:
        raise SchemaError("%s needs at least three points" % path)
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) < 2:
            raise SchemaError("%s[%d] must be an xy point" % (path, index))
        _finite(point[0], "%s[%d][0]" % (path, index))
        _finite(point[1], "%s[%d][1]" % (path, index))


def _validate_gate(value: Any, path: str) -> None:
    points = _require_list(value, path)
    if len(points) != 2:
        raise SchemaError("%s must have exactly two endpoints" % path)
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) < 2:
            raise SchemaError("%s[%d] must be an xy point" % (path, index))
        _finite(point[0], "%s[%d][0]" % (path, index))
        _finite(point[1], "%s[%d][1]" % (path, index))
    if points[0][:2] == points[1][:2]:
        raise SchemaError("%s endpoints must be distinct" % path)


def validate_contract(contract: Any, path: str = "contract") -> None:
    contract = _require_object(contract, path)
    _required(contract, ("id", "kind", "signature_set", "deadline_s"), path)
    if not isinstance(contract["id"], str) or not _OPAQUE_ID.fullmatch(contract["id"]):
        raise SchemaError("%s.id must be opaque" % path)
    kind = contract["kind"]
    if kind not in _CONTRACT_KINDS:
        raise SchemaError("%s.kind is unsupported: %r" % (path, kind))
    signatures = _require_list(contract["signature_set"], path + ".signature_set")
    if not signatures or any(not isinstance(item, str) or not item for item in signatures):
        raise SchemaError("%s.signature_set requires nonempty strings" % path)
    if len(signatures) != len(set(signatures)):
        raise SchemaError("%s.signature_set contains duplicates" % path)
    _finite(contract["deadline_s"], path + ".deadline_s", 0.0)
    if kind == "gate":
        _required(contract, ("target_gate", "opposing_gate"), path)
    elif kind in {"lane", "scoped_lane"}:
        _required(contract, ("target_lane", "scope", "stable_duration_s"), path)
        _finite(contract["stable_duration_s"], path + ".stable_duration_s", 0.0)
        if kind == "scoped_lane":
            _required(contract, ("entry_gate", "exit_gate"), path)
            if any(not isinstance(contract[k], str) or not contract[k] for k in ("entry_gate", "exit_gate")):
                raise SchemaError("%s scoped lane gates must be nonempty strings" % path)
            if contract["entry_gate"] == contract["exit_gate"]:
                raise SchemaError("%s scoped lane gates must be distinct" % path)
    elif kind in {"speed_band", "triggered_speed_band"}:
        _required(contract, ("speed_min_mps", "speed_max_mps", "scope", "entry_gate", "exit_gate"), path)
        for field in ("entry_gate", "exit_gate"):
            if not isinstance(contract[field], str) or not contract[field]:
                raise SchemaError("%s.%s must be a nonempty gate ID" % (path, field))
        if contract["entry_gate"] == contract["exit_gate"]:
            raise SchemaError("%s speed entry and exit gates must be distinct" % path)
        lower = _finite(contract["speed_min_mps"], path + ".speed_min_mps", 0.0)
        upper = _finite(contract["speed_max_mps"], path + ".speed_max_mps", 0.0)
        if not lower < upper:
            raise SchemaError("%s speed band must have positive width" % path)
        if kind == "triggered_speed_band":
            _required(contract, ("trigger_id",), path)
            if not isinstance(contract["trigger_id"], str) or not contract["trigger_id"]:
                raise SchemaError("%s.trigger_id must be a nonempty trigger ID" % path)
    elif kind in {"stop_region", "continue_region"}:
        _required(contract, ("region", "scope"), path)
        if kind == "stop_region":
            _required(contract, ("dwell_s", "max_speed_mps"), path)
            _finite(contract["dwell_s"], path + ".dwell_s", 0.0)
            _finite(contract["max_speed_mps"], path + ".max_speed_mps", 0.0)
        else:
            _required(contract, ("target_gate",), path)
    elif kind == "pass_side":
        _required(contract, ("actor_id", "side", "min_clearance_m"), path)
        if contract["side"] not in {"LEFT", "RIGHT"}:
            raise SchemaError("%s.side must be LEFT or RIGHT" % path)
        _finite(contract["min_clearance_m"], path + ".min_clearance_m", 0.0)


def _validate_contract_pair(
    first: Dict[str, Any], second: Dict[str, Any], path: str
) -> None:
    """Enforce semantic exclusivity in executable fields, not only labels."""

    kinds = (first["kind"], second["kind"])
    if kinds == ("gate", "gate"):
        if first["target_gate"] == second["target_gate"]:
            raise SchemaError("%s gate targets must be distinct" % path)
        if (
            first["opposing_gate"] != second["target_gate"]
            or second["opposing_gate"] != first["target_gate"]
        ):
            raise SchemaError("%s gate opposing targets must be reciprocal" % path)
    elif kinds[0] in {"lane", "scoped_lane"} and kinds[1] == kinds[0]:
        if first["target_lane"] == second["target_lane"]:
            raise SchemaError("%s lane targets must be distinct" % path)
        if first["scope"] != second["scope"]:
            raise SchemaError("%s lane arms must share one scope" % path)
        if kinds[0] == "scoped_lane" and any(first[k] != second[k] for k in ("entry_gate", "exit_gate", "stable_duration_s")):
            raise SchemaError("%s scoped lane arms must share gates and minimum duration" % path)
    elif kinds[0] in {"speed_band", "triggered_speed_band"} and kinds[1] == kinds[0]:
        if first["scope"] != second["scope"]:
            raise SchemaError("%s speed arms must share one scope" % path)
        if any(first[field] != second[field] for field in ("entry_gate", "exit_gate")):
            raise SchemaError("%s speed arms must share entry and exit gates" % path)
        if kinds[0] == "triggered_speed_band" and first["trigger_id"] != second["trigger_id"]:
            raise SchemaError("%s speed arms must share one trigger" % path)
        disjoint = (
            float(first["speed_max_mps"]) < float(second["speed_min_mps"])
            or float(second["speed_max_mps"]) < float(first["speed_min_mps"])
        )
        if not disjoint:
            raise SchemaError("%s speed bands must be non-overlapping" % path)
    elif set(kinds) == {"stop_region", "continue_region"}:
        stop = first if first["kind"] == "stop_region" else second
        drive = second if first["kind"] == "stop_region" else first
        if stop["scope"] != drive["scope"] or stop["region"] != drive["region"]:
            raise SchemaError(
                "%s stop/continue arms must share the scope and forbidden stop region"
                % path
            )
    elif kinds == ("stop_region", "stop_region"):
        if first["scope"] != second["scope"]:
            raise SchemaError("%s stop arms must share one scope" % path)
        if first["region"] == second["region"]:
            raise SchemaError("%s stop regions must be distinct" % path)
    elif kinds == ("pass_side", "pass_side"):
        if first["actor_id"] != second["actor_id"]:
            raise SchemaError("%s pass arms must reference the same actor" % path)
        if first["side"] == second["side"]:
            raise SchemaError("%s pass sides must be distinct" % path)
    else:
        raise SchemaError("%s has incompatible contract kinds %r" % (path, kinds))


def _validate_oracle_plans(pcu: Dict[str, Any], path: str) -> None:
    plans = _require_object(pcu["oracle_plans_by_seed"], path)
    expected_seeds = {str(seed) for seed in COUPLED_SEEDS}
    if set(plans) != expected_seeds:
        raise SchemaError("%s must contain exactly the frozen coupled seeds" % path)
    deadline = max(
        float(pcu["decision_window"]["deadline_s"]),
        float(pcu["hard_constraints"].get("deadline_s", 0.0)),
    )
    issuance_us = pcu["arms"]["R0"]["instruction_schedule"][0]["issue_at_us"]
    intervention_s = (
        int(issuance_us) - int(pcu["scene"]["start_timestamp_us"])
    ) / 1_000_000.0
    for seed in COUPLED_SEEDS:
        seed_path = "%s.%s" % (path, seed)
        arms = _require_object(plans[str(seed)], seed_path)
        if set(arms) != set(ARMS):
            raise SchemaError("%s must contain exactly R0 and R1" % seed_path)
        reviewed_plans = {}
        for arm_id in ARMS:
            plan_path = "%s.%s" % (seed_path, arm_id)
            plan = _require_object(arms[arm_id], plan_path)
            _required(plan, ("controller", "coordinate_frame", "waypoints"), plan_path)
            if plan["controller"] != ORACLE_CONTROLLER_ID:
                raise SchemaError("%s controller is not the frozen oracle" % plan_path)
            if plan["coordinate_frame"] != "nre_local":
                raise SchemaError("%s must use the nre_local coordinate frame" % plan_path)
            waypoints = _require_list(plan["waypoints"], plan_path + ".waypoints")
            if len(waypoints) < 2:
                raise SchemaError("%s needs at least two oracle waypoints" % plan_path)
            reviewed = []
            previous_time = -math.inf
            for index, raw_point in enumerate(waypoints):
                point_path = "%s.waypoints[%d]" % (plan_path, index)
                point = _require_object(raw_point, point_path)
                _required(point, ("time_s", "x", "y", "z", "yaw_rad"), point_path)
                values = {
                    key: _finite(point[key], point_path + "." + key, 0.0 if key == "time_s" else None)
                    for key in ("time_s", "x", "y", "z", "yaw_rad")
                }
                if values["time_s"] <= previous_time:
                    raise SchemaError("%s waypoint times must be strictly increasing" % plan_path)
                previous_time = values["time_s"]
                reviewed.append(values)
            if reviewed[0]["time_s"] != 0.0:
                raise SchemaError("%s must start at scenario-relative time zero" % plan_path)
            initial_pose = pcu["scene"]["initial_pose"]
            if any(
                abs(reviewed[0][key] - float(initial_pose[key])) > 1e-6
                for key in ("x", "y", "z", "yaw_rad")
            ):
                raise SchemaError("%s must start at the frozen initial pose" % plan_path)
            if reviewed[-1]["time_s"] < deadline:
                raise SchemaError("%s must cover the complete decision deadline" % plan_path)
            reviewed_plans[arm_id] = reviewed
        prefixes = {
            arm_id: [row for row in reviewed_plans[arm_id] if row["time_s"] <= intervention_s]
            for arm_id in ARMS
        }
        if prefixes["R0"] != prefixes["R1"] or not prefixes["R0"]:
            raise SchemaError(
                "%s oracle arms must share an identical pre-intervention prefix"
                % seed_path
            )


def validate_pcu(pcu: Any, path: str = "pcu", *, allow_pending_reviews: bool = False) -> None:
    pcu = _require_object(pcu, path)
    _required(
        pcu,
        (
            "schema_version",
            "pcu_id",
            "meta_group_id",
            "backend",
            "primary_layer",
            "task_family",
            "scene",
            "shared_non_language_input",
            "decision_window",
            "arms",
            "hard_constraints",
        ),
        path,
    )
    if pcu["schema_version"] != SCHEMA_VERSION:
        raise SchemaError("%s.schema_version must be %s" % (path, SCHEMA_VERSION))
    for field in ("pcu_id", "meta_group_id"):
        if not isinstance(pcu[field], str) or not _OPAQUE_ID.fullmatch(pcu[field]):
            raise SchemaError("%s.%s must be an opaque identifier" % (path, field))
    backend = _require_object(pcu["backend"], path + ".backend")
    _required(backend, ("type", "alpasim_commit", "renderer_type", "dataset_version"), path + ".backend")
    if backend["type"] != "alpasim":
        raise SchemaError("%s.backend.type must be alpasim" % path)
    if backend["renderer_type"] != "nurec":
        raise SchemaError("%s.backend.renderer_type must be nurec" % path)
    if backend["dataset_version"] != NUREC_DATASET_REVISION:
        raise SchemaError(
            "%s.backend.dataset_version must be %s"
            % (path, NUREC_DATASET_REVISION)
        )
    if backend["alpasim_commit"] != ALPASIM_COMMIT:
        raise SchemaError(
            "%s.backend.alpasim_commit must be the pinned commit %s"
            % (path, ALPASIM_COMMIT)
        )
    layer = pcu["primary_layer"]
    if layer not in LAYER_TARGETS:
        raise SchemaError("%s.primary_layer must be A, B, or C" % path)
    family = pcu["task_family"]
    if family not in FAMILY_TARGETS or not family.startswith(layer):
        raise SchemaError("%s.task_family disagrees with layer" % path)
    scene = _require_object(pcu["scene"], path + ".scene")
    _required(
        scene,
        (
            "artifact_id",
            "scene_id_private",
            "recording_id_private",
            "recording_start_timestamp_us",
            "start_timestamp_us",
            "start_time_offset_us",
            "initial_pose",
            "camera_configuration",
            "reconstruction_support_id",
        ),
        path + ".scene",
    )
    for field in (
        "recording_start_timestamp_us",
        "start_timestamp_us",
        "start_time_offset_us",
    ):
        if isinstance(scene[field], bool) or not isinstance(scene[field], int):
            raise SchemaError("%s.scene.%s must be an integer" % (path, field))
        if scene[field] < 0:
            raise SchemaError("%s.scene.%s must be nonnegative" % (path, field))
    if (
        scene["recording_start_timestamp_us"] + scene["start_time_offset_us"]
        != scene["start_timestamp_us"]
    ):
        raise SchemaError(
            "%s scene timestamps must satisfy recording start + offset = start timestamp"
            % path
        )
    if not _UUID.fullmatch(str(scene["artifact_id"])):
        raise SchemaError("%s.scene.artifact_id must be the catalog artifact UUID" % path)
    if not str(scene["scene_id_private"]).startswith("clipgt-") or not _UUID.fullmatch(str(scene["scene_id_private"])[7:]):
        raise SchemaError("%s.scene.scene_id_private must be a ClipGT UUID" % path)
    initial_pose = _require_object(scene["initial_pose"], path + ".scene.initial_pose")
    _required(initial_pose, ("x", "y", "z", "yaw_rad"), path + ".scene.initial_pose")
    for field in ("x", "y", "z", "yaw_rad"):
        _finite(initial_pose[field], path + ".scene.initial_pose." + field)
    camera = _require_object(
        scene["camera_configuration"], path + ".scene.camera_configuration"
    )
    _required(camera, ("logical_ids", "height", "width"), path + ".scene.camera_configuration")
    logical_ids = _require_list(
        camera["logical_ids"], path + ".scene.camera_configuration.logical_ids"
    )
    if not logical_ids or any(not isinstance(item, str) or not item for item in logical_ids):
        raise SchemaError("%s camera logical IDs must be nonempty strings" % path)
    if len(logical_ids) != len(set(logical_ids)):
        raise SchemaError("%s camera logical IDs must be unique" % path)
    for field in ("height", "width"):
        if isinstance(camera[field], bool) or not isinstance(camera[field], int) or camera[field] < 1:
            raise SchemaError("%s camera %s must be a positive integer" % (path, field))
    shared = _require_object(pcu["shared_non_language_input"], path + ".shared_non_language_input")
    _required(
        shared,
        (
            "route_envelope",
            "route_envelope_kind",
            "route_semantic_target",
            "route_neutrality_reviewed",
            "command_policy",
            "coupled_exogenous_seeds",
            "simulator_configuration",
        ),
        path + ".shared_non_language_input",
    )
    if shared["command_policy"] != "UNKNOWN_NEUTRAL":
        raise SchemaError("%s must disable target-specific waypoint commands" % path)
    if (
        shared["route_envelope_kind"] != "COMMON_SUPPORT"
        or shared["route_semantic_target"] is not None
        or type(shared["route_neutrality_reviewed"]) is not bool
        or (not allow_pending_reviews and shared["route_neutrality_reviewed"] is not True)
    ):
        raise SchemaError("%s route must be a reviewed common envelope without a semantic target" % path)
    if tuple(shared["coupled_exogenous_seeds"]) != COUPLED_SEEDS:
        raise SchemaError("%s must use the frozen three coupled seeds" % path)
    route_envelope = _require_list(
        shared["route_envelope"], path + ".shared_non_language_input.route_envelope"
    )
    if len(route_envelope) < 2:
        raise SchemaError("%s route envelope needs at least two points" % path)
    for index, point in enumerate(route_envelope):
        if not isinstance(point, list) or len(point) < 2:
            raise SchemaError("%s route point %d needs xy" % (path, index))
        _finite(point[0], "%s route point %d x" % (path, index))
        _finite(point[1], "%s route point %d y" % (path, index))
    _require_object(
        shared["simulator_configuration"],
        path + ".shared_non_language_input.simulator_configuration",
    )
    window = _require_object(pcu["decision_window"], path + ".decision_window")
    _required(window, ("trigger", "start_s", "deadline_s", "commitment_gates"), path + ".decision_window")
    if _finite(window["start_s"], path + ".decision_window.start_s", 0.0) >= _finite(
        window["deadline_s"], path + ".decision_window.deadline_s", 0.0
    ):
        raise SchemaError("%s decision window must have positive duration" % path)
    commitment_gates = _require_list(
        window["commitment_gates"], path + ".decision_window.commitment_gates"
    )
    if any(not isinstance(gate, str) or not gate for gate in commitment_gates):
        raise SchemaError("%s commitment gates must be nonempty strings" % path)
    if len(commitment_gates) != len(set(commitment_gates)):
        raise SchemaError("%s commitment gates must be unique" % path)

    arms = _require_object(pcu["arms"], path + ".arms")
    if set(arms) != set(ARMS):
        raise SchemaError("%s.arms must contain exactly R0 and R1" % path)
    seen_utterances = set()
    contracts = []
    issuance_times = []
    for arm_id in ARMS:
        arm = _require_object(arms[arm_id], "%s.arms.%s" % (path, arm_id))
        _required(arm, ("instruction_schedule", "semantic_contract", "opposing_contract"), "%s.arms.%s" % (path, arm_id))
        schedule = _require_list(arm["instruction_schedule"], "%s.arms.%s.instruction_schedule" % (path, arm_id))
        if len(schedule) != 1:
            raise SchemaError("%s/%s must have one v0.1 issuance" % (path, arm_id))
        issuance = _require_object(schedule[0], "%s.arms.%s.instruction_schedule[0]" % (path, arm_id))
        _required(issuance, ("issue_at_us", "utterances"), "%s.arms.%s.instruction_schedule[0]" % (path, arm_id))
        if isinstance(issuance["issue_at_us"], bool) or not isinstance(
            issuance["issue_at_us"], int
        ):
            raise SchemaError("%s/%s issue_at_us must be an integer" % (path, arm_id))
        latest_issue_us = scene["start_timestamp_us"] + int(
            float(window["deadline_s"]) * 1_000_000
        )
        if not scene["start_timestamp_us"] <= issuance["issue_at_us"] <= latest_issue_us:
            raise SchemaError(
                "%s/%s instruction issuance must fall inside the decision horizon"
                % (path, arm_id)
            )
        issuance_times.append(issuance["issue_at_us"])
        utterances = _require_list(issuance["utterances"], "%s.arms.%s.utterances" % (path, arm_id))
        if len(utterances) != 3 or any(not isinstance(text, str) or not text.strip() for text in utterances):
            raise SchemaError("%s/%s requires exactly three nonempty utterances" % (path, arm_id))
        normalized = {text.strip().casefold() for text in utterances}
        if len(normalized) != 3 or seen_utterances.intersection(normalized):
            raise SchemaError("%s arm utterances must be unique and disjoint" % path)
        seen_utterances.update(normalized)
        validate_contract(arm["semantic_contract"], "%s.arms.%s.semantic_contract" % (path, arm_id))
        validate_contract(arm["opposing_contract"], "%s.arms.%s.opposing_contract" % (path, arm_id))
        contracts.append(arm["semantic_contract"])

    if len(set(issuance_times)) != 1:
        raise SchemaError("%s arm instruction issue times must be identical" % path)

    actual_kinds = tuple(contract["kind"] for contract in contracts)
    expected_kinds = _FAMILY_CONTRACT_KINDS[family]
    if family == "A6_pull_over_continue":
        kinds_match = set(actual_kinds) == set(expected_kinds)
    else:
        kinds_match = actual_kinds == expected_kinds
    if not kinds_match:
        raise SchemaError(
            "%s task family requires contract kinds %r, got %r"
            % (path, expected_kinds, actual_kinds)
        )

    if arms["R0"]["opposing_contract"] != arms["R1"]["semantic_contract"]:
        raise SchemaError("%s R0 opposing contract must equal R1 contract" % path)
    if arms["R1"]["opposing_contract"] != arms["R0"]["semantic_contract"]:
        raise SchemaError("%s R1 opposing contract must equal R0 contract" % path)
    if set(contracts[0]["signature_set"]).intersection(contracts[1]["signature_set"]):
        raise SchemaError("%s semantic signature sets are not disjoint" % path)
    _validate_contract_pair(contracts[0], contracts[1], path + ".arms")
    if contracts[0]["kind"] == contracts[1]["kind"] == "gate" and set(
        commitment_gates
    ) != {contracts[0]["target_gate"], contracts[1]["target_gate"]}:
        raise SchemaError(
            "%s commitment gates must equal the two semantic gate targets" % path
        )

    hard = _require_object(pcu["hard_constraints"], path + ".hard_constraints")
    _required(
        hard,
        ("collision", "offroad", "minimum_progress_m", "legality", "deadline_s"),
        path + ".hard_constraints",
    )
    if hard["collision"] is not False or hard["offroad"] is not False:
        raise SchemaError("%s collision/offroad constraints must require false" % path)
    if hard["legality"] is not True:
        raise SchemaError("%s legality hard constraint must be true" % path)
    if _finite(hard["minimum_progress_m"], path + ".hard_constraints.minimum_progress_m", 0.0) <= 0.0:
        raise SchemaError("%s minimum progress must be positive" % path)
    hard_deadline = _finite(hard["deadline_s"], path + ".hard_constraints.deadline_s", 0.0)
    if hard_deadline < float(window["deadline_s"]):
        raise SchemaError("%s hard deadline cannot precede the decision deadline" % path)
    if any(float(contract["deadline_s"]) > hard_deadline for contract in contracts):
        raise SchemaError("%s semantic contract exceeds the hard deadline" % path)
    if "oracle_plans_by_seed" in pcu:
        _validate_oracle_plans(pcu, path + ".oracle_plans_by_seed")

    annotations = _require_object(pcu.get("private_annotations"), path + ".private_annotations")
    _required(
        annotations,
        (
            "gates",
            "regions",
            "lanes",
            "scopes",
            "triggers",
            "actor_ids",
            "reconstruction_support_polygon",
            "legality_reviewed",
            "balance_reviewed",
            "render_reviewed",
        ),
        path + ".private_annotations",
    )
    review_fields = ("legality_reviewed", "balance_reviewed", "render_reviewed")
    if (any(type(annotations[field]) is not bool for field in review_fields)
        or (not allow_pending_reviews and not all(annotations[field] is True for field in review_fields))):
        raise SchemaError("%s private legality/balance/render reviews must all pass" % path)
    try:
        native_render = native_render_protocol(annotations)
    except ValueError as error:
        raise SchemaError(str(error)) from error
    if not native_render:
        _validate_polygon(annotations["reconstruction_support_polygon"], path + ".private_annotations.reconstruction_support_polygon")
    geometry_sets = {name: _require_object(annotations[name], path + ".private_annotations." + name) for name in ("gates", "regions", "lanes", "scopes", "triggers")}
    actor_ids = _require_list(annotations["actor_ids"], path + ".private_annotations.actor_ids")
    for geometry_id, gate in geometry_sets["gates"].items():
        _validate_gate(gate, path + ".private_annotations.gates." + geometry_id)
    for registry_name in ("regions", "lanes", "scopes", "triggers"):
        for geometry_id, polygon in geometry_sets[registry_name].items():
            _validate_polygon(
                polygon,
                path + ".private_annotations." + registry_name + "." + geometry_id,
            )
    if any(not isinstance(actor_id, str) or not actor_id for actor_id in actor_ids):
        raise SchemaError("%s actor IDs must be nonempty strings" % path)
    if len(actor_ids) != len(set(actor_ids)):
        raise SchemaError("%s actor IDs must be unique" % path)
    if family == "B3_triggered_speed_constraint":
        # A geometric trigger alone does not establish a scene/object cue.
        # This binding stays private; both arms share the same fixed referent.
        referent_path = path + ".private_annotations.trigger_referent"
        referent = _require_object(annotations.get("trigger_referent"), referent_path)
        _required(referent, ("referent_id", "referent_kind", "trigger_id", "private_description",
                             "model_input_resolution", "frame_refs", "identity_reviewed",
                             "visibility_reviewed", "trigger_geometry_reviewed"), referent_path)
        if not isinstance(referent["referent_id"], str) or not _OPAQUE_ID.fullmatch(referent["referent_id"]):
            raise SchemaError("%s referent ID must be opaque" % referent_path)
        if referent["referent_kind"] not in ("actor", "landmark"):
            raise SchemaError("%s referent kind must be actor or landmark" % referent_path)
        if referent["trigger_id"] != contracts[0]["trigger_id"]:
            raise SchemaError("%s trigger disagrees with the shared B3 contract" % referent_path)
        if not isinstance(referent["private_description"], str) or not referent["private_description"].strip():
            raise SchemaError("%s needs a fixed private semantic description" % referent_path)
        if referent["model_input_resolution"] != [camera["height"], camera["width"]]:
            raise SchemaError("%s resolution must match the model input" % referent_path)
        frames = _require_list(referent["frame_refs"], referent_path + ".frame_refs")
        if not frames or any(not isinstance(item, str) or not item.strip() for item in frames):
            raise SchemaError("%s needs visible frame evidence references" % referent_path)
        if len(frames) != len(set(frames)):
            raise SchemaError("%s frame evidence references must be distinct" % referent_path)
        for field in ("identity_reviewed", "visibility_reviewed", "trigger_geometry_reviewed"):
            if not isinstance(referent[field], bool) or (not allow_pending_reviews and referent[field] is not True):
                raise SchemaError("%s.%s requires explicit review" % (referent_path, field))
        if referent["referent_kind"] == "actor":
            if referent.get("actor_id") not in actor_ids:
                raise SchemaError("%s actor is not registered" % referent_path)
        elif "actor_id" in referent:
            raise SchemaError("%s landmark must not impersonate an actor binding" % referent_path)
    trigger_id = window["trigger"]
    if trigger_id not in geometry_sets["triggers"]:
        raise SchemaError("%s decision trigger is not registered" % path)
    for arm_id, contract in zip(ARMS, contracts):
        contract_path = "%s.arms.%s.semantic_contract" % (path, arm_id)
        kind = contract["kind"]
        if kind == "gate" and contract["target_gate"] not in geometry_sets["gates"]:
            raise SchemaError("%s target gate is not registered" % contract_path)
        if kind in {"lane", "scoped_lane"}:
            if contract["target_lane"] not in geometry_sets["lanes"] or contract["scope"] not in geometry_sets["scopes"]:
                raise SchemaError("%s lane/scope geometry is not registered" % contract_path)
        if kind in {"speed_band", "triggered_speed_band", "scoped_lane"}:
            if contract["scope"] not in geometry_sets["scopes"]:
                raise SchemaError("%s speed scope is not registered" % contract_path)
            for field in ("entry_gate", "exit_gate"):
                if contract[field] not in geometry_sets["gates"]:
                    raise SchemaError("%s speed %s is not registered" % (contract_path, field))
            entry_geometry = geometry_sets["gates"][contract["entry_gate"]]
            exit_geometry = geometry_sets["gates"][contract["exit_gate"]]
            if entry_geometry == exit_geometry or entry_geometry == exit_geometry[::-1]:
                raise SchemaError("%s speed entry/exit resolve to identical geometry" % contract_path)
            if kind == "triggered_speed_band" and contract["trigger_id"] not in geometry_sets["triggers"]:
                raise SchemaError("%s speed trigger is not registered" % contract_path)
        if kind in {"stop_region", "continue_region"}:
            if contract["region"] not in geometry_sets["regions"] or contract["scope"] not in geometry_sets["scopes"]:
                raise SchemaError("%s stop/continue region or scope is not registered" % contract_path)
            if kind == "continue_region" and contract["target_gate"] not in geometry_sets["gates"]:
                raise SchemaError("%s continue gate is not registered" % contract_path)
        if kind == "pass_side" and contract["actor_id"] not in actor_ids:
            raise SchemaError("%s actor is not registered" % contract_path)
    for contract in contracts:
        if contract["kind"] == "gate" and (
            geometry_sets["gates"][contract["target_gate"]]
            == geometry_sets["gates"][contract["opposing_gate"]]
        ):
            raise SchemaError("%s gate contracts resolve to identical geometry" % path)
    if contracts[0]["kind"] == contracts[1]["kind"] and contracts[0]["kind"] in {"lane", "scoped_lane"} and (
        geometry_sets["lanes"][contracts[0]["target_lane"]]
        == geometry_sets["lanes"][contracts[1]["target_lane"]]
    ):
        raise SchemaError("%s lane contracts resolve to identical geometry" % path)
    if contracts[0]["kind"] == contracts[1]["kind"] == "stop_region" and (
        geometry_sets["regions"][contracts[0]["region"]]
        == geometry_sets["regions"][contracts[1]["region"]]
    ):
        raise SchemaError("%s stop contracts resolve to identical geometry" % path)
    if layer == "C":
        visibility = _require_object(annotations.get("visibility_review"), path + ".private_annotations.visibility_review")
        _required(visibility, ("model_input_resolution", "legible", "reviewed_frame_refs"), path + ".private_annotations.visibility_review")
        if visibility["legible"] is not True or not visibility["reviewed_frame_refs"]:
            raise SchemaError("%s Layer C visual grounding is not reviewed at model resolution" % path)
        if visibility["model_input_resolution"] != [camera["height"], camera["width"]]:
            raise SchemaError("%s Layer C review resolution must match model input" % path)
        referents = _require_object(
            annotations.get("semantic_referents"),
            path + ".private_annotations.semantic_referents",
        )
        if set(referents) != set(ARMS):
            raise SchemaError("%s Layer C requires one private referent per arm" % path)
        cue_type = {
            "C1_direction_sign_grounding": "direction_sign",
            "C2_road_exit_text_grounding": "road_exit_text",
            "C3_landmark_adjacent_branch": "landmark",
            "C4_attribute_landmark_target": "attribute_landmark",
        }[family]
        referent_ids = set()
        for arm_id, contract in zip(ARMS, contracts):
            referent_path = (
                path + ".private_annotations.semantic_referents." + arm_id
            )
            referent = _require_object(referents[arm_id], referent_path)
            _required(
                referent,
                (
                    "referent_id",
                    "cue_type",
                    "target_gate",
                    "private_description",
                    "reviewed_frame_refs",
                ),
                referent_path,
            )
            if (
                not isinstance(referent["referent_id"], str)
                or not _OPAQUE_ID.fullmatch(referent["referent_id"])
            ):
                raise SchemaError("%s referent ID must be opaque" % referent_path)
            referent_ids.add(referent["referent_id"])
            if referent["cue_type"] != cue_type:
                raise SchemaError("%s cue type disagrees with task family" % referent_path)
            if referent["target_gate"] != contract["target_gate"]:
                raise SchemaError("%s target gate disagrees with contract" % referent_path)
            if not isinstance(referent["private_description"], str) or not referent[
                "private_description"
            ].strip():
                raise SchemaError("%s needs a private semantic description" % referent_path)
            frame_refs = _require_list(
                referent["reviewed_frame_refs"], referent_path + ".reviewed_frame_refs"
            )
            if not frame_refs or any(not isinstance(item, str) or not item for item in frame_refs):
                raise SchemaError("%s needs reviewed visible frames" % referent_path)
        if len(referent_ids) != len(ARMS):
            raise SchemaError("%s Layer C arm referents must be distinct" % path)


def validate_source(source: Any, require_exact_release: bool = False) -> None:
    from .release_profile import release_layer_targets
    source = _require_object(source, "source")
    try:
        expected_families = family_targets(source)
        expected_layers = release_layer_targets(source)
    except ValueError as error:
        raise SchemaError(str(error)) from error
    _required(source, ("schema_version", "pcus"), "source")
    if source["schema_version"] != SCHEMA_VERSION:
        raise SchemaError("source.schema_version is invalid")
    pcus = _require_list(source["pcus"], "source.pcus")
    ids = set()
    group_ids = set()
    layer_counts = {layer: 0 for layer in LAYER_TARGETS}
    family_counts = {family: 0 for family in FAMILY_TARGETS}
    for index, pcu in enumerate(pcus):
        validate_pcu(pcu, "source.pcus[%d]" % index)
        if require_exact_release and "oracle_plans_by_seed" not in pcu:
            raise SchemaError(
                "source.pcus[%d] is missing frozen per-seed oracle plans" % index
            )
        if pcu["pcu_id"] in ids:
            raise SchemaError("duplicate PCU id")
        if pcu["meta_group_id"] in group_ids:
            raise SchemaError("duplicate meta group id")
        ids.add(pcu["pcu_id"])
        group_ids.add(pcu["meta_group_id"])
        layer_counts[pcu["primary_layer"]] += 1
        family_counts[pcu["task_family"]] += 1
    if require_exact_release:
        if len(pcus) != sum(expected_layers.values()) or layer_counts != expected_layers or family_counts != expected_families:
            raise SchemaError(
                "release distribution mismatch: total=%d layers=%r families=%r"
                % (len(pcus), layer_counts, family_counts)
            )
