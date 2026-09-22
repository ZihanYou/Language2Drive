"""Contract-aware privileged witness for Language2Drive v1.

Unlike :mod:`team_code.autopilot`, this is a small Bench2Drive-compatible
evaluation agent, not a data-collection agent.  It follows the evaluator-only
``oracle_path`` and compiles the private target into deterministic controls for
speed corridors, delayed starts, actor relations, conditional branches,
multi-stage programs, and exact reverse parking.  The evaluator binds that
private state through a capability-gated API on Native-System only.

The agent requests only a speedometer so Bench2Drive's result schema records a
non-empty sensor declaration.  Control uses authoritative CARLA state through
``CarlaDataProvider``; this privileged witness must never be used as a
Standardized-track submission.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import carla

from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from team_code.language2drive_context import (
    L2D_TRACK_NATIVE,
    PRIVILEGED_ORACLE_CAPABILITY,
    PublicContextState,
)
from team_code.language2drive_signal_free_proof import (
    SignalFreeCorridorProofError,
    validate_signal_free_corridor_proof,
)
from team_code.language2drive_parking_manoeuvre import (
    ParkingManoeuvreProofError,
    curvature_for_normalized_steer,
    normalized_steer_for_curvature,
    validate_parking_manoeuvre,
)


def get_entry_point():
    return "Language2DriveOracleAgent"


class OracleProgramError(RuntimeError):
    """The private route cannot be executed as a deterministic witness."""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _angle_degrees(value: float, target: float) -> float:
    return abs((float(value) - float(target) + 180.0) % 360.0 - 180.0)


def _polygon_center(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise OracleProgramError("parking target requires a polygon")
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    try:
        return (
            sum(float(point[0]) for point in vertices) / len(vertices),
            sum(float(point[1]) for point in vertices) / len(vertices),
        )
    except (IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise OracleProgramError("parking polygon is invalid") from exc


def _walk_named(expression: Any, name: str) -> Iterable[Mapping[str, Any]]:
    if isinstance(expression, Mapping):
        if name in expression:
            if not isinstance(expression[name], Mapping):
                raise OracleProgramError(
                    "contract predicate {} must be an object".format(name)
                )
            yield expression[name]
        for value in expression.values():
            yield from _walk_named(value, name)
    elif isinstance(expression, list):
        for value in expression:
            yield from _walk_named(value, name)


class Language2DriveOracleAgent(autonomous_agent.AutonomousAgent):
    """Private-path, contract-aware CARLA witness agent."""

    language2drive_private_context_capability = PRIVILEGED_ORACLE_CAPABILITY

    # Conservative, fixed controls are part of the oracle implementation
    # version.  No run result is used to tune them online.
    DT = 0.05
    CRUISE_SPEED_MPS = 7.0
    MAX_SPEED_MPS = 9.0
    MAX_THROTTLE = 0.75
    MAX_BRAKE = 1.0
    MAX_STEER = 0.8
    PATH_REACHED_M = 1.4
    PATH_ENDPOINT_STOP_M = 0.35
    PARKING_V5_PRELEAD_ENDPOINT_STOP_M = 0.10
    PARKING_V5_PRELEAD_REAR_DISTANCE_MAX_M = 0.10
    PARKING_V5_PRELEAD_YAW_ERROR_MAX_DEG = 3.0
    PARKING_STAGING_ENDPOINT_STOP_M = 0.10
    PARKING_STAGING_REAR_DISTANCE_MAX_M = 0.10
    PARKING_STAGING_YAW_ERROR_MAX_DEG = 3.0
    PARKING_HEADING_COMMAND_GAIN_PER_RAD = 5.0 / 7.0
    SPEED_LIMIT_EPSILON_MPS = 1e-6
    CONSERVATIVE_CARLA_CAP_MPS = 8.333333
    CURVE_MIN_TURN_DEG = 30.0
    CURVE_PREVIEW_M = 20.0
    CURVE_PASSED_GRACE_M = 3.0
    CURVE_LATERAL_ACCEL_MPS2 = 1.8
    CURVE_APPROACH_DECEL_MPS2 = 1.5

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self.track = autonomous_agent.Track.SENSORS
        self.controller_source = "privileged_oracle:constructed"
        self._public_state = PublicContextState()
        self._public_context = None
        self._private_context: Optional[Dict[str, Any]] = None
        self._route: Optional[Dict[str, Any]] = None
        self._parking_manoeuvre_registry = None
        self._target: Optional[Dict[str, Any]] = None
        self._conditional_branch: Optional[str] = None
        self._points: list[Tuple[float, float, float]] = []
        self._cumulative: list[float] = []
        self._curve_vertices: list[Tuple[float, float, float]] = []
        self._segment = 0
        self._along = 0.0
        self._start_time: Optional[float] = None
        self._tick = 0
        self._ego = None
        self._world = None
        self._world_map = None
        self._contract_speed_band: Optional[Tuple[float, float]] = None
        self._speed_band: Optional[Tuple[float, float]] = None
        self._speed_limit_proof_mps: Optional[float] = None
        self._speed_feasibility_cap_mps: Optional[float] = None
        self._signal_free_corridor_proof: Optional[Dict[str, Any]] = None
        self._setup_speed_limit_mps: Optional[float] = None
        self._until_end: Optional[Dict[str, Any]] = None
        self._until_end_entered = False
        self._until_progress_min: Optional[float] = None
        self._until_progress_entered = False
        self._wait_seconds = 0.0
        self._parking: Optional[Dict[str, Any]] = None
        self._parking_manoeuvre: Optional[Dict[str, Any]] = None
        self._parking_state = "none"
        self._terminal_pose: Optional[Dict[str, float]] = None
        self._terminal_state = "none"
        self._speed_integral = 0.0
        self._active_stop_sign_id: Optional[int] = None
        self._stop_sign_stopped_since: Optional[float] = None
        self._cleared_stop_sign_ids: set[int] = set()
        self._passing_complete = False
        self._passing_lane_key: Optional[Tuple[int, int, int]] = None
        self._plans = None
        self._destroyed = False
        self._native_plan_length = 0
        self._effective_control_tick = -1
        self._effective_target_speed_mps: Optional[float] = None
        self._effective_lookahead_m: Optional[float] = None
        self._curve_speed_cap_mps: Optional[float] = None
        self._curve_next_distance_m: Optional[float] = None
        self._parking_stage_rear_distance_m: Optional[float] = None
        self._parking_stage_cross_track_m: Optional[float] = None
        self._parking_stage_yaw_error_deg: Optional[float] = None
        self._parking_prelead_remaining_m: Optional[float] = None
        self._parking_prelead_rear_distance_m: Optional[float] = None
        self._parking_prelead_yaw_error_deg: Optional[float] = None
        self._parking_prelead_speed_mps: Optional[float] = None
        self._parking_prelead_gate_passed = False
        self._parking_prelead_handoff_tick: Optional[int] = None

    def set_language2drive_context(self, public_context):
        context = self._public_state.bind(public_context)
        if context.track != L2D_TRACK_NATIVE:
            raise OracleProgramError(
                "Language2Drive privileged oracle is Native-System only"
            )
        self._public_context = context

    def set_language2drive_private_context(self, private_context):
        if self._public_context is None:
            raise OracleProgramError("public context must be bound first")
        if self._public_context.track != L2D_TRACK_NATIVE:
            raise OracleProgramError("private context refused outside native track")
        if not isinstance(private_context, Mapping) or not isinstance(
            private_context.get("route"), Mapping
        ):
            raise OracleProgramError("private context has no route")
        self._private_context = copy.deepcopy(dict(private_context))
        self._route = dict(self._private_context["route"])
        self._parking_manoeuvre_registry = self._private_context.get(
            "parking_manoeuvre_registry"
        )

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # Native evaluator compatibility only.  The witness intentionally does
        # not steer from this dense plan; the hidden sidecar oracle_path below
        # is the sole geometric source.
        del global_plan_gps
        self._native_plan_length = len(global_plan_world_coord or [])

    def sensors(self):
        return [{
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed",
        }]

    def __call__(self):
        """Bench2Drive call protocol without per-frame wall-clock printing."""

        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        control = self.run_step(input_data, GameTime.get_time())
        control.manual_gear_shift = False
        return control

    def setup(self, path_to_conf_file):
        del path_to_conf_file  # The evaluator-bound private program is complete.
        if self._public_context is None or self._route is None:
            raise OracleProgramError(
                "oracle setup requires public and privileged private contexts"
            )
        instruction = str(self._route.get("instruction", "")).strip()
        if instruction != self._public_context.instruction:
            raise OracleProgramError("public/private instruction mismatch")
        if self._route.get("town") is None:
            raise OracleProgramError("private route has no town")
        self._compile_program()
        self._validate_live_speed_limit()
        record_dir = os.environ.get("L2D_RECORD_DIR")
        if record_dir:
            root = Path(record_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._plans = (root / "agent_plans.jsonl").open(
                "a", encoding="utf-8"
            )
        self.controller_source = "privileged_oracle:setup"

    def _compile_program(self) -> None:
        assert self._route is not None
        try:
            raw_deadline = self._route["deadline_seconds"]
            if isinstance(raw_deadline, bool):
                raise TypeError("boolean deadline")
            deadline_seconds = float(raw_deadline)
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleProgramError(
                "deadline_seconds must be finite and positive"
            ) from exc
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0.0:
            raise OracleProgramError(
                "deadline_seconds must be finite and positive"
            )
        raw_path = self._route.get("oracle_path")
        if not isinstance(raw_path, list) or len(raw_path) < 2:
            raise OracleProgramError("oracle_path requires at least two points")
        try:
            path_values = []
            for point in raw_path:
                if not isinstance(point, Mapping) or not {
                    "x", "y", "z", "yaw_deg"
                }.issubset(point):
                    raise KeyError("x/y/z/yaw_deg")
                path_values.append((
                    float(point["x"]), float(point["y"]),
                    float(point["z"]), float(point["yaw_deg"]),
                ))
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleProgramError(
                "oracle_path points require finite x/y/z/yaw_deg"
            ) from exc
        if any(
            not all(math.isfinite(value) for value in point)
            for point in path_values
        ):
            raise OracleProgramError("oracle_path contains a non-finite point")
        initial_forward = next((
            point for point in path_values[1:]
            if math.hypot(
                point[0] - path_values[0][0],
                point[1] - path_values[0][1],
            ) > 0.05
        ), None)
        if initial_forward is None:
            raise OracleProgramError(
                "oracle_path has no meaningful initial directed segment"
            )
        initial_tangent_yaw = math.degrees(math.atan2(
            initial_forward[1] - path_values[0][1],
            initial_forward[0] - path_values[0][0],
        ))
        if _angle_degrees(path_values[0][3], initial_tangent_yaw) > 90.0:
            raise OracleProgramError(
                "oracle_path spawn yaw opposes its initial directed segment"
            )
        points = [point[:3] for point in path_values]

        target = self._route.get("target")
        if not isinstance(target, Mapping):
            raise OracleProgramError("private route has no target program")
        supported = {
            "gate", "lane", "speed_band", "progress", "delayed_progress",
            "parking_slot", "until", "headway", "actor_order", "region",
            "sequence", "if_else", "memory", "hold",
        }
        target_type = str(target.get("type", ""))
        if target_type not in supported:
            raise OracleProgramError(
                "unsupported oracle target type {!r}".format(target_type)
            )
        self._target = self._resolve_conditional_target(dict(target))
        resolved_target_type = str(self._target.get("type", ""))
        if resolved_target_type not in supported - {"if_else"}:
            raise OracleProgramError(
                "unsupported resolved oracle target type {!r}".format(
                    resolved_target_type
                )
            )
        if self._target.get("type") == "hold":
            try:
                raw_seconds = self._target["seconds"]
                if isinstance(raw_seconds, bool):
                    raise TypeError("boolean duration")
                hold_seconds = float(raw_seconds)
            except (KeyError, TypeError, ValueError) as exc:
                raise OracleProgramError(
                    "hold target requires finite positive seconds"
                ) from exc
            if not math.isfinite(hold_seconds) or hold_seconds <= 0.0:
                raise OracleProgramError(
                    "hold target requires finite positive seconds"
                )

        try:
            self._parking_manoeuvre = validate_parking_manoeuvre(
                self._route,
                self._target,
                registry=self._parking_manoeuvre_registry,
            )
        except ParkingManoeuvreProofError as exc:
            raise OracleProgramError(str(exc)) from exc

        # The direct scenario route preserves sidecar yaw at every waypoint.
        # Reject an internally reversed route before it can corrupt route
        # criteria or the oracle controller. Exact parking is the sole frozen
        # exception: its audited final approach->slot segment is intentionally
        # executed in reverse by the parking state machine.
        parking_suffix_start = (
            int(self._parking_manoeuvre["oracle_path_suffix_start"])
            if self._parking_manoeuvre is not None
            else None
        )
        for index, (first, second) in enumerate(
            zip(path_values, path_values[1:])
        ):
            # The shared v4 validator already checked every actor/rear pose,
            # gear transition, motion-direction yaw and curvature.  Generic
            # forward-route validation applies only through the physical
            # approach pose; reverse path tangents intentionally oppose the
            # vehicle yaw.
            if parking_suffix_start is not None and index >= parking_suffix_start:
                continue
            if math.hypot(second[0] - first[0], second[1] - first[1]) <= 0.05:
                continue
            tangent_yaw = math.degrees(
                math.atan2(second[1] - first[1], second[0] - first[0])
            )
            if _angle_degrees(first[3], tangent_yaw) > 90.0:
                raise OracleProgramError(
                    "oracle_path yaw opposes directed segment {}".format(index)
                )
        if parking_suffix_start is None:
            terminal_tangent_yaw = math.degrees(math.atan2(
                path_values[-1][1] - path_values[-2][1],
                path_values[-1][0] - path_values[-2][0],
            ))
            if _angle_degrees(path_values[-1][3], terminal_tangent_yaw) > 90.0:
                raise OracleProgramError(
                    "oracle_path terminal yaw opposes its directed segment"
                )

        try:
            self._signal_free_corridor_proof = (
                validate_signal_free_corridor_proof(self._route)
            )
        except SignalFreeCorridorProofError as exc:
            raise OracleProgramError(str(exc)) from exc

        bands = []
        for body in _walk_named(self._route.get("contract", {}), "speed_band"):
            try:
                band = (float(body["min_mps"]), float(body["max_mps"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise OracleProgramError("invalid speed_band predicate") from exc
            if band not in bands:
                bands.append(band)
        if len(bands) > 1:
            raise OracleProgramError("own contract has ambiguous speed bands")
        if bands:
            low, high = bands[0]
            if not (0.0 <= low < high <= self.MAX_SPEED_MPS + 1.0):
                raise OracleProgramError("unsafe or empty oracle speed band")
            try:
                raw_limit = self._route["legal_speed_mps"]
                if isinstance(raw_limit, bool):
                    raise TypeError("boolean speed limit")
                proof = float(raw_limit)
            except (KeyError, TypeError, ValueError) as exc:
                raise OracleProgramError(
                    "speed contract requires route legal_speed_mps proof"
                ) from exc
            if not math.isfinite(proof) or proof <= 0.0:
                raise OracleProgramError(
                    "route legal_speed_mps proof must be finite and positive"
                )
            raw_proof = self._route.get("legal_speed_proof")
            if not isinstance(raw_proof, Mapping):
                raise OracleProgramError(
                    "speed contract requires route legal_speed_proof"
                )
            expected_proof_keys = {
                "method", "samples", "lane_types", "sample_spacing_max_m",
                "minimum_positive_mps", "town", "xodr_sha256",
            }
            if set(raw_proof) != expected_proof_keys:
                raise OracleProgramError(
                    "legal_speed_proof keys must be exactly {}".format(
                        sorted(expected_proof_keys)
                    )
                )
            if raw_proof.get("method") != "oracle_path_exact_waypoint_min_v1":
                raise OracleProgramError(
                    "legal_speed_proof has unsupported method"
                )
            if raw_proof.get("lane_types") != ["Driving", "Parking"]:
                raise OracleProgramError(
                    "legal_speed_proof lane_types must document "
                    "Driving->Parking precedence"
                )
            try:
                raw_samples = raw_proof["samples"]
                raw_minimum = raw_proof["minimum_positive_mps"]
                raw_spacing = raw_proof["sample_spacing_max_m"]
                if (
                    not isinstance(raw_samples, int)
                    or isinstance(raw_samples, bool)
                    or isinstance(raw_minimum, bool)
                    or isinstance(raw_spacing, bool)
                ):
                    raise TypeError("invalid proof numeric type")
                samples = raw_samples
                minimum = float(raw_minimum)
                spacing = float(raw_spacing)
            except (KeyError, TypeError, ValueError) as exc:
                raise OracleProgramError(
                    "legal_speed_proof numeric fields are invalid"
                ) from exc
            if samples <= 0:
                raise OracleProgramError(
                    "legal_speed_proof samples must be a positive integer"
                )
            if not all(math.isfinite(value) for value in (minimum, spacing)):
                raise OracleProgramError(
                    "legal_speed_proof numeric fields must be finite"
                )
            if abs(minimum - proof) > 1e-6:
                raise OracleProgramError(
                    "legal_speed_proof minimum disagrees with legal_speed_mps"
                )
            if spacing <= 0.0 or spacing > 0.1 + 1e-9:
                raise OracleProgramError(
                    "legal_speed_proof sample_spacing_max_m must be in (0,0.1]"
                )
            if str(raw_proof.get("town", "")) != str(self._route.get("town", "")):
                raise OracleProgramError(
                    "legal_speed_proof town disagrees with route town"
                )
            if re.fullmatch(
                r"[0-9a-f]{64}", str(raw_proof.get("xodr_sha256", ""))
            ) is None:
                raise OracleProgramError(
                    "legal_speed_proof xodr_sha256 must be lowercase SHA256"
                )
            # The controller deliberately reserves 0.35 m/s below CARLA's
            # current limit.  A band without that frozen margin can become
            # unreachable only after rollout start, wasting the full deadline.
            if proof - high < 0.35 - 1e-9:
                raise OracleProgramError(
                    "speed band max {:.3f}m/s lacks 0.35m/s margin below "
                    "legal_speed_mps {:.3f}m/s".format(high, proof)
                )
            layer = str(self._route.get("layer", ""))
            if layer not in {"A", "B", "C", "D", "Safety"}:
                raise OracleProgramError(
                    "speed contract has unsupported layer {!r}".format(layer)
                )
            raw_cap_proof = self._route.get("speed_feasibility_cap_proof")
            if layer in {"A", "B", "C", "D"} and not isinstance(
                raw_cap_proof, Mapping
            ):
                raise OracleProgramError(
                    "core speed contract requires speed_feasibility_cap_proof"
                )
            if raw_cap_proof is not None:
                if not isinstance(raw_cap_proof, Mapping):
                    raise OracleProgramError(
                        "speed_feasibility_cap_proof must be an object"
                    )
                expected_cap_keys = {
                    "method", "cap_mps", "control_margin_mps",
                }
                if set(raw_cap_proof) != expected_cap_keys:
                    raise OracleProgramError(
                        "speed_feasibility_cap_proof keys must be exactly {}"
                        .format(sorted(expected_cap_keys))
                    )
                if raw_cap_proof.get("method") != (
                    "carla_default_limit_conservative_cap_v1"
                ):
                    raise OracleProgramError(
                        "speed_feasibility_cap_proof has unsupported method"
                    )
                try:
                    raw_cap = raw_cap_proof["cap_mps"]
                    raw_margin = raw_cap_proof["control_margin_mps"]
                    if isinstance(raw_cap, bool) or isinstance(raw_margin, bool):
                        raise TypeError("boolean feasibility proof value")
                    cap = float(raw_cap)
                    margin = float(raw_margin)
                except (KeyError, TypeError, ValueError) as exc:
                    raise OracleProgramError(
                        "speed_feasibility_cap_proof numeric fields are invalid"
                    ) from exc
                if not all(math.isfinite(value) for value in (cap, margin)):
                    raise OracleProgramError(
                        "speed_feasibility_cap_proof values must be finite"
                    )
                if (
                    abs(cap - self.CONSERVATIVE_CARLA_CAP_MPS) > 1e-9
                    or abs(margin - 0.35) > 1e-9
                ):
                    raise OracleProgramError(
                        "speed_feasibility_cap_proof requires cap_mps="
                        "8.333333 and control_margin_mps=0.35"
                    )
                if high + margin > cap + 1e-9:
                    raise OracleProgramError(
                        "speed band max {:.3f}m/s plus {:.2f}m/s margin "
                        "exceeds conservative CARLA cap {:.6f}m/s".format(
                            high, margin, cap
                        )
                    )
                self._speed_feasibility_cap_mps = cap
            self._speed_limit_proof_mps = proof
            self._contract_speed_band = (low, high)
            self._speed_band = (low, high)

            target_band = None
            if self._target.get("type") == "speed_band":
                target_band = self._target
            elif self._target.get("type") == "until":
                target_band = self._target.get("constraint")
            if target_band is not None:
                try:
                    encoded = (
                        float(target_band["min_mps"]),
                        float(target_band["max_mps"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise OracleProgramError(
                        "speed target constraint is malformed"
                    ) from exc
                if any(
                    abs(observed - expected) > 1e-9
                    for observed, expected in zip(encoded, (low, high))
                ):
                    raise OracleProgramError(
                        "speed target constraint disagrees with own contract"
                    )

            if self._target.get("type") == "until":
                end = self._target.get("end")
                if not isinstance(end, Mapping) or end.get("type") != "gate":
                    raise OracleProgramError(
                        "speed until target requires an end gate"
                    )
                has_circle = (
                    isinstance(end.get("center"), Mapping)
                    and end.get("radius_m") is not None
                )
                has_polygon = isinstance(end.get("polygon"), list)
                if has_circle == has_polygon:
                    raise OracleProgramError(
                        "speed until end gate requires exactly one geometry"
                    )
                self._until_end = copy.deepcopy(dict(end))

            # Layer-A speed commitments end on normalized progress rather
            # than a geometry gate.  Do not begin generic terminal braking
            # while their ``until`` invariant is still active: doing so can
            # drive a valid high-speed witness below its band just before the
            # 0.9 progress edge.  Bind the controller latch to the exact own
            # contract condition, as the gate-based B/D targets do above.
            if self._target.get("type") == "speed_band":
                progress_thresholds = []
                for until_body in _walk_named(
                    self._route.get("contract", {}), "until"
                ):
                    invariant_bands = list(_walk_named(
                        until_body.get("invariant", {}), "speed_band"
                    ))
                    matches_band = False
                    for invariant_band in invariant_bands:
                        try:
                            encoded_invariant = (
                                float(invariant_band["min_mps"]),
                                float(invariant_band["max_mps"]),
                            )
                        except (KeyError, TypeError, ValueError) as exc:
                            raise OracleProgramError(
                                "speed until invariant is malformed"
                            ) from exc
                        if all(
                            abs(observed - expected) <= 1e-9
                            for observed, expected in zip(
                                encoded_invariant, (low, high)
                            )
                        ):
                            matches_band = True
                    if not matches_band:
                        continue
                    progress_conditions = list(_walk_named(
                        until_body.get("condition", {}), "progress"
                    ))
                    if len(progress_conditions) != 1:
                        raise OracleProgramError(
                            "speed until requires exactly one progress condition"
                        )
                    try:
                        raw_progress = progress_conditions[0]["min"]
                        if isinstance(raw_progress, bool):
                            raise TypeError("boolean progress threshold")
                        progress_min = float(raw_progress)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise OracleProgramError(
                            "speed until progress condition is malformed"
                        ) from exc
                    if not math.isfinite(progress_min) or not (
                        0.0 < progress_min <= 1.0
                    ):
                        raise OracleProgramError(
                            "speed until progress min must be in (0,1]"
                        )
                    progress_thresholds.append(progress_min)
                if progress_thresholds:
                    unique_thresholds = {
                        round(value, 12) for value in progress_thresholds
                    }
                    if len(unique_thresholds) != 1:
                        raise OracleProgramError(
                            "speed target has ambiguous until-progress thresholds"
                        )
                    self._until_progress_min = progress_thresholds[0]

        if target_type == "delayed_progress":
            raw_wait_seconds = target.get("wait_seconds", 0.0)
            try:
                if isinstance(raw_wait_seconds, bool):
                    raise TypeError("boolean wait")
                self._wait_seconds = float(raw_wait_seconds)
            except (TypeError, ValueError) as exc:
                raise OracleProgramError(
                    "delayed_progress requires finite positive wait"
                ) from exc
            if not math.isfinite(self._wait_seconds) or self._wait_seconds <= 0:
                raise OracleProgramError(
                    "delayed_progress requires finite positive wait"
                )
            # The 0--0.2 m/s band belongs to the wait stage, not the subsequent
            # progress stage. The explicit brake branch enforces it exactly.
            self._speed_band = None

        if self._target.get("type") == "parking_slot":
            if self._parking_manoeuvre is None:
                raise OracleProgramError(
                    "parking target has no compiled reverse_manoeuvre v4"
                )
            self._compile_parking(
                points,
                self._target,
                raw_path,
                self._parking_manoeuvre,
            )
        elif self._target.get("type") == "region":
            self._compile_region(points, self._target, raw_path)
        else:
            self._set_path(points)

        if target_type in {"headway", "actor_order"}:
            actor_id = self._target.get("actor_id")
            setup_ids = {
                str(spec.get("id"))
                for spec in self._route.get("actor_setup", {}).get("actors", [])
                if isinstance(spec, Mapping)
            }
            if not isinstance(actor_id, str) or actor_id not in setup_ids:
                raise OracleProgramError(
                    "actor-relation target has no matching actor program"
                )
        if self._target.get("type") == "headway":
            try:
                raw_minimum = self._target["min_seconds"]
                raw_maximum = self._target["max_seconds"]
                if isinstance(raw_minimum, bool) or isinstance(
                    raw_maximum, bool
                ):
                    raise TypeError("boolean headway")
                minimum = float(raw_minimum)
                maximum = float(raw_maximum)
            except (KeyError, TypeError, ValueError) as exc:
                raise OracleProgramError(
                    "headway target requires finite min/max seconds"
                ) from exc
            if not (
                math.isfinite(minimum)
                and math.isfinite(maximum)
                and 0.0 < minimum < maximum
            ):
                raise OracleProgramError(
                    "headway target requires 0 < min_seconds < max_seconds"
                )
        if (
            self._target.get("type") == "actor_order"
            and self._target.get("order") not in {"ego_ahead", "ego_behind"}
        ):
            raise OracleProgramError("actor_order requires ego_ahead or ego_behind")
        if (
            self._target.get("type") == "actor_order"
            and self._target.get("order") == "ego_ahead"
            and self._target.get("pass_lane_id") is not None
        ):
            try:
                pass_lane_id = int(self._target["pass_lane_id"])
            except (TypeError, ValueError) as exc:
                raise OracleProgramError("pass_lane_id must be an integer") from exc
            if pass_lane_id == 0:
                raise OracleProgramError("pass_lane_id cannot be zero")
            self._target["pass_lane_id"] = pass_lane_id

    def _validate_live_speed_limit(self) -> None:
        """Cross-check the contract band against the spawned CARLA hero.

        ``legal_speed_mps`` is the generator-audited minimum along the entire
        oracle path.  CARLA's live vehicle limit can be lower than that XODR
        value (custom maps commonly expose a conservative 30 km/h default),
        which is not itself a contradiction.  The witness is feasible only
        when the *contract band* retains the frozen 0.35 m/s control margin
        below both limits.  A missing/zero live limit still fails closed.
        """

        if self._speed_limit_proof_mps is None:
            return
        if self._contract_speed_band is None:
            raise OracleProgramError(
                "frozen speed proof has no compiled speed band"
            )
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            world = CarlaDataProvider.get_world()
            candidates = [] if world is None else [
                actor
                for actor in world.get_actors()
                if getattr(actor, "attributes", {}).get("role_name") == "hero"
            ]
            if len(candidates) == 1:
                hero = candidates[0]
        if hero is None:
            raise OracleProgramError(
                "speed proof cannot be checked: CARLA hero actor is unavailable"
            )
        actual = self._validate_live_band_limit(
            hero, self._contract_speed_band, "hero map"
        )
        self._setup_speed_limit_mps = actual

    def _validate_live_band_limit(
        self, actor, band: Tuple[float, float], label: str,
    ) -> float:
        try:
            actual = float(actor.get_speed_limit()) / 3.6
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise OracleProgramError(
                "speed proof cannot be checked: live {} speed limit "
                "unavailable".format(label)
            ) from exc
        if not math.isfinite(actual) or actual <= 0.0:
            raise OracleProgramError(
                "speed proof cannot be checked: live {} speed limit is "
                "non-positive".format(label)
            )
        if band[1] + 0.35 > actual + self.SPEED_LIMIT_EPSILON_MPS:
            raise OracleProgramError(
                "speed band max {:.3f}m/s lacks 0.35m/s margin below live "
                "{} limit {:.3f}m/s".format(band[1], label, actual)
            )
        return actual

    def _resolve_conditional_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        if target.get("type") != "if_else":
            return target
        assert self._route is not None
        condition = target.get("condition")
        if not isinstance(condition, str) or not condition.startswith("occupied:"):
            raise OracleProgramError("if_else oracle supports occupied:<slot> only")
        slot = condition.split(":", 1)[1]
        setup = self._route.get("actor_setup", {})
        declared = setup.get("slot_occupancy", {}).get(slot)
        if not isinstance(declared, bool):
            raise OracleProgramError("if_else has no boolean slot occupancy")
        spawned = any(
            isinstance(spec, Mapping) and str(spec.get("slot")) == slot
            for spec in setup.get("actors", [])
        )
        if spawned != declared:
            raise OracleProgramError(
                "if_else occupancy disagrees with evaluator actor program"
            )
        branch = "then" if declared else "else"
        selected = target.get(branch)
        if not isinstance(selected, Mapping):
            raise OracleProgramError("if_else selected branch has no target")
        self._conditional_branch = branch
        return dict(selected)

    def _compile_parking(
        self,
        points: list[Tuple[float, float, float]],
        target: Mapping[str, Any],
        raw_path: Sequence[Mapping[str, Any]],
        manoeuvre: Mapping[str, Any],
    ) -> None:
        del raw_path  # The shared validator closed its exact suffix already.
        center = tuple(manoeuvre["slot_center"])
        try:
            heading = float(target["heading_deg"])
            raw_dwell = target.get("dwell_seconds", 3.0)
            if isinstance(raw_dwell, bool):
                raise TypeError("boolean dwell")
            dwell = float(raw_dwell)
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleProgramError("parking target lacks heading/dwell") from exc
        if not math.isfinite(heading):
            raise OracleProgramError("parking heading_deg must be finite")
        if not math.isfinite(dwell) or dwell <= 0.0:
            raise OracleProgramError(
                "parking dwell_seconds must be finite and positive"
            )
        if math.hypot(points[-1][0] - center[0], points[-1][1] - center[1]) > 1.0:
            raise OracleProgramError(
                "parking oracle_path does not terminate in selected branch slot"
            )
        yaw = math.radians(heading)
        is_v5 = bool(manoeuvre.get("is_v5"))
        slot_axis_approach_pose = dict(manoeuvre["approach_pose"])
        ingress = manoeuvre.get("v5_ingress")
        if is_v5:
            if not isinstance(ingress, Mapping):
                raise OracleProgramError("parking v5 has no validated ingress")
            approach_pose = dict(ingress["prelead_pose"])
            prelead_rear_pose = dict(ingress["prelead_rear_pose"])
            approach_pose_source = (
                "reverse_manoeuvre.poses[0]:v5_stored_xodr_prelead"
            )
        else:
            approach_pose = slot_axis_approach_pose
            prelead_rear_pose = dict(manoeuvre["rear_axle_poses"][0])
            approach_pose_source = "target.approach_pose:v4_physical_lane"
        approach = (
            approach_pose["x"], approach_pose["y"], approach_pose["z"]
        )
        suffix_start = int(manoeuvre["oracle_path_suffix_start"])
        if suffix_start < 1:
            raise OracleProgramError(
                "parking oracle_path has no forward approach prefix"
            )
        approach_path = list(points[:suffix_start + 1])
        if math.sqrt(sum(
            (approach_path[-1][axis] - approach[axis]) ** 2
            for axis in range(3)
        )) > 1e-6:
            raise OracleProgramError(
                "parking forward prefix does not end at validated handoff pose"
            )

        poses = [dict(pose) for pose in manoeuvre["poses"]]
        rear_poses = [dict(pose) for pose in manoeuvre["rear_axle_poses"]]
        transition = int(manoeuvre["transition_index"])
        forward_poses = poses[:transition + 1]
        reverse_poses = poses[transition + 1:]
        forward_rear_poses = rear_poses[:transition + 1]
        reverse_rear_poses = rear_poses[transition + 1:]
        if (
            len(forward_poses) < 2
            or len(reverse_poses) < 2
            or len(forward_rear_poses) != len(forward_poses)
            or len(reverse_rear_poses) != len(reverse_poses)
        ):
            raise OracleProgramError(
                "parking manoeuvre requires nonempty forward staging and reverse paths"
            )
        self._parking = {
            "center": center,
            "heading_deg": heading,
            "heading_rad": yaw,
            "dwell_seconds": dwell,
            "schema_version": manoeuvre["schema_version"],
            "is_v5": is_v5,
            "approach": approach,
            "approach_yaw_deg": approach_pose["yaw_deg"],
            "approach_pose_source": approach_pose_source,
            "slot_axis_approach_pose": slot_axis_approach_pose,
            "prelead_rear": tuple(
                prelead_rear_pose[axis] for axis in ("x", "y", "z")
            ),
            "stage": tuple(
                forward_poses[-1][axis] for axis in ("x", "y", "z")
            ),
            "stage_yaw_deg": forward_poses[-1]["yaw_deg"],
            "stage_rear": tuple(
                forward_rear_poses[-1][axis] for axis in ("x", "y", "z")
            ),
            "forward_poses": forward_poses,
            "reverse_poses": reverse_poses,
            "forward_rear_poses": forward_rear_poses,
            "reverse_rear_poses": reverse_rear_poses,
            "proof": copy.deepcopy(manoeuvre["proof"]),
            "control_model": copy.deepcopy(manoeuvre["control_model"]),
            "normalized_steer_samples": copy.deepcopy(
                manoeuvre["normalized_steer_samples"]
            ),
            "polygon": copy.deepcopy(target.get("polygon")),
        }
        self._parking_prelead_remaining_m = None
        self._parking_prelead_rear_distance_m = None
        self._parking_prelead_yaw_error_deg = None
        self._parking_prelead_speed_mps = None
        self._parking_prelead_gate_passed = False
        self._parking_prelead_handoff_tick = None
        self._parking_state = "approach"
        # Preserve the exact final prelead while still filtering directionless
        # sub-5cm GRP de-duplication connectors elsewhere in the route.  A
        # generic cleanup that discards the final <=5cm point moves the live
        # capture endpoint outside the v5 10cm rear-axle contract.
        self._set_path(approach_path, preserve_endpoint=True)
        if math.sqrt(sum(
            (self._points[-1][axis] - approach[axis]) ** 2
            for axis in range(3)
        )) > 1e-9:
            raise OracleProgramError(
                "parking compiled path did not preserve its exact handoff endpoint"
            )

    def _compile_region(
        self,
        points: list[Tuple[float, float, float]],
        target: Mapping[str, Any],
        raw_path: Sequence[Mapping[str, Any]],
    ) -> None:
        pose = target.get("terminal_pose")
        if not isinstance(pose, Mapping):
            raise OracleProgramError(
                "region witness requires an audited terminal_pose"
            )
        if set(pose) != {"x", "y", "z", "yaw_deg"}:
            raise OracleProgramError(
                "region terminal_pose requires exactly x/y/z/yaw_deg"
            )
        try:
            terminal = {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose["z"]),
                "yaw_deg": float(pose["yaw_deg"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleProgramError("region terminal_pose is invalid") from exc
        if not all(math.isfinite(value) for value in terminal.values()):
            raise OracleProgramError("region terminal_pose is non-finite")
        endpoint_error = math.sqrt(sum(
            (points[-1][axis] - terminal[key]) ** 2
            for axis, key in enumerate(("x", "y", "z"))
        ))
        if endpoint_error > 0.25:
            raise OracleProgramError(
                "region oracle_path endpoint is {:.3f}m from terminal_pose".format(
                    endpoint_error
                )
            )
        try:
            endpoint_yaw = float(raw_path[-1]["yaw_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OracleProgramError(
                "region oracle_path terminal requires yaw_deg"
            ) from exc
        if not math.isfinite(endpoint_yaw):
            raise OracleProgramError("region oracle_path terminal yaw is non-finite")
        if _angle_degrees(endpoint_yaw, terminal["yaw_deg"]) > 1.0:
            raise OracleProgramError(
                "region oracle_path terminal yaw disagrees with terminal_pose"
            )
        self._terminal_pose = terminal
        self._terminal_state = "approach"
        self._set_path(points)

    def _set_path(
        self, points: list[Tuple[float, float, float]],
        *, minimum_spacing_m: float = 0.05,
        preserve_endpoint: bool = False,
    ) -> None:
        cleaned = [points[0]]
        for index, point in enumerate(points[1:], 1):
            spacing = math.hypot(
                point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]
            )
            if spacing > float(minimum_spacing_m) or (
                preserve_endpoint
                and index == len(points) - 1
                and spacing > 1e-9
            ):
                cleaned.append(point)
        if len(cleaned) < 2:
            raise OracleProgramError("oracle path collapses to one point")
        cumulative = [0.0]
        for first, second in zip(cleaned, cleaned[1:]):
            cumulative.append(
                cumulative[-1]
                + math.hypot(second[0] - first[0], second[1] - first[1])
            )
        self._points = cleaned
        self._cumulative = cumulative
        curve_vertices = []
        for index in range(1, len(cleaned) - 1):
            first, middle, last = (
                cleaned[index - 1], cleaned[index], cleaned[index + 1]
            )
            before = math.hypot(
                middle[0] - first[0], middle[1] - first[1]
            )
            after = math.hypot(last[0] - middle[0], last[1] - middle[1])
            first_yaw = math.atan2(
                middle[1] - first[1], middle[0] - first[0]
            )
            second_yaw = math.atan2(
                last[1] - middle[1], last[0] - middle[0]
            )
            turn = abs(_angle(second_yaw - first_yaw))
            if math.degrees(turn) < self.CURVE_MIN_TURN_DEG:
                continue
            curvature = turn / max(0.5, 0.5 * (before + after))
            vertex_speed = _clamp(
                math.sqrt(self.CURVE_LATERAL_ACCEL_MPS2 / curvature),
                1.8,
                3.5,
            )
            curve_vertices.append((cumulative[index], vertex_speed, turn))
        self._curve_vertices = curve_vertices
        if self._speed_band is not None and curve_vertices:
            minimum_band_speed = self._speed_band[0]
            minimum_curve_speed = min(item[1] for item in curve_vertices)
            if minimum_curve_speed + 0.15 < minimum_band_speed:
                raise OracleProgramError(
                    "speed band minimum {:.3f}m/s conflicts with private-path "
                    "curvature cap {:.3f}m/s".format(
                        minimum_band_speed, minimum_curve_speed
                    )
                )
        self._segment = 0
        self._along = 0.0

    def _curve_control_limits(
        self, along: float, requested_speed: float, base_lookahead: float,
    ) -> Tuple[float, float]:
        """Preview frozen polyline turns without slowing long straight legs."""

        speed_cap = float(requested_speed)
        lookahead = float(base_lookahead)
        raw_curve_cap = None
        next_curve_distance = None
        for vertex_along, vertex_speed, _ in self._curve_vertices:
            distance = vertex_along - float(along)
            if (
                distance < -self.CURVE_PASSED_GRACE_M
                or distance > self.CURVE_PREVIEW_M
            ):
                continue
            approach = max(0.0, distance - 1.0)
            reachable = math.sqrt(
                vertex_speed * vertex_speed
                + 2.0 * self.CURVE_APPROACH_DECEL_MPS2 * approach
            )
            raw_curve_cap = (
                reachable
                if raw_curve_cap is None
                else min(raw_curve_cap, reachable)
            )
            next_curve_distance = (
                distance
                if next_curve_distance is None
                else min(next_curve_distance, distance)
            )
            speed_cap = min(speed_cap, reachable)
            if distance <= 12.0:
                curve_lookahead = 2.0 + 0.1 * min(
                    10.0, max(0.0, distance)
                )
                lookahead = min(lookahead, curve_lookahead)
        self._curve_speed_cap_mps = raw_curve_cap
        self._curve_next_distance_m = next_curve_distance
        return speed_cap, lookahead

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, polygon: Sequence[Sequence[float]],
    ) -> bool:
        inside = False
        vertices = (
            polygon[:-1]
            if polygon and polygon[0] == polygon[-1]
            else polygon
        )
        for first, second in zip(vertices, vertices[1:] + vertices[:1]):
            x1, y1 = float(first[0]), float(first[1])
            x2, y2 = float(second[0]), float(second[1])
            if (y1 > y) != (y2 > y):
                crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < crossing:
                    inside = not inside
        return inside

    def _update_until_end(self, location) -> bool:
        if self._until_end is None or self._until_end_entered:
            return True
        x, y = float(location.x), float(location.y)
        center = self._until_end.get("center")
        if isinstance(center, Mapping):
            try:
                radius = float(self._until_end["radius_m"])
                entered = math.hypot(
                    x - float(center["x"]), y - float(center["y"])
                ) <= radius
            except (KeyError, TypeError, ValueError) as exc:
                raise OracleProgramError(
                    "speed until circle gate is malformed"
                ) from exc
        else:
            polygon = self._until_end.get("polygon")
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise OracleProgramError(
                    "speed until polygon gate is malformed"
                )
            entered = self._point_in_polygon(x, y, polygon)
        self._until_end_entered = bool(entered)
        return self._until_end_entered

    def _update_until_progress(self, along: float) -> bool:
        if self._until_progress_min is None or self._until_progress_entered:
            return True
        total = self._cumulative[-1]
        if not math.isfinite(total) or total <= 1e-6:
            raise OracleProgramError(
                "speed until-progress requires a positive private path length"
            )
        progress = _clamp(float(along) / total, 0.0, 1.0)
        if progress + 1e-9 >= self._until_progress_min:
            self._until_progress_entered = True
        return self._until_progress_entered

    def _project(self, location) -> Tuple[float, float]:
        start = max(0, self._segment - 1)
        stop = min(len(self._points) - 1, self._segment + 14)
        best = None
        for index in range(start, stop):
            first, second = self._points[index], self._points[index + 1]
            dx, dy = second[0] - first[0], second[1] - first[1]
            denominator = dx * dx + dy * dy
            ratio = 0.0 if denominator <= 1e-9 else (
                (float(location.x) - first[0]) * dx
                + (float(location.y) - first[1]) * dy
            ) / denominator
            ratio = _clamp(ratio, 0.0, 1.0)
            qx, qy = first[0] + ratio * dx, first[1] + ratio * dy
            distance = math.hypot(float(location.x) - qx, float(location.y) - qy)
            along = self._cumulative[index] + ratio * (
                self._cumulative[index + 1] - self._cumulative[index]
            )
            candidate = (distance, index, along)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise OracleProgramError("cannot project ego onto oracle path")
        _, index, along = best
        if along + 2.0 >= self._along:
            self._segment = max(self._segment, index)
            self._along = max(self._along, along)
        return self._along, best[0]

    def _point_at(self, along: float) -> Tuple[float, float, float]:
        distance = _clamp(along, 0.0, self._cumulative[-1])
        index = self._segment
        while index + 1 < len(self._cumulative) and self._cumulative[index + 1] < distance:
            index += 1
        index = min(index, len(self._points) - 2)
        span = self._cumulative[index + 1] - self._cumulative[index]
        ratio = 0.0 if span <= 1e-9 else (distance - self._cumulative[index]) / span
        first, second = self._points[index], self._points[index + 1]
        return tuple(first[axis] + ratio * (second[axis] - first[axis]) for axis in range(3))

    @staticmethod
    def _rear_axle_location(transform, rear_axle_x_m: float):
        """Project a live CARLA actor transform to the rear-axle centre."""

        yaw = math.radians(float(transform.rotation.yaw))
        return SimpleNamespace(
            x=(
                float(transform.location.x)
                + float(rear_axle_x_m) * math.cos(yaw)
            ),
            y=(
                float(transform.location.y)
                + float(rear_axle_x_m) * math.sin(yaw)
            ),
            z=float(transform.location.z),
        )

    @staticmethod
    def _parking_path_curvature(
        poses: Sequence[Mapping[str, Any]], desired_index: int, reverse: bool,
    ) -> float:
        """Return one proof-audited signed rear-axle segment curvature."""

        first_index = min(max(0, int(desired_index)), len(poses) - 2)
        last_index = first_index + 1
        first = poses[first_index]
        last = poses[last_index]
        distance = math.hypot(
            float(last["x"]) - float(first["x"]),
            float(last["y"]) - float(first["y"]),
        )
        if distance <= 1e-9:
            raise OracleProgramError(
                "parking rear-axle poses cannot define local curvature"
            )
        signed_distance = -distance if reverse else distance
        yaw_step = _angle(
            math.radians(float(last["yaw_deg"]))
            - math.radians(float(first["yaw_deg"]))
        )
        return yaw_step / signed_distance

    def _parking_steer_for_curvature(self, curvature: float) -> float:
        assert self._parking is not None
        samples = self._parking["normalized_steer_samples"]
        proof = self._parking["proof"]
        try:
            result = normalized_steer_for_curvature(curvature, samples)
        except ParkingManoeuvreProofError as exc:
            raise OracleProgramError(str(exc)) from exc
        required = float(proof["required_max_abs_normalized_steer"])
        if abs(result) > required + 1e-9:
            raise OracleProgramError(
                "parking path feedforward exceeds its proof-bound steer"
            )
        return result

    def _parking_pursuit_steer(self, curvature: float) -> float:
        assert self._parking is not None
        samples = self._parking["normalized_steer_samples"]
        try:
            lower = curvature_for_normalized_steer(-self.MAX_STEER, samples)
            upper = curvature_for_normalized_steer(self.MAX_STEER, samples)
            return normalized_steer_for_curvature(
                _clamp(curvature, lower, upper), samples
            )
        except ParkingManoeuvreProofError as exc:
            raise OracleProgramError(str(exc)) from exc

    def _ego_actor(self):
        if self._ego is None or not bool(getattr(self._ego, "is_alive", True)):
            self._ego = CarlaDataProvider.get_hero_actor()
        if self._ego is None:
            raise OracleProgramError("CARLA hero actor is unavailable")
        if self._world is None:
            self._world = self._ego.get_world()
            self._world_map = self._world.get_map()
        return self._ego

    def _semantic_actor(self, identifier: str):
        assert self._world is not None
        matches = []
        for actor in self._world.get_actors():
            attributes = getattr(actor, "attributes", {})
            if isinstance(attributes, Mapping) and attributes.get("role_name") == "l2d:{}".format(identifier):
                matches.append(actor)
        if len(matches) != 1:
            raise OracleProgramError(
                "semantic actor {} matched {} live actors".format(identifier, len(matches))
            )
        return matches[0]

    @staticmethod
    def _speed(actor) -> float:
        velocity = actor.get_velocity()
        return math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def _relation_speed(self, ego, base_speed: float) -> float:
        assert self._target is not None
        target_type = self._target.get("type")
        if target_type not in {"headway", "actor_order"}:
            return base_speed
        actor = self._semantic_actor(str(self._target["actor_id"]))
        transform = ego.get_transform()
        delta = actor.get_location() - transform.location
        forward = transform.get_forward_vector()
        longitudinal = float(delta.x) * float(forward.x) + float(delta.y) * float(forward.y)
        actor_speed = self._speed(actor)
        if target_type == "headway":
            desired_seconds = 0.5 * (
                float(self._target["min_seconds"])
                + float(self._target["max_seconds"])
            )
            desired_gap = desired_seconds * max(actor_speed, 2.0)
            if longitudinal < 3.5:
                return 0.0
            return _clamp(
                actor_speed + 0.35 * (longitudinal - desired_gap),
                0.5,
                self.MAX_SPEED_MPS,
            )
        if self._target.get("order") == "ego_behind":
            if longitudinal < 5.0:
                return max(0.0, actor_speed - 0.5)
            return min(base_speed, max(1.0, actor_speed))
        if self._target.get("order") != "ego_ahead":
            raise OracleProgramError("unknown actor_order program")
        if longitudinal < -6.0:
            self._passing_complete = True
        return base_speed if self._passing_complete else min(self.MAX_SPEED_MPS, 7.0)

    @staticmethod
    def _lane_key(waypoint) -> Tuple[int, int, int]:
        try:
            return (
                int(waypoint.road_id),
                int(waypoint.section_id),
                int(waypoint.lane_id),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise OracleProgramError(
                "pass waypoint lacks stable road/section/lane identity"
            ) from exc

    @staticmethod
    def _lane_change_allows(waypoint, side: str) -> bool:
        """Interpret CARLA's LaneChange bit flag without permissive fallback."""

        value = getattr(waypoint, "lane_change", None)
        flag_type = getattr(carla, "LaneChange", None)
        flag = getattr(flag_type, side.capitalize(), None)
        if value is not None and flag is not None:
            try:
                return bool(int(value) & int(flag))
            except (TypeError, ValueError):
                try:
                    return bool(value & flag)
                except TypeError:
                    pass
        text = str(value).casefold()
        return "both" in text or side.casefold() in text

    @staticmethod
    def _same_direction_driving(candidate, source) -> bool:
        lane_type = str(getattr(candidate, "lane_type", "")).casefold()
        if not lane_type.endswith("driving"):
            return False
        try:
            if int(candidate.lane_id) * int(source.lane_id) <= 0:
                return False
            yaw_error = _angle_degrees(
                candidate.transform.rotation.yaw,
                source.transform.rotation.yaw,
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return yaw_error < 60.0

    def _forward_on_locked_lane(self, waypoint):
        assert self._passing_lane_key is not None
        forward = [
            candidate
            for candidate in waypoint.next(5.0)
            if self._lane_key(candidate) == self._passing_lane_key
        ]
        if len(forward) != 1:
            raise OracleProgramError(
                "locked pass lane {} has {} deterministic forward waypoints".format(
                    self._passing_lane_key, len(forward)
                )
            )
        return forward[0]

    def _adjacent_pass_target(self, ego, fallback: Tuple[float, float, float]):
        assert self._target is not None
        if self._target.get("type") != "actor_order" or self._target.get("order") != "ego_ahead" or self._passing_complete:
            return fallback
        waypoint = self._world_map.get_waypoint(ego.get_location())
        if waypoint is None:
            raise OracleProgramError("pass contract has no live ego waypoint")

        if (
            self._passing_lane_key is not None
            and self._lane_key(waypoint) == self._passing_lane_key
        ):
            candidate = waypoint
        else:
            candidates = []
            for side, method in (
                ("left", "get_left_lane"),
                ("right", "get_right_lane"),
            ):
                if self._passing_lane_key is None and not self._lane_change_allows(
                    waypoint, side
                ):
                    continue
                candidate = getattr(waypoint, method)()
                if candidate is None or not self._same_direction_driving(
                    candidate, waypoint
                ):
                    continue
                key = self._lane_key(candidate)
                if self._passing_lane_key is not None and key != self._passing_lane_key:
                    continue
                frozen_lane_id = self._target.get("pass_lane_id")
                if frozen_lane_id is not None and key[2] != int(frozen_lane_id):
                    continue
                candidates.append((method, key, candidate))
            if not candidates:
                if self._passing_lane_key is None and self._target.get(
                    "pass_lane_id"
                ) is not None:
                    raise OracleProgramError(
                        "frozen pass_lane_id {} is not a legal adjacent lane".format(
                            self._target["pass_lane_id"]
                        )
                    )
                raise OracleProgramError(
                    "locked pass lane is no longer adjacent"
                    if self._passing_lane_key is not None
                    else "pass contract has no lane-change-permitted adjacent lane"
                )
            # Method name makes the left-before-right choice explicit and
            # deterministic when both changes are legal.
            candidates.sort(key=lambda item: 0 if item[0] == "get_left_lane" else 1)
            _, selected_key, candidate = candidates[0]
            if self._passing_lane_key is None:
                self._passing_lane_key = selected_key

        candidate = self._forward_on_locked_lane(candidate)
        location = candidate.transform.location
        return (float(location.x), float(location.y), float(location.z))

    def _red_light(self, ego) -> bool:
        try:
            state = ego.get_traffic_light_state()
            affected = bool(ego.is_at_traffic_light()) and "red" in str(state).casefold()
            if not affected:
                return False
            waypoint = self._world_map.get_waypoint(ego.get_location())
            # A signal can change after the vehicle legally entered the
            # junction. Braking inside it creates a deadlock and is not the
            # semantics of a red-light stop; clear the intersection instead.
            return not bool(getattr(waypoint, "is_junction", False))
        except (AttributeError, RuntimeError):
            return False

    def _obstacle_hazard(self, ego) -> Optional[str]:
        """Return a nearby forward vehicle/walker hazard, if any.

        This is deliberately a conservative last-resort layer. Contract actor
        programs still determine headway/order target speeds, but no witness is
        allowed to trade a requested relation for a collision.
        """

        assert self._world is not None
        transform = ego.get_transform()
        origin = transform.location
        forward = transform.get_forward_vector()
        right_x, right_y = -float(forward.y), float(forward.x)
        ego_speed = self._speed(ego)
        for actor in self._world.get_actors():
            if actor.id == ego.id or not bool(getattr(actor, "is_alive", True)):
                continue
            type_id = str(getattr(actor, "type_id", ""))
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
                continue
            # The frozen ego-ahead program deliberately changes to a checked,
            # same-direction adjacent lane to pass its semantic lead vehicle.
            # Treating that exact actor as a generic obstacle would command
            # zero speed before the lane change can begin.  No other actor is
            # exempt, and the exemption ends as soon as the pass is complete.
            attributes = getattr(actor, "attributes", {})
            if (
                not self._passing_complete
                and self._passing_lane_key is not None
                and self._target is not None
                and self._target.get("type") == "actor_order"
                and self._target.get("order") == "ego_ahead"
                and isinstance(attributes, Mapping)
                and attributes.get("role_name")
                == "l2d:{}".format(self._target.get("actor_id"))
            ):
                continue
            location = actor.get_location()
            dx = float(location.x) - float(origin.x)
            dy = float(location.y) - float(origin.y)
            longitudinal = dx * float(forward.x) + dy * float(forward.y)
            lateral = abs(dx * right_x + dy * right_y)
            if longitudinal <= 0.0:
                continue
            threshold = (
                5.0 + 1.4 * ego_speed
                if type_id.startswith("walker.")
                else 3.0 + 1.2 * ego_speed
            )
            if longitudinal < threshold and lateral < 2.2:
                return "walker" if type_id.startswith("walker.") else "vehicle"
        return None

    def _stop_sign_hazard(self, ego, now: float) -> bool:
        """Perform a deterministic one-second stop at an upcoming stop sign."""

        assert self._world is not None
        transform = ego.get_transform()
        origin = transform.location
        forward = transform.get_forward_vector()
        right_x, right_y = -float(forward.y), float(forward.x)
        candidates = []
        for actor in self._world.get_actors():
            if not str(getattr(actor, "type_id", "")).startswith("traffic.stop"):
                continue
            actor_id = int(actor.id)
            if actor_id in self._cleared_stop_sign_ids:
                continue
            try:
                trigger = actor.trigger_volume
                location = carla.Location(
                    x=float(trigger.location.x),
                    y=float(trigger.location.y),
                    z=float(trigger.location.z),
                )
                actor.get_transform().transform(location)
            except (AttributeError, RuntimeError):
                location = actor.get_location()
            dx = float(location.x) - float(origin.x)
            dy = float(location.y) - float(origin.y)
            longitudinal = dx * float(forward.x) + dy * float(forward.y)
            lateral = abs(dx * right_x + dy * right_y)
            if -1.0 <= longitudinal <= 9.0 and lateral <= 4.0:
                candidates.append((max(0.0, longitudinal), actor_id))
        if self._active_stop_sign_id is None and candidates:
            self._active_stop_sign_id = min(candidates)[1]
            self._stop_sign_stopped_since = None
        if self._active_stop_sign_id is None:
            return False
        if self._speed(ego) < 0.08:
            if self._stop_sign_stopped_since is None:
                self._stop_sign_stopped_since = now
            elif now - self._stop_sign_stopped_since >= 1.0:
                self._cleared_stop_sign_ids.add(self._active_stop_sign_id)
                self._active_stop_sign_id = None
                self._stop_sign_stopped_since = None
                return False
        else:
            self._stop_sign_stopped_since = None
        return True

    def _map_legal_speed(self, ego, requested: float) -> float:
        try:
            limit_mps = float(ego.get_speed_limit()) / 3.6
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            if self._speed_limit_proof_mps is not None:
                raise OracleProgramError(
                    "live map speed limit unavailable for frozen speed proof"
                ) from exc
            return requested
        if not math.isfinite(limit_mps) or limit_mps <= 0.5:
            if self._speed_limit_proof_mps is not None:
                raise OracleProgramError(
                    "live map speed limit is non-finite/non-positive for "
                    "frozen speed proof"
                )
            return requested
        if self._speed_limit_proof_mps is not None:
            # Delayed-progress programs clear ``_speed_band`` after compiling
            # their stationary wait stage; no speed-band predicate is active
            # while their later progress controller calls this method.
            if self._speed_band is not None:
                band_max = self._speed_band[1]
                if (
                    band_max + 0.35
                    > limit_mps + self.SPEED_LIMIT_EPSILON_MPS
                ):
                    raise OracleProgramError(
                        "speed band max {:.3f}m/s lacks 0.35m/s margin below "
                        "live map limit {:.3f}m/s".format(
                            band_max, limit_mps
                        )
                    )
        return min(requested, max(0.2, limit_mps - 0.35))

    def _path_control(self, ego, target_speed: float):
        return self._path_control_to_endpoint(
            ego, target_speed, self.PATH_ENDPOINT_STOP_M
        )

    def _parking_prelead_path_control(self, ego):
        """Capture the v5 prelead without the generic 0.35m early stop."""

        return self._path_control_to_endpoint(
            ego, 2.0, self.PARKING_V5_PRELEAD_ENDPOINT_STOP_M
        )

    def _path_control_to_endpoint(
        self, ego, target_speed: float, endpoint_stop_m: float,
    ):
        transform = ego.get_transform()
        speed = self._speed(ego)
        along, cross_track = self._project(transform.location)
        if cross_track > 8.0:
            raise OracleProgramError(
                "ego diverged {:.2f}m from private oracle path".format(cross_track)
            )
        remaining = max(0.0, self._cumulative[-1] - along)
        base_lookahead = _clamp(3.2 + 0.45 * speed, 3.2, 7.0)
        target_speed, lookahead = self._curve_control_limits(
            along, target_speed, base_lookahead
        )
        if (
            self._speed_band is not None
            and self._curve_speed_cap_mps is not None
            and target_speed < self._speed_band[0] + 0.15
        ):
            raise OracleProgramError(
                "curvature-safe speed {:.3f}m/s conflicts with active band "
                "minimum {:.3f}m/s".format(
                    target_speed, self._speed_band[0]
                )
            )
        target = self._point_at(along + lookahead)
        target = self._adjacent_pass_target(ego, target)
        dx = target[0] - float(transform.location.x)
        dy = target[1] - float(transform.location.y)
        bearing = math.atan2(dy, dx)
        yaw = math.radians(float(transform.rotation.yaw))
        steering = _clamp(1.35 * _angle(bearing - yaw), -self.MAX_STEER, self.MAX_STEER)

        # Stop in the final target after it has been entered. For an until
        # speed-band contract, stay within band up to the gate and brake only
        # in its final metre.
        terminal_braking_allowed = (
            self._update_until_end(transform.location)
            and self._update_until_progress(along)
        )
        if (
            terminal_braking_allowed
            and remaining < 12.0
            and self._target.get("type") not in {"headway", "actor_order"}
        ):
            braking_speed = math.sqrt(max(0.0, 2.4 * remaining))
            if self._speed_band and remaining > 0.8:
                braking_speed = max(self._speed_band[0] + 0.2, braking_speed)
            target_speed = min(target_speed, braking_speed)
        if terminal_braking_allowed and remaining < endpoint_stop_m:
            target_speed = 0.0
        self._effective_control_tick = self._tick
        self._effective_target_speed_mps = target_speed
        self._effective_lookahead_m = lookahead
        control = self._longitudinal_control(speed, target_speed)
        control.steer = steering
        self.controller_source = "privileged_oracle:path"
        return control, remaining, target

    def _longitudinal_control(self, speed: float, target_speed: float):
        control = carla.VehicleControl()
        error = float(target_speed) - float(speed)
        if target_speed <= 0.05:
            self._speed_integral = 0.0
            control.throttle = 0.0
            control.brake = self.MAX_BRAKE if speed > 0.08 else 0.65
        elif error >= -0.05:
            self._speed_integral = _clamp(
                self._speed_integral + error * self.DT, -2.0, 3.0
            )
            control.throttle = _clamp(
                0.30 + 0.20 * error + 0.04 * self._speed_integral,
                0.0,
                self.MAX_THROTTLE,
            )
            control.brake = 0.0
        else:
            self._speed_integral = 0.0
            control.throttle = 0.0
            control.brake = _clamp(0.18 + 0.35 * (-error), 0.0, self.MAX_BRAKE)
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def _region_control(self, ego):
        """Follow the audited approach and stop at its exact legal pose.

        Steering continues to use the final oracle-path tangent; the oracle
        never performs an in-place turn inside the semantic region.  A large
        residual heading error at the pose is a malformed witness and fails
        closed instead of widening map legality.
        """

        assert self._terminal_pose is not None
        transform = ego.get_transform()
        speed = self._speed(ego)
        terminal = self._terminal_pose
        distance = math.hypot(
            float(transform.location.x) - terminal["x"],
            float(transform.location.y) - terminal["y"],
        )
        heading_error = _angle_degrees(
            float(transform.rotation.yaw), terminal["yaw_deg"]
        )
        if distance <= 0.45:
            if heading_error > 15.0:
                raise OracleProgramError(
                    "region terminal heading error {:.2f}deg".format(
                        heading_error
                    )
                )
            self._terminal_state = "dwell"
            control = self._longitudinal_control(speed, 0.0)
            self.controller_source = "privileged_oracle:region_dwell"
            return control, distance, (
                terminal["x"], terminal["y"], terminal["z"]
            )

        # The last few metres are deliberately slow so the final tangent can
        # align yaw without a stationary steering manoeuvre.
        target_speed = 1.2 if distance < 6.0 else self.CRUISE_SPEED_MPS
        control, remaining, target_point = self._path_control(ego, target_speed)
        self._terminal_state = "align" if distance < 6.0 else "approach"
        self.controller_source = "privileged_oracle:region_{}".format(
            self._terminal_state
        )
        return control, min(distance, remaining), target_point

    def _parking_control(self, ego):
        assert self._parking is not None
        transform = ego.get_transform()
        speed = self._speed(ego)
        center = self._parking["center"]
        desired_yaw = float(self._parking["heading_rad"])
        current_yaw = math.radians(float(transform.rotation.yaw))
        heading_error = _angle(desired_yaw - current_yaw)

        if self._parking_state == "approach":
            if self._parking["is_v5"]:
                control, remaining, target = self._parking_prelead_path_control(ego)
                rear_location = self._rear_axle_location(
                    transform,
                    self._parking["control_model"]["rear_axle_x_m"],
                )
                prelead_rear = self._parking["prelead_rear"]
                rear_distance = math.sqrt(sum(
                    (
                        float(getattr(rear_location, axis))
                        - prelead_rear[index]
                    ) ** 2
                    for index, axis in enumerate(("x", "y", "z"))
                ))
                approach_error = _angle_degrees(
                    float(transform.rotation.yaw),
                    self._parking["approach_yaw_deg"],
                )
                self._parking_prelead_remaining_m = remaining
                self._parking_prelead_rear_distance_m = rear_distance
                self._parking_prelead_yaw_error_deg = approach_error
                self._parking_prelead_speed_mps = speed
                if (
                    remaining < self.PARKING_V5_PRELEAD_ENDPOINT_STOP_M
                    and speed < 0.15
                ):
                    if (
                        rear_distance
                        > self.PARKING_V5_PRELEAD_REAR_DISTANCE_MAX_M + 1e-6
                    ):
                        raise OracleProgramError(
                            "parking v5 prelead rear axle misses frozen handoff "
                            "by {:.4f}m".format(rear_distance)
                        )
                    if (
                        approach_error
                        > self.PARKING_V5_PRELEAD_YAW_ERROR_MAX_DEG + 1e-6
                    ):
                        raise OracleProgramError(
                            "parking v5 prelead heading error {:.4f}deg".format(
                                approach_error
                            )
                        )
                    self._set_path([
                        (pose["x"], pose["y"], pose["z"])
                        for pose in self._parking["forward_rear_poses"]
                    ], minimum_spacing_m=1e-6)
                    self._parking_prelead_gate_passed = True
                    self._parking_prelead_handoff_tick = self._tick
                    self._parking_state = "staging_forward"
                self.controller_source = "privileged_oracle:parking_v5_prelead_capture"
                return control, remaining, target

            control, remaining, target = self._path_control(ego, 2.0)
            if remaining < 0.4 and speed < 0.15:
                approach_error = _angle_degrees(
                    float(transform.rotation.yaw),
                    self._parking["approach_yaw_deg"],
                )
                if approach_error > 10.0:
                    raise OracleProgramError(
                        "parking physical approach heading error {:.2f}deg"
                        .format(approach_error)
                    )
                self._set_path([
                    (pose["x"], pose["y"], pose["z"])
                    for pose in self._parking["forward_rear_poses"]
                ], minimum_spacing_m=1e-6)
                self._parking_state = "staging_forward"
            self.controller_source = "privileged_oracle:parking_approach"
            return control, remaining, target

        if self._parking_state == "staging_forward":
            control, remaining, target = self._forward_parking_path_control(ego)
            # Accept at the exact radius where the audited staging follower
            # commands a full stop.  Keeping both decisions on one constant
            # prevents the threshold deadlock found by the route-528 smoke.
            if (
                remaining < self.PARKING_STAGING_ENDPOINT_STOP_M
                and speed < 0.15
            ):
                rear_location = self._rear_axle_location(
                    transform,
                    self._parking["control_model"]["rear_axle_x_m"],
                )
                stage_rear = self._parking["stage_rear"]
                stage_rear_distance = math.sqrt(sum(
                    (float(getattr(rear_location, axis)) - stage_rear[index]) ** 2
                    for index, axis in enumerate(("x", "y", "z"))
                ))
                stage_error = _angle_degrees(
                    float(transform.rotation.yaw),
                    self._parking["stage_yaw_deg"],
                )
                self._parking_stage_rear_distance_m = stage_rear_distance
                self._parking_stage_yaw_error_deg = stage_error
                if (
                    stage_rear_distance
                    > self.PARKING_STAGING_REAR_DISTANCE_MAX_M + 1e-6
                ):
                    raise OracleProgramError(
                        "parking staging rear axle misses frozen handoff by "
                        "{:.3f}m".format(stage_rear_distance)
                    )
                # This is a fail-closed capture gate at the serialized
                # forward/reverse handoff, not a swept-footprint clearance
                # proof.  The live multi-seed collision witness remains
                # authoritative for dynamic execution.
                if (
                    stage_error
                    > self.PARKING_STAGING_YAW_ERROR_MAX_DEG + 1e-6
                ):
                    raise OracleProgramError(
                        "parking staging heading error {:.2f}deg".format(
                            stage_error
                        )
                    )
                self._parking_state = "staging_stop"
            self.controller_source = "privileged_oracle:parking_staging_forward"
            return control, remaining, target

        if self._parking_state == "staging_stop":
            control = self._longitudinal_control(speed, 0.0)
            if speed < 0.08:
                reverse_poses = self._parking["reverse_poses"]
                self._set_path([
                    (pose["x"], pose["y"], pose["z"])
                    for pose in self._parking["reverse_rear_poses"]
                ], minimum_spacing_m=1e-6)
                self._parking_state = "reverse"
            self.controller_source = "privileged_oracle:parking_staging_stop"
            return control, self._cumulative[-1], self._parking["stage"]

        dx = center[0] - float(transform.location.x)
        dy = center[1] - float(transform.location.y)
        distance = math.hypot(dx, dy)
        if self._parking_state == "reverse":
            control, remaining, target = self._reverse_parking_path_control(ego)
            if (
                remaining < 0.16
                and distance < 0.20
                and abs(heading_error) <= math.radians(10.0)
                and speed < 0.12
            ):
                self._parking_state = "dwell"
            elif remaining < 0.16 and (
                distance >= 0.20
                or abs(heading_error) > math.radians(10.0)
            ):
                raise OracleProgramError(
                    "parking reverse terminal outside actor-center/heading "
                    "contract ({:.3f}m,{:.2f}deg)".format(
                        distance, math.degrees(abs(heading_error))
                    )
                )
            self.controller_source = "privileged_oracle:parking_reverse_path"
            return control, min(distance, remaining), target

        control = self._longitudinal_control(speed, 0.0)
        control.reverse = False
        self.controller_source = "privileged_oracle:parking_dwell"
        return control, distance, (center[0], center[1], transform.location.z)

    def _forward_parking_path_control(self, ego):
        """Track the explicit audited rear-axle staging curve.

        The generic route follower has a 3.2 m minimum lookahead.  A parking
        staging curve is only a few metres long, so that lookahead follows its
        endpoint chord and cannot reproduce the frozen terminal tangent.  This
        controller projects the live rear axle onto the serialized rear path
        and maps signed path curvature through the immutable 33-point response.
        """

        assert self._parking is not None
        transform = ego.get_transform()
        speed = self._speed(ego)
        rear_location = self._rear_axle_location(
            transform, self._parking["control_model"]["rear_axle_x_m"]
        )
        along, cross_track = self._project(rear_location)
        self._parking_stage_cross_track_m = cross_track
        if cross_track > 1.25:
            raise OracleProgramError(
                "parking staging forward diverged {:.2f}m from audited "
                "manoeuvre".format(cross_track)
            )
        remaining = max(0.0, self._cumulative[-1] - along)
        lookahead = 0.50
        target = self._point_at(along + lookahead)
        dx = target[0] - float(rear_location.x)
        dy = target[1] - float(rear_location.y)
        current_yaw = math.radians(float(transform.rotation.yaw))
        local_y = -math.sin(current_yaw) * dx + math.cos(current_yaw) * dy
        distance_squared = max(0.35 ** 2, dx * dx + dy * dy)
        proof = self._parking["proof"]
        max_steer = min(
            self.MAX_STEER,
            float(proof["max_normalized_control_steer"]),
        )
        if max_steer <= 0.0 or not math.isfinite(max_steer):
            raise OracleProgramError(
                "parking staging control proof has invalid steering bound"
            )
        pure_pursuit_steer = self._parking_pursuit_steer(
            2.0 * local_y / distance_squared
        )
        forward_poses = self._parking["forward_rear_poses"]
        desired_index, desired_pose = min(
            enumerate(forward_poses),
            key=lambda item: (
                (item[1]["x"] - target[0]) ** 2
                + (item[1]["y"] - target[1]) ** 2
            ),
        )
        curvature_feedforward = self._parking_steer_for_curvature(
            self._parking_path_curvature(
                forward_poses, desired_index, False
            )
        )
        heading_error = _angle(
            math.radians(float(desired_pose["yaw_deg"])) - current_yaw
        )
        # Pure pursuit becomes ill-conditioned as its target collapses onto
        # the endpoint: a few centimetres of live lateral offset can reverse
        # the steering sign and cancel the audited terminal tangent.  Taper
        # only that positional correction over the final metre while keeping
        # the serialized curvature and heading terms active.
        pursuit_blend = 0.50 * _clamp(remaining, 0.0, 1.0)
        steering = _clamp(
            curvature_feedforward
            + pursuit_blend * (
                pure_pursuit_steer - curvature_feedforward
            )
            + self.PARKING_HEADING_COMMAND_GAIN_PER_RAD * heading_error,
            -max_steer,
            max_steer,
        )
        target_speed = 0.85
        if remaining < 2.0:
            target_speed = min(
                target_speed, math.sqrt(max(0.0, 0.8 * remaining))
            )
        if remaining < self.PARKING_STAGING_ENDPOINT_STOP_M:
            target_speed = 0.0
        control = self._longitudinal_control(speed, target_speed)
        control.reverse = False
        control.steer = steering
        self._effective_control_tick = self._tick
        self._effective_target_speed_mps = target_speed
        self._effective_lookahead_m = lookahead
        return control, remaining, target

    def _reverse_parking_path_control(self, ego):
        """Track only the generator-audited rear-axle reverse sequence.

        Steering is applied against the rearward motion heading and every
        feedforward command is map-inverted from the proof-bound signed rear
        curvature.  There is no actor-origin projection or online fallback.
        """

        assert self._parking is not None
        transform = ego.get_transform()
        speed = self._speed(ego)
        rear_location = self._rear_axle_location(
            transform, self._parking["control_model"]["rear_axle_x_m"]
        )
        along, cross_track = self._project(rear_location)
        if cross_track > 1.25:
            raise OracleProgramError(
                "parking reverse diverged {:.2f}m from audited manoeuvre"
                .format(cross_track)
            )
        remaining = max(0.0, self._cumulative[-1] - along)
        lookahead = _clamp(0.65 + 0.2 * speed, 0.65, 1.1)
        target = self._point_at(along + lookahead)
        dx = target[0] - float(rear_location.x)
        dy = target[1] - float(rear_location.y)
        current_yaw = math.radians(float(transform.rotation.yaw))
        rearward_y = (
            -math.sin(current_yaw + math.pi) * dx
            + math.cos(current_yaw + math.pi) * dy
        )
        distance_squared = max(0.35 ** 2, dx * dx + dy * dy)
        proof = self._parking["proof"]
        max_steer = min(
            self.MAX_STEER,
            float(proof["max_normalized_control_steer"]),
        )
        if max_steer <= 0.0 or not math.isfinite(max_steer):
            raise OracleProgramError(
                "parking reverse control proof has invalid steering bound"
            )
        pure_pursuit_steer = self._parking_pursuit_steer(
            -2.0 * rearward_y / distance_squared
        )
        reverse_poses = self._parking["reverse_rear_poses"]
        desired_index, desired_pose = min(
            enumerate(reverse_poses),
            key=lambda item: (
                (item[1]["x"] - target[0]) ** 2
                + (item[1]["y"] - target[1]) ** 2
            ),
        )
        feedforward = self._parking_steer_for_curvature(
            self._parking_path_curvature(reverse_poses, desired_index, True)
        )
        heading_error = _angle(
            math.radians(float(desired_pose["yaw_deg"])) - current_yaw
        )
        steering = _clamp(
            feedforward
            + 0.50 * (pure_pursuit_steer - feedforward)
            - self.PARKING_HEADING_COMMAND_GAIN_PER_RAD * heading_error,
            -max_steer,
            max_steer,
        )
        target_speed = 0.85
        if remaining < 2.0:
            target_speed = min(
                target_speed, math.sqrt(max(0.0, 0.8 * remaining))
            )
        if remaining < 0.10:
            target_speed = 0.0
        control = self._longitudinal_control(speed, target_speed)
        control.reverse = True
        control.gear = -1
        control.steer = steering
        self._effective_control_tick = self._tick
        self._effective_target_speed_mps = target_speed
        self._effective_lookahead_m = lookahead
        return control, remaining, target

    def _write_plan(
        self,
        timestamp: float,
        target_speed: float,
        remaining: float,
        target: Tuple[float, float, float],
    ) -> None:
        if self._plans is None or self._tick % 20 != 0:
            return
        effective_target_speed = float(target_speed)
        effective_lookahead = None
        curve_speed_cap = None
        curve_distance = None
        if self._effective_control_tick == self._tick:
            if self._effective_target_speed_mps is not None:
                effective_target_speed = self._effective_target_speed_mps
            effective_lookahead = self._effective_lookahead_m
            curve_speed_cap = self._curve_speed_cap_mps
            curve_distance = self._curve_next_distance_m
        payload = {
            "sim_time": round(float(timestamp), 4),
            "source": "private_oracle_path",
            "controller_source": self.controller_source,
            "target_type": self._target.get("type") if self._target else None,
            "conditional_branch": self._conditional_branch,
            "parking_state": self._parking_state,
            "parking_schema_version": (
                self._parking.get("schema_version")
                if self._parking is not None else None
            ),
            "parking_approach_pose_source": (
                self._parking.get("approach_pose_source")
                if self._parking is not None else None
            ),
            "parking_prelead_remaining_m": (
                round(float(self._parking_prelead_remaining_m), 4)
                if self._parking_prelead_remaining_m is not None else None
            ),
            "parking_prelead_rear_distance_m": (
                round(float(self._parking_prelead_rear_distance_m), 4)
                if self._parking_prelead_rear_distance_m is not None else None
            ),
            "parking_prelead_yaw_error_deg": (
                round(float(self._parking_prelead_yaw_error_deg), 4)
                if self._parking_prelead_yaw_error_deg is not None else None
            ),
            "parking_prelead_speed_mps": (
                round(float(self._parking_prelead_speed_mps), 4)
                if self._parking_prelead_speed_mps is not None else None
            ),
            "parking_prelead_gate_passed": self._parking_prelead_gate_passed,
            "parking_prelead_handoff_tick": self._parking_prelead_handoff_tick,
            "parking_stage_rear_distance_m": (
                round(float(self._parking_stage_rear_distance_m), 4)
                if self._parking_stage_rear_distance_m is not None else None
            ),
            "parking_stage_cross_track_m": (
                round(float(self._parking_stage_cross_track_m), 4)
                if self._parking_stage_cross_track_m is not None else None
            ),
            "parking_stage_yaw_error_deg": (
                round(float(self._parking_stage_yaw_error_deg), 4)
                if self._parking_stage_yaw_error_deg is not None else None
            ),
            "terminal_state": self._terminal_state,
            "until_gate_entered": self._until_end_entered,
            "until_progress_min": self._until_progress_min,
            "until_progress_entered": self._until_progress_entered,
            "path_segment": self._segment,
            "target_speed_mps": round(float(effective_target_speed), 4),
            "requested_target_speed_mps": round(float(target_speed), 4),
            "lookahead_m": (
                round(float(effective_lookahead), 4)
                if effective_lookahead is not None else None
            ),
            "curve_speed_cap_mps": (
                round(float(curve_speed_cap), 4)
                if curve_speed_cap is not None else None
            ),
            "next_curve_distance_m": (
                round(float(curve_distance), 4)
                if curve_distance is not None else None
            ),
            "remaining_m": round(float(remaining), 4),
            "lookahead_target": {
                "x": round(float(target[0]), 4),
                "y": round(float(target[1]), 4),
                "z": round(float(target[2]), 4),
            },
        }
        self._plans.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._plans.flush()

    def run_step(self, input_data, timestamp):
        del input_data
        if self._destroyed or self._route is None or self._target is None:
            raise OracleProgramError("oracle was not set up or was already destroyed")
        ego = self._ego_actor()
        now = float(timestamp)
        if self._start_time is None:
            self._start_time = now
        self._tick += 1

        if now - self._start_time < self._wait_seconds:
            if self._contract_speed_band is not None:
                self._validate_live_band_limit(
                    ego, self._contract_speed_band, "map"
                )
            control = self._longitudinal_control(self._speed(ego), 0.0)
            self.controller_source = "privileged_oracle:delayed_start"
            target = self._points[0]
            self._write_plan(now, 0.0, self._cumulative[-1], target)
            return control

        if self._target.get("type") == "hold":
            if self._contract_speed_band is not None:
                self._validate_live_band_limit(
                    ego, self._contract_speed_band, "map"
                )
            control = self._longitudinal_control(self._speed(ego), 0.0)
            self.controller_source = "privileged_oracle:contract_hold"
            target = self._points[0]
            self._write_plan(now, 0.0, self._cumulative[-1], target)
            return control

        if self._parking is not None:
            control, remaining, target = self._parking_control(ego)
            self._write_plan(now, 0.0 if self._parking_state == "dwell" else 1.2, remaining, target)
            return control


        if self._terminal_pose is not None:
            control, remaining, target = self._region_control(ego)
            self._write_plan(
                now,
                0.0 if self._terminal_state == "dwell" else 1.2,
                remaining,
                target,
            )
            return control

        target_speed = self.CRUISE_SPEED_MPS
        if self._speed_band is not None:
            low, high = self._speed_band
            # Stay comfortably inside the band while making enough normalized
            # progress for long road1 control routes before their frozen
            # deadline. Never ride the upper boundary where numerical noise
            # could flip the predicate.
            target_speed = high - max(0.25, 0.15 * (high - low))
        target_speed = self._relation_speed(ego, target_speed)
        target_speed = self._map_legal_speed(ego, target_speed)
        safety_source = None
        if self._red_light(ego):
            target_speed = 0.0
            safety_source = "red_light_stop"
        elif self._stop_sign_hazard(ego, now):
            target_speed = 0.0
            safety_source = "stop_sign_stop"
        else:
            obstacle = self._obstacle_hazard(ego)
            if obstacle is not None:
                target_speed = 0.0
                safety_source = "{}_stop".format(obstacle)
        control, remaining, target = self._path_control(ego, target_speed)
        if safety_source is not None:
            self.controller_source = "privileged_oracle:{}".format(safety_source)
        self._write_plan(now, target_speed, remaining, target)
        return control

    def destroy(self):
        if self._destroyed:
            return
        self._destroyed = True
        if self._plans is not None:
            self._plans.flush()
            self._plans.close()
            self._plans = None
