"""Evaluator-only actor programs and Standardized-track termination.

The runtime owns private sidecar state.  It is deliberately constructed by the
leaderboard (never by the agent), spawns only declared semantic actors, observes
real CARLA state, and fails closed when a declared setup cannot be reproduced.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

from team_code.language2drive_trace import _project_progress


class Language2DriveRuntimeError(RuntimeError):
    """A private actor program or monitor could not be reproduced safely."""


SUPPORTED_ACTOR_PROGRAMS = frozenset({
    "constant_speed", "slow_constant_speed",
})


def validate_actor_setup_program(
    route: Mapping[str, Any], private_context: Mapping[str, Any]
) -> None:
    """Validate the complete evaluator-owned actor program without CARLA.

    The same validation is called by the release compiler and immediately
    before runtime spawning.  It prevents a partial setup from becoming
    observable (for example, actor 1 spawning before actor 2 is discovered to
    have an invalid program) and keeps release qualification isomorphic to the
    live evaluator.
    """

    setup = route.get("actor_setup", {})
    if not isinstance(setup, Mapping):
        raise Language2DriveRuntimeError("actor_setup must be an object")
    unknown_setup = sorted(
        set(setup) - {"actors", "condition_cell", "slot_occupancy"}
    )
    if unknown_setup:
        raise Language2DriveRuntimeError(
            "actor_setup has unsupported fields {}".format(
                ",".join(map(str, unknown_setup))
            )
        )
    actors = setup.get("actors", [])
    if not isinstance(actors, list):
        raise Language2DriveRuntimeError("actor_setup.actors must be a list")
    identifiers = set()
    occupied_actor_slots = set()
    allowed_actor_fields = {
        "id", "blueprint", "color", "program", "relative_spawn_m", "slot",
    }
    for index, spec in enumerate(actors):
        if not isinstance(spec, Mapping):
            raise Language2DriveRuntimeError(
                "actor_setup.actors[{}] must be an object".format(index)
            )
        unknown_actor = sorted(set(spec) - allowed_actor_fields)
        if unknown_actor:
            raise Language2DriveRuntimeError(
                "actor {} has unsupported fields {}".format(
                    index, ",".join(map(str, unknown_actor))
                )
            )
        identifier = spec.get("id")
        blueprint = spec.get("blueprint")
        if not isinstance(identifier, str) or not identifier:
            raise Language2DriveRuntimeError(
                "actor setup requires a non-empty id"
            )
        if identifier in identifiers:
            raise Language2DriveRuntimeError(
                "duplicate actor id {}".format(identifier)
            )
        identifiers.add(identifier)
        if not isinstance(blueprint, str) or not blueprint:
            raise Language2DriveRuntimeError(
                "actor {} requires a blueprint".format(identifier)
            )
        if "color" in spec and (
            not isinstance(spec.get("color"), str) or not spec.get("color")
        ):
            raise Language2DriveRuntimeError(
                "actor {} color must be a non-empty string".format(identifier)
            )

        slot = spec.get("slot")
        if slot is not None:
            if not isinstance(slot, str) or not slot:
                raise Language2DriveRuntimeError(
                    "actor {} slot must be a non-empty string".format(identifier)
                )
            if spec.get("program") is not None or spec.get(
                "relative_spawn_m"
            ) is not None:
                raise Language2DriveRuntimeError(
                    "parking actor {} cannot have a moving program".format(
                        identifier
                    )
                )
            if slot in occupied_actor_slots:
                raise Language2DriveRuntimeError(
                    "multiple evaluator actors occupy slot {}".format(slot)
                )
            occupied_actor_slots.add(slot)
            if slot not in private_context.get("actor_spawn_targets", {}):
                raise Language2DriveRuntimeError(
                    "parking actor {} has no unique geometry for slot {}".format(
                        identifier, slot
                    )
                )
            continue

        program = spec.get("program")
        if program not in SUPPORTED_ACTOR_PROGRAMS:
            raise Language2DriveRuntimeError(
                "actor {} has unsupported program {!r}".format(
                    identifier, program
                )
            )
        try:
            relative_spawn_m = float(spec["relative_spawn_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Language2DriveRuntimeError(
                "moving actor {} requires relative_spawn_m".format(identifier)
            ) from exc
        if not math.isfinite(relative_spawn_m) or relative_spawn_m < 0:
            raise Language2DriveRuntimeError(
                "moving actor {} relative_spawn_m must be finite and non-negative".format(
                    identifier
                )
            )
        path = route.get("oracle_path", [])
        try:
            path_xyz = [
                (float(point["x"]), float(point["y"]), float(point.get("z", 0.0)))
                for point in path
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise Language2DriveRuntimeError(
                "moving actor {} requires a valid oracle path".format(identifier)
            ) from exc
        path_length = sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(path_xyz, path_xyz[1:])
        )
        spawn_point = _path_point(path, relative_spawn_m)
        if (
            len(path_xyz) < 2
            or spawn_point is None
            or relative_spawn_m > path_length + 1e-6
        ):
            raise Language2DriveRuntimeError(
                "moving actor {} spawn distance exceeds oracle path".format(
                    identifier
                )
            )
        initial_clearance = math.hypot(
            spawn_point[0] - path_xyz[0][0],
            spawn_point[1] - path_xyz[0][1],
        )
        if initial_clearance < 5.0:
            raise Language2DriveRuntimeError(
                "moving actor {} spawn clearance {:.3f}m is unsafe".format(
                    identifier, initial_clearance
                )
            )

    occupancy = setup.get("slot_occupancy", {})
    if not isinstance(occupancy, Mapping):
        raise Language2DriveRuntimeError("slot_occupancy must be an object")
    for slot, declared in occupancy.items():
        if not isinstance(slot, str) or not slot or not isinstance(declared, bool):
            raise Language2DriveRuntimeError(
                "slot_occupancy requires non-empty string keys and booleans"
            )
        actually_occupied = slot in occupied_actor_slots
        if actually_occupied != declared:
            raise Language2DriveRuntimeError(
                "slot_occupancy for {} disagrees with evaluator actors".format(slot)
            )
    condition_cell = setup.get("condition_cell")
    if condition_cell is not None and condition_cell not in {"occupied", "empty"}:
        raise Language2DriveRuntimeError(
            "condition_cell must be occupied or empty"
        )

    required_actor_ids = set(
        map(str, private_context.get("requirements", {}).get("actor_ids", []))
    )
    unresolved_actor_ids = sorted(required_actor_ids - identifiers)
    if unresolved_actor_ids:
        raise Language2DriveRuntimeError(
            "contract actors are not declared by actor_setup: {}".format(
                ",".join(unresolved_actor_ids)
            )
        )
    required_occupancy = set(
        map(
            str,
            private_context.get("requirements", {}).get(
                "occupancy_slots", []
            ),
        )
    )
    missing_occupancy = sorted(required_occupancy - set(occupancy))
    if missing_occupancy:
        raise Language2DriveRuntimeError(
            "contract occupancy is not frozen for slot(s): {}".format(
                ",".join(missing_occupancy)
            )
        )


def _path_point(path, distance_m: float) -> Optional[Tuple[float, float, float]]:
    if not isinstance(path, list) or not path:
        return None
    try:
        points = [
            (float(point["x"]), float(point["y"]), float(point.get("z", 0.0)))
            for point in path
        ]
    except (KeyError, TypeError, ValueError):
        return None
    remaining = max(0.0, float(distance_m))
    for first, second in zip(points, points[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        if remaining <= length and length > 1e-9:
            ratio = remaining / length
            return tuple(
                first[index] + ratio * (second[index] - first[index])
                for index in range(3)
            )
        remaining -= length
    return points[-1]


def _polygon_center(polygon) -> Optional[Tuple[float, float]]:
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    try:
        return (
            sum(float(point[0]) for point in vertices) / len(vertices),
            sum(float(point[1]) for point in vertices) / len(vertices),
        )
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


class Language2DriveEvaluatorRuntime:
    """Run deterministic private actors and own the v1 rollout horizon."""

    # A CARLA ego that has fallen this far below every point in its frozen
    # route cannot return to the drivable world.  Letting it free-fall until
    # the semantic deadline can crash UE4 before the evaluator writes its
    # otherwise deterministic failed result.  Fifty metres is deliberately
    # much larger than normal road/bridge elevation changes in the release
    # suite and is evaluated only by the private runtime.
    EGO_BELOW_ROUTE_MARGIN_M = 50.0

    PROGRAM_SPEED_MPS = {
        "constant_speed": 5.0,
        "slow_constant_speed": 2.0,
    }

    def __init__(
        self,
        private_context: Mapping[str, Any],
        language2drive_root: str,
        *,
        provider,
        carla_module,
        traffic_manager=None,
        traffic_manager_port: Optional[int] = None,
        stability_seconds: float = 2.0,
        progress_minimum: float = 0.05,
    ) -> None:
        self.private_context = dict(private_context)
        self.route = self.private_context.get("route", {})
        self.language2drive_root = Path(language2drive_root).expanduser().resolve()
        self.provider = provider
        self.carla = carla_module
        self.traffic_manager = traffic_manager
        self.traffic_manager_port = traffic_manager_port
        self.stability_seconds = float(stability_seconds)
        self.progress_minimum = float(progress_minimum)
        if self.stability_seconds < 0 or self.progress_minimum < 0:
            raise ValueError("stability_seconds and progress_minimum must be non-negative")

        src = self.language2drive_root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from language2drive.contracts import (
            ContractEngine,
            semantic_instruction_verdict,
        )

        self._contract_engine = ContractEngine
        self._semantic_instruction_verdict = semantic_instruction_verdict
        self._actors: Dict[str, Any] = {}
        self._actor_specs: Dict[str, Dict[str, Any]] = {}
        self._pending_events = []
        self._final_events: List[Dict[str, Any]] = []
        self._spawned_actor_ids: List[str] = []
        self._destroyed_actor_ids: List[str] = []
        self._trace = []
        # ContractEngine sees the same immutable row objects as this trace
        # grows, allowing exact atomic-predicate reuse across online prefixes.
        self._contract_predicate_cache: Dict[
            Tuple[Any, ...], Dict[int, bool]
        ] = {}
        self._start_time: Optional[float] = None
        self._candidate_time: Optional[float] = None
        self._setup_complete = False
        self._closed = False
        self._cleanup_complete = False
        self.failure_reason: Optional[str] = None
        self.termination_reason: Optional[str] = None
        self._latest_monitor: Dict[str, Any] = {}
        route_z = []
        for point in self.route.get("oracle_path", []):
            if not isinstance(point, Mapping):
                continue
            try:
                value = float(point.get("z", 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                route_z.append(value)
        self._route_min_z = min(route_z) if route_z else None

    @property
    def active(self) -> bool:
        return self._setup_complete and not self._closed

    @property
    def should_terminate(self) -> bool:
        return self.termination_reason is not None or self.failure_reason is not None

    def _location(self, x: float, y: float, z: float):
        return self.carla.Location(x=float(x), y=float(y), z=float(z))

    def _lane_transform(self, point: Tuple[float, float, float]):
        location = self._location(*point)
        world_map = self.provider.get_map()
        kwargs = {"project_to_road": True}
        lane_type = getattr(getattr(self.carla, "LaneType", None), "Driving", None)
        if lane_type is not None:
            kwargs["lane_type"] = lane_type
        try:
            waypoint = world_map.get_waypoint(location, **kwargs)
        except TypeError:
            waypoint = world_map.get_waypoint(location)
        if waypoint is None:
            raise Language2DriveRuntimeError(
                "actor spawn point does not project to a driving lane"
            )
        source = waypoint.transform
        return self.carla.Transform(
            self._location(
                source.location.x, source.location.y, source.location.z
            ),
            self.carla.Rotation(
                pitch=float(source.rotation.pitch),
                yaw=float(source.rotation.yaw),
                roll=float(source.rotation.roll),
            ),
        )

    def _relative_transform(self, spec: Mapping[str, Any]):
        distance = spec.get("relative_spawn_m")
        if distance is None:
            raise Language2DriveRuntimeError(
                "moving actor {} requires relative_spawn_m".format(spec.get("id"))
            )
        point = _path_point(self.route.get("oracle_path"), float(distance))
        if point is None:
            raise Language2DriveRuntimeError(
                "moving actor {} requires a valid oracle path".format(spec.get("id"))
            )
        return self._lane_transform(point)

    def _slot_transform(self, spec: Mapping[str, Any]):
        slot = str(spec.get("slot", ""))
        target = self.private_context.get("actor_spawn_targets", {}).get(slot)
        if not isinstance(target, dict):
            raise Language2DriveRuntimeError(
                "parking actor {} has no unique geometry for slot {}".format(
                    spec.get("id"), slot
                )
            )
        center = _polygon_center(target.get("polygon"))
        if center is None:
            raise Language2DriveRuntimeError(
                "parking slot {} has invalid polygon geometry".format(slot)
            )
        # Preserve the exact annotated x/y while borrowing only road elevation.
        lane_transform = self._lane_transform((center[0], center[1], 0.0))
        heading = target.get("heading_deg")
        if heading is None:
            heading = lane_transform.rotation.yaw
        return self.carla.Transform(
            self._location(center[0], center[1], lane_transform.location.z),
            self.carla.Rotation(yaw=float(heading)),
        )

    def _spawn_one(self, spec: Mapping[str, Any], route_scenario) -> None:
        identifier = spec.get("id")
        blueprint = spec.get("blueprint")
        if not isinstance(identifier, str) or not identifier:
            raise Language2DriveRuntimeError("actor setup requires a non-empty id")
        if not isinstance(blueprint, str) or not blueprint:
            raise Language2DriveRuntimeError(
                "actor {} requires a blueprint".format(identifier)
            )
        if identifier in self._actors:
            raise Language2DriveRuntimeError(
                "duplicate actor id {}".format(identifier)
            )
        program = spec.get("program")
        if spec.get("slot") is not None:
            transform = self._slot_transform(spec)
        elif program in self.PROGRAM_SPEED_MPS:
            transform = self._relative_transform(spec)
        else:
            raise Language2DriveRuntimeError(
                "actor {} has unsupported program {!r}".format(identifier, program)
            )
        actor = self.provider.request_new_actor(
            blueprint,
            transform,
            rolename="l2d:{}".format(identifier),
            autopilot=False,
            color=spec.get("color"),
            tick=False,
        )
        if actor is None:
            raise Language2DriveRuntimeError(
                "CARLA failed to spawn actor {}".format(identifier)
            )
        self._actors[identifier] = actor
        self._actor_specs[identifier] = dict(spec)
        self._spawned_actor_ids.append(identifier)
        self._pending_events.append(
            {
                "type": "actor_spawn",
                "actor_id": identifier,
                "instance_id": int(actor.id),
                "actor_type": getattr(actor, "type_id", blueprint),
                "source": "evaluator_actor_runtime",
                "occurred_before_first_trace": True,
            }
        )

        if spec.get("slot") is not None:
            if hasattr(actor, "set_simulate_physics"):
                actor.set_simulate_physics(False)
            center = transform.location
            occupied = getattr(route_scenario, "occupied_parking_locations", None)
            if isinstance(occupied, list):
                occupied.append(center)
            return

        speed = self.PROGRAM_SPEED_MPS[program]
        if self.traffic_manager is None or self.traffic_manager_port is None:
            raise Language2DriveRuntimeError(
                "actor {} program {} requires TrafficManager".format(
                    identifier, program
                )
            )
        actor.set_autopilot(True, int(self.traffic_manager_port))
        if hasattr(self.traffic_manager, "auto_lane_change"):
            self.traffic_manager.auto_lane_change(actor, False)
        if hasattr(self.traffic_manager, "set_desired_speed"):
            self.traffic_manager.set_desired_speed(actor, speed * 3.6)
        else:
            raise Language2DriveRuntimeError(
                "TrafficManager lacks set_desired_speed for actor {}".format(identifier)
            )

    def setup(self, ego_vehicle, route_scenario) -> None:
        del ego_vehicle  # The hidden oracle path, not a public plan, selects the lane.
        if self._setup_complete:
            raise Language2DriveRuntimeError("runtime setup called twice")
        try:
            validate_actor_setup_program(self.route, self.private_context)
            actors = self.route.get("actor_setup", {}).get("actors", [])
            for spec in actors:
                self._spawn_one(spec, route_scenario)
            self._setup_complete = True
        except Exception as exc:
            self.failure_reason = "actor_setup_failed:{}:{}".format(
                type(exc).__name__, exc
            )
            self.cleanup()
            raise

    def tick_world(self) -> None:
        if not self.active:
            return
        for identifier, actor in self._actors.items():
            try:
                alive = bool(getattr(actor, "is_alive", True))
            except RuntimeError:
                alive = False
            if not alive:
                self.failure_reason = "actor_despawned_unexpectedly:{}".format(identifier)
                return

    def record_capture_failure(self, operation: str, exc: Exception) -> None:
        """Fail closed if the authoritative trace/contract path breaks."""

        if self.failure_reason is None:
            self.failure_reason = "capture_failed:{}:{}:{}".format(
                operation, type(exc).__name__, exc
            )

    def record_evaluator_failure(
        self, operation: str, exc: Optional[BaseException] = None,
    ) -> None:
        """Persist an evaluator/agent abort as an explicit private verdict.

        The contract monitor normally owns every v1 route termination.  If
        control leaves the scenario loop through an exception or an external
        manager stop, a partial trace must never look capture-complete merely
        because its JSONL handles closed cleanly.
        """

        if self.failure_reason is not None or self.termination_reason is not None:
            return
        label = str(operation).strip().lower()
        if not label or any(
            not (character.isalnum() or character == "_")
            for character in label
        ):
            label = "unknown"
        if exc is None:
            self.failure_reason = "evaluator_failed:{}".format(label)
            return
        detail = " ".join(str(exc).split())[:240]
        self.failure_reason = "evaluator_failed:{}:{}:{}".format(
            label, type(exc).__name__, detail
        )

    def drain_final_events(self) -> List[Dict[str, Any]]:
        """Return evaluator-owned lifecycle events exactly once."""

        events = list(self._final_events)
        self._final_events.clear()
        return events

    def _task_progress(self, row: Mapping[str, Any]) -> Optional[float]:
        target = self.route.get("target", {})
        if isinstance(target, dict) and target.get("type") == "hold":
            seconds = float(target.get("seconds", 0.0))
            if seconds <= 0 or self._start_time is None:
                return None
            return max(
                0.0,
                min(1.0, (float(row["sim_time"]) - self._start_time) / seconds),
            )
        ego = row.get("ego", {})
        location = ego.get("location", {}) if isinstance(ego, dict) else {}
        try:
            point = (float(location["x"]), float(location["y"]))
        except (KeyError, TypeError, ValueError):
            return None
        return _project_progress(point, self.route.get("oracle_path", []))

    def _ego_out_of_world_details(
        self, row: Mapping[str, Any],
    ) -> Optional[Dict[str, float]]:
        """Describe an irreversible below-map state, if one is observed."""

        ego = row.get("ego", {})
        location = ego.get("location", {}) if isinstance(ego, Mapping) else {}
        try:
            ego_z = float(location["z"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(ego_z) or self._route_min_z is None:
            return None
        route_min_z = self._route_min_z
        margin = float(self.EGO_BELOW_ROUTE_MARGIN_M)
        if ego_z >= route_min_z - margin:
            return None
        return {
            "ego_z": ego_z,
            "route_min_z": route_min_z,
            "below_route_m": route_min_z - ego_z,
            "threshold_m": margin,
        }

    def _always_horizon_reached(self, row: Mapping[str, Any]) -> bool:
        target = self.route.get("target", {})
        now = float(row["sim_time"])
        if isinstance(target, dict) and target.get("type") == "hold":
            seconds = float(target.get("seconds", 0.0))
            return self._start_time is not None and now - self._start_time >= seconds
        progress = row.get("progress")
        return isinstance(progress, (int, float)) and float(progress) >= 0.95

    def observe_frame(self, row: Dict[str, Any]) -> None:
        """Privately enrich a logger row and update the online stop monitor."""

        if self._closed:
            return
        now = float(row["sim_time"])
        if self._start_time is None:
            self._start_time = now
        progress = self._task_progress(row)
        if progress is not None:
            row["progress"] = progress
        events = row.setdefault("events", [])
        while self._pending_events:
            event = self._pending_events.pop(0)
            event["sim_time"] = now
            event["observed_at_frame"] = row.get("frame")
            events.append(event)

        out_of_world = self._ego_out_of_world_details(row)
        if out_of_world is not None:
            events.append(
                {
                    "type": "off_road",
                    "reason": "ego_out_of_world",
                    "source": "evaluator_terminal_guard",
                    "frame": row.get("frame"),
                    "sim_time": now,
                    "details": dict(out_of_world),
                }
            )

        self._trace.append(copy.deepcopy(row))
        try:
            engine = self._contract_engine(
                self._trace,
                predicate_cache=self._contract_predicate_cache,
            )
            own = engine.evaluate(self.route["contract"])
            opposite = engine.evaluate(self.route["opposite_contract"])
            safe = engine.evaluate({"safe_legal": {}})
            progress_verdict = engine.evaluate(
                {"eventually": {"progress": {"min": self.progress_minimum}}}
            )
            root_operator = next(iter(self.route["contract"]), "")
            horizon = (
                self._always_horizon_reached(row)
                if root_operator == "always"
                else True
            )
            candidate = (
                own.passed
                and not opposite.passed
                and safe.passed
                and progress_verdict.passed
                and horizon
            )
            semantic = self._semantic_instruction_verdict(
                self._trace,
                target=self.route.get("target", {}),
                oracle_path=self.route.get("oracle_path", []),
                own=own,
                layer=self.route.get("layer"),
            )
            semantic_candidate = semantic.passed
            self._latest_monitor = {
                "own_contract": own.passed,
                "semantic_success": semantic_candidate,
                "semantic_details": dict(semantic.details),
                "opposite_contract": opposite.passed,
                "safe_legal": safe.passed,
                "progress": progress_verdict.passed,
                "always_horizon_reached": horizon,
                "sim_time": now,
                "ego_out_of_world": dict(out_of_world) if out_of_world else None,
            }
            if out_of_world is not None:
                self.termination_reason = "ego_out_of_world"
                return
            if candidate:
                if self._candidate_time is None:
                    self._candidate_time = now
                if (
                    root_operator == "always"
                    or now - self._candidate_time >= self.stability_seconds - 1e-9
                ):
                    self.termination_reason = "contract_complete"
                    return
            else:
                self._candidate_time = None
            # Safety remains a hard veto for the benchmark's strict success,
            # but it must not truncate the evidence needed by the independent
            # semantic-only metric.  Once an unsafe rollout nevertheless
            # completes its requested contract, stop with an explicit reason.
            if semantic_candidate and not safe.passed:
                self.termination_reason = "semantic_contract_complete"
                return
            deadline = float(self.route["deadline_seconds"])
            if self._start_time is not None and now - self._start_time >= deadline:
                self.termination_reason = "deadline"
        except Exception as exc:
            self.failure_reason = "contract_monitor_failed:{}:{}".format(
                type(exc).__name__, exc
            )

    def report(self) -> Dict[str, Any]:
        return {
            "schema_version": "language2drive.evaluator_runtime.v1",
            "setup_complete": self._setup_complete,
            "spawned_actor_ids": sorted(self._spawned_actor_ids),
            "destroyed_actor_ids": sorted(self._destroyed_actor_ids),
            "cleanup_complete": self._cleanup_complete,
            "actor_program_defaults_mps": dict(self.PROGRAM_SPEED_MPS),
            "actor_program_source": "CARLA_TrafficManager_desired_speed",
            "traffic_manager_port": self.traffic_manager_port,
            "termination_owner": "private_contract_monitor",
            "terminal_guards": {
                "ego_below_route_margin_m": self.EGO_BELOW_ROUTE_MARGIN_M,
            },
            "termination_reason": self.termination_reason,
            "failure_reason": self.failure_reason,
            "latest_monitor": dict(self._latest_monitor),
        }

    def cleanup(self) -> None:
        if self._closed:
            return
        for identifier, actor in list(self._actors.items()):
            actor_id = None
            try:
                if hasattr(actor, "set_autopilot") and self.traffic_manager_port is not None:
                    actor.set_autopilot(False, int(self.traffic_manager_port))
                actor_id = int(actor.id)
                if hasattr(self.provider, "actor_id_exists") and self.provider.actor_id_exists(actor_id):
                    self.provider.remove_actor_by_id(actor_id)
                elif bool(getattr(actor, "is_alive", False)):
                    actor.destroy()
                self._destroyed_actor_ids.append(identifier)
                event = {
                    "type": "actor_despawn",
                    "actor_id": identifier,
                    "instance_id": actor_id,
                    "source": "evaluator_actor_runtime_cleanup",
                }
                if self._trace:
                    event["sim_time"] = float(self._trace[-1]["sim_time"])
                self._final_events.append(event)
            except Exception as exc:  # Cleanup cannot turn a completed run into success.
                if self.failure_reason is None:
                    self.failure_reason = "actor_cleanup_failed:{}:{}:{}".format(
                        identifier, type(exc).__name__, exc
                    )
        self._actors.clear()
        self._cleanup_complete = True
        self._closed = True
