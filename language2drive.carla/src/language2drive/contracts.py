from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class Verdict:
    """Result of evaluating one contract over a complete finite trace.

    ``first_satisfied_at`` is the simulation timestamp at which the contract was
    completed.  For a full-trace invariant (``always``/``safe_legal``), success
    can only be confirmed at the final timestamp.
    """

    passed: bool
    first_satisfied_at: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _point_segment_distance(point: tuple[float, float], start: Sequence[float], end: Sequence[float]) -> float:
    x, y = point
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    ratio = 0.0 if denominator == 0 else ((x - ax) * dx + (y - ay) * dy) / denominator
    ratio = max(0.0, min(1.0, ratio))
    return math.hypot(x - (ax + ratio * dx), y - (ay + ratio * dy))


def _point_in_polygon(point: tuple[float, float], polygon: list[list[float]], tolerance: float = 0.0) -> bool:
    """Boundary-inclusive point-in-polygon with an outward metric tolerance."""

    if len(polygon) < 3 or not all(len(item) >= 2 for item in polygon):
        return False
    if not all(_finite_number(v) for item in polygon for v in item[:2]):
        return False
    x, y = point
    if not (_finite_number(x) and _finite_number(y)):
        return False

    vertices = polygon[:-1] if polygon[0][:2] == polygon[-1][:2] else polygon
    if len(vertices) < 3:
        return False
    epsilon = max(1e-9, float(tolerance))
    for index, start in enumerate(vertices):
        if _point_segment_distance((float(x), float(y)), start, vertices[(index + 1) % len(vertices)]) <= epsilon:
            return True

    inside = False
    j = len(vertices) - 1
    for i, vertex in enumerate(vertices):
        xi, yi = float(vertex[0]), float(vertex[1])
        xj, yj = float(vertices[j][0]), float(vertices[j][1])
        if (yi > y) != (yj > y):
            intersection_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < intersection_x:
                inside = not inside
        j = i
    if inside or tolerance <= 0:
        return inside

    # The boundary check above used ``tolerance`` already.  Reaching here means
    # the point is farther than that distance outside the polygon.
    return False


def _angle_error_degrees(value: float, target: float) -> float:
    return abs((float(value) - float(target) + 180.0) % 360.0 - 180.0)


def _nested_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _event_matches(event: Any, expected: Any, kind: str | None = None) -> bool:
    if isinstance(event, str):
        return event == expected
    if not isinstance(event, dict):
        return False
    if kind is not None and event.get("type") not in (None, kind, f"enter_{kind}"):
        return False
    candidates = (
        event.get("id"), event.get("name"), event.get("event"), event.get("gate_id"),
        event.get("actor_id"), event.get("value"),
    )
    return expected in candidates


class ContractEngine:
    """Evaluate the versioned contract DSL over a complete normalized trace.

    Atomic event/state predicates mean "observed at least once" when used as a
    top-level goal.  ``safe_legal`` is deliberately exceptional: it is a hard
    full-rollout invariant even without an explicit ``always`` wrapper.

    Operators produce prefix-monitor masks.  A strict ``sequence`` rejects an
    early later-stage event, a repeated earlier stage, simultaneous stages, and
    state oscillation.  This is the behavior required by Layer D rather than a
    loose subsequence match.
    """

    def __init__(
        self,
        trace: Iterable[dict[str, Any]],
        *,
        predicate_cache: dict[tuple[Any, ...], dict[int, bool]] | None = None,
    ):
        self.trace = list(trace)
        # The evaluator repeatedly scores growing prefixes of the same frozen
        # row objects. Atomic predicates are pure functions of one row and
        # their fully-resolved arguments, so retain those booleans across
        # prefix engines. The cache is caller-owned: offline one-shot scoring
        # keeps its old lifetime, while one live rollout gets an isolated cache
        # that cannot leak across routes. Strict-sequence suffix engines share
        # that same route-local cache below.
        self._predicate_cache = predicate_cache
        # Several time-aware predicates access ``times`` inside a per-row
        # loop. Rebuilding the complete timestamp list on every property access
        # made dwell/sequence monitoring super-linear within each evaluation,
        # on top of the online prefix evaluation. Freeze it once alongside the
        # existing monotonicity validation; this is an exact representation of
        # the immutable ``self.trace`` snapshot.
        self._times: list[float] = []
        previous = -math.inf
        for index, row in enumerate(self.trace):
            if not isinstance(row, dict):
                raise ValueError(f"trace row {index} must be an object")
            if "sim_time" not in row or not _finite_number(row["sim_time"]):
                raise ValueError(f"trace row {index} requires a finite sim_time")
            timestamp = float(row["sim_time"])
            if timestamp < previous:
                raise ValueError("trace sim_time must be monotonic")
            previous = timestamp
            self._times.append(timestamp)
        self.bindings: dict[str, Any] = {}
        self._last_details: dict[str, Any] = {}

    @property
    def times(self) -> list[float]:
        return self._times

    def evaluate(self, expression: dict[str, Any]) -> Verdict:
        self.bindings = {}
        self._last_details = {}
        if not self.trace:
            return Verdict(False, None, {"reason": "empty_trace"})

        mask = self._mask(expression)
        name = next(iter(expression), "")
        if name == "safe_legal":
            passed = bool(mask) and all(mask)
            index = len(mask) - 1 if passed else None
        elif name in {"eventually", "always", "within", "until", "sequence", "if_else", "bind_reference"}:
            passed = bool(mask) and bool(mask[-1])
            index = next((i for i, value in enumerate(mask) if value), None) if passed else None
            if name == "always" and passed:
                index = len(mask) - 1
        else:
            index = next((i for i, value in enumerate(mask) if value), None)
            passed = index is not None

        details = dict(self._last_details)
        if not passed and "reason" not in details:
            details["reason"] = "contract_not_satisfied"
        return Verdict(passed, None if index is None else self.times[index], details)

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            return self.bindings.get(value[1:])
        if isinstance(value, dict) and set(value) == {"$ref"}:
            return self.bindings.get(str(value["$ref"]))
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        return value

    @staticmethod
    def _cache_key(value: Any) -> Any:
        """Return a lossless hashable key for validated JSON-like DSL data."""

        if isinstance(value, dict):
            return tuple(
                (key, ContractEngine._cache_key(item))
                for key, item in sorted(value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(ContractEngine._cache_key(item) for item in value)
        return value

    @staticmethod
    def _ego(row: dict[str, Any]) -> dict[str, Any]:
        return row.get("ego", {}) if isinstance(row.get("ego", {}), dict) else {}

    @staticmethod
    def _events(row: dict[str, Any]) -> list[Any]:
        events = row.get("events", [])
        return events if isinstance(events, list) else []

    @staticmethod
    def _ego_speed(ego: dict[str, Any]) -> float | None:
        for key in ("speed_mps", "speed"):
            if _finite_number(ego.get(key)):
                return float(ego[key])
        velocity = ego.get("velocity")
        if isinstance(velocity, dict) and all(_finite_number(velocity.get(axis)) for axis in ("x", "y", "z")):
            return math.sqrt(sum(float(velocity[axis]) ** 2 for axis in ("x", "y", "z")))
        return None

    @staticmethod
    def _scope_state(row: dict[str, Any], scope: str) -> bool | None:
        if "scope" in row:
            value = row["scope"]
            if isinstance(value, str):
                return value == scope
            if isinstance(value, list):
                return scope in value
            if isinstance(value, dict):
                return value.get(scope) is True
        for key in ("scopes", "trigger_states", "triggers"):
            value = row.get(key)
            if isinstance(value, list):
                return scope in value
            if isinstance(value, dict):
                return value.get(scope) is True
        return None

    def _ego_heading(self, ego: dict[str, Any]) -> float | None:
        for key in ("heading_deg", "yaw_deg", "yaw"):
            if _finite_number(ego.get(key)):
                return float(ego[key])
        for path in ("transform.rotation.yaw", "rotation.yaw"):
            value = _nested_get(ego, path)
            if _finite_number(value):
                return float(value)
        return None

    def _ego_position(self, ego: dict[str, Any]) -> tuple[float, float] | None:
        position = ego.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 2 and all(_finite_number(v) for v in position[:2]):
            return float(position[0]), float(position[1])
        for path in ("transform.location", "location"):
            value = _nested_get(ego, path)
            if isinstance(value, dict) and _finite_number(value.get("x")) and _finite_number(value.get("y")):
                return float(value["x"]), float(value["y"])
        return None

    def _ego_footprint(self, ego: dict[str, Any]) -> list[tuple[float, float]]:
        footprint = ego.get("footprint")
        if isinstance(footprint, list) and footprint:
            points = []
            for point in footprint:
                if not isinstance(point, (list, tuple)) or len(point) < 2 or not all(_finite_number(v) for v in point[:2]):
                    return []
                points.append((float(point[0]), float(point[1])))
            return points

        position = self._ego_position(ego)
        heading = self._ego_heading(ego)
        if position is None or heading is None:
            return []
        bbox = ego.get("bounding_box", ego.get("bbox", {}))
        if not isinstance(bbox, dict):
            return []
        extent = bbox.get("extent", {})
        if isinstance(extent, dict) and _finite_number(extent.get("x")) and _finite_number(extent.get("y")):
            half_length, half_width = float(extent["x"]), float(extent["y"])
        elif _finite_number(bbox.get("length")) and _finite_number(bbox.get("width")):
            half_length, half_width = float(bbox["length"]) / 2.0, float(bbox["width"]) / 2.0
        else:
            return []
        angle = math.radians(heading)
        cosine, sine = math.cos(angle), math.sin(angle)
        output = []
        for longitudinal, lateral in ((half_length, half_width), (half_length, -half_width),
                                      (-half_length, -half_width), (-half_length, half_width)):
            output.append((position[0] + longitudinal * cosine - lateral * sine,
                           position[1] + longitudinal * sine + lateral * cosine))
        return output

    def _dwell_mask(self, args: dict[str, Any]) -> list[bool]:
        seconds = float(args.get("min_seconds", args.get("seconds", 3.0)))
        if "condition" in args:
            condition = self._mask(args["condition"])
            measured: list[float | None] = [None] * len(self.trace)
        else:
            key = str(args.get("key", ""))
            condition, measured = [], []
            for row in self.trace:
                dwell = row.get("dwell", {})
                value = dwell.get(key) if isinstance(dwell, dict) else None
                if value is None:
                    states = row.get("states", {})
                    value = states.get(key) if isinstance(states, dict) else None
                if value is None:
                    slots = row.get("ego_inside_slots", [])
                    value = key in slots if isinstance(slots, list) else None
                if value is None:
                    value = self._ego(row).get("slot_id") == key if self._ego(row).get("slot_id") is not None else None
                if isinstance(value, bool):
                    condition.append(value)
                    measured.append(None)
                elif _finite_number(value):
                    condition.append(True)
                    measured.append(float(value))
                else:
                    condition.append(False)
                    measured.append(None)

        if args.get("max_speed_mps") is not None:
            maximum = float(args["max_speed_mps"])
            for index, row in enumerate(self.trace):
                speed = self._ego_speed(self._ego(row))
                if not (_finite_number(speed) and float(speed) < maximum):
                    condition[index], measured[index] = False, None

        output: list[bool] = []
        start: float | None = None
        for index, active in enumerate(condition):
            if measured[index] is not None:
                output.append(float(measured[index]) >= seconds)
                start = self.times[index] if active else None
            elif active:
                if start is None:
                    start = self.times[index]
                output.append(self.times[index] - start + 1e-9 >= seconds)
            else:
                start = None
                output.append(False)
        return output

    def _predicate(self, name: str, raw_args: dict[str, Any]) -> list[bool]:
        args = self._resolve(raw_args)
        scope = args.get("scope")
        scope_states = [self._scope_state(row, scope) for row in self.trace] if isinstance(scope, str) else None
        has_active_scope = scope_states is None or any(value is True for value in scope_states)
        cached_rows: dict[int, bool] | None = None
        if self._predicate_cache is not None:
            cache_key = (
                name,
                self._cache_key(args),
                # Before a scoped interval first activates, every sample must
                # fail instead of being vacuously true. Separate that trace-
                # level state so activation never reuses a pre-activation row.
                has_active_scope,
            )
            cached_rows = self._predicate_cache.setdefault(cache_key, {})
        result: list[bool] = []
        for row_index, row in enumerate(self.trace):
            row_identity = id(row)
            if cached_rows is not None and row_identity in cached_rows:
                result.append(cached_rows[row_identity])
                continue
            ego = self._ego(row)
            if not has_active_scope:
                value = False
                result.append(value)
                if cached_rows is not None:
                    cached_rows[row_identity] = value
                continue
            if scope_states is not None and scope_states[row_index] is not True:
                # Scoped invariants are vacuously true outside their explicitly
                # active interval, but a trace with no active sample fails.
                value = True
                result.append(value)
                if cached_rows is not None:
                    cached_rows[row_identity] = value
                continue
            if name == "enter_gate":
                expected = args.get("id")
                result.append(expected is not None and any(_event_matches(event, expected, "gate") for event in self._events(row)))
            elif name == "actor_order":
                expected = args.get("id", args.get("value"))
                current = row.get("actor_order")
                if "order" in args and isinstance(current, list):
                    result.append(current == args["order"])
                else:
                    matched = ((expected is not None and current == expected) or
                               (expected is not None and any(_event_matches(event, expected, "actor_order")
                                                             for event in self._events(row))))
                    if not matched and isinstance(expected, str) and expected.startswith("occupied:"):
                        slot = expected.split(":", 1)[1]
                        occupancy = row.get("slot_occupancy", row.get("occupancy", {}))
                        matched = isinstance(occupancy, dict) and occupancy.get(slot) is True
                    result.append(matched)
            elif name == "inside_region":
                position = self._ego_position(ego)
                result.append(position is not None and _point_in_polygon(position, args["polygon"], float(args.get("tolerance_m", 0))))
            elif name == "vehicle_footprint_inside_slot":
                corners = self._ego_footprint(ego)
                inside = bool(corners) and all(_point_in_polygon(point, args["polygon"], float(args.get("tolerance_m", 0.2)))
                                               for point in corners)
                target_heading = args.get("heading_deg", args.get("target_heading_deg"))
                if target_heading is not None:
                    heading = self._ego_heading(ego)
                    targets = target_heading if isinstance(target_heading, list) else [target_heading]
                    tolerance = float(args.get("heading_tolerance_deg", 15.0))
                    inside = inside and heading is not None and any(_angle_error_degrees(heading, float(target)) <= tolerance
                                                                   for target in targets)
                if args.get("max_speed_mps") is not None:
                    speed = self._ego_speed(ego)
                    inside = inside and _finite_number(speed) and float(speed) < float(args["max_speed_mps"])
                result.append(inside)
            elif name == "lane_relation":
                result.append(ego.get("lane_relation") == args.get("value"))
            elif name == "speed_band":
                speed = self._ego_speed(ego)
                result.append(_finite_number(speed) and float(args["min_mps"]) <= float(speed) <= float(args["max_mps"]))
            elif name == "headway_band":
                actor_id = args.get("actor_id")
                value = None
                if actor_id is not None:
                    headways = row.get("headways", {})
                    if isinstance(headways, dict):
                        value = headways.get(actor_id)
                    if value is None:
                        actors = row.get("actors", [])
                        if isinstance(actors, list):
                            actor = next((item for item in actors if isinstance(item, dict) and
                                          actor_id in (item.get("id"), item.get("instance_id"), item.get("actor_id"),
                                                       item.get("semantic_id"), item.get("role_name"))), None)
                            if actor is not None:
                                value = actor.get("headway_seconds")
                                if value is None:
                                    distance = actor.get("longitudinal_distance_m", actor.get("relative_distance_m"))
                                    ego_speed = self._ego_speed(ego)
                                    if _finite_number(distance) and _finite_number(ego_speed) and float(ego_speed) > 1e-3:
                                        value = max(0.0, float(distance)) / float(ego_speed)
                else:
                    value = row.get("headway_seconds")
                result.append(_finite_number(value) and float(args["min_seconds"]) <= float(value) <= float(args["max_seconds"]))
            elif name == "passing_side":
                result.append(row.get("passing_side") == args.get("value"))
            elif name == "progress":
                value = row.get("progress")
                result.append(_finite_number(value) and float(value) >= float(args.get("min", 1.0)))
            elif name == "safe_legal":
                forbidden = set(args.get("forbidden_events", ["collision", "off_road", "red_light"]))
                event_names: set[Any] = set()
                for event in self._events(row):
                    if isinstance(event, str):
                        event_names.add(event)
                    elif isinstance(event, dict):
                        event_names.update(event.get(key) for key in ("type", "id", "event")
                                           if isinstance(event.get(key), str))
                result.append(not bool(forbidden.intersection(event_names)))
            elif name == "dwell":
                raise AssertionError("dwell is evaluated as a time-aware predicate")
            else:  # validate_contract should prevent this; fail closed for raw callers.
                result.append(False)
            if cached_rows is not None:
                cached_rows[row_identity] = result[-1]
        return result

    @staticmethod
    def _rising_edges(mask: list[bool]) -> list[int]:
        return [index for index, value in enumerate(mask) if value and (index == 0 or not mask[index - 1])]

    def _occurrences(self, expression: dict[str, Any], mask: list[bool] | None = None) -> list[int]:
        if not self.trace:
            return []
        mask = self._mask(expression) if mask is None else mask
        name = next(iter(expression))
        if name in {"eventually", "within", "until", "sequence", "if_else", "bind_reference"}:
            if not mask[-1]:
                return []
            first = next((index for index, value in enumerate(mask) if value), None)
            return [] if first is None else [first]
        if name == "always" or name == "safe_legal":
            return [len(mask) - 1] if all(mask) else []
        if name == "dwell":
            return self._rising_edges(mask)
        if name in {"enter_gate", "actor_order"}:
            # Event rows are discrete occurrences; consecutive rows carrying
            # the same event are duplicates, not one persistent state region.
            return [index for index, value in enumerate(mask) if value]
        return self._rising_edges(mask)

    def _latch(self, occurrences: list[int]) -> list[bool]:
        if not occurrences:
            return [False] * len(self.trace)
        first = occurrences[0]
        return [index >= first for index in range(len(self.trace))]

    def _strict_sequence_mask(self, expressions: list[dict[str, Any]]) -> list[bool]:
        """Evaluate stages on successive trace suffixes.

        A temporal sequence activates stage N only after stage N-1 completes.
        Evaluating every child against the entire trace is subtly wrong for a
        program such as ``dwell(speed), until(speed, gate)``: the vehicle must
        be allowed to accelerate *before* the dwell establishes the corridor
        invariant.  Each suffix starts on the sample following the previous
        completion, which also makes simultaneous stage events invalid.

        Discrete gate stages retain the stricter Layer-D rules.  A later gate
        seen before its turn, or a previously completed gate entered again,
        invalidates the complete program.  Persistent state predicates are
        not treated as duplicate events merely because they remain true.
        """

        def discrete_occurrences(expression: dict[str, Any]) -> list[int]:
            if not isinstance(expression, dict) or len(expression) != 1:
                return []
            name, body = next(iter(expression.items()))
            if name == "enter_gate":
                return self._occurrences(expression)
            if name in {"eventually", "within"} and isinstance(body, dict):
                return discrete_occurrences(body)
            return []

        discrete = [discrete_occurrences(item) for item in expressions]
        cursor = 0
        completions: list[int] = []
        for stage, expression in enumerate(expressions):
            if cursor >= len(self.trace):
                self._last_details.update({
                    "reason": "sequence_incomplete",
                    "completed_stages": stage,
                    "required_stages": len(expressions),
                })
                return [False] * len(self.trace)

            suffix = ContractEngine(
                self.trace[cursor:], predicate_cache=self._predicate_cache,
            )
            suffix.bindings = dict(self.bindings)
            mask = suffix._mask(expression)
            occurrences = suffix._occurrences(expression, mask)
            if not occurrences:
                for key, value in suffix._last_details.items():
                    self._last_details.setdefault(key, value)
                self._last_details.update({
                    "reason": "sequence_incomplete",
                    "completed_stages": stage,
                    "required_stages": len(expressions),
                })
                return [False] * len(self.trace)

            completion = cursor + occurrences[0]
            early_later = [
                later for later in range(stage + 1, len(expressions))
                if any(index <= completion for index in discrete[later])
            ]
            repeated_earlier = [
                earlier for earlier in range(stage)
                if any(index >= cursor for index in discrete[earlier])
            ]
            if early_later or repeated_earlier:
                self._last_details.update({
                    "reason": "sequence_order_violation",
                    "at": self.times[completion],
                    "expected_stage": stage,
                    "early_later_stages": early_later,
                    "repeated_earlier_stages": repeated_earlier,
                })
                return [False] * len(self.trace)
            completions.append(completion)
            cursor = completion + 1

        # A repeat after final completion is still a strict-program error.
        final_completion = completions[-1]
        repeated = [
            stage for stage, occurrences in enumerate(discrete)
            if any(index > completions[stage] for index in occurrences)
        ]
        if repeated:
            self._last_details.update({
                "reason": "sequence_order_violation",
                "at": self.times[min(
                    index for stage in repeated for index in discrete[stage]
                    if index > completions[stage]
                )],
                "repeated_stages": repeated,
            })
            return [False] * len(self.trace)
        return [index >= final_completion for index in range(len(self.trace))]

    def _bind_reference_mask(self, body: dict[str, Any]) -> list[bool]:
        reference = str(body["reference"])
        eligible = self._mask(body["when"]) if "when" in body else [True] * len(self.trace)
        binding_index: int | None = None
        binding_value: Any = None
        for index, row in enumerate(self.trace):
            if not eligible[index]:
                continue
            if "value" in body:
                value = body["value"]
            elif "source" in body:
                value = _nested_get(row, str(body["source"]))
            else:
                references = row.get("references", {})
                bindings = row.get("bindings", {})
                value = references.get(reference) if isinstance(references, dict) else None
                if value is None and isinstance(bindings, dict):
                    value = bindings.get(reference)
                if value is None:
                    value = row.get(reference)
            if value is not None:
                binding_index, binding_value = index, value
                break
        # Backward-compatible constant binding: fixed-layout programs often put
        # the canonical semantic entity directly in ``reference`` and do not
        # use it as a lookup key in the body.
        if binding_index is None and "source" not in body and "value" not in body:
            binding_index, binding_value = 0, reference
        if binding_index is None:
            self._last_details.update({"reason": "reference_not_bound", "reference": reference})
            return [False] * len(self.trace)

        self.bindings[reference] = binding_value
        # Evaluate the body on the suffix beginning at the bind point.  Merely
        # masking earlier rows would incorrectly make a suffix ``always`` fail
        # and would allow pre-binding events to poison a strict sequence.
        suffix_engine = ContractEngine(
            self.trace[binding_index:], predicate_cache=self._predicate_cache,
        )
        suffix_engine.bindings = dict(self.bindings)
        child = suffix_engine._mask(body["body"])
        occurrences = [binding_index + index for index in suffix_engine._occurrences(body["body"], child)]
        for key, value in suffix_engine._last_details.items():
            self._last_details.setdefault(key, value)
        return self._latch(occurrences)

    def _mask(self, expression: dict[str, Any]) -> list[bool]:
        if not isinstance(expression, dict) or len(expression) != 1:
            return [False] * len(self.trace)
        name, body = next(iter(expression.items()))
        if name == "dwell":
            return self._dwell_mask(body)
        if name not in {"eventually", "always", "within", "until", "sequence", "if_else", "bind_reference"}:
            child = self._predicate(name, body)
            if body.get("stable_seconds") is not None:
                seconds = float(body["stable_seconds"])
                output: list[bool] = []
                start: float | None = None
                for index, active in enumerate(child):
                    if active:
                        if start is None:
                            start = self.times[index]
                        output.append(self.times[index] - start + 1e-9 >= seconds)
                    else:
                        start = None
                        output.append(False)
                return output
            return child
        if name == "eventually":
            child = self._mask(body)
            return self._latch(self._occurrences(body, child))
        if name == "within":
            child_expression = body["condition"]
            child = self._mask(child_expression)
            cutoff = self.times[0] + float(body["seconds"])
            occurrences = [
                index for index in self._occurrences(child_expression, child)
                if self.times[index] <= cutoff + 1e-9
            ]
            if not occurrences:
                self._last_details.setdefault(
                    "reason", "within_condition_not_reached",
                )
            return self._latch(occurrences)
        if name == "always":
            child = self._mask(body)
            valid = True
            output = []
            for value in child:
                valid = valid and value
                output.append(valid)
            return output
        if name == "until":
            invariant = self._mask(body["invariant"])
            condition = self._mask(body["condition"])
            condition_points = set(self._occurrences(body["condition"], condition))
            valid, completed = True, False
            output = []
            for index, inv in enumerate(invariant):
                if not completed and index in condition_points:
                    completed = valid
                elif not completed:
                    valid = valid and inv
                output.append(completed)
            if not completed:
                self._last_details.setdefault("reason", "until_condition_not_reached" if valid else "until_invariant_violated")
            return output
        if name == "sequence":
            return self._strict_sequence_mask(body)
        if name == "if_else":
            condition = self._mask(body["condition"])
            # The condition is sampled when the program activates (the first
            # trace row).  Looking for it anywhere in the future would let a
            # later road closure retroactively change an already executed
            # branch.  Dynamic/revision programs require an explicit trigger
            # and are outside fixed-layout v1.
            selected = "then" if condition and condition[0] else "else"
            self._last_details.update({"selected_branch": selected,
                                       "condition_evaluated_at": self.times[0] if self.trace else None})
            branch = self._mask(body[selected])
            return self._latch(self._occurrences(body[selected], branch))
        return self._bind_reference_mask(body)


def _semantic_ego(row: dict[str, Any]) -> dict[str, Any]:
    ego = row.get("ego", {})
    return ego if isinstance(ego, dict) else {}


def _semantic_position(row: dict[str, Any]) -> tuple[float, float] | None:
    ego = _semantic_ego(row)
    for key in ("location", "position"):
        value = ego.get(key)
        if isinstance(value, dict) and all(
            _finite_number(value.get(axis)) for axis in ("x", "y")
        ):
            return float(value["x"]), float(value["y"])
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(_finite_number(item) for item in value[:2])
        ):
            return float(value[0]), float(value[1])
    value = _nested_get(ego, "transform.location")
    if isinstance(value, dict) and all(
        _finite_number(value.get(axis)) for axis in ("x", "y")
    ):
        return float(value["x"]), float(value["y"])
    return None


def _semantic_heading(row: dict[str, Any]) -> float | None:
    ego = _semantic_ego(row)
    for key in ("heading_deg", "yaw_deg", "yaw"):
        if _finite_number(ego.get(key)):
            return float(ego[key])
    for path in ("rotation.yaw", "transform.rotation.yaw"):
        value = _nested_get(ego, path)
        if _finite_number(value):
            return float(value)
    return None


def _semantic_speed(row: dict[str, Any]) -> float | None:
    ego = _semantic_ego(row)
    for key in ("speed_mps", "speed"):
        if _finite_number(ego.get(key)):
            return float(ego[key])
    return None


def _semantic_progress(row: dict[str, Any]) -> float | None:
    value = row.get("progress")
    return float(value) if _finite_number(value) else None


def _semantic_altitude(row: dict[str, Any]) -> float | None:
    ego = _semantic_ego(row)
    for path in ("location.z", "transform.location.z"):
        value = _nested_get(ego, path)
        if _finite_number(value):
            return float(value)
    return None


def _semantic_first(
    trace: list[dict[str, Any]], predicate,
) -> Verdict:
    for row in trace:
        if predicate(row):
            return Verdict(True, float(row["sim_time"]), {})
    return Verdict(False, None, {})


def _semantic_event(row: dict[str, Any], identifier: str) -> bool:
    events = row.get("events", [])
    return isinstance(events, list) and any(
        _event_matches(event, identifier) for event in events
    )


def _semantic_gate_verdict(
    trace: list[dict[str, Any]],
    target: dict[str, Any],
    oracle_path: list[dict[str, Any]],
) -> Verdict:
    """Accept clear destination/turn intent without requiring exact gate entry."""

    initial_position = next(
        (position for row in trace if (position := _semantic_position(row)) is not None),
        None,
    )
    initial_altitude = next(
        (value for row in trace if (value := _semantic_altitude(row)) is not None),
        None,
    )
    endpoint = oracle_path[-1] if oracle_path else {}
    endpoint_yaw = endpoint.get("yaw_deg") if isinstance(endpoint, dict) else None
    maneuver = target.get("maneuver")
    is_turn = maneuver in {"left", "right"}
    progress_minimum = 0.25 if is_turn else 0.45
    displacement_minimum = 8.0 if is_turn else 12.0
    heading_tolerance = 45.0 if is_turn else 35.0

    def heading_proxy(row: dict[str, Any]) -> bool:
        position = _semantic_position(row)
        heading = _semantic_heading(row)
        progress = _semantic_progress(row)
        altitude = _semantic_altitude(row)
        if (
            position is None
            or initial_position is None
            or heading is None
            or not _finite_number(endpoint_yaw)
            or progress is None
            or progress < progress_minimum
            or math.dist(position, initial_position) < displacement_minimum
        ):
            return False
        # Ignore physically invalid falling/teleport frames without consulting
        # any collision, red-light, lane-legality or safety event.
        if (
            initial_altitude is not None
            and altitude is not None
            and abs(altitude - initial_altitude) > 3.0
        ):
            return False
        return _angle_error_degrees(heading, float(endpoint_yaw)) <= heading_tolerance

    heading = _semantic_first(trace, heading_proxy)
    if heading.passed:
        heading.details.update({
            "method": "loose_endpoint_heading",
            "maneuver": maneuver,
            "target_heading_deg": float(endpoint_yaw),
            "heading_tolerance_deg": heading_tolerance,
            "progress_minimum": progress_minimum,
            "displacement_minimum_m": displacement_minimum,
        })
        return heading

    center = target.get("center")
    if isinstance(center, dict) and all(
        _finite_number(center.get(axis)) for axis in ("x", "y")
    ):
        radius = float(target.get("radius_m", 0.0)) + 5.0
        near = _semantic_first(
            trace,
            lambda row: (
                (position := _semantic_position(row)) is not None
                and math.dist(position, (float(center["x"]), float(center["y"])))
                <= radius
            ),
        )
        if near.passed:
            near.details.update({"method": "loose_gate_radius", "radius_m": radius})
            return near

    polygon = target.get("polygon")
    if isinstance(polygon, list):
        near = _semantic_first(
            trace,
            lambda row: (
                (position := _semantic_position(row)) is not None
                and _point_in_polygon(position, polygon, 4.0)
            ),
        )
        if near.passed:
            near.details.update({"method": "loose_gate_polygon", "tolerance_m": 4.0})
            return near
    return Verdict(False, None, {"reason": "no_loose_gate_evidence"})


def _semantic_parking_verdict(
    trace: list[dict[str, Any]], target: dict[str, Any]
) -> Verdict:
    polygon = target.get("polygon")
    heading_target = target.get("heading_deg")
    if not isinstance(polygon, list):
        return Verdict(False, None, {"reason": "parking_polygon_missing"})

    def predicate(row: dict[str, Any]) -> bool:
        position = _semantic_position(row)
        heading = _semantic_heading(row)
        return bool(
            position is not None
            and _point_in_polygon(position, polygon, 1.5)
            and (
                not _finite_number(heading_target)
                or (
                    heading is not None
                    and _angle_error_degrees(heading, float(heading_target)) <= 45.0
                )
            )
        )

    verdict = _semantic_first(trace, predicate)
    verdict.details.update({
        "method": "loose_parking_center_and_heading",
        "polygon_tolerance_m": 1.5,
        "heading_tolerance_deg": 45.0,
        "dwell_and_speed_required": False,
    })
    return verdict


def _semantic_stage_verdict(
    trace: list[dict[str, Any]], stages: list[Any]
) -> Verdict:
    expected = [
        value if ":" in value else f"parking-stage:{value}"
        for value in stages
        if isinstance(value, str) and value
    ]
    if not expected:
        return Verdict(False, None, {"reason": "semantic_stages_missing"})
    cursor = 0
    completed_at = None
    for row in trace:
        if _semantic_event(row, expected[cursor]):
            cursor += 1
            completed_at = float(row["sim_time"])
            if cursor == len(expected):
                return Verdict(True, completed_at, {
                    "method": "loose_ordered_stage_events", "stages": expected,
                })
    return Verdict(False, None, {
        "reason": "ordered_stages_incomplete", "completed_stages": cursor,
        "required_stages": len(expected),
    })


def _semantic_target_verdict(
    trace: list[dict[str, Any]],
    target: dict[str, Any],
    oracle_path: list[dict[str, Any]],
) -> Verdict:
    kind = target.get("type")
    if kind == "gate":
        return _semantic_gate_verdict(trace, target, oracle_path)
    if kind == "lane":
        identifier = target.get("id")
        verdict = _semantic_first(trace, lambda row: (
            _semantic_ego(row).get("lane_relation") == identifier
            and (_semantic_progress(row) or 0.0) >= 0.2
        ))
        verdict.details.update({"method": "loose_lane_observation", "progress_minimum": 0.2})
        return verdict
    if kind in {"speed_band", "headway"}:
        minimum = float(target.get("min_mps", target.get("min_seconds", 0.0)))
        maximum = float(target.get("max_mps", target.get("max_seconds", 0.0)))
        slack = 1.0
        if kind == "speed_band":
            value_at = _semantic_speed
        else:
            actor_id = target.get("actor_id")

            def value_at(row: dict[str, Any]) -> float | None:
                values = row.get("headways", {})
                value = values.get(actor_id) if isinstance(values, dict) else None
                return float(value) if _finite_number(value) else None

        verdict = _semantic_first(trace, lambda row: (
            (value := value_at(row)) is not None
            and minimum - slack <= value <= maximum + slack
        ))
        verdict.details.update({
            "method": f"loose_{kind}_observation", "range_slack": slack,
            "dwell_required": False,
        })
        return verdict
    if kind in {"progress", "delayed_progress"}:
        threshold = 0.8 * float(target.get("min", 1.0))
        start = float(trace[0]["sim_time"])
        wait = 0.5 * float(target.get("wait_seconds", 0.0))
        verdict = _semantic_first(trace, lambda row: (
            float(row["sim_time"]) - start >= wait
            and (_semantic_progress(row) or 0.0) >= threshold
        ))
        verdict.details.update({
            "method": "loose_delayed_progress" if kind == "delayed_progress" else "loose_progress",
            "progress_minimum": threshold, "wait_seconds": wait,
        })
        return verdict
    if kind == "parking_slot":
        return _semantic_parking_verdict(trace, target)
    if kind == "region":
        polygon = target.get("polygon")
        if isinstance(polygon, list):
            verdict = _semantic_first(trace, lambda row: (
                (position := _semantic_position(row)) is not None
                and _point_in_polygon(position, polygon, 2.0)
            ))
        else:
            pose = target.get("terminal_pose", {})
            verdict = _semantic_first(trace, lambda row: bool(
                isinstance(pose, dict)
                and all(_finite_number(pose.get(axis)) for axis in ("x", "y"))
                and (position := _semantic_position(row)) is not None
                and math.dist(position, (float(pose["x"]), float(pose["y"]))) <= 8.0
            ))
        verdict.details.update({"method": "loose_region_position"})
        return verdict
    if kind == "until":
        constraint = dict(target.get("constraint", {}))
        constraint.setdefault("type", "speed_band")
        constraint_verdict = _semantic_target_verdict(trace, constraint, oracle_path)
        end = target.get("end", {})
        end_verdict = _semantic_target_verdict(trace, end, oracle_path)
        passed = constraint_verdict.passed and end_verdict.passed
        completion = max(
            value for value in (
                constraint_verdict.first_satisfied_at,
                end_verdict.first_satisfied_at,
            ) if value is not None
        ) if passed else None
        return Verdict(passed, completion, {
            "method": "loose_constraint_and_endpoint",
            "constraint": constraint_verdict.details,
            "endpoint": end_verdict.details,
        })
    if kind == "actor_order":
        expected = f"{target.get('order')}:{target.get('actor_id')}"
        verdict = _semantic_first(trace, lambda row: _semantic_event(row, expected))
        verdict.details.update({"method": "actor_order_event", "expected": expected})
        return verdict
    if kind in {"sequence", "memory"}:
        return _semantic_stage_verdict(trace, target.get("stages", []))
    if kind == "if_else":
        condition = str(target.get("condition", ""))
        slot = condition.split(":", 1)[1] if condition.startswith("occupied:") else ""
        occupancy = trace[0].get("occupancy", trace[0].get("slot_occupancy", {}))
        selected = "then" if isinstance(occupancy, dict) and occupancy.get(slot) is True else "else"
        branch = target.get(selected, {})
        verdict = _semantic_target_verdict(trace, branch, oracle_path)
        verdict.details.update({"selected_branch": selected})
        return verdict
    return Verdict(False, None, {"reason": f"no_loose_proxy_for_{kind}"})


def semantic_instruction_verdict(
    trace: Iterable[dict[str, Any]],
    *,
    target: dict[str, Any],
    oracle_path: list[dict[str, Any]],
    own: Verdict | None = None,
    layer: str | None = None,
) -> Verdict:
    """Score instruction meaning independently from legality and strict gates.

    Exact contract completion always passes.  Otherwise this uses a deliberately
    permissive target-specific proxy: endpoint heading for junction choices,
    observed lane/speed/headway/order for behavioral instructions, and expanded
    center/heading geometry without dwell for destinations and parking.
    """

    rows = list(trace)
    if own is not None and own.passed:
        return Verdict(True, own.first_satisfied_at, {
            "method": "exact_own_contract", "layer": layer,
        })
    if not rows:
        return Verdict(False, None, {"reason": "empty_trace", "layer": layer})
    verdict = _semantic_target_verdict(rows, target, oracle_path)
    verdict.details.setdefault("layer", layer)
    return verdict


def score_rollout(
    own: Verdict,
    opposite: Verdict,
    safe: Verdict,
    progress: Verdict,
    sim_time: float,
    deadline: float,
    stable: Verdict,
    *,
    start_time: float = 0.0,
    semantic: Verdict | None = None,
) -> dict[str, Any]:
    """Apply the benchmark's conjunctive success rule.

    Deadline is evaluated against task completion, not logger shutdown.  The
    completion timestamp is the latest known timestamp of own-contract,
    progress, and stability; ``sim_time`` is a conservative fallback for legacy
    verdicts without timestamps.
    """

    if not all(_finite_number(value) for value in (sim_time, deadline, start_time)):
        raise ValueError("sim_time, deadline, and start_time must be finite numbers")
    if float(deadline) < 0:
        raise ValueError("deadline must be non-negative")

    completion_candidates = [
        verdict.first_satisfied_at if verdict.first_satisfied_at is not None else float(sim_time)
        for verdict in (own, progress, stable)
        if verdict.passed
    ]
    completion_at = max(completion_candidates) if completion_candidates else float(sim_time)
    elapsed = max(0.0, float(completion_at) - float(start_time))
    within_deadline = elapsed <= float(deadline) + 1e-9
    passed = (
        own.passed
        and not opposite.passed
        and safe.passed
        and progress.passed
        and within_deadline
        and stable.passed
    )
    semantic_verdict = semantic if semantic is not None else own
    semantic_time = semantic_verdict.first_satisfied_at
    return {
        "success": passed,
        # An intentionally loose diagnostic axis.  Its verdict is supplied by
        # ``semantic_instruction_verdict`` and is independent of every safety
        # and strict-success gate.  Falling back to ``own`` keeps callers that
        # do not have route metadata backwards compatible.
        "semantic_success": semantic_verdict.passed,
        "semantic_completion_time": (
            None
            if semantic_time is None
            else max(0.0, float(semantic_time) - float(start_time))
        ),
        "own_contract": own.passed,
        "opposite_contract": opposite.passed,
        "safe_legal": safe.passed,
        "progress": progress.passed,
        "deadline": within_deadline,
        "stability": stable.passed,
        "completion_time": elapsed,
        "deadline_seconds": float(deadline),
    }
