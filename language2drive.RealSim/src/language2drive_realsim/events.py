"""Simulator-independent semantic event representation and geometry helpers."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SemanticEvent:
    time_s: float
    kind: str
    subject: str
    value: Any = True


def polygon_parts(polygon):
    """Read legacy XY rings or explicit GeoJSON Polygon/MultiPolygon parts.

    Coordinates are local XY, not geographic longitude/latitude. The source
    schema validates rings; this reader never closes, buffers or repairs them.
    """
    if isinstance(polygon, dict):
        if set(polygon) != {"type", "coordinates"}:
            raise ValueError("polygon geometry requires only type and coordinates")
        kind, coordinates = polygon["type"], polygon["coordinates"]
        if kind not in {"Polygon", "MultiPolygon"} or not isinstance(coordinates, list) or not coordinates:
            raise ValueError("nonempty Polygon or MultiPolygon required")
        parts = [coordinates] if kind == "Polygon" else coordinates
        if any(not isinstance(part, list) or not part for part in parts):
            raise ValueError("polygon parts require an exterior ring")
        return parts
    return [[polygon]]


def _ring_location(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> int:
    """Return -1 outside, 0 on boundary, 1 strictly inside one ring."""

    if len(polygon) < 3:
        return -1
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        ax, ay = float(previous[0]), float(previous[1])
        bx, by = float(current[0]), float(current[1])
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) <= 1e-9 and min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9 and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9:
            return 0
        if (ay > y) != (by > y):
            intersection = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < intersection:
                inside = not inside
        previous = current
    return 1 if inside else -1


def point_in_polygon(point: Sequence[float], polygon) -> bool:
    """Boundary-inclusive polygon union, excluding hole interiors and gaps."""
    return any(_ring_location(point, part[0]) >= 0
               and all(_ring_location(point, hole) <= 0 for hole in part[1:])
               for part in polygon_parts(polygon))


def point_to_polygon_distance(
    point: Sequence[float], polygon
) -> float:
    """Euclidean xy distance to a polygon (zero on or inside it)."""

    if point_in_polygon(point, polygon):
        return 0.0
    return min(_ring_distance(point, ring) for part in polygon_parts(polygon) for ring in part)


def _ring_distance(point, polygon):
    """Distance to a ring boundary, including when inside a hole."""
    if len(polygon) < 2:
        raise ValueError("polygon needs at least two points")
    px, py = float(point[0]), float(point[1])
    minimum = math.inf
    previous = polygon[-1]
    for current in polygon:
        ax, ay = float(previous[0]), float(previous[1])
        bx, by = float(current[0]), float(current[1])
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        fraction = 0.0 if denominator == 0.0 else (
            (px - ax) * dx + (py - ay) * dy
        ) / denominator
        fraction = min(1.0, max(0.0, fraction))
        closest_x, closest_y = ax + fraction * dx, ay + fraction * dy
        minimum = min(minimum, math.hypot(px - closest_x, py - closest_y))
        previous = current
    return minimum


def _orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def segments_intersect(
    first_a: Sequence[float],
    first_b: Sequence[float],
    second_a: Sequence[float],
    second_b: Sequence[float],
) -> bool:
    o1 = _orientation(first_a, first_b, second_a)
    o2 = _orientation(first_a, first_b, second_b)
    o3 = _orientation(second_a, second_b, first_a)
    o4 = _orientation(second_a, second_b, first_b)
    epsilon = 1e-9
    if o1 * o2 < -epsilon and o3 * o4 < -epsilon:
        return True

    def on_segment(
        start: Sequence[float], end: Sequence[float], point: Sequence[float]
    ) -> bool:
        return (
            abs(_orientation(start, end, point)) <= epsilon
            and min(float(start[0]), float(end[0])) - epsilon
            <= float(point[0])
            <= max(float(start[0]), float(end[0])) + epsilon
            and min(float(start[1]), float(end[1])) - epsilon
            <= float(point[1])
            <= max(float(start[1]), float(end[1])) + epsilon
        )

    return (
        on_segment(first_a, first_b, second_a)
        or on_segment(first_a, first_b, second_b)
        or on_segment(second_a, second_b, first_a)
        or on_segment(second_a, second_b, first_b)
    )


def validate_trace(trace: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .render_validity import NATIVE_MODE
    result = list(trace)
    previous = -math.inf
    previous_progress = -math.inf
    for index, row in enumerate(result):
        if not isinstance(row, dict):
            raise ValueError("trace row %d must be an object" % index)
        time_s = row.get("time_s")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            raise ValueError("trace row %d needs finite time_s" % index)
        if float(time_s) < 0.0:
            raise ValueError("trace timestamps must be nonnegative")
        if float(time_s) <= previous:
            raise ValueError("trace timestamps must be strictly increasing")
        previous = float(time_s)
        position = row.get("position")
        if not isinstance(position, list) or len(position) < 2:
            raise ValueError("trace row %d needs position" % index)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in position
        ):
            raise ValueError("trace row %d position must be finite" % index)
        for boolean in ("collision", "offroad", "render_valid"):
            if row.get(boolean) not in (True, False):
                raise ValueError("trace row %d needs boolean %s" % (index, boolean))
        for optional_boolean in ("image_black", "reconstruction_support_valid"):
            if optional_boolean in row and row[optional_boolean] not in (True, False):
                raise ValueError(
                    "trace row %d needs boolean %s" % (index, optional_boolean)
                )
        if row.get("render_validity_mode") == NATIVE_MODE:
            if (type(row.get("image_black")) is not bool
                or row["render_valid"] is not (not row["image_black"])
                or "reconstruction_support_valid" in row
                or row.get("recorded_path_deviation_m") is None):
                raise ValueError("native render trace has inconsistent or missing validity evidence")
        for field in ("speed_mps", "progress_m", "reconstruction_deviation_m", "recorded_path_deviation_m"):
            value = row.get(field)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError("trace row %d needs nonnegative finite %s" % (index, field))
        if row.get("progress_m") is not None:
            progress = float(row["progress_m"])
            if progress < previous_progress:
                raise ValueError("trace progress must be monotonic")
            previous_progress = progress
        for field in ("lane_ids", "regions", "scopes", "triggers", "gates_crossed"):
            values = row.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError("trace row %d has invalid %s" % (index, field))
        for field in ("relative_side", "relative_longitudinal_m", "clearance_m"):
            values = row.get(field, {})
            if not isinstance(values, dict) or any(
                not isinstance(key, str) or not key for key in values
            ):
                raise ValueError("trace row %d has invalid %s" % (index, field))
        if any(
            side not in {"LEFT", "RIGHT"}
            for side in row.get("relative_side", {}).values()
        ):
            raise ValueError("trace row %d has invalid relative side" % index)
        for field in ("relative_longitudinal_m", "clearance_m"):
            for value in row.get(field, {}).values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or (field == "clearance_m" and value < 0.0)
                ):
                    raise ValueError("trace row %d has invalid %s" % (index, field))
    return result


def derive_events(
    trace: Iterable[Dict[str, Any]], annotations: Dict[str, Any]
) -> List[SemanticEvent]:
    """Derive registered events from a normalized trajectory and private geometry."""

    rows = validate_trace(trace)
    events: List[SemanticEvent] = []
    prior_regions = set()
    prior_lanes = set()
    crossed_gates = set()
    previous_position: Optional[Sequence[float]] = None
    for row in rows:
        time_s = float(row["time_s"])
        position = row["position"]
        regions = set(row.get("regions", [])) | {
            region_id
            for region_id, polygon in annotations.get("regions", {}).items()
            if point_in_polygon(position, polygon)
        }
        lanes = set(row.get("lane_ids", [])) | {
            lane_id
            for lane_id, polygon in annotations.get("lanes", {}).items()
            if point_in_polygon(position, polygon)
        }
        for region_id in sorted(regions - prior_regions):
            events.append(SemanticEvent(time_s, "EnterRegion", region_id))
        for region_id in sorted(prior_regions - regions):
            events.append(SemanticEvent(time_s, "ExitRegion", region_id))
        for lane_id in sorted(lanes):
            events.append(SemanticEvent(time_s, "LaneOccupancy", lane_id))
        if previous_position is not None:
            for gate_id, endpoints in annotations.get("gates", {}).items():
                if gate_id not in crossed_gates and segments_intersect(
                    previous_position, position, endpoints[0], endpoints[1]
                ):
                    crossed_gates.add(gate_id)
                    events.append(SemanticEvent(time_s, "CrossGate", gate_id))
        for gate_id in row.get("gates_crossed", []):
            if gate_id not in crossed_gates:
                crossed_gates.add(gate_id)
                events.append(SemanticEvent(time_s, "CrossGate", gate_id))
        for actor_id, side in row.get("relative_side", {}).items():
            events.append(SemanticEvent(time_s, "RelativeSide", actor_id, side))
        for actor_id, clearance in row.get("clearance_m", {}).items():
            events.append(SemanticEvent(time_s, "Clearance", actor_id, float(clearance)))
        for actor_id, longitudinal in row.get("relative_longitudinal_m", {}).items():
            events.append(
                SemanticEvent(
                    time_s, "RelativeLongitudinal", actor_id, float(longitudinal)
                )
            )
        if row["collision"]:
            events.append(SemanticEvent(time_s, "Collision", "ego"))
        if row["offroad"]:
            events.append(SemanticEvent(time_s, "OffRoad", "ego"))
        if not row["render_valid"]:
            events.append(SemanticEvent(time_s, "InvalidRender", "ego"))
        progress = row.get("progress_m")
        if progress is not None:
            events.append(SemanticEvent(time_s, "Progress", "route", float(progress)))
        speed = row.get("speed_mps")
        if speed is not None:
            events.append(SemanticEvent(time_s, "Speed", "ego", float(speed)))
        prior_regions = regions
        prior_lanes = lanes
        previous_position = position
    return events


def event_rows(events: Iterable[SemanticEvent]) -> List[Dict[str, Any]]:
    return [
        {"time_s": event.time_s, "kind": event.kind, "subject": event.subject, "value": event.value}
        for event in events
    ]
