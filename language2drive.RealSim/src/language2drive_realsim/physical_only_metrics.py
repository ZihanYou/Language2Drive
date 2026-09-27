"""Versioned RealSim v0.2 scorer: physical safety, not traffic-law safety.

This is a separate projection of existing single-seed component records. It
never mutates the old six-rule records or treats absent evidence as a pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


PROFILE = "realsim-physical-safety-v0.2-20260921"
METRICS = ("SISR", "SemISR", "Safety", "Progress", "CFPS", "Sem-CFPS", "DrivingScore")
STRICT = ("own_contract", "opposing_contract", "progress", "deadline", "stability")
ROUTE_CORRIDOR_M = 5.0


def validated_trace_rows(path: Path) -> list[dict]:
    """Load normalized trajectory, rejecting malformed or invalid simulation rows."""
    rows = []
    previous_time = -math.inf
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError(f"blank trace row: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid trace row: {path}")
            time_s = row.get("time_s")
            if (type(time_s) not in (float, int) or not math.isfinite(time_s)
                    or time_s <= previous_time):
                raise ValueError(f"non-monotone trace time: {path}")
            previous_time = time_s
            for field in ("collision", "offroad", "render_valid"):
                if type(row.get(field)) is not bool:
                    raise ValueError(f"missing or invalid {field}: {path}")
            if ("infrastructure_failure" in row
                    and type(row["infrastructure_failure"]) is not bool):
                raise ValueError(f"invalid infrastructure_failure: {path}")
            if row["render_valid"] is False or row.get("infrastructure_failure") is True:
                raise ValueError(f"invalid simulation sample: {path}")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"incomplete trace: {path}")
    return rows


def physical_safety_from_trace(path: Path) -> bool:
    """Require a valid normalized trajectory and check collision/offroad only."""
    return not any(row["collision"] or row["offroad"] for row in validated_trace_rows(path))


def _number(value, label: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"invalid {label}")
    return float(value)


def _reference_route(pcu: dict, arm_id: str, seed: int, *, horizon_s: float | None = None) -> tuple[list[tuple[float, float]], float]:
    """Use the frozen arm-specific oracle route, clipped at the scoring horizon."""
    own = pcu["arms"][arm_id]["semantic_contract"]
    opposing = pcu["arms"][arm_id]["opposing_contract"]
    horizon = max(_number(pcu["decision_window"]["deadline_s"], "window deadline"),
                  _number(own["deadline_s"], "own deadline"),
                  _number(opposing["deadline_s"], "opposing deadline"))
    if horizon_s is not None:
        horizon = _number(horizon_s, "scoring horizon")
        if horizon <= 0:
            raise ValueError("scoring horizon must be positive")
    waypoints = pcu["oracle_plans_by_seed"][str(seed)][arm_id]["waypoints"]
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("missing frozen oracle route")
    points = []
    previous = None
    for waypoint in waypoints:
        time_s = _number(waypoint["time_s"], "route time")
        point = (_number(waypoint["x"], "route x"), _number(waypoint["y"], "route y"))
        if previous is not None and time_s <= previous[0]:
            raise ValueError("non-monotone frozen route time")
        if time_s <= horizon + 1e-9:
            points.append(point)
        elif previous is not None and previous[0] < horizon:
            fraction = (horizon - previous[0]) / (time_s - previous[0])
            points.append((previous[1][0] + fraction * (point[0] - previous[1][0]),
                           previous[1][1] + fraction * (point[1] - previous[1][1])))
            break
        else:
            break
        previous = (time_s, point)
    if len(points) < 2 or waypoints[0]["time_s"] > 0 or waypoints[-1]["time_s"] < horizon:
        raise ValueError("frozen route does not cover evaluation horizon")
    return points, horizon


def route_completion_percent(rows: list[dict], pcu: dict, arm_id: str, seed: int = 1701,
                             *, horizon_s: float | None = None) -> float:
    """Longest along-route position within 5 m, / route length to fixed horizon."""
    points, horizon = _reference_route(pcu, arm_id, seed, horizon_s=horizon_s)
    if rows[0]["time_s"] != 0 or rows[-1]["time_s"] + 1e-9 < horizon:
        raise ValueError("trace lacks fixed route-completion horizon")
    segments = []
    cumulative = 0.0
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 1e-9:
            segments.append((start, dx, dy, length, cumulative))
            cumulative += length
    if cumulative <= 0:
        raise ValueError("frozen route has zero length")
    maximum_progress = 0.0
    for row in rows:
        if row["time_s"] > horizon + 1e-9:
            break
        position = row.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ValueError("trace lacks route-completion position")
        x, y = _number(position[0], "trace x"), _number(position[1], "trace y")
        closest = (math.inf, 0.0)
        for start, dx, dy, length, base in segments:
            fraction = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / (length * length)))
            px, py = start[0] + fraction * dx, start[1] + fraction * dy
            distance = math.hypot(x - px, y - py)
            candidate = (distance, base + fraction * length)
            if distance < closest[0] - 1e-9 or (abs(distance - closest[0]) <= 1e-9 and candidate[1] > closest[1]):
                closest = candidate
        if closest[0] <= ROUTE_CORRIDOR_M:
            maximum_progress = max(maximum_progress, closest[1])
    return 100.0 * min(1.0, maximum_progress / cumulative)


def _metric(values: list[bool | None], denominator: int) -> dict:
    known = sum(value is not None for value in values)
    numerator = sum(value is True for value in values)
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator, "known": known,
            "unknown": denominator - known}


def _driving_score(values: list[float | None]) -> dict:
    known = sum(value is not None for value in values)
    total_percent = sum(value for value in values if value is not None)
    return {"percent": total_percent / 60, "known_only_percent": total_percent / known if known else None,
            "denominator": 60, "known": known, "unknown": 60 - known,
            "aggregation": "mean_route_completion_percent_over_fixed_60_slots"}


def score_document(source: dict, source_label: str, *, pcu_by_id: dict | None = None) -> dict:
    """Score one fixed 30-PCU/60-slot document under v0.2.

    `evidence_valid` remains an upstream assertion. This function validates
    trace shape/physical flags but does not certify model or ASL provenance.
    """
    rows = source.get("results")
    if not isinstance(rows, list) or len(rows) != 60 or source.get("expected_runs") != 60:
        raise ValueError(f"expected exactly 60 frozen slots: {source_label}")
    if source.get("seed") != 1701:
        raise ValueError(f"unexpected seed: {source_label}")
    slots = set()
    pairs: dict[str, dict[str, dict]] = {}
    scored = []
    for row in rows:
        key = (row.get("pcu_id"), row.get("arm_id"))
        if not isinstance(key[0], str) or key[1] not in ("R0", "R1") or key in slots:
            raise ValueError(f"duplicate or invalid slot: {source_label}")
        slots.add(key)
        if row.get("seed") != 1701:
            raise ValueError(f"slot seed mismatch: {source_label}")
        usable = row.get("execution_status") == "completed" and row.get("evidence_valid") is True
        components = row.get("components", {})
        if not isinstance(components, dict):
            raise ValueError(f"invalid component record: {key}")
        safety = None
        driving_score = None
        if usable:
            trace = row.get("origin", {}).get("trace")
            if not isinstance(trace, str) or not trace:
                raise ValueError(f"valid slot lacks trace: {key}")
            trace_path = Path(trace)
            if not trace_path.is_file():
                raise ValueError(f"valid slot trace missing: {trace_path}")
            trace_rows = validated_trace_rows(trace_path)
            safety = not any(sample["collision"] or sample["offroad"] for sample in trace_rows)
            if pcu_by_id is not None:
                pcu = pcu_by_id.get(key[0])
                if pcu is None:
                    raise ValueError(f"unknown frozen PCU: {key[0]}")
                driving_score = route_completion_percent(trace_rows, pcu, key[1])
        strict_values = [components.get(name) for name in STRICT]
        strict_known = usable and safety is not None and all(type(value) is bool for value in strict_values)
        strict = (components["own_contract"] is True
                  and components["opposing_contract"] is False
                  and safety is True
                  and all(components[name] is True for name in ("progress", "deadline", "stability"))) if strict_known else None
        semantic = None
        progress = None
        if usable:
            own, relaxed = components.get("own_contract"), components.get("semantic_success")
            if own is True or type(relaxed) is bool:
                semantic = own is True or relaxed is True
            if type(components.get("progress")) is bool:
                progress = components["progress"]
        item = {"pcu_id": key[0], "arm_id": key[1], "evidence_valid": usable,
                "SISR": strict, "SemISR": semantic, "Safety": safety, "Progress": progress,
                "DrivingScore": driving_score}
        scored.append(item)
        pairs.setdefault(key[0], {})[key[1]] = item
    if len(pairs) != 30 or any(set(pair) != {"R0", "R1"} for pair in pairs.values()):
        raise ValueError(f"expected 30 complete PCU identities: {source_label}")
    if pcu_by_id is not None and set(pairs) != set(pcu_by_id):
        raise ValueError(f"slot PCU identities differ from frozen cohort: {source_label}")
    metrics = {name: _metric([row[name] for row in scored], 60)
               for name in ("SISR", "SemISR", "Safety", "Progress")}
    for name, arm_metric in (("CFPS", "SISR"), ("Sem-CFPS", "SemISR")):
        values = []
        for pair in pairs.values():
            arms = [pair[arm][arm_metric] for arm in ("R0", "R1")]
            values.append(None if any(value is None for value in arms) else all(arms))
        metrics[name] = _metric(values, 30)
    metrics["DrivingScore"] = _driving_score([row["DrivingScore"] for row in scored])
    return {"model": source["model_name"], "configuration_id": source["configuration_id"],
            "source": source_label, "profile": PROFILE,
            "cohort_pcu_ids": sorted(pairs),
            "evidence_valid_slots": sum(row["evidence_valid"] for row in scored),
            "metrics": metrics}


def score_source(path: Path, *, pcu_by_id: dict | None = None) -> dict:
    return score_document(json.loads(path.read_text(encoding="utf-8")), str(path), pcu_by_id=pcu_by_id)
