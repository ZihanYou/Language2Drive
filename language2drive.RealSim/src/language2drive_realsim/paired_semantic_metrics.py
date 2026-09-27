"""Continuous paired language-response metrics for frozen RealSim PCUs.

This module implements the SCS/PCS/SRG/NBD definitions without changing the
existing binary verifier.  The continuous metrics apply to PCUs whose two
contracts expose persistent speed-band or lane-region targets.  Gate-only
straight/turn PCUs are deliberately not coerced into a region metric.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path
from typing import Iterable

from .events import point_in_polygon, point_to_polygon_distance
from .physical_only_metrics import validated_trace_rows


PROFILE = "realsim-paired-semantic-v0.1-20260923"
METRICS = ("SCS", "PCS", "SRG", "NBD")
APPLICABLE_KINDS = {"speed_band", "triggered_speed_band", "lane", "scoped_lane"}
SPEED_KINDS = {"speed_band", "triggered_speed_band"}
LANE_KINDS = {"lane", "scoped_lane"}
RESAMPLE_INTERVAL_S = 0.1
PROGRESS_NORMALIZER_M = 8.0


def _number(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid " + label)
    return float(value)


def task_group(pcu: dict) -> str | None:
    """Return the continuous-metric group, or None for gate-only PCUs."""
    kinds = {pcu["arms"][arm]["semantic_contract"]["kind"] for arm in ("R0", "R1")}
    if kinds <= SPEED_KINDS:
        return "speed"
    if kinds <= LANE_KINDS:
        return "lane"
    if kinds.isdisjoint(APPLICABLE_KINDS):
        return None
    raise ValueError("PCU mixes incompatible continuous semantic target kinds")


def _horizon(pcu: dict, arm_id: str) -> tuple[float, float]:
    contract = pcu["arms"][arm_id]["semantic_contract"]
    return (_number(pcu["decision_window"]["start_s"], "decision start"),
            min(_number(pcu["decision_window"]["deadline_s"], "decision deadline"),
                _number(contract["deadline_s"], "contract deadline")))


def _geometry(pcu: dict, collection: str, identifier: str):
    try:
        return pcu["private_annotations"][collection][identifier]
    except KeyError as exc:
        raise ValueError(f"missing frozen {collection} geometry: {identifier}") from exc


def _scope_geometry(pcu: dict, arm_id: str):
    contract = pcu["arms"][arm_id]["semantic_contract"]
    scope_id = contract.get("scope")
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError("continuous semantic contract lacks a scope")
    return _geometry(pcu, "scopes", scope_id)


def _target_distance(pcu: dict, arm_id: str, *, position, speed_mps) -> float:
    contract = pcu["arms"][arm_id]["semantic_contract"]
    kind = contract["kind"]
    if kind in SPEED_KINDS:
        speed = _number(speed_mps, "speed")
        lower = _number(contract["speed_min_mps"], "speed lower bound")
        upper = _number(contract["speed_max_mps"], "speed upper bound")
        if lower > upper:
            raise ValueError("reversed speed band")
        return max(lower - speed, 0.0, speed - upper)
    if kind in LANE_KINDS:
        lane = contract.get("target_lane")
        return point_to_polygon_distance(position, _geometry(pcu, "lanes", lane))
    raise ValueError("continuous target distance requested for unsupported contract")


def _target_satisfied(pcu: dict, arm_id: str, row: dict) -> bool:
    return _target_distance(pcu, arm_id, position=row["position"],
                            speed_mps=row.get("speed_mps")) <= 1e-9


def _active_rows(pcu: dict, arm_id: str, rows: list[dict]) -> list[dict]:
    start, end = _horizon(pcu, arm_id)
    scope = _scope_geometry(pcu, arm_id)
    return [row for row in rows
            if start - 1e-9 <= _number(row["time_s"], "trace time") <= end + 1e-9
            and point_in_polygon(row["position"], scope)]


def _interpolate(rows: list[dict], time_s: float) -> dict:
    times = [float(row["time_s"]) for row in rows]
    if time_s < times[0] - 1e-9 or time_s > times[-1] + 1e-9:
        raise ValueError("trace does not cover paired comparison time")
    index = bisect_right(times, time_s)
    if index == 0:
        left = right = rows[0]
    elif index == len(rows):
        left = right = rows[-1]
    else:
        left, right = rows[index - 1], rows[index]
    if left is right or abs(float(left["time_s"]) - time_s) <= 1e-12:
        fraction = 0.0
    else:
        fraction = ((time_s - float(left["time_s"])) /
                    (float(right["time_s"]) - float(left["time_s"])))

    def linear(field: str) -> float:
        first, second = left.get(field), right.get(field)
        return _number(first, field) + fraction * (_number(second, field) - _number(first, field))

    first_position, second_position = left["position"], right["position"]
    if len(first_position) < 2 or len(second_position) < 2:
        raise ValueError("trace position lacks xy coordinates")
    return {
        "time_s": time_s,
        "position": [
            _number(first_position[0], "x") + fraction * (
                _number(second_position[0], "x") - _number(first_position[0], "x")),
            _number(first_position[1], "y") + fraction * (
                _number(second_position[1], "y") - _number(first_position[1], "y")),
        ],
        "speed_mps": linear("speed_mps"),
        "progress_m": linear("progress_m"),
    }


def _grid(start: float, end: float) -> list[float]:
    if end < start:
        return []
    count = int(math.floor((end - start) / RESAMPLE_INTERVAL_S + 1e-9))
    values = [start + index * RESAMPLE_INTERVAL_S for index in range(count + 1)]
    if not values or end - values[-1] > 1e-8:
        values.append(end)
    return values


def score_arm(pcu: dict, arm_id: str, rows: list[dict]) -> dict:
    """Compute one arm's C_j and its factual factors."""
    if task_group(pcu) is None:
        raise ValueError("gate-only PCU has no continuous semantic target region")
    start, end = _horizon(pcu, arm_id)
    if rows[0]["time_s"] > 0.0 or rows[-1]["time_s"] + 1e-9 < end:
        raise ValueError("trace lacks the fixed continuous-metric horizon")
    active = _active_rows(pcu, arm_id, rows)
    satisfied = sum(_target_satisfied(pcu, arm_id, row) for row in active)
    satisfaction_ratio = satisfied / len(active) if active else 0.0
    initial = _interpolate(rows, 0.0)["progress_m"]
    terminal = _interpolate(rows, end)["progress_m"]
    forward = max(0.0, terminal - initial)
    progress_factor = min(1.0, forward / PROGRESS_NORMALIZER_M)
    compliance = satisfaction_ratio * progress_factor
    return {
        "arm_id": arm_id,
        "start_s": start,
        "end_s": end,
        "active_samples": len(active),
        "satisfied_samples": satisfied,
        "satisfaction_ratio": satisfaction_ratio,
        "terminal_forward_distance_m": forward,
        "progress_factor": progress_factor,
        "compliance": compliance,
    }


def score_pair(pcu: dict, rows_r0: list[dict], rows_r1: list[dict]) -> dict:
    """Compute one PCU's continuous pair values on validated real traces."""
    group = task_group(pcu)
    if group is None:
        return {"applicable": False, "reason": "gate_only_without_persistent_target_region"}
    arms = {"R0": score_arm(pcu, "R0", rows_r0),
            "R1": score_arm(pcu, "R1", rows_r1)}
    pcs = min(arms["R0"]["compliance"], arms["R1"]["compliance"])

    starts_ends = [_horizon(pcu, arm) for arm in ("R0", "R1")]
    start = max(value[0] for value in starts_ends)
    end = min(value[1] for value in starts_ends)
    scope0, scope1 = _scope_geometry(pcu, "R0"), _scope_geometry(pcu, "R1")
    paired = []
    for time_s in _grid(start, end):
        sample0, sample1 = _interpolate(rows_r0, time_s), _interpolate(rows_r1, time_s)
        if (point_in_polygon(sample0["position"], scope0)
                and point_in_polygon(sample1["position"], scope1)):
            paired.append((sample0, sample1))

    preferences = {}
    for trajectory_arm, index in (("R0", 0), ("R1", 1)):
        values = []
        for samples in paired:
            sample = samples[index]
            distances = [_target_distance(
                pcu, target_arm, position=sample["position"], speed_mps=sample["speed_mps"])
                         for target_arm in ("R0", "R1")]
            denominator = distances[0] + distances[1]
            values.append(0.0 if denominator <= 1e-12
                          else (distances[1] - distances[0]) / denominator)
        preferences[trajectory_arm] = sum(values) / len(values) if values else 0.0
    srg = (preferences["R0"] - preferences["R1"]) / 2.0

    position0, position1 = rows_r0[0]["position"], rows_r1[0]["position"]
    if math.hypot(float(position0[0]) - float(position1[0]),
                  float(position0[1]) - float(position1[1])) > 1e-6:
        raise ValueError("paired trajectories do not share the same scene origin")
    origin = (float(position0[0]), float(position0[1]))
    numerator = denominator = 0.0
    nbd_samples = _grid(start, end)
    for time_s in nbd_samples:
        sample0, sample1 = _interpolate(rows_r0, time_s), _interpolate(rows_r1, time_s)
        xy0 = (sample0["position"][0] - origin[0], sample0["position"][1] - origin[1])
        xy1 = (sample1["position"][0] - origin[0], sample1["position"][1] - origin[1])
        numerator += math.hypot(xy0[0] - xy1[0], xy0[1] - xy1[1])
        denominator += math.hypot(*xy0) + math.hypot(*xy1)
    nbd = 0.0 if denominator <= 1e-12 else numerator / denominator
    # Triangle inequality guarantees <= 1 except for floating roundoff.
    nbd = min(1.0, max(0.0, nbd))
    return {
        "applicable": True,
        "task_group": group,
        "arms": arms,
        "PCS": pcs,
        "preference_R0_trajectory": preferences["R0"],
        "preference_R1_trajectory": preferences["R1"],
        "SRG": srg,
        "NBD": nbd,
        "common_effective_samples": len(paired),
        "nbd_samples": len(nbd_samples),
        "comparison_window_s": [start, end],
        "scene_origin_xy": list(origin),
    }


def _bounded_metric(values: Iterable[float | None], *, minimum: float, maximum: float) -> dict:
    values = list(values)
    known_values = [value for value in values if value is not None]
    if any(not minimum - 1e-9 <= value <= maximum + 1e-9 for value in known_values):
        raise ValueError("continuous metric escaped its defined range")
    denominator, known = len(values), len(known_values)
    unknown = denominator - known
    total = sum(known_values)
    return {
        "percent": 100.0 * total / denominator if unknown == 0 and denominator else None,
        "known_only_percent": 100.0 * total / known if known else None,
        "lower_bound_percent": 100.0 * (total + unknown * minimum) / denominator if denominator else None,
        "upper_bound_percent": 100.0 * (total + unknown * maximum) / denominator if denominator else None,
        "known": known,
        "unknown": unknown,
        "denominator": denominator,
    }


def score_document(document: dict, pcus: list[dict], source_label: str = "") -> dict:
    """Score one 30-PCU component document and retain missing evidence as unknown."""
    pcu_by_id = {pcu["pcu_id"]: pcu for pcu in pcus}
    if len(pcus) != 30 or len(pcu_by_id) != 30:
        raise ValueError("expected 30 unique frozen PCUs")
    rows = document.get("results")
    if not isinstance(rows, list) or len(rows) != 60 or document.get("expected_runs") != 60:
        raise ValueError("expected a fixed 60-slot experiment document")
    if document.get("seed") != 1701:
        raise ValueError("continuous metrics require the frozen seed 1701 cohort")
    slots = {}
    for row in rows:
        key = (row.get("pcu_id"), row.get("arm_id"))
        if (key[0] not in pcu_by_id or key[1] not in {"R0", "R1"} or key in slots):
            raise ValueError("invalid, unknown, or duplicate experiment slot")
        slots[key] = row
    expected = {(pcu_id, arm) for pcu_id in pcu_by_id for arm in ("R0", "R1")}
    if set(slots) != expected:
        raise ValueError("experiment slots differ from frozen cohort")

    details = []
    for pcu in pcus:
        group = task_group(pcu)
        if group is None:
            details.append({"pcu_id": pcu["pcu_id"], "task_family": pcu["task_family"],
                            "applicable": False,
                            "reason": "gate_only_without_persistent_target_region"})
            continue
        arm_rows, missing = {}, []
        for arm in ("R0", "R1"):
            record = slots[(pcu["pcu_id"], arm)]
            usable = (record.get("execution_status") == "completed"
                      and record.get("evidence_valid") is True)
            trace = record.get("origin", {}).get("trace") if usable else None
            if usable and (not isinstance(trace, str) or not trace or not Path(trace).is_file()):
                raise ValueError("evidence-valid slot lacks its normalized trace")
            if not usable:
                missing.append(arm)
                continue
            arm_rows[arm] = validated_trace_rows(Path(trace))
        if missing:
            arm_scores = {arm: score_arm(pcu, arm, trace_rows)
                          for arm, trace_rows in arm_rows.items()}
            details.append({"pcu_id": pcu["pcu_id"], "task_family": pcu["task_family"],
                            "task_group": group, "applicable": True,
                            "evidence_complete": False, "missing_arms": missing,
                            "arms": arm_scores})
            continue
        pair = score_pair(pcu, arm_rows["R0"], arm_rows["R1"])
        pair.update(pcu_id=pcu["pcu_id"], task_family=pcu["task_family"],
                    evidence_complete=True)
        details.append(pair)

    applicable = [item for item in details if item.get("applicable")]
    metrics = {
        "SCS": _bounded_metric([
            item.get("arms", {}).get(arm, {}).get("compliance")
            for item in applicable for arm in ("R0", "R1")], minimum=0.0, maximum=1.0),
        "PCS": _bounded_metric([
            item.get("PCS") if item.get("evidence_complete") else None
            for item in applicable], minimum=0.0, maximum=1.0),
        "SRG": _bounded_metric([
            item.get("SRG") if item.get("evidence_complete") else None
            for item in applicable], minimum=-1.0, maximum=1.0),
        "NBD": _bounded_metric([
            item.get("NBD") if item.get("evidence_complete") else None
            for item in applicable], minimum=0.0, maximum=1.0),
    }
    by_group = {}
    for group in ("speed", "lane"):
        subset = [item for item in applicable if item["task_group"] == group]
        by_group[group] = {
            "pcus": len(subset),
            "SCS": _bounded_metric([
                item.get("arms", {}).get(arm, {}).get("compliance")
                for item in subset for arm in ("R0", "R1")], minimum=0.0, maximum=1.0),
            "PCS": _bounded_metric([
                item.get("PCS") if item.get("evidence_complete") else None for item in subset],
                minimum=0.0, maximum=1.0),
            "SRG": _bounded_metric([
                item.get("SRG") if item.get("evidence_complete") else None for item in subset],
                minimum=-1.0, maximum=1.0),
            "NBD": _bounded_metric([
                item.get("NBD") if item.get("evidence_complete") else None for item in subset],
                minimum=0.0, maximum=1.0),
        }
    return {
        "schema_version": "l2d.realsim.paired_semantic_metrics.v0.1",
        "profile": PROFILE,
        "model": document.get("model_name"),
        "configuration_id": document.get("configuration_id"),
        "source": source_label,
        "applicable_pcus": len(applicable),
        "excluded_gate_pcus": len(details) - len(applicable),
        "complete_applicable_pairs": sum(item.get("evidence_complete") is True for item in applicable),
        "metrics": metrics,
        "by_task_group": by_group,
        "pcu_details": details,
    }
