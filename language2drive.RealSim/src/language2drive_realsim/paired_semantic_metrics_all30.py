"""All-30-PCU continuous paired language-response metrics.

The aggregation order is part of the metric definition: first average rewrite
pairs within each PCU, then average PCUs.  A single-seed document contributes
one observed rewrite pair per PCU.  The scorer keeps the rewrite dimension
explicit so future multi-rewrite evidence cannot accidentally become a
micro-average over executions.

Unlike the earlier v0.1 profile, this profile also scores the three frozen
gate-contract PCUs.  A gate instruction is complete iff its target is the
first commitment gate crossed within the decision window.  This uses the
already frozen contract and geometry and does not invent a lane/region target.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path
from typing import Iterable

from .events import derive_events, point_in_polygon, point_to_polygon_distance
from .physical_only_metrics import validated_trace_rows


PROFILE = "realsim-paired-semantic-v0.2-all30-20260923"
METRICS = ("SCS", "PCS", "SRG", "NBD")
SPEED_KINDS = {"speed_band", "triggered_speed_band"}
LANE_KINDS = {"lane", "scoped_lane"}
GATE_KINDS = {"gate"}
SUPPORTED_KINDS = SPEED_KINDS | LANE_KINDS | GATE_KINDS
RESAMPLE_INTERVAL_S = 0.1
PROGRESS_NORMALIZER_M = 8.0
EPSILON = 1e-9


def _number(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid " + label)
    return float(value)


def task_group(pcu: dict) -> str:
    kinds = {pcu["arms"][arm]["semantic_contract"]["kind"] for arm in ("R0", "R1")}
    if kinds <= SPEED_KINDS:
        return "speed"
    if kinds <= LANE_KINDS:
        return "lane"
    if kinds <= GATE_KINDS:
        return "route"
    if not kinds <= SUPPORTED_KINDS:
        raise ValueError("unsupported semantic contract kind")
    raise ValueError("PCU mixes incompatible semantic target kinds")


def _horizon(pcu: dict, arm_id: str) -> tuple[float, float]:
    contract = pcu["arms"][arm_id]["semantic_contract"]
    return (
        _number(pcu["decision_window"]["start_s"], "decision start"),
        min(
            _number(pcu["decision_window"]["deadline_s"], "decision deadline"),
            _number(contract["deadline_s"], "contract deadline"),
        ),
    )


def _geometry(pcu: dict, collection: str, identifier: str):
    try:
        return pcu["private_annotations"][collection][identifier]
    except KeyError as exc:
        raise ValueError(f"missing frozen {collection} geometry: {identifier}") from exc


def _scope_geometry(pcu: dict, arm_id: str):
    contract = pcu["arms"][arm_id]["semantic_contract"]
    scope_id = contract.get("scope")
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError("persistent semantic contract lacks a scope")
    return _geometry(pcu, "scopes", scope_id)


def _interpolate(rows: list[dict], time_s: float) -> dict:
    times = [float(row["time_s"]) for row in rows]
    if time_s < times[0] - EPSILON or time_s > times[-1] + EPSILON:
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
        return _number(first, field) + fraction * (
            _number(second, field) - _number(first, field)
        )

    first_position, second_position = left["position"], right["position"]
    if len(first_position) < 2 or len(second_position) < 2:
        raise ValueError("trace position lacks xy coordinates")
    return {
        "time_s": time_s,
        "position": [
            _number(first_position[0], "x") + fraction * (
                _number(second_position[0], "x") - _number(first_position[0], "x")
            ),
            _number(first_position[1], "y") + fraction * (
                _number(second_position[1], "y") - _number(first_position[1], "y")
            ),
        ],
        "speed_mps": linear("speed_mps"),
        "progress_m": linear("progress_m"),
    }


def _grid(start: float, end: float) -> list[float]:
    if end < start:
        return []
    count = int(math.floor((end - start) / RESAMPLE_INTERVAL_S + EPSILON))
    values = [start + index * RESAMPLE_INTERVAL_S for index in range(count + 1)]
    if not values or end - values[-1] > 1e-8:
        values.append(end)
    return values


def _point_to_segment_distance(point, endpoints) -> float:
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("gate must contain exactly two endpoints")
    px, py = float(point[0]), float(point[1])
    ax, ay = float(endpoints[0][0]), float(endpoints[0][1])
    bx, by = float(endpoints[1][0]), float(endpoints[1][1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    fraction = 0.0 if denominator == 0.0 else (
        (px - ax) * dx + (py - ay) * dy
    ) / denominator
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))


def _gate_crossings(pcu: dict, rows: list[dict]) -> dict[str, float]:
    gate_ids = set(pcu["decision_window"]["commitment_gates"])
    result = {}
    for event in derive_events(rows, pcu["private_annotations"]):
        if event.kind == "CrossGate" and event.subject in gate_ids:
            result.setdefault(event.subject, float(event.time_s))
    return result


def _first_commitment(pcu: dict, rows: list[dict], start: float, end: float):
    gate_ids = set(pcu["decision_window"]["commitment_gates"])
    events = [event for event in derive_events(rows, pcu["private_annotations"])
              if event.kind == "CrossGate" and event.subject in gate_ids
              and start - EPSILON <= event.time_s <= end + EPSILON]
    return min(events, key=lambda event: event.time_s) if events else None


def _persistent_target_distance(pcu: dict, target_arm: str, sample: dict) -> float:
    contract = pcu["arms"][target_arm]["semantic_contract"]
    kind = contract["kind"]
    if kind in SPEED_KINDS:
        speed = _number(sample["speed_mps"], "speed")
        lower = _number(contract["speed_min_mps"], "speed lower bound")
        upper = _number(contract["speed_max_mps"], "speed upper bound")
        if lower > upper:
            raise ValueError("reversed speed band")
        return max(lower - speed, 0.0, speed - upper)
    if kind in LANE_KINDS:
        return point_to_polygon_distance(
            sample["position"], _geometry(pcu, "lanes", contract["target_lane"])
        )
    raise ValueError("persistent target distance requested for event contract")


def _progress_factor(rows: list[dict], end: float) -> tuple[float, float]:
    initial = _interpolate(rows, 0.0)["progress_m"]
    terminal = _interpolate(rows, end)["progress_m"]
    forward = max(0.0, terminal - initial)
    return forward, min(1.0, forward / PROGRESS_NORMALIZER_M)


def score_candidate(pcu: dict, target_arm: str, rows: list[dict]) -> dict:
    """Compute C_j(tau) for either candidate j on any trajectory tau."""
    group = task_group(pcu)
    start, end = _horizon(pcu, target_arm)
    if rows[0]["time_s"] > EPSILON or rows[-1]["time_s"] + EPSILON < end:
        raise ValueError("trace lacks the fixed metric horizon")
    forward, progress_factor = _progress_factor(rows, end)
    contract = pcu["arms"][target_arm]["semantic_contract"]
    if group == "route":
        first = _first_commitment(pcu, rows, start, end)
        semantic_fraction = float(bool(first and first.subject == contract["target_gate"]))
        active_samples = 1
        satisfied_samples = int(semantic_fraction)
        first_gate = first.subject if first else None
        first_gate_time = first.time_s if first else None
        scoring_mode = "first_commitment_gate_by_deadline"
    else:
        scope = _scope_geometry(pcu, target_arm)
        active = []
        for time_s in _grid(start, end):
            sample = _interpolate(rows, time_s)
            if point_in_polygon(sample["position"], scope):
                active.append(sample)
        satisfied_samples = sum(
            _persistent_target_distance(pcu, target_arm, sample) <= EPSILON
            for sample in active
        )
        active_samples = len(active)
        semantic_fraction = satisfied_samples / active_samples if active_samples else 0.0
        first_gate = first_gate_time = None
        scoring_mode = "target_satisfaction_fraction_in_frozen_scope"
    compliance = semantic_fraction * progress_factor
    return {
        "target_arm": target_arm,
        "start_s": start,
        "end_s": end,
        "scoring_mode": scoring_mode,
        "active_samples": active_samples,
        "satisfied_samples": satisfied_samples,
        "satisfaction_ratio": semantic_fraction,
        "first_commitment_gate": first_gate,
        "first_commitment_time_s": first_gate_time,
        "terminal_forward_distance_m": forward,
        "progress_factor": progress_factor,
        "compliance": compliance,
    }


def _preference(distance0: float, distance1: float) -> float:
    denominator = distance0 + distance1
    return 0.0 if denominator <= 1e-12 else (distance1 - distance0) / denominator


def _paired_preference_samples(
    pcu: dict, rows_r0: list[dict], rows_r1: list[dict], start: float, end: float
) -> dict[str, list[float]]:
    """Use exactly the same effective timestamps for the two preferences."""
    group = task_group(pcu)
    crossings = {
        "R0": _gate_crossings(pcu, rows_r0) if group == "route" else {},
        "R1": _gate_crossings(pcu, rows_r1) if group == "route" else {},
    }
    values = {"R0": [], "R1": []}
    for time_s in _grid(start, end):
        samples = {
            "R0": _interpolate(rows_r0, time_s),
            "R1": _interpolate(rows_r1, time_s),
        }
        if group != "route" and not (
            point_in_polygon(samples["R0"]["position"], _scope_geometry(pcu, "R0"))
            and point_in_polygon(samples["R1"]["position"], _scope_geometry(pcu, "R1"))
        ):
            continue
        for trajectory_arm, sample in samples.items():
            distances = []
            for target_arm in ("R0", "R1"):
                if group != "route":
                    distance = _persistent_target_distance(pcu, target_arm, sample)
                else:
                    gate = pcu["arms"][target_arm]["semantic_contract"]["target_gate"]
                    crossed = crossings[trajectory_arm].get(gate)
                    distance = (
                        0.0 if crossed is not None and crossed <= time_s + EPSILON
                        else _point_to_segment_distance(
                            sample["position"], _geometry(pcu, "gates", gate)
                        )
                    )
                distances.append(distance)
            values[trajectory_arm].append(_preference(distances[0], distances[1]))
    return values


def score_pair(pcu: dict, rows_r0: list[dict], rows_r1: list[dict]) -> dict:
    """Score one observed rewrite pair for one PCU."""
    group = task_group(pcu)
    trajectories = {"R0": rows_r0, "R1": rows_r1}
    cross = {
        trajectory_arm: {
            target_arm: score_candidate(pcu, target_arm, rows)
            for target_arm in ("R0", "R1")
        }
        for trajectory_arm, rows in trajectories.items()
    }
    c0 = cross["R0"]["R0"]["compliance"]
    c1 = cross["R1"]["R1"]["compliance"]
    scs = (c0 + c1) / 2.0
    pcs = min(c0, c1)
    imbalance = abs(c0 - c1) / 2.0
    residual = scs - pcs - imbalance
    if abs(residual) > 1e-12:
        raise AssertionError("SCS/PCS imbalance identity failed at pair level")

    starts_ends = [_horizon(pcu, arm) for arm in ("R0", "R1")]
    start = max(value[0] for value in starts_ends)
    end = min(value[1] for value in starts_ends)
    preference_values = _paired_preference_samples(pcu, rows_r0, rows_r1, start, end)
    preferences = {
        arm: sum(values) / len(values) if values else 0.0
        for arm, values in preference_values.items()
    }
    srg = (preferences["R0"] - preferences["R1"]) / 2.0

    position0, position1 = rows_r0[0]["position"], rows_r1[0]["position"]
    if math.hypot(float(position0[0]) - float(position1[0]),
                  float(position0[1]) - float(position1[1])) > 1e-6:
        raise ValueError("paired trajectories do not share the same scene origin")
    origin = (float(position0[0]), float(position0[1]))
    numerator = denominator = 0.0
    nbd_samples = _grid(start, end)
    for time_s in nbd_samples:
        sample0 = _interpolate(rows_r0, time_s)
        sample1 = _interpolate(rows_r1, time_s)
        xy0 = (sample0["position"][0] - origin[0], sample0["position"][1] - origin[1])
        xy1 = (sample1["position"][0] - origin[0], sample1["position"][1] - origin[1])
        numerator += math.hypot(xy0[0] - xy1[0], xy0[1] - xy1[1])
        denominator += math.hypot(*xy0) + math.hypot(*xy1)
    nbd = 0.0 if denominator <= 1e-12 else numerator / denominator
    nbd = min(1.0, max(0.0, nbd))
    return {
        "task_group": group,
        "cross_compliance": {
            "C0_tau0": cross["R0"]["R0"]["compliance"],
            "C1_tau0": cross["R0"]["R1"]["compliance"],
            "C0_tau1": cross["R1"]["R0"]["compliance"],
            "C1_tau1": cross["R1"]["R1"]["compliance"],
        },
        "candidate_details": cross,
        "SCS": scs,
        "PCS": pcs,
        "imbalance_half_abs": imbalance,
        "identity_residual": residual,
        "preference_R0_trajectory": preferences["R0"],
        "preference_R1_trajectory": preferences["R1"],
        "SRG": srg,
        "NBD": nbd,
        "common_effective_samples": len(preference_values["R0"]),
        "nbd_samples": len(nbd_samples),
        "comparison_window_s": [start, end],
        "scene_origin_xy": list(origin),
    }


def _interval(lower: float, upper: float, exact: float | None = None) -> dict:
    if lower > upper + EPSILON:
        raise ValueError("reversed metric interval")
    return {"lower": lower, "upper": upper, "exact": exact}


def _pair_intervals(c0: float | None, c1: float | None) -> dict:
    values = [c0, c1]
    known = [value for value in values if value is not None]
    scs_lower = sum(known) / 2.0
    scs_upper = (sum(known) + 2 - len(known)) / 2.0
    scs_exact = scs_lower if len(known) == 2 else None
    if len(known) == 2:
        pcs_lower = pcs_upper = min(known)
        pcs_exact = pcs_lower
    elif len(known) == 1:
        pcs_lower, pcs_upper, pcs_exact = 0.0, known[0], None
    else:
        pcs_lower, pcs_upper, pcs_exact = 0.0, 1.0, None
    return {
        "SCS": _interval(scs_lower, scs_upper, scs_exact),
        "PCS": _interval(pcs_lower, pcs_upper, pcs_exact),
    }


def _macro_summary(pcu_details: list[dict], metric: str) -> dict:
    minimum, maximum = (-1.0, 1.0) if metric == "SRG" else (0.0, 1.0)
    pcu_intervals = []
    exact_values = []
    rewrite_count = 0
    for pcu in pcu_details:
        rewrites = pcu["rewrites"]
        if not rewrites:
            pcu_intervals.append(_interval(minimum, maximum))
            continue
        rewrite_count += len(rewrites)
        intervals = []
        for rewrite in rewrites:
            value = rewrite["intervals"].get(metric)
            if value is None:
                exact = rewrite.get(metric) if rewrite.get("evidence_complete") else None
                value = _interval(exact if exact is not None else minimum,
                                  exact if exact is not None else maximum, exact)
            intervals.append(value)
        lower = sum(value["lower"] for value in intervals) / len(intervals)
        upper = sum(value["upper"] for value in intervals) / len(intervals)
        exact = lower if all(value["exact"] is not None for value in intervals) else None
        pcu_intervals.append(_interval(lower, upper, exact))
        if exact is not None:
            exact_values.append(exact)
    denominator = len(pcu_details)
    lower = sum(value["lower"] for value in pcu_intervals) / denominator
    upper = sum(value["upper"] for value in pcu_intervals) / denominator
    all_exact = all(value["exact"] is not None for value in pcu_intervals)
    exact = lower if all_exact else None
    return {
        "percent": 100.0 * exact if exact is not None else None,
        "known_only_percent": (
            100.0 * sum(exact_values) / len(exact_values) if exact_values else None
        ),
        "lower_bound_percent": 100.0 * lower,
        "upper_bound_percent": 100.0 * upper,
        "known": len(exact_values),
        "unknown": denominator - len(exact_values),
        "denominator": denominator,
        "observed_rewrite_pairs": rewrite_count,
        "aggregation": "equal_rewrites_within_each_PCU_then_equal_PCUs",
    }


def _identity_summary(details: list[dict]) -> dict:
    complete = [rewrite for pcu in details for rewrite in pcu["rewrites"]
                if rewrite.get("evidence_complete")]
    all_complete = len(complete) == sum(len(pcu["rewrites"]) for pcu in details)
    if not all_complete or len(details) != 30:
        return {"applicable": False, "reason": "requires_complete_two_arm_evidence_for_all_PCUs"}
    pcu_lhs, pcu_rhs = [], []
    for pcu in details:
        rewrites = pcu["rewrites"]
        pcu_lhs.append(sum(item["SCS"] - item["PCS"] for item in rewrites) / len(rewrites))
        pcu_rhs.append(sum(item["imbalance_half_abs"] for item in rewrites) / len(rewrites))
    lhs = 100.0 * sum(pcu_lhs) / len(pcu_lhs)
    rhs = 100.0 * sum(pcu_rhs) / len(pcu_rhs)
    return {
        "applicable": True,
        "SCS_minus_PCS_percent": lhs,
        "fifty_mean_absolute_arm_gap_percent": rhs,
        "residual_percent": lhs - rhs,
        "passed": abs(lhs - rhs) <= 1e-10,
    }


def score_document(document: dict, pcus: list[dict], source_label: str = "") -> dict:
    """Score a fixed 60-slot, one-seed document over every frozen PCU."""
    pcu_by_id = {pcu["pcu_id"]: pcu for pcu in pcus}
    if len(pcus) != 30 or len(pcu_by_id) != 30:
        raise ValueError("expected 30 unique frozen PCUs")
    rows = document.get("results")
    if not isinstance(rows, list) or len(rows) != 60 or document.get("expected_runs") != 60:
        raise ValueError("expected a fixed 60-slot experiment document")
    seed = document.get("seed")
    if type(seed) is not int:
        raise ValueError("experiment document lacks integer seed")
    slots = {}
    for row in rows:
        key = (row.get("pcu_id"), row.get("arm_id"))
        if key[0] not in pcu_by_id or key[1] not in {"R0", "R1"} or key in slots:
            raise ValueError("invalid, unknown, or duplicate experiment slot")
        slots[key] = row
    expected = {(pcu_id, arm) for pcu_id in pcu_by_id for arm in ("R0", "R1")}
    if set(slots) != expected:
        raise ValueError("experiment slots differ from frozen cohort")

    details = []
    rewrite_id = f"seed-{seed}"
    for pcu in pcus:
        group = task_group(pcu)
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
            else:
                arm_rows[arm] = validated_trace_rows(Path(trace))
        diagonal = {
            arm: score_candidate(pcu, arm, trace_rows)["compliance"]
            for arm, trace_rows in arm_rows.items()
        }
        intervals = _pair_intervals(diagonal.get("R0"), diagonal.get("R1"))
        if missing:
            rewrite = {
                "rewrite_id": rewrite_id,
                "evidence_complete": False,
                "missing_arms": missing,
                "diagonal_compliance": diagonal,
                "intervals": intervals,
            }
        else:
            rewrite = score_pair(pcu, arm_rows["R0"], arm_rows["R1"])
            rewrite.update(
                rewrite_id=rewrite_id,
                evidence_complete=True,
                missing_arms=[],
                intervals={
                    "SCS": _interval(rewrite["SCS"], rewrite["SCS"], rewrite["SCS"]),
                    "PCS": _interval(rewrite["PCS"], rewrite["PCS"], rewrite["PCS"]),
                },
            )
        details.append({
            "pcu_id": pcu["pcu_id"],
            "task_family": pcu["task_family"],
            "task_group": group,
            "rewrite_count": 1,
            "rewrites": [rewrite],
        })

    metrics = {name: _macro_summary(details, name) for name in METRICS}
    by_group = {}
    for group in ("speed", "lane", "route"):
        subset = [pcu for pcu in details if pcu["task_group"] == group]
        by_group[group] = {
            "pcus": len(subset),
            **{name: _macro_summary(subset, name) for name in METRICS},
        }
    identity = _identity_summary(details)
    if identity.get("applicable") and (
        abs(metrics["SCS"]["percent"] - metrics["PCS"]["percent"]
            - identity["fifty_mean_absolute_arm_gap_percent"]) > 1e-10
    ):
        raise AssertionError("reported SCS/PCS identity failed")
    return {
        "schema_version": "l2d.realsim.paired_semantic_metrics.v0.2",
        "profile": PROFILE,
        "model": document.get("model_name"),
        "configuration_id": document.get("configuration_id"),
        "source": source_label,
        "seed": seed,
        "pcu_coverage": len(details),
        "pcu_denominator": 30,
        "observed_rewrites_per_pcu": 1,
        "rewrite_definition": "one paired execution realization identified by experiment seed",
        "complete_pairs": sum(
            pcu["rewrites"][0]["evidence_complete"] for pcu in details
        ),
        "metrics": metrics,
        "identity_audit": identity,
        "by_task_group": by_group,
        "pcu_details": details,
    }
