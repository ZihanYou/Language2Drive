"""Versioned supplemental FDQ metrics derived from authoritative rollouts.

The canonical Bench2Drive ``fdq_result.json`` remains unchanged.  This module
materializes a separate ``fdq_metrics.json`` so SI/GPE/PSM/MDQ/RLS can evolve
without silently redefining the upstream RC/DS/IP contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "language2drive.fdq_metrics.v1"
REFERENCE_SCHEMA_VERSION = "language2drive.fdq_reference_profiles.v1"
DEFAULT_REFERENCE_PROFILES = (
    Path(__file__).resolve().parents[2]
    / "build/suite/fdq_reference_profiles.v1.json"
)

POLICY: dict[str, Any] = {
    "schema_version": "language2drive.fdq_metric_policy.v1",
    "RC": {
        "source": "fdq_result.json.RC",
        "mutation": "none",
    },
    "SI": {
        "formula": "exp(-(3*N_pedestrian+2*N_vehicle+1*N_layout))",
        "collision_authority": "Bench2Drive ego-involved collision infraction lists",
        "fault_scope": (
            "Bench2Drive does not adjudicate legal fault; ego-involved collisions "
            "are a conservative at-fault proxy"
        ),
    },
    "GPE": {
        "formula": "r*exp(-beta*max(0,T/T_ref(r)-1))",
        "beta": 0.5,
        "progress_source": "fdq_result.json.RC/100",
        "reference": (
            "same-route versioned privileged-reference progress/time curve "
            "using Bench2Drive ROUTE_COMPLETION criterion progress"
        ),
    },
    "PSM": {
        "formula": "time_average(clip(TTC_min/tau_safe,0,1))",
        "tau_safe_seconds": 3.0,
        "relevant_actor_types": ["vehicle.*", "walker.pedestrian.*"],
        "observation_radius_m": 75.0,
        "collision_geometry": "constant-velocity ego-frame swept oriented-box entry time",
        "box_clearance_m": 0.2,
    },
    "MDQ": {
        "formula": "exp(-sum_q(w_q*E_q))",
        "smoothing": "centered_mean_7_samples_before_finite_difference",
        "angular_velocity_unit": "CARLA degrees/s converted to radians/s",
        "weights": {
            "longitudinal_acceleration": 1.0 / 6.0,
            "lateral_acceleration": 1.0 / 6.0,
            "yaw_rate": 1.0 / 6.0,
            "yaw_acceleration": 1.0 / 6.0,
            "longitudinal_jerk": 1.0 / 6.0,
            "jerk_magnitude": 1.0 / 6.0,
        },
        "expert_ranges": {
            "longitudinal_acceleration": [-4.05, 2.40],
            "lateral_acceleration": [-4.89, 4.89],
            "yaw_rate": [-0.95, 0.95],
            "yaw_acceleration": [-1.93, 1.93],
            "longitudinal_jerk": [-4.13, 4.13],
            "jerk_magnitude": [0.0, 8.37],
        },
        "normalization_scale": "maximum absolute expert-range endpoint",
    },
    "RLS": {
        "formula": "mean(u_j*exp(-t_recovery_j/tau_recovery))",
        "tau_recovery_seconds": 5.0,
        "opportunities": {
            "temporary_blockage": {
                "entry_speed_mps": 0.1,
                "entry_duration_seconds": 2.0,
                "recovered_speed_mps": 1.0,
            },
            "predictive_risk_braking": {
                "brake_min": 0.5,
                "ttc_entry_seconds": 3.0,
                "recovered_ttc_seconds": 3.0,
                "recovered_speed_mps": 0.5,
            },
            "lane_deviation": {"recovered_when_map_legal": True},
        },
        "no_opportunity": "not_applicable",
        "terminal_liveness_failures": [
            "Failed - Agent got blocked",
            "Failed - Agent timed out",
        ],
    },
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


POLICY_SHA256 = _stable_hash(POLICY)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _metric(value: float | None, *, status: str = "available", **details: Any) -> dict[str, Any]:
    if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError("FDQ supplemental metric is outside [0,1]")
    return {"status": status, "value": value, **details}


def _record(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _load_object(run_dir / "bench2drive_result.json")
    records = checkpoint.get("_checkpoint", {}).get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("FDQ checkpoint must contain exactly one record")
    fdq_result = _load_object(run_dir / "fdq_result.json")
    return records[0], fdq_result


def _score_si(record: dict[str, Any]) -> dict[str, Any]:
    infractions = record.get("infractions", {})
    if not isinstance(infractions, dict):
        return _metric(None, status="unavailable", reason="missing_infraction_lists")
    keys = {
        "pedestrian": ("collisions_pedestrian", 3.0),
        "vehicle": ("collisions_vehicle", 2.0),
        "layout": ("collisions_layout", 1.0),
    }
    counts: dict[str, int] = {}
    severity = 0.0
    for label, (key, weight) in keys.items():
        values = infractions.get(key)
        if not isinstance(values, list):
            return _metric(None, status="unavailable", reason=f"invalid_{key}")
        counts[label] = len(values)
        severity += weight * len(values)
    return _metric(
        math.exp(-severity),
        severity=severity,
        collision_counts=counts,
        fault_scope=POLICY["SI"]["fault_scope"],
    )


def _interpolate_reference_time(samples: list[dict[str, Any]], progress: float) -> float | None:
    points: list[tuple[float, float]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        r = _finite_number(sample.get("progress"))
        t = _finite_number(sample.get("time_seconds"))
        if r is None or t is None or not 0.0 <= r <= 1.0 or t < 0.0:
            return None
        points.append((r, t))
    points.sort()
    if not points or progress > points[-1][0] + 1e-9:
        return None
    if progress <= points[0][0]:
        return points[0][1]
    for (r0, t0), (r1, t1) in zip(points, points[1:]):
        if progress <= r1 + 1e-12:
            if r1 <= r0:
                return t1
            alpha = (progress - r0) / (r1 - r0)
            return t0 + alpha * (t1 - t0)
    return points[-1][1]


def _score_gpe(
    record: dict[str, Any], fdq_result: dict[str, Any], route_id: str,
    reference_path: Path,
) -> dict[str, Any]:
    rc = _finite_number(fdq_result.get("RC"))
    meta = record.get("meta")
    duration = _finite_number(
        record.get("duration_game")
        if record.get("duration_game") is not None
        else (meta.get("duration_game") if isinstance(meta, dict) else None)
    )
    if rc is None or duration is None or duration < 0.0:
        return _metric(None, status="unavailable", reason="missing_progress_or_game_time")
    progress = min(1.0, max(0.0, rc / 100.0))
    # GPE = r * exp(-beta * max(0, T / T_ref(r) - 1)).  At exactly zero
    # progress the score is identically zero, regardless of T_ref(0).  The
    # shipped reference curves start at (0, 0), so requiring a strictly
    # positive interpolated reference time here incorrectly made every
    # stationary/blocked run unavailable and caused formal workers to archive
    # and rerun otherwise complete captures indefinitely.
    if progress == 0.0:
        return _metric(
            0.0,
            progress=progress,
            duration_game_seconds=duration,
            zero_progress_identity=True,
            reference_profile_sha256=(
                hashlib.sha256(reference_path.read_bytes()).hexdigest()
                if reference_path.is_file() else None
            ),
        )
    if not reference_path.is_file():
        return _metric(
            None, status="unavailable", reason="reference_profile_missing",
            progress=progress, duration_game_seconds=duration,
            expected_reference_path=str(reference_path),
        )
    try:
        reference = _load_object(reference_path)
        if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
            raise ValueError("reference profile schema differs")
        routes = reference.get("routes")
        profile = routes.get(str(route_id)) if isinstance(routes, dict) else None
        if not isinstance(profile, dict):
            raise ValueError("route reference profile is missing")
        samples = profile.get("samples")
        if not isinstance(samples, list):
            raise ValueError("route reference samples are missing")
        reference_time = _interpolate_reference_time(samples, progress)
        if reference_time is None or reference_time <= 0.0:
            raise ValueError("reference does not cover achieved progress")
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        return _metric(
            None, status="unavailable", reason=f"invalid_reference_profile: {error}",
            progress=progress, duration_game_seconds=duration,
            reference_profile_sha256=(
                hashlib.sha256(reference_path.read_bytes()).hexdigest()
                if reference_path.is_file() else None
            ),
        )
    beta = float(POLICY["GPE"]["beta"])
    value = progress * math.exp(-beta * max(0.0, duration / reference_time - 1.0))
    return _metric(
        value, progress=progress, duration_game_seconds=duration,
        reference_time_seconds=reference_time,
        reference_profile_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        reference_policy=profile.get("reference_policy"),
    )


def _time_weights(rows: list[dict[str, Any]]) -> list[float]:
    times = [_finite_number(row.get("sim_time")) for row in rows]
    if any(value is None for value in times):
        return []
    numeric = [float(value) for value in times if value is not None]
    deltas = [b - a for a, b in zip(numeric, numeric[1:]) if b > a]
    if not deltas:
        return [1.0] * len(rows)
    ordered = sorted(deltas)
    typical = ordered[len(ordered) // 2]
    return [
        (numeric[index + 1] - numeric[index])
        if index + 1 < len(numeric) and numeric[index + 1] > numeric[index]
        else typical
        for index in range(len(numeric))
    ]


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = list(zip(values, weights))
    denominator = sum(weight for _, weight in pairs)
    if denominator <= 0.0:
        raise ValueError("non-positive trace duration")
    return sum(value * weight for value, weight in pairs) / denominator


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    alpha = position - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def _fdq_capture_supported(capabilities: dict[str, Any]) -> bool:
    observation = capabilities.get("fdq_metric_observation")
    return bool(
        isinstance(observation, dict)
        and observation.get("schema_version")
        == "language2drive.fdq_observation.v1"
        and observation.get("frame_samples") == capabilities.get("controller_source_samples")
        and observation.get("ttc_supported") is True
    )


def _score_psm(rows: list[dict[str, Any]], capabilities: dict[str, Any]) -> dict[str, Any]:
    if not _fdq_capture_supported(capabilities):
        return _metric(None, status="unavailable", reason="fdq_ttc_capture_not_supported")
    weights = _time_weights(rows)
    if not weights:
        return _metric(None, status="unavailable", reason="invalid_trace_time")
    tau = float(POLICY["PSM"]["tau_safe_seconds"])
    margins: list[float] = []
    finite_ttc: list[float] = []
    for row in rows:
        observation = row.get("fdq_observation")
        if not isinstance(observation, dict) or type(observation.get("relevant_actor_count")) is not int:
            return _metric(None, status="unavailable", reason="incomplete_fdq_observation")
        raw = observation.get("min_ttc_seconds")
        if raw is None:
            margins.append(1.0)
            continue
        ttc = _finite_number(raw)
        if ttc is None or ttc < 0.0:
            return _metric(None, status="unavailable", reason="invalid_ttc_observation")
        finite_ttc.append(ttc)
        margins.append(min(1.0, max(0.0, ttc / tau)))
    return _metric(
        _weighted_mean(margins, weights),
        ttc_5_percent_seconds=_percentile(finite_ttc, 0.05),
        collision_course_frame_count=len(finite_ttc),
        frame_count=len(rows),
    )


def _centered_mean(values: list[float], window: int = 7) -> list[float]:
    radius = window // 2
    output: list[float] = []
    for index in range(len(values)):
        segment = values[max(0, index - radius):min(len(values), index + radius + 1)]
        output.append(sum(segment) / len(segment))
    return output


def _derivative(values: list[float], times: list[float]) -> list[float]:
    if len(values) != len(times) or not values:
        return []
    output: list[float] = []
    for index in range(len(values)):
        left = max(0, index - 1)
        right = min(len(values) - 1, index + 1)
        dt = times[right] - times[left]
        output.append(0.0 if dt <= 0.0 else (values[right] - values[left]) / dt)
    return output


def _score_mdq(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times: list[float] = []
    ax: list[float] = []
    ay: list[float] = []
    yaw_rate: list[float] = []
    for row in rows:
        time_value = _finite_number(row.get("sim_time"))
        ego = row.get("ego")
        if time_value is None or not isinstance(ego, dict):
            return _metric(None, status="unavailable", reason="invalid_ego_trace")
        acceleration = ego.get("acceleration")
        forward = ego.get("forward_vector")
        right = ego.get("right_vector")
        angular = ego.get("angular_velocity")
        if not all(isinstance(value, dict) for value in (acceleration, forward, right, angular)):
            return _metric(None, status="unavailable", reason="missing_dynamics_fields")
        components = [
            _finite_number(value.get(axis))
            for value, axis in (
                (acceleration, "x"), (acceleration, "y"),
                (forward, "x"), (forward, "y"),
                (right, "x"), (right, "y"), (angular, "z"),
            )
        ]
        if any(value is None for value in components):
            return _metric(None, status="unavailable", reason="nonfinite_dynamics_fields")
        acc_x, acc_y, fwd_x, fwd_y, right_x, right_y, yaw_deg = (
            float(value) for value in components if value is not None
        )
        times.append(time_value)
        ax.append(acc_x * fwd_x + acc_y * fwd_y)
        ay.append(acc_x * right_x + acc_y * right_y)
        yaw_rate.append(math.radians(yaw_deg))
    if len(rows) < 3 or any(b <= a for a, b in zip(times, times[1:])):
        return _metric(None, status="unavailable", reason="insufficient_or_nonmonotonic_trace")
    ax = _centered_mean(ax)
    ay = _centered_mean(ay)
    yaw_rate = _centered_mean(yaw_rate)
    yaw_acceleration = _derivative(yaw_rate, times)
    longitudinal_jerk = _derivative(ax, times)
    lateral_jerk = _derivative(ay, times)
    jerk_magnitude = [math.hypot(lon, lat) for lon, lat in zip(longitudinal_jerk, lateral_jerk)]
    signals = {
        "longitudinal_acceleration": ax,
        "lateral_acceleration": ay,
        "yaw_rate": yaw_rate,
        "yaw_acceleration": yaw_acceleration,
        "longitudinal_jerk": longitudinal_jerk,
        "jerk_magnitude": jerk_magnitude,
    }
    weights = _time_weights(rows)
    exceedances: dict[str, float] = {}
    for name, values in signals.items():
        lower, upper = (float(value) for value in POLICY["MDQ"]["expert_ranges"][name])
        scale = max(abs(lower), abs(upper))
        normalized = []
        for value in values:
            raw = (lower - value) / scale if value < lower else (
                (value - upper) / scale if value > upper else 0.0
            )
            normalized.append(min(1.0, max(0.0, raw)))
        exceedances[name] = _weighted_mean(normalized, weights)
    penalty = sum(
        float(POLICY["MDQ"]["weights"][name]) * value
        for name, value in exceedances.items()
    )
    return _metric(math.exp(-penalty), normalized_exceedance=exceedances, frame_count=len(rows))


def _rls_events(rows: list[dict[str, Any]], record_status: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    active: dict[str, float | None] = {
        "temporary_blockage": None,
        "predictive_risk_braking": None,
        "lane_deviation": None,
    }
    low_speed_start: float | None = None
    blockage = POLICY["RLS"]["opportunities"]["temporary_blockage"]
    for row in rows:
        now = float(row["sim_time"])
        ego = row.get("ego", {})
        control = row.get("control", {})
        observation = row.get("fdq_observation", {})
        speed = _finite_number(ego.get("speed_mps", ego.get("speed")))
        brake = _finite_number(control.get("brake"))
        ttc = _finite_number(observation.get("min_ttc_seconds"))
        map_legal = observation.get("map_legal")
        if speed is None or brake is None:
            continue
        if speed < float(blockage["entry_speed_mps"]):
            low_speed_start = now if low_speed_start is None else low_speed_start
            if (
                active["temporary_blockage"] is None
                and now - low_speed_start >= float(blockage["entry_duration_seconds"])
            ):
                active["temporary_blockage"] = low_speed_start
        else:
            low_speed_start = None
        start = active["temporary_blockage"]
        if start is not None and speed >= float(blockage["recovered_speed_mps"]):
            events.append({"type": "temporary_blockage", "recovered": True, "recovery_seconds": now - start})
            active["temporary_blockage"] = None

        risk = POLICY["RLS"]["opportunities"]["predictive_risk_braking"]
        if (
            active["predictive_risk_braking"] is None
            and brake >= float(risk["brake_min"])
            and ttc is not None and ttc < float(risk["ttc_entry_seconds"])
        ):
            active["predictive_risk_braking"] = now
        start = active["predictive_risk_braking"]
        if (
            start is not None and speed >= float(risk["recovered_speed_mps"])
            and (ttc is None or ttc >= float(risk["recovered_ttc_seconds"]))
        ):
            events.append({"type": "predictive_risk_braking", "recovered": True, "recovery_seconds": now - start})
            active["predictive_risk_braking"] = None

        if map_legal is False and active["lane_deviation"] is None:
            active["lane_deviation"] = now
        if map_legal is True and active["lane_deviation"] is not None:
            start = float(active["lane_deviation"])
            events.append({"type": "lane_deviation", "recovered": True, "recovery_seconds": now - start})
            active["lane_deviation"] = None

    end_time = float(rows[-1]["sim_time"])
    for event_type, start in active.items():
        if start is not None:
            events.append({"type": event_type, "recovered": False, "recovery_seconds": end_time - float(start)})
    if (
        record_status in POLICY["RLS"]["terminal_liveness_failures"]
        and not any(event["recovered"] is False for event in events)
    ):
        events.append({
            "type": "terminal_liveness_failure",
            "recovered": False,
            "recovery_seconds": max(0.0, end_time - float(rows[0]["sim_time"])),
        })
    return events


def _score_rls(
    rows: list[dict[str, Any]], capabilities: dict[str, Any], record_status: str,
) -> dict[str, Any]:
    if not _fdq_capture_supported(capabilities):
        return _metric(None, status="unavailable", reason="fdq_recovery_capture_not_supported")
    events = _rls_events(rows, record_status)
    if not events:
        return _metric(None, status="not_applicable", opportunity_count=0, events=[])
    tau = float(POLICY["RLS"]["tau_recovery_seconds"])
    terms = [
        math.exp(-float(event["recovery_seconds"]) / tau)
        if event["recovered"] else 0.0
        for event in events
    ]
    return _metric(
        sum(terms) / len(terms), opportunity_count=len(events),
        recovered_count=sum(event["recovered"] is True for event in events),
        events=events,
    )


def score_fdq_metrics(
    run: dict[str, Any], run_dir: Path,
    *, reference_profiles: Path = DEFAULT_REFERENCE_PROFILES,
) -> dict[str, Any]:
    """Purely score one finalized FDQ route into the supplemental schema."""

    root = Path(run_dir)
    record, fdq_result = _record(root)
    rows = _load_trace(root / "trace.jsonl")
    capabilities = _load_object(root / "capture_capabilities.json")
    identity = {
        key: run[key] for key in ("run_id", "ability", "seed")
    }
    if any(fdq_result.get(key) != value for key, value in identity.items()):
        raise ValueError("FDQ metric identity differs from fdq_result")
    route_id = str(fdq_result.get("route_id"))
    metrics = {
        "SI": _score_si(record),
        "GPE": _score_gpe(record, fdq_result, route_id, Path(reference_profiles)),
        "PSM": _score_psm(rows, capabilities),
        "MDQ": _score_mdq(rows),
        "RLS": _score_rls(rows, capabilities, str(record.get("status", ""))),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "route_id": route_id,
        "policy_sha256": POLICY_SHA256,
        "policy": POLICY,
        "rc_unchanged": True,
        "RC": fdq_result.get("RC"),
        "metrics": metrics,
        "coverage": {
            "available": sum(metric["status"] == "available" for metric in metrics.values()),
            "requested": len(metrics),
            "unavailable": sorted(
                name for name, metric in metrics.items()
                if metric["status"] == "unavailable"
            ),
            "not_applicable": sorted(
                name for name, metric in metrics.items()
                if metric["status"] == "not_applicable"
            ),
        },
    }


def validate_fdq_metrics(value: Any, run: dict[str, Any]) -> bool:
    """Validate the supplemental artifact without trusting self-declared keys."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version", "run_id", "ability", "seed", "route_id",
        "policy_sha256", "policy", "rc_unchanged", "RC", "metrics", "coverage",
    }:
        return False
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("run_id") != run.get("run_id")
        or value.get("ability") != run.get("ability")
        or value.get("seed") != run.get("seed")
        or value.get("policy") != POLICY
        or value.get("policy_sha256") != POLICY_SHA256
        or value.get("rc_unchanged") is not True
    ):
        return False
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"SI", "GPE", "PSM", "MDQ", "RLS"}:
        return False
    for metric in metrics.values():
        if not isinstance(metric, dict) or metric.get("status") not in {
            "available", "unavailable", "not_applicable",
        }:
            return False
        number = metric.get("value")
        if metric["status"] == "available":
            if not isinstance(number, (int, float)) or isinstance(number, bool):
                return False
            if not math.isfinite(float(number)) or not 0.0 <= float(number) <= 1.0:
                return False
        elif number is not None:
            return False
    coverage = value.get("coverage")
    return bool(
        isinstance(coverage, dict)
        and coverage.get("requested") == 5
        and coverage.get("available")
        == sum(metric["status"] == "available" for metric in metrics.values())
    )
