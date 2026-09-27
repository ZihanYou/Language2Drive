"""Independent component judgements on unchanged, normalized real trajectories.

Contract predicates reuse the frozen semantic primitives without their outer
safety/progress short circuit. Relaxed semantics is evaluated directly, never
by changing safety flags or interpreting a legacy failure code. This module
does not attest ASL/model identity or supply missing traffic-rule observations.
"""

import math

from .events import derive_events, validate_trace
from .single_seed_metrics import COMPONENTS
from .verifier import (
    _between, _first_event, _gate_verifier, _lane_verifier,
    _scoped_lane_verifier, _speed_verifier, _stable_interval, _unsafe_result,
)

SUPPORTED_KINDS = {"gate", "lane", "scoped_lane", "speed_band", "triggered_speed_band"}


def _number(value, name, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError("invalid " + name)
    return float(value)


def _contract_result(contract, opposing, rows, events, window, layer, epsilon):
    kind = contract["kind"]
    if kind == "gate":
        return _gate_verifier(contract, rows, events, window, layer)
    if kind == "lane":
        return _lane_verifier(contract, opposing, rows)
    if kind == "scoped_lane":
        return _scoped_lane_verifier(contract, opposing, rows, events, window)
    return _speed_verifier(contract, rows, epsilon, events, window)


def _scope_facts(contract, rows, events, start):
    entry_id, exit_id = contract.get("entry_gate"), contract.get("exit_gate")
    if not entry_id or not exit_id or entry_id == exit_id:
        raise ValueError("distinct scope gates required")
    entry = _first_event(events, "CrossGate", [entry_id], start)
    exit_event = _first_event(events, "CrossGate", [exit_id], start)
    ordered = bool(entry and exit_event and exit_event.time_s > entry.time_s)
    if not ordered:
        return dict(complete=False, samples=[], completion_time=None, duration=0.0,
                    trigger_satisfied=contract["kind"] != "triggered_speed_band")
    traversal = [r for r in rows if entry.time_s <= r["time_s"] < exit_event.time_s]
    checked = [r for r in rows if entry.time_s <= r["time_s"] <= exit_event.time_s
               and contract["scope"] in r.get("scopes", [])]
    stayed = bool(traversal) and all(contract["scope"] in r.get("scopes", []) for r in traversal)
    triggered = (contract["kind"] != "triggered_speed_band" or any(
        start <= r["time_s"] <= entry.time_s and contract["trigger_id"] in r.get("triggers", [])
        for r in rows))
    return dict(complete=stayed, samples=checked, completion_time=exit_event.time_s,
                duration=exit_event.time_s - entry.time_s, trigger_satisfied=triggered)


def _relaxed_components(contract, opposing, rows, events, window, epsilon):
    """Return semantic success, deadline, stability, and their factual basis."""
    start = float(window["start_s"])
    deadline = min(float(window["deadline_s"]), float(contract["deadline_s"]))
    kind = contract["kind"]
    if kind == "gate":
        first = _first_event(events, "CrossGate", window["commitment_gates"], start)
        semantic = bool(first and first.subject == contract["target_gate"])
        # Gate contracts have no additional dwell requirement: commitment is
        # the frozen first gate event, not an invented post-hoc holding period.
        return semantic, bool(first and first.time_s <= deadline), bool(first), dict(
            definition="first_commitment_gate_without_deadline",
            completion_time_s=first.time_s if first else None)
    if kind == "lane":
        interval = _stable_interval(rows, contract["target_lane"], contract["scope"],
                                    float(contract["stable_duration_s"]))
        semantic = any(r["time_s"] >= start and contract["scope"] in r.get("scopes", [])
                       and contract["target_lane"] in r.get("lane_ids", []) for r in rows)
        return semantic, bool(interval and interval[1] <= deadline), bool(interval), dict(
            definition="target_lane_occupied_in_scope_without_dwell_or_deadline",
            stable_interval_s=list(interval) if interval else None)
    facts = _scope_facts(contract, rows, events, start)
    checked = facts["samples"]
    complete = facts["complete"] and facts["trigger_satisfied"] and bool(checked)
    on_time = facts["completion_time"] is not None and facts["completion_time"] <= deadline
    if kind == "scoped_lane":
        semantic = complete and all(
            contract["target_lane"] in r.get("lane_ids", [])
            and opposing["target_lane"] not in r.get("lane_ids", []) for r in checked)
        stable = facts["complete"] and len(checked) >= 2 and facts["duration"] >= float(contract["stable_duration_s"])
    else:
        missing = any(r.get("speed_mps") is None for r in checked)
        semantic = None if complete and missing else (complete and all(
            r.get("speed_mps") is not None and _between(
                float(r["speed_mps"]), float(contract["speed_min_mps"]),
                float(contract["speed_max_mps"]), epsilon) for r in checked))
        stable = facts["complete"] and len(checked) >= 2
    return semantic, on_time, stable, dict(
        definition="complete_ordered_scope_semantics_without_deadline_or_minimum_dwell",
        completion_time_s=facts["completion_time"], scope_duration_s=facts["duration"],
        scope_complete=facts["complete"], trigger_satisfied=facts["trigger_satisfied"],
        evaluated_samples=len(checked))


def score_trace_components(pcu, arm_id, trace, *, annotations=None,
                           threshold_profile=None, legality_passed=None,
                           required_end_s=None):
    """Score supported frozen contracts; unknown legality remains null.

    `legality_passed` is an optional independently established upstream result,
    not a claim this function validates. Do not supply True from absent metrics
    or confuse upstream's heuristic wrong_lane with complete traffic legality.
    Missing progress/legality never defaults to success. Infrastructure-invalid
    traces yield no trustworthy component credits.
    """
    if legality_passed is not None and type(legality_passed) is not bool:
        raise ValueError("legality_passed must be boolean or null")
    thresholds = threshold_profile or {}
    maximum_gap = _number(thresholds.get("maximum_sample_gap_s", 0.25), "sample gap", True)
    epsilon = _number(thresholds.get("speed_epsilon_mps", 1e-6), "speed epsilon")
    rows = validate_trace(trace)
    if len(rows) < 2:
        raise ValueError("complete trajectory requires at least two samples")
    for row in rows:
        for field in ("collision", "offroad", "render_valid"):
            if type(row[field]) is not bool:
                raise ValueError("trace requires actual boolean " + field)
        if "infrastructure_failure" in row and type(row["infrastructure_failure"]) is not bool:
            raise ValueError("invalid infrastructure flag")
    if any(b["time_s"] - a["time_s"] > maximum_gap for a, b in zip(rows, rows[1:])):
        raise ValueError("trace sample gap")
    window = pcu["decision_window"]
    arm = pcu["arms"][arm_id]
    own, opposite = arm["semantic_contract"], arm["opposing_contract"]
    if own["kind"] not in SUPPORTED_KINDS or opposite["kind"] not in SUPPORTED_KINDS:
        raise ValueError("unsupported contract kind in component scorer")
    contract_horizon = max(_number(value, "contract deadline") for value in (
        window["deadline_s"], own["deadline_s"], opposite["deadline_s"]))
    horizon = _number(required_end_s if required_end_s is not None else contract_horizon,
                      "required trace end")
    if horizon < contract_horizon:
        raise ValueError("required trace end cannot shorten contract horizon")
    if rows[0]["time_s"] != 0 or rows[-1]["time_s"] < horizon:
        raise ValueError("trace lacks complete required horizon")
    events = derive_events(rows, annotations if annotations is not None else pcu.get("private_annotations", {}))
    own_result = _contract_result(own, opposite, rows, events, window, pcu["primary_layer"], epsilon)
    opposing_result = _contract_result(opposite, own, rows, events, window, pcu["primary_layer"], epsilon)
    semantic, deadline, stability, semantic_basis = _relaxed_components(own, opposite, rows, events, window, epsilon)
    own_passed = None if own_result.failure == "INVALID_SIM" else own_result.passed
    opposite_passed = None if opposing_result.failure == "INVALID_SIM" else opposing_result.passed
    if own_passed is True:
        semantic = True  # Exact predicate implies relaxed semantic success.
    minimum = _number(pcu["hard_constraints"]["minimum_progress_m"], "minimum progress")
    values = [r.get("progress_m") for r in rows]
    progress = max(values) >= minimum if all(v is not None for v in values) else None
    # Inspect every row: a later invalid render/infrastructure failure must not
    # be hidden by the legacy verifier's earlier collision short circuit.
    safety_results = [_unsafe_result([row], float(window["start_s"])) for row in rows]
    invalid = any(r is not None and r.failure == "INVALID_SIM" for r in safety_results)
    physical_safe = not any(r is not None for r in safety_results)
    safe_legal = False if not physical_safe or legality_passed is False else legality_passed
    components = dict(own_contract=own_passed, opposing_contract=opposite_passed,
                      semantic_success=semantic, safe_legal=safe_legal, progress=progress,
                      deadline=deadline, stability=stability)
    if invalid:
        components = {name: None for name in COMPONENTS}
    return dict(
        schema_version="l2d.realsim.trace_components.v0.1",
        evaluation_scope="trajectory_components_without_model_or_legality_authority",
        formal_model_evaluation_complete=False, trace_usable=not invalid,
        components=components,
        diagnostics=dict(own_contract_result=own_result.to_dict(),
                         opposing_contract_result=opposing_result.to_dict(),
                         semantic_basis=semantic_basis,
                         physical_safe=None if invalid else physical_safe,
                         legality_passed=legality_passed,
                         safety_events=[r.to_dict() for r in safety_results if r is not None],
                         required_end_s=horizon, sample_count=len(rows),
                         maximum_progress_m=max(v for v in values if v is not None)
                         if any(v is not None for v in values) else None),
    )
