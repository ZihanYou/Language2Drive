"""Fail-closed contract for authoritative runtime semantic captures.

This module validates a *small normalized export* produced by a future runtime
semantic authority.  It does not infer rules from images, mutate ASLs, or turn
the absence of a warning into compliance.  A complete capture is an input to a
later legality assessor; it is not itself a legal verdict.
"""

from __future__ import annotations

from .io import stable_hash
from .release_evidence import IDENTITY_FIELDS
from .single_seed_metrics import COMPONENT_SCHEMA


CAPTURE_SCHEMA = "l2d.realsim.runtime_semantic_capture.v0.2"
RECEIPT_SCHEMA = "l2d.realsim.runtime_semantic_capture_receipt.v0.2"
FORMAL_ELIGIBILITY_SCHEMA = "l2d.realsim.formal_score_slot_eligibility.v0.1"
SIGNAL_STATES = {"red", "yellow", "green", "flashing_red", "flashing_yellow", "off"}
SPEED_UNITS = {"m/s", "km/h", "mph"}
ZERO_SEMANTICS = {"literal_zero_mps", "no_posted_limit", "not_mapped", "not_applicable"}
OCCUPANCY_KINDS = {
    "within_lane", "straddling_lane_boundary", "crossing_lane_boundary",
    "outside_drivable_lane",
}
CROSSING_KINDS = {"lane_change", "merge", "split", "road_departure", "road_entry"}
SPEED_SCOPE_TYPES = {"directed_lane", "road_segment", "regulatory_zone"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in {float("inf"), float("-inf")}
    )


def _converted_mps(raw_value, unit):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not _finite_number(value):
        return None
    return value if unit == "m/s" else value / (3.6 if unit == "km/h" else 2.2369362920544)


def _identity(component_record):
    _require(isinstance(component_record, dict), "component record must be an object")
    _require(
        component_record.get("schema_version") == COMPONENT_SCHEMA,
        "unsupported component record schema",
    )
    _require(
        component_record.get("execution_status") == "completed"
        and component_record.get("evidence_valid") is True,
        "semantic capture requires a completed evidence-valid component record",
    )
    result = {field: component_record.get(field) for field in IDENTITY_FIELDS}
    _require(
        all(_text(result[field]) for field in IDENTITY_FIELDS if field != "seed")
        and type(result["seed"]) is int
        and result["seed"] >= 0,
        "invalid component identity",
    )
    return result


def _reference(reference, label, reference_verifier, gaps):
    if not (
        isinstance(reference, dict)
        and set(reference) == {"label", "path", "sha256", "size_bytes"}
        and _text(reference.get("label"))
        and _text(reference.get("path"))
        and _digest(reference.get("sha256"))
        and type(reference.get("size_bytes")) is int
        and reference["size_bytes"] >= 0
    ):
        gaps.append(label + ":content_reference_invalid")
        return
    if reference_verifier is None:
        gaps.append(label + ":content_reference_not_authenticated")
        return
    try:
        authenticated = reference_verifier(reference)
    except Exception:
        authenticated = False
    if authenticated is not True:
        gaps.append(label + ":content_reference_authentication_failed")


def _intervals_cover(intervals, start_us, end_us, label, gaps, item_validator=None):
    if not isinstance(intervals, list) or not intervals:
        gaps.append(label + ":intervals_missing")
        return
    cursor = start_us
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            gaps.append(label + ":invalid_interval")
            return
        interval_start = interval.get("start_timestamp_us")
        interval_end = interval.get("end_timestamp_us")
        if type(interval_start) is not int or type(interval_end) is not int:
            gaps.append(label + ":invalid_interval_clock")
            return
        if interval_start != cursor or interval_end <= interval_start or interval_end > end_us:
            gaps.append(label + ":interval_gap_overlap_or_range_error")
            return
        if item_validator is not None:
            item_validator(interval, index, gaps)
        cursor = interval_end
    if cursor != end_us:
        gaps.append(label + ":full_trajectory_coverage_missing")


def _lane_interval(interval, _index, gaps):
    if not _text(interval.get("directed_lane_id")):
        gaps.append("directed_lane_occupancy:lane_identity_missing")
    if interval.get("occupancy_kind") not in OCCUPANCY_KINDS:
        gaps.append("directed_lane_occupancy:occupancy_kind_missing")
    if interval.get("ego_footprint_bound") is not True:
        gaps.append("directed_lane_occupancy:ego_footprint_binding_missing")
    if not _text(interval.get("map_definition_version")):
        gaps.append("directed_lane_occupancy:map_definition_version_missing")
    lane_direction = interval.get("lane_travel_direction_rad")
    if not _finite_number(lane_direction) or not -3.141592654 <= lane_direction <= 3.141592654:
        gaps.append("directed_lane_occupancy:lane_travel_direction_missing")
    if not _text(interval.get("direction_definition_version")):
        gaps.append("directed_lane_occupancy:direction_definition_version_missing")


def _speed_interval(interval, _index, gaps):
    if not _text(interval.get("directed_lane_id")):
        gaps.append("speed_limit:directed_lane_identity_missing")
    if not isinstance(interval.get("raw_value"), str):
        gaps.append("speed_limit:raw_value_missing")
    if interval.get("source_unit") not in SPEED_UNITS:
        gaps.append("speed_limit:interval_source_unit_missing")
    if not _text(interval.get("definition_version")):
        gaps.append("speed_limit:interval_definition_version_missing")
    if not _text(interval.get("scope_definition_version")):
        gaps.append("speed_limit:interval_scope_definition_version_missing")
    if interval.get("scope_type") not in SPEED_SCOPE_TYPES or not _text(
        interval.get("scope_id")
    ):
        gaps.append("speed_limit:scope_identity_missing")
    if not _text(interval.get("vehicle_class")):
        gaps.append("speed_limit:vehicle_class_missing")
    applicability = interval.get("applicability")
    if applicability not in {"posted_limit", "no_posted_limit", "not_applicable"}:
        gaps.append("speed_limit:invalid_applicability")
    limit = interval.get("limit_mps")
    if applicability == "posted_limit":
        if not _finite_number(limit) or limit < 0:
            gaps.append("speed_limit:normalized_positive_limit_missing")
    elif limit is not None:
        gaps.append("speed_limit:non_applicable_interval_has_limit")


def _signal_state_intervals(signal, start_us, end_us, gaps):
    signal_id = signal.get("signal_id") if isinstance(signal, dict) else None
    if not _text(signal_id):
        gaps.append("traffic_signal:signal_identity_missing")
        return

    def validate(interval, _index, local_gaps):
        if interval.get("state") not in SIGNAL_STATES:
            local_gaps.append("traffic_signal:unsupported_or_unknown_dynamic_state")

    _intervals_cover(
        signal.get("state_intervals"),
        start_us,
        end_us,
        "traffic_signal_state:" + signal_id,
        gaps,
        validate,
    )


def _interval_at(intervals, timestamp_us):
    if not isinstance(intervals, list):
        return None
    for interval in intervals:
        if (
            isinstance(interval, dict)
            and type(interval.get("start_timestamp_us")) is int
            and type(interval.get("end_timestamp_us")) is int
            and interval["start_timestamp_us"] <= timestamp_us < interval["end_timestamp_us"]
        ):
            return interval
    return None


def _partitions_have_same_lane(left, right, start_us, end_us):
    try:
        boundaries = sorted(
            {start_us, end_us}
            | {item["start_timestamp_us"] for item in left}
            | {item["end_timestamp_us"] for item in left}
            | {item["start_timestamp_us"] for item in right}
            | {item["end_timestamp_us"] for item in right}
        )
    except (KeyError, TypeError):
        return False
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower == upper:
            continue
        timestamp = lower + (upper - lower) // 2
        lhs = _interval_at(left, timestamp)
        rhs = _interval_at(right, timestamp)
        if (
            lhs is None
            or rhs is None
            or lhs.get("directed_lane_id") != rhs.get("directed_lane_id")
        ):
            return False
    return True


def assess_runtime_semantic_capture(
    capture, component_record, reference_verifier=None, authority_verifier=None
):
    """Validate capture completeness and return a content-bound receipt.

    ``capture_ready_for_legality_assessment=True`` only means the two semantic
    rule families have authoritative, full-trajectory inputs.  It never means
    that either rule passed.
    """

    _require(isinstance(capture, dict), "semantic capture must be an object")
    _require(capture.get("schema_version") == CAPTURE_SCHEMA, "unsupported capture schema")
    identity = _identity(component_record)
    _require(capture.get("identity") == identity, "capture/component identity mismatch")
    _require(
        capture.get("component_record_binding") == stable_hash(component_record),
        "capture/component record binding mismatch",
    )
    gaps = []
    for field in (
        "pcu_contract_reference",
        "asl_reference",
        "semantic_export_reference",
        "semantic_definition_reference",
    ):
        _reference(capture.get(field), field, reference_verifier, gaps)

    authority = capture.get("capture_authority")
    if not (
        isinstance(authority, dict)
        and _text(authority.get("provider_id"))
        and _text(authority.get("version_id"))
        and _text(authority.get("interface_id"))
        and authority.get("clock") == "simulation_timestamp_us"
        and _text(authority.get("map_definition_version"))
        and _text(authority.get("semantic_definition_version"))
        and authority.get("export_format_version") == CAPTURE_SCHEMA
    ):
        gaps.append("capture_authority:identity_or_clock_missing")
    elif authority_verifier is None:
        gaps.append("capture_authority:not_independently_authenticated")
    else:
        try:
            authority_ok = authority_verifier(authority, capture["semantic_export_reference"])
        except Exception:
            authority_ok = False
        if authority_ok is not True:
            gaps.append("capture_authority:authentication_failed")

    authority_map_version = authority.get("map_definition_version") if isinstance(authority, dict) else None
    trajectory = capture.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory capture missing")
    start_us = trajectory.get("start_timestamp_us")
    end_us = trajectory.get("end_timestamp_us")
    _require(
        type(start_us) is int
        and type(end_us) is int
        and 0 <= start_us < end_us,
        "invalid trajectory interval",
    )
    if trajectory.get("full_pose_coverage") is not True:
        gaps.append("trajectory:full_pose_coverage_missing")
    if trajectory.get("full_ego_heading_coverage") is not True:
        gaps.append("trajectory:full_ego_heading_coverage_missing")
    if type(trajectory.get("pose_sample_count")) is not int or trajectory["pose_sample_count"] < 2:
        gaps.append("trajectory:pose_samples_missing")
    lane_intervals = trajectory.get("directed_lane_occupancy_intervals")
    _intervals_cover(
        lane_intervals,
        start_us,
        end_us,
        "directed_lane_occupancy",
        gaps,
        _lane_interval,
    )
    if isinstance(lane_intervals, list):
        for interval in lane_intervals:
            if (
                isinstance(interval, dict)
                and interval.get("map_definition_version")
                != authority_map_version
            ):
                gaps.append("directed_lane_occupancy:authority_map_version_mismatch")
    lane_crossings = trajectory.get("lane_boundary_crossing_events")
    if not isinstance(lane_crossings, list):
        gaps.append("directed_lane_occupancy:lane_crossing_inventory_missing")
    else:
        for event in lane_crossings:
            if not isinstance(event, dict):
                gaps.append("directed_lane_occupancy:invalid_lane_crossing_event")
                continue
            timestamp = event.get("timestamp_us")
            if (
                type(timestamp) is not int
                or not start_us <= timestamp < end_us
                or not all(
                    _text(event.get(field))
                    for field in (
                        "from_directed_lane_id", "to_directed_lane_id", "boundary_id",
                        "map_definition_version",
                    )
                )
                or event.get("crossing_kind") not in CROSSING_KINDS
                or event.get("ego_footprint_bound") is not True
            ):
                gaps.append("directed_lane_occupancy:invalid_lane_crossing_event")
                continue
            if event["from_directed_lane_id"] == event["to_directed_lane_id"]:
                gaps.append("directed_lane_occupancy:lane_crossing_endpoints_equal")
            if event["map_definition_version"] != authority_map_version:
                gaps.append("directed_lane_occupancy:lane_crossing_map_version_mismatch")
            intervals = lane_intervals if isinstance(lane_intervals, list) else []
            from_matches = [
                interval for interval in intervals
                if isinstance(interval, dict)
                and interval.get("end_timestamp_us") == timestamp
                and interval.get("directed_lane_id") == event["from_directed_lane_id"]
            ]
            to_matches = [
                interval for interval in intervals
                if isinstance(interval, dict)
                and interval.get("start_timestamp_us") == timestamp
                and interval.get("directed_lane_id") == event["to_directed_lane_id"]
            ]
            if len(from_matches) != 1 or len(to_matches) != 1:
                gaps.append("directed_lane_occupancy:lane_crossing_transition_mismatch")

    traffic = capture.get("traffic_signal")
    if not isinstance(traffic, dict):
        raise ValueError("traffic-signal capture missing")
    if traffic.get("full_trajectory_applicability_inventory") is not True:
        gaps.append("traffic_signal:full_trajectory_applicability_inventory_missing")
    bindings = traffic.get("stopline_control_bindings")
    if not isinstance(bindings, list):
        gaps.append("traffic_signal:stopline_control_bindings_missing")
        bindings = []
    binding_keys = set()
    signal_ids = set()
    occupied_lanes = {
        interval.get("directed_lane_id")
        for interval in lane_intervals
        if isinstance(interval, dict)
    } if isinstance(lane_intervals, list) else set()
    for binding in bindings:
        if not isinstance(binding, dict) or not all(
            _text(binding.get(field))
            for field in (
                "directed_lane_id", "stopline_id", "signal_id", "movement_id",
                "source_version", "map_definition_version",
            )
        ):
            gaps.append("traffic_signal:invalid_stopline_lane_signal_binding")
            continue
        if binding.get("applies_to_ego_movement") is not True:
            gaps.append("traffic_signal:binding_applicability_not_established")
        if binding.get("map_definition_version") != authority_map_version:
            gaps.append("traffic_signal:binding_authority_map_version_mismatch")
        key = (binding["directed_lane_id"], binding["stopline_id"], binding["signal_id"])
        if key in binding_keys:
            gaps.append("traffic_signal:duplicate_stopline_control_binding")
        binding_keys.add(key)
        signal_ids.add(binding["signal_id"])
        if binding["directed_lane_id"] not in occupied_lanes:
            gaps.append("traffic_signal:binding_lane_not_on_ego_trajectory")
    states = traffic.get("dynamic_signal_states")
    if not isinstance(states, list):
        gaps.append("traffic_signal:dynamic_state_time_series_missing")
        states = []
    state_ids = set()
    state_intervals_by_id = {}
    for signal in states:
        if isinstance(signal, dict) and _text(signal.get("signal_id")):
            if signal["signal_id"] in state_ids:
                gaps.append("traffic_signal:duplicate_signal_state_series")
            state_ids.add(signal["signal_id"])
            state_intervals_by_id[signal["signal_id"]] = signal.get("state_intervals")
        _signal_state_intervals(signal, start_us, end_us, gaps)
    if signal_ids != state_ids:
        gaps.append("traffic_signal:binding_and_dynamic_state_identity_mismatch")
    crossings = traffic.get("crossing_events")
    if not isinstance(crossings, list):
        gaps.append("traffic_signal:crossing_event_inventory_missing")
    else:
        for event in crossings:
            if not isinstance(event, dict):
                gaps.append("traffic_signal:invalid_crossing_event")
                continue
            key = (
                event.get("directed_lane_id"),
                event.get("stopline_id"),
                event.get("signal_id"),
            )
            timestamp = event.get("timestamp_us")
            state = event.get("state_at_crossing")
            if (
                key not in binding_keys
                or type(timestamp) is not int
                or not start_us <= timestamp < end_us
                or event.get("ego_footprint_bound") is not True
                or event.get("crossing_direction") != "front_to_back"
            ):
                gaps.append("traffic_signal:crossing_event_not_bound")
            if state not in SIGNAL_STATES:
                gaps.append("traffic_signal:crossing_dynamic_state_missing")
            if type(timestamp) is int:
                lane_interval = _interval_at(lane_intervals, timestamp) if isinstance(lane_intervals, list) else None
                state_interval = _interval_at(
                    state_intervals_by_id.get(event.get("signal_id"), []), timestamp
                )
                if lane_interval is None or lane_interval.get("directed_lane_id") != event.get("directed_lane_id"):
                    gaps.append("traffic_signal:crossing_lane_occupancy_mismatch")
                if state_interval is None or state_interval.get("state") != state:
                    gaps.append("traffic_signal:crossing_state_series_mismatch")

    speed = capture.get("speed_limit")
    if not isinstance(speed, dict):
        raise ValueError("speed-limit capture missing")
    if speed.get("authoritative_unit") not in SPEED_UNITS:
        gaps.append("speed_limit:authoritative_unit_missing")
    if speed.get("zero_value_semantics") not in ZERO_SEMANTICS:
        gaps.append("speed_limit:zero_value_semantics_missing")
    if not _text(speed.get("definition_version")):
        gaps.append("speed_limit:definition_version_missing")
    if not _text(speed.get("scope_definition_version")):
        gaps.append("speed_limit:scope_definition_version_missing")
    if speed.get("full_trajectory_scope") is not True:
        gaps.append("speed_limit:full_trajectory_scope_missing")
    speed_intervals = speed.get("applicable_intervals")
    _intervals_cover(
        speed_intervals,
        start_us,
        end_us,
        "speed_limit",
        gaps,
        _speed_interval,
    )
    if isinstance(speed_intervals, list):
        for interval in speed_intervals:
            if not isinstance(interval, dict):
                continue
            if interval.get("source_unit") != speed.get("authoritative_unit"):
                gaps.append("speed_limit:interval_and_authority_unit_mismatch")
            if interval.get("definition_version") != speed.get("definition_version"):
                gaps.append("speed_limit:interval_and_authority_definition_mismatch")
            if interval.get("scope_definition_version") != speed.get("scope_definition_version"):
                gaps.append("speed_limit:interval_and_authority_scope_definition_mismatch")
            if (
                interval.get("applicability") == "posted_limit"
                and interval.get("limit_mps") == 0
                and speed.get("zero_value_semantics") != "literal_zero_mps"
            ):
                gaps.append("speed_limit:zero_limit_contradicts_zero_value_semantics")
            if interval.get("applicability") == "posted_limit":
                converted = _converted_mps(
                    interval.get("raw_value"), interval.get("source_unit")
                )
                if (
                    converted is None
                    or not _finite_number(interval.get("limit_mps"))
                    or abs(converted - interval["limit_mps"]) > 1e-6
                ):
                    gaps.append("speed_limit:raw_value_conversion_mismatch")
    if (
        isinstance(lane_intervals, list)
        and isinstance(speed_intervals, list)
        and not _partitions_have_same_lane(
            lane_intervals, speed_intervals, start_us, end_us
        )
    ):
        gaps.append("speed_limit:directed_lane_occupancy_mismatch")

    ready = not gaps
    claim = capture.get("claimed_capture_ready")
    _require(claim is None or type(claim) is bool, "claimed_capture_ready must be boolean or null")
    _require(
        claim is (True if ready else None),
        "capture readiness claim is not supported by authoritative full-trajectory evidence",
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "verification_scope": "runtime_semantic_input_completeness_not_legality_verdict",
        "identity": identity,
        "component_record_binding": capture["component_record_binding"],
        "capture_binding": stable_hash(capture),
        "capture_ready_for_legality_assessment": True if ready else None,
        # Shared authority/reference gaps invalidate both rule families.  Do
        # not advertise a partially authenticated family as complete.
        "traffic_signal_evidence_complete": True if ready else None,
        "speed_limit_evidence_complete": True if ready else None,
        "directed_lane_occupancy_evidence_complete": True if ready else None,
        "legality_verdict": None,
        "evidence_gaps": sorted(set(gaps)),
        "existing_asl_mutated": False,
    }


def assess_formal_score_slot_eligibility(
    component_record,
    capture_receipt,
    legality_assessment,
    execution_source_assessment,
    pairing_assessment,
    denominator_slot,
):
    """Gate one slot without turning semantic completeness into a legal pass.

    This consumes small, already verified assessments.  Cohort-level formal
    completion still requires every fixed-denominator slot and every R0/R1 pair;
    a single eligible slot is never a model-level completion claim.
    """

    identity = _identity(component_record)
    binding = stable_hash(component_record)
    gaps = []
    if not (
        isinstance(capture_receipt, dict)
        and capture_receipt.get("schema_version") == RECEIPT_SCHEMA
        and capture_receipt.get("identity") == identity
        and capture_receipt.get("component_record_binding") == binding
        and capture_receipt.get("capture_ready_for_legality_assessment") is True
        and capture_receipt.get("traffic_signal_evidence_complete") is True
        and capture_receipt.get("speed_limit_evidence_complete") is True
        and capture_receipt.get("directed_lane_occupancy_evidence_complete") is True
        and capture_receipt.get("legality_verdict") is None
        and capture_receipt.get("evidence_gaps") == []
    ):
        gaps.append("runtime_semantic_capture_assessment_incomplete")
    if not (
        isinstance(legality_assessment, dict)
        and legality_assessment.get("schema_version")
        == "l2d.realsim.release_legality_assessment.v0.1"
        and legality_assessment.get("component_record_binding") == binding
        and type(legality_assessment.get("legality_passed")) is bool
    ):
        gaps.append("release_legality_assessment_not_boolean_or_not_bound")
    if not (
        isinstance(execution_source_assessment, dict)
        and execution_source_assessment.get("schema_version")
        == "l2d.realsim.execution_source_assessment.v0.1"
        and execution_source_assessment.get("component_record_binding") == binding
        and execution_source_assessment.get("source_equivalence") is True
        and execution_source_assessment.get("execution_source_evidence_complete") is True
    ):
        gaps.append("execution_source_equivalence_not_established")
    if not (
        isinstance(pairing_assessment, dict)
        and pairing_assessment.get("schema_version")
        == "l2d.realsim.paired_input_equality_assessment.v0.1"
        and pairing_assessment.get("identity") == identity
        and pairing_assessment.get("component_record_binding") == binding
        and pairing_assessment.get("preintervention_non_language_equal") is True
        and pairing_assessment.get("model_visible_input_contract_verified") is True
        and pairing_assessment.get("route_command_target_leakage_absent") is True
    ):
        gaps.append("paired_non_language_or_leakage_gate_incomplete")
    if not (
        isinstance(denominator_slot, dict)
        and denominator_slot.get("schema_version")
        == "l2d.realsim.fixed_denominator_slot.v0.1"
        and denominator_slot.get("identity") == identity
        and denominator_slot.get("component_record_binding") == binding
        and denominator_slot.get("fixed_denominator_retained") is True
    ):
        gaps.append("fixed_denominator_slot_binding_incomplete")
    components = component_record.get("components")
    if not (
        isinstance(components, dict)
        and all(
            type(components.get(name)) is bool
            for name in (
                "own_contract", "opposing_contract", "progress", "deadline",
                "stability", "semantic_success",
            )
        )
    ):
        gaps.append("non_legality_score_components_incomplete")
    return {
        "schema_version": FORMAL_ELIGIBILITY_SCHEMA,
        "verification_scope": "single_fixed_denominator_slot_not_model_or_cohort_completion",
        "identity": identity,
        "component_record_binding": binding,
        "slot_formal_score_eligible": True if not gaps else None,
        "legality_passed": legality_assessment.get("legality_passed")
        if isinstance(legality_assessment, dict)
        and type(legality_assessment.get("legality_passed")) is bool
        else None,
        "eligibility_gaps": sorted(set(gaps)),
        "formal_model_evaluation_complete": False,
        "existing_component_record_mutated": False,
    }
