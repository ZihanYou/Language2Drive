"""Fail-closed contracts for release legality and execution-source evidence.

This module validates small, independently produced sidecars.  It does not
inspect ASLs, infer traffic rules, hash model weights, or turn a current
workspace snapshot into historical execution evidence.
"""

from datetime import datetime, timezone

from .io import stable_hash
from .single_seed_metrics import COMPONENT_SCHEMA


LEGALITY_SCHEMA = "l2d.realsim.release_legality_evidence.v0.1"
EXECUTION_SOURCE_SCHEMA = "l2d.realsim.execution_source_evidence.v0.1"
IDENTITY_FIELDS = (
    "model_name", "configuration_id", "protocol_id", "pcu_id", "arm_id",
    "seed", "session_uuid",
)
LEGALITY_RULES = (
    "collision", "offroad", "wrong_way", "traffic_signal", "speed_limit",
    "other_applicable_rules",
)
VERDICTS = {"pass", "fail", "not_applicable", "unknown"}
SPEED_UNITS = {"m/s", "km/h", "mph"}
ZERO_SEMANTICS = {
    "literal_zero_mps", "no_posted_limit", "not_mapped", "not_applicable",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _identity(component_record):
    _require(isinstance(component_record, dict), "component record must be an object")
    _require(component_record.get("schema_version") == COMPONENT_SCHEMA,
             "unsupported component record schema")
    _require(component_record.get("execution_status") == "completed"
             and component_record.get("evidence_valid") is True,
             "release sidecars require a completed evidence-valid component record")
    result = {field: component_record.get(field) for field in IDENTITY_FIELDS}
    _require(all(_text(result[field]) for field in IDENTITY_FIELDS if field != "seed")
             and type(result["seed"]) is int and result["seed"] >= 0,
             "invalid component record identity")
    return result


def component_record_binding(component_record):
    """Return a small-JSON binding; this never reads referenced evidence files."""

    _identity(component_record)
    return stable_hash(component_record)


def _validate_sidecar_binding(sidecar, component_record):
    identity = _identity(component_record)
    _require(sidecar.get("identity") == identity, "sidecar/component identity mismatch")
    _require(sidecar.get("component_record_binding") == stable_hash(component_record),
             "sidecar/component record binding mismatch")
    return identity


def _validate_reference(reference):
    _require(isinstance(reference, dict)
             and set(reference) == {"label", "path", "sha256", "size_bytes"},
             "evidence reference requires label, path, sha256 and size_bytes")
    _require(_text(reference["label"]) and _text(reference["path"]),
             "evidence reference label/path missing")
    _require(_digest(reference["sha256"]), "evidence reference content binding missing")
    _require(type(reference["size_bytes"]) is int and reference["size_bytes"] >= 0,
             "invalid evidence reference size")


def _authenticate_references(references, verifier, label):
    if not references:
        return []
    if verifier is None:
        return [label + ":content_bound_evidence_not_authenticated"]
    for reference in references:
        try:
            authenticated = verifier(reference)
        except Exception:
            authenticated = False
        if authenticated is not True:
            return [label + ":content_bound_evidence_authentication_failed"]
    return []


def _rule_evidence_gaps(name, rule, reference_verifier):
    _require(isinstance(rule, dict)
             and set(rule) == {"verdict", "evidence_references", "facts"},
             "invalid legality rule record: " + name)
    verdict = rule["verdict"]
    _require(verdict in VERDICTS, "invalid legality verdict: " + name)
    refs, facts = rule["evidence_references"], rule["facts"]
    _require(isinstance(refs, list) and isinstance(facts, dict),
             "invalid legality evidence/facts: " + name)
    for reference in refs:
        _validate_reference(reference)
    gaps = []
    if verdict == "unknown":
        return [name + ":verdict_unknown"]
    if not refs:
        gaps.append(name + ":content_bound_evidence_missing")
    else:
        gaps.extend(_authenticate_references(refs, reference_verifier, name))
    if verdict == "fail" and facts.get("violation_event_bound") is not True:
        gaps.append(name + ":violation_event_not_bound")

    if name in {"collision", "offroad"}:
        _require(verdict != "not_applicable", name + " cannot be not_applicable")
        if not _text(facts.get("metric_name")):
            gaps.append(name + ":metric_identity_missing")
        if verdict == "pass" and facts.get("full_trajectory_metric_coverage") is not True:
            gaps.append(name + ":full_trajectory_metric_coverage_missing")

    elif name == "wrong_way":
        _require(verdict != "not_applicable", "wrong_way cannot be not_applicable")
        for field, gap in (
            ("directed_lane_applicability_bound", "directed_lane_applicability_missing"),
            ("vehicle_heading_bound", "vehicle_heading_binding_missing"),
        ):
            if facts.get(field) is not True:
                gaps.append(name + ":" + gap)
        if verdict == "pass" and facts.get("full_trajectory_lane_coverage") is not True:
            gaps.append(name + ":full_trajectory_lane_coverage_missing")

    elif name == "traffic_signal":
        for field, gap in (
            ("stopline_lane_binding", "stopline_lane_binding_missing"),
            ("full_trajectory_signal_applicability", "signal_applicability_coverage_missing"),
        ):
            if facts.get(field) is not True:
                gaps.append(name + ":" + gap)
        if verdict in {"pass", "fail"}:
            for field, gap in (
                ("controlling_signal_binding", "controlling_signal_binding_missing"),
                ("dynamic_state_time_series", "dynamic_signal_state_missing"),
                ("crossing_time_bound", "stopline_crossing_time_missing"),
            ):
                if facts.get(field) is not True:
                    gaps.append(name + ":" + gap)
        elif facts.get("not_applicable_basis_bound") is not True:
            gaps.append(name + ":not_applicable_basis_missing")

    elif name == "speed_limit":
        for field, gap in (
            ("directed_lane_binding", "directed_lane_binding_missing"),
            ("applicable_intervals_bound", "applicable_range_missing"),
        ):
            if facts.get(field) is not True:
                gaps.append(name + ":" + gap)
        if facts.get("source_unit") not in SPEED_UNITS:
            gaps.append(name + ":authoritative_unit_missing")
        if facts.get("zero_value_semantics") not in ZERO_SEMANTICS:
            gaps.append(name + ":zero_value_semantics_missing")
        if verdict in {"pass", "not_applicable"} and facts.get(
                "full_trajectory_speed_limit_scope") is not True:
            gaps.append(name + ":full_trajectory_scope_missing")

    elif name == "other_applicable_rules":
        for field, gap in (
            ("rule_inventory_complete", "rule_inventory_incomplete"),
            ("full_trajectory_applicability", "applicability_coverage_missing"),
        ):
            if facts.get(field) is not True:
                gaps.append(name + ":" + gap)
    return gaps


def assess_legality_evidence(sidecar, component_record, reference_verifier=None):
    """Validate a legality overlay and derive bool/null without guessing gaps.

    A valid violation can establish False even when other rules are unknown.
    True requires every rule to pass (or be proven not applicable) with all
    rule-specific evidence.  Incompleteness remains None, never a convenient
    measured zero.
    """

    _require(isinstance(sidecar, dict), "legality sidecar must be an object")
    _require(sidecar.get("schema_version") == LEGALITY_SCHEMA,
             "unsupported legality sidecar schema")
    _validate_sidecar_binding(sidecar, component_record)
    rules = sidecar.get("rules")
    _require(isinstance(rules, dict) and set(rules) == set(LEGALITY_RULES),
             "legality sidecar must contain the exact release rule set")
    gaps = {}
    for name in LEGALITY_RULES:
        rule_gaps = _rule_evidence_gaps(name, rules[name], reference_verifier)
        if rule_gaps:
            gaps[name] = rule_gaps

    supported_failures = [name for name in LEGALITY_RULES
                          if rules[name]["verdict"] == "fail" and name not in gaps]
    if supported_failures:
        verdict = False
    elif not gaps and all(rules[name]["verdict"] in {"pass", "not_applicable"}
                          for name in LEGALITY_RULES):
        verdict = True
    else:
        verdict = None
    claim = sidecar.get("claimed_legality_passed")
    _require(claim is None or type(claim) is bool,
             "claimed_legality_passed must be boolean or null")
    _require(claim is verdict,
             "claimed legality verdict is not supported by the supplied rule evidence")
    existing = component_record.get("components", {}).get("safe_legal")
    _require(existing is None or existing is verdict,
             "legality sidecar contradicts the frozen component record")
    return {
        "schema_version": "l2d.realsim.release_legality_assessment.v0.1",
        "verification_scope": "small_sidecar_contract_not_underlying_evidence_authenticity",
        "legality_passed": verdict,
        "supported_failure_rules": supported_failures,
        "evidence_gaps": gaps,
        "component_record_binding": sidecar["component_record_binding"],
    }


def _utc_timestamp(value, name):
    _require(_text(value), "missing " + name)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ValueError("invalid " + name) from error
    _require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
             name + " must carry a UTC offset")
    return parsed


def assess_execution_source_evidence(sidecar, component_record,
                                     reference_verifier=None, capture_verifier=None):
    """Gate execution-source claims without retroactively blessing a checkout.

    The function checks the small sidecar and its recorded content bindings; it
    intentionally does not open or hash referenced files.  A current workspace
    or other retrospective capture therefore remains incomplete by design.
    """

    _require(isinstance(sidecar, dict), "execution-source sidecar must be an object")
    _require(sidecar.get("schema_version") == EXECUTION_SOURCE_SCHEMA,
             "unsupported execution-source sidecar schema")
    _validate_sidecar_binding(sidecar, component_record)
    gaps = []
    capture = sidecar.get("capture")
    _require(isinstance(capture, dict), "execution-source capture record missing")
    if capture.get("mode") != "pre_and_post_execution":
        gaps.append("retrospective_capture_cannot_attest_execution_source")
    else:
        pre = _utc_timestamp(capture.get("pre_capture_at_utc"), "pre capture time")
        first = _utc_timestamp(capture.get("first_submission_at_utc"), "first submission time")
        last = _utc_timestamp(capture.get("last_terminal_at_utc"), "last terminal time")
        post = _utc_timestamp(capture.get("post_capture_at_utc"), "post capture time")
        if not pre <= first <= last <= post:
            gaps.append("capture_timeline_does_not_enclose_execution")
        if capture.get("references_unchanged") is not True:
            gaps.append("pre_post_reference_stability_missing")
        if not _digest(capture.get("immutable_run_binding")):
            gaps.append("immutable_run_binding_missing")
        if capture_verifier is None:
            gaps.append("execution_time_capture_not_independently_authenticated")
        else:
            try:
                capture_authenticated = capture_verifier(capture)
            except Exception:
                capture_authenticated = False
            if capture_authenticated is not True:
                gaps.append("execution_time_capture_authentication_failed")

    model = sidecar.get("model")
    _require(isinstance(model, dict), "model source record missing")
    for field in ("model_id", "snapshot_revision", "identity_binding"):
        if not _text(model.get(field)):
            gaps.append("model_" + field + "_missing")
    if not _digest(model.get("identity_binding")):
        gaps.append("model_identity_binding_not_content_bound")
    try:
        checkpoint_reference = model.get("checkpoint_reference")
        _validate_reference(checkpoint_reference)
        gaps.extend(_authenticate_references(
            [checkpoint_reference], reference_verifier, "checkpoint"))
    except ValueError:
        gaps.append("checkpoint_content_reference_missing")

    adapter = sidecar.get("adapter")
    _require(isinstance(adapter, dict), "adapter source record missing")
    if not _text(adapter.get("driver_version_id")):
        gaps.append("driver_version_missing")
    for field, gap in (
        ("source_tree_reference", "adapter_source_content_reference_missing"),
        ("input_contract_reference", "input_contract_content_reference_missing"),
        ("input_equivalence_review_reference",
         "input_equivalence_review_content_reference_missing"),
    ):
        try:
            reference = adapter.get(field)
            _validate_reference(reference)
            gaps.extend(_authenticate_references(
                [reference], reference_verifier, field))
        except ValueError:
            gaps.append(gap)
    if adapter.get("official_input_equivalence") is not True:
        gaps.append("official_input_equivalence_not_established")

    launch = sidecar.get("launch")
    _require(isinstance(launch, dict), "launch source record missing")
    if not _digest(launch.get("manifest_binding")):
        gaps.append("launch_manifest_binding_missing")
    try:
        launch_context_reference = launch.get("launch_context_reference")
        _validate_reference(launch_context_reference)
        gaps.extend(_authenticate_references(
            [launch_context_reference], reference_verifier, "launch_context"))
    except ValueError:
        gaps.append("launch_context_content_reference_missing")
    refs = launch.get("execution_file_references")
    if not isinstance(refs, list) or not refs:
        gaps.append("execution_file_references_missing")
    else:
        labels = []
        for reference in refs:
            try:
                _validate_reference(reference)
                labels.append(reference["label"])
            except ValueError:
                gaps.append("execution_file_content_reference_invalid")
        if len(labels) != len(set(labels)):
            gaps.append("duplicate_execution_file_reference_label")
        if len(labels) == len(refs):
            gaps.extend(_authenticate_references(
                refs, reference_verifier, "execution_files"))
    if launch.get("complete_file_set") is not True:
        gaps.append("complete_execution_file_set_not_attested")

    complete = not gaps
    claim = sidecar.get("claimed_source_equivalence")
    _require(claim is None or type(claim) is bool,
             "claimed_source_equivalence must be boolean or null")
    _require(claim is (True if complete else None),
             "source-equivalence claim is not supported by execution-time evidence")
    return {
        "schema_version": "l2d.realsim.execution_source_assessment.v0.1",
        "verification_scope": "recorded_binding_contract_not_recomputed_file_authenticity",
        "execution_source_evidence_complete": complete,
        "source_equivalence": True if complete else None,
        "evidence_gaps": sorted(set(gaps)),
        "component_record_binding": sidecar["component_record_binding"],
    }
