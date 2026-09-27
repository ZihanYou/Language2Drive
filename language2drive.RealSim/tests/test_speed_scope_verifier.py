"""Synthetic adversaries for complete, ordered, shared spatial speed scopes."""
from copy import deepcopy

import pytest

from language2drive_realsim.schema import SchemaError, validate_source
from language2drive_realsim.verifier import verify_trace
from language2drive_realsim.verifier_selftest import _pcu, _semantic_trace
from .factories import make_release_source


@pytest.mark.parametrize("kind", ["speed_band", "triggered_speed_band"])
@pytest.mark.parametrize("mutation,tag", [
    ("truncated", "incomplete_scope"),
    ("no_exit", "incomplete_scope"),
    ("reverse", "wrong_scope_order"),
    ("simultaneous", "wrong_scope_order"),
    ("late_exit", "late"),
    ("window_deadline", "late"),
    ("preintervention_entry", "entry_not_crossed"),
    ("escape_reentry", "left_scope_before_exit"),
    ("one_sample", "insufficient_scope_samples"),
])
def test_incomplete_or_wrong_scope_cannot_pass(kind, mutation, tag):
    pcu, rows = _pcu(kind, "B"), _semantic_trace(kind, "positive")
    if mutation == "truncated":
        rows = rows[:6]  # Enough global progress, still no spatial completion.
    elif mutation == "no_exit":
        rows[6]["gates_crossed"] = []
    elif mutation == "reverse":
        rows[1]["gates_crossed"], rows[6]["gates_crossed"] = ["gate-exit"], ["gate-entry"]
    elif mutation == "simultaneous":
        rows[1]["gates_crossed"] = ["gate-entry", "gate-exit"]
    elif mutation == "late_exit":
        rows[6]["gates_crossed"] = []
        rows[6]["scopes"] = ["scope-main"]
        rows[7]["gates_crossed"] = ["gate-exit"]
    elif mutation == "window_deadline":
        pcu["decision_window"]["deadline_s"] = 5.5
    elif mutation == "preintervention_entry":
        rows[0]["gates_crossed"] = ["gate-entry"]
    elif mutation == "escape_reentry":
        rows[3]["scopes"] = []
        rows[3]["speed_mps"] = 20.0  # Cannot hide speed by leaving scope.
    elif mutation == "one_sample":
        rows[1]["gates_crossed"] = []
        rows[5]["gates_crossed"] = ["gate-entry"]
    result = verify_trace(pcu, "R0", rows)
    assert not result.passed and result.failure == "F6_ORDER_SCOPE"
    assert tag in result.secondary_tags


@pytest.mark.parametrize("kind", ["speed_band", "triggered_speed_band"])
def test_complete_scope_passes_but_opposing_and_single_speed_spike_fail(kind):
    pcu, rows = _pcu(kind, "B"), _semantic_trace(kind, "positive")
    result = verify_trace(pcu, "R0", rows)
    assert result.passed
    assert result.diagnostics["shared_scope_traversed"] is True
    assert result.diagnostics["exit_time_s"] == pcu["decision_window"]["deadline_s"]
    assert verify_trace(pcu, "R1", rows).failure == "F5_QUANTITATIVE"
    rows[3]["speed_mps"] = 9.0
    assert verify_trace(pcu, "R0", rows).failure == "F5_QUANTITATIVE"


def test_exit_boundary_sample_is_checked_when_still_in_scope():
    pcu, rows = _pcu("speed_band", "B"), _semantic_trace("speed_band", "positive")
    rows[6]["scopes"] = ["scope-main"]
    rows[6]["speed_mps"] = 20.0
    assert verify_trace(pcu, "R0", rows).failure == "F5_QUANTITATIVE"


@pytest.mark.parametrize("mutation,expected", [
    ("one_activation", None), ("outside_scope_activation", None),
    ("post_trigger_spike", "F5_QUANTITATIVE"), ("post_trigger_missing_speed", "INVALID_SIM"),
    ("late_trigger", "F6_ORDER_SCOPE"), ("preinstruction_only", "F6_ORDER_SCOPE"),
    ("missing_trigger", "F6_ORDER_SCOPE"), ("different_trigger", "F6_ORDER_SCOPE"),
])
def test_b3_activation_latches_until_complete_scope_exit(mutation, expected):
    pcu, rows = _pcu("triggered_speed_band", "B"), _semantic_trace("triggered_speed_band", "positive")
    for row in rows:
        row["triggers"] = []
    rows[1]["triggers"] = ["trigger-main"]
    if mutation == "outside_scope_activation":
        # Trigger at 1 s, before the common segment begins at 2 s.
        rows[1]["scopes"] = []
        rows[1]["gates_crossed"] = []
        rows[2]["gates_crossed"] = ["gate-entry"]
    elif mutation == "post_trigger_spike":
        rows[3]["speed_mps"] = 20.
    elif mutation == "post_trigger_missing_speed":
        del rows[3]["speed_mps"]
    elif mutation in {"late_trigger", "preinstruction_only", "missing_trigger", "different_trigger"}:
        rows[1]["triggers"] = []
        if mutation == "late_trigger":
            rows[3]["triggers"] = ["trigger-main"]
        elif mutation == "preinstruction_only":
            rows[0]["triggers"] = ["trigger-main"]
        elif mutation == "different_trigger":
            rows[1]["triggers"] = ["trigger-other"]
    result = verify_trace(pcu, "R0", rows)
    assert result.passed is (expected is None)
    assert result.failure == expected
    if expected is None:
        assert result.diagnostics["trigger_activation_time_s"] == 1.
        assert result.diagnostics["trigger_latched_through_scope_exit"] is True
        assert result.diagnostics["evaluated_samples"] == (4 if mutation == "outside_scope_activation" else 5)
    elif expected == "F6_ORDER_SCOPE":
        assert "trigger_avoidance" in result.secondary_tags


def test_scope_gates_are_derived_from_registered_geometry():
    pcu, rows = _pcu("speed_band", "B"), _semantic_trace("speed_band", "positive")
    for row in rows:
        row["gates_crossed"] = []
    annotations = {"gates": {"gate-entry": [[0.5, -1], [0.5, 1]],
                              "gate-exit": [[5.5, -1], [5.5, 1]]}}
    assert verify_trace(pcu, "R0", rows, annotations).passed
    assert not verify_trace(pcu, "R0", rows[:6], annotations).passed


@pytest.mark.parametrize("mutation", [
    "missing_entry", "empty_exit", "same_id", "different_arm_entry",
    "unregistered", "same_geometry", "reversed_same_geometry",
])
def test_source_rejects_unbound_or_nonshared_speed_scope_gates(mutation):
    source = make_release_source()
    pcu = next(p for p in source["pcus"] if p["task_family"].startswith("B1"))
    c = pcu["arms"]["R0"]["semantic_contract"]
    other = pcu["arms"]["R1"]["semantic_contract"]
    if mutation == "missing_entry":
        del c["entry_gate"]
    elif mutation == "empty_exit":
        c["exit_gate"] = ""
    elif mutation == "same_id":
        c["exit_gate"] = c["entry_gate"]
    elif mutation == "different_arm_entry":
        other["entry_gate"] = "gate-different"
    elif mutation == "unregistered":
        c["entry_gate"] = other["entry_gate"] = "gate-unregistered"
    else:
        gates = pcu["private_annotations"]["gates"]
        gates[c["exit_gate"]] = deepcopy(gates[c["entry_gate"]])
        if mutation == "reversed_same_geometry":
            gates[c["exit_gate"]].reverse()
    pcu["arms"]["R0"]["opposing_contract"] = deepcopy(other)
    pcu["arms"]["R1"]["opposing_contract"] = deepcopy(c)
    with pytest.raises(SchemaError, match="gate|geometry"):
        validate_source(source)
