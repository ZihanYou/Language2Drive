"""Synthetic B2 full-segment adversaries; these are not scene qualifications."""
from copy import deepcopy

import pytest

from language2drive_realsim.schema import SchemaError, validate_source
from language2drive_realsim.verifier import verify_trace
from language2drive_realsim.verifier_selftest import _pcu, _semantic_trace
from .factories import make_release_source


def test_b2_cannot_pass_short_dwell_then_leave_before_the_end_marker():
    pcu, rows = _pcu("scoped_lane", "B"), _semantic_trace("scoped_lane", "post_dwell_departure")
    old = deepcopy(pcu)
    for arm in old["arms"].values():
        arm["semantic_contract"]["kind"] = "lane"
        arm["opposing_contract"]["kind"] = "lane"
    assert verify_trace(old, "R0", rows).passed  # Reproduce the old false positive.
    result = verify_trace(pcu, "R0", rows)
    assert not result.passed and result.failure == "F2_WRONG_SEMANTIC"
    assert result.diagnostics["nonexclusive_lane_times_s"] == [5.]


def test_b2_requires_full_exclusive_traversal_and_opposing_arm_fails():
    pcu, rows = _pcu("scoped_lane", "B"), _semantic_trace("scoped_lane", "positive")
    result = verify_trace(pcu, "R0", rows)
    assert result.passed and result.diagnostics["exclusive_lane_through_scope_exit"]
    assert result.diagnostics["entry_time_s"] == 2. and result.diagnostics["exit_time_s"] == 6.
    assert verify_trace(pcu, "R1", rows).failure == "F2_WRONG_SEMANTIC"


@pytest.mark.parametrize("fault,tag", [("truncated", "incomplete_scope"), ("no_exit", "incomplete_scope"),
    ("reverse", "wrong_scope_order"), ("simultaneous", "wrong_scope_order"), ("late", "late"),
    ("window_deadline", "late"), ("preinstruction_only", "entry_not_crossed"),
    ("escape_reentry", "left_scope_before_exit"), ("short_scope", "insufficient_scope_duration")])
def test_b2_rejects_incomplete_late_or_escaped_shared_scope(fault, tag):
    pcu, rows = _pcu("scoped_lane", "B"), _semantic_trace("scoped_lane", "positive")
    if fault == "truncated": rows = rows[:6]
    elif fault == "no_exit": rows[6]["gates_crossed"] = []
    elif fault == "reverse": rows[2]["gates_crossed"], rows[6]["gates_crossed"] = ["gate-exit"], ["gate-entry"]
    elif fault == "simultaneous": rows[2]["gates_crossed"] = ["gate-entry", "gate-exit"]
    elif fault == "late":
        rows[6]["gates_crossed"] = []
        rows[7]["gates_crossed"] = ["gate-exit"]
    elif fault == "window_deadline": pcu["decision_window"]["deadline_s"] = 5.5
    elif fault == "preinstruction_only":
        rows[2]["gates_crossed"] = []
        rows[0]["gates_crossed"] = ["gate-entry"]
    elif fault == "escape_reentry": rows[3]["scopes"] = []
    else:
        rows[2]["gates_crossed"] = []
        rows[5]["gates_crossed"] = ["gate-entry"]
    result = verify_trace(pcu, "R0", rows)
    assert not result.passed and result.failure == "F6_ORDER_SCOPE" and tag in result.secondary_tags


@pytest.mark.parametrize("fault", ["exit_violation", "ambiguous", "missing_lane"])
def test_b2_checks_exit_boundary_and_every_in_scope_sample(fault):
    pcu, rows = _pcu("scoped_lane", "B"), _semantic_trace("scoped_lane", "positive")
    if fault == "exit_violation":
        rows[6]["scopes"] = ["scope-main"]
        rows[6]["lane_ids"] = ["lane-0001"]
    elif fault == "ambiguous": rows[4]["lane_ids"] = ["lane-0000", "lane-0001"]
    else: rows[4]["lane_ids"] = []
    assert verify_trace(pcu, "R0", rows).failure == "F2_WRONG_SEMANTIC"


@pytest.mark.parametrize("fault", ["legacy_dwell_kind", "missing_gate", "unregistered", "same_ids", "same_geometry",
    "reversed_geometry", "different_arm_gate", "different_duration"])
def test_formal_b2_contract_requires_registered_shared_scope_boundaries(fault):
    source = make_release_source()
    pcu = next(p for p in source["pcus"] if p["task_family"] == "B2_scoped_lane_preference")
    c = pcu["arms"]["R0"]["semantic_contract"]
    if fault == "legacy_dwell_kind":
        for arm in pcu["arms"].values():
            arm["semantic_contract"]["kind"] = arm["opposing_contract"]["kind"] = "lane"
    elif fault == "missing_gate": del c["entry_gate"]
    elif fault == "unregistered": c["entry_gate"] = "gate-ffff"
    elif fault == "same_ids": c["exit_gate"] = c["entry_gate"]
    elif fault in {"same_geometry", "reversed_geometry"}:
        geometry = pcu["private_annotations"]["gates"]
        geometry[c["exit_gate"]] = geometry[c["entry_gate"]][::-1] if fault == "reversed_geometry" else deepcopy(geometry[c["entry_gate"]])
    elif fault == "different_arm_gate":
        pcu["private_annotations"]["gates"]["gate-ffff"] = [[10., -10.], [10., 10.]]
        c["exit_gate"] = "gate-ffff"
    else: c["stable_duration_s"] += .5
    if fault != "legacy_dwell_kind": pcu["arms"]["R1"]["opposing_contract"] = deepcopy(c)
    with pytest.raises(SchemaError): validate_source(source, require_exact_release=True)
