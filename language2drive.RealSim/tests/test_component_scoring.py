"""Synthetic trajectories exercise component semantics, not real authority."""

from copy import deepcopy

import pytest

from language2drive_realsim.component_scoring import score_trace_components
from language2drive_realsim.verifier import verify_trace
from language2drive_realsim.verifier_selftest import _pcu, _semantic_trace

KINDS = ("gate", "lane", "scoped_lane", "speed_band", "triggered_speed_band")
TEST_THRESHOLDS = {"maximum_sample_gap_s": 1.0, "speed_epsilon_mps": 1e-6}


def fixture(kind, mode="positive"):
    return _pcu(kind, "A" if kind in {"gate", "lane"} else "B"), _semantic_trace(kind, mode)


def score(pcu, rows, **kwargs):
    return score_trace_components(pcu, "R0", rows, threshold_profile=TEST_THRESHOLDS,
                                  **kwargs)


@pytest.mark.parametrize("kind", KINDS)
def test_positive_all_components_with_independent_legality(kind):
    pcu, rows = fixture(kind)
    result = score(pcu, rows, legality_passed=True)
    assert result["components"] == dict(own_contract=True, opposing_contract=False,
        semantic_success=True, safe_legal=True, progress=True, deadline=True, stability=True)
    assert result["trace_usable"]
    assert not result["formal_model_evaluation_complete"]


@pytest.mark.parametrize("kind", KINDS)
def test_missing_legality_stays_unknown(kind):
    pcu, rows = fixture(kind)
    result = score(pcu, rows)
    assert result["components"]["safe_legal"] is None
    assert result["components"]["own_contract"] is True
    assert result["components"]["semantic_success"] is True


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("unsafe_field", ["collision", "offroad"])
def test_unsafe_does_not_erase_semantics_or_mutate_trace(kind, unsafe_field):
    pcu, rows = fixture(kind)
    rows[4][unsafe_field] = True
    before = deepcopy((pcu, rows))
    result = score(pcu, rows)
    assert result["components"]["safe_legal"] is False
    assert result["components"]["semantic_success"] is True
    assert result["components"]["own_contract"] is True
    assert verify_trace(pcu, "R0", rows).passed is False
    assert (pcu, rows) == before


def test_legal_violation_is_independent_of_collision():
    pcu, rows = fixture("gate")
    result = score(pcu, rows, legality_passed=False)
    assert result["diagnostics"]["physical_safe"] is True
    assert result["components"]["safe_legal"] is False


@pytest.mark.parametrize("kind", ["gate", "lane"])
def test_late_semantics_survives_deadline_failure(kind):
    pcu, rows = fixture(kind, "late")
    result = score(pcu, rows)
    assert result["components"]["own_contract"] is False
    assert result["components"]["semantic_success"] is True
    assert result["components"]["deadline"] is False


@pytest.mark.parametrize("kind", ["scoped_lane", "speed_band", "triggered_speed_band"])
def test_late_scope_without_geometry_change(kind):
    pcu, rows = fixture(kind)
    pcu["decision_window"]["deadline_s"] = 5.5
    pcu["arms"]["R0"]["semantic_contract"]["deadline_s"] = 5.5
    result = score(pcu, rows)
    assert result["components"]["own_contract"] is False
    assert result["components"]["semantic_success"] is True
    assert result["components"]["deadline"] is False
    assert result["components"]["stability"] is True


def test_transient_lane_semantics_without_stability():
    pcu, rows = fixture("lane", "partial")
    result = score(pcu, rows)
    assert result["components"]["semantic_success"] is True
    assert result["components"]["stability"] is False
    assert result["components"]["own_contract"] is False


def test_too_short_scoped_lane_semantics_without_stability():
    pcu, rows = fixture("scoped_lane")
    pcu["arms"]["R0"]["semantic_contract"]["stable_duration_s"] = 5.0
    result = score(pcu, rows)
    assert result["components"]["semantic_success"] is True
    assert result["components"]["stability"] is False
    assert result["components"]["own_contract"] is False


@pytest.mark.parametrize("kind", ["speed_band", "triggered_speed_band"])
def test_full_scope_required_even_for_semantics(kind):
    pcu, rows = fixture(kind)
    rows[6]["gates_crossed"] = []
    result = score(pcu, rows)
    assert result["components"]["semantic_success"] is False
    assert result["components"]["stability"] is False
    assert result["components"]["deadline"] is False


def test_trigger_cannot_be_relaxed_away():
    pcu, rows = fixture("triggered_speed_band")
    for row in rows:
        row["triggers"] = []
    result = score(pcu, rows)
    assert result["components"]["semantic_success"] is False
    assert result["components"]["own_contract"] is False


def test_speed_violation_independent_of_deadline_and_stability():
    pcu, rows = fixture("speed_band")
    rows[3]["speed_mps"] = 20.0
    result = score(pcu, rows)
    assert result["components"]["semantic_success"] is False
    assert result["components"]["deadline"] is True
    assert result["components"]["stability"] is True


def test_missing_speed_not_a_measured_negative():
    pcu, rows = fixture("speed_band")
    del rows[3]["speed_mps"]
    result = score(pcu, rows)
    assert result["components"]["semantic_success"] is None
    assert result["components"]["own_contract"] is None


def test_missing_progress_does_not_default_to_zero_or_success():
    pcu, rows = fixture("gate")
    del rows[3]["progress_m"]
    assert score(pcu, rows)["components"]["progress"] is None


def test_insufficient_progress_does_not_erase_semantics():
    pcu, rows = fixture("gate")
    for row in rows:
        row["progress_m"] = 0.0
    result = score(pcu, rows)
    assert result["components"]["progress"] is False
    assert result["components"]["semantic_success"] is True


@pytest.mark.parametrize("field", ["infrastructure_failure", "render_valid"])
def test_later_invalidity_not_hidden_by_prior_collision(field):
    pcu, rows = fixture("gate")
    rows[2]["collision"] = True
    rows[8][field] = field == "infrastructure_failure"
    if field == "render_valid":
        del rows[8]["reconstruction_support_valid"]
    result = score(pcu, rows, legality_passed=True)
    assert not result["trace_usable"]
    assert all(v is None for v in result["components"].values())


def test_policy_support_exit_scored_unsafe_not_invalid():
    pcu, rows = fixture("gate")
    rows[8].update(render_valid=False, reconstruction_support_valid=False)
    result = score(pcu, rows)
    assert result["trace_usable"]
    assert result["components"]["safe_legal"] is False
    assert result["components"]["semantic_success"] is True


@pytest.mark.parametrize("mutation", ["short", "late_start", "gap", "bad_bool"])
def test_invalid_trace_coverage_or_types_rejected(mutation):
    pcu, rows = fixture("gate")
    if mutation == "short":
        rows = rows[:5]
    elif mutation == "late_start":
        rows = rows[1:]
    elif mutation == "gap":
        del rows[3]
    else:
        rows[3]["collision"] = 0
    with pytest.raises(ValueError):
        score(pcu, rows)


@pytest.mark.parametrize("value", [1, "yes", [], {}])
def test_legality_must_not_be_truthy_coercion(value):
    pcu, rows = fixture("gate")
    with pytest.raises(ValueError):
        score(pcu, rows, legality_passed=value)


def test_unsupported_family_explicitly_rejected():
    pcu, rows = fixture("stop_region")
    with pytest.raises(ValueError, match="unsupported"):
        score(pcu, rows)


def test_frozen_speed_epsilon_preserved():
    pcu, rows = fixture("speed_band", "edge_pass")
    assert score(pcu, rows)["components"]["own_contract"] is True
    rows[3]["speed_mps"] = 8.001
    assert score(pcu, rows)["components"]["own_contract"] is False


def test_explicit_horizon_cannot_shorten_frozen_contract():
    pcu, rows = fixture("gate")
    with pytest.raises(ValueError, match="cannot shorten"):
        score(pcu, rows, required_end_s=1.0)


def test_explicit_longer_horizon_requires_actual_coverage():
    pcu, rows = fixture("gate")
    with pytest.raises(ValueError, match="complete required horizon"):
        score(pcu, rows, required_end_s=rows[-1]["time_s"] + 1.0)
