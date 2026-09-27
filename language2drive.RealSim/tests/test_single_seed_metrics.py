"""Synthetic component judgements test arithmetic, not actual model evidence."""

from copy import deepcopy

import pytest

from language2drive_realsim.constants import COUPLED_SEEDS
from language2drive_realsim.single_seed_metrics import (
    COMPONENT_SCHEMA, aggregate_single_seed, single_seed_slots,
)


@pytest.fixture
def cohort():
    source = {"pcus": [dict(pcu_id="p" + str(i), primary_layer="A" if i < 6 else "B",
                            arms={"R0": {}, "R1": {}}) for i in range(30)]}
    sidecar = {"sessions": {"%s-%s-%s" % (p["pcu_id"], arm, seed):
        dict(pcu_id=p["pcu_id"], arm_id=arm, seed=seed)
        for p in source["pcus"] for arm in ("R0", "R1") for seed in COUPLED_SEEDS}}
    return source, sidecar


def complete_rows(cohort):
    return [dict(**slot, schema_version=COMPONENT_SCHEMA, model_name="test-model",
                 configuration_id="fixed-config", protocol_id="test-protocol",
                 execution_status="completed", evidence_valid=True,
                 components=dict(own_contract=True, opposing_contract=False,
                                 safe_legal=True, progress=True, deadline=True,
                                 stability=True, semantic_success=True))
            for slot in single_seed_slots(*cohort)]


def score(cohort, rows):
    return aggregate_single_seed(*cohort, rows, model_name="test-model",
                                 configuration_id="fixed-config", protocol_id="test-protocol")


def test_full_ledger_and_layer_denominators(cohort):
    report = score(cohort, complete_rows(cohort))
    assert report["expected_runs"] == report["supplied_runs"] == 60
    assert report["valid_completed_runs"] == 60
    assert report["component_scoring_complete"]
    assert not report["formal_model_evaluation_complete"]
    assert report["metrics"]["SISR"]["rate"] == 1
    assert report["metrics"]["CFPS"]["numerator"] == 30
    assert report["by_layer"]["A"]["SISR"]["denominator"] == 12
    assert report["by_layer"]["B"]["SISR"]["denominator"] == 48
    assert report["by_layer"]["A"]["CFPS"]["denominator"] == 6
    assert report["by_layer"]["B"]["CFPS"]["denominator"] == 24
    assert report["by_layer"]["C"]["SISR"]["rate"] is None
    assert not report["by_layer"]["C"]["SISR"]["applicable"]
    assert all(v is None for v in report["not_applicable"].values())


def test_missing_runs_zero_without_shrinking_denominator(cohort):
    report = score(cohort, complete_rows(cohort)[:1])
    assert report["metrics"]["SISR"]["rate"] == 1 / 60
    assert report["metrics"]["CFPS"]["rate"] == 0
    assert report["execution_status_counts"] == {"completed": 1, "missing": 59}
    assert not report["component_scoring_complete"]


def test_no_runs_produces_all_slots(cohort):
    report = score(cohort, [])
    assert len(report["results"]) == 60
    assert report["metrics"]["SISR"]["numerator"] == 0
    assert report["metrics"]["SISR"]["unknown_slots"] == 60
    assert report["runtime_completed_runs"] == 0


@pytest.mark.parametrize("status", ["missing", "crash", "timeout", "no_trace", "invalid_evidence"])
def test_failed_execution_never_earns_credit(cohort, status):
    rows = complete_rows(cohort)
    rows[0]["execution_status"] = status
    report = score(cohort, rows)
    assert report["runtime_completed_runs"] == 59
    assert all(report["metrics"][m]["numerator"] == 59 for m in ("SISR", "SemISR", "Safety", "Progress"))
    assert report["metrics"]["CFPS"]["numerator"] == 29
    assert not report["component_scoring_complete"]


@pytest.mark.parametrize("status", ["invalid_model_output", "completed_pair_uncertified"])
def test_explicit_terminal_failure_preserves_denominator_and_nulls(cohort, status):
    rows = complete_rows(cohort)
    rows[0].update(execution_status=status, evidence_valid=False,
                   components=dict.fromkeys(rows[0]["components"]))
    original = deepcopy(rows)
    report = score(cohort, rows)
    assert rows == original
    assert report["expected_runs"] == report["supplied_runs"] == 60
    assert report["valid_completed_runs"] == 59
    assert report["execution_status_counts"][status] == 1
    assert report["results"][0]["execution_status"] == status
    assert all(value is None for value in report["results"][0]["components"].values())
    for name in ("SISR", "SemISR", "Safety", "Progress"):
        assert report["metrics"][name]["numerator"] == 59
        assert report["metrics"][name]["denominator"] == 60
        assert report["metrics"][name]["unknown_slots"] == 1
    assert report["metrics"]["CFPS"]["numerator"] == 29
    assert not report["formal_model_evaluation_complete"]


@pytest.mark.parametrize("status", ["invalid_model_output", "completed_pair_uncertified"])
@pytest.mark.parametrize("mutation", ["valid", "true", "false", "missing_component"])
def test_explicit_terminal_failure_rejects_fabricated_scoring(cohort, status, mutation):
    rows = complete_rows(cohort)
    rows[0].update(execution_status=status, evidence_valid=False,
                   components=dict.fromkeys(rows[0]["components"]))
    if mutation == "valid":
        rows[0]["evidence_valid"] = True
    elif mutation == "missing_component":
        rows[0]["components"].pop("progress")
    else:
        rows[0]["components"]["progress"] = mutation == "true"
    with pytest.raises(ValueError, match="null unscored components"):
        score(cohort, rows)


@pytest.mark.parametrize("field,value", [("own_contract", False), ("opposing_contract", True),
    ("safe_legal", False), ("progress", False), ("deadline", False), ("stability", False)])
def test_sisr_requires_every_strict_component(cohort, field, value):
    rows = complete_rows(cohort)
    rows[0]["components"][field] = value
    report = score(cohort, rows)
    assert report["metrics"]["SISR"]["numerator"] == 59
    assert report["metrics"]["SemISR"]["numerator"] == 60
    assert report["metrics"]["CFPS"]["numerator"] == 29
    assert report["component_scoring_complete"]


def test_safe_but_stationary_does_not_pass(cohort):
    rows = complete_rows(cohort)
    for row in rows:
        row["components"].update(own_contract=False, semantic_success=False, progress=False)
    report = score(cohort, rows)
    assert report["metrics"]["Safety"]["rate"] == 1
    assert all(report["metrics"][m]["rate"] == 0 for m in ("SISR", "SemISR", "Progress", "CFPS"))


def test_unsafe_semantic_success_is_independent(cohort):
    rows = complete_rows(cohort)
    rows[0]["components"]["safe_legal"] = False
    report = score(cohort, rows)
    assert report["results"][0]["scores"] == {"SISR": 0, "SemISR": 1, "Safety": 0, "Progress": 1}


def test_unknown_component_not_inferred_from_legacy_pass(cohort):
    rows = complete_rows(cohort)
    rows[0]["components"] = {}
    rows[0]["passed"] = True
    report = score(cohort, rows)
    assert not report["component_scoring_complete"]
    assert all(v is None for v in report["results"][0]["components"].values())
    assert all(value == 0 for value in report["results"][0]["scores"].values())


def test_own_contract_implies_semantic_but_not_other_metrics(cohort):
    rows = complete_rows(cohort)
    rows[0]["components"] = {"own_contract": True}
    report = score(cohort, rows)
    assert report["results"][0]["scores"] == {"SISR": 0, "SemISR": 1, "Safety": 0, "Progress": 0}
    assert not report["component_scoring_complete"]


def test_invalid_authority_zeroes_all_metrics(cohort):
    rows = complete_rows(cohort)
    rows[0]["evidence_valid"] = False
    report = score(cohort, rows)
    assert report["runtime_completed_runs"] == 60
    assert report["valid_completed_runs"] == 59
    assert all(v == 0 for v in report["results"][0]["scores"].values())


@pytest.mark.parametrize("field,value", [("model_name", "other"), ("configuration_id", "mixed"),
    ("protocol_id", "old"), ("schema_version", "old"), ("session_uuid", "wrong"),
    ("seed", 1702), ("seed", True), ("seed", "1701"), ("pcu_id", "unknown"),
    ("arm_id", "R2"), ("execution_status", "ok"), ("evidence_valid", 1)])
def test_mixed_or_invalid_identity_rejected(cohort, field, value):
    rows = complete_rows(cohort)
    rows[0][field] = value
    with pytest.raises(ValueError):
        score(cohort, rows)


@pytest.mark.parametrize("value", [1, "true", [], {}])
def test_component_requires_actual_bool(cohort, value):
    rows = complete_rows(cohort)
    rows[0]["components"]["progress"] = value
    with pytest.raises(ValueError, match="boolean or null"):
        score(cohort, rows)


def test_contradictory_semantic_flag_rejected(cohort):
    rows = complete_rows(cohort)
    rows[0]["components"]["semantic_success"] = False
    with pytest.raises(ValueError, match="implies"):
        score(cohort, rows)


def test_duplicate_results_rejected_even_if_equal(cohort):
    rows = complete_rows(cohort)
    with pytest.raises(ValueError, match="duplicate result"):
        score(cohort, rows + [deepcopy(rows[0])])


def test_duplicate_compiled_identity_rejected(cohort):
    cohort[1]["sessions"]["duplicate"] = deepcopy(cohort[1]["sessions"]["p0-R0-1701"])
    with pytest.raises(ValueError, match="duplicate selected"):
        single_seed_slots(*cohort)


def test_missing_compiled_session_rejected(cohort):
    del cohort[1]["sessions"]["p0-R0-1701"]
    with pytest.raises(ValueError, match="incomplete compiled"):
        single_seed_slots(*cohort)


def test_duplicate_pcu_rejected(cohort):
    cohort[0]["pcus"].append(deepcopy(cohort[0]["pcus"][0]))
    with pytest.raises(ValueError, match="duplicate PCU"):
        single_seed_slots(*cohort)


@pytest.mark.parametrize("seed", [True, "1701", -1, None])
def test_invalid_seed_rejected(cohort, seed):
    with pytest.raises(ValueError):
        single_seed_slots(*cohort, seed=seed)


def test_preserves_input_and_legacy_protocol(cohort):
    original = deepcopy(cohort)
    rows = complete_rows(cohort)
    rows[0]["origin"] = {"path": "old-evidence"}
    before = deepcopy(rows)
    report = score(cohort, reversed(rows))
    assert rows == before and cohort == original
    report["results"][0]["origin"]["path"] = "edited-report-only"
    assert rows == before
    assert COUPLED_SEEDS == (1701, 1702, 1703)


def test_order_independent(cohort):
    rows = complete_rows(cohort)
    assert score(cohort, rows) == score(cohort, reversed(rows))


@pytest.mark.parametrize("field", ["own_contract", "opposing_contract", "safe_legal",
                                  "progress", "deadline", "stability"])
def test_unknown_strict_component_prevents_success(cohort, field):
    rows = complete_rows(cohort)
    rows[0]["components"][field] = None
    report = score(cohort, rows)
    assert report["metrics"]["SISR"]["numerator"] == 59
    assert report["metrics"]["SISR"]["unknown_slots"] == 1
    assert report["metrics"]["CFPS"]["unknown_slots"] == 1
    assert not report["component_scoring_complete"]


def test_only_one_arm_succeeds_in_every_pair(cohort):
    rows = complete_rows(cohort)
    for row in rows:
        if row["arm_id"] == "R1":
            row["components"]["deadline"] = False
    report = score(cohort, rows)
    assert report["metrics"]["SISR"]["rate"] == 0.5
    assert report["metrics"]["CFPS"]["rate"] == 0


def test_unknown_component_name_rejected(cohort):
    rows = complete_rows(cohort)
    rows[0]["components"]["passed"] = True
    with pytest.raises(ValueError, match="unknown components"):
        score(cohort, rows)


def test_empty_cohort_rejected():
    with pytest.raises(ValueError, match="empty"):
        single_seed_slots({"pcus": []}, {"sessions": {}})


@pytest.mark.parametrize("field,value", [("primary_layer", "D"), ("primary_layer", []),
                                       ("arms", {"R0": {}}), ("pcu_id", "")])
def test_invalid_pcu_shape_rejected(cohort, field, value):
    cohort[0]["pcus"][0][field] = value
    with pytest.raises(ValueError):
        single_seed_slots(*cohort)


@pytest.mark.parametrize("field,value", [("pcu_id", "unknown"), ("arm_id", []),
                                       ("seed", True), ("seed", "1701")])
def test_bad_selected_session_rejected(cohort, field, value):
    cohort[1]["sessions"]["p0-R0-1701"][field] = value
    with pytest.raises(ValueError):
        single_seed_slots(*cohort)
