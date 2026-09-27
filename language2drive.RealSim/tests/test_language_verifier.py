import pytest

from language2drive_realsim.compiler import compile_source
from language2drive_realsim.events import segments_intersect, validate_trace
from language2drive_realsim.language import SidecarLanguageConditionProvider
from language2drive_realsim.verifier import verify_trace
from language2drive_realsim.verifier_selftest import run_verifier_selftests

from .factories import basic_trace, make_pcu, make_release_source


@pytest.mark.parametrize("family", ["A6_pull_over_continue", "B5_relative_stopping_region"])
@pytest.mark.parametrize("interruption", ["scope", "region", "speed"])
def test_stop_dwell_requires_continuous_scoped_samples(family, interruption):
    pcu = make_pcu(1, family)
    contract = pcu["arms"]["R0"]["semantic_contract"]
    contract["dwell_s"] = 2.0
    trace = basic_trace()
    for index in (3, 4, 5):
        trace[index].update(scopes=[contract["scope"]], regions=[contract["region"]], speed_mps=0.0)
    if interruption == "scope":
        trace[4]["scopes"] = []
    elif interruption == "region":
        trace[4]["regions"] = []
    else:
        trace[4]["speed_mps"] = 1.0
    result = verify_trace(pcu, "R0", trace)
    assert result.failure == "F1_NO_COMMIT"


@pytest.mark.parametrize("family,arm", [("A6_pull_over_continue", "R1"), ("B5_relative_stopping_region", "R0")])
def test_disjoint_opposing_stop_does_not_reject_valid_commitment(family, arm):
    pcu = make_pcu(1, family)
    contract = pcu["arms"][arm]["semantic_contract"]
    opposing = pcu["arms"][arm]["opposing_contract"]
    opposing["dwell_s"] = 2.0
    trace = basic_trace()
    for index in (2, 4):
        trace[index].update(scopes=[contract["scope"]], regions=[opposing["region"]], speed_mps=0.0)
    trace[3].update(regions=[opposing["region"]], speed_mps=0.0)
    for index in (6, 7):
        trace[index]["scopes"] = [contract["scope"]]
        if contract["kind"] == "stop_region":
            trace[index].update(regions=[contract["region"]], speed_mps=0.0)
    if contract["kind"] == "continue_region":
        trace[7]["gates_crossed"] = [contract["target_gate"]]
    result = verify_trace(pcu, arm, trace)
    assert result.passed is True
    assert result.first_commitment["time_s"] == 7.0


def test_stop_dwell_restarts_after_scope_reentry():
    pcu = make_pcu(1, "A6_pull_over_continue")
    contract = pcu["arms"]["R0"]["semantic_contract"]
    contract["dwell_s"] = 2.0
    trace = basic_trace()
    for index in range(2, 7):
        trace[index].update(scopes=[contract["scope"]], regions=[contract["region"]], speed_mps=0.0)
    trace[3]["scopes"] = []
    result = verify_trace(pcu, "R0", trace)
    assert result.passed is True
    assert result.first_commitment["time_s"] == 6.0


def test_gate_segment_intersection_rejects_disjoint_collinear_segments():
    assert segments_intersect([0, 0], [1, 0], [2, 0], [3, 0]) is False
    assert segments_intersect([0, 0], [2, 0], [1, -1], [1, 1]) is True
    assert segments_intersect([0, 0], [1, 0], [1, 0], [2, 1]) is True


def test_sidecar_exposes_only_snapshot_payload():
    public, sidecar, _ = compile_source(make_release_source())
    session_uuid = public["runs"][0]["session_uuid"]
    provider = SidecarLanguageConditionProvider(sidecar)
    before = provider.get_instruction(session_uuid, 999999)
    after = provider.get_instruction(session_uuid, 1000000)
    assert before.text == ""
    payload = provider.model_payload(after)
    assert set(payload) == {"text", "version", "issued_at_us"}
    assert "pcu_id" not in payload and "arm_id" not in payload


def test_adversarial_verifier_matrix_passes():
    report = run_verifier_selftests()
    assert report["pass"] is True
    required = {
        "positive", "opposing", "wrong_first", "late", "partial",
        "oscillation", "no_commit", "always_stop", "slow_creep",
        "trigger_avoidance", "collision", "offroad", "edge_pass", "edge_fail",
        "preintervention_collision", "invalid_render", "infrastructure_failure",
    }
    for implementation in report["implementations"].values():
        assert required <= {row["name"] for row in implementation["cases"]}
    assert report["layer_c_wrong_referent"]["actual"]["failure"] == (
        "F4_WRONG_REFERENT_OUTCOME"
    )


def test_layer_c_wrong_gate_is_referent_failure():
    pcu = make_pcu(1, "C1_direction_sign_grounding")
    trace = basic_trace()
    wrong_gate = pcu["arms"]["R1"]["semantic_contract"]["target_gate"]
    trace[3]["gates_crossed"] = [wrong_gate]
    result = verify_trace(pcu, "R0", trace)
    assert result.passed is False
    assert result.failure == "F4_WRONG_REFERENT_OUTCOME"


def test_wrong_first_event_cannot_be_repaired_later():
    pcu = make_pcu(1, "A1_intersection_left_right")
    trace = basic_trace()
    trace[3]["gates_crossed"] = [pcu["arms"]["R1"]["semantic_contract"]["target_gate"]]
    trace[4]["gates_crossed"] = [pcu["arms"]["R0"]["semantic_contract"]["target_gate"]]
    result = verify_trace(pcu, "R0", trace)
    assert result.failure == "F2_WRONG_SEMANTIC"


def test_trace_gap_is_invalid_sim_under_frozen_sampling_threshold():
    pcu = make_pcu(1, "A1_intersection_left_right")
    trace = basic_trace()
    result = verify_trace(
        pcu,
        "R0",
        trace,
        threshold_profile={"maximum_sample_gap_s": 0.25},
    )
    assert result.passed is False
    assert result.failure == "INVALID_SIM"
    assert result.diagnostics["reason"] == "trace_sample_gap"


def test_trace_validation_rejects_nan_and_nonmonotonic_progress():
    trace = basic_trace()
    trace[2]["speed_mps"] = float("nan")
    try:
        validate_trace(trace)
    except ValueError as error:
        assert "speed_mps" in str(error)
    else:
        raise AssertionError("NaN speed was accepted")

    trace = basic_trace()
    trace[2]["progress_m"] = trace[1]["progress_m"] - 0.1
    try:
        validate_trace(trace)
    except ValueError as error:
        assert "progress must be monotonic" in str(error)
    else:
        raise AssertionError("decreasing progress was accepted")
