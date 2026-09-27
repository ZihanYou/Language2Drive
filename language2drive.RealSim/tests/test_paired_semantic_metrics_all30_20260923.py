import pytest

from language2drive_realsim.paired_semantic_metrics_all30 import (
    _interval,
    _macro_summary,
    score_candidate,
    score_pair,
)


SCOPE = {
    "type": "Polygon",
    "coordinates": [[[-1, -10], [20, -10], [20, 10], [-1, 10], [-1, -10]]],
}
LANE0 = {
    "type": "Polygon",
    "coordinates": [[[-1, -2], [20, -2], [20, 0], [-1, 0], [-1, -2]]],
}
LANE1 = {
    "type": "Polygon",
    "coordinates": [[[-1, 0], [20, 0], [20, 2], [-1, 2], [-1, 0]]],
}


def speed_pcu():
    def arm(lower, upper):
        return {"semantic_contract": {
            "kind": "speed_band", "scope": "scope",
            "speed_min_mps": lower, "speed_max_mps": upper, "deadline_s": 2.0,
        }}
    return {
        "pcu_id": "p", "task_family": "B1_segment_speed_band",
        "decision_window": {"start_s": 1.0, "deadline_s": 2.0},
        "arms": {"R0": arm(1.0, 3.0), "R1": arm(9.0, 11.0)},
        "private_annotations": {"scopes": {"scope": SCOPE}, "lanes": {}},
    }


def gate_pcu():
    gate0 = "gate0"
    gate1 = "gate1"
    return {
        "pcu_id": "g", "task_family": "A2_straight_turn",
        "decision_window": {
            "start_s": 1.0, "deadline_s": 2.0,
            "commitment_gates": [gate0, gate1],
        },
        "arms": {
            "R0": {"semantic_contract": {
                "kind": "gate", "target_gate": gate0,
                "opposing_gate": gate1, "deadline_s": 2.0,
            }},
            "R1": {"semantic_contract": {
                "kind": "gate", "target_gate": gate1,
                "opposing_gate": gate0, "deadline_s": 2.0,
            }},
        },
        "private_annotations": {
            "gates": {
                gate0: [[4.0, -2.0], [4.0, -0.1]],
                gate1: [[4.0, 0.1], [4.0, 2.0]],
            },
            "regions": {}, "lanes": {},
        },
    }


def trace(speed, y=0.0, progress=8.0):
    return [
        {
            "time_s": time,
            "position": [2.5 * time, y * time / 2.0],
            "speed_mps": speed,
            "progress_m": progress * time / 2.0,
            "collision": False,
            "offroad": False,
            "render_valid": True,
        }
        for time in (0.0, 1.0, 2.0)
    ]


def test_candidate_can_cross_score_either_instruction_on_one_trajectory():
    rows = trace(2.0, y=-1.0)
    assert score_candidate(speed_pcu(), "R0", rows)["compliance"] == 1.0
    assert score_candidate(speed_pcu(), "R1", rows)["compliance"] == 0.0


def test_gate_pair_is_included_without_inventing_target_regions():
    result = score_pair(
        gate_pcu(), trace(5.0, y=-1.0), trace(5.0, y=1.0)
    )
    assert result["task_group"] == "route"
    assert result["SCS"] == 1.0
    assert result["PCS"] == 1.0
    assert result["cross_compliance"] == {
        "C0_tau0": 1.0, "C1_tau0": 0.0,
        "C0_tau1": 0.0, "C1_tau1": 1.0,
    }
    assert result["SRG"] > 0.0


def test_pair_identity_is_exact_for_unbalanced_arms():
    result = score_pair(speed_pcu(), trace(2.0), trace(8.0, progress=4.0))
    assert result["SCS"] - result["PCS"] == pytest.approx(
        result["imbalance_half_abs"]
    )
    assert result["identity_residual"] == pytest.approx(0.0)


def test_macro_average_weights_rewrites_inside_pcu_then_pcus_equally():
    def rewrite(value):
        return {
            "evidence_complete": True,
            "SCS": value,
            "intervals": {"SCS": _interval(value, value, value)},
        }
    details = [
        {"rewrites": [rewrite(0.0), rewrite(1.0)]},
        {"rewrites": [rewrite(1.0)]},
    ]
    result = _macro_summary(details, "SCS")
    assert result["percent"] == pytest.approx(75.0)
    assert result["aggregation"] == "equal_rewrites_within_each_PCU_then_equal_PCUs"
