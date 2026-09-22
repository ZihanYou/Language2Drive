from __future__ import annotations

import math

import pytest

from language2drive.core_driving_score import driving_score_percent, valid_driving_score
from language2drive.metrics import _leaderboard_summary


def test_driving_score_uses_furthest_route_progress_not_last_frame():
    rows = [{"progress": 0.1}, {"progress": 0.7}, {"progress": 0.4}]
    assert driving_score_percent(rows) == pytest.approx(70.0)
    assert driving_score_percent(rows, hold_target=True) is None
    assert driving_score_percent([{"progress": math.nan}, {}, {"progress": -1}]) is None
    assert driving_score_percent([{"progress": 0.0}]) == 0.0


def test_driving_score_validation():
    assert valid_driving_score(0.0)
    assert valid_driving_score(100.0)
    for value in (None, True, -0.1, 100.1, math.nan, math.inf, "70"):
        assert not valid_driving_score(value)


def test_leaderboard_driving_score_mean_keeps_binary_progress_unchanged():
    rows = []
    for layer, progress, score in (("A", True, 10.0), ("A", True, 70.0), ("B", False, 0.0)):
        rows.append({
            "layer": layer, "pcu_id": f"{layer}-{len(rows)}", "arm_id": "left",
            "paraphrase": 0, "seed": 1, "result_valid": True,
            "success": False, "semantic_success": False, "safe_legal": True,
            "progress": progress, "driving_score": score,
        })
    summary = _leaderboard_summary(rows, {})
    assert summary["Progress"] == pytest.approx(2 / 3)
    assert summary["DrivingScore"] == pytest.approx(80 / 3)
    assert summary["DrivingScore_by_layer"]["A"] == pytest.approx(40.0)
    assert summary["DrivingScore_by_layer"]["B"] == 0.0
    assert summary["DrivingScore_by_layer"]["C"] is None
    assert summary["DrivingScore_valid_runs"] == 3
    assert summary["DrivingScore_valid_runs_by_layer"] == {"A": 2, "B": 1, "C": 0, "D": 0}
