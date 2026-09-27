from copy import deepcopy

import pytest

from language2drive_realsim.compiler import compile_source
from language2drive_realsim.pairing import LeakageError, assert_paired_inputs
from language2drive_realsim.schema import SchemaError, validate_source

from .factories import make_release_source


def test_exact_release_distribution_is_valid():
    source = make_release_source()
    validate_source(source, require_exact_release=True)
    assert len(source["pcus"]) == 50


def test_scene_absolute_time_and_runtime_offset_must_agree():
    source = make_release_source()
    source["pcus"][0]["scene"]["start_time_offset_us"] += 1
    with pytest.raises(SchemaError, match="recording start \\+ offset"):
        validate_source(source)


def test_source_must_bind_the_pinned_alpasim_commit():
    source = make_release_source()
    source["pcus"][0]["backend"]["alpasim_commit"] = "0" * 40
    with pytest.raises(SchemaError, match="pinned commit"):
        validate_source(source)


def test_signature_sets_must_be_disjoint():
    source = make_release_source()
    source["pcus"][0]["arms"]["R1"]["semantic_contract"]["signature_set"] = source["pcus"][0]["arms"]["R0"]["semantic_contract"]["signature_set"]
    source["pcus"][0]["arms"]["R0"]["opposing_contract"] = deepcopy(source["pcus"][0]["arms"]["R1"]["semantic_contract"])
    with pytest.raises(SchemaError, match="not disjoint"):
        validate_source(source)


def test_arm_instruction_times_must_match():
    source = make_release_source()
    source["pcus"][0]["arms"]["R1"]["instruction_schedule"][0][
        "issue_at_us"
    ] += 1
    with pytest.raises(SchemaError, match="issue times must be identical"):
        validate_source(source)


def test_release_requires_all_seed_oracle_plans():
    source = make_release_source()
    del source["pcus"][0]["oracle_plans_by_seed"]
    validate_source(source)
    with pytest.raises(SchemaError, match="missing frozen per-seed oracle plans"):
        validate_source(source, require_exact_release=True)


def test_oracle_arms_require_an_identical_preintervention_prefix():
    source = make_release_source()
    pcu = source["pcus"][0]
    for arm_id in ("R0", "R1"):
        pcu["arms"][arm_id]["instruction_schedule"][0]["issue_at_us"] = 1_500_000
        pcu["oracle_plans_by_seed"]["1701"][arm_id]["waypoints"].insert(
            1,
            {
                "time_s": 0.5,
                "x": 0.5,
                "y": 0.0 if arm_id == "R0" else 0.1,
                "z": 0.0,
                "yaw_rad": 0.0,
            },
        )
    with pytest.raises(SchemaError, match="identical pre-intervention prefix"):
        validate_source(source)


def test_executable_gate_targets_must_be_distinct():
    source = make_release_source()
    pcu = source["pcus"][0]
    first = pcu["arms"]["R0"]["semantic_contract"]
    second = pcu["arms"]["R1"]["semantic_contract"]
    second["target_gate"] = first["target_gate"]
    pcu["arms"]["R0"]["opposing_contract"] = deepcopy(second)
    with pytest.raises(SchemaError, match="gate targets must be distinct"):
        validate_source(source)


def test_task_family_must_match_the_executable_contract_kind():
    source = make_release_source()
    source["pcus"][0]["task_family"] = "A4_lane_change_left_right"
    with pytest.raises(SchemaError, match="requires contract kinds"):
        validate_source(source)


def test_distinct_gate_ids_cannot_resolve_to_identical_geometry():
    source = make_release_source()
    pcu = source["pcus"][0]
    first = pcu["arms"]["R0"]["semantic_contract"]["target_gate"]
    second = pcu["arms"]["R1"]["semantic_contract"]["target_gate"]
    pcu["private_annotations"]["gates"][second] = deepcopy(
        pcu["private_annotations"]["gates"][first]
    )
    with pytest.raises(SchemaError, match="identical geometry"):
        validate_source(source)


def test_executable_speed_bands_must_not_overlap():
    source = make_release_source()
    pcu = next(row for row in source["pcus"] if row["task_family"].startswith("B1"))
    first = pcu["arms"]["R0"]["semantic_contract"]
    second = pcu["arms"]["R1"]["semantic_contract"]
    second["speed_min_mps"] = first["speed_max_mps"]
    pcu["arms"]["R0"]["opposing_contract"] = deepcopy(second)
    with pytest.raises(SchemaError, match="speed bands must be non-overlapping"):
        validate_source(source)


def test_compiler_makes_six_runs_per_pcu_without_public_private_ids():
    public, sidecar, audit = compile_source(make_release_source())
    assert len(public["runs"]) == 300
    assert len(sidecar["sessions"]) == 300
    assert audit["pass"] is True
    assert len(audit["pairs"]) == 150
    for row in public["runs"]:
        assert "pcu_id" not in row
        assert "arm_id" not in row
        assert "semantic_contract" not in row
        assert row["command"] == "UNKNOWN"
        assert row["route_use_waypoint_commands"] is False
    assert all(session["oracle_plan"] for session in sidecar["sessions"].values())


def test_runtime_pair_assertion_rejects_arm_route_change():
    public, sidecar, audit = compile_source(make_release_source())
    first = dict(public["runs"][0])
    second = dict(public["runs"][1])
    second["route"] = [[0.0, 0.0], [0.0, 25.0]]
    with pytest.raises(LeakageError, match="route"):
        assert_paired_inputs(first, second)
