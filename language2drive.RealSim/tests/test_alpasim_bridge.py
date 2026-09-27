from dataclasses import dataclass

from types import SimpleNamespace
from pathlib import Path

import pytest

from language2drive_realsim.alpasim_bridge import (
    _observation_record,
    _preintervention_evidence,
    normalize_simulation_result,
    validate_asl_authority,
)
from language2drive_realsim.constants import ALPASIM_COMMIT, ORACLE_CONTROLLER_ID
from language2drive_realsim.io import file_hash
from language2drive_realsim.qualification import _authority_check


@pytest.mark.parametrize("overrides", [
    {"diagnostic_without_map": 1},
    {"diagnostic_without_map": True},
    {"diagnostic_without_map": True, "diagnostic_runtime_metrics": Path("metrics.parquet")},
    {"diagnostic_without_map": True, "diagnostic_runtime_metrics": Path("metrics.parquet"),
     "diagnostic_run_uuid": "run"},
])
def test_map_free_mode_cannot_replace_formal_conversion(overrides):
    from language2drive_realsim.alpasim_bridge import _load_alpasim_result
    # Reject before attempting imports or reading nonexistent files.
    with pytest.raises(ValueError, match="map-free diagnostics require"):
        _load_alpasim_result(Path("missing.asl"), Path("missing.yaml"), "missing.usdz", **overrides)


@pytest.mark.parametrize("kind", ["driver_camera_image", "driver_ego_trajectory", "route_request"])
def test_observation_hash_skips_after_cutoff_and_keeps_boundary(monkeypatch, kind):
    calls = []
    monkeypatch.setattr("language2drive_realsim.alpasim_bridge._protobuf_payload_hash",
                        lambda request: calls.append(request) or "payload")
    request = SimpleNamespace(
        camera_image=SimpleNamespace(frame_end_us=10, logical_id="front"),
        trajectory=SimpleNamespace(poses=[SimpleNamespace(timestamp_us=9), SimpleNamespace(timestamp_us=10)]),
        route=SimpleNamespace(timestamp_us=10),
    )
    message = SimpleNamespace(WhichOneof=lambda _: kind, **{kind: request})
    assert _observation_record(message, 9) is None
    assert calls == []  # No serialization or hashing of excluded observations.
    assert _observation_record(message, 10) == _observation_record(message)
    assert len(calls) == 2


def test_cutoff_preserves_existing_prefix_digest(monkeypatch):
    monkeypatch.setattr("language2drive_realsim.alpasim_bridge._protobuf_payload_hash", lambda _: "payload")
    messages = []
    for timestamp in (9, 10, 11):
        for kind in ("driver_camera_image", "driver_ego_trajectory", "route_request"):
            request = SimpleNamespace(
                camera_image=SimpleNamespace(frame_end_us=timestamp, logical_id="front"),
                trajectory=SimpleNamespace(poses=[SimpleNamespace(timestamp_us=timestamp)]),
                route=SimpleNamespace(timestamp_us=timestamp),
            )
            messages.append(SimpleNamespace(WhichOneof=lambda _, k=kind: k, **{kind: request}))
    full = [_observation_record(message) for message in messages]
    bounded = [row for message in messages if (row := _observation_record(message, 10)) is not None]
    assert len(full) == 9 and len(bounded) == 6
    assert _preintervention_evidence(full, 10, ["front"]) == _preintervention_evidence(bounded, 10, ["front"])


def test_preintervention_camera_contract_accepts_frozen_model_specific_set():
    records = [
        {"kind": "camera", "timestamp_us": 10, "logical_id": camera, "payload_sha256": camera}
        for camera in ("front", "rear", "left")
    ] + [{"kind": "egomotion", "timestamp_us": 10, "payload_sha256": "ego"}]
    evidence = _preintervention_evidence(records, 10, ["rear", "front", "left"])
    assert evidence["preintervention_camera_logical_ids"] == ["front", "left", "rear"]
    with pytest.raises(ValueError, match="frozen model input"):
        _preintervention_evidence(records, 10, ["front"])


class Pose:
    def __init__(self, x, y):
        self.vec3 = (x, y, 0.0)

    def yaw(self):
        return 0.0


class Trajectory:
    def interpolate_pose(self, timestamp_us):
        return Pose((timestamp_us - 1_500_000_000_000) / 1_000_000.0, 0.0)


class StaticActorTrajectory:
    def interpolate_pose(self, timestamp_us):
        return Pose(1.0, -2.0)


class Polygon:
    def distance(self, other):
        return 1.25


class ActorPolygons:
    def get_polygon_for_agent_at_time(self, actor_id, timestamp_us):
        return Polygon()


class Result:
    timestamps_us = [1_500_000_000_000, 1_500_001_000_000, 1_500_002_000_000]
    session_metadata = SimpleNamespace(start_timestamp_us=1_500_000_000_000)
    actor_trajectories = {"EGO": Trajectory()}


@dataclass
class Metric:
    name: str
    timestamps_us: list
    values: list
    valid: list


def test_bridge_uses_alpasim_metrics_and_reconstruction_support():
    metrics = [
        Metric("collision_any", Result.timestamps_us, [0, 0, 1], [True] * 3),
        Metric("offroad", Result.timestamps_us, [0, 0, 0], [True] * 3),
        Metric("img_is_black", Result.timestamps_us, [0, 0, 0], [True] * 3),
    ]
    annotations = {
        "reconstruction_support_polygon": [[-1, -1], [1.5, -1], [1.5, 1], [-1, 1]],
        "lanes": {"lane-0000": [[-1, -1], [3, -1], [3, 1], [-1, 1]]},
        "regions": {},
        "scopes": {},
        "triggers": {},
        "actor_ids": [],
    }
    rows = normalize_simulation_result(Result(), metrics, annotations)
    assert [row["time_s"] for row in rows] == [0.0, 1.0, 2.0]
    assert rows[0]["timestamp_us"] == 1_500_000_000_000
    assert rows[1]["speed_mps"] == 1.0
    assert rows[1]["progress_m"] == 1.0
    assert rows[1]["lane_ids"] == ["lane-0000"]
    assert rows[1]["render_valid"] is True
    assert rows[1]["image_black"] is False
    assert rows[2]["collision"] is True
    assert rows[2]["reconstruction_support_valid"] is False
    assert rows[2]["reconstruction_deviation_m"] == pytest.approx(0.5)
    assert rows[2]["render_valid"] is False


def test_bridge_exposes_completed_pass_state_and_polygon_clearance():
    result = Result()
    result.actor_trajectories = {
        "EGO": Trajectory(),
        "actor-0000": StaticActorTrajectory(),
    }
    result.actor_polygons = ActorPolygons()
    annotations = {
        "reconstruction_support_polygon": [[-1, -3], [3, -3], [3, 3], [-1, 3]],
        "lanes": {},
        "regions": {},
        "scopes": {},
        "triggers": {},
        "actor_ids": ["actor-0000"],
    }
    metrics = [Metric(name, Result.timestamps_us, [0] * 3, [True] * 3)
               for name in ("collision_any", "offroad", "img_is_black")]
    rows = normalize_simulation_result(result, metrics, annotations)

    assert [row["relative_longitudinal_m"]["actor-0000"] for row in rows] == [
        1.0,
        0.0,
        -1.0,
    ]
    assert all(row["relative_side"]["actor-0000"] == "LEFT" for row in rows)
    assert all(row["clearance_m"]["actor-0000"] == 1.25 for row in rows)


@pytest.mark.parametrize("fault", ["missing", "gap", "invalid", "nan", "infinity", "length", "duplicate_name", "duplicate_time", "non_binary"])
def test_bridge_rejects_missing_or_invalid_safety_evidence(fault):
    metrics = [Metric(name, list(Result.timestamps_us), [0] * 3, [True] * 3)
               for name in ("collision_any", "offroad", "img_is_black")]
    metric = metrics[-1]
    if fault == "missing":
        metrics.pop()
    elif fault == "gap":
        metric.timestamps_us.pop()
        metric.values.pop()
        metric.valid.pop()
    elif fault == "invalid":
        metric.valid[1] = False
    elif fault == "nan":
        metric.values[1] = float("nan")
    elif fault == "infinity":
        metric.values[1] = float("inf")
    elif fault == "length":
        metric.values.pop()
    elif fault == "duplicate_name":
        metrics.append(metric)
    elif fault == "duplicate_time":
        metric.timestamps_us[1] = metric.timestamps_us[0]
    elif fault == "non_binary":
        metric.values[1] = -1
    with pytest.raises(ValueError, match="safety metric"):
        normalize_simulation_result(Result(), metrics, {})


def test_asl_authority_binds_real_scene_start_and_seed():
    metadata = SimpleNamespace(
        session_uuid="opaque-session",
        scene_id="clipgt-scene",
        nre_uuid="artifact-uuid",
        start_timestamp_us=1_500_000,
    )
    driver = {
        "session_uuid": "opaque-session",
        "scene_id": "clipgt-scene",
        "random_seed": 1701,
    }
    session = {
        "scene_id_private": "clipgt-scene",
        "artifact_id": "artifact-uuid",
        "start_timestamp_us": 1_500_000,
        "seed": 1701,
    }
    assert validate_asl_authority(metadata, driver, session)["random_seed"] == 1701

    wrong_seed = dict(driver, random_seed=1702)
    with pytest.raises(ValueError, match="frozen sidecar"):
        validate_asl_authority(metadata, wrong_seed, session)


def test_oracle_receipt_requires_the_distinct_privileged_driver(tmp_path):
    source_asl = tmp_path / "rollout.asl"
    trace = tmp_path / "opaque-session.jsonl"
    source_asl.write_bytes(b"asl")
    trace.write_text("{}\n", encoding="utf-8")
    session = {
        "artifact_id": "artifact-uuid",
        "scene_id_private": "clipgt-scene",
        "start_timestamp_us": 1_500_000,
        "seed": 1701,
    }
    receipt = {
        "schema_version": "l2d.realsim.asl_receipt.v0.1",
        "synthetic": False,
        "session_uuid": "opaque-session",
        "artifact_id": "artifact-uuid",
        "scene_id_private": "clipgt-scene",
        "start_timestamp_us": 1_500_000,
        "random_seed": 1701,
        "alpasim_commit": ALPASIM_COMMIT,
        "egodriver_version_id": "l2d_alpamayo1_5-driver-0.91.0",
        "oracle_controller": None,
        "source_asl": str(source_asl),
        "source_asl_sha256": file_hash(source_asl),
        "normalized_trace_sha256": file_hash(trace),
    }
    errors = _authority_check(trace, receipt, session, ALPASIM_COMMIT)
    assert "evidence_not_generated_by_privileged_oracle_driver" in errors
    assert "oracle_controller_authority_missing" in errors

    receipt["egodriver_version_id"] = "l2d_privileged_oracle-driver-0.91.0"
    receipt["oracle_controller"] = ORACLE_CONTROLLER_ID
    assert _authority_check(trace, receipt, session, ALPASIM_COMMIT) == []
