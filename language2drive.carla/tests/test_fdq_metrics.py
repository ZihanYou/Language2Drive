from __future__ import annotations

import json
import math
from pathlib import Path

from language2drive.fdq_metrics import (
    POLICY_SHA256,
    score_fdq_metrics,
    validate_fdq_metrics,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, record_status: str = "Completed") -> tuple[dict, Path]:
    run = {"run_id": "fdq__overtaking__s1", "ability": "overtaking", "seed": 1}
    result = {
        "schema_version": "language2drive.fdq_result.v1",
        "status": "complete",
        "success": False,
        "capture_complete": True,
        **run,
        "route_id": "24240",
        "town": "Town10HD",
        "scenario": "Accident",
        "record_status": record_status,
        "DS": 27.3,
        "RC": 50.0,
        "IP": 0.273,
        "infraction_free": False,
        "source": "unused",
        "source_sha256": "0" * 64,
        "checkpoint_record_sha256": "1" * 64,
    }
    _write_json(tmp_path / "fdq_result.json", result)
    _write_json(tmp_path / "bench2drive_result.json", {
        "_checkpoint": {"records": [{
            "status": record_status,
            "meta": {"duration_game": 10.0},
            "infractions": {
                "collisions_pedestrian": [],
                "collisions_vehicle": ["vehicle"],
                "collisions_layout": ["layout"],
            },
        }]},
    })
    _write_json(tmp_path / "capture_capabilities.json", {
        "controller_source_samples": 5,
        "fdq_metric_observation": {
            "schema_version": "language2drive.fdq_observation.v1",
            "frame_samples": 5,
            "ttc_supported": True,
        },
    })
    rows = []
    for index, (speed, brake, ttc) in enumerate((
        (2.0, 0.0, None),
        (2.0, 0.6, 1.5),
        (1.0, 0.0, 2.0),
        (2.0, 0.0, 3.5),
        (2.0, 0.0, None),
    )):
        rows.append({
            "frame": index,
            "sim_time": float(index),
            "ego": {
                "speed_mps": speed,
                "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "forward_vector": {"x": 1.0, "y": 0.0, "z": 0.0},
                "right_vector": {"x": 0.0, "y": 1.0, "z": 0.0},
            },
            "control": {"brake": brake},
            "fdq_observation": {
                "schema_version": "language2drive.fdq_observation.v1",
                "route_progress": index / 4,
                "relevant_actor_count": 1,
                "min_ttc_seconds": ttc,
                "map_legal": True,
            },
        })
    (tmp_path / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run, tmp_path


def test_fdq_supplemental_metrics_keep_rc_and_score_all_signals(tmp_path: Path) -> None:
    run, run_dir = _fixture(tmp_path)
    reference = tmp_path / "reference.json"
    _write_json(reference, {
        "schema_version": "language2drive.fdq_reference_profiles.v1",
        "routes": {
            "24240": {
                "reference_policy": "test-expert-v1",
                "samples": [
                    {"progress": 0.0, "time_seconds": 0.1},
                    {"progress": 0.5, "time_seconds": 5.0},
                    {"progress": 1.0, "time_seconds": 10.0},
                ],
            },
        },
    })
    value = score_fdq_metrics(run, run_dir, reference_profiles=reference)
    assert validate_fdq_metrics(value, run) is True
    assert value["policy_sha256"] == POLICY_SHA256
    assert value["rc_unchanged"] is True
    assert value["RC"] == 50.0
    assert value["metrics"]["SI"]["value"] == math.exp(-3.0)
    assert value["metrics"]["GPE"]["value"] == 0.5 * math.exp(-0.5)
    assert value["metrics"]["PSM"]["value"] == (1 + 0.5 + 2 / 3 + 1 + 1) / 5
    assert value["metrics"]["MDQ"]["value"] == 1.0
    assert value["metrics"]["RLS"]["status"] == "available"
    assert value["metrics"]["RLS"]["opportunity_count"] == 1


def test_fdq_missing_capture_and_reference_are_explicitly_unavailable(tmp_path: Path) -> None:
    run, run_dir = _fixture(tmp_path)
    _write_json(run_dir / "capture_capabilities.json", {"controller_source_samples": 5})
    value = score_fdq_metrics(
        run, run_dir, reference_profiles=tmp_path / "absent-reference.json"
    )
    assert value["metrics"]["GPE"]["status"] == "unavailable"
    assert value["metrics"]["PSM"]["status"] == "unavailable"
    assert value["metrics"]["RLS"]["status"] == "unavailable"
    assert value["metrics"]["SI"]["status"] == "available"
    assert value["metrics"]["MDQ"]["status"] == "available"
    assert validate_fdq_metrics(value, run) is True


def test_fdq_zero_progress_gpe_is_exact_zero_without_positive_reference_time(
    tmp_path: Path,
) -> None:
    run, run_dir = _fixture(tmp_path)
    result_path = run_dir / "fdq_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["RC"] = 0.0
    _write_json(result_path, result)
    reference = tmp_path / "reference.json"
    _write_json(reference, {
        "schema_version": "language2drive.fdq_reference_profiles.v1",
        "routes": {
            "24240": {
                "reference_policy": "test-expert-v1",
                "samples": [
                    {"progress": 0.0, "time_seconds": 0.0},
                    {"progress": 1.0, "time_seconds": 10.0},
                ],
            },
        },
    })

    value = score_fdq_metrics(run, run_dir, reference_profiles=reference)

    assert value["metrics"]["GPE"]["status"] == "available"
    assert value["metrics"]["GPE"]["value"] == 0.0
    assert value["metrics"]["GPE"]["zero_progress_identity"] is True
    assert validate_fdq_metrics(value, run) is True
