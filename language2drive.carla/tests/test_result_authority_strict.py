from __future__ import annotations

import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import pytest

import language2drive.scoring as scoring_module
from language2drive.io import atomic_json, atomic_jsonl, stable_hash
from language2drive.metrics import _read_result, aggregate
from language2drive.report import (
    _current_release_identity_gate, _exact_baseline_matrix,
    _exact_denominator, _exact_fdq_matrix,
)
from language2drive.scoring import (
    _fdq_summary, _materialize_json_object, _qualification_status,
    _strict_release_ledger, _validate_baseline_manifest,
    _validate_exact_release_ledgers, _verify_materialized_release_ledgers,
    result_snapshot,
)
from language2drive.fdq import FDQ_ROUTE_BINDING_ROWS, FDQ_SEEDS


SHA = "a" * 64


def _full_bundle(*, workload: str = "core") -> dict:
    value = {
        "schema_version": "language2drive.execution_bundle.v2",
        "route_xml_sha256": SHA,
        "agent_sha256": SHA,
        "agent_config_sha256": SHA,
        "evaluator_sha256": SHA,
        "workload": workload,
        "benchmark_track": "fdq" if workload == "fdq" else "standardized",
        "evaluator_track": "native" if workload == "fdq" else "standardized",
        "dependency_profile": "autovla-runtime.v1",
        "dependency_hashes": {"module.py": SHA},
        "dependency_manifest_sha256": SHA,
        "execution_environment_sha256": SHA,
        "storage_policy_sha256": SHA,
        "wall_timeout_policy_sha256": SHA,
        "carla_runtime_root": "/frozen/carla",
        "carla_runtime_asset_manifest_sha256": SHA,
        "carla_runtime_asset_inventory_sha256": SHA,
        "carla_runtime_asset_files_sha256": SHA,
        "carla_runtime_asset_file_count": 1,
        "carla_runtime_asset_total_size_bytes": 1,
        "carla_runtime_shipping_binary_sha256": SHA,
        "carla_runtime_asset_provenance_sha256": SHA,
        "carla_runtime_available_maps_sha256": SHA,
        "carla_runtime_available_map_count": 1,
        "carla_runtime_client_version": "0.9.14",
        "carla_runtime_server_version": "0.9.14",
        "carla_runtime_boot_id": "boot",
        "carla_runtime_freeze_monotonic_ticks": 1,
        "carla_runtime_clock_ticks_per_second": 100,
        "carla_runtime_server_readonly_permissions_sha256": SHA,
        "carla_runtime_server_readonly_permission_entry_count": 1,
        "carla_runtime_server_uid": 1000,
        "carla_runtime_server_gids": [1000],
        "carla_runtime_asset_validator_sha256": SHA,
        "carla_server_process_receipt": {"pid": 123},
        "carla_server_process_receipt_sha256": SHA,
    }
    if workload != "fdq":
        value["private_sidecar_sha256"] = SHA
    value["dependency_manifest_sha256"] = stable_hash(value["dependency_hashes"])
    value["carla_server_process_receipt_sha256"] = stable_hash(
        value["carla_server_process_receipt"]
    )
    return value


def _core_run() -> dict:
    return {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "pcu_id": "A-001", "arm_id": "left",
        "paraphrase": 0, "seed": 1, "route_entry_id": 7,
    }


def _core_result(run: dict, *, success: bool = True) -> dict:
    return {
        "schema_version": "language2drive.contract_result.v1",
        "status": "complete", "success": success, "capture_complete": True,
        "own_contract": success, "opposite_contract": False,
        "safe_legal": success, "progress": success, "deadline": success,
        "stability": success, "scorer": "language2drive.v1.offline_contract_engine",
        **{key: run[key] for key in (
            "run_id", "pcu_id", "arm_id", "paraphrase", "seed", "track",
            "route_entry_id",
        )},
    }


def _write_core(root: Path, run: dict, bundle: dict, result: dict | None = None) -> Path:
    run_dir = root / run["run_id"]
    atomic_json(run_dir / "runner_command.json", {"run": run, "input_hashes": bundle})
    atomic_json(run_dir / "contract_result.json", result or _core_result(run))
    atomic_jsonl(run_dir / "trace.jsonl", [{"sim_time": 0.0}, {"sim_time": 1.0}])
    return run_dir


def test_modern_core_result_requires_full_bundle_identity_capture_and_conjunction(tmp_path: Path):
    run, bundle = _core_run(), _full_bundle()
    run_dir = _write_core(tmp_path, run, bundle)
    assert _read_result(run, tmp_path, expected_input_hashes=bundle)["success"] is True

    scored = _core_result(run)
    scored["driving_score"] = 42.5
    atomic_json(run_dir / "contract_result.json", scored)
    assert _read_result(run, tmp_path, expected_input_hashes=bundle)["driving_score"] == 42.5
    scored["driving_score"] = 101.0
    atomic_json(run_dir / "contract_result.json", scored)
    assert _read_result(run, tmp_path, expected_input_hashes=bundle)["status"] == "invalid_result"

    mutations = {
        "missing_capture": lambda value: value.pop("capture_complete"),
        "false_capture": lambda value: value.__setitem__("capture_complete", False),
        "missing_conjunct": lambda value: value.pop("deadline"),
        "identity_swap": lambda value: value.__setitem__("run_id", "other"),
        "success_conflict": lambda value: value.__setitem__("safe_legal", False),
    }
    for mutate in mutations.values():
        candidate = _core_result(run)
        mutate(candidate)
        atomic_json(run_dir / "contract_result.json", candidate)
        row = _read_result(run, tmp_path, expected_input_hashes=bundle)
        assert row["result_valid"] is False
        assert row["success"] is False

    atomic_json(run_dir / "contract_result.json", _core_result(run))
    incomplete = copy.deepcopy(bundle)
    incomplete.pop("dependency_manifest_sha256")
    assert _read_result(run, tmp_path, expected_input_hashes=incomplete)["status"] == "stale_provenance"
    swapped = copy.deepcopy(run)
    swapped["route_entry_id"] = 8
    atomic_json(run_dir / "runner_command.json", {"run": swapped, "input_hashes": bundle})
    assert _read_result(run, tmp_path, expected_input_hashes=bundle)["status"] == "stale_provenance"


def test_duplicate_result_key_and_nonmonotonic_trace_fail_closed(tmp_path: Path):
    run, bundle = _core_run(), _full_bundle()
    run_dir = _write_core(tmp_path, run, bundle)
    text = json.dumps(_core_result(run)).replace('"capture_complete": true', '"capture_complete": true, "capture_complete": true')
    (run_dir / "contract_result.json").write_text(text, encoding="utf-8")
    assert _read_result(run, tmp_path, expected_input_hashes=bundle)["status"] == "invalid_result"

    atomic_json(run_dir / "contract_result.json", _core_result(run))
    atomic_jsonl(run_dir / "trace.jsonl", [{"sim_time": 1.0}, {"sim_time": 0.0}])
    row = _read_result(run, tmp_path, expected_input_hashes=bundle)
    assert row["status"] == "no_trace"
    assert row["success"] is False


def _fdq_run() -> dict:
    return {
        "run_id": "fdq__overtaking__s1", "ability": "overtaking",
        "seed": 1, "status": "expected",
    }


def _fdq_result(run: dict) -> dict:
    return {
        "schema_version": "language2drive.fdq_result.v1",
        "status": "complete", "success": True, "capture_complete": True,
        "run_id": run["run_id"], "ability": run["ability"], "seed": run["seed"],
        "route_id": "24240", "town": "Town10HD", "scenario": "Accident",
        "record_status": "Completed", "DS": 100.0, "RC": 100.0, "IP": 1.0,
        "infraction_free": True, "source": "pending",
        "source_sha256": SHA, "checkpoint_record_sha256": SHA,
    }


def _write_fdq(root: Path, run: dict, bundle: dict, result: dict) -> tuple[Path, Path]:
    ledger = root / "fdq-ledger.json"
    atomic_json(ledger, {"expected_count": 1, "runs": [run]})
    run_dir = root / "results" / run["run_id"]
    binding = {"route_id": "24240", "town": "Town10HD", "scenario": "Accident"}
    record = {
        "route_id": f"RouteScenario_{binding['route_id']}_rep0",
        "town_name": binding["town"], "scenario_name": f"{binding['scenario']}_1",
        "status": "Completed", "num_infractions": 0,
        "scores": {"score_composed": 100.0, "score_route": 100.0, "score_penalty": 1.0},
        "infractions": {key: [] for key in (
            "collisions_layout", "collisions_pedestrian", "collisions_vehicle",
            "red_light", "stop_infraction", "outside_route_lanes",
            "min_speed_infractions", "yield_emergency_vehicle_infractions",
            "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout",
        )},
    }
    checkpoint = run_dir / "bench2drive_result.json"
    atomic_json(checkpoint, {"_checkpoint": {"records": [record]}})
    result.setdefault("source", str(checkpoint.resolve()))
    if result.get("source") == "pending":
        result["source"] = str(checkpoint.resolve())
    result.setdefault("source_sha256", SHA)
    result["source_sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
    result.setdefault("checkpoint_record_sha256", stable_hash(record))
    if result["checkpoint_record_sha256"] == SHA:
        result["checkpoint_record_sha256"] = stable_hash(record)
    atomic_json(run_dir / "fdq_result.json", result)
    atomic_json(run_dir / "runner_command.json", {"run": run, "input_hashes": bundle})
    return ledger, root / "results"


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("status"),
    lambda value: value.__setitem__("success", "true"),
    lambda value: value.update(status="crash", success=True, capture_complete=False),
    lambda value: value.__setitem__("route_id", "wrong"),
    lambda value: value.__setitem__("DS", True),
    lambda value: value.update(driving_score=value.pop("DS")),
])
def test_fdq_summary_rejects_schema_identity_metric_and_alias_mutations(tmp_path: Path, mutation):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    result = _fdq_result(run)
    mutation(result)
    ledger, results = _write_fdq(tmp_path, run, bundle, result)
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["finalized_runs"] == 0
    assert summary["passed_runs"] == 0
    assert summary["invalid_runs"] == 1


def test_fdq_summary_accepts_only_canonical_fdq_result_file(tmp_path: Path):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    ledger, results = _write_fdq(tmp_path, run, bundle, _fdq_result(run))
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["finalized_runs"] == summary["passed_runs"] == 1
    assert summary["DS_RC_IP"]["DS"]["mean_valid_only"] == 100.0

    (results / run["run_id"] / "fdq_result.json").unlink()
    atomic_json(results / run["run_id"] / "contract_result.json", _fdq_result(run))
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["observed_runs"] == 0
    assert summary["missing_runs"] == 1


@pytest.mark.parametrize("record_status", [
    "Completed", "Perfect", "Failed", "Failed - Agent timed out",
    "Failed - Agent deviated from the route", "Failed - Agent got blocked",
])
def test_fdq_record_status_exact_whitelist(tmp_path: Path, record_status: str):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    ledger, results = _write_fdq(tmp_path, run, bundle, _fdq_result(run))
    run_dir = results / run["run_id"]
    checkpoint_path = run_dir / "bench2drive_result.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["_checkpoint"]["records"][0]
    record["status"] = record_status
    atomic_json(checkpoint_path, checkpoint)
    result_path = run_dir / "fdq_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["record_status"] = record_status
    result["success"] = record_status in {"Completed", "Perfect"}
    result["source_sha256"] = sha256(checkpoint_path.read_bytes()).hexdigest()
    result["checkpoint_record_sha256"] = stable_hash(record)
    atomic_json(result_path, result)
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["finalized_runs"] == 1
    assert summary["passed_runs"] == int(result["success"])


def test_fdq_unknown_self_consistent_record_status_is_invalid(tmp_path: Path):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    ledger, results = _write_fdq(tmp_path, run, bundle, _fdq_result(run))
    run_dir = results / run["run_id"]
    checkpoint_path = run_dir / "bench2drive_result.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["_checkpoint"]["records"][0]
    record["status"] = "Banana"
    atomic_json(checkpoint_path, checkpoint)
    result_path = run_dir / "fdq_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update({
        "record_status": "Banana", "success": False,
        "source_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_record_sha256": stable_hash(record),
    })
    atomic_json(result_path, result)
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["finalized_runs"] == 0
    assert summary["invalid_runs"] == 1


def test_fdq_checkpoint_boolean_score_is_invalid_even_if_equal_to_result(tmp_path: Path):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    ledger, results = _write_fdq(tmp_path, run, bundle, _fdq_result(run))
    run_dir = results / run["run_id"]
    checkpoint_path = run_dir / "bench2drive_result.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["_checkpoint"]["records"][0]
    record["scores"]["score_penalty"] = True
    atomic_json(checkpoint_path, checkpoint)
    result_path = run_dir / "fdq_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update({
        "IP": 1.0,
        "source_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_record_sha256": stable_hash(record),
    })
    atomic_json(result_path, result)
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["finalized_runs"] == 0
    assert summary["invalid_runs"] == 1


@pytest.mark.parametrize("invalid_source", [None, ["not", "a", "path"]])
def test_fdq_nonstring_source_is_invalid_not_exception(
    tmp_path: Path, invalid_source,
):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    result = _fdq_result(run)
    result["source"] = invalid_source
    ledger, results = _write_fdq(tmp_path, run, bundle, result)
    result_path = results / run["run_id"] / "fdq_result.json"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    stored["source"] = invalid_source
    atomic_json(result_path, stored)
    summary = _fdq_summary(ledger, results, expected_input_hashes=bundle, expected_count=1)
    assert summary["finalized_runs"] == 0
    assert summary["invalid_runs"] == 1


def test_release_denominators_and_fdq_matrix_cannot_shrink_or_duplicate(tmp_path: Path):
    run, bundle = _fdq_run(), _full_bundle(workload="fdq")
    ledger, results = _write_fdq(tmp_path, run, bundle, _fdq_result(run))
    with pytest.raises(ValueError, match="denominator"):
        _fdq_summary(ledger, results, expected_input_hashes=bundle)
    with pytest.raises(ValueError, match="denominator"):
        _strict_release_ledger(ledger, len(FDQ_ROUTE_BINDING_ROWS) * len(FDQ_SEEDS))

    expected_count = len(FDQ_ROUTE_BINDING_ROWS) * len(FDQ_SEEDS)
    duplicated = [dict(run, seed=((index % 3) + 1), run_id=f"dup-{index}") for index in range(expected_count)]
    duplicated[1]["run_id"] = duplicated[0]["run_id"]
    atomic_json(ledger, {"expected_count": expected_count, "runs": duplicated})
    with pytest.raises(ValueError, match="duplicated|denominator"):
        _strict_release_ledger(ledger, 12)

    assert _qualification_status(
        {"expected_runs": 1, "finalized_runs": 1, "success_count": 1},
        expected_runs=expected_count,
    ) == "invalid_denominator"
    assert _exact_denominator({"expected_runs": expected_count}, expected_count) is True
    shrunk = {
        "expected_runs": 1,
        "ability_groups": {"overtaking": {"expected": 1}},
    }
    assert _exact_fdq_matrix(shrunk) is False
    exact = {
        "expected_runs": expected_count,
        "ability_groups": {
            ability: {"expected": 3}
            for ability, _, _, _ in FDQ_ROUTE_BINDING_ROWS
        },
    }
    assert _exact_fdq_matrix(exact) is True


def test_aggregate_uses_same_turn_fail_closed_result_not_stale_disk_success(tmp_path: Path):
    run, bundle = _core_run(), _full_bundle()
    _write_core(tmp_path / "results", run, bundle, _core_result(run))
    ledger = tmp_path / "ledger.json"
    atomic_json(ledger, {"runs": [run]})
    failed = _core_result(run)
    failed.update({
        "status": "capture_failure", "success": False, "capture_complete": False,
        "own_contract": False, "safe_legal": False, "progress": False,
        "deadline": False, "stability": False,
    })
    value = aggregate(
        ledger, tmp_path / "results", expected_input_hashes=bundle,
        result_overrides={run["run_id"]: failed},
    )
    assert value["success_count"] == 0
    assert value["status_counts"] == {"capture_failure": 1}


def test_result_snapshot_detects_same_size_same_mtime_content_mutation(tmp_path: Path):
    run_dir = tmp_path / "run"
    for name, text in {
        "contract_result.json": '{"value":"aaaa"}\n',
        "runner_command.json": '{"value":"aaaa"}\n',
        "route_meta.json": '{"value":"aaaa"}\n',
        "run_meta.json": '{"value":"aaaa"}\n',
        "events.jsonl": '{"sim_time":0}\n',
        "trace.jsonl": '{"sim_time":0}\n',
    }.items():
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    before = result_snapshot(tmp_path)
    target = run_dir / "contract_result.json"
    stat = target.stat()
    target.write_text('{"value":"bbbb"}\n', encoding="utf-8")
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = result_snapshot(tmp_path)
    assert before["file_count"] == after["file_count"]
    assert before["content_sha256"] != after["content_sha256"]


def test_result_snapshot_and_reader_reject_directory_symlink(tmp_path: Path):
    run, bundle = _core_run(), _full_bundle()
    external = tmp_path / "external"
    _write_core(external, run, bundle)
    results = tmp_path / "results"
    results.mkdir()
    (results / run["run_id"]).symlink_to(
        external / run["run_id"], target_is_directory=True,
    )
    snapshot = result_snapshot(results)
    assert snapshot["all_files_stable_and_regular"] is False
    assert snapshot["file_count"] >= 1
    row = _read_result(run, results, expected_input_hashes=bundle)
    assert row["status"] == "stale_provenance"
    assert row["success"] is False


def test_result_snapshot_rejects_atomic_path_replacement_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "run" / "contract_result.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"value":"aaaa"}\n', encoding="utf-8")
    original_read = scoring_module.os.read
    replaced = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, count)
        if not replaced and chunk:
            replaced = True
            metadata = target.stat()
            replacement = target.with_suffix(".replacement")
            replacement.write_text('{"value":"bbbb"}\n', encoding="utf-8")
            os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
            os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(scoring_module.os, "read", replacing_read)
    snapshot = result_snapshot(tmp_path)
    assert replaced is True
    assert snapshot["all_files_stable_and_regular"] is False


def _copied_release_tree(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "release"
    for relative in ("build/suite", "build/safety", "build/baselines"):
        shutil.copytree(source / relative, root / relative)
    return root


def _validate_copied_release(root: Path) -> None:
    suite, safety = root / "build/suite", root / "build/safety"
    core_sidecar, _ = _materialize_json_object(suite / "sidecar.private.json")
    safety_sidecar, _ = _materialize_json_object(safety / "sidecar.private.json")
    baseline_manifest, _ = _materialize_json_object(root / "build/baselines/manifest.json")
    _validate_exact_release_ledgers(
        suite, core_sidecar, safety, baseline_manifest, root,
        safety_sidecar=safety_sidecar,
    )


def test_exact_release_gate_rejects_shrunk_and_semantically_substituted_ledgers(
    tmp_path: Path,
):
    root = _copied_release_tree(tmp_path)
    ledger_path = root / "build/suite/pilot_expected_runs.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["runs"] = ledger["runs"][:-1]
    ledger["expected_count"] -= 1
    atomic_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="denominator"):
        _validate_copied_release(root)

    root = _copied_release_tree(tmp_path / "semantic")
    ledger_path = root / "build/suite/expected_runs.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["runs"][0]["arm_id"] = "semantic-substitution"
    atomic_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="rows differ"):
        _validate_copied_release(root)


def test_baseline_manifest_gate_rejects_control_semantic_substitution(tmp_path: Path):
    root = _copied_release_tree(tmp_path)
    manifest_path = root / "build/baselines/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_baseline_manifest(root, manifest)
    manifest["baselines"]["no-language"]["evaluator_track"] = "native"
    with pytest.raises(ValueError, match="specification semantics"):
        _validate_baseline_manifest(root, manifest)


def test_materialized_ledger_gate_rejects_same_byte_atomic_replacement(tmp_path: Path):
    ledger = tmp_path / "ledger.json"
    atomic_json(ledger, {"expected_count": 1, "runs": [{"run_id": "one"}]})
    _runs, snapshot = scoring_module._materialize_release_ledger(ledger, 1)
    replacement = tmp_path / "replacement.json"
    shutil.copy2(ledger, replacement)
    os.replace(replacement, ledger)
    with pytest.raises(ValueError, match="changed during scoring"):
        _verify_materialized_release_ledgers({str(ledger.resolve()): snapshot})


def test_score_suite_exact_gate_runs_before_any_trace_rescore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    rescored = []

    def reject(*args, **kwargs):
        raise ValueError("exact-ledger-entry-gate")

    monkeypatch.setattr(scoring_module, "_validate_exact_release_ledgers", reject)
    monkeypatch.setattr(
        scoring_module, "score_ledger_traces",
        lambda *args, **kwargs: rescored.append(True),
    )
    project = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="exact-ledger-entry-gate"):
        scoring_module.score_suite(project, tmp_path / "results", tmp_path / "score.json")
    assert rescored == []


def test_report_independently_rejects_current_baseline_semantic_drift(tmp_path: Path):
    root = _copied_release_tree(tmp_path)
    assert _current_release_identity_gate(root) is True
    manifest_path = root / "build/baselines/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baselines"]["dense-route-only"]["input_strategy"] = "substituted"
    atomic_json(manifest_path, manifest)
    assert _current_release_identity_gate(root) is False


def test_report_score_gate_rejects_six_substituted_baseline_ids():
    substituted = {
        f"substituted-{index}": {"expected_runs": 432}
        for index in range(6)
    }
    assert _exact_baseline_matrix(substituted) is False
    exact = {
        baseline_id: {"expected_runs": 432}
        for baseline_id in scoring_module._BASELINE_SEMANTICS
    }
    assert _exact_baseline_matrix(exact) is True


def test_release_score_keeps_missing_jobs_in_plan_denominator(tmp_path, monkeypatch):
    from language2drive import release_scoring
    from types import SimpleNamespace
    ledger = tmp_path / 'ledger.json'
    runs = [_core_run(), {**_core_run(), 'run_id':'standardized__A-001__right__p0__s1', 'arm_id':'right', 'route_entry_id':8}]
    for run in runs:
        run.update(split='test', layer='A')
    atomic_json(ledger, {'runs':runs})
    sidecar = tmp_path/'sidecar.private.json'
    atomic_json(sidecar, {})
    bundle = _full_bundle()
    bundle['private_sidecar_sha256'] = sha256(sidecar.read_bytes()).hexdigest()
    plan = {'agent':'onevl', 'track':'standardized', 'workload':'core',
            'input_hashes':bundle, 'execution_authority':{'sidecar':str(sidecar)},
            'jobs':[{'run':runs[0]}]}
    args = SimpleNamespace(command='run', seeds=[1], split='test', tier='base', route_entry_ids=None)
    release_scoring.stamp_scope(plan, args, ledger, tmp_path)
    path = tmp_path/'plan.json'
    atomic_json(path, plan)
    monkeypatch.setattr(release_scoring, '_targets_for_suite', lambda *_: {})
    result = release_scoring.score_plan(tmp_path,path,tmp_path/'results',tmp_path/'score.json')
    assert result['metrics']['expected_runs'] == 2
    assert result['metrics']['missing_runs'] == 2
    assert result['metrics']['valid_result_runs'] == 0
    assert result['metrics']['CFPS']['A'] == 0
    assert result['metrics']['Sem-CFPS']['A'] == 0
    atomic_json(ledger, {'runs':runs[:1]})
    with pytest.raises(ValueError, match='ledger differs'):
        release_scoring.score_plan(tmp_path,path,tmp_path/'results',tmp_path/'score.json')


def test_release_cli_labels_custom_agents_and_selects_released_default(monkeypatch):
    from language2drive import cli
    seen=[]
    def record(**kwargs):
        seen.append(kwargs['agent'])
        raise ValueError('end of interface check')
    monkeypatch.setattr(cli, 'resolve_plan_inputs', record)
    assert cli.main(['run','--track','standardized','--split','dev']) != 0
    assert cli.main(['run','--track','standardized','--split','dev','--agent-path','custom.py']) != 0
    assert seen == ['simlingo','custom']
