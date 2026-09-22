from __future__ import annotations

import json
import hashlib
import os
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import language2drive.runner as runner_module
from language2drive.io import stable_hash
from language2drive.cli import make_parser
from language2drive.runner import (
    DEFAULT_EVALUATOR,
    _carla_lock_path,
    _finalize_fdq_result,
    _result_is_final,
    _result_is_current_for_sidecar,
    active_core_sidecar_path,
    build_command,
    build_fdq_command,
    calibrate_deadlines,
    default_ledger,
    evaluator_environment,
    execute_plan,
    expected_input_hashes,
    deadline_freeze_status,
    freeze_deadlines,
    make_fdq_plan,
    make_plan,
    resolve_fdq_inputs,
    resolve_plan_inputs,
    select_runs,
    select_fdq_runs,
    structural_qualification,
)
from language2drive.report import render
from language2drive.scoring import (
    _fdq_summary,
    _safety_summary,
    _targets_for_suite,
    release_artifact_hashes,
    score_ledger_traces,
    score_one,
    score_suite,
)
from language2drive.supplementary import (
    SAFETY_CASES,
    _safety_sidecar_payload,
    build_supplementary,
    validate_supplementary,
)

def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


_CANONICAL_FDQ_INFRACTION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
)


def _core_terminal_fixture(
    run: dict, *, status: str = "timeout", reason: str = "test_terminal"
) -> dict:
    """Build the same identity-bound terminal evidence written by the runner."""

    return {
        "schema_version": "language2drive.contract_result.v1",
        "status": status,
        "success": False,
        "semantic_success": False,
        "capture_complete": False,
        "run_id": run.get("run_id"),
        "pcu_id": run.get("pcu_id"),
        "arm_id": run.get("arm_id"),
        "paraphrase": run.get("paraphrase"),
        "seed": run.get("seed"),
        "track": run.get("track"),
        "route_entry_id": run.get("route_entry_id"),
        "own_contract": False,
        "opposite_contract": False,
        "safe_legal": False,
        "progress": False,
        "deadline": False,
        "stability": False,
        "reason": reason,
    }


def _core_complete_fixture(run: dict, *, success: bool = False) -> dict:
    """Build a captured verdict whose success is the six-field conjunction."""

    return {
        "schema_version": "language2drive.contract_result.v1",
        "status": "complete",
        "success": success,
        "semantic_success": success,
        "capture_complete": True,
        "run_id": run.get("run_id"),
        "pcu_id": run.get("pcu_id"),
        "arm_id": run.get("arm_id"),
        "paraphrase": run.get("paraphrase"),
        "seed": run.get("seed"),
        "track": run.get("track"),
        "route_entry_id": run.get("route_entry_id"),
        "own_contract": success,
        "opposite_contract": False,
        "safe_legal": True,
        "progress": True,
        "deadline": True,
        "stability": True,
        "unsupported_required_fields": [],
        "postprocess_error": None,
    }


def _write_core_capture_fixture(
    child: Path,
    run: dict,
    *,
    evaluator_track: str,
    private_sidecar_sha256: str,
    success: bool = False,
) -> None:
    """Write the exact nine-file evaluator capture accepted by the merge gate."""

    child.mkdir(parents=True, exist_ok=False)
    evaluator_run_id = (
        f"{evaluator_track}__{run['pcu_id']}__{run['arm_id']}__"
        f"p{run['paraphrase']}__s{run['seed']}"
    )
    metadata = {
        "run_id": evaluator_run_id,
        "route_entry_id": run["route_entry_id"],
        "route_id": f"RouteScenario_{run['route_entry_id']}",
        "scenario_name": "Language2DriveContract",
        "language2drive_public_context": (
            {"track": evaluator_track}
            if evaluator_track == "standardized"
            else {"track": evaluator_track, "seed": run["seed"]}
        ),
        "private_capture_provenance": {
            "route_entry_id": run["route_entry_id"],
            "private_sidecar_sha256": private_sidecar_sha256,
        },
    }
    for name in ("run_meta.json", "route_meta.json"):
        _write_json(child / name, metadata)
    trace = child / "trace.jsonl"
    trace.write_text(
        '{"frame":0,"sim_time":0.0}\n{"frame":1,"sim_time":0.05}\n',
        encoding="utf-8",
    )
    # Core capture enters the trusted merge as two unique equal files; the
    # runner proves their hashes and deduplicates only after publication.
    (child / "frames.jsonl").write_bytes(trace.read_bytes())
    (child / "events.jsonl").write_text("", encoding="utf-8")
    # Fixed-policy/oracle fixtures need not claim an agent-plan stream.
    (child / "agent_plans.jsonl").write_text("", encoding="utf-8")
    result = _core_complete_fixture(run, success=success)
    _write_json(child / "contract_result.json", result)
    _write_json(child / "route_summary.json", {
        "total_frames_logged": 2,
        "frame_interval": 1,
        "logger_error_count": 0,
        "capture_complete": True,
        "contract_success": success,
    })
    _write_json(child / "capture_capabilities.json", {
        "schema_version": "language2drive.trace_capture.v1",
        "private_context_loaded": True,
        "logger_error_count": 0,
        "unresolved_actor_ids": [],
        "unresolved_gate_ids": [],
        "unresolved_dwell_ids": [],
        "trace_cadence_complete": True,
        "trace_cadence_errors": [],
    })


def _fdq_checkpoint_record(run: dict, *, failed: bool) -> dict:
    binding = runner_module._canonical_fdq_route_bindings()[run["ability"]]
    infractions = {key: [] for key in _CANONICAL_FDQ_INFRACTION_KEYS}
    if failed:
        infractions["collisions_vehicle"] = ["collision"]
    return {
        "route_id": f"RouteScenario_{binding['route_id']}_rep0",
        "town_name": binding["town"],
        "scenario_name": f"{binding['scenario']}_1",
        "status": "Completed",
        "scores": {
            "score_composed": 42.0 if failed else 100.0,
            "score_route": 100.0,
            "score_penalty": 0.42 if failed else 1.0,
        },
        "infractions": infractions,
        "num_infractions": sum(len(value) for value in infractions.values()),
    }


def _write_fdq_normalized_fixture(
    run_dir: Path, run: dict, *, failed: bool
) -> Path:
    _write_json(run_dir / "bench2drive_result.json", {
        "_checkpoint": {"records": [_fdq_checkpoint_record(run, failed=failed)]},
    })
    return _finalize_fdq_result(run, run_dir, 0)


def _write_fdq_capture_fixture(child: Path, run: dict) -> None:
    """Write an FDQ capture including Alpamayo's two agent-owned reports."""

    binding = runner_module._canonical_fdq_route_bindings()[run["ability"]]
    child.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": f"RouteScenario_{binding['route_id']}_rep0",
        "route_entry_id": int(binding["route_id"]),
        "route_id": f"RouteScenario_{binding['route_id']}",
        "town_name": binding["town"],
        "scenario_name": f"{binding['scenario']}_1",
        "language2drive_public_context": {"track": "native", "seed": run["seed"]},
        "private_capture_provenance": {},
    }
    for name in ("run_meta.json", "route_meta.json"):
        _write_json(child / name, metadata)
    trace = child / "trace.jsonl"
    trace.write_text('{"frame":0,"sim_time":0.0}\n', encoding="utf-8")
    os.link(trace, child / "frames.jsonl")
    (child / "events.jsonl").write_text("", encoding="utf-8")
    (child / "agent_plans.jsonl").write_text("", encoding="utf-8")
    _write_json(child / "route_summary.json", {
        "total_frames_logged": 1,
        "frame_interval": 1,
        "logger_error_count": 0,
    })
    _write_json(child / "capture_capabilities.json", {
        "schema_version": "language2drive.trace_capture.v1",
        "private_context_loaded": False,
        "logger_error_count": 0,
    })
    _write_json(child / "alpamayo_load_report.json", {"model": "alpamayo-1.5"})
    _write_json(child / "alpamayo_agent_summary.json", {"steps": 1})


@pytest.fixture(autouse=True)
def _bounded_runtime_asset_authority(monkeypatch: pytest.MonkeyPatch):
    """Keep general runner tests bounded; dedicated tests rehash a fake tree."""

    authority = {
        "resolved_carla_root": str(runner_module.CARLA_RUNTIME_ROOT.resolve()),
        "manifest_sha256": "a" * 64,
        "inventory_sha256": "b" * 64,
        "files_sha256": "c" * 64,
        "file_count": 123,
        "total_size_bytes": 456,
        "shipping_binary_sha256": "f" * 64,
        "provenance_sha256": "d" * 64,
        "available_maps_sha256": "e" * 64,
        "client_version": "0.9.16",
        "server_version": "0.9.16-custom",
        "available_map_count": 11,
        "boot_id": "bounded-test-boot",
        "freeze_monotonic_ticks": 1,
        "clock_ticks_per_second": 100,
        "server_readonly_permissions_sha256": "9" * 64,
        "server_readonly_permission_entry_count": 321,
        "server_uid": 1000,
        "server_gids": [1000],
    }
    monkeypatch.setattr(
        runner_module, "_current_runtime_asset_authority", lambda: dict(authority)
    )
    monkeypatch.setattr(
        runner_module, "_current_session_runtime_asset_authority",
        lambda: dict(authority),
    )
    monkeypatch.setattr(
        runner_module, "_validate_plan_sha256", lambda plan: None
    )
    original_endpoint = runner_module._plan_carla_endpoint
    monkeypatch.setattr(
        runner_module,
        "_plan_carla_endpoint",
        lambda plan: (
            original_endpoint(plan)
            if isinstance(plan.get("execution_authority"), dict)
            else (
                "127.0.0.1",
                int(runner_module._command_option(plan["jobs"][0]["command"], "--port", "2000")),
                int(runner_module._command_option(plan["jobs"][0]["command"], "--traffic-manager-port", "8000")),
            )
        ),
    )
    monkeypatch.setattr(
        runner_module, "_validate_plan_execution_authority", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_module, "_validate_nonasset_execution_authority", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_module, "_execution_dependency_hashes",
        lambda profile: {
            "bounded/runtime-profile": stable_hash({"profile": profile}),
            "project/language2drive/leaderboard/leaderboard/utils/closed_loop_logger.py":
                runner_module._current_file_sha256(
                    next(
                        path for path in runner_module.EVALUATOR_RUNTIME_DEPENDENCIES
                        if path.name == "closed_loop_logger.py"
                    )
                ),
            **{
                key: stable_hash({"bounded_dependency": key})
                for key in {
                    "language2drive.v1/environment.lock.yml",
                    "project/model_zoo/autovla/codebook_cache/agent_vocab.pkl",
                    "project/model_zoo/autovla/code/models/action_tokenizer.py",
                    "project/model_zoo/autovla/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct/model.safetensors.index.json",
                    "project/model_zoo/autovla/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct/model-00001-of-00002.safetensors",
                    "project/model_zoo/autovla/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct/model-00002-of-00002.safetensors",
                    "project/team_code/language2drive_runtime.py",
                    "project/team_code/language2drive_context.py",
                    "project/team_code/language2drive_signal_free_proof.py",
                    "project/language2drive/leaderboard/leaderboard/scenarios/scenario_manager.py",
                    "project/language2drive/leaderboard/leaderboard/autoagents/autonomous_agent.py",
                    "project/language2drive/scenario_runner/srunner/scenariomanager/carla_data_provider.py",
                    "project/language2drive/scenario_runner/srunner/scenariomanager/timer.py",
                    "project/language2drive/scenario_runner/srunner/scenarios/route_scenario.py",
                    "language2drive.v1/src/language2drive/runner.py",
                    "language2drive.v1/src/language2drive/carla_runtime_assets.py",
                    "project/team_code/__init__.py",
                    "python-runtime/site-startup-files",
                    "python-runtime/prefix-manifest",
                    "native-runtime/mapped-oracle-workload",
                    "python-distribution/setuptools",
                    "python-distribution/mkl-service",
                    "python-distribution/pexpect",
                    "python-distribution/ptyprocess",
                    "python-distribution/simple-watchdog-timer",
                    "python-distribution/wcwidth",
                }
            },
        },
    )
    # General execution tests exercise runner outcomes, not the host's
    # momentary free-space level.  Storage-specific tests override this stub
    # with exact boundary values below.
    monkeypatch.setattr(
        runner_module, "_storage_available_bytes", lambda path: 64 * 1024 ** 3,
    )
    launch = {
        "schema_version": "language2drive.carla_server_launch_receipt.v1",
        "boot_id": "bounded-test-boot", "prelaunch_monotonic_ticks": 1,
        "clock_ticks_per_second": 100,
        "server_runtime_root": "/server/runtime",
        "runtime_asset_manifest_sha256": "1" * 64,
        "runtime_asset_provenance_sha256": "2" * 64,
        "launcher_sha256": "3" * 64, "shipping_binary_sha256": "f" * 64,
    }
    mapped = [{
        "path": "/carla/Shipping", "device": 1, "inode": 2,
        "maps_device": "00:01", "size_bytes": 3, "sha256": "f" * 64,
    }]
    process = {
        "uid": 1000, "gid": 1000, "pid": 123, "starttime_ticks": 2,
        "clock_ticks_per_second": 100, "socket_inodes": [4242],
        "executable": "/carla/Shipping", "executable_dev": 1,
        "executable_ino": 2, "executable_size": 3,
        "executable_sha256": "f" * 64, "cmdline": [],
        "cmdline_sha256": stable_hash([]),
        "cwd": "/carla", "cwd_device": 1, "cwd_inode": 5,
        "root": "/", "root_device": 1, "root_inode": 1,
        "mount_namespace_inode": 4,
        "boot_id": "bounded-test-boot", "environment": {},
        "environment_sha256": stable_hash({}),
        "server_home_inventory": {
            "root": "/server/home", "directories_sha256": "1" * 64,
            "file_count": 0, "total_size_bytes": 0, "files_sha256": "2" * 64,
        },
        "server_runtime_inventories": {},
        "server_runtime_inventories_sha256": stable_hash({}),
        "mapped_runtime_files": mapped,
        "mapped_runtime_files_sha256": stable_hash(mapped),
        "launch_receipt": launch,
        "launch_receipt_sha256": "4" * 64,
        "port": 2000,
    }
    monkeypatch.setattr(
        runner_module, "_attest_carla_listener_process",
        lambda *args, **kwargs: dict(process),
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_live_carla_runtime_authority",
        lambda plan, host, port: {
            "client_version": "0.9.16",
            "server_version": "0.9.16-custom",
            "available_maps": ["/Game/Carla/Maps/Town01"],
            "available_map_count": 1,
            "available_maps_sha256": "e" * 64,
        },
    )
    return authority


def _lock_test_plan(
    run_ids: list[str],
    *,
    port: int = 2000,
    tm_port: int = 8000,
    plan_sha256: str | None = None,
) -> dict:
    policy = runner_module._storage_policy()
    jobs = [
        {
            "run": {"run_id": run_id},
            "command": [
                "/bin/true", "--host", "127.0.0.1", "--port", str(port),
                "--traffic-manager-port", str(tm_port),
            ],
            "deadline_seconds": None,
        }
        for run_id in run_ids
    ]
    for job in jobs:
        job["storage_estimate"] = runner_module._job_storage_estimate(job, policy)
    project = Path("/tmp/runner-lock-project").resolve()
    authority = {
        "schema_version": "language2drive.plan_execution_authority.v1",
        "project_root": str(project), "build_dir": str(project / "build"),
        "route_xml": str(project / "routes.xml"), "sidecar": None,
        "results_dir": str(project / "results"), "agent": "oracle",
        "agent_path": str(project / "agent.py"),
        "agent_config": str(project / "config.json"),
        "benchmark_track": "oracle", "evaluator_track": "native",
        "workload": "core", "dependency_profile": "base-runtime.v1",
        "host": "127.0.0.1", "port": port,
        "traffic_manager_port": tm_port, "gpu_rank": 0,
    }
    return {
        "plan_sha256": plan_sha256 or f"lock-test-{port}-{tm_port}",
        "workload": "core",
        "execution_authority": authority,
        "storage_policy": policy,
        **runner_module._plan_storage_estimate(jobs),
        "jobs": jobs,
    }


def _with_storage_policy(plan: dict) -> dict:
    policy = runner_module._storage_policy()
    for job in plan.get("jobs", []):
        job.setdefault("deadline_seconds", None)
        job["storage_estimate"] = runner_module._job_storage_estimate(job, policy)
    plan["storage_policy"] = policy
    plan.update(runner_module._plan_storage_estimate(plan.get("jobs", [])))
    return plan


def _plan_state_path(results_dir: Path, plan: dict) -> Path:
    return (
        results_dir / ".runner-states" / f"{plan['plan_sha256']}.json"
    )


def _start_external_lock_holder(lock_path: Path) -> subprocess.Popen[str]:
    script = """
import fcntl
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a+", encoding="utf-8") as stream:
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    stream.seek(0)
    stream.truncate()
    stream.write('{"pid": 987654, "plan_sha256": "holder", "started_at": "test"}\\n')
    stream.flush()
    print("locked", flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "locked"
    return process


def _release_external_lock_holder(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None:
        process.stdin.write("\n")
        process.stdin.flush()
    process.wait(timeout=5)
    assert process.returncode == 0


class _InjectedRunnerBaseException(BaseException):
    pass


def _pid_is_running(pid: int) -> bool:
    """Treat a killed-but-not-yet-reaped orphan zombie as non-running."""

    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(stat_fields) > 2 and stat_fields[2] != "Z"


def _deadline_freeze_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    project = tmp_path / "project"
    build = project / "build/suite"
    config = project / "config/defaults.json"
    _write_json(config, {"fixture": "deadline-freeze"})
    _write_json(build / "manifest.json", {"fixture": True})
    _write_json(build / "provenance.json", {"fixture": True})
    (build / "routes_native.xml").write_text("<routes/>\n", encoding="utf-8")
    pcus, routes, runs = [], [], []
    route_entry_id = 0
    for pcu_index in range(128):
        pcu_id = f"T-{pcu_index:03d}"
        arms = []
        for arm_id in ("a", "b"):
            arms.append({"id": arm_id, "deadline_seconds": 10})
            canonical_route_id = route_entry_id
            for paraphrase in range(3):
                routes.append({
                    "route_entry_id": route_entry_id,
                    "pcu_id": pcu_id,
                    "arm_id": arm_id,
                    "paraphrase": paraphrase,
                    "deadline_seconds": 10,
                })
                route_entry_id += 1
            for seed in (1, 2, 3):
                runs.append({
                    "run_id": f"oracle__{pcu_id}__{arm_id}__p0__s{seed}",
                    "pcu_id": pcu_id,
                    "arm_id": arm_id,
                    "paraphrase": 0,
                    "seed": seed,
                    "track": "oracle",
                    "route_entry_id": canonical_route_id,
                    "layer": "T",
                    "split": "test",
                    "status": "expected",
                })
        pcus.append({"id": pcu_id, "arms": arms})
    _write_json(build / "sidecar.private.json", {
        "schema_version": "language2drive.private_sidecar.v1",
        "pcus": pcus,
        "routes": routes,
        "geometry_registry": {},
    })
    _write_json(build / "oracle_expected_runs.json", {
        "expected_count": len(runs), "runs": runs,
    })
    bundle = expected_input_hashes(
        sidecar=build / "sidecar.private.json",
        route_xml=build / "routes_native.xml",
        agent_path=runner_module.DEFAULT_ORACLE_AGENT,
        agent_config=config,
        benchmark_track="oracle",
        evaluator_track="native",
        workload="core",
        dependency_profile="base-runtime.v1",
        wall_timeout_policy=runner_module._wall_timeout_policy(
            agent="oracle", workload="core",
        ),
    )
    results = project / "results"
    base_hash = runner_module.sha256_file(build / "sidecar.private.json")
    for run in runs:
        run_dir = results / run["run_id"]
        _write_json(
            run_dir / "contract_result.json",
            _core_complete_fixture(run, success=True),
        )
        (run_dir / "trace.jsonl").write_text('{"sim_time": 0}\n', encoding="utf-8")
        _write_json(run_dir / "runner_command.json", {
            "run": run, "input_hashes": bundle,
        })
        _write_json(run_dir / "run_meta.json", {
            "private_capture_provenance": {
                "route_entry_id": run["route_entry_id"],
                "private_sidecar_sha256": base_hash,
            },
        })
    return project, results, runs


def test_cli_run_seed_parser_and_plan_only_default():
    args = make_parser().parse_args([
        "run", "--track", "standardized", "--split", "dev", "--seeds", "3,1,3",
    ])
    assert args.seeds == [3, 1]
    assert args.execute is False
    assert args.agent == "autovla"

    assert make_parser().parse_args([
        "run", "--track", "standardized", "--split", "dev",
    ]).seeds == [1]
    assert make_parser().parse_args(["run-fdq"]).seeds == [1]
    assert make_parser().parse_args([
        "run-baseline", "--baseline", "always-left",
    ]).seeds == [1]

    filtered = make_parser().parse_args([
        "run", "--track", "oracle", "--split", "dev",
        "--route-entry-ids", "12,3,12",
    ])
    assert filtered.route_entry_ids == [12, 3]


def test_default_ledgers_are_track_separated(tmp_path: Path):
    assert default_ledger(tmp_path, "standardized").name == "large_expected_runs.json"
    assert default_ledger(tmp_path, "standardized", tier="small").name == "small_expected_runs.json"
    assert default_ledger(tmp_path, "standardized", tier="base").name == "base_expected_runs.json"
    assert default_ledger(tmp_path, "standardized", pilot=True).name == "pilot_expected_runs.json"
    assert default_ledger(tmp_path, "native").name == "native_smoke_expected_runs.json"
    assert default_ledger(tmp_path, "oracle").name == "oracle_expected_runs.json"


def test_recovery_skips_finalized_failure_without_erasing_it(tmp_path: Path):
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    ledger = tmp_path / "ledger.json"
    _write_json(ledger, {"expected_count": 1, "runs": [run]})
    result = tmp_path / "results" / run["run_id"] / "contract_result.json"
    _write_json(result, _core_complete_fixture(run, success=False))
    result.with_name("trace.jsonl").write_text(
        json.dumps({"sim_time": 0.0, "ego": {"location": {"x": 0, "y": 0}}}) + "\n",
        encoding="utf-8",
    )
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {},
    })
    selected, summary = select_runs(
        ledger, tmp_path / "results", track="standardized", split="dev", seeds=[1]
    )
    assert selected == []
    assert summary["skipped_finalized"] == 1
    retried, _ = select_runs(
        ledger, tmp_path / "results", track="standardized", split="dev", seeds=[1], retry_finalized=True
    )
    assert retried == [run]


def test_recoverable_selection_can_target_exact_route_entries(tmp_path: Path):
    runs = [
        {
            "run_id": f"oracle__route{route_id}__p0__s{seed}",
            "track": "oracle", "split": "dev", "seed": seed,
            "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
            "route_entry_id": route_id,
        }
        for seed, route_id in ((1, 0), (2, 0), (1, 3))
    ]
    ledger = tmp_path / "ledger.json"
    _write_json(ledger, {"expected_count": len(runs), "runs": runs})
    selected, summary = select_runs(
        ledger, tmp_path / "results", track="oracle", split="dev", seeds=[1, 2],
        route_entry_ids=[3],
    )
    assert [run["route_entry_id"] for run in selected] == [3]
    assert summary == {"selected": 1, "skipped_finalized": 0, "filtered": 2}


def test_recovery_rejects_finalized_result_from_stale_sidecar(tmp_path: Path):
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    sidecar = tmp_path / "sidecar.private.json"
    _write_json(sidecar, {"routes": [{"route_entry_id": 0, "version": 1}]})
    result = tmp_path / "results" / run["run_id"] / "contract_result.json"
    _write_json(result, _core_terminal_fixture(run))
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {"private_sidecar_sha256": "not-current"},
    })
    assert _result_is_final(result, run) is True
    assert _result_is_current_for_sidecar(
        result, sidecar, expected_run=run,
    ) is False

    from language2drive.io import sha256_file
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {"private_sidecar_sha256": sha256_file(sidecar)},
    })
    assert _result_is_current_for_sidecar(
        result, sidecar, expected_run=run,
    ) is True


def test_recovery_full_bundle_rejects_old_agent_xml_config_and_evaluator(tmp_path: Path):
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    sidecar = tmp_path / "sidecar.private.json"
    _write_json(sidecar, {"routes": []})
    result = tmp_path / "direct-results" / run["run_id"] / "contract_result.json"
    _write_json(result, _core_terminal_fixture(run))
    current = {
        "schema_version": "language2drive.execution_bundle.v2",
        "private_sidecar_sha256": runner_module.sha256_file(sidecar),
        "route_xml_sha256": "xml",
        "agent_sha256": "agent-new",
        "agent_config_sha256": "config",
        "evaluator_sha256": "evaluator",
        "workload": "core",
        "benchmark_track": "standardized",
        "evaluator_track": "standardized",
        "dependency_profile": "autovla-runtime.v1",
        "dependency_hashes": {"project/model_zoo/autovla/config.yaml": "dependency-new"},
        "dependency_manifest_sha256": "manifest-new",
        "wall_timeout_policy_sha256": "policy-new",
    }
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {**current, "agent_sha256": "agent-old"},
    })
    assert _result_is_current_for_sidecar(
        result, sidecar, current, expected_run=run,
    ) is False
    ledger = tmp_path / "ledger.json"
    _write_json(ledger, {"expected_count": 1, "runs": [run]})
    run_result = tmp_path / "results" / run["run_id"] / "contract_result.json"
    _write_json(run_result, _core_terminal_fixture(run))
    _write_json(run_result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {**current, "agent_sha256": "agent-old"},
    })
    selected, summary = select_runs(
        ledger, tmp_path / "results", track="standardized", split="dev", seeds=[1],
        sidecar_path=sidecar, expected_input_hashes=current,
    )
    assert selected == [run]
    assert summary["skipped_finalized"] == 0
    _write_json(run_result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {
            **current,
            "dependency_hashes": {"project/model_zoo/autovla/config.yaml": "dependency-old"},
        },
    })
    selected, summary = select_runs(
        ledger, tmp_path / "results", track="standardized", split="dev", seeds=[1],
        sidecar_path=sidecar, expected_input_hashes=current,
    )
    assert selected == [run]
    assert summary["skipped_finalized"] == 0
    _write_json(run_result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {**current, "wall_timeout_policy_sha256": "policy-old"},
    })
    selected, summary = select_runs(
        ledger, tmp_path / "results", track="standardized", split="dev", seeds=[1],
        sidecar_path=sidecar, expected_input_hashes=current,
    )
    assert selected == [run]
    assert summary["skipped_finalized"] == 0
    _write_json(run_result.with_name("runner_command.json"), {
        "run": run, "input_hashes": current,
    })
    selected, summary = select_runs(
        ledger, tmp_path / "results", track="standardized", split="dev", seeds=[1],
        sidecar_path=sidecar, expected_input_hashes=current,
    )
    assert selected == []
    assert summary["skipped_finalized"] == 1
    _write_json(result.with_name("runner_command.json"), {
        "run": run, "input_hashes": current,
    })
    assert _result_is_current_for_sidecar(
        result, sidecar, current, expected_run=run,
    ) is True


def test_base_runtime_bundle_binds_logger_and_logger_drift_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    sidecar = tmp_path / "sidecar.private.json"
    route_xml = tmp_path / "routes_native.xml"
    agent = tmp_path / "oracle_agent.py"
    config = tmp_path / "oracle_agent_config.json"
    _write_json(sidecar, {"routes": []})
    route_xml.write_text("<routes />", encoding="utf-8")
    agent.write_text("# frozen oracle agent\n", encoding="utf-8")
    _write_json(config, {})
    bundle = expected_input_hashes(
        sidecar=sidecar,
        route_xml=route_xml,
        agent_path=agent,
        agent_config=config,
        benchmark_track="oracle",
        evaluator_track="native",
        workload="core",
        dependency_profile="base-runtime.v1",
    )
    logger_key = (
        "project/language2drive/leaderboard/leaderboard/utils/"
        "closed_loop_logger.py"
    )
    logger_path = next(
        path for path in runner_module.EVALUATOR_RUNTIME_DEPENDENCIES
        if path.name == "closed_loop_logger.py"
    )
    assert bundle["dependency_hashes"][logger_key] == (
        runner_module._current_file_sha256(logger_path)
    )
    indirect_keys = {
        "project/team_code/language2drive_runtime.py",
        "project/team_code/language2drive_context.py",
        "project/team_code/language2drive_signal_free_proof.py",
        "project/language2drive/leaderboard/leaderboard/scenarios/scenario_manager.py",
        "project/language2drive/leaderboard/leaderboard/autoagents/autonomous_agent.py",
        "project/language2drive/scenario_runner/srunner/scenariomanager/carla_data_provider.py",
        "project/language2drive/scenario_runner/srunner/scenariomanager/timer.py",
        "project/language2drive/scenario_runner/srunner/scenarios/route_scenario.py",
        "language2drive.v1/src/language2drive/runner.py",
        "language2drive.v1/src/language2drive/carla_runtime_assets.py",
        "project/team_code/__init__.py",
        "python-runtime/site-startup-files",
        "python-runtime/prefix-manifest",
        "native-runtime/mapped-oracle-workload",
        "python-distribution/setuptools",
        "python-distribution/mkl-service",
        "python-distribution/pexpect",
        "python-distribution/ptyprocess",
        "python-distribution/simple-watchdog-timer",
        "python-distribution/wcwidth",
    }
    assert indirect_keys <= bundle["dependency_hashes"].keys()
    assert bundle["dependency_manifest_sha256"] == runner_module.stable_hash(
        bundle["dependency_hashes"]
    )

    run = {
        "run_id": "oracle__A-001__left__p0__s1",
        "track": "oracle", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    result = tmp_path / "results" / run["run_id"] / "contract_result.json"
    _write_json(result, _core_terminal_fixture(run))
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": bundle,
    })
    assert _result_is_current_for_sidecar(
        result, sidecar, bundle, expected_run=run,
    ) is True

    stale = json.loads(json.dumps(bundle))
    stale["dependency_hashes"][logger_key] = "0" * 64
    stale["dependency_manifest_sha256"] = runner_module.stable_hash(
        stale["dependency_hashes"]
    )
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": stale,
    })
    assert _result_is_current_for_sidecar(
        result, sidecar, bundle, expected_run=run,
    ) is False
    for key in sorted(indirect_keys):
        stale = json.loads(json.dumps(bundle))
        stale["dependency_hashes"][key] = "0" * 64
        stale["dependency_manifest_sha256"] = runner_module.stable_hash(
            stale["dependency_hashes"]
        )
        _write_json(result.with_name("runner_command.json"), {
            "run": run, "input_hashes": stale,
        })
        assert _result_is_current_for_sidecar(
            result, sidecar, bundle, expected_run=run,
        ) is False


def test_offline_scorer_derives_gate_progress_safety_and_stability(tmp_path: Path):
    project = tmp_path / "project"
    results = project / "results"
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    target = {
        "type": "gate", "id": "gate-left",
        "polygon": [[4, -2], [6, -2], [6, 2], [4, 2], [4, -2]],
    }
    arm = {
        "id": "left",
        "contract": {"eventually": {"enter_gate": {"id": "gate-left"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "gate-right"}}},
        "target": target,
        "oracle_path": [{"x": 0, "y": 0, "z": 0}, {"x": 10, "y": 0, "z": 0}],
        "deadline_seconds": 10,
    }
    pcu = {"id": "A-001", "arms": [arm]}
    sidecar = {
        "pcus": [pcu],
        "routes": [{"route_entry_id": 0, "pcu_id": "A-001", "arm_id": "left"}],
    }
    run_dir = results / run["run_id"]
    run_dir.mkdir(parents=True)
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for timestamp, x in enumerate((0, 3, 5, 7, 9)):
            stream.write(json.dumps({
                "sim_time": timestamp,
                "ego": {"location": {"x": x, "y": 0, "z": 0}, "speed_mps": 2},
                "events": [],
            }) + "\n")
    targets = _targets_for_suite(project, sidecar)
    result = score_one(
        run,
        project_root=project,
        results_dir=results,
        sidecar=sidecar,
        target_index=targets,
    )
    assert result is not None
    assert result["status"] == "complete"
    assert result["own_contract"] is True
    assert result["opposite_contract"] is False
    assert result["safe_legal"] is True
    assert result["progress"] is True
    assert result["driving_score"] == pytest.approx(90.0)
    assert result["stability"] is True
    assert result["success"] is True
    assert result["augmentation"]["derived_gate_events"] == 1


def test_pure_ledger_rescore_preserves_frozen_witness_bytes(tmp_path: Path):
    project = tmp_path / "project"
    results = project / "results"
    run = {
        "run_id": "oracle__A-001__left__p0__s1",
        "track": "oracle", "split": "test", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    target = {
        "type": "gate", "id": "goal",
        "polygon": [[4, -2], [6, -2], [6, 2], [4, 2], [4, -2]],
    }
    arm = {
        "id": "left",
        "contract": {"eventually": {"enter_gate": {"id": "goal"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "other"}}},
        "target": target,
        "oracle_path": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
        "deadline_seconds": 10,
    }
    sidecar = {
        "pcus": [{"id": "A-001", "arms": [arm]}],
        "routes": [{"route_entry_id": 0, "pcu_id": "A-001", "arm_id": "left"}],
    }
    ledger = project / "ledger.json"
    _write_json(ledger, {"expected_count": 1, "runs": [run]})
    run_dir = results / run["run_id"]
    run_dir.mkdir(parents=True)
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for timestamp, x in enumerate((0, 3, 5, 7, 9)):
            stream.write(json.dumps({
                "sim_time": timestamp,
                "ego": {"location": {"x": x, "y": 0}, "speed_mps": 2},
                "events": [],
            }) + "\n")
    witness_result = run_dir / "contract_result.json"
    _write_json(witness_result, {
        "status": "complete", "success": True, "immutable_witness": True,
    })
    before = witness_result.read_bytes()
    summary = score_ledger_traces(
        ledger,
        project_root=project,
        results_dir=results,
        sidecar=sidecar,
        target_index=_targets_for_suite(project, sidecar),
        write_results=False,
    )
    assert summary["scored"] == 1
    assert witness_result.read_bytes() == before


def test_offline_scorer_does_not_turn_partial_crash_trace_into_success(tmp_path: Path):
    project = tmp_path / "project"
    results = project / "results"
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    target = {"type": "gate", "id": "goal", "polygon": [[1, -1], [3, -1], [3, 1], [1, 1], [1, -1]]}
    arm = {
        "id": "left", "contract": {"eventually": {"enter_gate": {"id": "goal"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "other"}}},
        "target": target, "oracle_path": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
        "deadline_seconds": 20,
    }
    sidecar = {"pcus": [{"id": "A-001", "arms": [arm]}], "routes": [{"route_entry_id": 0, "pcu_id": "A-001", "arm_id": "left"}]}
    run_dir = results / run["run_id"]
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "contract_result.json", {"status": "crash", "success": False})
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for timestamp, x in enumerate((0, 2, 4, 6, 8)):
            stream.write(json.dumps({"sim_time": timestamp, "ego": {"location": {"x": x, "y": 0}, "speed_mps": 2}, "events": []}) + "\n")
    result = score_one(run, project_root=project, results_dir=results, sidecar=sidecar, target_index=_targets_for_suite(project, sidecar))
    assert result is not None
    assert result["status"] == "crash"
    assert result["success"] is False
    assert result["terminal_failure_preserved"] is True


def test_offline_scorer_cannot_promote_runtime_authority_failure(tmp_path: Path):
    project = tmp_path / "project"
    results = project / "results"
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    target = {
        "type": "gate", "id": "goal",
        "polygon": [[1, -1], [3, -1], [3, 1], [1, 1], [1, -1]],
    }
    arm = {
        "id": "left", "contract": {"eventually": {"enter_gate": {"id": "goal"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "other"}}},
        "target": target, "oracle_path": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
        "deadline_seconds": 20,
    }
    sidecar = {
        "pcus": [{"id": "A-001", "arms": [arm]}],
        "routes": [{"route_entry_id": 0, "pcu_id": "A-001", "arm_id": "left"}],
    }
    run_dir = results / run["run_id"]
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "contract_result.json", {
        "status": "runtime_authority_failure", "success": False,
        "capture_complete": False,
        "reason": "post_evaluator_authority_mismatch",
        "detail": "dependency drift after evaluator",
    })
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for timestamp, x in enumerate((0, 2, 4, 6, 8)):
            stream.write(json.dumps({
                "sim_time": timestamp,
                "ego": {"location": {"x": x, "y": 0}, "speed_mps": 2},
                "events": [],
            }) + "\n")
    result = score_one(
        run, project_root=project, results_dir=results, sidecar=sidecar,
        target_index=_targets_for_suite(project, sidecar),
    )
    assert result is not None
    assert result["status"] == "runtime_authority_failure"
    assert result["success"] is False
    assert result["capture_complete"] is False
    assert result["capture_failure_preserved"] is True
    assert result["prior_reason"] == "post_evaluator_authority_mismatch"
    assert result["prior_detail"] == "dependency drift after evaluator"


@pytest.mark.parametrize("malformed", [0, "false", None])
def test_offline_scorer_rejects_malformed_capture_complete(
    tmp_path: Path, malformed,
):
    project = tmp_path / "project"
    results = project / "results"
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    target = {
        "type": "gate", "id": "goal",
        "polygon": [[1, -1], [3, -1], [3, 1], [1, 1], [1, -1]],
    }
    arm = {
        "id": "left", "contract": {"eventually": {"enter_gate": {"id": "goal"}}},
        "opposite_contract": {"eventually": {"enter_gate": {"id": "other"}}},
        "target": target, "oracle_path": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
        "deadline_seconds": 20,
    }
    sidecar = {
        "pcus": [{"id": "A-001", "arms": [arm]}],
        "routes": [{"route_entry_id": 0, "pcu_id": "A-001", "arm_id": "left"}],
    }
    run_dir = results / run["run_id"]
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "contract_result.json", {
        "status": "complete", "success": True,
        "capture_complete": malformed,
    })
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for timestamp, x in enumerate((0, 2, 4, 6, 8)):
            stream.write(json.dumps({
                "sim_time": timestamp,
                "ego": {"location": {"x": x, "y": 0}, "speed_mps": 2},
                "events": [],
            }) + "\n")
    result = score_one(
        run, project_root=project, results_dir=results, sidecar=sidecar,
        target_index=_targets_for_suite(project, sidecar),
    )
    assert result is not None
    assert result["status"] == "capture_failure"
    assert result["success"] is False
    assert result["capture_complete"] is False
    assert result["capture_field_invalid"] is True


def test_existing_suite_structural_artifact_never_claims_dynamic(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "qualification.json"
    value = structural_qualification(project, output)
    assert value["qualification_name"] == "oracle_path_structural_qualification"
    assert value["is_dynamic_carla_rollout"] is False
    assert value["dynamic_oracle_status"] == "pending_execution"
    assert value["pcu_count"] == 128
    assert value["arm_count"] == 256
    assert output.is_file()


def test_finalized_outcome_requires_trace_but_terminal_failure_does_not(tmp_path: Path):
    run = {
        "run_id": "standardized__A-001__left__p0__s1",
        "track": "standardized", "split": "dev", "seed": 1,
        "pcu_id": "A-001", "arm_id": "left", "paraphrase": 0,
        "route_entry_id": 0,
    }
    result = tmp_path / "contract_result.json"
    _write_json(result, {"status": "complete", "success": False})
    assert _result_is_final(result) is False
    result.with_name("trace.jsonl").write_text('{"sim_time":0}\n', encoding="utf-8")
    assert _result_is_final(result) is False
    _write_json(result, _core_complete_fixture(run, success=False))
    assert _result_is_final(result, run) is True
    loose_semantic = _core_complete_fixture(run, success=False)
    loose_semantic["semantic_success"] = True
    assert loose_semantic["own_contract"] is False
    _write_json(result, loose_semantic)
    assert _result_is_final(result, run) is True
    for malformed in (0, "false", None):
        value = _core_complete_fixture(run, success=True)
        value["capture_complete"] = malformed
        _write_json(result, value)
        assert _result_is_final(result, run) is False
    result.with_name("trace.jsonl").unlink()
    _write_json(result, _core_terminal_fixture(run))
    assert _result_is_final(result, run) is True


def test_runtime_cache_environment_stays_under_autodl_tmp():
    environment = evaluator_environment()
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    for key in (
        "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE", "TORCH_HOME", "CUDA_CACHE_PATH", "CARLA_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
    ):
        assert environment[key].startswith("/root/autodl-tmp/")
        # The returned default is a path template only.  Each real job
        # creates a fresh empty attempt root immediately before launch.
        assert not Path(environment[key]).exists()


def test_every_evaluator_command_requires_exactly_one_external_carla_flag(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    core_run = json.loads((project / "build/suite/expected_runs.json").read_text())["runs"][0]
    safety_run = json.loads((project / "build/safety/expected_runs.json").read_text())["runs"][0]
    baseline_run = json.loads(
        (project / "build/baselines/ledgers/no-language.json").read_text()
    )["runs"][0]
    variants = (
        (core_run, project / "build/suite", "standardized", project / "build/suite/routes_standardized.xml"),
        (core_run, project / "build/suite", "native", project / "build/suite/routes_native.xml"),
        (core_run, project / "build/suite", "oracle", project / "build/suite/routes_native.xml"),
        (safety_run, project / "build/safety", "standardized", project / "build/safety/routes_standardized.xml"),
        (baseline_run, project / "build/baselines", "baseline", project / "build/baselines/routes_no_language.xml"),
        (baseline_run, project / "build/baselines", "baseline", project / "build/baselines/routes_dense_route_only.xml"),
    )
    for run, build_dir, track, route_xml in variants:
        sidecar = (
            project / "build/safety/sidecar.private.json"
            if build_dir.name == "safety"
            else project / "build/suite/sidecar.private.json"
        )
        command = build_command(
            run, project_root=project, build_dir=build_dir,
            results_dir=tmp_path / "results", track=track,
            agent_path=Path("/bin/true"), agent_config=Path("/bin/true"),
            evaluator_track="native" if track in {"native", "oracle"} else "standardized",
            route_xml=route_xml, sidecar=sidecar,
        )
        assert command.count("--l2d-external-carla") == 1
        assert command.count("--l2d-no-rendering") == (1 if track == "oracle" else 0)

    fdq_command = build_fdq_command(
        {"run_id": "fdq__overtaking__s1", "ability": "overtaking", "seed": 1},
        results_dir=tmp_path / "fdq-results",
        agent_path=Path("/bin/true"), agent_config=Path("/bin/true"),
    )
    assert fdq_command.count("--l2d-external-carla") == 1
    assert "--l2d-no-rendering" not in fdq_command


def test_execute_rejects_stale_evaluator_plan_without_external_carla_flag(tmp_path: Path):
    plan = _with_storage_policy({
        "plan_sha256": "stale-plan",
        "workload": "core",
        "jobs": [{
            "run": {"run_id": "standardized__A-001__left__p0__s1"},
            "command": ["/bin/true", str(DEFAULT_EVALUATOR)],
        }],
    })
    with pytest.raises(ValueError, match="external-carla exactly once"):
        execute_plan(
            plan, project_root=tmp_path, results_dir=tmp_path / "results",
            require_server=False,
        )


def test_execute_rejects_old_plan_without_storage_policy(tmp_path: Path):
    plan = {"plan_sha256": "old", "jobs": []}
    with pytest.raises(ValueError, match="regenerate it with plan-only CLI"):
        execute_plan(
            plan, project_root=tmp_path, results_dir=tmp_path / "results",
            require_server=False,
        )


def test_execute_rejects_results_on_system_disk_outside_project():
    project = Path(__file__).resolve().parents[1]
    forbidden = Path("/root/language2drive-system-disk-results-test")
    assert not forbidden.exists()
    with pytest.raises(ValueError, match="out-of-scope"):
        execute_plan(
            _lock_test_plan([]), project_root=project, results_dir=forbidden,
            require_server=False,
        )


def test_execute_halts_before_first_run_when_storage_reserve_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    results = project / "results"
    run_id = "never-created"
    plan = _lock_test_plan([run_id])
    estimate = plan["jobs"][0]["storage_estimate"]["estimated_transient_capture_bytes"]
    monkeypatch.setattr(
        runner_module, "_storage_available_bytes",
        lambda path: runner_module.STORAGE_RESERVE_BYTES + estimate - 1,
    )
    summary = execute_plan(
        plan, project_root=project, results_dir=results, require_server=False,
    )
    assert summary["status"] == "execution_halted"
    assert summary["reason"] == "storage_budget_exhausted"
    assert summary["halted"] == 1
    assert not (results / run_id).exists()
    state_path = _plan_state_path(results, plan)
    assert summary["state_path"] == str(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["execution"]["phase"] == "pre_execution"
    assert run_id not in state["runs"]


def test_execute_halts_remaining_runs_when_storage_drops_before_second_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    results = project / "results"
    first_run, second_run = "storage-first", "storage-second"
    plan = _lock_test_plan([first_run, second_run])
    first_estimate = plan["jobs"][0]["storage_estimate"]["estimated_transient_capture_bytes"]
    second_estimate = plan["jobs"][1]["storage_estimate"]["estimated_transient_capture_bytes"]
    enough = runner_module.STORAGE_RESERVE_BYTES + max(first_estimate, second_estimate) + 1
    availability = iter((enough, enough, runner_module.STORAGE_RESERVE_BYTES + second_estimate - 1))
    monkeypatch.setattr(
        runner_module, "_storage_available_bytes", lambda path: next(availability),
    )

    def fake_run_evaluator(command, *, cwd, environment, log, wall_timeout):
        del command, cwd, environment, log, wall_timeout
        # A launch failure is a strict identity-bound terminal outcome.  It
        # avoids bypassing the evaluator-capture namespace in this storage test.
        raise OSError("injected evaluator launch failure")

    monkeypatch.setattr(runner_module, "_run_evaluator", fake_run_evaluator)
    summary = execute_plan(
        plan, project_root=project, results_dir=results, require_server=False,
    )
    assert summary["completed"] == 1
    assert summary["halted"] == 1
    assert summary["reason"] == "storage_budget_exhausted"
    assert (results / first_run / "contract_result.json").is_file()
    assert not (results / second_run).exists()
    state_path = _plan_state_path(results, plan)
    assert summary["state_path"] == str(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["execution"]["phase"] == "before_job"
    assert state["execution"]["next_run_id"] == second_run
    assert second_run not in state["runs"]


def test_per_plan_state_allows_disjoint_completed_plans_in_one_results_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    results = project / "results"

    def fail_evaluator(command, *, cwd, environment, log, wall_timeout):
        del command, cwd, environment, log, wall_timeout
        raise OSError("injected finalized launch failure")

    monkeypatch.setattr(runner_module, "_run_evaluator", fail_evaluator)
    plan_a = _lock_test_plan(
        ["completed-plan-a-run"], plan_sha256="completed-plan-a",
    )
    summary_a = execute_plan(
        plan_a, project_root=project, results_dir=results,
        require_server=False,
    )
    state_a_path = _plan_state_path(results, plan_a)
    state_a_before = state_a_path.read_bytes()
    assert summary_a == {
        "planned": 1, "completed": 1, "failed": 0, "halted": 0,
        "status": "completed", "state_path": str(state_a_path),
    }

    plan_b = _lock_test_plan(
        ["completed-plan-b-run"], plan_sha256="completed-plan-b",
    )
    summary_b = execute_plan(
        plan_b, project_root=project, results_dir=results,
        require_server=False,
    )
    state_b_path = _plan_state_path(results, plan_b)
    assert summary_b == {
        "planned": 1, "completed": 1, "failed": 0, "halted": 0,
        "status": "completed", "state_path": str(state_b_path),
    }
    assert state_a_path != state_b_path
    assert state_a_path.read_bytes() == state_a_before
    assert set(json.loads(state_a_before)["runs"]) == {"completed-plan-a-run"}
    assert set(json.loads(state_b_path.read_text(encoding="utf-8"))["runs"]) == {
        "completed-plan-b-run",
    }
    assert (results / "completed-plan-a-run" / "contract_result.json").is_file()
    assert (results / "completed-plan-b-run" / "contract_result.json").is_file()
    assert not (results / "run_ledger_state.json").exists()


def test_pending_only_recovery_plan_uses_independent_per_plan_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    results = project / "results"
    first_run, pending_run = "partial-plan-a-done", "partial-plan-a-pending"
    plan_a = _lock_test_plan(
        [first_run, pending_run], plan_sha256="partial-plan-a",
    )
    estimate = plan_a["jobs"][0]["storage_estimate"][
        "estimated_transient_capture_bytes"
    ]
    enough = runner_module.STORAGE_RESERVE_BYTES + estimate + 1
    insufficient = runner_module.STORAGE_RESERVE_BYTES + estimate - 1
    availability = iter((enough, enough, insufficient))
    monkeypatch.setattr(
        runner_module, "_storage_available_bytes", lambda path: next(availability),
    )

    def fail_evaluator(command, *, cwd, environment, log, wall_timeout):
        del command, cwd, environment, log, wall_timeout
        raise OSError("injected finalized launch failure")

    monkeypatch.setattr(runner_module, "_run_evaluator", fail_evaluator)
    partial = execute_plan(
        plan_a, project_root=project, results_dir=results,
        require_server=False,
    )
    state_a_path = _plan_state_path(results, plan_a)
    state_a_before = state_a_path.read_bytes()
    assert partial["status"] == "execution_halted"
    assert partial["reason"] == "storage_budget_exhausted"
    assert partial["completed"] == 1
    pending_ids = [
        job["run"]["run_id"] for job in plan_a["jobs"]
        if not (results / job["run"]["run_id"] / "contract_result.json").is_file()
    ]
    assert pending_ids == [pending_run]

    recovery = _lock_test_plan(
        pending_ids, plan_sha256="partial-plan-a-pending-only-recovery",
    )
    monkeypatch.setattr(
        runner_module, "_storage_available_bytes", lambda path: enough,
    )
    recovered = execute_plan(
        recovery, project_root=project, results_dir=results,
        require_server=False,
    )
    state_recovery_path = _plan_state_path(results, recovery)
    assert recovered == {
        "planned": 1, "completed": 1, "failed": 0, "halted": 0,
        "status": "completed", "state_path": str(state_recovery_path),
    }
    assert state_recovery_path != state_a_path
    assert state_a_path.read_bytes() == state_a_before
    assert set(json.loads(state_recovery_path.read_text(encoding="utf-8"))["runs"]) == {
        pending_run,
    }
    assert (results / first_run / "contract_result.json").is_file()
    assert (results / pending_run / "contract_result.json").is_file()


def test_evaluator_log_compaction_retains_bounded_head_tail_and_digest(tmp_path: Path):
    log = tmp_path / "evaluator.stdout.log"
    original = b"HEAD_TOKEN\n" + b"a" * 4000 + b"\nTAIL_TOKEN"
    log.write_bytes(original)
    metadata = runner_module._compact_evaluator_log(log, max_bytes=1024)
    stored = log.read_bytes()
    assert metadata["compacted"] is True
    assert metadata["original_size_bytes"] == len(original)
    assert metadata["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert len(stored) <= 1024
    assert b"HEAD_TOKEN" in stored
    assert b"TAIL_TOKEN" in stored
    assert metadata["original_sha256"].encode("ascii") in stored
    sidecar = json.loads(
        log.with_name("evaluator.stdout.log.bounded.json").read_text(encoding="utf-8")
    )
    assert sidecar == metadata


def test_evaluator_stdout_is_streamed_into_bounded_head_tail(tmp_path: Path):
    log_path = tmp_path / "evaluator.stdout.log"
    capture = runner_module._StreamingBoundedLog(log_path, max_bytes=1024)
    original = b"STREAM_HEAD\n" + b"z" * 8192 + b"\nSTREAM_TAIL"
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(" + repr(original) + "); sys.stdout.flush()",
    ]
    returncode, timed_out = runner_module._run_evaluator(
        command,
        cwd=tmp_path,
        environment=dict(os.environ),
        log=capture,
        wall_timeout=10.0,
    )
    metadata = capture.finalize()
    stored = log_path.read_bytes()
    assert (returncode, timed_out) == (0, False)
    assert metadata["capture_mode"] == "streaming_pipe_incremental_sha256_head_tail"
    assert metadata["original_size_bytes"] == len(original)
    assert metadata["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["compacted"] is True
    assert len(stored) <= 1024
    assert b"STREAM_HEAD" in stored
    assert b"STREAM_TAIL" in stored


def test_successful_evaluator_leader_cannot_leave_a_process_group_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    child_pid_path = tmp_path / "child.pid"
    child = (
        "import os,pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()),encoding='utf-8');"
        "time.sleep(300)"
    )
    leader = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]]);"
        "deadline=time.monotonic()+5;"
        "exec(\"while not pathlib.Path(sys.argv[1]).is_file():\\n"
        "    assert time.monotonic()<deadline\\n"
        "    time.sleep(.01)\");"
        "print('leader-complete',flush=True)"
    )
    monkeypatch.setattr(
        runner_module, "EVALUATOR_TERMINATE_GRACE_SECONDS", 0.25,
    )
    capture = runner_module._StreamingBoundedLog(
        tmp_path / "successful-leader.log", max_bytes=4096,
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="leader exited with a surviving process-group member",
        ):
            runner_module._run_evaluator(
                [sys.executable, "-c", leader, str(child_pid_path), child],
                cwd=tmp_path, environment=dict(os.environ), log=capture,
                wall_timeout=10.0,
            )
    finally:
        capture.finalize()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while _pid_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _pid_is_running(child_pid)


@pytest.mark.parametrize(
    "injected",
    [
        KeyboardInterrupt("injected keyboard interrupt"),
        SystemExit(17),
        _InjectedRunnerBaseException("injected base exception"),
    ],
    ids=["keyboard-interrupt", "system-exit", "base-exception"],
)
def test_evaluator_base_exception_kills_group_before_lock_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
):
    project = tmp_path / "project"
    results = project / "results"
    pids_path = tmp_path / "evaluator-pids.json"
    grandchild_ready = tmp_path / "grandchild-ready.txt"
    grandchild = (
        "import os, pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(300)"
    )
    evaluator = (
        "import json, os, pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "child=subprocess.Popen([sys.executable, '-c', sys.argv[3], sys.argv[2]]); "
        "ready=pathlib.Path(sys.argv[2]); "
        "deadline=time.monotonic()+5; "
        "exec(\"while not ready.is_file():\\n"
        "    assert time.monotonic() < deadline\\n"
        "    time.sleep(0.01)\"); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'leader': os.getpid(), "
        "'grandchild': child.pid}), encoding='utf-8'); "
        "print('evaluator-ready', flush=True); "
        "time.sleep(300)"
    )
    plan = _lock_test_plan(["interrupted-evaluator"])
    plan["jobs"][0]["command"] = [
        sys.executable,
        "-c",
        evaluator,
        str(pids_path),
        str(grandchild_ready),
        grandchild,
    ]

    def injected_select(*args, **kwargs):
        del args, kwargs
        deadline = time.monotonic() + 5.0
        while not pids_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError("evaluator process group did not become ready")
            time.sleep(0.01)
        raise injected

    monkeypatch.setattr(runner_module.select, "select", injected_select)
    monkeypatch.setattr(
        runner_module, "EVALUATOR_TERMINATE_GRACE_SECONDS", 0.05
    )
    real_killpg = runner_module.os.killpg
    termination_signals: list[int] = []

    def tracked_killpg(process_group_id, sent_signal):
        if sent_signal in {signal.SIGTERM, signal.SIGKILL}:
            termination_signals.append(sent_signal)
        return real_killpg(process_group_id, sent_signal)

    monkeypatch.setattr(runner_module.os, "killpg", tracked_killpg)
    real_flock = runner_module.fcntl.flock
    running_at_unlock: list[dict[str, bool]] = []

    def checked_flock(descriptor, operation):
        if operation == runner_module.fcntl.LOCK_UN:
            pids = json.loads(pids_path.read_text(encoding="utf-8"))
            running_at_unlock.append({
                name: _pid_is_running(pid) for name, pid in pids.items()
            })
        return real_flock(descriptor, operation)

    monkeypatch.setattr(runner_module.fcntl, "flock", checked_flock)
    observed_pids: dict[str, int] = {}
    try:
        with pytest.raises(type(injected)) as caught:
            execute_plan(
                plan,
                project_root=project,
                results_dir=results,
                require_server=False,
            )
        assert caught.value is injected
        observed_pids = json.loads(pids_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2.0
        while any(_pid_is_running(pid) for pid in observed_pids.values()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert not any(_pid_is_running(pid) for pid in observed_pids.values())
        with pytest.raises(ChildProcessError):
            os.waitpid(observed_pids["leader"], os.WNOHANG)
        assert termination_signals == [signal.SIGTERM, signal.SIGKILL]
        assert running_at_unlock == [{"leader": False, "grandchild": False}]
        lock_record = json.loads(
            _carla_lock_path(
                project, "127.0.0.1", 2000, 8000
            ).read_text(encoding="utf-8")
        )
        assert lock_record["status"] == "released"
    finally:
        # Keep a failing regression test from leaking the exact process group
        # whose lifecycle it is intended to diagnose.
        if observed_pids:
            try:
                os.killpg(observed_pids["leader"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_core_capture_merge_requires_exact_namespace_provenance_and_no_collision(
    tmp_path: Path,
):
    run = {
        "run_id": "oracle__A-001__left__p0__s1", "pcu_id": "A-001",
        "arm_id": "left", "paraphrase": 0, "seed": 1,
        "route_entry_id": 0, "track": "oracle", "split": "dev",
    }
    run_dir = tmp_path / "run"
    alias = run_dir / "_evaluator_record"
    child = alias / "native__A-001__left__p0__s1"
    _write_core_capture_fixture(
        child, run, evaluator_track="native",
        private_sidecar_sha256="a" * 64,
    )
    runner_module._merge_core_evaluator_capture(
        alias, run_dir, run, evaluator_track="native",
        private_sidecar_sha256="a" * 64,
    )
    assert (run_dir / "trace.jsonl").is_file()
    assert not alias.exists()

    _write_core_capture_fixture(
        child, run, evaluator_track="native",
        private_sidecar_sha256="a" * 64,
    )
    (run_dir / "trace.jsonl").write_text("runner-owned\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="collides"):
        runner_module._merge_core_evaluator_capture(
            alias, run_dir, run, evaluator_track="native",
            private_sidecar_sha256="a" * 64,
        )
    assert (run_dir / "trace.jsonl").read_text(encoding="utf-8") == "runner-owned\n"

    for path in child.iterdir():
        path.unlink()
    child.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    child.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="namespace is not a real directory"):
        runner_module._merge_core_evaluator_capture(
            alias, run_dir, run, evaluator_track="native",
            private_sidecar_sha256="a" * 64,
        )


def test_standardized_capture_merge_accepts_private_seed_omission_and_rejects_leak(
    tmp_path: Path,
):
    run = {
        "run_id": "standardized__A-001__left__p0__s1", "pcu_id": "A-001",
        "arm_id": "left", "paraphrase": 0, "seed": 1,
        "route_entry_id": 0, "track": "standardized", "split": "dev",
    }
    accepted_dir = tmp_path / "accepted" / run["run_id"]
    accepted_alias = accepted_dir / "_evaluator_record"
    accepted_child = accepted_alias / run["run_id"]
    _write_core_capture_fixture(
        accepted_child, run, evaluator_track="standardized",
        private_sidecar_sha256="a" * 64,
    )
    runner_module._merge_core_evaluator_capture(
        accepted_alias, accepted_dir, run, evaluator_track="standardized",
        private_sidecar_sha256="a" * 64,
    )
    assert (accepted_dir / "trace.jsonl").is_file()
    current_bundle = {
        "evaluator_track": "standardized",
        "private_sidecar_sha256": "a" * 64,
    }
    _write_json(accepted_dir / "runner_command.json", {
        "run": run,
        "input_hashes": current_bundle,
    })
    assert _result_is_current_for_sidecar(
        accepted_dir / "contract_result.json",
        None,
        current_bundle,
        expected_run=run,
    ) is True

    rejected_dir = tmp_path / "rejected" / run["run_id"]
    rejected_alias = rejected_dir / "_evaluator_record"
    rejected_child = rejected_alias / run["run_id"]
    _write_core_capture_fixture(
        rejected_child, run, evaluator_track="standardized",
        private_sidecar_sha256="a" * 64,
    )
    for name in ("run_meta.json", "route_meta.json"):
        path = rejected_child / name
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["language2drive_public_context"]["seed"] = run["seed"]
        _write_json(path, metadata)
    with pytest.raises(RuntimeError, match="identity/provenance differs"):
        runner_module._merge_core_evaluator_capture(
            rejected_alias, rejected_dir, run, evaluator_track="standardized",
            private_sidecar_sha256="a" * 64,
        )


def test_fdq_capture_merge_accepts_alpamayo_agent_reports(tmp_path: Path):
    run = {
        "run_id": "fdq__overtaking__s1", "ability": "overtaking", "seed": 1,
    }
    run_dir = tmp_path / run["run_id"]
    alias = run_dir / "_evaluator_record"
    binding = runner_module._canonical_fdq_route_bindings()[run["ability"]]
    child = alias / f"RouteScenario_{binding['route_id']}_rep0"
    _write_fdq_capture_fixture(child, run)
    (child / "model_plans.jsonl").write_text(
        '{"step":0,"waypoints":[[0.0,0.0],[1.0,0.0]]}\n'
        '{"step":1,"waypoints":[[1.0,0.0],[2.0,0.0]]}\n',
        encoding="utf-8",
    )

    runner_module._merge_fdq_evaluator_capture(alias, run_dir, run)

    assert json.loads(
        (run_dir / "alpamayo_load_report.json").read_text(encoding="utf-8")
    )["model"] == "alpamayo-1.5"
    assert json.loads(
        (run_dir / "alpamayo_agent_summary.json").read_text(encoding="utf-8")
    )["steps"] == 1
    assert (run_dir / "model_plans.jsonl").is_file()
    assert not alias.exists()


def test_trace_alias_is_hardlinked_only_after_exact_hash_proof(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace = run_dir / "trace.jsonl"
    frames = run_dir / "frames.jsonl"
    payload = b'{"frame": 1}\n'
    trace.write_bytes(payload)
    frames.write_bytes(payload)
    assert trace.stat().st_ino != frames.stat().st_ino
    metadata = runner_module._deduplicate_trace_alias(run_dir)
    assert metadata["status"] == "deduplicated_verified_equal"
    assert metadata["projection_satisfied"] is True
    assert metadata["bytes_saved"] == len(payload)
    assert trace.stat().st_ino == frames.stat().st_ino

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    (mismatched / "trace.jsonl").write_bytes(b"same-size-a")
    (mismatched / "frames.jsonl").write_bytes(b"same-size-b")
    mismatch = runner_module._deduplicate_trace_alias(mismatched)
    assert mismatch["status"] == "not_deduplicated_hash_mismatch"
    assert mismatch["projection_satisfied"] is False
    assert (mismatched / "trace.jsonl").stat().st_ino != (mismatched / "frames.jsonl").stat().st_ino


def test_execute_plan_lock_contention_fails_before_any_verdict_write(tmp_path: Path):
    project = tmp_path / "project"
    results = project / "results"
    plan = _lock_test_plan(["contended-run"])
    state_path = _plan_state_path(results, plan)
    _write_json(state_path, {"sentinel": "unchanged"})
    state_path.parent.chmod(0o700)
    state_before = state_path.read_bytes()
    lock_path = _carla_lock_path(project, "127.0.0.1", 2000, 8000)
    holder = _start_external_lock_holder(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already owned by another runner"):
            execute_plan(
                plan,
                project_root=project,
                results_dir=results,
                state_path=state_path,
                require_server=False,
        )
        assert state_path.read_bytes() == state_before
        assert not (results / "contended-run").exists()
    finally:
        _release_external_lock_holder(holder)


def test_execute_plan_lock_releases_after_exception_and_records_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    results = project / "results"
    bad_plan = _lock_test_plan(["unused"])
    original_execute_locked = runner_module._execute_plan_locked
    monkeypatch.setattr(
        runner_module, "_execute_plan_locked",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("run_id")),
    )
    with pytest.raises(KeyError, match="run_id"):
        execute_plan(
            bad_plan, project_root=project, results_dir=results, require_server=False,
        )

    released = json.loads(
        _carla_lock_path(project, "127.0.0.1", 2000, 8000).read_text(encoding="utf-8")
    )
    assert released["pid"] == os.getpid()
    assert released["status"] == "released"
    assert isinstance(released["released_at"], str)

    monkeypatch.setattr(runner_module, "_execute_plan_locked", original_execute_locked)
    good_plan = _lock_test_plan(["after-exception"])
    summary = execute_plan(
        good_plan, project_root=project, results_dir=results, require_server=False,
    )
    assert summary["planned"] == 1
    assert summary["halted"] == 0
    lock_record = json.loads(
        _carla_lock_path(project, "127.0.0.1", 2000, 8000).read_text(encoding="utf-8")
    )
    assert lock_record["pid"] == os.getpid()
    assert lock_record["status"] == "released"
    assert isinstance(lock_record["released_at"], str)


def test_execute_plan_lock_is_independent_for_different_carla_port(tmp_path: Path):
    project = tmp_path / "project"
    held_path = _carla_lock_path(project, "127.0.0.1", 2000, 8000)
    holder = _start_external_lock_holder(held_path)
    try:
        summary = execute_plan(
            _lock_test_plan(["other-port"], port=2001),
            project_root=project,
            results_dir=project / "results/other-port",
            require_server=False,
        )
        assert summary["planned"] == 1
        assert summary["halted"] == 0
        assert _carla_lock_path(project, "127.0.0.1", 2001, 8000).is_file()
    finally:
        _release_external_lock_holder(holder)


def test_execute_plan_halts_remaining_jobs_when_server_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    results = project / "results"
    first_run, second_run = "first-run", "second-run"
    availability = iter((True, True, False))
    monkeypatch.setattr(
        runner_module,
        "carla_server_available",
        lambda host, port: next(availability),
    )

    def fake_run_evaluator(command, *, cwd, environment, log, wall_timeout):
        del command, cwd, environment, log, wall_timeout
        raise OSError("injected evaluator launch failure")

    monkeypatch.setattr(runner_module, "_run_evaluator", fake_run_evaluator)
    plan = _lock_test_plan([first_run, second_run])
    summary = execute_plan(
        plan,
        project_root=project,
        results_dir=results,
        require_server=True,
    )
    assert summary == {
        "planned": 2,
        "completed": 1,
        "failed": 0,
        "halted": 1,
        "status": "execution_halted",
        "reason": "server_unavailable",
        "state_path": str(_plan_state_path(results, plan)),
    }
    assert (results / first_run / "contract_result.json").is_file()
    assert not (results / second_run).exists()
    state = json.loads(
        _plan_state_path(results, plan).read_text(encoding="utf-8")
    )
    assert state["execution"]["status"] == "execution_halted"
    assert state["execution"]["reason"] == "server_unavailable"
    assert state["execution"]["halted_runs"] == 1
    assert second_run not in state["runs"]


def test_supplementary_workloads_are_semantic_opaque_and_complete():
    project = Path(__file__).resolve().parents[1]
    built = build_supplementary(project)
    assert built["safety_priority"]["expected_run_count"] == 6
    assert built["dev_antigaming"]["baseline_count"] == 6
    assert built["dev_antigaming"]["total_expected_runs"] == 2592
    assert validate_supplementary(project)["valid"] is True
    assert len(SAFETY_CASES) == 6
    assert {case["hazard_class"] for case in SAFETY_CASES} >= {
        "dangerous", "illegal", "physically_infeasible",
    }
    assert {case["response_mode"] for case in SAFETY_CASES} >= {
        "safe_refusal", "safe_fallback", "constrained_alternative",
    }
    forbidden = {"pcu_id", "arm_id", "paraphrase", "layer", "source_pcu_id"}
    for xml_path in (
        project / "build/safety/routes_standardized.xml",
        project / "build/baselines/routes_no_language.xml",
        project / "build/baselines/routes_shuffled_language.xml",
        project / "build/baselines/routes_standardized_dev.xml",
    ):
        for route in ET.parse(xml_path).getroot().findall("route"):
            for node in (route, route.find("simlingo_metadata"), route.find("l2d_ref")):
                assert node is None or not forbidden.intersection(node.attrib)
    sidecar_mode = stat.S_IMODE((project / "build/safety/sidecar.private.json").stat().st_mode)
    assert sidecar_mode == 0o600
    main_sidecar = json.loads(
        (project / "build/suite/sidecar.private.json").read_text(encoding="utf-8")
    )
    safety_sidecar = json.loads(
        (project / "build/safety/sidecar.private.json").read_text(encoding="utf-8")
    )
    assert safety_sidecar["parking_manoeuvre_registry"] == main_sidecar["parking_manoeuvre_registry"]
    assert safety_sidecar["source_files"] == main_sidecar["source_files"]
    main_routes = {
        (route["pcu_id"], route["arm_id"]): route for route in main_sidecar["routes"]
    }
    parking_safety_routes = [
        route for route in safety_sidecar["routes"]
        if "parking_v5_route_proof" in route
    ]
    assert len(parking_safety_routes) == 2
    for route in parking_safety_routes:
        safety_pcu = next(pcu for pcu in safety_sidecar["pcus"] if pcu["id"] == route["pcu_id"])
        safety_arm = safety_pcu["arms"][0]
        source = main_routes[(safety_arm["source_pcu_id"], safety_arm["source_arm_id"])]
        assert route["parking_v5_route_proof"] == safety_arm["parking_v5_route_proof"] == source["parking_v5_route_proof"]
        assert route["intended_path_length_m"] == safety_arm["intended_path_length_m"] == source["intended_path_length_m"]


def test_safety_sidecar_parking_proof_closure_is_exact_and_fail_closed():
    registry = {"schema_version": "parking-test.v1", "towns": {"T": {"proof": "x"}}}
    source_files = {"/data/source.trace": {"sha256": "a" * 64}}
    main = {
        "parking_manoeuvre_registry": registry,
        "source_files": source_files,
    }
    sidecar = _safety_sidecar_payload(main, [{"id": "S-001"}], [{"route_entry_id": 0}])
    assert sidecar["parking_manoeuvre_registry"] == registry
    assert sidecar["source_files"] == source_files
    assert sidecar["parking_manoeuvre_registry"] is not registry
    assert sidecar["source_files"] is not source_files
    for missing in ("parking_manoeuvre_registry", "source_files"):
        incomplete = dict(main)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match="parking-proof closure"):
            _safety_sidecar_payload(incomplete, [], [])


def test_safety_missing_runs_remain_in_denominator(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    ledger = project / "build/safety/expected_runs.json"
    value = _safety_summary(ledger, tmp_path / "results")
    assert value["expected_runs"] == 6
    assert value["missing_runs"] == 6
    assert value["Safety-Priority-Success"] == 0.0
    assert value["Safe-Refusal"]["expected"] == 2
    assert value["Clarification-Proxy-Correctness"]["expected"] == 0
    assert value["Fallback-or-Constrained-Alternative"]["expected"] == 4
    assert value["Unnecessary-Refusal"]["value"] is None
    assert value["four_quadrants"]["Unobserved-or-invalid"] == 6


def test_deadline_calibration_fails_closed_without_oracle_witnesses(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "deadline.json"
    build_before = {
        path.name: path.read_bytes()
        for path in (project / "build/suite").iterdir()
        if path.is_file()
    }
    value = calibrate_deadlines(project, tmp_path / "empty-results", output)
    assert value["status"] == "blocked_pending_oracle_witnesses"
    assert value["required_runs"] == 768
    assert value["valid_runs"] == 0
    assert value["calibrated_arms"] == 0
    assert value["entries"] == []
    assert not output.exists()
    build_after = {
        path.name: path.read_bytes()
        for path in (project / "build/suite").iterdir()
        if path.is_file()
    }
    assert build_after == build_before
    with pytest.raises(FileNotFoundError):
        freeze_deadlines(project, output)
    assert {
        path.name: path.read_bytes()
        for path in (project / "build/suite").iterdir()
        if path.is_file()
    } == build_before

    _write_json(output, {"status": "blocked_pending_oracle_witnesses", "stale": True})
    value = calibrate_deadlines(project, tmp_path / "empty-results", output)
    assert value["valid_runs"] == 0
    assert not output.exists()


def test_fdq_recovery_uses_exact_sidecar_free_execution_bundle(tmp_path: Path):
    run = {
        "run_id": "fdq__overtaking__s1", "ability": "overtaking",
        "seed": 1, "status": "expected",
    }
    ledger = tmp_path / "fdq.json"
    _write_json(ledger, {"expected_count": 1, "runs": [run]})
    run_dir = tmp_path / "results" / run["run_id"]
    result = _write_fdq_normalized_fixture(run_dir, run, failed=False)
    assert _result_is_final(result, run) is True
    current = resolve_fdq_inputs(agent="oracle")["input_hashes"]
    assert "private_sidecar_sha256" not in current
    _write_json(result.with_name("runner_command.json"), {
        "run": run,
        "input_hashes": {**current, "evaluator_sha256": "stale"},
    })
    selected, summary = select_fdq_runs(
        ledger, tmp_path / "results", seeds=[1], expected_input_hashes=current,
    )
    assert selected == [run]
    assert summary["skipped_finalized"] == 0
    _write_json(result.with_name("runner_command.json"), {
        "run": run, "input_hashes": current,
    })
    selected, summary = select_fdq_runs(
        ledger, tmp_path / "results", seeds=[1], expected_input_hashes=current,
    )
    assert selected == []
    assert summary["skipped_finalized"] == 1


def test_active_core_sidecar_is_base_when_no_freeze_exists(tmp_path: Path):
    project = tmp_path / "project"
    base = project / "build/suite/sidecar.private.json"
    _write_json(base, {"pcus": [], "routes": []})
    assert active_core_sidecar_path(project) == base
    assert deadline_freeze_status(project)["status"] == "not_frozen"


def test_plan_describes_private_sidecar_as_trusted_api_boundary(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    plan = make_plan(
        [],
        project_root=project,
        build_dir=project / "build/suite",
        results_dir=tmp_path / "results",
        track="oracle",
        agent="oracle",
    )
    note = plan["private_sidecar_visibility"]
    assert "Trusted same-process engineering API boundary" in note
    assert "not OS/process security isolation" in note
    assert "Evaluator process only" not in note
    assert plan["storage_policy"]["schema_version"] == "language2drive.storage_policy.v1"
    assert plan["storage_policy"]["minimum_free_reserve_bytes"] == 4 * 1024 ** 3
    assert plan["storage_policy"]["historical_upper_bytes_per_sim_second"] == 150_000
    assert plan["storage_policy"]["hardlink_projection"]["projected_bytes_per_sim_second"] == 75_000
    assert plan["storage_policy"]["fixed_overhead_bytes"] == 17 * 1024 ** 2
    assert plan["estimated_plan_storage_bytes"] == 0
    assert len(plan["input_hashes"]["storage_policy_sha256"]) == 64


def test_autovla_wall_timeout_policy_covers_low_and_high_deadlines():
    policy = runner_module._wall_timeout_policy(agent="autovla", workload="core")
    assert policy["sim_time_multiplier"] == 12.0
    assert policy["provenance"]["observed_system_to_sim_time_ratios"] == [10.70, 11.05]
    assert runner_module._job_wall_timeout(
        {"deadline_seconds": 30.0}, None, policy,
    ) == 900.0
    assert runner_module._job_wall_timeout(
        {"deadline_seconds": 67.5}, None, policy,
    ) == 1110.0
    assert runner_module._job_wall_timeout(
        {"deadline_seconds": 240.0}, None, policy,
    ) == 3180.0
    assert runner_module._job_wall_timeout({}, None, policy) == 900.0

    oracle_policy = runner_module._wall_timeout_policy(agent="oracle", workload="core")
    assert runner_module._job_wall_timeout(
        {"deadline_seconds": 240.0}, None, oracle_policy,
    ) == 540.0
    assert runner_module._job_wall_timeout(
        {"deadline_seconds": 240.0}, 123.0, policy,
    ) == 123.0
    storage = runner_module._storage_policy()
    low = runner_module._job_storage_estimate({"deadline_seconds": 30.0}, storage)
    high = runner_module._job_storage_estimate({"deadline_seconds": 240.0}, storage)
    fdq = runner_module._job_storage_estimate({"deadline_seconds": None}, storage)
    fixed = 17 * 1024 ** 2
    assert low["estimated_bytes"] == fixed + 2_250_000
    assert low["estimated_transient_capture_bytes"] == fixed + 4_500_000
    assert high["estimated_bytes"] == fixed + 18_000_000
    assert high["estimated_transient_capture_bytes"] == fixed + 36_000_000
    assert high["estimated_bytes_with_trace_hardlink_projection"] == fixed + 18_000_000
    assert fdq["estimated_bytes"] == fixed + 67_500_000
    assert fdq["estimated_transient_capture_bytes"] == fixed + 135_000_000


def test_deadline_freeze_is_exact_atomic_and_live_hash_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project, results, runs = _deadline_freeze_fixture(tmp_path)
    build = project / "build/suite"
    base_path = build / "sidecar.private.json"
    base_before = base_path.read_bytes()

    def fake_rescore(run, **kwargs):
        completion = float(run["seed"] * 2)
        return {
            "status": "complete", "success": True,
            "completion_time": completion,
            "deadline_seconds": 10.0,
        }

    monkeypatch.setattr(runner_module, "_score_deadline_witness", fake_rescore)
    calibration_path = project / "generated/deadlines.calibrated.json"
    calibration = calibrate_deadlines(project, results, calibration_path)
    assert calibration["status"] == "passed"
    assert calibration["valid_runs"] == 768
    assert calibration["calibrated_arms"] == 256
    assert {entry["deadline_seconds"] for entry in calibration["entries"]} == {14}
    assert base_path.read_bytes() == base_before

    frozen = freeze_deadlines(project, calibration_path)
    assert frozen["status"] == "frozen"
    assert frozen["valid"] is True
    assert frozen["changed"] is True
    assert base_path.read_bytes() == base_before
    derived_path = build / "sidecar.deadlines-frozen.private.json"
    overlay_path = build / "deadlines.frozen.private.json"
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    assert {
        arm["deadline_seconds"] for pcu in derived["pcus"] for arm in pcu["arms"]
    } == {14}
    assert {route["deadline_seconds"] for route in derived["routes"]} == {14}
    assert active_core_sidecar_path(project) == derived_path
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay["new_deadline_rescore"]["valid_runs"] == 768
    assert len(overlay["witnesses"]) == 768

    route_xml = build / "routes_native.xml"
    route_xml_before = route_xml.read_bytes()
    route_xml.write_text("<routes><!-- changed --></routes>\n", encoding="utf-8")
    status = deadline_freeze_status(project, verify_witnesses=False)
    assert status["valid"] is False
    assert "deadline freeze oracle execution bundle is no longer current" in status["errors"]
    route_xml.write_bytes(route_xml_before)
    assert deadline_freeze_status(project, verify_witnesses=False)["valid"] is True

    first_trace = results / runs[0]["run_id"] / "trace.jsonl"
    first_trace.write_text('{"sim_time": 99}\n', encoding="utf-8")
    status = deadline_freeze_status(project)
    assert status["valid"] is False
    assert any("frozen witness hash mismatch for trace" in error for error in status["errors"])


def test_deadline_freeze_revalidates_bundle_and_never_applies_stale_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    project, results, runs = _deadline_freeze_fixture(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "_score_deadline_witness",
        lambda run, **kwargs: {
            "status": "complete", "success": True,
            "completion_time": float(run["seed"]), "deadline_seconds": 10.0,
        },
    )
    calibration_path = project / "generated/deadlines.calibrated.json"
    assert calibrate_deadlines(project, results, calibration_path)["status"] == "passed"
    build = project / "build/suite"
    snapshot = {
        path.name: path.read_bytes() for path in build.iterdir() if path.is_file()
    }
    stale_command = results / runs[0]["run_id"] / "runner_command.json"
    command = json.loads(stale_command.read_text(encoding="utf-8"))
    command["input_hashes"]["agent_sha256"] = "changed-agent"
    _write_json(stale_command, command)
    with pytest.raises(ValueError, match="no longer complete/current"):
        freeze_deadlines(project, calibration_path)
    assert not (build / "sidecar.deadlines-frozen.private.json").exists()
    assert not (build / "deadlines.frozen.private.json").exists()
    assert {
        path.name: path.read_bytes() for path in build.iterdir() if path.is_file()
    } == snapshot


def test_runner_wall_timeout_writes_terminal_verdict(tmp_path: Path):
    run_id = "standardized__A-001__left__p0__s1"
    plan = _with_storage_policy({
        "plan_sha256": "unit-test",
        "workload": "core",
        "carla_server_prerequisite": "disabled in unit test",
        "jobs": [{
            "run": {"run_id": run_id},
            "command": ["/bin/sleep", "10"],
            "deadline_seconds": 1,
        }],
    })
    value = execute_plan(
        plan, project_root=tmp_path, results_dir=tmp_path / "results", require_server=False,
        wall_timeout_seconds=0.05,
    )
    result = json.loads((tmp_path / "results" / run_id / "contract_result.json").read_text())
    assert value["completed"] == 1
    assert result["status"] == "timeout"
    assert result["reason"] == "runner_wall_timeout"


def test_runner_launch_error_writes_terminal_verdict(tmp_path: Path):
    run_id = "standardized__A-001__left__p0__s1"
    plan = _with_storage_policy({
        "plan_sha256": "unit-test-launch",
        "workload": "core",
        "jobs": [{"run": {"run_id": run_id}, "command": [str(tmp_path / "absent-command")]}],
    })
    value = execute_plan(
        plan, project_root=tmp_path, results_dir=tmp_path / "results", require_server=False,
    )
    result = json.loads((tmp_path / "results" / run_id / "contract_result.json").read_text())
    assert value["completed"] == 1
    assert result["status"] == "crash"
    assert result["reason"] == "runner_launch_error"


def test_runner_zero_exit_without_result_is_retryable_capture_failure(tmp_path: Path):
    run_id = "standardized__A-001__left__p0__s1"
    plan = _with_storage_policy({
        "plan_sha256": "unit-test-capture",
        "workload": "core",
        "jobs": [{"run": {"run_id": run_id}, "command": ["/bin/true"]}],
    })
    value = execute_plan(
        plan, project_root=tmp_path, results_dir=tmp_path / "results", require_server=False,
    )
    result_path = tmp_path / "results" / run_id / "contract_result.json"
    result = json.loads(result_path.read_text())
    assert value["failed"] == 1
    assert result["status"] == "capture_failure"
    assert result["capture_complete"] is False
    assert _result_is_final(result_path) is False


def test_execute_rejects_existing_run_directory_before_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    run_id = "standardized__A-001__left__p0__s1"
    results = tmp_path / "results"
    run_dir = results / run_id
    run_dir.mkdir(parents=True)
    stale = run_dir / "contract_result.json"
    stale.write_text('{"status":"complete","success":true}\n', encoding="utf-8")
    before = stale.read_bytes()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("evaluator must not run over an existing result directory")

    monkeypatch.setattr(runner_module, "_run_evaluator", forbidden)
    plan = _with_storage_policy({
        "plan_sha256": "unit-test-existing-attempt", "workload": "core",
        "jobs": [{"run": {"run_id": run_id}, "command": ["/bin/true"]}],
    })
    summary = execute_plan(
        plan, project_root=tmp_path, results_dir=results, require_server=False,
    )
    assert summary["status"] == "execution_halted"
    assert summary["reason"] == "existing_run_directory_requires_archive"
    assert stale.read_bytes() == before
    assert called is False


@pytest.mark.parametrize("run_id", ["../escape", "a/b", ".", "", "x" * 256])
def test_execute_rejects_unsafe_run_id_before_artifact_write(
    tmp_path: Path, run_id: str,
):
    plan = _with_storage_policy({
        "plan_sha256": "unit-test-unsafe-run", "workload": "core",
        "jobs": [{"run": {"run_id": run_id}, "command": ["/bin/true"]}],
    })
    with pytest.raises(ValueError, match="run_id is invalid"):
        execute_plan(
            plan, project_root=tmp_path, results_dir=tmp_path / "results",
            require_server=False,
        )
    assert not (tmp_path / "results").exists()


def test_post_evaluator_runtime_drift_invalidates_result_and_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    run_id = "standardized__A-001__left__p0__s1"
    results = tmp_path / "results"
    calls = 0

    def runtime_gate(plan, host, port):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("server changed during evaluator")
        return {"status": "same-server"}

    def fake_run(command, *, cwd, environment, log, wall_timeout):
        run_dir = results / run_id
        _write_json(run_dir / "contract_result.json", {
            "status": "complete", "success": True,
        })
        (run_dir / "trace.jsonl").write_text('{"sim_time":0}\n', encoding="utf-8")
        return 0, False

    monkeypatch.setattr(runner_module, "_validate_live_carla_runtime_authority", runtime_gate)
    monkeypatch.setattr(runner_module, "_run_evaluator", fake_run)
    plan = _with_storage_policy({
        "plan_sha256": "unit-test-post-runtime", "workload": "core",
        "jobs": [{"run": {"run_id": run_id}, "command": ["/bin/true"]}],
    })
    summary = execute_plan(
        plan, project_root=tmp_path, results_dir=results, require_server=False,
    )
    result = json.loads((results / run_id / "contract_result.json").read_text())
    assert summary["status"] == "execution_halted"
    assert summary["reason"] == "post_evaluator_authority_mismatch"
    assert result["status"] == "runtime_authority_failure"
    assert result["capture_complete"] is False
    assert _result_is_final(results / run_id / "contract_result.json") is False


def test_release_score_and_report_keep_constructs_separate(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    score_path = tmp_path / "score.json"
    value = score_suite(project, tmp_path / "empty-results", score_path)
    release_hashes = release_artifact_hashes(project)
    assert value["source"]["release_artifacts_sha256"] == release_hashes
    assert "build/safety/sidecar.private.json" in release_hashes
    assert "build/safety/routes_standardized.xml" in release_hashes
    assert "build/baselines/ledgers/no-language.json" in release_hashes
    assert "build/baselines/ledgers/dense-route-only.json" in release_hashes
    assert "build/baselines/routes_shuffled_language.xml" in release_hashes
    assert value["leaderboard"] == "standardized"
    assert set(value["dev_antigaming_baselines"]) == {
        "no-language", "shuffled-language", "always-left", "always-right",
        "fixed-branch", "dense-route-only",
    }
    assert value["safety_priority"]["expected_runs"] == 6
    assert value["fdq"]["expected_runs"] == 72
    assert value["fdq"]["aggregation"] == "mean_over_actual_applicable_ability_groups"
    autovla_bundle = resolve_plan_inputs(
        project_root=project,
        build_dir=project / "build/suite",
        track="standardized",
        agent="autovla",
    )["input_hashes"]
    assert autovla_bundle["schema_version"] == "language2drive.execution_bundle.v2"
    assert autovla_bundle["dependency_profile"] == "autovla-runtime.v1"
    assert len(autovla_bundle["wall_timeout_policy_sha256"]) == 64
    dependencies = autovla_bundle["dependency_hashes"]
    assert "language2drive.v1/environment.lock.yml" in dependencies
    assert "project/model_zoo/autovla/codebook_cache/agent_vocab.pkl" in dependencies
    assert "project/team_code/language2drive_context.py" in dependencies
    assert "project/model_zoo/autovla/code/models/action_tokenizer.py" in dependencies
    assert "project/model_zoo/autovla/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct/model.safetensors.index.json" in dependencies
    assert sum(key.endswith(".safetensors") for key in dependencies) == 2
    assert len(autovla_bundle["dependency_manifest_sha256"]) == 64
    assert set(value["standardized"]["C_task_breakdown"]) == {
        "branch_semantic_navigation", "region", "parking",
    }
    report_path = tmp_path / "REPORT.md"
    report = render(score_path, report_path, project_root=project, json_output=tmp_path / "REPORT.json")
    markdown = report_path.read_text(encoding="utf-8")
    assert report["paper_release_qualified"] is False
    assert report["FCS-2x2"]["reason"] == "fixed_layout_only"
    assert report["train_overlap"] == "unknown"
    assert report["parking_core_inventory"]["unique_slot_tokens"] == 52
    assert report["parking_core_inventory"]["unique_physical_xodr_parking_space_ids"] == 52
    assert report["parking_core_inventory"]["by_layer_unique_tokens"] == {
        "A": 8, "B": 0, "C": 15, "D": 42,
    }
    assert report["minimal_human_review"]["automatic_geometry_arm_joins"] == 16
    assert report["minimal_human_review"]["human_actions_remaining"] == 8
    assert "no linear total" in report["score_policy"]
    assert "## Safety Priority" in markdown
    assert "## Native-System secondary leaderboard" in markdown
    assert "## Dev anti-gaming controls" in markdown
    assert "Deadline calibration" in markdown
    assert "All 16 C-RD arm gate/name bindings already passed" in markdown
    assert "52 unique slot tokens" in markdown
    assert "trusted same-process" in markdown


def test_fdq_normalization_and_recovery_preserve_failed_outcome(tmp_path: Path):
    run = {
        "run_id": "fdq__overtaking__s1", "ability": "overtaking",
        "seed": 1, "status": "expected",
    }
    run_dir = tmp_path / "results" / run["run_id"]
    result_path = _write_fdq_normalized_fixture(run_dir, run, failed=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["success"] is False
    assert (result["DS"], result["RC"], result["IP"]) == (42.0, 100.0, 0.42)
    ledger = tmp_path / "fdq-ledger.json"
    _write_json(ledger, {"expected_count": 1, "runs": [run]})
    current = resolve_fdq_inputs(agent="oracle")["input_hashes"]
    _write_json(run_dir / "runner_command.json", {
        "run": run, "input_hashes": current,
    })
    selected, summary = select_fdq_runs(
        ledger, tmp_path / "results", seeds=[1],
        expected_input_hashes=current,
    )
    assert selected == []
    assert summary["skipped_finalized"] == 1
    value = _fdq_summary(
        ledger, tmp_path / "results", expected_input_hashes=current,
        expected_count=1,
    )
    assert value["finalized_runs"] == 1
    assert value["passed_runs"] == 0
    assert value["DS_RC_IP"]["DS"]["mean_valid_only"] == 42.0
