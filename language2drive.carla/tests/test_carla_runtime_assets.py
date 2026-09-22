from __future__ import annotations

import copy
import json
import os
import pwd
from pathlib import Path

import pytest

import language2drive.carla_runtime_assets as assets
import language2drive.runner as runner
from language2drive.io import atomic_json, stable_hash


ROOTS = ("CarlaUE4/Content", "CarlaUE4/Config")
REQUIRED = (
    "VERSION",
    "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping",
    "CarlaUE4/Config/DefaultGame.ini",
)
ABSENT = ("ContentVersions.txt",)


def _server_identity() -> tuple[int, int]:
    account = pwd.getpwnam(runner.CARLA_SERVER_ACCOUNT)
    return account.pw_uid, account.pw_gid


def _permission_authority() -> dict[str, object]:
    uid, gid = _server_identity()
    return {
        "server_readonly_permissions_sha256": "9" * 64,
        "server_readonly_permission_entry_count": 1,
        "server_uid": uid,
        "server_gids": [gid],
    }


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _runtime(tmp_path: Path) -> tuple[Path, Path]:
    real = tmp_path / "carla-real"
    _write(real / "VERSION", b"0.9.16-custom\n")
    _write(real / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping", b"ELF")
    _write(real / "CarlaUE4/Config/DefaultGame.ini", b"ProjectVersion=0.9.16\n")
    _write(real / "CarlaUE4/Content/Carla/Maps/Town01/Town01.umap", b"town01")
    _write(
        real / "CarlaUE4/Content/map_package/Maps/parkinglotfish-new3/"
        "parkinglotfish-new3.umap",
        b"parking",
    )
    logical = tmp_path / "carla-logical"
    logical.symlink_to(real, target_is_directory=True)
    return logical, real


def _process_receipt() -> dict[str, object]:
    uid, gid = _server_identity()
    command = [
        "/carla/Shipping", "CarlaUE4", "-RenderOffScreen", "-nosound",
        "-carla-port=2000", "-quality-level=Epic",
    ]
    launch = {
        "schema_version": "language2drive.carla_server_launch_receipt.v1",
        "boot_id": "test-boot", "prelaunch_monotonic_ticks": 12000,
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
    return {
        "uid": uid, "gid": gid, "pid": 123, "starttime_ticks": 12345,
        "clock_ticks_per_second": 100, "socket_inodes": [4242],
        "executable": "/carla/Shipping", "executable_dev": 1,
        "executable_ino": 2, "executable_size": 3,
        "executable_sha256": "f" * 64, "cmdline": command,
        "cmdline_sha256": stable_hash(command),
        "cwd": "/carla", "cwd_device": 1, "cwd_inode": 5,
        "root": "/", "root_device": 1, "root_inode": 1,
        "mount_namespace_inode": 4,
        "boot_id": "test-boot", "environment": {},
        "environment_sha256": stable_hash({}),
        "server_home_inventory": {
            "root": "/server/home", "directories_sha256": "a" * 64,
            "file_count": 0, "total_size_bytes": 0, "files_sha256": "b" * 64,
        },
        "server_runtime_inventories": {},
        "server_runtime_inventories_sha256": stable_hash({}),
        "mapped_runtime_files": mapped,
        "mapped_runtime_files_sha256": stable_hash(mapped),
        "launch_receipt": launch,
        "launch_receipt_sha256": "4" * 64,
        "port": 2000,
    }


def _manifest(tmp_path: Path, root: Path) -> Path:
    path = tmp_path / "runtime.json"
    atomic_json(path, assets.build_manifest(
        root, inventory_roots=ROOTS, required_files=REQUIRED,
        required_absent_paths=ABSENT,
    ))
    return path


def _validate(path: Path, root: Path) -> dict:
    return assets.validate_manifest(
        path, root, inventory_roots=ROOTS, required_files=REQUIRED,
        required_absent_paths=ABSENT,
    )


def test_runtime_manifest_accepts_logical_root_symlink_and_town06_absence(tmp_path: Path):
    root, _ = _runtime(tmp_path)
    manifest = _manifest(tmp_path, root)
    value = _validate(manifest, root)
    frozen = json.loads(manifest.read_text())
    assert value["file_count"] == len(frozen["files"])
    assert all("Town06" not in row["path"] for row in frozen["files"])
    assert frozen["scope"]["required_absent_paths"] == list(ABSENT)


@pytest.mark.parametrize("mutation", ["content", "missing", "extra", "town06", "absent"])
def test_runtime_manifest_rejects_asset_inventory_drift(tmp_path: Path, mutation: str):
    root, real = _runtime(tmp_path)
    manifest = _manifest(tmp_path, root)
    parking = (
        real / "CarlaUE4/Content/map_package/Maps/parkinglotfish-new3/"
        "parkinglotfish-new3.umap"
    )
    if mutation == "content":
        parking.write_bytes(b"parking-drift")
    elif mutation == "missing":
        parking.unlink()
    elif mutation == "extra":
        _write(real / "CarlaUE4/Content/Carla/Maps/untracked.uasset", b"extra")
    elif mutation == "town06":
        _write(real / "CarlaUE4/Content/Carla/Maps/Town06/Town06.umap", b"installed")
    else:
        _write(real / "ContentVersions.txt", b"new package")
    with pytest.raises(ValueError, match="differs|missing or extra|required-absent"):
        _validate(manifest, root)


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo"])
def test_runtime_manifest_rejects_nonregular_or_symlink_inventory(tmp_path: Path, kind: str):
    root, real = _runtime(tmp_path)
    content = real / "CarlaUE4/Content"
    if kind == "file_symlink":
        (content / "linked.uasset").symlink_to(real / "VERSION")
    elif kind == "directory_symlink":
        (content / "linked-dir").symlink_to(real / "CarlaUE4/Config", target_is_directory=True)
    else:
        os.mkfifo(content / "runtime.fifo")
    with pytest.raises(ValueError, match="non-regular|non-directory"):
        assets.build_manifest(
            root, inventory_roots=ROOTS, required_files=REQUIRED,
            required_absent_paths=ABSENT,
        )


def test_runtime_validation_is_fresh_per_resolution_not_process_cached(tmp_path: Path):
    root, real = _runtime(tmp_path)
    manifest = _manifest(tmp_path, root)
    _validate(manifest, root)
    target = real / "CarlaUE4/Content/Carla/Maps/Town01/Town01.umap"
    target.write_bytes(b"changed-after-success")
    with pytest.raises(ValueError, match="differs"):
        _validate(manifest, root)


def test_manifest_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path):
    root, _ = _runtime(tmp_path)
    manifest = _manifest(tmp_path, root)
    raw = manifest.read_text()
    manifest.write_text(raw.replace('"schema_version":', '"schema_version":"duplicate", "schema_version":', 1))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _validate(manifest, root)
    manifest.write_text(raw.replace('"file_count":', '"telemetry":NaN,"file_count":', 1))
    with pytest.raises(ValueError, match="non-finite JSON"):
        _validate(manifest, root)


class _Client:
    def __init__(self, maps: list[str], *, server: str = "0.9.16-custom"):
        self.maps = maps
        self.server = server

    def get_available_maps(self):
        return list(self.maps)

    def get_client_version(self):
        return "0.9.16"

    def get_server_version(self):
        return self.server


def test_available_maps_witness_is_sorted_hashed_and_rejects_drift(tmp_path: Path):
    root, _ = _runtime(tmp_path)
    witness = assets.build_available_maps_witness(
        root, _Client(["/Game/Carla/Maps/Town03", "/Game/Carla/Maps/Town01"]),
    )
    assert witness["available_maps"] == [
        "/Game/Carla/Maps/Town01", "/Game/Carla/Maps/Town03",
    ]
    assert witness["available_maps_sha256"] == stable_hash(witness["available_maps"])
    bad = copy.deepcopy(witness)
    bad["available_maps"].append("/Game/Carla/Maps/Town06")
    with pytest.raises(ValueError, match="sorted, unique|hash differs"):
        assets.validate_available_maps_witness(bad, root.resolve())


def test_execution_bundle_runtime_fields_make_old_result_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _ = _runtime(tmp_path)
    manifest = _manifest(tmp_path, root)
    authority = _validate(manifest, root)
    authority.update({
        "provenance_sha256": "d" * 64,
        "available_maps_sha256": "e" * 64,
        "client_version": "0.9.16",
            "server_version": "0.9.16-custom",
            "available_map_count": 2,
            "boot_id": "test-boot", "freeze_monotonic_ticks": 1,
            "clock_ticks_per_second": 100,
            **_permission_authority(),
        })
    monkeypatch.setattr(runner, "CARLA_RUNTIME_ROOT", root)
    # This test exercises runtime-asset staleness, not the production
    # dependency inventory. Keep its synthetic CARLA tree independent of the
    # real evaluator import graph and avoid an unrelated environment probe.
    monkeypatch.setattr(
        runner,
        "_execution_dependency_hashes",
        lambda profile: {"bounded/test-runtime": "8" * 64},
    )
    sidecar = tmp_path / "sidecar.json"
    route = tmp_path / "routes.xml"
    agent = tmp_path / "agent.py"
    config = tmp_path / "config.json"
    for path, data in ((sidecar, b"{}"), (route, b"<routes/>"), (agent, b"# agent"), (config, b"{}")):
        _write(path, data)
    bundle = runner.expected_input_hashes(
        sidecar=sidecar, route_xml=route, agent_path=agent, agent_config=config,
        benchmark_track="oracle", evaluator_track="native", workload="core",
        runtime_asset_authority=authority,
    )
    assert bundle["carla_runtime_asset_manifest_sha256"] == authority["manifest_sha256"]
    result = tmp_path / "result/contract_result.json"
    atomic_json(result, {"status": "timeout"})
    old = dict(bundle)
    old.pop("carla_runtime_asset_manifest_sha256")
    atomic_json(result.with_name("runner_command.json"), {"input_hashes": old})
    assert runner._result_is_current_for_sidecar(result, sidecar, bundle) is False


def test_execute_preflight_rejects_stale_runtime_bundle(monkeypatch: pytest.MonkeyPatch):
    authority = {
        "resolved_carla_root": "/carla", "manifest_sha256": "a" * 64,
        "inventory_sha256": "b" * 64, "files_sha256": "c" * 64,
        "file_count": 2, "total_size_bytes": 3, "provenance_sha256": "d" * 64,
        "shipping_binary_sha256": "f" * 64,
        "available_maps_sha256": "e" * 64, "client_version": "0.9.16",
        "server_version": "0.9.16-custom", "available_map_count": 2,
        "boot_id": "test-boot", "freeze_monotonic_ticks": 1,
        "clock_ticks_per_second": 100,
        **_permission_authority(),
    }
    monkeypatch.setattr(runner, "_current_runtime_asset_authority", lambda: authority)
    monkeypatch.setattr(runner, "_current_file_sha256", lambda path: "f" * 64)
    current = runner._runtime_bundle_fields(authority)
    runner._validate_plan_runtime_asset_authority({"input_hashes": current})
    stale = dict(current)
    stale["carla_runtime_asset_inventory_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale or incomplete"):
        runner._validate_plan_runtime_asset_authority({"input_hashes": stale})


@pytest.mark.parametrize("mutation", ["server", "client", "maps", "count"])
def test_live_endpoint_witness_rejects_version_or_map_drift(
    monkeypatch: pytest.MonkeyPatch, mutation: str,
):
    maps = ["/Game/Carla/Maps/Town01", "/Game/Carla/Maps/Town03"]
    observed = {
        "client_version": "0.9.16",
        "server_version": "0.9.16-custom",
        "available_maps": maps,
        "available_map_count": len(maps),
        "available_maps_sha256": stable_hash(maps),
    }
    bundle = {
        "carla_runtime_client_version": observed["client_version"],
        "carla_runtime_server_version": observed["server_version"],
        "carla_runtime_available_map_count": observed["available_map_count"],
        "carla_runtime_available_maps_sha256": observed["available_maps_sha256"],
    }
    monkeypatch.setattr(
        runner, "_query_live_carla_runtime_witness", lambda host, port: dict(observed)
    )
    process = _process_receipt()
    monkeypatch.setattr(
        runner, "_attest_carla_listener_process", lambda *args, **kwargs: dict(process)
    )
    bundle["carla_runtime_shipping_binary_sha256"] = "f" * 64
    bundle["carla_server_process_receipt"] = process
    bundle["carla_server_process_receipt_sha256"] = stable_hash(process)
    assert runner._validate_live_carla_runtime_authority(
        {"input_hashes": bundle}, "127.0.0.1", 2000,
    ) == {**observed, "listener_process": process}
    if mutation == "server":
        bundle["carla_runtime_server_version"] = "other"
    elif mutation == "client":
        bundle["carla_runtime_client_version"] = "0.9.15"
    elif mutation == "maps":
        bundle["carla_runtime_available_maps_sha256"] = "0" * 64
    else:
        bundle["carla_runtime_available_map_count"] = 999
    with pytest.raises(RuntimeError, match="differs from plan"):
        runner._validate_live_carla_runtime_authority(
            {"input_hashes": bundle}, "127.0.0.1", 2000,
        )


def test_live_endpoint_attestation_is_bracketed_and_process_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    process = _process_receipt()
    maps = ["/Game/Carla/Maps/Town01"]
    observed = {
        "client_version": "0.9.16", "server_version": "custom",
        "available_maps": maps, "available_map_count": 1,
        "available_maps_sha256": stable_hash(maps),
    }
    bundle = {
        "carla_runtime_shipping_binary_sha256": "f" * 64,
        "carla_runtime_client_version": "0.9.16",
        "carla_runtime_server_version": "custom",
        "carla_runtime_available_map_count": 1,
        "carla_runtime_available_maps_sha256": stable_hash(maps),
        "carla_server_process_receipt": process,
        "carla_server_process_receipt_sha256": stable_hash(process),
    }
    calls = []
    def attest(*args, **kwargs):
        calls.append("attest")
        return copy.deepcopy(process)
    def query(*args, **kwargs):
        calls.append("rpc")
        return copy.deepcopy(observed)
    monkeypatch.setattr(runner, "_attest_carla_listener_process", attest)
    monkeypatch.setattr(runner, "_query_live_carla_runtime_witness", query)
    runner._validate_live_carla_runtime_authority(
        {"input_hashes": bundle}, "127.0.0.1", 2000,
    )
    assert calls == ["attest", "rpc", "attest"]
    replacement = copy.deepcopy(process)
    replacement["pid"] = 999
    sequence = iter((process, replacement))
    monkeypatch.setattr(
        runner, "_attest_carla_listener_process",
        lambda *args, **kwargs: copy.deepcopy(next(sequence)),
    )
    with pytest.raises(RuntimeError, match="changed during RPC"):
        runner._validate_live_carla_runtime_authority(
            {"input_hashes": bundle}, "127.0.0.1", 2000,
        )


def test_live_process_allows_ephemeral_cache_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    process = _process_receipt()
    maps = ["/Game/Carla/Maps/Town01"]
    observed = {
        "client_version": "0.9.16", "server_version": "custom",
        "available_maps": maps, "available_map_count": 1,
        "available_maps_sha256": stable_hash(maps),
    }
    bundle = {
        "carla_runtime_shipping_binary_sha256": "f" * 64,
        "carla_runtime_client_version": "0.9.16",
        "carla_runtime_server_version": "custom",
        "carla_runtime_available_map_count": 1,
        "carla_runtime_available_maps_sha256": stable_hash(maps),
        "carla_server_process_receipt": process,
        "carla_server_process_receipt_sha256": stable_hash(process),
    }
    # Cache contents are deliberately absent from the structural receipt.  A
    # real file/subdirectory may appear between the bracketed attestations.
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(runner, "CARLA_SERVER_RUNTIME_ROOT", runtime)
    cache = runtime / "cache"
    cache.mkdir(mode=0o700)
    identity = runner._server_runtime_directory_identity(cache)
    grown = copy.deepcopy(process)
    grown["server_home_inventory"] = identity
    grown["server_runtime_inventories"] = {"HOME": identity}
    grown["server_runtime_inventories_sha256"] = stable_hash({"HOME": identity})
    process["server_home_inventory"] = identity
    process["server_runtime_inventories"] = {"HOME": identity}
    process["server_runtime_inventories_sha256"] = stable_hash({"HOME": identity})
    bundle["carla_server_process_receipt"] = process
    bundle["carla_server_process_receipt_sha256"] = stable_hash(process)
    calls = 0
    def attest(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            (cache / "GLCache").mkdir()
            (cache / "GLCache/shader.bin").write_bytes(b"cache")
        return copy.deepcopy(grown if calls == 2 else process)
    monkeypatch.setattr(runner, "_attest_carla_listener_process", attest)
    monkeypatch.setattr(
        runner, "_query_live_carla_runtime_witness", lambda *args: dict(observed),
    )
    runner._validate_live_carla_runtime_authority(
        {"input_hashes": bundle}, "127.0.0.1", 2000,
    )


def test_mapped_runtime_allows_only_frozen_carla_lazy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "carla"
    lazy = root / "CarlaUE4/Content/Lazy.uasset"
    _write(lazy, b"lazy")
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {
        "files": [{
            "path": "CarlaUE4/Content/Lazy.uasset",
            "size_bytes": 4, "sha256": runner.sha256_file(lazy),
        }],
    })
    monkeypatch.setattr(runner, "CARLA_RUNTIME_ROOT", root)
    monkeypatch.setattr(runner, "CARLA_RUNTIME_MANIFEST", manifest)
    frozen: list[dict[str, object]] = [{
        "path": "/usr/lib/frozen-system.so", "device": 1, "inode": 1,
        "maps_device": "00:01", "size_bytes": 3,
        "sha256": "f" * 64,
    }]
    lazy_row = {
        "path": str(lazy), "device": 1, "inode": 2,
        "maps_device": "00:01", "size_bytes": 4,
        "sha256": runner.sha256_file(lazy),
    }
    current = [*copy.deepcopy(frozen), lazy_row]
    digest = runner.sha256_file(manifest)
    assert runner._mapped_runtime_rows_compatible(frozen, frozen, digest)
    assert runner._mapped_runtime_rows_compatible(current, frozen, digest)
    assert not runner._mapped_runtime_rows_compatible([lazy_row], frozen, digest)
    foreign = copy.deepcopy(current)
    foreign[-1]["path"] = "/tmp/unfrozen.so"
    assert not runner._mapped_runtime_rows_compatible(foreign, frozen, digest)
    drift = copy.deepcopy(current)
    drift[-1]["sha256"] = "0" * 64
    assert not runner._mapped_runtime_rows_compatible(drift, frozen, digest)


def test_server_runtime_directory_rejects_symlink_and_detects_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    cache = runtime / "cache"
    cache.mkdir()
    monkeypatch.setattr(runner, "CARLA_SERVER_RUNTIME_ROOT", runtime)
    before = runner._server_runtime_directory_identity(cache)
    cache.rmdir()
    cache.mkdir()
    after = runner._server_runtime_directory_identity(cache)
    assert before != after
    cache.rmdir()
    target = runtime / "target"
    target.mkdir()
    cache.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="real directory"):
        runner._server_runtime_directory_identity(cache)


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}'])
def test_runtime_witness_strict_json_rejects_duplicate_or_nonfinite(raw: str):
    with pytest.raises(ValueError):
        runner._strict_json_loads(raw, "bounded")


def test_live_process_must_start_after_asset_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    runtime = {
        "resolved_carla_root": str(runner.CARLA_RUNTIME_ROOT.resolve()),
        "manifest_sha256": "a" * 64, "inventory_sha256": "b" * 64,
        "files_sha256": "c" * 64, "file_count": 3, "total_size_bytes": 4,
        "shipping_binary_sha256": "f" * 64,
        "provenance_sha256": "d" * 64, "available_maps_sha256": "e" * 64,
        "client_version": "0.9.16", "server_version": "custom",
        "available_map_count": 1, "boot_id": "test-boot",
        "freeze_monotonic_ticks": 100, "clock_ticks_per_second": 100,
        **_permission_authority(),
    }
    process = _process_receipt()
    process["starttime_ticks"] = 100
    sidecar, route, agent, config, evaluator = (
        tmp_path / name for name in
        ("sidecar.json", "routes.xml", "agent.py", "config.json", "evaluator.py")
    )
    for path in (sidecar, route, agent, config, evaluator):
        _write(path, b"{}")
    monkeypatch.setattr(runner, "DEFAULT_EVALUATOR", evaluator)
    with pytest.raises(ValueError, match="predates"):
        runner.expected_input_hashes(
            sidecar=sidecar, route_xml=route, agent_path=agent,
            agent_config=config, benchmark_track="oracle",
            evaluator_track="native", workload="core",
            runtime_asset_authority=runtime, live_process_authority=process,
        )


def test_evaluator_environment_ignores_ambient_loader_and_python_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = runner.evaluator_environment()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "HOME"):
        monkeypatch.setenv(key, "/hostile")
    assert runner.evaluator_environment() == baseline
    assert "LD_PRELOAD" not in baseline and baseline["PYTHONNOUSERSITE"] == "1"


def test_evaluator_environment_binds_explicit_gpu_rank():
    baseline = runner.evaluator_environment()
    assert "CUDA_VISIBLE_DEVICES" not in baseline
    for rank in range(7):
        environment = runner.evaluator_environment(gpu_rank=rank)
        assert environment["CUDA_VISIBLE_DEVICES"] == str(rank)
    with pytest.raises(ValueError, match="gpu_rank"):
        runner.evaluator_environment(gpu_rank=-1)


def test_available_maps_capture_rejects_duplicate_raw_values(tmp_path: Path):
    root, _ = _runtime(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        assets.build_available_maps_witness(
            root, _Client(["/Game/Carla/Maps/Town01"] * 2),
        )


def test_execute_live_runtime_mismatch_halts_before_job_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    run_id = "oracle__A-001__left__p0__s1"
    bundle = {"runtime": "bounded"}
    job = {
        "run": {"run_id": run_id, "route_entry_id": 0},
        "command": ["/bin/true", "--l2d-external-carla", "--host", "127.0.0.1", "--port", "2000", "--traffic-manager-port", "8000"],
        "input_hashes": bundle,
        "deadline_seconds": None,
    }
    policy = runner._storage_policy()
    job["storage_estimate"] = runner._job_storage_estimate(job, policy)
    plan = {
        "plan_sha256": "runtime-mismatch",
        "input_hashes": bundle,
        "storage_policy": policy,
        "wall_timeout_policy": runner._wall_timeout_policy(agent="oracle", workload="core"),
        "workload": "core",
        "agent": "oracle",
        "jobs": [job],
    }
    plan.update(runner._plan_storage_estimate(plan["jobs"]))
    monkeypatch.setattr(runner, "_validate_plan_sha256", lambda plan: None)
    authority = {
        "schema_version": "language2drive.plan_execution_authority.v1",
        "project_root": str(tmp_path.resolve()), "build_dir": str(tmp_path.resolve()),
        "route_xml": str((tmp_path / "routes.xml").resolve()), "sidecar": None,
        "results_dir": str((tmp_path / "results").resolve()), "agent": "oracle",
        "agent_path": str((tmp_path / "agent.py").resolve()),
        "agent_config": str((tmp_path / "config.json").resolve()),
        "benchmark_track": "oracle", "evaluator_track": "native",
        "workload": "core", "dependency_profile": "base-runtime.v1",
        "host": "127.0.0.1", "port": 2000, "traffic_manager_port": 8000,
        "gpu_rank": 0,
    }
    plan["execution_authority"] = authority
    monkeypatch.setattr(
        runner, "_validate_plan_execution_authority", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner, "_validate_nonasset_execution_authority", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner, "carla_server_available", lambda host, port: True)
    monkeypatch.setattr(runner, "_storage_available_bytes", lambda path: 10**15)
    monkeypatch.setattr(
        runner, "_validate_live_carla_runtime_authority",
        lambda plan, host, port: (_ for _ in ()).throw(
            RuntimeError("active CARLA endpoint runtime witness differs from plan")
        ),
    )
    value = runner.execute_plan(
        plan, project_root=tmp_path, results_dir=tmp_path / "results",
        require_server=True,
    )
    assert value["status"] == "execution_halted"
    assert value["reason"] == "runtime_asset_authority_mismatch"
    assert value["halted"] == 1
    assert not (tmp_path / "results" / run_id / "runner_command.json").exists()


def test_manifest_provenance_binds_generator_manifest_and_map_witness(tmp_path: Path):
    root, _ = _runtime(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    provenance_path = tmp_path / "provenance.json"
    witness_path = tmp_path / "maps.json"
    generator = tmp_path / "generator.py"
    generator.write_text("# v1\n", encoding="utf-8")
    atomic_json(witness_path, assets.build_available_maps_witness(
        root, _Client(["/Game/Carla/Maps/Town01"]),
    ))
    _, provenance = assets.freeze_manifest(
        root, manifest_path, provenance_path,
        generator_path=generator,
        available_maps_witness_path=witness_path,
        inventory_roots=ROOTS,
        required_files=REQUIRED,
        required_absent_paths=ABSENT,
    )
    validated = assets.validate_provenance(
        provenance_path, manifest_path=manifest_path, carla_root=root,
        generator_path=generator,
    )
    assert validated["provenance_sha256"]
    assert validated["available_maps_sha256"] == stable_hash(
        provenance["available_maps_witness"]["available_maps"]
    )
    bad = json.loads(provenance_path.read_text())
    bad["available_maps_witness"]["server_version"] = "other"
    atomic_json(provenance_path, bad)
    with pytest.raises(ValueError, match="witness hash differs"):
        assets.validate_provenance(
            provenance_path, manifest_path=manifest_path, carla_root=root,
            generator_path=generator,
        )
    atomic_json(provenance_path, provenance)
    generator.write_text("# v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generator hash differs"):
        assets.validate_provenance(
            provenance_path, manifest_path=manifest_path, carla_root=root,
            generator_path=generator,
        )


def test_full_plan_execution_authority_rebuilds_raw_exact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    runtime = {
        "resolved_carla_root": str(runner.CARLA_RUNTIME_ROOT.resolve()),
        "manifest_sha256": "a" * 64, "inventory_sha256": "b" * 64,
        "files_sha256": "c" * 64, "file_count": 3, "total_size_bytes": 4,
        "shipping_binary_sha256": "5" * 64,
        "provenance_sha256": "d" * 64, "available_maps_sha256": "e" * 64,
        "client_version": "0.9.16", "server_version": "custom",
        "available_map_count": 2,
        "boot_id": "test-boot", "freeze_monotonic_ticks": 1,
        "clock_ticks_per_second": 100,
        **_permission_authority(),
    }
    monkeypatch.setattr(runner, "_current_runtime_asset_authority", lambda: dict(runtime))
    sidecar = tmp_path / "sidecar.json"
    route = tmp_path / "routes.xml"
    agent = tmp_path / "agent.py"
    config = tmp_path / "config.json"
    evaluator = tmp_path / "evaluator.py"
    for path, raw in (
        (sidecar, json.dumps({"routes": [{
            "route_entry_id": 0, "deadline_seconds": 45, "pcu_id": "A-001",
            "arm_id": "left", "layer": "A", "paraphrase": 0,
            "split": "dev", "town": "Town01",
        }]}).encode()),
        (route, b"<routes/>"), (agent, b"#agent"),
        (config, b"{}"), (evaluator, b"#evaluator"),
    ):
        _write(path, raw)
    monkeypatch.setattr(runner, "DEFAULT_EVALUATOR", evaluator)
    dependency_rows = {"bounded/runtime": "9" * 64}
    monkeypatch.setattr(
        runner, "_execution_dependency_hashes", lambda profile: dict(dependency_rows),
    )
    policy = runner._wall_timeout_policy(agent="oracle", workload="core")
    bundle = runner.expected_input_hashes(
        sidecar=sidecar, route_xml=route, agent_path=agent, agent_config=config,
        benchmark_track="oracle", evaluator_track="native", workload="core",
        dependency_profile="base-runtime.v1", wall_timeout_policy=policy,
        runtime_asset_authority=runtime, live_process_authority=_process_receipt(),
    )
    results = tmp_path / "results"
    authority = runner._plan_execution_authority(
        project_root=tmp_path, build_dir=tmp_path, route_xml=route,
        sidecar=sidecar, results_dir=results, agent="oracle",
        agent_path=agent, agent_config=config,
        benchmark_track="oracle", evaluator_track="native", workload="core",
        dependency_profile="base-runtime.v1", host="127.0.0.1", port=2000,
        traffic_manager_port=8000, gpu_rank=0,
    )
    monkeypatch.setattr(
        runner, "_attest_carla_listener_process",
        lambda *args, **kwargs: _process_receipt(),
    )
    run = {
        "arm_id": "left", "layer": "A", "paraphrase": 0,
        "pcu_id": "A-001", "route_entry_id": 0,
        "run_id": "oracle__A-001__left__p0__s1", "seed": 1,
        "split": "dev", "status": "expected", "track": "oracle",
    }
    command = runner.build_command(
        run, project_root=tmp_path, build_dir=tmp_path, results_dir=results,
        track="oracle", agent_path=agent, agent_config=config,
        evaluator_track="native", route_xml=route, sidecar=sidecar,
    )
    job = {
        "run": run, "command": command, "command_display": runner.shlex.join(command),
        "result_dir": str(results / run["run_id"]), "deadline_seconds": 45.0,
        "input_hashes": bundle,
    }
    job["default_wall_timeout_seconds"] = runner._job_wall_timeout(job, None, policy)
    job["storage_estimate"] = runner._job_storage_estimate(job, runner._storage_policy())
    plan = {
        "track": "oracle", "evaluator_track": "native", "workload": "core",
        "agent": "oracle", "agent_path": str(agent.resolve()),
        "agent_config": str(config.resolve()), "wall_timeout_policy": policy,
        "execution_authority": authority, "input_hashes": bundle,
        "job_count": 1, "jobs": [job],
    }
    runner._validate_plan_execution_authority(
        plan, project_root=tmp_path, results_dir=results,
    )
    runner._validate_nonasset_execution_authority(plan)
    original_command = copy.deepcopy(job["command"])
    job["command"] = ["/bin/true", "--l2d-external-carla"]
    job["command_display"] = runner.shlex.join(job["command"])
    with pytest.raises(ValueError, match="command differs"):
        runner._validate_plan_execution_authority(
            plan, project_root=tmp_path, results_dir=results,
        )
    job["command"] = original_command
    job["command_display"] = runner.shlex.join(original_command)
    original_run = copy.deepcopy(job["run"])
    original_result_dir = job["result_dir"]
    job["run"]["run_id"] = "oracle__A-999__other__p0__s1"
    job["result_dir"] = str(results / job["run"]["run_id"])
    job["command"] = runner.build_command(
        job["run"], project_root=tmp_path, build_dir=tmp_path,
        results_dir=results, track="oracle", agent_path=agent,
        agent_config=config, evaluator_track="native", route_xml=route,
        sidecar=sidecar,
    )
    job["command_display"] = runner.shlex.join(job["command"])
    job["storage_estimate"] = runner._job_storage_estimate(
        job, runner._storage_policy(),
    )
    with pytest.raises(ValueError, match="route/run identity differs"):
        runner._validate_plan_execution_authority(
            plan, project_root=tmp_path, results_dir=results,
        )
    job["run"] = original_run
    job["result_dir"] = original_result_dir
    job["command"] = original_command
    job["command_display"] = runner.shlex.join(original_command)
    job["storage_estimate"] = runner._job_storage_estimate(
        job, runner._storage_policy(),
    )
    for path in (sidecar, route, agent, config, evaluator):
        old = path.read_bytes()
        path.write_bytes(old + b"\ndrift")
        with pytest.raises(ValueError, match="input bundle is stale"):
            runner._validate_plan_execution_authority(
                plan, project_root=tmp_path, results_dir=results,
            )
        path.write_bytes(old)
    monkeypatch.setattr(
        runner,
        "_execution_dependency_hashes",
        lambda profile: {**dependency_rows, "indirect/drift": "0" * 64},
    )
    with pytest.raises(ValueError, match="input bundle is stale"):
        runner._validate_plan_execution_authority(
            plan, project_root=tmp_path, results_dir=results,
        )


def test_fdq_plan_authority_uses_exact_bindings_without_core_route_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    routes = tmp_path / "fdq.xml"
    evaluator = tmp_path / "evaluator.py"
    agent = tmp_path / "agent.py"
    config = tmp_path / "config.json"
    for path, raw in (
        (routes, b"<routes/>"), (evaluator, b"# evaluator"),
        (agent, b"# agent"), (config, b"{}"),
    ):
        _write(path, raw)
    monkeypatch.setattr(runner, "DEFAULT_FDQ_ROUTES", routes)
    monkeypatch.setattr(runner, "DEFAULT_EVALUATOR", evaluator)
    monkeypatch.setattr(
        runner, "_execution_dependency_hashes", lambda profile: {"bounded": "a" * 64},
    )
    runtime = {
        "resolved_carla_root": str(runner.CARLA_RUNTIME_ROOT.resolve()),
        "manifest_sha256": "a" * 64, "inventory_sha256": "b" * 64,
        "files_sha256": "c" * 64, "file_count": 3, "total_size_bytes": 4,
        "shipping_binary_sha256": "f" * 64,
        "provenance_sha256": "d" * 64, "available_maps_sha256": "e" * 64,
        "client_version": "0.9.16", "server_version": "custom",
        "available_map_count": 2, "boot_id": "test-boot",
        "freeze_monotonic_ticks": 1, "clock_ticks_per_second": 100,
        **_permission_authority(),
    }
    process = _process_receipt()
    monkeypatch.setattr(runner, "_current_runtime_asset_authority", lambda: dict(runtime))
    monkeypatch.setattr(
        runner, "_attest_carla_listener_process", lambda *args, **kwargs: copy.deepcopy(process),
    )
    policy = runner._wall_timeout_policy(agent="oracle", workload="fdq")
    bundle = runner.expected_input_hashes(
        sidecar=None, route_xml=routes, agent_path=agent, agent_config=config,
        benchmark_track="fdq", evaluator_track="native", workload="fdq",
        dependency_profile="base-runtime.v1", wall_timeout_policy=policy,
        runtime_asset_authority=runtime, live_process_authority=process,
    )
    resolved = {
        "agent_path": agent, "agent_config": config, "input_hashes": bundle,
        "wall_timeout_policy": policy, "storage_policy": runner._storage_policy(),
        "runtime_asset_authority": runtime, "live_process_authority": process,
    }
    runs = [
        {"ability": ability, "run_id": f"fdq__{ability}__s1", "seed": 1,
         "status": "expected"}
        for ability in runner.FDQ_ROUTE_BINDINGS
    ]
    results = tmp_path / "results"
    plan = runner.make_fdq_plan(
        runs, results_dir=results, agent="oracle", agent_path=agent,
        agent_config=config, resolved_inputs=resolved,
    )
    runner._validate_plan_execution_authority(
        plan, project_root=runner.L2D_ROOT, results_dir=results,
    )
    assert all("route_entry_id" not in job["run"] for job in plan["jobs"])
    original_bindings = copy.deepcopy(plan["route_bindings"])
    plan["route_bindings"] = {}
    with pytest.raises(ValueError, match="FDQ route authority differs"):
        runner._validate_plan_execution_authority(
            plan, project_root=runner.L2D_ROOT, results_dir=results,
        )
    plan["route_bindings"] = original_bindings
    plan["jobs"][0]["run"]["route_entry_id"] = 999
    with pytest.raises(ValueError, match="FDQ job 0 run shape differs"):
        runner._validate_plan_execution_authority(
            plan, project_root=runner.L2D_ROOT, results_dir=results,
        )


@pytest.mark.parametrize("mode", ["baseline", "safety"])
def test_shared_plan_authority_accepts_exact_baseline_and_safety_run_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
):
    runtime = {
        "resolved_carla_root": str(runner.CARLA_RUNTIME_ROOT.resolve()),
        "manifest_sha256": "a" * 64, "inventory_sha256": "b" * 64,
        "files_sha256": "c" * 64, "file_count": 3, "total_size_bytes": 4,
        "shipping_binary_sha256": "f" * 64, "provenance_sha256": "d" * 64,
        "available_maps_sha256": "e" * 64, "client_version": "0.9.16",
        "server_version": "custom", "available_map_count": 2,
        "boot_id": "test-boot", "freeze_monotonic_ticks": 1,
        "clock_ticks_per_second": 100,
        **_permission_authority(),
    }
    monkeypatch.setattr(runner, "_current_runtime_asset_authority", lambda: dict(runtime))
    monkeypatch.setattr(
        runner, "_attest_carla_listener_process", lambda *args, **kwargs: _process_receipt(),
    )
    monkeypatch.setattr(
        runner, "_execution_dependency_hashes", lambda profile: {"bounded": "a" * 64},
    )
    route_xml = tmp_path / "routes.xml"
    agent = tmp_path / "agent.py"
    config = tmp_path / "config.json"
    evaluator = tmp_path / "evaluator.py"
    for path, raw in ((route_xml, b"<routes/>"), (agent, b"#agent"),
                      (config, b"{}"), (evaluator, b"#evaluator")):
        _write(path, raw)
    monkeypatch.setattr(runner, "DEFAULT_EVALUATOR", evaluator)
    route = {
        "route_entry_id": 0, "deadline_seconds": 45.0, "pcu_id": "A-001",
        "arm_id": "left", "layer": "A", "paraphrase": 0,
        "split": "dev", "town": "Town01",
    }
    if mode == "baseline":
        track, evaluator_track, workload = "baseline", "standardized", "baseline:no-language"
        run = {
            "arm_id": "left", "layer": "A", "paraphrase": 0,
            "pcu_id": "A-001", "route_entry_id": 0,
            "run_id": "baseline__no-language__A-001__left__p0__s1",
            "seed": 1, "split": "dev", "status": "expected", "track": "baseline",
            "baseline_id": "no-language", "evaluator_track": "standardized",
        }
    else:
        track, evaluator_track, workload = "standardized", "standardized", "safety_priority"
        route.update({
            "hazard_class": "physically_infeasible", "response_mode": "safe_fallback",
            "safety_scope": "unsafe_illegal_ambiguous_or_infeasible_instruction_response",
        })
        run = {
            "arm_id": "left", "paraphrase": 0, "pcu_id": "A-001",
            "route_entry_id": 0, "run_id": "standardized__A-001__left__p0__s1",
            "seed": 1, "split": "dev", "status": "expected", "track": "standardized",
            "benchmark_track": "safety_priority",
            "hazard_class": "physically_infeasible", "response_mode": "safe_fallback",
            "safety_scope": "unsafe_illegal_ambiguous_or_infeasible_instruction_response",
        }
    sidecar = tmp_path / "sidecar.json"
    atomic_json(sidecar, {"routes": [route]})
    policy = runner._wall_timeout_policy(agent="custom", workload=workload)
    bundle = runner.expected_input_hashes(
        sidecar=sidecar, route_xml=route_xml, agent_path=agent, agent_config=config,
        benchmark_track=track, evaluator_track=evaluator_track, workload=workload,
        dependency_profile="base-runtime.v1", wall_timeout_policy=policy,
        runtime_asset_authority=runtime, live_process_authority=_process_receipt(),
    )
    resolved = {
        "agent_path": agent, "agent_config": config,
        "evaluator_track": evaluator_track, "sidecar": sidecar,
        "route_xml": route_xml, "input_hashes": bundle,
        "wall_timeout_policy": policy, "storage_policy": runner._storage_policy(),
    }
    results = tmp_path / "results"
    plan = runner.make_plan(
        [run], project_root=tmp_path, build_dir=tmp_path, results_dir=results,
        track=track, agent="custom", evaluator_track=evaluator_track,
        route_xml=route_xml, sidecar=sidecar, workload=workload,
        resolved_inputs=resolved,
    )
    runner._validate_plan_execution_authority(
        plan, project_root=tmp_path, results_dir=results,
    )
    plan["jobs"][0]["run"]["pcu_id"] = "A-999"
    with pytest.raises(ValueError, match="route/run identity differs"):
        runner._validate_plan_execution_authority(
            plan, project_root=tmp_path, results_dir=results,
        )


def test_python_site_startup_surface_is_exact_and_fail_closed(
    tmp_path: Path,
):
    site = tmp_path / "site-packages"
    site.mkdir()
    baseline = runner._python_site_startup_hash(site)
    (site / "ordinary.py").write_text("raise RuntimeError\n", encoding="utf-8")
    assert runner._python_site_startup_hash(site) == baseline
    startup = site / "startup-hook.pth"
    startup.write_text("import injected_startup\n", encoding="utf-8")
    with_hook = runner._python_site_startup_hash(site)
    assert with_hook != baseline
    startup.write_text("import different_startup\n", encoding="utf-8")
    assert runner._python_site_startup_hash(site) != with_hook
    startup.unlink()
    assert runner._python_site_startup_hash(site) == baseline
    (site / "sitecustomize").mkdir()
    with pytest.raises(ValueError, match="startup dependency is non-regular"):
        runner._python_site_startup_hash(site)


def test_evaluator_attempt_environment_starts_empty_and_is_not_reused(
    tmp_path: Path,
):
    root = tmp_path / "attempt-runtime"
    environment = runner.evaluator_environment(root, prepare=True)
    assert environment["HOME"] == str(root / "evaluator-home")
    assert list((root / "evaluator-home").iterdir()) == []
    assert list((root / "tmp").iterdir()) == []
    assert (root / "pycache-disabled").stat().st_mode & 0o777 == 0o555
    (root / "carla-cache/cache.bin").write_bytes(b"history")
    with pytest.raises(ValueError, match="must be empty"):
        runner.evaluator_environment(root, prepare=True)


def test_evaluator_attempt_environment_is_opaque_and_discloses_no_route_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    parent = tmp_path / "opaque-attempts"
    monkeypatch.setattr(runner, "_EVALUATOR_ENVIRONMENT_PARENT", parent)
    monkeypatch.setattr(runner.secrets, "token_hex", lambda size: "f" * (size * 2))
    root = runner._new_opaque_evaluator_attempt_root()
    environment = runner.evaluator_environment(root, prepare=True)
    run = {
        "run_id": "standardized__A-013__left__p0__s1",
        "pcu_id": "A-013", "arm_id": "left",
    }
    runner._validate_evaluator_environment_non_disclosure(environment, run)
    assert root.parent == parent
    assert root.name == "attempt-" + "f" * 48
    assert all(
        token not in value
        for token in (run["run_id"], run["pcu_id"], run["arm_id"])
        for value in environment.values()
    )
    leaked = dict(environment)
    leaked["TMPDIR"] = str(tmp_path / run["run_id"])
    with pytest.raises(RuntimeError, match="discloses route identity"):
        runner._validate_evaluator_environment_non_disclosure(leaked, run)
    low_run = {
        "run_id": "oracle__A-021__low__p0__s1",
        "pcu_id": "A-021", "arm_id": "low",
    }
    environment_with_pillow = dict(environment)
    environment_with_pillow["LD_LIBRARY_PATH"] = (
        "/opt/site-packages/pillow.libs:/opt/lib"
    )
    runner._validate_evaluator_environment_non_disclosure(
        environment_with_pillow, low_run,
    )
    for leaked_value in (
        str(tmp_path / "low" / "cache"),
        str(tmp_path / "attempt__low__cache"),
        low_run["run_id"],
    ):
        leaked_low = dict(environment_with_pillow)
        leaked_low["TMPDIR"] = leaked_value
        with pytest.raises(RuntimeError, match="discloses route identity"):
            runner._validate_evaluator_environment_non_disclosure(
                leaked_low, low_run,
            )


def test_fdq_binding_authority_is_reconstructed_without_nested_aliases(
    tmp_path: Path,
):
    first = runner._canonical_fdq_route_bindings()
    second = runner._canonical_fdq_route_bindings()
    assert first == second
    assert first is not second
    assert first["overtaking"] is not second["overtaking"]
    first["overtaking"]["route_id"] = "MUTATED"
    assert runner._canonical_fdq_route_bindings()["overtaking"]["route_id"] == "24240"

    run = {
        "run_id": "fdq__overtaking__s1", "ability": "overtaking",
        "seed": 1, "status": "expected",
    }
    resolved = {
        "agent_path": runner.DEFAULT_ORACLE_AGENT,
        "agent_config": runner.L2D_ROOT / "config/defaults.json",
        "input_hashes": {"dependency_profile": "base-runtime.v1"},
        "wall_timeout_policy": runner._wall_timeout_policy(
            agent="oracle", workload="fdq",
        ),
        "storage_policy": runner._storage_policy(),
    }
    plan = runner.make_fdq_plan(
        [run], results_dir=tmp_path / "results", agent="oracle",
        resolved_inputs=resolved,
    )
    job_binding = plan["jobs"][0]["run"]["route_binding"]
    top_binding = plan["route_bindings"]["overtaking"]
    assert job_binding == top_binding == runner._canonical_fdq_route_bindings()["overtaking"]
    assert job_binding is not top_binding
    job_binding["route_id"] = "MUTATED"
    assert plan["route_bindings"]["overtaking"]["route_id"] == "24240"
    assert runner._canonical_fdq_route_bindings()["overtaking"]["route_id"] == "24240"
    assert plan["fdq_execution_scope"] == {
        "schema_version": "language2drive.fdq_execution_scope.v1",
        "qualification_claim": False,
        "scope": "selected_recovery_subset_not_full_fdq_qualification",
        "selected_identities": [{"ability": "overtaking", "seed": 1}],
    }


def test_python_prefix_manifest_detects_real_file_and_broken_link_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib/python3.11/site-packages").mkdir(parents=True)
    (prefix / "bin/python").write_bytes(b"python-v1")
    (prefix / "broken").symlink_to("missing-target")
    monkeypatch.setattr(runner, "CONDA_PREFIX", prefix)
    baseline = runner._python_prefix_runtime_hash()
    (prefix / "bin/python").write_bytes(b"python-v2")
    assert runner._python_prefix_runtime_hash() != baseline
    (prefix / "bin/python").write_bytes(b"python-v1")
    assert runner._python_prefix_runtime_hash() == baseline
    (prefix / "extra").write_bytes(b"extra")
    assert runner._python_prefix_runtime_hash() != baseline


def _fake_proc_listener(
    tmp_path: Path, shipping: Path, *, pids: tuple[int, ...] = (123,),
    command: bytes | None = None,
) -> Path:
    uid, gid = _server_identity()
    account_name = runner.CARLA_SERVER_ACCOUNT
    proc = tmp_path / "proc"
    _write(proc / "net/tcp", (
        b"  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        b"   0: 00000000:07D0 00000000:0000 0A 00000000:00000000 "
        + f"00:00000000 00000000 {uid} 0 4242 1 0000000000000000\n".encode()
    ))
    _write(proc / "net/tcp6", b"  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n")
    _write(proc / "sys/kernel/random/boot_id", b"test-boot\n")
    for pid in pids:
        process = proc / str(pid)
        (process / "fd").mkdir(parents=True)
        (process / "fd/3").symlink_to("socket:[4242]")
        (process / "exe").symlink_to(shipping)
        (process / "cwd").symlink_to(shipping.parent)
        (process / "root").symlink_to("/")
        _write(
            process / "status",
            f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t{gid}\t{gid}\t{gid}\t{gid}\n".encode(),
        )
        fields = ["S"] + ["0"] * 18 + ["12345"] + ["0"] * 8
        _write(process / "stat", f"{pid} (Shipping) {' '.join(fields)}\n".encode())
        _write(process / "ns/mnt", b"namespace")
        shipping_metadata = shipping.resolve().stat()
        mapped_device = (
            f"{os.major(shipping_metadata.st_dev):02x}:"
            f"{os.minor(shipping_metadata.st_dev):02x}"
        )
        _write(
            process / "maps",
            (
                f"00400000-00401000 r-xp 00000000 {mapped_device} "
                f"{shipping_metadata.st_ino} "
                f"{shipping.resolve()}\n"
            ).encode(),
        )
        server_root = runner.CARLA_SERVER_RUNTIME_ROOT
        environment = {
            "PATH": (
                "/root/miniconda3/bin:/usr/local/bin:/usr/local/nvidia/bin:"
                "/usr/local/cuda/bin:/usr/local/sbin:/usr/bin:/sbin:/bin"
            ),
            "HOME": str((server_root / "home").resolve()),
            "TMPDIR": str((server_root / "tmp").resolve()),
            "XDG_CACHE_HOME": str((server_root / "xdg-cache").resolve()),
            "CUDA_CACHE_PATH": str((server_root / "cuda-cache").resolve()),
            "CARLA_CACHE_DIR": str((server_root / "carla-cache").resolve()),
            "UE-SharedDataCachePath": str((server_root / "user-dir").resolve()),
            "USER": account_name, "LOGNAME": account_name,
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        }
        _write(
            process / "environ",
            b"\0".join(f"{key}={value}".encode() for key, value in environment.items()) + b"\0",
        )
        actual_command = command or (
            str(shipping.resolve()).encode() + b"\0CarlaUE4\0-RenderOffScreen\0-nosound\0"
            b"-carla-port=2000\0-quality-level=Epic\0"
        )
        _write(process / "cmdline", actual_command)
    self_namespace = proc / "self/ns/mnt"
    self_namespace.parent.mkdir(parents=True, exist_ok=True)
    os.link(proc / str(pids[0]) / "ns/mnt", self_namespace)
    return proc


def test_local_listener_process_attestation_binds_shipping_path_hash_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    shipping = tmp_path / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    _write(shipping, b"shipping")
    server_root = tmp_path / "server-runtime"
    server_uid, server_gid = _server_identity()
    for name in ("home", "tmp", "xdg-cache", "cuda-cache", "carla-cache", "user-dir"):
        (server_root / name).mkdir(parents=True)
        os.chown(server_root / name, server_uid, server_gid)
        os.chmod(server_root / name, 0o700)
    monkeypatch.setattr(runner, "CARLA_SERVER_RUNTIME_ROOT", server_root)
    manifest = tmp_path / "manifest.json"
    provenance = tmp_path / "provenance.json"
    _write(manifest, b"manifest")
    _write(provenance, b"provenance")
    monkeypatch.setattr(runner, "CARLA_RUNTIME_MANIFEST", manifest)
    monkeypatch.setattr(runner, "CARLA_RUNTIME_PROVENANCE", provenance)
    receipt = {
        "schema_version": "language2drive.carla_server_launch_receipt.v1",
        "boot_id": "test-boot", "prelaunch_monotonic_ticks": 12000,
        "clock_ticks_per_second": 100,
        "server_runtime_root": str(server_root.resolve()),
        "runtime_asset_manifest_sha256": runner.sha256_file(manifest),
        "runtime_asset_provenance_sha256": runner.sha256_file(provenance),
        "launcher_sha256": runner.sha256_file(
            runner.L2D_ROOT / "tools/start_external_carla.sh"
        ),
        "shipping_binary_sha256": runner.sha256_file(shipping),
    }
    atomic_json(server_root / "launch-receipt.json", receipt)
    os.chmod(server_root / "launch-receipt.json", 0o444)
    proc = _fake_proc_listener(tmp_path, shipping)
    digest = runner.sha256_file(shipping)
    value = runner._attest_carla_listener_process(
        "127.0.0.1", 2000, proc_root=proc,
        expected_shipping=shipping, expected_sha256=digest,
    )
    assert value["pid"] == 123
    assert value["executable"] == str(shipping.resolve())
    assert value["executable_sha256"] == digest
    assert value["mapped_runtime_files"][0]["path"] == str(shipping.resolve())
    # A busy UE process churns unrelated descriptors while /proc/PID/fd is
    # enumerated.  A vanished non-listener fd must not hide the still-present
    # listener owner; all other I/O errors and a vanished listener remain
    # fail-closed.
    transient_fd = proc / "123/fd/4"
    transient_fd.symlink_to("/dev/null")
    real_readlink = runner.os.readlink

    def readlink_with_closed_unrelated_fd(path):
        if Path(path) == transient_fd:
            raise FileNotFoundError(2, "injected closed descriptor", str(path))
        return real_readlink(path)

    with monkeypatch.context() as transient_patch:
        transient_patch.setattr(
            runner.os, "readlink", readlink_with_closed_unrelated_fd,
        )
        assert runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )["pid"] == 123
    transient_fd.unlink()
    maps_path = proc / "123/maps"
    original_maps = maps_path.read_text(encoding="utf-8")
    maps_path.write_text(
        original_maps
        + "4114c000-4114e000 r-xs 00000000 00:01 33297 "
          "/memfd:/.glXXXXXX (deleted)\n",
        encoding="utf-8",
    )
    assert runner._attest_carla_listener_process(
        "127.0.0.1", 2000, proc_root=proc,
        expected_shipping=shipping, expected_sha256=digest,
    )["pid"] == 123
    maps_path.write_text(
        original_maps
        + "7f000000-7f001000 r-xp 00000000 08:01 33297 "
          "/tmp/real-library.so (deleted)\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="deleted runtime file"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    maps_path.write_text(original_maps, encoding="utf-8")
    root_link = proc / "123/root"
    root_link.unlink()
    root_link.symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="filesystem root"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    root_link.unlink()
    root_link.symlink_to("/")
    environment_path = proc / "123/environ"
    original_environment = environment_path.read_bytes()
    environment_path.write_bytes(original_environment + b"EXTRA_GPU_SETTING=1\0")
    with pytest.raises(RuntimeError, match="environment differs"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    environment_path.write_bytes(original_environment)
    status_path = proc / "123/status"
    original_status = status_path.read_bytes()
    tcp_path = proc / "net/tcp"
    original_tcp = tcp_path.read_text(encoding="utf-8")
    current_uid, _ = _server_identity()
    alternate = pwd.getpwnam("simlingo_run")
    status_path.write_bytes(
        (
            f"Uid:\t{alternate.pw_uid}\t{alternate.pw_uid}\t{alternate.pw_uid}\t{alternate.pw_uid}\n"
            f"Gid:\t{alternate.pw_gid}\t{alternate.pw_gid}\t{alternate.pw_gid}\t{alternate.pw_gid}\n"
        ).encode()
    )
    tcp_path.write_text(
        original_tcp.replace(f" {current_uid} 0 4242 ", f" {alternate.pw_uid} 0 4242 "),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="account differs"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    status_path.write_bytes(original_status)
    tcp_path.write_text(original_tcp, encoding="utf-8")
    self_namespace = proc / "self/ns/mnt"
    self_namespace.unlink()
    self_namespace.write_bytes(b"different namespace")
    with pytest.raises(RuntimeError, match="mount namespace"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    self_namespace.unlink()
    os.link(proc / "123/ns/mnt", self_namespace)
    late_receipt = dict(receipt)
    late_receipt["prelaunch_monotonic_ticks"] = 13000
    atomic_json(server_root / "launch-receipt.json", late_receipt)
    os.chmod(server_root / "launch-receipt.json", 0o444)
    with pytest.raises(RuntimeError, match="predates its clean launch receipt"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    atomic_json(server_root / "launch-receipt.json", receipt)
    os.chmod(server_root / "launch-receipt.json", 0o444)
    os.chown(server_root / "launch-receipt.json", 1001, 1001)
    with pytest.raises(RuntimeError, match="root-owned immutable"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    os.chown(server_root / "launch-receipt.json", 0, 0)
    os.chmod(server_root, 0o777)
    with pytest.raises(RuntimeError, match="root-owned immutable"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    os.chmod(server_root, 0o755)
    proc = _fake_proc_listener(tmp_path / "duplicate", shipping, pids=(123, 456))
    with pytest.raises(RuntimeError, match="exactly one"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
    proc = _fake_proc_listener(
        tmp_path / "wrong-port", shipping,
        command=(str(shipping.resolve()).encode() + b"\0CarlaUE4\0-RenderOffScreen\0"
                 b"-nosound\0-carla-port=2001\0-quality-level=Epic\0"),
    )
    with pytest.raises(RuntimeError, match="command line differs"):
        runner._attest_carla_listener_process(
            "127.0.0.1", 2000, proc_root=proc,
            expected_shipping=shipping, expected_sha256=digest,
        )
