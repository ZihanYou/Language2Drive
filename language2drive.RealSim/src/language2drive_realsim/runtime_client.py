"""Submit frozen opaque sessions to an AlpaSim daemon runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import atomic_json, load_json, stable_hash
from .model_identity import checkpoint_identity
from .pairing import assert_paired_inputs


def build_run_plan(public: Dict[str, Any], sidecar: Dict[str, Any]) -> Dict[str, Any]:
    if public.get("source_binding") != sidecar.get("source_binding"):
        raise ValueError("public workload and private sidecar bindings differ")
    sessions = sidecar["sessions"]
    public_session_ids = [run.get("session_uuid") for run in public["runs"]]
    if len(public_session_ids) != len(set(public_session_ids)):
        raise ValueError("public workload contains duplicate session UUIDs")
    if set(public_session_ids) != set(sessions):
        raise ValueError("public workload and private sidecar session ledgers differ")
    plans = []
    public_by_identity = {}
    for run in public["runs"]:
        session_uuid = run["session_uuid"]
        private = sessions.get(session_uuid)
        if private is None:
            raise ValueError("public session is absent from the trusted sidecar")
        identity = (private["pcu_id"], private["seed"], private["arm_id"])
        if identity in public_by_identity:
            raise ValueError("duplicate public PCU/seed/arm identity")
        public_by_identity[identity] = run
        plans.append(
            {
                "session_uuid": session_uuid,
                "scenario_id": private["scene_id_private"],
                "expected_artifact_id": private["artifact_id"],
                "expected_nre_version_string": private["nre_version_string"],
                "start_time_offset_us": int(private["start_time_offset_us"]),
                "expected_start_timestamp_us": int(private["start_timestamp_us"]),
                "random_seed": int(private["seed"]),
                "pcu_id": private["pcu_id"],
                "arm_id": private["arm_id"],
            }
        )
    if not plans:
        raise ValueError("run plan contains no sessions")
    paired_input_hashes = {}
    pair_keys = sorted({(pcu_id, seed) for pcu_id, seed, _ in public_by_identity})
    for pcu_id, seed in pair_keys:
        first = public_by_identity.get((pcu_id, seed, "R0"))
        second = public_by_identity.get((pcu_id, seed, "R1"))
        if first is None or second is None:
            raise ValueError("missing public arm for %s seed %s" % (pcu_id, seed))
        paired_input_hashes["%s:%s" % (pcu_id, seed)] = assert_paired_inputs(first, second)
    return {
        "schema_version": "l2d.realsim.run_plan.v0.1",
        "source_binding": public["source_binding"],
        "run_count": len(plans),
        "runtime_paired_input_assertions": paired_input_hashes,
        "runs": plans,
    }


def paired_retry_plan(plan: Dict[str, Any], runtime_return: Dict[str, Any]) -> Dict[str, Any]:
    """Expand any infrastructure failure to the complete coupled arm pair."""

    by_session = {row["session_uuid"]: row for row in plan["runs"]}
    failed_pairs = set()
    returned = {}
    for result in runtime_return.get("rollout_returns", []):
        session_uuid = result.get("session_uuid")
        if session_uuid not in by_session:
            raise ValueError("runtime returned an unknown session")
        if session_uuid in returned:
            raise ValueError("runtime returned a duplicate session")
        returned[session_uuid] = result
    for session_uuid, original in by_session.items():
        result = returned.get(session_uuid)
        if result is None or result.get("success") is not True:
            failed_pairs.add((original["pcu_id"], original["random_seed"]))
    retries = [
        row
        for row in plan["runs"]
        if (row["pcu_id"], row["random_seed"]) in failed_pairs
    ]
    return {
        "schema_version": "l2d.realsim.retry_plan.v0.1",
        "source_binding": plan["source_binding"],
        "policy": "complete_R0_R1_pair",
        "failed_pair_count": len(failed_pairs),
        "run_count": len(retries),
        "runs": retries,
    }


def simulation_request_payload(
    plan: Dict[str, Any],
    driver_address: str,
    concurrency: int,
    use_registered_driver: bool = False,
) -> Dict[str, Any]:
    """Return the exact JSON-shaped payload used to build SimulationRequest.

    Prospective campaigns persist this small payload immediately before the
    RPC.  ``submit_run_plan`` consumes the same helper, so the recorded request
    cannot silently drift from the protobuf fields that are actually sent.
    """

    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("simulation concurrency must be a positive integer")
    runs = plan.get("runs")
    declared_count = plan.get("run_count")
    if not isinstance(runs, list) or (
        declared_count is not None and declared_count != len(runs)
    ):
        raise ValueError("run plan count is inconsistent")
    driver_host, separator, driver_port_text = driver_address.rpartition(":")
    if not separator or not driver_host:
        raise ValueError("driver address must be host:port")
    try:
        driver_port = int(driver_port_text)
    except ValueError as error:
        raise ValueError("driver port must be an integer") from error
    if not 1 <= driver_port <= 65535:
        raise ValueError("driver port is outside the TCP range")

    specs = []
    session_uuids = set()
    required = (
        "scenario_id", "session_uuid", "start_time_offset_us", "random_seed"
    )
    for row in runs:
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError("run plan row lacks a SimulationRequest field")
        session_uuid = row["session_uuid"]
        if not isinstance(session_uuid, str) or not session_uuid or session_uuid in session_uuids:
            raise ValueError("run plan has an empty or duplicate session UUID")
        session_uuids.add(session_uuid)
        specs.append(
            {
                "scenario_id": row["scenario_id"],
                "nr_rollouts": 1,
                "session_uuids": [session_uuid],
                "start_time_offset_us": int(row["start_time_offset_us"]),
                "random_seed": int(row["random_seed"]),
            }
        )
    return {
        "available_drivers": [] if use_registered_driver else [
            {"ip": driver_host, "port": driver_port}
        ],
        "rollout_specs": specs,
        "n_concurrent_per_driver": concurrency,
    }


def submit_run_plan(
    plan: Dict[str, Any],
    runtime_address: str,
    driver_address: str,
    concurrency: int,
    use_registered_driver: bool = False,
) -> Dict[str, Any]:
    try:
        import grpc
        from alpasim_grpc.v0 import runtime_pb2, runtime_pb2_grpc
    except ImportError as error:
        raise RuntimeError("live submission must run in the pinned AlpaSim environment") from error
    payload = simulation_request_payload(
        plan, driver_address, concurrency, use_registered_driver
    )
    specs = [runtime_pb2.RolloutSpec(**row) for row in payload["rollout_specs"]]
    request = runtime_pb2.SimulationRequest(
        available_drivers=[
            runtime_pb2.SimulationRequest.DriverAddress(**row)
            for row in payload["available_drivers"]
        ],
        rollout_specs=specs,
        n_concurrent_per_driver=payload["n_concurrent_per_driver"],
    )
    # Simulator service traffic is direct (or through an explicit SSH tunnel),
    # never through the user's model-download HTTP proxy.
    with grpc.insecure_channel(
        runtime_address, options=(("grpc.enable_http_proxy", 0),)
    ) as channel:
        stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
        if use_registered_driver:
            from alpasim_grpc.v0 import common_pb2

            info = stub.get_runtime_info(common_pb2.Empty(), timeout=30)
            if info.driver_version.version_id in {"", "<skip>"}:
                raise ValueError("runtime has no registered driver version authority")
        response = stub.simulate(request)
    returns = []
    for result in response.rollout_returns:
        # Short rollouts can legitimately have no valid long-horizon minADE.
        # JSON has no NaN/Infinity; retain the missing value explicitly, not 0.
        metrics = {name: float(value) if math.isfinite(value) else None
                   for name, value in result.aggregated_metrics.items()}
        returns.append(
            {
                "session_uuid": result.rollout_uuid,
                "success": bool(result.success),
                "error": result.error,
                "error_code": int(result.error_code),
                "aggregated_metrics": metrics,
                "unavailable_aggregated_metrics": sorted(name for name, value in metrics.items() if value is None),
            }
        )
    return {
        "schema_version": "l2d.realsim.runtime_return.v0.1",
        "source_binding": plan["source_binding"],
        "runtime_version": response.runtime_version.version_id,
        "runtime_git_hash": response.runtime_version.git_hash,
        "nre_version": response.nre_version.version_id,
        "driver_version": response.driver_version.version_id,
        "rollout_returns": returns,
    }


def create_run_manifest(
    plan: Dict[str, Any],
    runtime_address: str,
    driver_address: str,
    dry_run: bool,
    track: str = "benchmark_model",
    model_identity: Optional[Dict[str, Any]] = None,
    use_registered_driver: bool = False,
) -> Dict[str, Any]:
    if track == "benchmark_model" and not model_identity:
        raise ValueError("benchmark-model run manifest requires a model identity")
    manifest = {
        "schema_version": "l2d.realsim.run.v0.1",
        "source_binding": plan["source_binding"],
        "run_plan_hash": stable_hash(plan),
        "expected_run_count": plan["run_count"],
        "runtime_address": runtime_address,
        "driver_address": driver_address,
        "driver_binding": "registered_static" if use_registered_driver else "per_request_dynamic",
        "dry_run": bool(dry_run),
        "track": track,
        "model_identity": model_identity,
        "denominator_policy": "all frozen eligible sessions remain; missing/crash/timeout are failures",
        "invalid_sim_policy": "rerun the complete R0/R1 pair for the coupled seed",
    }
    manifest["manifest_binding"] = stable_hash(manifest)
    return manifest


def run(
    public_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    runtime_address: str,
    driver_address: str,
    concurrency: int = 1,
    dry_run: bool = False,
    track: str = "benchmark_model",
    model_id: Optional[str] = None,
    model_checkpoint: Optional[Path] = None,
    use_registered_driver: bool = False,
) -> Dict[str, Any]:
    if track not in {"benchmark_model", "privileged_oracle"}:
        raise ValueError("unknown RealSim execution track")
    public = load_json(public_path)
    sidecar = load_json(sidecar_path)
    plan = build_run_plan(public, sidecar)
    output_dir = Path(output_dir)
    atomic_json(output_dir / "run_plan.private.json", plan)
    identity = None
    if track == "benchmark_model":
        if not model_id or model_checkpoint is None:
            raise ValueError("benchmark-model run requires model ID and checkpoint path")
        identity = checkpoint_identity(model_checkpoint, model_id)
        if not identity["complete"]:
            raise ValueError("benchmark-model checkpoint is incomplete")
    manifest = create_run_manifest(
        plan, runtime_address, driver_address, dry_run, track, identity, use_registered_driver
    )
    atomic_json(output_dir / "run_manifest.json", manifest)
    if not dry_run:
        result = submit_run_plan(plan, runtime_address, driver_address, concurrency, use_registered_driver)
        oracle_driver = result["driver_version"].startswith(
            "l2d_privileged_oracle-driver-"
        )
        if track == "privileged_oracle" and not oracle_driver:
            raise ValueError("oracle run used a non-oracle driver")
        if track == "benchmark_model" and oracle_driver:
            raise ValueError("benchmark run used the privileged oracle driver")
        if track == "benchmark_model":
            expected_prefix = {
                "nvidia/Alpamayo-1.5-10B": "l2d_alpamayo1_5-driver-",
                "nvidia/Alpamayo2-Super": "l2d_alpamayo2_super-driver-",
            }.get(model_id)
            if expected_prefix is None or not result["driver_version"].startswith(
                expected_prefix
            ):
                raise ValueError("benchmark run used a driver that does not match model ID")
        atomic_json(output_dir / "runtime_return.json", result)
        atomic_json(output_dir / "retry_plan.private.json", paired_retry_plan(plan, result))
    return manifest
