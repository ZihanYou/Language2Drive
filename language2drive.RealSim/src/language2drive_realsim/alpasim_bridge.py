"""Convert pinned AlpaSim SimulationResult/ASL state into normalized traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from .constants import ALPASIM_COMMIT, ORACLE_CONTROLLER_ID
from .events import point_in_polygon, point_to_polygon_distance
from .io import atomic_json, file_hash, load_json, stable_hash
from .render_validity import NATIVE_MODE, POLYGON_MODE, native_render_protocol


def _protobuf_payload_hash(message: Any) -> str:
    """Hash a gRPC observation after removing its arm-specific session UUID."""

    clone = message.__class__()
    clone.CopyFrom(message)
    clone.ClearField("session_uuid")
    return stable_hash(clone.SerializeToString(deterministic=True))


def _preintervention_evidence(
    records: Iterable[Dict[str, Any]],
    instruction_issued_at_us: int,
    expected_camera_ids: Iterable[str],
) -> Dict[str, Any]:
    selected = [
        dict(record)
        for record in records
        if int(record["timestamp_us"]) <= int(instruction_issued_at_us)
    ]
    selected.sort(
        key=lambda row: (
            int(row["timestamp_us"]),
            str(row["kind"]),
            str(row.get("logical_id", "")),
            str(row["payload_sha256"]),
        )
    )
    counts = {
        kind: sum(row["kind"] == kind for row in selected)
        for kind in ("camera", "egomotion", "route")
    }
    if counts["camera"] < 1 or counts["egomotion"] < 1:
        raise ValueError(
            "ASL lacks camera or egomotion evidence before language intervention"
        )
    observed_camera_ids = sorted(
        {row["logical_id"] for row in selected if row["kind"] == "camera"}
    )
    if observed_camera_ids != sorted(expected_camera_ids):
        raise ValueError(
            "ASL pre-intervention cameras do not match the frozen model input"
        )
    return {
        "instruction_issued_at_us": int(instruction_issued_at_us),
        "preintervention_camera_logical_ids": observed_camera_ids,
        "preintervention_record_counts": counts,
        "preintervention_non_language_sha256": stable_hash(selected),
    }


REQUIRED_SAFETY_METRICS = ("collision_any", "offroad", "img_is_black")


def load_runtime_safety_metrics(path: Path, session_uuid: str, scene_id: str, run_uuid: str) -> List[Any]:
    """Reuse official per-sample parquet for diagnostics, not formal attestation."""
    import polars as pl

    frame = pl.read_parquet(path).filter(pl.col("name").is_in(REQUIRED_SAFETY_METRICS))
    grouped = {}
    for row in frame.iter_rows(named=True):
        if (row["rollout_id"] != session_uuid or row["clipgt_id"] != scene_id
            or row["run_uuid"] != run_uuid):
            raise ValueError("runtime safety parquet has different session/scene/run authority")
        timestamp = row["timestamps_us"]
        # Official mixed-scorer dataframes may promote microseconds to Float64.
        # Accept only exactly representable integral values, never truncation.
        if (type(row["valid"]) is not bool or type(timestamp) not in (int, float)
            or not math.isfinite(timestamp) or timestamp < 0
            or (type(timestamp) is float and (not timestamp.is_integer() or timestamp > 2**53 - 1))):
            raise ValueError("runtime safety parquet has invalid timestamp/valid types")
        metric = grouped.setdefault(row["name"], SimpleNamespace(
            name=row["name"], timestamps_us=[], values=[], valid=[]))
        metric.timestamps_us.append(int(timestamp))
        metric.values.append(row["values"])
        metric.valid.append(row["valid"])
    if set(grouped) != set(REQUIRED_SAFETY_METRICS):
        raise ValueError("runtime parquet lacks required safety metrics")
    metrics = list(grouped.values())
    _metric_lookup(metrics)  # Retain strict duplicate/nonfinite/invalid/binary checks.
    return metrics


def _metric_lookup(metrics: Iterable[Any]) -> Dict[str, Dict[int, float]]:
    lookup: Dict[str, Dict[int, float]] = {}
    for metric in metrics:
        if metric.name not in REQUIRED_SAFETY_METRICS:
            continue
        if metric.name in lookup:
            raise ValueError("duplicate safety metric: " + metric.name)
        if not len(metric.timestamps_us) == len(metric.values) == len(metric.valid):
            raise ValueError("safety metric arrays differ in length: " + metric.name)
        samples = {}
        for timestamp, value, valid in zip(metric.timestamps_us, metric.values, metric.valid):
            timestamp = int(timestamp)
            if timestamp in samples:
                raise ValueError("duplicate safety metric timestamp: " + metric.name)
            # Preserve invalid samples as absent evidence, never a safe zero.
            number = float(value)
            if not valid or not math.isfinite(number) or number not in (0.0, 1.0):
                raise ValueError("invalid safety metric sample: " + metric.name)
            samples[timestamp] = number
        lookup[metric.name] = samples
    return lookup


def _relative_actor_state(
    simulation_result: Any, actor_id: str, timestamp_us: int, ego_position: Any, ego_yaw: float
) -> Any:
    actor = simulation_result.actor_trajectories.get(actor_id)
    if actor is None:
        return None, None, None
    pose = actor.interpolate_pose(timestamp_us)
    dx = float(pose.vec3[0]) - float(ego_position[0])
    dy = float(pose.vec3[1]) - float(ego_position[1])
    # The actor's local lateral coordinate is positive to ego-left. Contracts
    # describe which side *ego* passes on, so the reported side is inverted.
    lateral = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
    side = "RIGHT" if lateral >= 0.0 else "LEFT"
    longitudinal = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
    try:
        ego_polygon = simulation_result.actor_polygons.get_polygon_for_agent_at_time("EGO", timestamp_us)
        actor_polygon = simulation_result.actor_polygons.get_polygon_for_agent_at_time(actor_id, timestamp_us)
        clearance = float(ego_polygon.distance(actor_polygon))
    except (AttributeError, KeyError, ValueError):
        # Centre distance can substantially overstate surface clearance and is
        # therefore not a safe fallback for oracle qualification.
        clearance = None
    return side, clearance, longitudinal


def normalize_simulation_result(
    simulation_result: Any,
    metric_returns: Iterable[Any],
    annotations: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Use public AlpaSim eval state, never GT-path distance, for semantics."""

    metrics = _metric_lookup(metric_returns)
    timestamps = [int(value) for value in simulation_result.timestamps_us]
    if not timestamps:
        return []
    for name in REQUIRED_SAFETY_METRICS:
        missing = set(timestamps) - metrics.get(name, {}).keys()
        if missing:
            raise ValueError("missing safety metric coverage: %s (%d timestamps)" % (name, len(missing)))
    metadata = getattr(simulation_result, "session_metadata", None)
    time_origin_us = int(
        metadata.start_timestamp_us if metadata is not None else timestamps[0]
    )
    ego = simulation_result.actor_trajectories["EGO"]
    native_render = native_render_protocol(annotations)
    recorded = getattr(simulation_result, "ego_recorded_ground_truth_trajectory", None)
    if native_render and recorded is None:
        raise ValueError("native render protocol requires recorded-path deviation diagnostics")
    rows: List[Dict[str, Any]] = []
    distance = 0.0
    previous_position = None
    previous_time = None
    for timestamp_us in timestamps:
        pose = ego.interpolate_pose(timestamp_us)
        # AlpaSim eval actor poses are AABB-centre poses in NRE-local space.
        # Privileged path waypoints instead use the rig origin; do not compare
        # these directly when diagnosing tracking or freezing gate geometry.
        position = [float(pose.vec3[0]), float(pose.vec3[1]), float(pose.vec3[2])]
        if previous_position is not None:
            distance += math.hypot(
                position[0] - previous_position[0], position[1] - previous_position[1]
            )
        speed = 0.0
        if previous_position is not None and previous_time is not None and timestamp_us > previous_time:
            speed = math.hypot(
                position[0] - previous_position[0], position[1] - previous_position[1]
            ) / ((timestamp_us - previous_time) / 1_000_000.0)
        lane_ids = [
            identifier
            for identifier, polygon in annotations.get("lanes", {}).items()
            if point_in_polygon(position, polygon)
        ]
        regions = [
            identifier
            for identifier, polygon in annotations.get("regions", {}).items()
            if point_in_polygon(position, polygon)
        ]
        scopes = [
            identifier
            for identifier, polygon in annotations.get("scopes", {}).items()
            if point_in_polygon(position, polygon)
        ]
        triggers = [
            identifier
            for identifier, polygon in annotations.get("triggers", {}).items()
            if point_in_polygon(position, polygon)
        ]
        relative_side = {}
        clearance_m = {}
        relative_longitudinal_m = {}
        for actor_id in annotations.get("actor_ids", []):
            side, clearance, longitudinal = _relative_actor_state(
                simulation_result, actor_id, timestamp_us, position, float(pose.yaw())
            )
            if side is not None:
                relative_side[actor_id] = side
                relative_longitudinal_m[actor_id] = longitudinal
            if clearance is not None:
                clearance_m[actor_id] = clearance
        collision = metrics["collision_any"][timestamp_us] > 0.0
        offroad = metrics["offroad"][timestamp_us] > 0.0
        black = metrics["img_is_black"][timestamp_us] > 0.0
        support_polygon = annotations.get("reconstruction_support_polygon")
        support_valid = bool(support_polygon) and point_in_polygon(position, support_polygon)
        support_deviation = (
            point_to_polygon_distance(position, support_polygon)
            if support_polygon
            else None
        )
        recorded_deviation = None
        if native_render:
            reference = recorded.interpolate_pose(timestamp_us).vec3
            recorded_deviation = math.hypot(position[0] - float(reference[0]), position[1] - float(reference[1]))
            if not math.isfinite(recorded_deviation):
                raise ValueError("invalid recorded-path diagnostic pose")
        rows.append(
            {
                "time_s": (timestamp_us - time_origin_us) / 1_000_000.0,
                "timestamp_us": timestamp_us,
                "time_origin_us": time_origin_us,
                "position": position,
                "yaw_rad": float(pose.yaw()),
                "speed_mps": speed,
                "progress_m": distance,
                "lane_ids": lane_ids,
                "regions": regions,
                "scopes": scopes,
                "triggers": triggers,
                "relative_side": relative_side,
                "relative_longitudinal_m": relative_longitudinal_m,
                "clearance_m": clearance_m,
                "collision": collision,
                "offroad": offroad,
                "image_black": black,
                "render_valid": not black and (native_render or support_valid),
                "render_validity_mode": NATIVE_MODE if native_render else POLYGON_MODE,
                **({} if native_render else {"reconstruction_support_valid": support_valid}),
                "reconstruction_deviation_m": support_deviation,
                "recorded_path_deviation_m": recorded_deviation,
            }
        )
        previous_position = position
        previous_time = timestamp_us
    return rows


def _observation_record(message: Any, cutoff_us: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Hash only observations needed for the shared-prefix check."""
    kind = message.WhichOneof("log_entry")
    if kind == "driver_camera_image":
        request = message.driver_camera_image
        row = {"kind": "camera", "timestamp_us": int(request.camera_image.frame_end_us),
               "logical_id": request.camera_image.logical_id}
    elif kind == "driver_ego_trajectory":
        request = message.driver_ego_trajectory
        timestamps = [int(p.timestamp_us) for p in request.trajectory.poses]
        if not timestamps:
            return None
        row = {"kind": "egomotion", "timestamp_us": max(timestamps)}
    elif kind == "route_request":
        request = message.route_request
        row = {"kind": "route", "timestamp_us": int(request.route.timestamp_us)}
    else:
        return None
    if cutoff_us is not None and row["timestamp_us"] > cutoff_us:
        return None
    row["payload_sha256"] = _protobuf_payload_hash(request)
    return row


def _load_alpasim_result(
    asl: Path, eval_config: Path, usdz_glob: str, *,
    collect_observation_hashes: bool = True,
    observation_cutoff_us: Optional[int] = None,
    diagnostic_runtime_metrics: Optional[Path] = None,
    diagnostic_run_uuid: Optional[str] = None,
    diagnostic_without_map: bool = False,
) -> Any:
    # Spatial diagnostics can use separately bound original geometry and the
    # complete official metric stream. Formal conversion/scoring keeps its map.
    if type(diagnostic_without_map) is not bool or (diagnostic_without_map and (
        diagnostic_runtime_metrics is None or not diagnostic_run_uuid
        or collect_observation_hashes
    )):
        raise ValueError("map-free diagnostics require bound runtime metrics and no formal observation collection")
    try:
        from alpasim_utils.artifact import Artifact
        from .asl import iter_asl
        from eval.accumulator import EvalDataAccumulator
        from eval.data import SimulationResult
        from eval.schema import EvalConfig
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("ASL conversion must run in the pinned AlpaSim eval environment") from error
    cfg = OmegaConf.merge(EvalConfig, OmegaConf.load(str(eval_config)))
    artifacts = Artifact.discover_from_glob(usdz_glob)

    async def load() -> Any:
        accumulator = EvalDataAccumulator(cfg=cfg)
        driver_authority = None
        version_authority = None
        non_language_records = []
        for message in iter_asl(asl):
            accumulator.handle_message(message)
            entry_kind = message.WhichOneof("log_entry")
            if entry_kind == "driver_session_request":
                request = message.driver_session_request
                observed = {
                    "session_uuid": request.session_uuid,
                    "random_seed": int(request.random_seed),
                    "scene_id": (
                        request.debug_info.scene_id
                        if request.HasField("debug_info")
                        else None
                    ),
                }
                if driver_authority is not None and driver_authority != observed:
                    raise ValueError("ASL contains conflicting driver session requests")
                driver_authority = observed
            elif entry_kind == "rollout_metadata":
                version = message.rollout_metadata.version_ids.egodriver_version
                observed_version = {
                    "egodriver_version_id": version.version_id,
                    "egodriver_git_hash": version.git_hash,
                }
                if version_authority is not None and version_authority != observed_version:
                    raise ValueError("ASL contains conflicting driver versions")
                version_authority = observed_version
            elif collect_observation_hashes:
                record = _observation_record(message, observation_cutoff_us)
                if record is not None:
                    non_language_records.append(record)
        if accumulator.session_metadata is None:
            raise ValueError("ASL is missing rollout session metadata")
        scene_id = accumulator.session_metadata.scene_id
        if scene_id not in artifacts:
            raise ValueError("ASL scene is absent from the supplied USDZ set")
        scenario = accumulator.build_scenario_eval_input(
            run_uuid="l2d-realsim",
            run_name="l2d-realsim",
            vec_map=None if diagnostic_without_map else artifacts[scene_id].map,
        )
        if version_authority is None or not version_authority["egodriver_version_id"]:
            raise ValueError("ASL is missing ego-driver version authority")
        return scenario, driver_authority, version_authority, non_language_records

    scenario_input, driver_authority, version_authority, non_language_records = asyncio.run(load())
    simulation_result = SimulationResult.from_scenario_input(scenario_input, cfg)
    if diagnostic_runtime_metrics is not None:
        if (diagnostic_runtime_metrics.resolve() != asl.resolve().with_name("metrics.parquet")
            or not diagnostic_run_uuid):
            raise ValueError("diagnostic metrics must be bound to the same actual rollout")
        metric_returns = load_runtime_safety_metrics(
            diagnostic_runtime_metrics, simulation_result.session_metadata.session_uuid,
            simulation_result.session_metadata.scene_id, diagnostic_run_uuid)
    else:
        from eval.scorers import create_scorer_group
        metric_returns = create_scorer_group(cfg).calculate(simulation_result)
    return (
        simulation_result,
        metric_returns,
        driver_authority,
        version_authority,
        non_language_records,
    )


def validate_asl_authority(
    session_metadata: Any,
    driver_authority: Optional[Dict[str, Any]],
    session: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind a normalized trace to the exact frozen scene, start, and seed."""

    if (
        driver_authority is None
        or session_metadata.session_uuid != driver_authority.get("session_uuid")
    ):
        raise ValueError("ASL driver session identity is missing or inconsistent")
    observed = {
        "session_uuid": session_metadata.session_uuid,
        "scene_id_private": session_metadata.scene_id,
        "artifact_id": session_metadata.nre_uuid,
        "start_timestamp_us": int(session_metadata.start_timestamp_us),
        "random_seed": int(driver_authority["random_seed"]),
    }
    expected = {
        "session_uuid": driver_authority["session_uuid"],
        "scene_id_private": session["scene_id_private"],
        "artifact_id": session["artifact_id"],
        "start_timestamp_us": int(session["start_timestamp_us"]),
        "random_seed": int(session["seed"]),
    }
    if observed != expected:
        raise ValueError("ASL authority does not match the frozen sidecar session")
    driver_scene = driver_authority.get("scene_id")
    if driver_scene and driver_scene != observed["scene_id_private"]:
        raise ValueError("ASL driver scene does not match rollout metadata")
    return observed


def export_asl(
    asl: Path,
    eval_config: Path,
    usdz_glob: str,
    sidecar: Path,
    output: Path,
    alpasim_commit: str,
    expected_camera_ids: Optional[Iterable[str]] = None,
) -> None:
    if alpasim_commit != ALPASIM_COMMIT:
        raise ValueError(
            "ASL conversion requires pinned AlpaSim commit %s" % ALPASIM_COMMIT
        )
    private = load_json(sidecar)
    (
        simulation_result,
        metric_returns,
        driver_authority,
        version_authority,
        non_language_records,
    ) = _load_alpasim_result(asl, eval_config, usdz_glob)
    session_uuid = simulation_result.session_metadata.session_uuid
    session = private["sessions"].get(session_uuid)
    if session is None:
        raise ValueError("ASL session is absent from the trusted sidecar")
    authority = validate_asl_authority(
        simulation_result.session_metadata, driver_authority, session
    )
    schedule = session.get("instruction_schedule", [])
    if len(schedule) != 1:
        raise ValueError("frozen sidecar session must contain one instruction issuance")
    camera_ids = list(expected_camera_ids) if expected_camera_ids is not None else list(
        session.get("camera_configuration", {}).get("logical_ids", [])
    )
    if not camera_ids or len(camera_ids) != len(set(camera_ids)) or any(
        not isinstance(camera_id, str) or not camera_id for camera_id in camera_ids
    ):
        raise ValueError("expected model-input cameras must be unique non-empty logical IDs")
    preintervention = _preintervention_evidence(
        non_language_records,
        int(schedule[0]["issued_at_us"]),
        camera_ids,
    )
    rows = normalize_simulation_result(simulation_result, metric_returns, session["annotations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_json(
        output.with_suffix(".receipt.json"),
        {
            "schema_version": "l2d.realsim.asl_receipt.v0.1",
            "synthetic": False,
            **authority,
            **version_authority,
            **preintervention,
            "oracle_controller": (
                ORACLE_CONTROLLER_ID
                if version_authority["egodriver_version_id"].startswith(
                    "l2d_privileged_oracle-driver-"
                )
                else None
            ),
            "alpasim_commit": alpasim_commit,
            "source_asl": str(asl.resolve()),
            "source_asl_sha256": file_hash(asl),
            "normalized_trace_sha256": file_hash(output),
        },
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asl", required=True, type=Path)
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--usdz-glob", required=True)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpasim-commit", required=True)
    parser.add_argument(
        "--expected-camera-id", action="append", dest="expected_camera_ids",
        help="Frozen actual model-input camera logical ID; repeat once per camera",
    )
    args = parser.parse_args(argv)
    export_asl(
        args.asl,
        args.eval_config,
        args.usdz_glob,
        args.sidecar,
        args.output,
        args.alpasim_commit,
        args.expected_camera_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
