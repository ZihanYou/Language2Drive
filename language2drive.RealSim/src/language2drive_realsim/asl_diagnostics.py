"""Lightweight real-pose/prefix diagnostics using official runtime safety samples."""
import numpy as np

from .compat import strict_zip
from .alpasim_bridge import (
    _metric_lookup, _observation_record, _preintervention_evidence,
    load_runtime_safety_metrics, validate_asl_authority,
)
from .asl import iter_asl
from .trajectory import sample_plan


def inspect_pair_arm(asl, private, candidate, run_uuid):
    from alpasim_utils.geometry import pose_from_grpc

    duration = private.get("rollout_duration_s", 7.)
    if type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid requested rollout duration")

    metadata, driver = None, None
    observations, poses = [], {}
    for message in iter_asl(asl):
        kind = message.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            current = message.rollout_metadata
            if metadata is not None and metadata != current:
                raise ValueError("conflicting ASL rollout metadata")
            metadata = current
        elif kind == "driver_session_request":
            request = message.driver_session_request
            current = {"session_uuid": request.session_uuid, "random_seed": int(request.random_seed),
                       "scene_id": request.debug_info.scene_id if request.HasField("debug_info") else None}
            if driver is not None and driver != current:
                raise ValueError("conflicting ASL driver session requests")
            driver = current
        elif kind == "actor_poses":
            timestamp = int(message.actor_poses.timestamp_us)
            for actor in message.actor_poses.actor_poses:
                if actor.actor_id == "EGO":
                    if timestamp in poses:
                        raise ValueError("duplicate actual ego pose timestamp")
                    poses[timestamp] = pose_from_grpc(actor.actor_pose)
        else:
            record = _observation_record(message, private["instruction_issued_at_us"])
            if record is not None:
                observations.append(record)
    if metadata is None or not metadata.HasField("transform_ego_coords_rig_to_aabb"):
        raise ValueError("ASL lacks authoritative rig transform or metadata")
    if metadata.session_metadata.session_uuid != asl.parent.name:
        raise ValueError("ASL session does not match its requested session directory")
    validate_asl_authority(metadata.session_metadata, driver, {
        "scene_id_private": candidate["scene_id_private"], "artifact_id": candidate["artifact_uuid"],
        "start_timestamp_us": private["start_timestamp_us"], "seed": private["seed"]})
    version = metadata.version_ids.egodriver_version
    if not version.version_id.startswith("l2d_privileged_oracle-driver-"):
        raise ValueError("ASL has no oracle driver authority")
    cameras = ["camera_cross_left_120fov", "camera_front_wide_120fov",
               "camera_cross_right_120fov", "camera_front_tele_30fov"]
    prefix = _preintervention_evidence(observations, private["instruction_issued_at_us"], cameras)
    timestamps = sorted(poses)
    times = (np.asarray(timestamps, dtype=np.int64) - private["start_timestamp_us"]) / 1e6
    if len(times) < 2 or times[0] != 0 or times[-1] < duration or np.any(np.diff(times) > 0.200001):
        raise ValueError("actual ego trace lacks complete dense requested-duration coverage")
    centre_to_rig = pose_from_grpc(metadata.transform_ego_coords_rig_to_aabb).inverse()
    actual = np.asarray([poses[t].vec3 for t in timestamps])
    actual_rig = np.asarray([(poses[t] @ centre_to_rig).vec3 for t in timestamps])
    if not np.isfinite(actual_rig).all():
        raise ValueError("nonfinite actual rig poses")
    metrics_path = asl.with_name("metrics.parquet")
    if metrics_path.resolve().parent != asl.resolve().parent:
        raise ValueError("metrics must belong to the same ASL directory")
    metrics = _metric_lookup(load_runtime_safety_metrics(
        metrics_path, metadata.session_metadata.session_uuid, candidate["scene_id_private"], run_uuid))
    if any(not set(timestamps).issubset(values) for values in metrics.values()):
        raise ValueError("runtime safety metrics do not cover every actual ego pose")
    expected, _ = sample_plan(private["oracle_plan"], times)
    error = np.linalg.norm(actual_rig[:, :2] - expected[:, :2], axis=1)
    stat, metrics_stat = asl.stat(), metrics_path.stat()
    return {"session_uuid": metadata.session_metadata.session_uuid,
            "versions": {"egodriver_version_id": version.version_id, "egodriver_git_hash": version.git_hash},
            "source_asl": str(asl), "source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "source_identity_scope": "diagnostic_metadata_not_integrity_attestation",
            "safety_metric_source": "official_runtime_parquet_not_recomputed",
            "metrics_size_bytes": metrics_stat.st_size, "metrics_mtime_ns": metrics_stat.st_mtime_ns,
            "prefix": prefix, "samples": len(times),
            "required_duration_s": duration,
            "collision_samples": sum(bool(metrics["collision_any"][t]) for t in timestamps),
            "offroad_samples": sum(bool(metrics["offroad"][t]) for t in timestamps),
            "black_image_samples": sum(bool(metrics["img_is_black"][t]) for t in timestamps),
            "maximum_reference_tracking_error_m": float(error.max()),
            "tracking_error_reference": "rig_origin_in_nre_local",
            "final_position_xyz": actual[-1].tolist(), "position_reference": "ego_aabb_center_in_nre_local",
            "final_rig_position_xyz": actual_rig[-1].tolist(), "final_time_s": float(times[-1]),
            "rig_samples_for_diagnostics": [{"time_s": float(t), "position_xyz": p.tolist()}
                                            for t, p in strict_zip(times, actual_rig)]}
