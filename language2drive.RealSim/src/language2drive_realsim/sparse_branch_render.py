"""Sparse planned-pose renders for rejection triage, never support qualification."""
import base64
import math
from pathlib import Path

import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2 as pb
from alpasim_runtime.camera_catalog import CameraCatalog, CameraDefinition
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import RuntimeCamera
from alpasim_runtime.unbound_rollout import _build_rollout_timing
from alpasim_utils.geometry import Pose, Trajectory

from .compat import strict_zip

CAMERAS = frozenset(("camera_front_wide_120fov", "camera_front_tele_30fov",
                     "camera_cross_left_120fov", "camera_cross_right_120fov"))
SCOPE = "sparse_planned_branch_render_not_closed_loop_or_continuous_support"


def validate_geometry_probe_source(search, candidate, geometry, input_path):
    """Allow rejection renders before traffic passes; never grant run authority."""
    from .branch_pair_smoke import validate_branch_search
    times, rows = validate_branch_search(search, candidate)
    binding = candidate["scene_execution_proposal_not_frozen"]
    artifact = Path(binding["artifact_path"]).resolve()
    stat = artifact.stat()
    expected = {"start_timestamp_us": binding["start_timestamp_us"],
                "selected_arm_pair": candidate["selected_arm_pair"]}
    if (search.get("scope") != "curvature_branch_geometry_dynamics_search_not_traffic_or_qualification"
        or geometry.get("scope") != "actual_map_branch_join_diagnostic_not_qualification"
        or geometry.get("qualified") is not False
        or geometry.get("candidate_execution_binding") != expected
        or geometry.get("offline_candidate_source") != str(Path(input_path).resolve())
        or search.get("offline_candidate_source") != str(Path(input_path).resolve())
        or any(geometry[k] != candidate[k] for k in ("proposal_key", "artifact_uuid", "review_revision"))
        or geometry["scene_id"] != candidate["scene_id_private"]
        or Path(geometry["source_artifact"]).resolve() != artifact
        or geometry["source_size_bytes"] != stat.st_size
        or geometry["source_mtime_ns"] != stat.st_mtime_ns
        or not np.array_equal(times, geometry["time_s"])):
        raise ValueError("geometry-only probe must retain the exact source, window and pair")
    recorded = np.column_stack((times, geometry["recorded_xyz"], geometry["recorded_yaws"]))
    if not np.array_equal(rows["R0"][times <= 1.7], recorded[times <= 1.7]):
        raise ValueError("geometry-only probe changed the original non-language prefix")
    return rows


def sample_indices(requested_times, duration_s, render_delay_us, interval_us):
    """Select actual common camera ticks, retaining requested and actual times."""
    if (type(duration_s) not in (int, float) or not math.isfinite(duration_s)
        or not 7 <= duration_s <= 30
        or not 1 <= len(requested_times) <= 5
        or any(type(t) not in (int, float) or not math.isfinite(t)
               or not 0 <= t <= duration_s for t in requested_times)
        or type(interval_us) is not int or interval_us != 100000
        or type(render_delay_us) is not int or not 0 <= render_delay_us <= 100000):
        raise ValueError("one to five finite in-window samples on the supported clock required")
    indices = [max(0, round((t * 1e6 - render_delay_us) / interval_us)) for t in requested_times]
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("distinct increasing actual camera ticks required")
    return indices


class PlannedRigTrajectory:
    """Recorded force-GT prefix, then planned XYZ/yaw; no physics correction."""
    def __init__(self, rows, source_start_us, recorded, force_gt_end_us):
        rows = np.asarray(rows, dtype=float)
        if (rows.ndim != 2 or rows.shape[1] != 5 or len(rows) < 3
            or not np.isfinite(rows).all() or rows[0, 0] != 0
            or np.any(np.diff(rows[:, 0]) <= 0) or np.any(np.diff(rows[:, 0]) > .200001)
            or type(source_start_us) is not int or source_start_us < 0
            or type(force_gt_end_us) is not int or force_gt_end_us <= source_start_us):
            raise ValueError("finite dense planned poses and exact prefix boundary required")
        timestamps = source_start_us + np.rint(rows[:, 0] * 1e6).astype(np.uint64)
        if force_gt_end_us >= int(timestamps[-1]):
            raise ValueError("planned future after force-GT required")
        poses = [Pose(r[1:4].astype(np.float32),
                      np.array([0., 0., np.sin(r[4]/2), np.cos(r[4]/2)], dtype=np.float32))
                 for r in rows]
        self.planned = Trajectory.from_poses(timestamps=timestamps, poses=poses)
        self.time_range_us = self.planned.time_range_us
        self.recorded, self.force_gt_end_us = recorded, force_gt_end_us

    def interpolate_pose(self, timestamp_us):
        if timestamp_us not in self.time_range_us:
            raise ValueError("render shutter outside complete planned trajectory")
        if timestamp_us <= self.force_gt_end_us:
            return self.recorded.interpolate_pose(timestamp_us)
        return self.planned.interpolate_pose(timestamp_us)


def runtime_render_traffic(traffic_objects, timing, min_duration_us):
    """Mirror pinned UnboundRollout's window/short-track policy, without a map."""
    if type(min_duration_us) is not int or min_duration_us < 0:
        raise ValueError("nonnegative original minimum traffic duration required")
    window = traffic_objects.clip_trajectories(
        timing.egomotion_context_start_us, timing.end_timestamp_us + 1, exclude_empty=True)
    active = window.filter_short_trajectories(min_duration_us)
    hidden = set(window.keys()) - set(active.keys())
    # CameraEvent excludes retained static tracks. StepEvent accumulates replay
    # poses on policy ticks; shutter-start poses interpolate this history, not
    # the original dense trajectory directly.
    trajectories = {}
    for key, obj in active.items():
        if obj.is_static:
            continue
        accumulated = obj.trajectory.clip(timing.egomotion_context_start_us,
                                           timing.first_policy_timestamp_us + 1)
        for timestamp in range(timing.first_policy_timestamp_us + 100000,
                               timing.end_timestamp_us + 1, 100000):
            if timestamp in obj.trajectory.time_range_us:
                accumulated.update_absolute(timestamp, obj.trajectory.interpolate_pose(timestamp))
        trajectories[key] = accumulated
    hide_offset = Pose(np.array([0., 0., -100.], dtype=np.float32),
                       np.array([0., 0., 0., 1.], dtype=np.float32))
    for key in sorted(hidden):
        trajectories[key] = window[key].trajectory.transform(hide_offset, is_relative=True)
    return trajectories, {"active_ids": sorted(active.keys()), "hidden_short_track_ids": sorted(hidden),
                          "static_ids_not_sent_by_official_camera_event": sorted(k for k, o in active.items() if o.is_static),
                          "window_ids": sorted(window.keys()), "min_duration_us": min_duration_us,
                          "policy": "pinned_window_short_track_lowering_static_filter_and_100ms_replay_accumulation"}


def build_sparse_plan(artifact, candidate, rows, config, cameras, masks, requested_times):
    """Use official calibration, clocks and RGB request construction; no RPCs."""
    binding = candidate["scene_execution_proposal_not_frozen"]
    scene = candidate["scene_id_private"]
    if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != scene:
        raise ValueError("actual artifact identity mismatch")
    camera_ids = [c.logical_id for c in config.cameras]
    if (set(rows) != {"R0", "R1"} or len(camera_ids) != 4 or set(camera_ids) != CAMERAS
        or config.control_timestep_us != 100000 or config.force_gt_duration_us != 1700000
        or config.assert_zero_decision_delay is not True
        or any(c.frame_interval_us != 100000 or c.shutter_duration_us != 30000 for c in config.cameras)
        or config.image_format != "jpeg"):
        raise ValueError("paired four-camera zero-delay JPEG configuration required")
    offset = binding["start_time_offset_us"]
    if (type(offset) is not int or offset < 0
        or binding["recording_start_timestamp_us"] + offset != binding["start_timestamp_us"]):
        raise ValueError("exact source window required")
    rig = artifact.rig.clip_start(offset) if offset else artifact.rig
    if int(rig.trajectory.time_range_us.start) != binding["start_timestamp_us"]:
        raise ValueError("actual rig start differs from bound source window")
    definitions = {}
    for camera in cameras.available_cameras:
        if camera.logical_id in definitions:
            raise ValueError("duplicate renderer camera definition")
        definitions[camera.logical_id] = CameraDefinition.from_proto(camera)
    if not CAMERAS <= definitions.keys():
        raise ValueError("renderer lacks configured camera calibration")
    catalog = CameraCatalog(None)
    catalog.register_scene_definitions(scene, definitions)
    service = SensorsimService("127.0.0.1:6005", False, catalog)
    timing = _build_rollout_timing(config, rig, list(config.cameras), renderer_service=service)
    if timing.n_sim_steps != config.n_sim_steps:
        raise ValueError("do not clip requested probe task horizon")
    duration = config.n_sim_steps * config.control_timestep_us / 1e6
    delay = int(timing.render_start_timestamp_us) - binding["start_timestamp_us"]
    indices = sample_indices(requested_times, duration, delay, config.control_timestep_us)
    for arm, values in rows.items():
        values = np.asarray(values, dtype=float)
        if (values.ndim != 2 or values.shape[1] != 5 or len(values) < 3
            or not np.isfinite(values).all()):
            raise ValueError("finite dense planned poses required")
        if values[-1, 0] < duration + 6 or not np.array_equal(values[:, 0], np.asarray(rows["R0"])[:, 0]):
            raise ValueError("identical dense clocks with complete six-second future required")
        prefix = values[:, 0] <= 1.7
        if not np.array_equal(values[prefix], np.asarray(rows["R0"])[prefix]):
            raise ValueError("planned non-language prefixes differ")
    traffic, policy = runtime_render_traffic(artifact.traffic_objects, timing, config.min_traffic_duration_us)
    plan = {"scope": SCOPE, "qualified": False, "human_reviewed": False,
            "continuous_support_verified": False, "actual_oracle_witness": False,
            "scene_id": scene, "artifact_uuid": candidate["artifact_uuid"],
            "proposal_key": candidate["proposal_key"], "rollout_duration_s": duration,
            "pose_method": "official recorded force-GT prefix then planned XYZ/yaw without physics ground or attitude correction",
            "timing": {"ego_start_us": binding["start_timestamp_us"],
                       "render_start_us": int(timing.render_start_timestamp_us),
                       "closed_loop_start_us": int(timing.closed_loop_start_us),
                       "end_timestamp_us": int(timing.end_timestamp_us)},
            "traffic_policy": policy, "frames": [],
            "ego_mask_policy": "official per-camera lookup; absent masks remain absent",
            "limitations": ["sparse frames cannot prove continuous support or safety",
                            "planned camera poses are not closed-loop tracked poses",
                            "successful decoding is not visual approval",
                            "full oracle, traffic legality, all-camera review and three seeds still required"]}
    for arm in ("R0", "R1"):
        trajectory = PlannedRigTrajectory(rows[arm], binding["start_timestamp_us"],
                                          rig.trajectory, int(timing.closed_loop_start_us))
        for requested, index in strict_zip(requested_times, indices):
            for camera_config in config.cameras:
                camera = RuntimeCamera.from_camera_config(
                    camera_config, timing.first_camera_frame_ranges_us[camera_config.logical_id])
                trigger = camera.clock.ith_trigger(index)
                mask = service.determine_ego_mask_id(masks, camera.logical_id, config.ego_mask_rig_config_id)
                request = service.construct_rgb_render_request(
                    trajectory, traffic, camera, trigger, scene, pb.JPEG, mask)
                plan["frames"].append({"arm": arm, "camera": camera.logical_id,
                    "requested_time_s": requested, "time_s": (request.frame_end_us-binding["start_timestamp_us"])/1e6,
                    "frame_index": index, "native_size_wh": list(definitions[camera.logical_id].native_resolution_hw[::-1]),
                    "insert_ego_mask": request.insert_ego_mask,
                    "requested_size_wh": [request.resolution_w, request.resolution_h],
                    "request_base64": base64.b64encode(request.SerializeToString()).decode()})
    return plan
