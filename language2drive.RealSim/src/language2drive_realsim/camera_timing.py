"""Read official camera anchors before delegated GPU pilots, without hashes."""


def aligned_camera_window(artifact, binding, camera_ids, horizon_us=13000000):
    """Select an actual shutter opening; callers must regenerate all plans."""
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("distinct configured cameras required")
    if type(horizon_us) is not int or horizon_us < 13000000:
        raise ValueError("complete 13-second or longer horizon required")
    offset = binding["start_time_offset_us"]
    if (type(offset) is not int or offset < 0
        or binding["recording_start_timestamp_us"] + offset != binding["start_timestamp_us"]):
        raise ValueError("consistent nonnegative source window required")
    rig = artifact.rig.clip_start(offset) if offset else artifact.rig
    if int(rig.trajectory.time_range_us.start) != binding["start_timestamp_us"]:
        raise ValueError("actual source ego start differs from requested window")
    ranges = rig.first_camera_frame_ranges_us(camera_ids)
    if set(ranges) != set(camera_ids) or any(r.stop <= r.start for r in ranges.values()):
        raise ValueError("complete nonempty actual camera ranges required")
    new_start = min(r.start for r in ranges.values())
    new_offset = new_start - binding["recording_start_timestamp_us"]
    if (new_start < binding["start_timestamp_us"]
        or new_start + horizon_us >= binding["recording_end_timestamp_us"]):
        raise ValueError("aligned camera window must retain the complete real horizon")
    aligned = artifact.rig.clip_start(new_offset) if new_offset else artifact.rig
    timing = check_camera_anchor(aligned, camera_ids, new_start)
    return {"old_start_timestamp_us": binding["start_timestamp_us"],
            "old_start_offset_us": offset, "new_start_timestamp_us": new_start,
            "new_start_offset_us": new_offset, "reused_old_plans": False,
            "new_cpu_traffic_search_required": True, "actual_preflight": timing}


def check_camera_anchor(rig, camera_ids, expected_start_us):
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("distinct configured cameras required")
    start = int(rig.trajectory.time_range_us.start)
    if start != expected_start_us:
        raise ValueError("actual clipped ego start differs from requested window")
    ranges = rig.first_camera_frame_ranges_us(camera_ids)
    render_start = int(rig.first_camera_frame_end_us(camera_ids))
    delay = render_start - start
    # Match existing real-pose/video coverage checks; do not reinterpret a late
    # first render as if an image had been observed at the requested start.
    if not 0 <= delay <= 100000:
        raise ValueError(f"first camera render delay {delay} us exceeds the 100000 us review-coverage limit; choose an exposure-aligned window")
    return {"scope": "actual_camera_anchor_preflight_not_qualification", "qualified": False,
            "ego_start_us": start, "render_start_us": render_start, "render_delay_us": delay,
            "first_camera_ranges_us": {k: [r.start, r.stop] for k, r in ranges.items()}}


def camera_timing_preflight(candidate, camera_ids):
    from alpasim_utils.artifact import Artifact
    binding = candidate["scene_execution_proposal_not_frozen"]
    artifact = Artifact(binding["artifact_path"])
    if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != candidate["scene_id_private"]:
        raise ValueError("camera timing artifact identity mismatch")
    offset = binding["start_time_offset_us"]
    rig = artifact.rig.clip_start(offset) if offset else artifact.rig
    return check_camera_anchor(rig, camera_ids, binding["start_timestamp_us"])
