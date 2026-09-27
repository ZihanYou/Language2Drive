"""Fixed lateral perturbations of authentic render requests; not support proof."""
import math


def lateral_request(request, offset_m, start_yaw_rad, end_yaw_rad):
    """Shift both shutter poses along actual ego left, preserving other fields."""
    if (any(type(v) not in (int, float) or not math.isfinite(v)
            for v in (offset_m, start_yaw_rad, end_yaw_rad)) or abs(offset_m) > .25):
        raise ValueError("probe requires finite headings and a lateral offset within 0.25 m")
    result = type(request)()
    result.CopyFrom(request)
    for pose, yaw in ((result.sensor_pose.start_pose, start_yaw_rad),
                      (result.sensor_pose.end_pose, end_yaw_rad)):
        pose.vec.x += -math.sin(yaw) * offset_m
        pose.vec.y += math.cos(yaw) * offset_m
    return result
