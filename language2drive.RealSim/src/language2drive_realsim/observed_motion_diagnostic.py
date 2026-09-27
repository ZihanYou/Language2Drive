"""Past estimated-pose finite difference; diagnostic only, never a state repair."""
import math


def velocity_from_observed_poses(poses, frame_timestamp_us):
    """Return xyz velocity in latest rig axes from two timestamped local poses.

Each pose contains timestamp_us, position_xyz, quaternion_xyzw. This is an
interval-average approximation rotated at the current heading, not a measured
instantaneous velocity. No GT/future state or acceleration is inferred.
"""
    if type(frame_timestamp_us) is not int or frame_timestamp_us < 0:
        raise ValueError("frame timestamp must be a nonnegative integer")
    if not isinstance(poses, list) or len(poses) != 2:
        raise ValueError("exactly two observed poses required")
    times, positions, quaternions = [], [], []
    for pose in poses:
        if not isinstance(pose, dict) or set(pose) != {"timestamp_us", "position_xyz", "quaternion_xyzw"}:
            raise ValueError("invalid observed pose schema")
        timestamp = pose["timestamp_us"]
        if type(timestamp) is not int or not 0 <= timestamp <= frame_timestamp_us:
            raise ValueError("invalid/future pose timestamp")
        for key, count, destination in (("position_xyz", 3, positions), ("quaternion_xyzw", 4, quaternions)):
            vector = pose[key]
            if not isinstance(vector, list) or len(vector) != count or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in vector
            ):
                raise ValueError("invalid/nonfinite observed pose vector")
            destination.append([float(v) for v in vector])
        times.append(timestamp)
    if times[0] >= times[1] or times[1] != frame_timestamp_us:
        raise ValueError("poses must increase strictly and end at frame timestamp")
    for quaternion in quaternions:
        norm = math.sqrt(sum(v * v for v in quaternion))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
            raise ValueError("observed quaternion must be near unit length")
    dt = (times[1] - times[0]) * 1e-6
    velocity = [(b - a) / dt for a, b in zip(positions[0], positions[1])]
    norm = math.sqrt(sum(v * v for v in quaternions[1]))
    x, y, z, w = [v / norm for v in quaternions[1]]
    q = [-x, -y, -z]  # Inverse current rotation: local displacement -> current rig.

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    twice_cross = [2 * v for v in cross(q, velocity)]
    correction = cross(q, twice_cross)
    result = [v + w * t + c for v, t, c in zip(velocity, twice_cross, correction)]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite finite-difference velocity")
    return result
