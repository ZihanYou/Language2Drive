"""CPU-only novel rear-view request construction, never a native/approved view.

Poses are parent_from_child transforms. Optical axes x-right, y-down,
z-forward map to rig +Y, -Z, -X for the central rear orientation.
Caller-supplied translation/calibration are hypotheses, not official extrinsics.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2


REAR_ROTATION = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])


def _finite_message(message):
    for field, value in message.ListFields():
        values = value if field.label == field.LABEL_REPEATED else [value]
        for item in values:
            if field.type == field.TYPE_MESSAGE:
                _finite_message(item)
            elif field.type in (field.TYPE_DOUBLE, field.TYPE_FLOAT):
                if not math.isfinite(item):
                    raise ValueError("nonfinite protobuf field: " + field.name)


def _matrix(pose):
    if not isinstance(pose, common_pb2.Pose) or not pose.HasField("quat"):
        raise ValueError("explicit common.Pose quaternion required")
    _finite_message(pose)
    quat = np.array([pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w])
    if not np.isclose(np.linalg.norm(quat), 1., rtol=0., atol=1e-5):
        raise ValueError("quaternion must already be normalized")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
    matrix[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    return matrix


def _pose(matrix):
    result = common_pb2.Pose()
    result.vec.x, result.vec.y, result.vec.z = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    result.quat.x, result.quat.y, result.quat.z, result.quat.w = quat
    _matrix(result)  # Also reject overflow when converted to protobuf float32.
    return result


def central_rear_pose(translation_xyz):
    """Build rig_from_rear; no default or guessed official translation."""
    translation = np.asarray(translation_xyz, dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("three finite caller-specified translation coordinates required")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = REAR_ROTATION, translation
    return _pose(matrix)


def _camera(spec):
    if not isinstance(spec, sensorsim_pb2.CameraSpec):
        raise ValueError("CameraSpec required")
    if not spec.logical_id or not spec.resolution_w or not spec.resolution_h:
        raise ValueError("explicit camera identity and positive dimensions required")
    kind = spec.WhichOneof("camera_param")
    if kind is None or spec.shutter_type not in (1, 2, 3, 4, 5):
        raise ValueError("explicit calibration and supported shutter required")
    _finite_message(spec)
    params = getattr(spec, kind)
    if kind in ("opencv_pinhole_param", "opencv_fisheye_param"):
        if params.focal_length_x <= 0 or params.focal_length_y <= 0:
            raise ValueError("positive focal lengths required")
    elif not params.pixeldist_to_angle_poly and not params.angle_to_pixeldist_poly:
        raise ValueError("ftheta calibration polynomial required")


def transform_rear_request(source, rig_from_source, rig_from_target, target_intrinsics):
    """Return (deep-copied RGBRenderRequest, diagnostic-only provenance).

    Both shutter poses use W_source @ inverse(R_source) @ R_target.
    Source mask may only be disabled: this API cannot enable/reuse a hood mask.
    It does not attest visual support, optical calibration, or camera approval.
    """
    if not isinstance(source, sensorsim_pb2.RGBRenderRequest):
        raise ValueError("RGBRenderRequest required")
    _finite_message(source)
    _camera(source.camera_intrinsics)
    _camera(target_intrinsics)
    if not source.scene_id or not source.resolution_w or not source.resolution_h:
        raise ValueError("source scene and positive dimensions required")
    if source.frame_start_us > source.frame_end_us:
        raise ValueError("reversed shutter timestamps")
    if source.image_format not in (sensorsim_pb2.PNG, sensorsim_pb2.JPEG):
        raise ValueError("AlpaSim PNG/JPEG request required")
    if source.camera_intrinsics.logical_id == target_intrinsics.logical_id:
        raise ValueError("novel target must use a distinct logical_id")
    source_rig, target_rig = _matrix(rig_from_source), _matrix(rig_from_target)
    for obj in source.dynamic_objects:
        _matrix(obj.pose_pair.start_pose)
        _matrix(obj.pose_pair.end_pose)
    result = sensorsim_pb2.RGBRenderRequest()
    result.CopyFrom(source)
    for endpoint in ("start_pose", "end_pose"):
        world_source = _matrix(getattr(source.sensor_pose, endpoint))
        getattr(result.sensor_pose, endpoint).CopyFrom(
            _pose(world_source @ np.linalg.inv(source_rig) @ target_rig))
    result.camera_intrinsics.CopyFrom(target_intrinsics)
    result.resolution_w = target_intrinsics.resolution_w
    result.resolution_h = target_intrinsics.resolution_h
    result.insert_ego_mask = False
    result.ClearField("ego_mask_id")
    provenance = dict(
        diagnostic_only=True, formal_evaluation=False, native_support_verified=False,
        camera_approved=False, official_equivalence_claimed=False,
        source_camera=source.camera_intrinsics.logical_id,
        target_camera=target_intrinsics.logical_id, source_ego_mask_disabled=True,
        formula="world_from_source @ inverse(rig_from_source) @ rig_from_target",
        target_translation_xyz=target_rig[:3, 3].tolist(),
        optical_axes_in_rig_columns=target_rig[:3, :3].T.tolist(),
        central_rear_orientation=bool(np.allclose(
            target_rig[:3, :3], REAR_ROTATION, rtol=0., atol=1e-5)),
        limitations=["caller-specified extrinsics/intrinsics, not official calibration",
                     "novel-view visual support and image quality remain unverified"],
    )
    return result, provenance
