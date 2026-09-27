import math

import pytest
pb = pytest.importorskip('alpasim_grpc.v0.sensorsim_pb2',
                        reason='Optional upstream simulator protobuf integration')

from language2drive_realsim.render_neighborhood import lateral_request


def request():
    r = pb.RGBRenderRequest(scene_id="scene-test", frame_start_us=100, frame_end_us=200,
                            resolution_w=1900, resolution_h=1080, image_format=pb.JPEG,
                            image_quality=95, insert_ego_mask=True)
    for pose in (r.sensor_pose.start_pose, r.sensor_pose.end_pose):
        pose.vec.x, pose.vec.y, pose.vec.z = 10, 20, 1
        pose.quat.w = 1
    r.dynamic_objects.add(track_id="actor-test")
    r.camera_intrinsics.logical_id = "camera_front_wide_120fov"
    return r


@pytest.mark.parametrize("offset", [-.25, .25])
def test_translation_preserves_exact_nonpose_fields_and_original_request(offset):
    original = request()
    unchanged = original.SerializeToString()
    shifted = lateral_request(original, offset, 0, math.pi/2)
    assert shifted.sensor_pose.start_pose.vec.x == 10
    assert shifted.sensor_pose.start_pose.vec.y == pytest.approx(20+offset)
    assert shifted.sensor_pose.end_pose.vec.x == pytest.approx(10-offset)
    assert shifted.sensor_pose.end_pose.vec.y == 20
    assert shifted.sensor_pose.start_pose.vec.z == shifted.sensor_pose.end_pose.vec.z == 1
    assert shifted.sensor_pose.start_pose.quat == original.sensor_pose.start_pose.quat
    assert shifted.sensor_pose.end_pose.quat == original.sensor_pose.end_pose.quat
    shifted.sensor_pose.CopyFrom(original.sensor_pose)
    assert shifted == original
    assert original.SerializeToString() == unchanged


def test_zero_offset_replays_exact_request():
    r = request()
    assert lateral_request(r, 0., .2, -.3) == r


@pytest.mark.parametrize("args", [(float("nan"),0,0),(.5,0,0),(-.5,0,0),(.25,float("inf"),0),(.25,0,float("nan")),(True,0,0)])
def test_nonfinite_or_out_of_protocol_offsets_rejected(args):
    with pytest.raises(ValueError):
        lateral_request(request(), *args)
