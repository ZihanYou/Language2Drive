from types import SimpleNamespace

import pytest

from language2drive_realsim.camera_timing import aligned_camera_window, check_camera_anchor


def rig(delay):
    return SimpleNamespace(trajectory=SimpleNamespace(time_range_us=range(1000000, 20000000)),
        first_camera_frame_ranges_us=lambda ids: {k: range(1000000 + delay - 30000, 1000000 + delay) for k in ids},
        first_camera_frame_end_us=lambda ids: 1000000 + delay)


def test_actual_camera_anchor_is_recorded_without_claiming_qualification():
    report = check_camera_anchor(rig(30559), ["front", "left", "right", "tele"], 1000000)
    assert report["render_delay_us"] == 30559
    assert report["qualified"] is False


@pytest.mark.parametrize("delay", [-1, 100001, 213722])
def test_bad_initial_camera_coverage_rejected_before_gpu(delay):
    with pytest.raises(ValueError, match="render delay"):
        check_camera_anchor(rig(delay), ["front"], 1000000)


def test_mismatched_window_or_camera_selection_rejected():
    with pytest.raises(ValueError, match="ego start"):
        check_camera_anchor(rig(30000), ["front"], 1000001)
    for ids in ([], ["front", "front"]):
        with pytest.raises(ValueError, match="distinct"):
            check_camera_anchor(rig(30000), ids, 1000000)


def alignment_fixture(opening=1050000, end=21000000, missing=False):
    def clipped(offset=0):
        start = 1000000 + offset
        return SimpleNamespace(
            trajectory=SimpleNamespace(time_range_us=range(start, end)),
            first_camera_frame_ranges_us=lambda ids: {
                key: range(opening, opening + 30559)
                for key in (ids[:-1] if missing else ids)},
            first_camera_frame_end_us=lambda ids: opening + 30559,
            clip_start=clipped)
    return SimpleNamespace(rig=clipped()), {
        "recording_start_timestamp_us": 1000000, "start_time_offset_us": 0,
        "start_timestamp_us": 1000000, "recording_end_timestamp_us": end}


def test_camera_alignment_is_new_window_not_retimed_plan():
    artifact, binding = alignment_fixture()
    result = aligned_camera_window(artifact, binding, ["front", "tele"])
    assert result["new_start_timestamp_us"] == 1050000
    assert result["new_start_offset_us"] == 50000
    assert result["actual_preflight"]["render_delay_us"] == 30559
    assert result["new_cpu_traffic_search_required"] is True
    assert result["reused_old_plans"] is False
    assert binding["start_timestamp_us"] == 1000000


@pytest.mark.parametrize("opening,end", [(999999, 21000000), (8000000, 21000000), (8000001, 21000000)])
def test_camera_alignment_rejects_backward_or_incomplete_horizon(opening, end):
    artifact, binding = alignment_fixture(opening, end)
    with pytest.raises(ValueError, match="complete real horizon"):
        aligned_camera_window(artifact, binding, ["front"])


def test_camera_alignment_rejects_missing_camera():
    artifact, binding = alignment_fixture(missing=True)
    with pytest.raises(ValueError, match="complete nonempty"):
        aligned_camera_window(artifact, binding, ["front", "tele"])


@pytest.mark.parametrize("horizon", [12000000, 13000000.0, True])
def test_camera_alignment_cannot_shorten_horizon(horizon):
    artifact, binding = alignment_fixture()
    with pytest.raises(ValueError, match="13-second"):
        aligned_camera_window(artifact, binding, ["front"], horizon)


def test_camera_alignment_rejects_inconsistent_offset():
    artifact, binding = alignment_fixture()
    binding["start_time_offset_us"] = 1
    with pytest.raises(ValueError, match="consistent nonnegative"):
        aligned_camera_window(artifact, binding, ["front"])
