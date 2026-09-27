"""Cheap original-label exclusions before costly A4 map and motion loading."""
import io
from pathlib import Path
import zipfile


def audit_bilateral_lane_use(candidate, raw_lanes):
    """Reject explicit shoulders/non-car lanes; absence of labels is not legality."""
    pair = candidate["selected_arm_pair"]
    if pair["kind"] != "left_vs_right":
        raise ValueError("original bilateral lane candidate required")
    groups = {"middle": str(candidate["initial_middle_lane_hypothesis"])}
    if [(a["arm"], a["role"]) for a in pair["arms"]] != [
        ("R0", "change_left"), ("R1", "change_right")
    ]:
        raise ValueError("exact left/right arm roles required")
    groups.update({a["arm"]: str(a["lane_id"]) for a in pair["arms"]})
    if len(set(groups.values())) != 3:
        raise ValueError("three distinct original lane identities required")
    clip = candidate["scene_id_private"].removeprefix("clipgt-")
    checked = []
    for group, identifier in groups.items():
        matches = [r for r in raw_lanes if r["key"]["map_id"] == identifier]
        if len(matches) != 1 or matches[0]["key"]["clip_id"] != clip:
            raise ValueError("one matching original clip/lane record required")
        row = matches[0]
        lane = row["lane"]
        use, vehicles = lane["use_types"], lane["vehicle_types"]
        if any(not isinstance(v, list) or any(not isinstance(x, str) for x in v) for v in (use, vehicles)):
            raise ValueError("original lane use and vehicle label lists required")
        reasons = []
        if "SHOULDER_LANE" in use:
            reasons.append("explicit_original_shoulder_not_ordinary_lane_change_target")
        if vehicles and "CAR" not in vehicles:
            reasons.append("explicit_original_vehicle_types_exclude_car")
        checked.append(dict(group=group, source_lane_record=row, exclusion_reasons=reasons))
    versions = {r["source_lane_record"]["key"]["map_id_version"] for r in checked}
    if len(versions) != 1:
        raise ValueError("original three-lane map versions differ")
    return dict(scope="original_A4_lane_use_exclusion_not_legality_approval", qualified=False,
                human_reviewed=False, legality_approved=False,
                selected_lanes_only=True, full_corridor_use_not_checked=True,
                passed=not any(r["exclusion_reasons"] for r in checked), checks=checked)


def read_bilateral_lane_use(candidate):
    import pyarrow.parquet as pq
    path = Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"]).resolve()
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        rows = pq.read_table(io.BytesIO(archive.read("clipgt/lane.parquet")), use_threads=False).to_pylist()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("original asset changed during lane-use read")
    return {**audit_bilateral_lane_use(candidate, rows), "source_artifact": str(path),
            "source_size_bytes": after.st_size, "source_mtime_ns": str(after.st_mtime_ns)}
