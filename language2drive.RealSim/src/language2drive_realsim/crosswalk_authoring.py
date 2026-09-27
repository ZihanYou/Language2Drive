"""Bind real crosswalk-relative geometry and frames for pending B3 drafts."""
from pathlib import Path
import math

from .io import load_json


def crosswalk_speed_utterances(lower_mps, upper_mps, far_m, near_m):
    values = [lower_mps, upper_mps, far_m, near_m]
    if not all(math.isfinite(v) for v in values) or not 0 <= lower_mps < upper_mps or not 0 < near_m < far_m:
        raise ValueError("ordered finite speed and approach distance bands required")
    lower, upper = lower_mps * 3.6, upper_mps * 3.6
    return [
        f"As you approach the upcoming marked pedestrian crossing, keep {lower:g}–{upper:g} km/h in the current lane's stretch from approximately {far_m:g} to {near_m:g} metres before it.",
        f"On the approach to the pedestrian crossing ahead, maintain between {lower:g} and {upper:g} km/h along the current lane, from about {far_m:g} metres to {near_m:g} metres before the crossing.",
        f"While travelling along the current lane's {far_m:g}-to-{near_m:g}-metre approach to the marked crossing ahead, hold a speed of {lower:g}–{upper:g} km/h.",
    ]


def validate_crosswalk_authoring_inputs(root, candidate, native, approach_path, summary):
    from alpasim_utils.artifact import Artifact
    from .crosswalk_speed_scope import read_crosswalk_approach

    root, approach_path = Path(root).resolve(), Path(approach_path).resolve()
    if not approach_path.is_relative_to(root / ".runtime/curation_review/spatial"):
        raise ValueError("isolated local original crosswalk evidence required")
    approach = load_json(approach_path)
    if (approach.get("scope") != "original_crosswalk_approach_geometry_not_grounding_or_qualification"
        or approach.get("qualified") is not False or approach.get("human_reviewed") is not False
        or approach["source_lane_ids"][0] != candidate["selected_arm_pair"]["arms"][0]["lane_id"]
        or approach["source_lane_ids"][0] != native["source_lane_id"]):
        raise ValueError("unreviewed original crosswalk approach must match the selected lane")
    path = Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"]).resolve()
    if (str(path) != approach["source_artifact"] or path.stat().st_size != approach["source_size_bytes"]
        or str(path.stat().st_mtime_ns) != approach["source_mtime_ns"]):
        raise ValueError("original crosswalk asset binding changed")
    artifact = Artifact(str(path))
    if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != candidate["scene_id_private"]:
        raise ValueError("crosswalk artifact identity differs")
    actual = read_crosswalk_approach(artifact, approach["source_lane_ids"], approach["source_crosswalk"]["key"]["map_id"],
        far_distance_m=approach["segment"]["far_distance_m"], near_distance_m=approach["segment"]["near_distance_m"])
    if actual != approach:
        raise ValueError("crosswalk scope differs from independently reread original geometry")
    for key in ("scope_polygon", "entry_gate", "exit_gate"):
        if native["segment"][key] != approach["segment"][key]:
            raise ValueError("actual native speed scope differs from the fixed crosswalk-relative scope")
    refs, bindings = validate_actual_approach_frames(root, candidate, native, summary)
    return approach, refs, bindings


def validate_actual_approach_frames(root, candidate, native, summary):
    """Bind actual model-sized frames, without granting referent visibility."""
    root = Path(root).resolve()
    first = Path(summary["pairs"][0]["run_dir"])
    folder = root / ".runtime/curation_review/agent-feasibility" / candidate["proposal_key"] / first.name
    inventory = load_json(folder / "inventory.private.json")
    if (inventory["proposal_key"] != candidate["proposal_key"] or inventory["seed"] != summary["pairs"][0]["seed"]
        or inventory["final_human_review_pending"] is not True):
        raise ValueError("matching pending actual review packet required")
    inspected = load_json(first / "speed-pair-inspection.private.json")
    # Pre-instruction context plus both arms before the common active segment.
    targets = [("R0", "camera_front_wide_120fov", 1.5), ("R0", "camera_front_tele_30fov", 1.5),
               ("R0", "camera_front_wide_120fov", 3.), ("R1", "camera_front_wide_120fov", 3.)]
    deadline = min(row["own"]["diagnostics"]["entry_time_s"] for row in native["arms"])
    refs, bindings = [], []
    for arm, camera, requested in targets:
        matches = [r for r in inventory["frames"] if (r["arm"], r["camera"], r["requested_time_s"]) == (arm, camera, requested)]
        if len(matches) != 1:
            raise ValueError("one actual frame for each proposed referent view required")
        row = matches[0]
        path = (folder / row["preview_file"]).resolve()
        if (not path.is_relative_to(folder) or row["preview_size_wh"] != [576, 320]
            or path.stat().st_size != row["preview_size_bytes"] or not 0 <= row["actual_time_s"] < deadline
            or (requested == 1.5 and row["actual_time_s"] >= 1.7)
            or row["source_asl"] != inspected["arms"][arm]["source_asl"]):
            raise ValueError("actual referent view timing, ASL, dimensions or file binding differs")
        refs.append(str(path))
        bindings.append({**row, "preview_path": str(path), "preview_mtime_ns": str(path.stat().st_mtime_ns)})
    return refs, bindings
