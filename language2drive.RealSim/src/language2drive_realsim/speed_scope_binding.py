"""Re-read a preselected original crosswalk scope, never choose it from outcomes."""
from pathlib import Path

from .io import load_json


def selected_speed_scope(payload):
    """One explicit typed landmark scope, or the legacy unbound B1 path."""
    keys = [k for k in ("crosswalk_scope_reference", "sign_scope_reference") if k in payload]
    if len(keys) > 1:
        raise ValueError("crosswalk and sign scopes are mutually exclusive")
    if not keys:
        return None, None
    key = keys[0]
    if not isinstance(payload[key], dict) or not payload[key]:
        raise ValueError("explicit nonempty speed scope reference required")
    return key, payload[key]


def read_bound_speed_scope(key, reference, candidate):
    if key == "crosswalk_scope_reference":
        return read_bound_crosswalk_scope(reference, candidate)
    if key == "sign_scope_reference":
        return read_bound_sign_scope(reference, candidate)
    raise ValueError("unknown speed landmark scope kind")


def read_bound_sign_scope(reference, candidate):
    """Bind the exact original sign scope; infer no speed units or review."""
    from alpasim_utils.artifact import Artifact
    from alpasim_runtime.unbound_rollout import get_ds_rig_to_aabb_center_transform
    from .sign_speed_scope import read_sign_approach

    root = Path(__file__).resolve().parents[2]
    path = Path(reference["path"]).resolve()
    if not path.is_relative_to(root / ".runtime/curation_review/spatial"):
        raise ValueError("bound sign scope must be isolated spatial evidence")
    stat = path.stat()
    if stat.st_size != reference["size_bytes"] or str(stat.st_mtime_ns) != reference["mtime_ns"]:
        raise ValueError("preselected sign scope changed")
    approach = load_json(path)
    asset = Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"]).resolve()
    if (approach.get("scope") != "original_speed_limit_sign_approach_not_grounding_or_qualification"
        or approach.get("qualified") is not False or approach.get("human_reviewed") is not False
        or approach.get("speed_limit_value_or_units_inferred") is not False
        or Path(approach["source_artifact"]).resolve() != asset
        or approach["source_lane_ids"][0] != candidate["selected_arm_pair"]["arms"][0]["lane_id"]
        or asset.stat().st_size != approach["source_size_bytes"]
        or str(asset.stat().st_mtime_ns) != approach["source_mtime_ns"]):
        raise ValueError("preselected sign asset, lane or review binding changed")
    artifact = Artifact(str(asset))
    if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != candidate["scene_id_private"]:
        raise ValueError("sign original artifact identity differs")
    actual = read_sign_approach(artifact, approach["source_lane_ids"], approach["source_sign"]["key"]["map_id"],
        far_distance_m=approach["segment"]["far_distance_m"], near_distance_m=approach["segment"]["near_distance_m"])
    if actual != approach:
        raise ValueError("preselected sign differs from the original map")
    return approach, get_ds_rig_to_aabb_center_transform(artifact.rig.vehicle_config).as_se3()


def read_bound_crosswalk_scope(reference, candidate):
    from alpasim_utils.artifact import Artifact
    from alpasim_runtime.unbound_rollout import get_ds_rig_to_aabb_center_transform
    from .crosswalk_speed_scope import read_crosswalk_approach

    root = Path(__file__).resolve().parents[2]
    path = Path(reference["path"]).resolve()
    if not path.is_relative_to(root / ".runtime/curation_review/spatial"):
        raise ValueError("bound crosswalk scope must be isolated spatial evidence")
    stat = path.stat()
    if stat.st_size != reference["size_bytes"] or str(stat.st_mtime_ns) != reference["mtime_ns"]:
        raise ValueError("preselected crosswalk scope changed")
    approach = load_json(path)
    asset = Path(candidate["scene_execution_proposal_not_frozen"]["artifact_path"]).resolve()
    if (approach.get("qualified") is not False or approach.get("human_reviewed") is not False
        or Path(approach["source_artifact"]).resolve() != asset
        or approach["source_lane_ids"][0] != candidate["selected_arm_pair"]["arms"][0]["lane_id"]
        or asset.stat().st_size != approach["source_size_bytes"]
        or str(asset.stat().st_mtime_ns) != approach["source_mtime_ns"]):
        raise ValueError("preselected crosswalk asset or lane binding changed")
    artifact = Artifact(str(asset))
    if artifact.metadata.uuid != candidate["artifact_uuid"] or artifact.metadata.scene_id != candidate["scene_id_private"]:
        raise ValueError("crosswalk original artifact identity differs")
    actual = read_crosswalk_approach(artifact, approach["source_lane_ids"], approach["source_crosswalk"]["key"]["map_id"],
        far_distance_m=approach["segment"]["far_distance_m"], near_distance_m=approach["segment"]["near_distance_m"])
    if actual != approach:
        raise ValueError("preselected crosswalk differs from the original map")
    return approach, get_ds_rig_to_aabb_center_transform(artifact.rig.vehicle_config).as_se3()
