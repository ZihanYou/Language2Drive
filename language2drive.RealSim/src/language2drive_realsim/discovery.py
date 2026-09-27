"""Scene catalog and downloaded-artifact discovery.

Catalog presence alone never implies eligibility.  A candidate only passes the
automatic pre-filter when its exact catalog artifact is present, is a readable
USDZ with matching private identity metadata, and has a complete reviewed
annotation.  This prevents a public CSV row or Git LFS pointer from being
mistaken for real simulator evidence.
"""

import csv
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree

from .io import load_json


def _suite_uuids(path: Path, suite: str) -> set:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {
            row["uuid"]
            for row in csv.DictReader(stream)
            if row["test_suite_id"] == suite
        }


def _yaml_scalar(text: str, key: str) -> Optional[str]:
    """Read the simple top-level scalars used by NuRec ``metadata.yaml``."""

    match = re.search(r"^%s:\s*['\"]?([^'\"\n]+?)['\"]?\s*$" % re.escape(key), text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _yaml_time_range(text: str) -> Dict[str, Optional[int]]:
    match = re.search(
        r"^time_range:\s*\n(?:[ \t]+.*\n)*?[ \t]+end:\s*(\d+)\s*\n[ \t]+start:\s*(\d+)\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        return {"start_timestamp_us": None, "end_timestamp_us": None, "duration_s": None}
    end = int(match.group(1))
    start = int(match.group(2))
    return {
        "start_timestamp_us": start,
        "end_timestamp_us": end,
        "duration_s": (end - start) / 1_000_000.0,
    }


def _xodr_inventory(payload: bytes) -> Dict[str, Any]:
    root = ElementTree.fromstring(payload)
    roads = root.findall("road")
    junctions = root.findall("junction")
    driving_lane_counts: List[int] = []
    three_lane_sections = 0
    for section in root.findall("./road/lanes/laneSection"):
        left = len(section.findall("./left/lane[@type='driving']"))
        right = len(section.findall("./right/lane[@type='driving']"))
        driving_lane_counts.extend((left, right))
        three_lane_sections += int(left >= 3) + int(right >= 3)
    return {
        "road_count": len(roads),
        "junction_count": len(junctions),
        "junction_connection_count": len(root.findall("./junction/connection")),
        "driving_lane_count_max_one_direction": max(driving_lane_counts, default=0),
        "three_lane_section_count": three_lane_sections,
        "signal_count": len(root.findall("./road/signals/signal")),
        "object_count": len(root.findall("./road/objects/object")),
    }


def inspect_downloaded_artifact(path: Path, catalog_row: Dict[str, str]) -> Dict[str, Any]:
    """Inspect one local USDZ without extracting its multi-gigabyte payload."""

    path = Path(path)
    result: Dict[str, Any] = {
        "present": path.is_file(),
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "readable_usdz": False,
        "identity_matches_catalog": False,
        "required_members_present": False,
        "metadata": {},
        "topology_inventory": {},
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("artifact_not_downloaded")
        return result
    if path.stat().st_size < 1024:
        result["errors"].append("artifact_is_pointer_or_placeholder")
        return result
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {"metadata.yaml", "data_info.json", "rig_trajectories.json"}
            map_members = {"map.xodr", "clipgt/lane.parquet"}
            missing = sorted(required - names)
            if missing:
                result["errors"].append("missing_members:" + ",".join(missing))
            if not (map_members & names):
                result["errors"].append("missing_map_input")
            result["required_members_present"] = not missing and bool(map_members & names)

            metadata_text = archive.read("metadata.yaml").decode("utf-8")
            metadata = {
                "scene_id": _yaml_scalar(metadata_text, "scene_id"),
                "artifact_uuid": _yaml_scalar(metadata_text, "uuid"),
                "nre_version": _yaml_scalar(metadata_text, "version_string"),
            }
            metadata.update(_yaml_time_range(metadata_text))
            metadata["first_frame_camera_count"] = len(
                {
                    name.split("/")[1]
                    for name in names
                    if name.startswith("frames/") and name.lower().endswith((".jpg", ".jpeg", ".png"))
                }
            )
            result["metadata"] = metadata
            result["identity_matches_catalog"] = (
                metadata["scene_id"] == catalog_row["scene_id"]
                and metadata["artifact_uuid"] == catalog_row["uuid"]
                and metadata["nre_version"] == catalog_row["nre_version_string"]
            )
            if not result["identity_matches_catalog"]:
                result["errors"].append("artifact_identity_mismatch")
            if "map.xodr" in names:
                try:
                    result["topology_inventory"] = _xodr_inventory(archive.read("map.xodr"))
                except (ElementTree.ParseError, ValueError) as error:
                    result["errors"].append("invalid_xodr:%s" % type(error).__name__)
            result["readable_usdz"] = True
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        result["errors"].append("unreadable_usdz:%s" % type(error).__name__)
    return result


def _annotation_review_errors(annotation: Any) -> List[str]:
    """Reject placeholders before a scene can pass the automatic pre-filter."""

    if not isinstance(annotation, dict):
        return ["annotation_not_an_object"]
    required = {
        "topology",
        "reconstruction_support",
        "camera_visibility",
        "decision_windows",
    }
    missing = sorted(required - set(annotation))
    if missing:
        return ["annotation_missing:" + ",".join(missing)]

    errors = []
    for name in ("topology", "reconstruction_support", "camera_visibility"):
        section = annotation[name]
        if not isinstance(section, dict) or section.get("reviewed") is not True:
            errors.append("annotation_unreviewed:" + name)

    support = annotation["reconstruction_support"]
    if not isinstance(support, dict) or support.get("both_arms_supported") is not True:
        errors.append("annotation_reconstruction_support_not_dual_arm")

    visibility = annotation["camera_visibility"]
    resolution = visibility.get("model_input_resolution") if isinstance(visibility, dict) else None
    if (
        not isinstance(visibility, dict)
        or visibility.get("observable") is not True
        or not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolution)
    ):
        errors.append("annotation_camera_visibility_incomplete")

    windows = annotation["decision_windows"]
    if not isinstance(windows, list) or not windows:
        errors.append("annotation_decision_windows_missing")
    else:
        for index, window in enumerate(windows):
            if not isinstance(window, dict) or any(
                window.get(field) is not True
                for field in ("reviewed", "both_arms_legal", "both_arms_reachable")
            ):
                errors.append("annotation_decision_window_unreviewed:%d" % index)
    return errors


def discover_candidates(
    scenes_csv: Path,
    suites_csv: Path,
    suite: str = "public_2601",
    limit: int = 120,
    annotations_path: Optional[Path] = None,
    scene_cache: Optional[Path] = None,
    selected_scene_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    annotations = {}
    if annotations_path is not None and Path(annotations_path).exists():
        payload = load_json(annotations_path)
        annotations = payload.get("scenes", {})
    allowed = _suite_uuids(suites_csv, suite)
    selected = set(selected_scene_ids) if selected_scene_ids is not None else None
    if scene_cache is None:
        scene_cache = Path(scenes_csv).parents[1] / "nre-artifacts" / "all-usdzs"
    candidates: List[Dict[str, Any]] = []
    with Path(scenes_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["uuid"] not in allowed:
                continue
            bare_scene_id = (
                row["scene_id"][len("clipgt-") :]
                if row["scene_id"].startswith("clipgt-")
                else row["scene_id"]
            )
            if selected is not None and bare_scene_id not in selected:
                continue
            annotation = annotations.get(row["uuid"])
            reasons = []
            artifact = inspect_downloaded_artifact(Path(scene_cache) / row["path"], row)
            reasons.extend(artifact["errors"])
            if artifact["readable_usdz"] and not artifact["required_members_present"]:
                reasons.append("artifact_incomplete")
            if annotation is None:
                reasons.append("missing_private_topology_semantic_annotation")
            else:
                reasons.extend(_annotation_review_errors(annotation))
            candidates.append(
                {
                    "candidate_id": "candidate-%04x" % (len(candidates) + 1),
                    "artifact_id": row["uuid"],
                    "scene_id_private": row["scene_id"],
                    "dataset_version": row["hf_revision"],
                    "nre_version": row["nre_version_string"],
                    "artifact_path": row["path"],
                    "catalog_source": suite,
                    "downloaded_artifact": artifact,
                    "annotation_present": annotation is not None,
                    "automatic_prefilter_pass": not reasons,
                    "rejection_reasons": reasons,
                }
            )
            if limit > 0 and len(candidates) >= limit:
                break
    return {
        "schema_version": "l2d.realsim.candidates.v0.1",
        "suite": suite,
        "candidate_scope": "reconstructed_scene_acquisition_candidate_not_pcu",
        "selection_scope": "explicit_scene_ids" if selected is not None else "suite_catalog",
        "candidate_count": len(candidates),
        "downloaded_artifact_count": sum(row["downloaded_artifact"]["present"] for row in candidates),
        "verified_artifact_count": sum(
            row["downloaded_artifact"]["readable_usdz"]
            and row["downloaded_artifact"]["required_members_present"]
            and row["downloaded_artifact"]["identity_matches_catalog"]
            for row in candidates
        ),
        "prefilter_pass_count": sum(row["automatic_prefilter_pass"] for row in candidates),
        "candidates": candidates,
    }
