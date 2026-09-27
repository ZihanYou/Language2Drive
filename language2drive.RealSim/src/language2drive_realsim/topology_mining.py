"""Fail-closed OpenDRIVE feature mining for downloaded NuRec shortlist scenes."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree

from .acquisition import _catalog_rows, _scene_ids
from .discovery import inspect_downloaded_artifact
from .io import load_json
from .route_topology_mining import mine_route_aligned_artifact


def mine_xodr(payload: bytes) -> Dict[str, Any]:
    root = ElementTree.fromstring(payload)
    road_rows: List[Dict[str, Any]] = []
    three_lane_sections = 0
    two_lane_sections = 0
    for road in root.findall("road"):
        section_rows = []
        for section in road.findall("./lanes/laneSection"):
            left = len(section.findall("./left/lane[@type='driving']"))
            right = len(section.findall("./right/lane[@type='driving']"))
            three_lane_sections += int(left >= 3) + int(right >= 3)
            two_lane_sections += int(left >= 2) + int(right >= 2)
            section_rows.append({"s": float(section.get("s", "0")), "left": left, "right": right})
        road_rows.append(
            {
                "road_id": road.get("id"),
                "name": road.get("name", ""),
                "length_m": float(road.get("length", "0")),
                "junction_id": road.get("junction"),
                "lane_sections": section_rows,
                "signal_count": len(road.findall("./signals/signal")),
                "object_count": len(road.findall("./objects/object")),
            }
        )

    decisions = []
    for junction in root.findall("junction"):
        by_incoming: Dict[str, List[Dict[str, Any]]] = {}
        for connection in junction.findall("connection"):
            lane_links = connection.findall("laneLink")
            if not lane_links:
                continue
            incoming = connection.get("incomingRoad", "")
            by_incoming.setdefault(incoming, []).append(
                {
                    "connection_id": connection.get("id"),
                    "connecting_road": connection.get("connectingRoad"),
                    "contact_point": connection.get("contactPoint"),
                    "lane_links": [
                        {"from": link.get("from"), "to": link.get("to")} for link in lane_links
                    ],
                }
            )
        for incoming, connections in sorted(by_incoming.items()):
            movements = sorted({item["connecting_road"] for item in connections})
            if len(movements) >= 2:
                decisions.append(
                    {
                        "junction_id": junction.get("id"),
                        "incoming_road": incoming,
                        "movement_choice_count": len(movements),
                        "connecting_roads": movements,
                        "connections": connections,
                    }
                )

    object_descriptors = sorted(
        {
            value
            for element in root.findall("./road/objects/object")
            for value in (element.get("name"), element.get("type"), element.get("subtype"))
            if value and value.lower() not in {"none", "-1"}
        }
    )
    signal_descriptors = sorted(
        {
            value
            for element in root.findall("./road/signals/signal")
            for value in (element.get("name"), element.get("type"), element.get("subtype"))
            if value and value.lower() not in {"none", "-1"}
        }
    )
    longest_road = max((row["length_m"] for row in road_rows), default=0.0)
    max_choice = max((row["movement_choice_count"] for row in decisions), default=0)
    hypotheses = []
    if max_choice >= 2:
        hypotheses.extend(["A1_INTERSECTION_BRANCH", "A2_STRAIGHT_VS_TURN"])
    if max_choice >= 3:
        hypotheses.append("A3_FORK")
    if three_lane_sections:
        hypotheses.append("A4_LANE_CHANGE_LEFT_VS_RIGHT")
    if two_lane_sections:
        hypotheses.extend(["A5_LANE_CHANGE_VS_KEEP", "B2_SCOPED_LANE_PREFERENCE"])
    if longest_road >= 80.0:
        hypotheses.append("B1_SEGMENT_SPEED_BAND")
    if object_descriptors:
        hypotheses.extend(["B3_OBJECT_TRIGGERED_SPEED", "B4_OR_B5_OBJECT_RELATIVE"])
    return {
        "road_count": len(road_rows),
        "junction_count": len(root.findall("junction")),
        "decision_incoming_road_count": len(decisions),
        "movement_choice_count_max": max_choice,
        "three_lane_section_count": three_lane_sections,
        "two_lane_section_count": two_lane_sections,
        "longest_road_m": longest_road,
        "object_descriptors": object_descriptors[:100],
        "signal_descriptors": signal_descriptors[:100],
        "topology_hypotheses_not_qualified": hypotheses,
        "decisions": decisions,
        "roads": road_rows,
    }


def mine_shortlist_topology(
    shortlist_path: Path,
    scenes_csv: Path,
    scene_cache: Path,
) -> Dict[str, Any]:
    shortlist = load_json(shortlist_path)
    selected = _scene_ids(shortlist)
    catalog = _catalog_rows(scenes_csv, shortlist["revision"])
    scenes = []
    for scene_id in selected:
        row = catalog[scene_id]
        artifact_path = Path(scene_cache) / row["path"]
        inspection = inspect_downloaded_artifact(artifact_path, row)
        mined: Dict[str, Any] = {}
        route_aligned: Dict[str, Any] = {}
        errors = list(inspection["errors"])
        if inspection["readable_usdz"] and inspection["identity_matches_catalog"]:
            try:
                with zipfile.ZipFile(artifact_path) as archive:
                    if "map.xodr" not in archive.namelist():
                        errors.append("xodr_not_present")
                    else:
                        mined = mine_xodr(archive.read("map.xodr"))
                route_aligned = mine_route_aligned_artifact(artifact_path)
            except (OSError, zipfile.BadZipFile, KeyError, ElementTree.ParseError, ValueError) as error:
                errors.append("topology_mining_failed:%s" % type(error).__name__)
        scenes.append(
            {
                "scene_id_private": row["scene_id"],
                "artifact_uuid": row["uuid"],
                "artifact_path": row["path"],
                "full_artifact_verified": bool(
                    inspection["readable_usdz"]
                    and inspection["identity_matches_catalog"]
                    and inspection["required_members_present"]
                ),
                "topology": mined,
                "route_aligned_topology": route_aligned,
                "errors": errors,
                "qualification_state": "OFFLINE_TOPOLOGY_HYPOTHESIS_NOT_QUALIFIED",
            }
        )
    route_branch_hypothesis_count = sum(
        len(row["route_aligned_topology"].get("route_branch_hypotheses", []))
        for row in scenes
    )
    lane_choice_hypothesis_count = sum(
        len(row["route_aligned_topology"].get("lane_choice_hypotheses", []))
        for row in scenes
    )
    return {
        "schema_version": "l2d.realsim.topology_mining.v0.1",
        "evidence_scope": "offline_xodr_plus_optional_authoritative_recorded_route",
        "selected_scene_count": len(selected),
        "verified_artifact_count": sum(row["full_artifact_verified"] for row in scenes),
        "xodr_mined_count": sum(bool(row["topology"]) for row in scenes),
        "route_aligned_mined_count": sum(
            row["route_aligned_topology"].get("available") is True for row in scenes
        ),
        "route_branch_hypothesis_count": route_branch_hypothesis_count,
        "lane_choice_hypothesis_count": lane_choice_hypothesis_count,
        "route_aligned_opportunity_count": (
            route_branch_hypothesis_count + lane_choice_hypothesis_count
        ),
        "scene_error_count": sum(bool(row["errors"]) for row in scenes),
        "qualification_warning": (
            "Topology hypotheses do not establish visual observability, reconstruction support, "
            "dual-arm feasibility, oracle safety, or PCU eligibility."
        ),
        "scenes": scenes,
    }
