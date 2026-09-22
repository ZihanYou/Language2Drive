#!/usr/bin/env python3
"""Inventory and normalize the pre-v1 Language2Drive annotations.

The script is deliberately read-only with respect to all source repositories.  It
emits one deterministic JSON document to stdout (the default) or to ``--output``.
Only Python's standard library is required, so it can be run in simlingo-py311.

With ``--review-output-dir`` it also emits the deterministic eight-row C-RD
visibility/sign-arrow review template derived from the compiled suite.  This is
an *evidence extractor*, not a benchmark builder.  In particular, a
hand-authored trajectory graph is never promoted to lane-level reachability truth,
and an XODR parking object is kept road-relative until the v1 map pipeline turns it
into a world-space polygon.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SIMLINGO_FZH = Path("/root/autodl-tmp/simlingo_fzh")
DEFAULT_V1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATOR_ROOT = DEFAULT_V1_ROOT

SLOT_RE = re.compile(r"^[A-F]\d{3}$")
TOWN_FILE_RE = re.compile(r"recorded_routes_town(\d+)\.json$", re.IGNORECASE)
VERTEX_RE = re.compile(r"^V(\d+)_(.+)$", re.IGNORECASE)

# These are the frozen limits used by the live-CARLA/urban-edge spatial join.
# They establish an auditable geometry correspondence.  They do not establish
# that a road-name sign or its arrow is readable from the agent camera.
C_RD_JOIN_MAX_DISTANCE_M = 15.0
C_RD_JOIN_MAX_HEADING_ERROR_DEG = 20.0

FDQ_ABILITIES = {
    "overtaking": [
        "Accident",
        "AccidentTwoWays",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "HazardAtSideLaneTwoWays",
        "HazardAtSideLane",
        "ParkedObstacleTwoWays",
        "ParkedObstacle",
        "VehicleOpensDoorTwoWays",
    ],
    "merging": [
        "CrossingBicycleFlow",
        "EnterActorFlow",
        "HighwayExit",
        "InterurbanActorFlow",
        "HighwayCutIn",
        "InterurbanAdvancedActorFlow",
        "MergerIntoSlowTrafficV2",
        "MergerIntoSlowTraffic",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow",
        "ParkingExit",
        "SequentialLaneChange",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
    ],
    "emergency_brake": [
        "BlockedIntersection",
        "DynamicObjectCrossing",
        "HardBreakRoute",
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "ParkingCutIn",
        "PedestrianCrossing",
        "ParkingCrossingPedestrian",
        "StaticCutIn",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "ControlLoss",
    ],
    "give_way": ["InvadingTurn", "YieldToEmergencyVehicle"],
}

KNOWN_TOWN_LABEL_ISSUES = {
    "speed_limit_3": "probable truncation",
    "postal_": "probable truncation",
    "traffo": "probable truncation",
    "traffic_limit": "probable typo",
    "speeed_limit_90": "probable typo",
    "speed-limit_30": "inconsistent spelling",
    "unlabeled": "not a semantic target",
    "unnamed": "not a semantic target",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required source does not exist: {path}")
    return path


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file()) if path.is_dir() else 0


def xyz_key(value: dict[str, Any]) -> tuple[float, float, float]:
    return tuple(round(float(value.get(axis, 0.0)), 4) for axis in ("x", "y", "z"))


def town_from_file(path: Path) -> tuple[str, str]:
    match = TOWN_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer town from {path}")
    number = int(match.group(1))
    source_town = f"Town{number:02d}"
    canonical_town = "Town10HD" if number == 10 else source_town
    return source_town, canonical_town


def scan_town_landmarks(root: Path) -> dict[str, Any]:
    test_root = require(root / "data/test")
    files = sorted(test_root.glob("recorded_routes_town*.json"))
    files += sorted((test_root / "loacation").glob("recorded_routes_town*.json"))
    files.sort(key=lambda path: int(TOWN_FILE_RE.search(path.name).group(1)))
    if len(files) != 8:
        raise RuntimeError(f"expected 8 Town label files, found {len(files)}")

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    total_geometries = 0
    total_endpoints = 0
    for path in files:
        source_town, canonical_town = town_from_file(path)
        payload = load_json(path)
        if not isinstance(payload, list):
            raise TypeError(f"Town label source must be a list: {path}")
        word_counts = collections.Counter()
        starts: set[tuple[float, float, float]] = set()
        ends: set[tuple[float, float, float]] = set()
        geometries: set[tuple[Any, Any]] = set()
        labels_by_end: dict[tuple[float, float, float], list[str]] = collections.defaultdict(list)
        issue_words: dict[str, str] = {}
        for index, item in enumerate(payload):
            start = item.get("start") or {}
            end = item.get("end") or {}
            word = str(end.get("word") or "")
            start_key = xyz_key(start.get("location") or {})
            end_key = xyz_key(end.get("location") or {})
            starts.add(start_key)
            ends.add(end_key)
            geometries.add((start_key, end_key))
            labels_by_end[end_key].append(word)
            word_counts[word] += 1
            if word in KNOWN_TOWN_LABEL_ISSUES:
                issue_words[word] = KNOWN_TOWN_LABEL_ISSUES[word]
            records.append(
                {
                    "source_file": str(path),
                    "source_index": index,
                    "source_town": source_town,
                    "canonical_town": canonical_town,
                    "route_id": item.get("route_id"),
                    "timestamp": item.get("timestamp"),
                    "word": word,
                    "start": start,
                    "end": end,
                    "quality_flags": (
                        [f"label_issue:{issue_words[word]}"] if word in issue_words else []
                    ),
                }
            )
        multi_label_endpoints = [
            {"end_xyz": list(key), "words": words}
            for key, words in sorted(labels_by_end.items())
            if len(words) > 1
        ]
        total_geometries += len(geometries)
        total_endpoints += len(ends)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "source_town": source_town,
                "canonical_town": canonical_town,
                "entries": len(payload),
                "unique_words": len(word_counts),
                "unique_start_points": len(starts),
                "unique_end_points": len(ends),
                "unique_start_end_geometries": len(geometries),
                "repeated_word_counts": {
                    word: count for word, count in sorted(word_counts.items()) if count > 1
                },
                "multi_label_endpoints": multi_label_endpoints,
                "known_label_issues": issue_words,
            }
        )
    return {
        "summary": {
            "source_files": len(files),
            "entries": len(records),
            "unique_start_end_geometries_within_town": total_geometries,
            "unique_end_points_within_town": total_endpoints,
            "map_coverage": [source["canonical_town"] for source in sources],
            "location_directory_is_misspelled": str(test_root / "loacation"),
        },
        "sources": sources,
        "records": records,
    }


def split_parking_variants(end_name: str) -> list[str]:
    parts = end_name.split("_")
    if len(parts) == 2 and all(re.fullmatch(r"[A-Za-z0-9]+", part) for part in parts):
        return parts
    return [end_name]


def scan_custom_points(create_root: Path) -> dict[str, Any]:
    point_root = require(create_root / "carla_point_jsons")
    result: dict[str, Any] = {}
    for map_type in ("parking", "port", "urban"):
        path = require(point_root / f"{map_type}.json")
        payload = load_json(path)
        words = [str(item.get("word") or "") for item in payload]
        word_counts = collections.Counter(words)
        coordinates = {xyz_key(item.get("location") or {}) for item in payload}
        records = []
        slot_occurrences: list[tuple[str, int]] = []
        non_slot_components: collections.Counter[str] = collections.Counter()
        for index, item in enumerate(payload):
            word = str(item.get("word") or "")
            match = VERTEX_RE.match(word)
            vertex_id = int(match.group(1)) if match else None
            label = match.group(2) if match else word
            slot_tokens = [part for part in label.split("_") if SLOT_RE.fullmatch(part)]
            for token in slot_tokens:
                slot_occurrences.append((token, vertex_id if vertex_id is not None else -1))
            for part in label.split("_"):
                if part and not SLOT_RE.fullmatch(part) and not part.isdigit():
                    non_slot_components[part] += 1
            records.append(
                {
                    "source_index": index,
                    "vertex_id": vertex_id,
                    "word": word,
                    "label": label,
                    "slot_tokens": slot_tokens,
                    "location": item.get("location"),
                    "rotation": item.get("rotation"),
                    "timestamp": item.get("timestamp"),
                }
            )
        slot_vertices: dict[str, list[int]] = collections.defaultdict(list)
        for token, vertex in slot_occurrences:
            slot_vertices[token].append(vertex)
        duplicate_slot_bindings = {
            token: vertices
            for token, vertices in sorted(slot_vertices.items())
            if len(vertices) > 1
        }
        result[map_type] = {
            "source": {"path": str(path), "sha256": sha256_file(path)},
            "summary": {
                "entries": len(payload),
                "unique_words": len(word_counts),
                "unique_coordinates": len(coordinates),
                "duplicate_words": {
                    word: count for word, count in sorted(word_counts.items()) if count > 1
                },
                "slot_token_occurrences": len(slot_occurrences),
                "unique_slot_tokens": len(slot_vertices),
                "duplicate_slot_bindings": duplicate_slot_bindings,
                "non_slot_components": dict(sorted(non_slot_components.items())),
            },
            "records": records,
        }
    return result


def normalized_graph_path(map_type: str, item: dict[str, Any]) -> dict[str, Any]:
    vertices = item.get("vertex_indices_along_path") or item.get("vertices_along_path") or []
    waypoints = item.get("waypoints") or []
    null_indices = [index for index, waypoint in enumerate(waypoints) if waypoint is None]
    quality_flags = ["hand_authored_undirected_graph_not_lane_truth"]
    valid_targets: list[str] = []
    invalid_targets: list[str] = []
    if map_type == "port":
        quality_flags.append("placeholder_topology_matrix")
    elif map_type == "parking":
        variants = split_parking_variants(str(item.get("end_name") or ""))
        valid_targets = [variant for variant in variants if SLOT_RE.fullmatch(variant)]
        invalid_targets = [variant for variant in variants if not SLOT_RE.fullmatch(variant)]
        if invalid_targets:
            quality_flags.append("contains_pseudo_target")
    elif map_type == "urban":
        quality_flags.append("road_names_require_visual_confirmation")
    if null_indices:
        quality_flags.append("missing_waypoints")
    return {
        "path_id": item.get("path_id"),
        "start_vertex": item.get("start_vertex"),
        "start_name": item.get("start_name"),
        "end_vertex": item.get("end_vertex"),
        "end_name": item.get("end_name"),
        "end_vertex_type": item.get("end_vertex_type"),
        "instruction": item.get("instruction"),
        "reachable": item.get("reachable"),
        "graph_distance": item.get("graph_distance"),
        "sequence_vertex_edge": item.get("sequence_vertex_edge"),
        "vertices_along_path": vertices,
        "vertex_real_names": item.get("vertex_real_names"),
        "edge_tokens_along_path": item.get("edge_tokens_along_path"),
        "road_names_along_path": item.get("road_names_along_path"),
        "waypoints": waypoints,
        "null_waypoint_indices": null_indices,
        "valid_parking_target_tokens": valid_targets,
        "invalid_parking_target_tokens": invalid_targets,
        "quality_flags": quality_flags,
    }


def scan_route_graphs(create_root: Path) -> dict[str, Any]:
    graph_root = require(create_root / "trj_seq_jsons")
    names = {
        "parking": "parking_traj_graph_sequence_starts_150_151_158_159.json",
        "port": "port_traj_graph_sequence_all_start.json",
        "urban": "urban_traj_graph_sequence_manual.json",
    }
    result: dict[str, Any] = {}
    for map_type, filename in names.items():
        path = require(graph_root / filename)
        payload = load_json(path)
        raw_paths = payload.get("paths") or []
        paths = [normalized_graph_path(map_type, item) for item in raw_paths]
        by_start: dict[str, set[str]] = collections.defaultdict(set)
        for item in paths:
            start = str(item.get("start_vertex"))
            end = str(item.get("end_vertex"))
            by_start[start].add(end)
        raw_pair_candidates = sum(len(ends) * (len(ends) - 1) // 2 for ends in by_start.values())
        geometry_groups: dict[tuple[str, ...], list[Any]] = collections.defaultdict(list)
        for item in paths:
            geometry_groups[tuple(str(value) for value in item["vertices_along_path"])].append(
                item["path_id"]
            )
        duplicate_geometries = [
            {"vertices": list(vertices), "path_ids": ids}
            for vertices, ids in sorted(geometry_groups.items())
            if len(ids) > 1
        ]
        summary: dict[str, Any] = {
            "map_type": payload.get("map_type"),
            "directed": payload.get("directed"),
            "graph_mode": payload.get("graph_mode"),
            "num_vertices": payload.get("num_vertices"),
            "num_edges_declared": payload.get("num_edges_declared"),
            "declared_path_count": payload.get("path_count_total"),
            "actual_path_count": len(paths),
            "starts_included": payload.get("starts_included"),
            "reachable_paths": sum(item.get("reachable") is not False for item in paths),
            "paths_with_null_waypoints": sum(bool(item["null_waypoint_indices"]) for item in paths),
            "path_ids_with_null_waypoints": [
                item["path_id"] for item in paths if item["null_waypoint_indices"]
            ],
            "unique_path_geometries": len(geometry_groups),
            "duplicate_geometry_groups": duplicate_geometries,
            "raw_same_start_unordered_endpoint_pairs": raw_pair_candidates,
        }
        if map_type == "parking":
            categories: collections.Counter[str] = collections.Counter()
            unique_slots: set[str] = set()
            for item in paths:
                for variant in item["valid_parking_target_tokens"]:
                    categories["slot"] += 1
                    unique_slots.add(variant)
                for variant in item["invalid_parking_target_tokens"]:
                    category = "numeric" if variant.isdigit() else re.sub(r"\d+$", "", variant)
                    categories[category] += 1
            summary["expanded_target_variants_across_paths"] = sum(categories.values())
            summary["expanded_target_variant_categories"] = dict(sorted(categories.items()))
            summary["unique_valid_slot_tokens"] = len(unique_slots)
            summary["generated_xml_count_by_target_category"] = {
                category: count * 30 for category, count in sorted(categories.items())
            }
        if map_type == "urban":
            road_registry = payload.get("road_names") or {}
            noncanonical_keys = []
            for key in road_registry:
                try:
                    left, right = (int(value) for value in key.split("-", 1))
                except (TypeError, ValueError):
                    noncanonical_keys.append(key)
                    continue
                if left > right:
                    noncanonical_keys.append(key)
            null_road_name_paths = [
                {
                    "path_id": item["path_id"],
                    "null_indices": [
                        index
                        for index, value in enumerate(item.get("road_names_along_path") or [])
                        if value is None
                    ],
                }
                for item in paths
                if any(value is None for value in (item.get("road_names_along_path") or []))
            ]
            point_words = {
                str(item.get("word") or "").lower()
                for item in load_json(create_root / "carla_point_jsons/urban.json")
            }
            referenced_vertices = {
                str(vertex).lower() for item in paths for vertex in item["vertices_along_path"]
            }
            missing_vertices = sorted(referenced_vertices - point_words)
            summary["road_name_registry"] = road_registry
            summary["unique_road_names"] = sorted(
                set(road_registry.values())
            )
            summary["noncanonical_road_name_keys"] = noncanonical_keys
            summary["null_road_name_occurrences"] = sum(
                len(item["null_indices"]) for item in null_road_name_paths
            )
            summary["paths_with_null_road_names"] = null_road_name_paths
            summary["referenced_vertex_keys"] = len(referenced_vertices)
            summary["missing_vertex_keys_in_point_registry"] = missing_vertices
            summary["available_N_suffix_aliases"] = {
                vertex: f"{vertex}_n"
                for vertex in missing_vertices
                if f"{vertex}_n" in point_words
            }
        result[map_type] = {
            "source": {"path": str(path), "sha256": sha256_file(path)},
            "summary": summary,
            "paths": paths,
        }
    return result


def scan_intersections(create_root: Path, xodr_root: Path) -> dict[str, Any]:
    path = require(create_root / "american_city_intersection_benchmark.json")
    payload = load_json(path)
    episodes = payload.get("episodes") or []
    by_approach: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for episode in episodes:
        by_approach[str(episode.get("approach_id"))].append(episode)
    maneuver_sets = collections.Counter(
        "+".join(sorted(str(item.get("maneuver")) for item in group))
        for group in by_approach.values()
    )
    pair_types: collections.Counter[str] = collections.Counter()
    pair_candidates = []
    for approach_id, group in sorted(by_approach.items()):
        for left, right in itertools.combinations(sorted(group, key=lambda item: item["maneuver"]), 2):
            pair_type = "+".join(sorted((left["maneuver"], right["maneuver"])))
            pair_types[pair_type] += 1
            pair_candidates.append(
                {
                    "candidate_id": f"{approach_id}:{pair_type}",
                    "approach_id": approach_id,
                    "junction_id": left.get("junction_id"),
                    "junction_type": left.get("junction_type"),
                    "common_start": left.get("start"),
                    "arms": [
                        {
                            "episode_id": left.get("episode_id"),
                            "maneuver": left.get("maneuver"),
                            "exit_road_id": left.get("exit_road_id"),
                            "success_region": left.get("success_region"),
                            "instruction_en": left.get("instruction_en"),
                        },
                        {
                            "episode_id": right.get("episode_id"),
                            "maneuver": right.get("maneuver"),
                            "exit_road_id": right.get("exit_road_id"),
                            "success_region": right.get("success_region"),
                            "instruction_en": right.get("instruction_en"),
                        },
                    ],
                    "quality_flags": [
                        "road_reference_geometry_requires_lane_projection",
                        "requires_oracle_validation",
                    ],
                }
            )
    same_start = all(
        len(
            {
                tuple((item.get("start") or {}).get(key) for key in ("x", "y", "z", "yaw_deg"))
                for item in group
            }
        )
        == 1
        for group in by_approach.values()
    )
    distinct_pair_targets = all(
        pair["arms"][0]["exit_road_id"] != pair["arms"][1]["exit_road_id"]
        and pair["arms"][0]["success_region"]["center"]
        != pair["arms"][1]["success_region"]["center"]
        for pair in pair_candidates
    )
    canonical_xodr = require(xodr_root / "American-city0428.xodr")
    meta_xodr_value = (payload.get("meta") or {}).get("xodr_path")
    meta_xodr = Path(meta_xodr_value) if meta_xodr_value else None
    canonical_hash = sha256_file(canonical_xodr)
    meta_hash = sha256_file(meta_xodr) if meta_xodr and meta_xodr.is_file() else None
    return {
        "source": {"path": str(path), "sha256": sha256_file(path)},
        "meta": payload.get("meta"),
        "summary": {
            "episodes": len(episodes),
            "approaches": len(by_approach),
            "junctions": len({str(item.get("junction_id")) for item in episodes}),
            "maneuver_counts": counter_dict(item.get("maneuver") for item in episodes),
            "junction_type_counts": counter_dict(item.get("junction_type") for item in episodes),
            "maneuver_set_per_approach_counts": dict(sorted(maneuver_sets.items())),
            "unordered_pcu_pair_candidates": len(pair_candidates),
            "pcu_pair_type_counts": dict(sorted(pair_types.items())),
            "all_arms_in_approach_share_start": same_start,
            "all_pair_targets_are_distinct": distinct_pair_targets,
            "canonical_xodr_path": str(canonical_xodr),
            "canonical_xodr_sha256": canonical_hash,
            "metadata_xodr_exists": bool(meta_xodr and meta_xodr.is_file()),
            "metadata_xodr_sha256": meta_hash,
            "metadata_xodr_matches_canonical": meta_hash == canonical_hash,
        },
        "pcu_pair_candidates": pair_candidates,
        "episodes": episodes,
    }


def object_type(value: ET.Element) -> str:
    return value.get("type") or value.get("name") or "<none>"


def float_attr(element: ET.Element, name: str) -> float:
    return float(element.get(name, "0") or 0.0)


def scan_xodr(xodr_root: Path) -> dict[str, Any]:
    names = [
        "Town01",
        "Town02",
        "Town03",
        "Town04",
        "Town05",
        "Town06",
        "Town07",
        "Town10HD",
        "yangluogang1",
        "parkinglotfish-new3",
        "American-city0428",
        "hanghailou3",
        "hubeiwenli4",
        "road1",
    ]
    maps: dict[str, Any] = {}
    dedup_parking: dict[tuple[Any, ...], dict[str, Any]] = {}
    raw_parking_count = 0
    for name in names:
        path = require(xodr_root / f"{name}.xodr")
        root = ET.parse(path).getroot()
        roads = root.findall("road")
        junctions = root.findall("junction")
        objects = root.findall(".//object")
        signals = root.findall(".//signal")
        maps[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "roads": len(roads),
            "junctions": len(junctions),
            "objects": len(objects),
            "signals": len(signals),
            "object_type_counts": counter_dict(object_type(item) for item in objects),
            "signal_type_counts": counter_dict(
                f"{item.get('type', '')}:{item.get('subtype', '')}" for item in signals
            ),
        }
        if name != "parkinglotfish-new3":
            continue
        for road in roads:
            road_id = str(road.get("id"))
            for obj in road.findall("./objects/object"):
                if obj.get("type") != "parkingSpace":
                    continue
                raw_parking_count += 1
                values = {
                    "road_id": road_id,
                    "s": float_attr(obj, "s"),
                    "t": float_attr(obj, "t"),
                    "z_offset": float_attr(obj, "zOffset"),
                    "hdg": float_attr(obj, "hdg"),
                    "roll": float_attr(obj, "roll"),
                    "pitch": float_attr(obj, "pitch"),
                    "width": float_attr(obj, "width"),
                    "length": float_attr(obj, "length"),
                    "orientation": obj.get("orientation"),
                }
                key = (
                    road_id,
                    *(round(values[field], 9) for field in ("s", "t", "z_offset", "hdg", "width", "length")),
                    values["orientation"],
                )
                if key not in dedup_parking:
                    dedup_parking[key] = {
                        "candidate_index": len(dedup_parking),
                        **values,
                        "source_object_ids": [],
                        "quality_flags": [
                            "anonymous_xodr_object",
                            "road_relative_rectangle_not_world_polygon",
                            "slot_number_unbound",
                        ],
                    }
                dedup_parking[key]["source_object_ids"].append(obj.get("id"))
    candidates = list(dedup_parking.values())
    duplicate_groups = [
        {
            "candidate_index": candidate["candidate_index"],
            "source_object_ids": candidate["source_object_ids"],
        }
        for candidate in candidates
        if len(candidate["source_object_ids"]) > 1
    ]
    return {
        "maps": maps,
        "parking_spaces": {
            "source": maps["parkinglotfish-new3"]["path"],
            "raw_parking_space_objects": raw_parking_count,
            "deduplicated_road_relative_rectangles": len(candidates),
            "duplicate_object_count": raw_parking_count - len(candidates),
            "duplicate_groups": duplicate_groups,
            "candidates": candidates,
        },
    }


def route_positions(route: ET.Element) -> list[dict[str, float]]:
    result = []
    for node in route.findall("./waypoints/position"):
        result.append({axis: float(node.get(axis, "0")) for axis in ("x", "y", "z")})
    return result


def scan_xml_corpora(root: Path, create_root: Path) -> dict[str, Any]:
    generated_root = require(root / "data/xmls/generated_route_xmls")
    per_town = {
        directory.name: count_files(directory, "*.xml")
        for directory in sorted(generated_root.iterdir())
        if directory.is_dir()
    }
    enriched_root = require(root / "data/xmls/town01")
    enriched_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(enriched_root.glob("*.xml")):
        route = ET.parse(path).getroot().find(".//route")
        if route is None:
            continue
        meta = route.find("simlingo_metadata")
        attrs = dict(meta.attrib) if meta is not None else {}
        waypoints = route_positions(route)
        key = (
            route.get("town"),
            attrs.get("source_route_id"),
            attrs.get("task_type"),
            attrs.get("word"),
            tuple((item["x"], item["y"], item["z"]) for item in waypoints),
        )
        if key not in enriched_groups:
            enriched_groups[key] = {
                "town": route.get("town"),
                "source_route_id": attrs.get("source_route_id"),
                "task_type": attrs.get("task_type"),
                "word": attrs.get("word"),
                "waypoints": waypoints,
                "start_speed_kmh": set(),
                "end_speed_kmh": set(),
                "weathers": set(),
                "file_count": 0,
                "representative_xml": str(path),
            }
        group = enriched_groups[key]
        group["start_speed_kmh"].add(attrs.get("start_speed_kmh"))
        group["end_speed_kmh"].add(attrs.get("end_speed_kmh"))
        group["weathers"].add(attrs.get("weather_name"))
        group["file_count"] += 1
    enriched_records = []
    for group in enriched_groups.values():
        for field in ("start_speed_kmh", "end_speed_kmh", "weathers"):
            group[field] = sorted(value for value in group[field] if value is not None)
        enriched_records.append(group)
    enriched_records.sort(key=lambda item: (str(item["source_route_id"]), str(item["task_type"])))
    custom_counts = {
        map_type: count_files(require(create_root / f"layerc_xmls/{map_type}"), "*.xml")
        for map_type in ("parking", "port", "urban")
    }
    return {
        "town_weather_expansion": {
            "path": str(generated_root),
            "xml_files": sum(per_town.values()),
            "per_source_directory": per_town,
            "canonical_source_records": 580,
            "note": "ten weather files per recorded_routes entry; not independent annotations",
        },
        "town01_enriched_tasks": {
            "path": str(enriched_root),
            "xml_files": count_files(enriched_root, "*.xml"),
            "canonical_task_geometries": len(enriched_records),
            "task_file_counts": counter_dict(
                item["task_type"]
                for item in enriched_records
                for _ in range(int(item["file_count"]))
            ),
            "canonical_task_counts": counter_dict(item["task_type"] for item in enriched_records),
            "records": enriched_records,
        },
        "custom_layer_c_weather_template_expansion": {
            "path": str(create_root / "layerc_xmls"),
            "xml_files": custom_counts,
            "total_xml_files": sum(custom_counts.values()),
            "note": "weather/template/slot-token variants; route-graph paths are the canonical source",
        },
    }


def classify_parking_target(target: str) -> str:
    if SLOT_RE.fullmatch(target):
        return "slot"
    if target.isdigit():
        return "numeric"
    return re.sub(r"\d+$", "", target) or "<empty>"


def scan_v0_layer_c(evaluator_root: Path) -> dict[str, Any]:
    data_root = require(evaluator_root / "assets/legacy_input/data_l2d")
    result: dict[str, Any] = {}
    for subset in ("town", "parking_lot"):
        subset_root = require(data_root / "layer_C" / subset)
        records = []
        targets: list[str] = []
        for path in sorted(subset_root.glob("*.xml")):
            route = ET.parse(path).getroot().find(".//route")
            if route is None:
                continue
            meta_node = route.find("simlingo_metadata")
            meta = dict(meta_node.attrib) if meta_node is not None else {}
            instruction_node = route.find("instruction")
            instruction = (
                (instruction_node.text or "").strip()
                if instruction_node is not None
                else meta.get("instruction", "")
            )
            target = meta.get("end_name_variant") or meta.get("end_name") or meta.get("word") or ""
            targets.append(target)
            records.append(
                {
                    "source_xml": str(path),
                    "route_id": route.get("id"),
                    "town": route.get("town"),
                    "metadata": meta,
                    "instruction": instruction,
                    "waypoints": route_positions(route),
                    "target": target,
                    "quality_flags": (
                        ["pseudo_parking_target"]
                        if subset == "parking_lot" and classify_parking_target(target) != "slot"
                        else []
                    ),
                }
            )
        result[subset] = {
            "path": str(subset_root),
            "xml_files": len(records),
            "town_counts": counter_dict(item["town"] for item in records),
            "unique_source_route_or_path_ids": len(
                {
                    item["metadata"].get("source_route_id") or item["metadata"].get("path_id")
                    for item in records
                }
            ),
            "unique_targets": len(set(targets)),
            "target_category_counts": (
                counter_dict(classify_parking_target(target) for target in targets)
                if subset == "parking_lot"
                else None
            ),
            "records_without_explicit_instruction_text": sum(
                not item["instruction"] for item in records
            ),
            "records": records,
        }
    merged = require(data_root / "layer_C_merged.xml")
    manifest = require(data_root / "layer_C_manifest.json")
    manifest_payload = load_json(manifest)
    result["stale_merged_artifacts"] = {
        "merged_xml": str(merged),
        "merged_xml_route_count": len(ET.parse(merged).getroot().findall(".//route")),
        "manifest": str(manifest),
        "manifest_declared_total_routes": manifest_payload.get("total_routes"),
        "manifest_actual_route_entries": len(manifest_payload.get("routes") or []),
    }
    layer_b = require(data_root / "layer_B_merged.xml")
    layer_b_routes = ET.parse(layer_b).getroot().findall(".//route")
    result["invalid_layer_b_artifact"] = {
        "path": str(layer_b),
        "route_count": len(layer_b_routes),
        "town_counts": counter_dict(item.get("town") for item in layer_b_routes),
        "routes_with_instruction": sum(
            item.find("instruction") is not None or item.find("simlingo_metadata") is not None
            for item in layer_b_routes
        ),
        "reason": "ordinary Bench2Drive routes without an instruction-conditioned opposite arm",
    }
    return result


def scan_fdq(evaluator_root: Path) -> dict[str, Any]:
    split_root = require(
        evaluator_root / "language2drive/leaderboard/data/bench2drive_split"
    )
    records = []
    for path in sorted(split_root.glob("*.xml")):
        xml_root = ET.parse(path).getroot()
        for route in xml_root.findall(".//route"):
            scenario = route.find("./scenarios/scenario")
            trigger = scenario.find("trigger_point") if scenario is not None else None
            records.append(
                {
                    "source_xml": str(path),
                    "route_id": route.get("id"),
                    "town": route.get("town"),
                    "scenario_type": scenario.get("type") if scenario is not None else None,
                    "scenario_name": scenario.get("name") if scenario is not None else None,
                    "trigger_point": dict(trigger.attrib) if trigger is not None else None,
                    "waypoint_count": len(route.findall("./waypoints/position")),
                }
            )
    scenario_types = {str(item["scenario_type"]) for item in records}
    covered = set(itertools.chain.from_iterable(FDQ_ABILITIES.values()))
    return {
        "path": str(split_root),
        "xml_files": count_files(split_root, "*.xml"),
        "routes": len(records),
        "town_counts": counter_dict(item["town"] for item in records),
        "scenario_types": len(scenario_types),
        "scenario_type_counts": counter_dict(item["scenario_type"] for item in records),
        "ability_route_counts": {
            ability: sum(item["scenario_type"] in types for item in records)
            for ability, types in FDQ_ABILITIES.items()
        },
        "unmapped_scenario_types": sorted(scenario_types - covered),
        "unmapped_routes": sum(item["scenario_type"] not in covered for item in records),
        "records": records,
    }


def scan_rollouts(root: Path) -> dict[str, Any]:
    dataset_root = root / "data/trajectory_data/simlingo_v2_parking_no_inter_wp_200"
    episode_root = dataset_root / "data/parking"
    if not episode_root.is_dir():
        return {"path": str(dataset_root), "exists": False}
    episodes = []
    for directory in sorted(item for item in episode_root.iterdir() if item.is_dir()):
        abort_path = directory / "abort_reason.txt"
        abort_reason = abort_path.read_text(encoding="utf-8", errors="replace").strip() if abort_path.is_file() else None
        result_path = directory / "results.json.gz"
        status = None
        result_error = None
        if result_path.is_file():
            try:
                with gzip.open(result_path, "rt", encoding="utf-8") as handle:
                    status = json.load(handle).get("status")
            except (OSError, ValueError) as error:
                result_error = str(error)
        episodes.append(
            {
                "directory": str(directory),
                "abort_reason": abort_reason,
                "result_status": status,
                "result_error": result_error,
                "has_results": result_path.is_file(),
                "has_records": (directory / "records.json.gz").is_file(),
                "has_measurements": (directory / "measurements").is_dir(),
                "has_rgb_front": (directory / "camera/rgb_front").is_dir(),
            }
        )
    return {
        "path": str(dataset_root),
        "exists": True,
        "episode_directories": len(episodes),
        "result_status_counts": counter_dict(
            item["result_status"] if item["has_results"] else "<missing>" for item in episodes
        ),
        "abort_reason_counts": counter_dict(item["abort_reason"] or "<none>" for item in episodes),
        "non_aborted_episodes": sum(item["abort_reason"] is None for item in episodes),
        "non_aborted_perfect_results": sum(
            item["abort_reason"] is None and item["result_status"] == "Perfect"
            for item in episodes
        ),
        "annotations_json_exists": (dataset_root / "annotations.json").is_file(),
        "episodes": episodes,
    }


def scan_review_aids(root: Path, create_root: Path) -> dict[str, Any]:
    town_png_root = root / "data/test/loacation"
    plot_root = create_root / "plots"
    return {
        "town_rgb_examples": {
            "path": str(town_png_root),
            "png_files": count_files(town_png_root, "*.png"),
        },
        "custom_graph_node_plots": {
            "path": str(plot_root),
            "png_files": count_files(plot_root, "*_graph_nodes.png"),
        },
        "custom_route_spotchecks": {
            "path": str(plot_root / "spotcheck"),
            "png_files": count_files(plot_root / "spotcheck", "*.png"),
        },
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_road_name(value: str) -> str:
    """Normalize spelling variants for *name evidence*, never for map binding."""

    words = re.findall(r"[a-z0-9]+", str(value).lower())
    abbreviations = {
        "avenue": "ave",
        "street": "st",
        "boulevard": "blvd",
        "road": "rd",
    }
    return " ".join(abbreviations.get(word, word) for word in words)


def urban_replacement_candidates(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Return eight evidence-backed *candidate* C-VIA pairs.

    They are deliberately not marked accepted: the lane projection, disjoint
    terminal regions and three-seed oracle still have to pass the normal suite
    builder.  The first six share one source/topology group (dev), and the last
    two share another (test), so the proposal does not split either group.
    """

    paths = {
        int(item["path_id"]): item
        for item in catalog["custom_route_graphs"]["urban"]["paths"]
    }
    point_records = catalog["custom_points"]["urban"]["records"]
    point_lookup = {str(item["word"]).lower(): item for item in point_records}
    # Six unordered endpoint pairs from the v0_2 group, then two from v14_1.
    # Path 9 is an authored paraphrase of path 8 and is retained as such rather
    # than miscounted as another outcome.
    pair_specs = [
        ("C-VIA-R01", "C-001", "dev", 2, 1),
        ("C-VIA-R02", "C-002", "dev", 2, 3),
        ("C-VIA-R03", "C-003", "dev", 2, 4),
        ("C-VIA-R04", "C-004", "dev", 1, 3),
        ("C-VIA-R05", "C-005", "dev", 1, 4),
        ("C-VIA-R06", "C-006", "dev", 3, 4),
        ("C-VIA-R07", "C-007", "test", 8, 7),
        ("C-VIA-R08", "C-008", "test", 7, 6),
    ]

    def endpoint(path: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        waypoints = path.get("waypoints") or []
        if waypoints and isinstance(waypoints[-1], dict):
            return waypoints[-1], None
        key = str(path.get("end_vertex") or "").lower()
        record = point_lookup.get(key)
        alias = None
        if record is None:
            alias = f"{key}_n"
            record = point_lookup.get(alias)
        return ((record or {}).get("location"), alias if record else None)

    output = []
    for proposal_id, replaces, split, path_a_id, path_b_id in pair_specs:
        path_a, path_b = paths[path_a_id], paths[path_b_id]
        endpoint_a, alias_a = endpoint(path_a)
        endpoint_b, alias_b = endpoint(path_b)
        output.append(
            {
                "proposal_id": proposal_id,
                "replaces_pcu": replaces,
                "split": split,
                "town": "American-city0428",
                "category": "C-VIA_terminal_stop",
                "same_start_vertex": path_a["start_vertex"],
                "arm_a": {
                    "source_path_id": path_a_id,
                    "end_vertex": path_a["end_vertex"],
                    "instruction": path_a["instruction"],
                    "endpoint": endpoint_a,
                    "repaired_endpoint_alias": alias_a,
                    "authored_paraphrase_path_ids": [9] if path_a_id == 8 else [],
                },
                "arm_b": {
                    "source_path_id": path_b_id,
                    "end_vertex": path_b["end_vertex"],
                    "instruction": path_b["instruction"],
                    "endpoint": endpoint_b,
                    "repaired_endpoint_alias": alias_b,
                    "authored_paraphrase_path_ids": [9] if path_b_id == 8 else [],
                },
                "source_evidence": "authored instruction, vertex sequence and endpoint pose",
                "human_label_rows_required_for_engineering_pre_release": 0,
                "contract_requirement": (
                    "derive disjoint lane-bound terminal polygons and require low-speed dwell; "
                    "do not use historical enter_gate alone because several paths are nested"
                ),
                "automated_acceptance_required": [
                    "repair declared _N endpoint aliases",
                    "project every waypoint to the directed lane graph",
                    "validate legal path and route-length ratio",
                    "validate terminal polygons are disjoint",
                    "pass both arms for CARLA/TM seeds 1,2,3",
                ],
                "status": "candidate_not_compiled_or_oracle_witnessed",
            }
        )
    return output


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_slot_tokens(value: Any) -> set[str]:
    """Collect slots used by contracts/targets, excluding incidental prose."""

    found: set[str] = set()
    if isinstance(value, dict):
        if value.get("type") == "parking_slot" and SLOT_RE.fullmatch(str(value.get("slot_number") or "")):
            found.add(str(value["slot_number"]))
        for child in value.values():
            found.update(semantic_slot_tokens(child))
    elif isinstance(value, list):
        for child in value:
            found.update(semantic_slot_tokens(child))
    elif isinstance(value, str):
        # D strict-order and memory programs use evaluator-only stage gates
        # instead of a parking-slot terminal object.  They are still semantic
        # references to numbered physical XODR spaces and must be included in
        # the release inventory alongside exact-parking and occupancy targets.
        found.update(re.findall(r"(?:occupied|parking-stage):([A-F]\d{3})", value))
    return found


def scan_release_core(v1_root: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    """Audit how the generated 128-PCU snapshot uses the extracted evidence."""

    suite_path = v1_root / "build/suite/sidecar.private.json"
    source_pcus_path = v1_root / "sources/core_pcus.json"
    review_root = v1_root / "assets/review"
    if not suite_path.is_file():
        return {
            "status": "suite_snapshot_missing",
            "suite_path": str(suite_path),
            "replacement_candidates": urban_replacement_candidates(catalog),
        }

    suite = load_json(suite_path)
    pcus = suite.get("pcus") or []
    source_payload = load_json(source_pcus_path) if source_pcus_path.is_file() else None
    source_pcus = (
        source_payload.get("pcus") if isinstance(source_payload, dict) else source_payload
    ) if source_payload is not None else None
    c_road_pcus = []
    # Keep this split assignment separate so a malformed non-list cannot be
    # mistaken for usable benchmark evidence.
    if isinstance(pcus, list):
        c_road_pcus = [
            item for item in pcus
            if item.get("layer") == "C" and item.get("category") == "road_name"
        ]

    urban_names = catalog["custom_route_graphs"]["urban"]["summary"]["unique_road_names"]
    urban_name_keys = {canonical_road_name(value) for value in urban_names}

    capture_manifest_path = v1_root / "generated/review/rgb/capture_manifest.json"
    capture_manifest = (
        load_json(capture_manifest_path) if capture_manifest_path.is_file() else {}
    )
    capture_cases = {
        str(item.get("pcu_id")): item
        for item in (capture_manifest.get("cases") or [])
        if isinstance(item, dict)
    }

    def capture_integrity(case: dict[str, Any]) -> dict[str, Any]:
        checked = []
        for item in [*(case.get("camera_images") or []), {
            "relative_path": case.get("contact_sheet"),
            "sha256": case.get("contact_sheet_sha256"),
            "label": "contact_sheet",
        }]:
            relative = str(item.get("relative_path") or "")
            path = v1_root / relative if relative else Path("/__missing__")
            expected = str(item.get("sha256") or "")
            actual = sha256_file(path) if path.is_file() else None
            checked.append(
                {
                    "label": item.get("label"),
                    "relative_path": relative,
                    "expected_sha256": expected or None,
                    "actual_sha256": actual,
                    "valid": bool(actual and expected and actual == expected),
                }
            )
        return {
            "files": checked,
            "all_files_present_and_hash_valid": bool(checked) and all(
                item["valid"] for item in checked
            ),
        }

    mapping_rows = []
    for pcu in c_road_pcus:
        arms = pcu.get("arms") or []
        fixed_mapping = (pcu.get("metadata") or {}).get("fixed_semantic_mapping") or {}
        capture_case = capture_cases.get(str(pcu.get("id"))) or {}
        normalized_arms = []
        for arm in arms:
            target = arm.get("target") or {}
            semantic_name = str(target.get("semantic_name") or "")
            semantic_label = str(target.get("semantic_label") or semantic_name)
            join = fixed_mapping.get(semantic_label)
            if not isinstance(join, dict):
                # Fall back to the unique entry that names this exact gate.  This
                # keeps older compiled snapshots auditable without guessing by
                # string alone.
                matches = [
                    value for value in fixed_mapping.values()
                    if isinstance(value, dict) and value.get("gate") == target.get("id")
                ]
                join = matches[0] if len(matches) == 1 else {}
            try:
                distance_m = float(join.get("distance_m"))
                heading_error_deg = float(join.get("heading_error_deg"))
            except (TypeError, ValueError):
                distance_m = None
                heading_error_deg = None
            automatic_join_valid = bool(
                join
                and join.get("gate") == target.get("id")
                and join.get("edge_token")
                and distance_m is not None
                and distance_m <= C_RD_JOIN_MAX_DISTANCE_M
                and heading_error_deg is not None
                and heading_error_deg <= C_RD_JOIN_MAX_HEADING_ERROR_DEG
            )
            normalized_arms.append(
                {
                    "arm_id": arm.get("id"),
                    "semantic_name": semantic_name,
                    "semantic_label": semantic_label,
                    "name_occurs_in_urban_registry": canonical_road_name(semantic_name) in urban_name_keys,
                    "target_gate": target.get("id"),
                    "exit_road_id": target.get("exit_road_id"),
                    "maneuver": target.get("maneuver"),
                    "urban_edge_token": join.get("edge_token"),
                    "spatial_distance_m": distance_m,
                    "heading_error_deg": heading_error_deg,
                    "automatic_geometry_join_valid": automatic_join_valid,
                }
            )
        capture_check = capture_integrity(capture_case) if capture_case else {
            "files": [], "all_files_present_and_hash_valid": False
        }
        capture_matches = bool(
            capture_case
            and set(capture_case.get("gate_ids") or [])
            == {str(item.get("target_gate")) for item in normalized_arms}
            and set(canonical_road_name(value) for value in (capture_case.get("names") or []))
            == set(canonical_road_name(item.get("semantic_label") or "") for item in normalized_arms)
        )
        mapping_rows.append(
            {
                "review_id": f"C-RD-{pcu.get('id')}",
                "pcu_id": pcu.get("id"),
                "split": pcu.get("split"),
                "town": pcu.get("town"),
                "arms": normalized_arms,
                "automatic_geometry_join_valid": bool(normalized_arms) and all(
                    item["automatic_geometry_join_valid"] for item in normalized_arms
                ),
                "geometry_join_method": (
                    "live CARLA junction outgoing-lane geometry joined to the existing urban "
                    "named edge by spatial distance and directed heading"
                ),
                "join_thresholds": {
                    "maximum_distance_m": C_RD_JOIN_MAX_DISTANCE_M,
                    "maximum_heading_error_deg": C_RD_JOIN_MAX_HEADING_ERROR_DEG,
                },
                "geometry_evidence_scope": (
                    "supports the fixed-layout name-to-outgoing-lane mapping; does not prove "
                    "camera readability, a visible sign, or sign-arrow meaning"
                ),
                "rgb_contact_sheet": capture_case.get("contact_sheet"),
                "rgb_contact_sheet_sha256": capture_case.get("contact_sheet_sha256"),
                "rgb_capture_matches_current_suite": capture_matches,
                "rgb_capture_integrity": capture_check,
                "topdown_composite": f"generated/visualizations/{pcu.get('id')}/composite.png",
                "blocking": False,
                "blocking_scope": "paper_grade_human_visibility_witness_only",
                "minimum_review_action": (
                    "confirm or reject whether the RGB views show a readable road-name/sign-arrow "
                    "cue for both mapped exits; no free-form relabeling is requested"
                ),
            }
        )

    c_pcus = [item for item in pcus if isinstance(item, dict) and item.get("layer") == "C"]
    serialized_c = json.dumps(c_pcus, ensure_ascii=False, sort_keys=True)
    serialized_core = json.dumps(pcus, ensure_ascii=False, sort_keys=True)
    port_ids = sorted(
        {
            str((arm.get("target") or {}).get("id"))
            for pcu in c_pcus if pcu.get("category") == "port_target"
            for arm in (pcu.get("arms") or [])
        }
    )
    selected_slots: set[str] = set()
    selected_regions: set[str] = set()
    for pcu in c_pcus:
        for arm in pcu.get("arms") or []:
            target = arm.get("target") or {}
            target_id = str(target.get("id") or "")
            match = re.search(r"(?:^|_)([A-F]\d{3})(?:_|$)", target_id)
            if match:
                selected_slots.add(match.group(1))
            if target.get("type") == "region":
                selected_regions.add(str(target.get("id")))

    parking_tokens_by_layer: dict[str, list[str]] = {}
    parking_pcus_by_layer: dict[str, list[dict[str, Any]]] = {}
    all_core_slot_tokens: set[str] = set()
    for layer in ("A", "B", "C", "D"):
        rows = []
        layer_tokens: set[str] = set()
        for pcu in pcus:
            if not isinstance(pcu, dict) or pcu.get("layer") != layer:
                continue
            tokens = sorted(semantic_slot_tokens(pcu))
            if tokens:
                rows.append({"pcu_id": pcu.get("id"), "category": pcu.get("category"), "slot_tokens": tokens})
                layer_tokens.update(tokens)
        parking_tokens_by_layer[layer] = sorted(layer_tokens)
        parking_pcus_by_layer[layer] = rows
        all_core_slot_tokens.update(layer_tokens)

    review_tables: dict[str, Any] = {}
    review_table_rows: dict[str, list[dict[str, str]]] = {}
    for filename in ("asset_review.csv", "slot_registry.csv", "sign_routes.csv"):
        path = review_root / filename
        rows = read_csv_rows(path)
        review_table_rows[filename] = rows
        review_tables[filename] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "rows": len(rows),
            "status_counts": counter_dict(row.get("status", "") for row in rows),
        }

    fabricated_or_unregistered = sorted(
        {
            arm["semantic_name"]
            for row in mapping_rows for arm in row["arms"]
            if not arm["name_occurs_in_urban_registry"]
        }
    )
    selected_slot_rows = [
        row for row in review_table_rows.get("slot_registry.csv", [])
        if row.get("slot_number") in all_core_slot_tokens
    ]
    fallback_slot_tokens = sorted(
        row.get("slot_number", "") for row in selected_slot_rows
        if "fallback" in str(row.get("notes") or "").lower()
        or "unbound" in str(row.get("notes") or "").lower()
    )
    missing_slot_tokens = sorted(
        all_core_slot_tokens - {row.get("slot_number", "") for row in selected_slot_rows}
    )
    bound_slot_tokens = sorted(
        row.get("slot_number", "") for row in selected_slot_rows
        if "bound to " in str(row.get("notes") or "").lower()
    )
    selected_physical_ids = [
        row.get("physical_space_id", "") for row in selected_slot_rows
        if row.get("physical_space_id")
    ]
    duplicate_selected_physical_ids = sorted(
        physical_id for physical_id, count in collections.Counter(selected_physical_ids).items()
        if count > 1
    )
    valid_join_arms = sum(
        arm["automatic_geometry_join_valid"]
        for row in mapping_rows for arm in row["arms"]
    )
    valid_capture_cases = sum(
        row["rgb_capture_matches_current_suite"]
        and row["rgb_capture_integrity"]["all_files_present_and_hash_valid"]
        for row in mapping_rows
    )
    return {
        "status": "audited_generated_snapshot",
        "suite_snapshot": {
            "path": str(suite_path),
            "sha256": sha256_file(suite_path),
            "pcus": len(pcus),
            "routes": len(suite.get("routes") or []),
        },
        "canonical_source_snapshot": {
            "path": str(source_pcus_path),
            "sha256": sha256_file(source_pcus_path) if source_pcus_path.is_file() else None,
            "pcus": len(source_pcus) if isinstance(source_pcus, list) else None,
            "canonical_pcus_sha256": canonical_json_sha256(source_pcus) if isinstance(source_pcus, list) else None,
        },
        "source_compiled_consistency": {
            "pcus_equal": isinstance(source_pcus, list) and source_pcus == pcus,
            "compiled_canonical_pcus_sha256": canonical_json_sha256(pcus),
        },
        "review_status_semantics": {
            "confirmed_existing_annotation": (
                "machine-migrated source row whose pose projected within the configured lane threshold; "
                "it is not an independent human witness"
            ),
            "migrated_existing_annotation": "provenance label, not a human reviewer identity",
        },
        "review_tables": review_tables,
        "current_core_usage": {
            "c_road_name_pcus": len(c_road_pcus),
            "c_road_name_arms": sum(len(row["arms"]) for row in mapping_rows),
            "c_port_target_ids": port_ids,
            "c_selected_slot_tokens": sorted(selected_slots),
            "c_selected_region_ids": sorted(selected_regions),
            "parking_slot_tokens_by_layer": parking_tokens_by_layer,
            "parking_slot_pcus_by_layer": parking_pcus_by_layer,
        },
        "c_rd_mapping_audit": {
            "source_urban_road_names": urban_names,
            "current_names_not_in_source_registry": fabricated_or_unregistered,
            "pair_rows": mapping_rows,
            "automatic_geometry_join_valid_arms": valid_join_arms,
            "expected_arms": sum(len(row["arms"]) for row in mapping_rows),
            "rgb_capture_cases_with_valid_hashes_and_current_bindings": valid_capture_cases,
            "expected_rgb_capture_cases": len(mapping_rows),
            "pair_rows_requiring_visibility_sign_arrow_witness_for_paper_release": len(mapping_rows),
            "engineering_core_blocking_rows": 0,
            "decision": (
                "fixed_layout_mapping_has_automatic_geometry_evidence; "
                "eight_paired_visibility_sign_arrow_decisions_remain_for_paper_release"
            ),
        },
        "c_rd_rgb_capture": {
            "manifest": str(capture_manifest_path),
            "manifest_sha256": (
                sha256_file(capture_manifest_path) if capture_manifest_path.is_file() else None
            ),
            "declared_case_count": capture_manifest.get("case_count"),
            "actual_case_count": len(capture_cases),
            "camera_image_count": sum(
                len(case.get("camera_images") or []) for case in capture_cases.values()
            ),
            "contact_sheet_count": sum(bool(case.get("contact_sheet")) for case in capture_cases.values()),
        },
        "port_v13_v14": {
            "port:v13_reference_count_in_core_C": serialized_c.count("port:v13"),
            "port:v14_reference_count_in_core_C": serialized_c.count("port:v14"),
            "port:v13_reference_count_in_all_core_layers": serialized_core.count("port:v13"),
            "port:v14_reference_count_in_all_core_layers": serialized_core.count("port:v14"),
            "status": "non_blocking_excluded_from_core",
            "reason": "neither excluded outlier is referenced by any retained core layer",
        },
        "parking_semantics": {
            "status": "core_slots_one_to_one_bound_to_xodr_not_independent_human_confirmation",
            "core_blocking_human_rows": 0,
            "reason": (
                "the selected engineering-core slot/region identities are explicitly present in the existing "
                "manual labels and selected exact slots have one-to-one physical XODR bindings; "
                "footprint/heading/dwell and dynamic oracle checks remain automated acceptance gates"
            ),
            "selected_slot_registry_rows": len(selected_slot_rows),
            "all_core_selected_slot_tokens": sorted(all_core_slot_tokens),
            "all_core_selected_slot_tokens_bound_to_xodr": bound_slot_tokens,
            "selected_physical_space_ids": sorted(selected_physical_ids),
            "selected_physical_space_ids_are_unique": (
                len(selected_physical_ids) == len(set(selected_physical_ids))
                and len(selected_physical_ids) == len(all_core_slot_tokens)
            ),
            "duplicate_selected_physical_space_ids": duplicate_selected_physical_ids,
            "selected_slot_tokens_missing_from_registry": missing_slot_tokens,
            "selected_slot_tokens_using_unbound_pose_fallback": fallback_slot_tokens,
            "engineering_acceptance_pending": bool(
                missing_slot_tokens or fallback_slot_tokens or duplicate_selected_physical_ids
                or len(selected_physical_ids) != len(all_core_slot_tokens)
            ),
            "required_nonhuman_action": (
                "bind selected poses to world-space XODR parking polygons where possible, then validate "
                "footprint, heading and dwell with the privileged oracle; reject any unresolved candidate"
            ),
            "paper_release_requires_independent_visual_witness": True,
        },
        "replacement_candidates": urban_replacement_candidates(catalog),
        "release_decision": {
            "current_C_RD": (
                "engineering_mapping_supported_by_live_CARLA_spatial_heading_join; "
                "visibility_and_sign_arrow_witness_pending"
            ),
            "engineering_pre_release_core_blocking_human_rows": 0,
            "paper_grade_visibility_sign_arrow_rows": len(mapping_rows),
            "contingency_if_a_row_is_rejected": (
                "reject that PCU and remine from existing evidence; the historical C-VIA table "
                "is retained only as non-core candidate material"
            ),
            "paper_grade_human_witness": "eight paired confirm/reject decisions still required",
            "do_not_claim": ["human_confirmed semantics", "unseen-map generalization", "FCS-2x2"],
        },
    }


def build_catalog(
    simlingo_fzh: Path,
    evaluator_root: Path,
    v1_root: Path = DEFAULT_V1_ROOT,
) -> dict[str, Any]:
    simlingo_fzh = require(simlingo_fzh.resolve())
    evaluator_root = require(evaluator_root.resolve())
    v1_root = require(v1_root.resolve())
    create_root = require(simlingo_fzh / "tools/create_data")
    xodr_root = require(simlingo_fzh / "software/road_mesh/OpenDrive")
    catalog = {
        "schema_version": 1,
        "generator": "tools/extract_existing_labels.py",
        "roots": {
            "simlingo_fzh": str(simlingo_fzh),
            "legacy_evaluator_root": str(evaluator_root),
            "language2drive_v1": str(v1_root),
            "create_data": str(create_root),
            "xodr": str(xodr_root),
        },
        "town_landmarks": scan_town_landmarks(simlingo_fzh),
        "custom_points": scan_custom_points(create_root),
        "custom_route_graphs": scan_route_graphs(create_root),
        "american_city_intersections": scan_intersections(create_root, xodr_root),
        "xodr": scan_xodr(xodr_root),
        "xml_corpora": scan_xml_corpora(simlingo_fzh, create_root),
        "v0_artifacts": scan_v0_layer_c(evaluator_root),
        "fdq": scan_fdq(evaluator_root),
        "trajectory_rollouts": scan_rollouts(simlingo_fzh),
        "review_aids": scan_review_aids(simlingo_fzh, create_root),
        "layer_reuse": {
            "FDQ": ["fdq"],
            "A": ["american_city_intersections", "xodr.maps", "xml_corpora.town01_enriched_tasks"],
            "B": ["xodr.maps", "xml_corpora.town01_enriched_tasks", "fdq.records as actor-program substrate only"],
            "C": ["town_landmarks", "custom_points", "custom_route_graphs", "xodr.parking_spaces"],
            "D": ["compose new FSMs from validated A/B/C assets; no existing D set"],
        },
        "hard_risks": [
            "Weather, instruction-template, and initial-speed XML variants are not independent annotations.",
            "The custom parking and port route graphs are hand-authored undirected graphs, not lane reachability truth.",
            "The port topology is explicitly labelled placeholder in generate_traj_graph_sequence.py.",
            "American-city fixed-layout name-to-lane joins have spatial/heading evidence, but sign/name/arrow camera readability still needs the eight-row visual witness.",
            "Four urban path records contain null intermediate waypoints; generated XML silently omits those points.",
            "Parking XML expansion creates targets named WALL, CROSS, PILLAR, START, END, INCROSS, and bare numbers.",
            "The 247 unique authored parking slot tokens do not form a clean bijection with 249 deduplicated XODR rectangles; D314 is bound twice.",
            "Town labels contain typos, truncations, synonyms at identical coordinates, and actor/object labels that are not necessarily unique destinations.",
            "v0 layer_C_merged.xml and layer_C_manifest.json are stale two-route artifacts despite a declared total of 200.",
            "v0 layer_B_merged.xml is not an instruction-conditioned paired counterfactual dataset.",
            "The 75 collected parking rollouts are training/review evidence only, not v1 oracle witnesses.",
            "No existing compiled D-layer FSM/opposite-arm annotation was found.",
            "No semantic point/sign annotation was found for hanghailou3 or hubeiwenli4.",
        ],
    }
    catalog["release_core"] = scan_release_core(v1_root, catalog)
    return catalog


REVIEW_FIELDS = [
    "review_id",
    "pcu_id",
    "split",
    "town",
    "engineering_core_blocking",
    "paper_release_blocking",
    "automatic_geometry_join_status",
    "arm_a_semantic_label",
    "arm_a_maneuver",
    "arm_a_gate",
    "arm_a_urban_edge_token",
    "arm_a_distance_m",
    "arm_a_heading_error_deg",
    "arm_b_semantic_label",
    "arm_b_maneuver",
    "arm_b_gate",
    "arm_b_urban_edge_token",
    "arm_b_distance_m",
    "arm_b_heading_error_deg",
    "rgb_contact_sheet",
    "rgb_contact_sheet_clickable",
    "topdown_composite",
    "topdown_composite_clickable",
    "capture_hash_status",
    "question",
    "visibility_sign_arrow_decision_confirm_or_reject",
    "notes",
]


def relative_review_path(value: str | None) -> str:
    prefix = "generated/review/"
    text = str(value or "")
    return text[len(prefix):] if text.startswith(prefix) else text


def review_csv_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    audit_rows = catalog["release_core"]["c_rd_mapping_audit"]["pair_rows"]
    for item in audit_rows:
        arms = item.get("arms") or []
        if len(arms) != 2:
            raise ValueError(f"{item.get('pcu_id')}: expected exactly two C-RD arms")
        arm_a, arm_b = arms
        rgb = relative_review_path(item.get("rgb_contact_sheet"))
        topdown = f"../visualizations/{item.get('pcu_id')}/composite.png"
        geometry_status = (
            "pass:live_carla_spatial_heading_join"
            if item.get("automatic_geometry_join_valid")
            else "FAIL:geometry_join_missing_or_out_of_threshold"
        )
        capture_valid = bool(
            item.get("rgb_capture_matches_current_suite")
            and (item.get("rgb_capture_integrity") or {}).get(
                "all_files_present_and_hash_valid"
            )
        )
        question = (
            "Do the contact RGB views show a readable road-name/sign-arrow cue for both exits, "
            f"consistent with {arm_a.get('semantic_label')} -> {arm_a.get('maneuver')} and "
            f"{arm_b.get('semantic_label')} -> {arm_b.get('maneuver')}?"
        )
        rows.append(
            {
                "review_id": item.get("review_id"),
                "pcu_id": item.get("pcu_id"),
                "split": item.get("split"),
                "town": item.get("town"),
                "engineering_core_blocking": "false",
                "paper_release_blocking": "true",
                "automatic_geometry_join_status": geometry_status,
                "arm_a_semantic_label": arm_a.get("semantic_label"),
                "arm_a_maneuver": arm_a.get("maneuver"),
                "arm_a_gate": arm_a.get("target_gate"),
                "arm_a_urban_edge_token": arm_a.get("urban_edge_token"),
                "arm_a_distance_m": arm_a.get("spatial_distance_m"),
                "arm_a_heading_error_deg": arm_a.get("heading_error_deg"),
                "arm_b_semantic_label": arm_b.get("semantic_label"),
                "arm_b_maneuver": arm_b.get("maneuver"),
                "arm_b_gate": arm_b.get("target_gate"),
                "arm_b_urban_edge_token": arm_b.get("urban_edge_token"),
                "arm_b_distance_m": arm_b.get("spatial_distance_m"),
                "arm_b_heading_error_deg": arm_b.get("heading_error_deg"),
                "rgb_contact_sheet": rgb,
                "rgb_contact_sheet_clickable": f"[open RGB contact]({rgb})",
                "topdown_composite": topdown,
                "topdown_composite_clickable": f"[open top-down]({topdown})",
                "capture_hash_status": "pass" if capture_valid else "FAIL",
                "question": question,
                "visibility_sign_arrow_decision_confirm_or_reject": "",
                "notes": "",
            }
        )
    return rows


def format_slots(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) or "none"


def render_review_markdown(catalog: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    release = catalog["release_core"]
    parking = release["parking_semantics"]
    usage = release["current_core_usage"]["parking_slot_tokens_by_layer"]
    port = release["port_v13_v14"]
    lines = [
        "# Language2Drive v1 最小人工确认包",
        "",
        "当前 8 个 C-RD PCU 已由 **live CARLA junction outgoing lane ↔ 既有 urban 命名边** 的空间距离与有向航向自动 join 建立固定布局几何证据。所有 16 个 arm 均须满足距离不超过 15 m、航向误差不超过 20°，因此不再要求人工重新绑定 gate 或填写道路名称。",
        "",
        "人工只做论文级可见性 witness：逐行查看真实 AutoVLA 相机 rig 的 RGB contact sheet，确认两侧道路名/路牌箭头是否可读且与已绑定出口一致。每行只填 `confirm` 或 `reject`；不要求自由文本标注。该项不阻塞 `engineering/pre-release`，但阻塞“human-witnessed semantics”论文声明。",
        "",
        "## 8 行确认表",
        "",
        "| PCU | split | 自动绑定证据（arm A；arm B） | RGB | 俯视合图 | 决策 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = (
            f"{row['arm_a_maneuver']} → {row['arm_a_semantic_label']} "
            f"(`{row['arm_a_urban_edge_token']}`, {row['arm_a_distance_m']} m / "
            f"{row['arm_a_heading_error_deg']}°)；"
            f"{row['arm_b_maneuver']} → {row['arm_b_semantic_label']} "
            f"(`{row['arm_b_urban_edge_token']}`, {row['arm_b_distance_m']} m / "
            f"{row['arm_b_heading_error_deg']}°)"
        )
        lines.append(
            f"| {row['pcu_id']} | {row['split']} | {evidence} | "
            f"[打开 RGB]({row['rgb_contact_sheet']}) | "
            f"[打开 top-down]({row['topdown_composite']}) | ☐ confirm ☐ reject |"
        )
    lines += [
        "",
        "机器填写表为 [`MINIMAL_HUMAN_REVIEW.csv`](MINIMAL_HUMAN_REVIEW.csv)。`confirm` 表示两臂所需名称/箭头提示均可读且方向一致；只要任一臂不可读、没有可解释箭头或方向矛盾，就填 `reject`。不确定时也填 `reject`。",
        "",
        "## RGB 证据完整性",
        "",
        f"捕获清单含 {release['c_rd_rgb_capture']['camera_image_count']} 张独立 RGB（每个 PCU 6 张）和 {release['c_rd_rgb_capture']['contact_sheet_count']} 张 contact sheet。CSV 的 `capture_hash_status` 必须全部为 `pass`；SHA256 来源为 [`rgb/capture_manifest.json`](rgb/capture_manifest.json)。这些图像来自实际 AutoVLA camera rig，而非俯视图或合成路牌。",
        "",
        "## 停车与港口的零人工阻塞结论",
        "",
        f"当前核心的停车语义（精确停车目标、占用条件与 D 阶段 gate）共引用 {len(parking['all_core_selected_slot_tokens'])} 个唯一 token，且 {len(parking['selected_physical_space_ids'])} 个 physical XODR `parkingSpace` ID 一一对应；重复 physical ID 数为 {len(parking['duplicate_selected_physical_space_ids'])}。按层实际引用：",
        "",
        f"- A：{format_slots(usage.get('A') or [])}",
        f"- C：{format_slots(usage.get('C') or [])}",
        f"- D：{format_slots(usage.get('D') or [])}",
        "",
        f"`port:v13` / `port:v14` 的全核心引用数为 {port['port:v13_reference_count_in_all_core_layers']} / {port['port:v14_reference_count_in_all_core_layers']}。它们约 7.8 m / 14.5 m 的旧点投影偏差继续保留在复核清单，但因未进入核心而不占本期人工时间。",
        "",
        "## 严格发布措辞",
        "",
        "自动 join 证明的是固定布局几何对应关系，不是人类可见性确认。8 行未全部 `confirm` 前，报告必须写 `semantic_visibility_witness=pending`，不得写 `human-confirmed semantics`。固定布局仍为 `FCS-2×2=N/A: fixed_layout_only`，也不能据此声称 unseen-map generalization。",
        "",
        "若任何一行被拒绝，应淘汰并重新 mine 对应 PCU；[`C_RD_REPLACEMENT_CANDIDATES.csv`](C_RD_REPLACEMENT_CANDIDATES.csv) 仅保留为历史候选池，不是当前推荐替代或已验收核心。",
        "",
    ]
    return "\n".join(lines)


def render_rgb_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# C-RD RGB visibility/sign-arrow review",
        "",
        "These are actual AutoVLA camera-rig captures. Geometry/name binding has already passed the live-CARLA spatial+heading join; review only whether the road-name/sign-arrow cue is readable and consistent with both mapped exits. Choose **confirm** or **reject**—no relabeling is requested.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['pcu_id']}: [{row['arm_a_semantic_label']} / {row['arm_b_semantic_label']}]"
            f"(./{Path(row['rgb_contact_sheet']).name}) — ☐ confirm ☐ reject"
        )
    lines += [
        "",
        "Image and contact-sheet SHA256 values are frozen in [`capture_manifest.json`](capture_manifest.json). A reject means the required cue is absent, unreadable, ambiguous, or points to a different exit.",
        "",
    ]
    return "\n".join(lines)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False, newline=""
        ) as handle:
            temp_name = handle.name
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o644)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False, newline=""
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o644)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def write_review_package(catalog: dict[str, Any], review_output_dir: Path) -> None:
    output = review_output_dir.expanduser().resolve()
    rows = review_csv_rows(catalog)
    if len(rows) != 8:
        raise ValueError(f"review package requires exactly 8 C-RD rows, found {len(rows)}")
    if any(row["automatic_geometry_join_status"].startswith("FAIL") for row in rows):
        raise ValueError("cannot emit review package: one or more automatic geometry joins failed")
    if any(row["capture_hash_status"] != "pass" for row in rows):
        raise ValueError("cannot emit review package: RGB capture binding/hash validation failed")
    capture_source = Path(catalog["release_core"]["c_rd_rgb_capture"]["manifest"])
    capture_at_output = output / "rgb/capture_manifest.json"
    if (
        not capture_at_output.is_file()
        or not capture_source.is_file()
        or sha256_file(capture_at_output) != sha256_file(capture_source)
    ):
        raise ValueError(
            "review output directory must already contain the validated RGB tree "
            "and rgb/capture_manifest.json (normally generated/review)"
        )

    # A current top-down composite is part of every review row.  Require its
    # provenance manifest to reference the exact sidecar snapshot audited
    # above; a visually plausible image from an older build is not evidence.
    visualization_manifest_path = output.parent / "visualizations/MANIFEST.json"
    if not visualization_manifest_path.is_file():
        raise ValueError(
            "review output requires generated/visualizations/MANIFEST.json; "
            "run `l2d visualize` for the current suite first"
        )
    visualization_manifest = load_json(visualization_manifest_path)
    current_sidecar_sha256 = catalog["release_core"]["suite_snapshot"]["sha256"]
    if (
        visualization_manifest.get("source_hashes", {}).get(
            "build/suite/sidecar.private.json"
        )
        != current_sidecar_sha256
    ):
        raise ValueError("review output requires visualizations built from the current sidecar")
    visualization_files = {
        file.get("path")
        for item in (visualization_manifest.get("artifacts") or [])
        for file in (item.get("files") or [])
        if isinstance(file, dict)
    }
    for row in rows:
        expected = f"{row['pcu_id']}/composite.png"
        if expected not in visualization_files or not (output.parent / "visualizations" / expected).is_file():
            raise ValueError(f"review output missing current top-down composite: {expected}")

    atomic_write_csv(output / "MINIMAL_HUMAN_REVIEW.csv", REVIEW_FIELDS, rows)
    atomic_write_text(output / "MINIMAL_HUMAN_REVIEW.md", render_review_markdown(catalog, rows))
    atomic_write_text(output / "rgb/README.md", render_rgb_readme(rows))

    candidates = catalog["release_core"].get("replacement_candidates") or []
    candidate_fields = [
        "proposal_id", "replaces_pcu", "split", "town", "category", "same_start_vertex",
        "arm_a_source_path_id", "arm_a_end_vertex", "arm_a_instruction",
        "arm_b_source_path_id", "arm_b_end_vertex", "arm_b_instruction",
        "source_evidence", "human_rows_required", "status",
    ]
    candidate_rows = []
    for item in candidates:
        candidate_rows.append(
            {
                "proposal_id": item.get("proposal_id"),
                "replaces_pcu": item.get("replaces_pcu"),
                "split": item.get("split"),
                "town": item.get("town"),
                "category": item.get("category"),
                "same_start_vertex": item.get("same_start_vertex"),
                "arm_a_source_path_id": (item.get("arm_a") or {}).get("source_path_id"),
                "arm_a_end_vertex": (item.get("arm_a") or {}).get("end_vertex"),
                "arm_a_instruction": (item.get("arm_a") or {}).get("instruction"),
                "arm_b_source_path_id": (item.get("arm_b") or {}).get("source_path_id"),
                "arm_b_end_vertex": (item.get("arm_b") or {}).get("end_vertex"),
                "arm_b_instruction": (item.get("arm_b") or {}).get("instruction"),
                "source_evidence": item.get("source_evidence"),
                "human_rows_required": item.get(
                    "human_label_rows_required_for_engineering_pre_release"
                ),
                "status": "historical_contingency_not_current_core",
            }
        )
    atomic_write_csv(
        output / "C_RD_REPLACEMENT_CANDIDATES.csv", candidate_fields, candidate_rows
    )

    tracked = [
        output / "MINIMAL_HUMAN_REVIEW.csv",
        output / "MINIMAL_HUMAN_REVIEW.md",
        output / "C_RD_REPLACEMENT_CANDIDATES.csv",
        output / "rgb/README.md",
        output / "rgb/capture_manifest.json",
    ]
    capture_manifest = load_json(capture_at_output)
    release = catalog["release_core"]
    manifest = {
        "schema_version": 2,
        "row_count": len(rows),
        "decision_field": "visibility_sign_arrow_decision_confirm_or_reject",
        "allowed_decisions": ["confirm", "reject"],
        "source_provenance": {
            "current_core_source_sha256": release["canonical_source_snapshot"]["sha256"],
            "current_sidecar_sha256": current_sidecar_sha256,
            "current_source_and_compiled_pcus_equal": release["source_compiled_consistency"]["pcus_equal"],
            "visualization_manifest_sha256": sha256_file(visualization_manifest_path),
            "visualization_sidecar_sha256": visualization_manifest["source_hashes"][
                "build/suite/sidecar.private.json"
            ],
            "rgb_capture_manifest_sha256": sha256_file(capture_at_output),
            "rgb_capture_origin_sidecar_sha256": capture_manifest.get("sidecar_sha256"),
            "rgb_capture_origin_full_sidecar_matches_current": (
                capture_manifest.get("sidecar_sha256") == current_sidecar_sha256
            ),
            "current_c_rd_gate_name_bindings_and_rgb_hashes": "pass",
            "current_c_rd_validated_cases": release["c_rd_mapping_audit"][
                "rgb_capture_cases_with_valid_hashes_and_current_bindings"
            ],
        },
        "files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
            }
            for path in tracked
            if path.is_file()
        ],
    }
    atomic_write_text(
        output / "REVIEW_PACKAGE_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def write_json(payload: dict[str, Any], output: str, compact: bool) -> None:
    kwargs = {"ensure_ascii": False, "sort_keys": True}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = 2
    if output == "-":
        json.dump(payload, sys.stdout, **kwargs)
        sys.stdout.write("\n")
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, **kwargs)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o644)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simlingo-fzh-root", type=Path, default=DEFAULT_SIMLINGO_FZH)
    parser.add_argument(
        "--evaluator-root", type=Path, default=DEFAULT_EVALUATOR_ROOT
    )
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--output", default="-", help="JSON path, or '-' for stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument(
        "--review-output-dir",
        type=Path,
        help=(
            "also regenerate the deterministic eight-row visibility/sign-arrow "
            "review package in this directory"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        catalog = build_catalog(
            args.simlingo_fzh_root, args.evaluator_root, args.v1_root
        )
        write_json(catalog, args.output, args.compact)
        if args.review_output_dir is not None:
            write_review_package(catalog, args.review_output_dir)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, ET.ParseError) as error:
        print(f"extract_existing_labels: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
