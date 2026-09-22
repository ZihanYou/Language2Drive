from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_json, sha256_file, stable_hash, topology_spawn_token
from .maps import _import_carla
from .parking import (
    BBOX_ASSET as PARKING_BBOX_ASSET,
    SLOT_COLLISION_OCCUPANCY_ASSET,
    VEHICLE_DYNAMICS_ASSET as PARKING_VEHICLE_DYNAMICS_ASSET,
    VEHICLE_BBOX_ASSET as PARKING_VEHICLE_BBOX_ASSET,
    manoeuvre_path as _parking_manoeuvre_path,
    mine as _mine_parking_manoeuvre,
    registry as _parking_registry,
    source_paths as _parking_source_paths,
)


DATA_ROOT = (Path(__file__).resolve().parents[2] / "assets/annotations")
INTERSECTIONS = DATA_ROOT / "american_city_intersection_benchmark.json"
TRAJECTORIES = DATA_ROOT / "trj_seq_jsons"
PARKING_SLOT_OCCUPANCY_ASSET = SLOT_COLLISION_OCCUPANCY_ASSET

# Lane-change and keep-lane arms share a long physical prefix.  Their outcome
# is therefore a terminal commitment, not a mid-route observation: at 50%
# progress a correct lane change can still be travelling in the source lane.
LANE_TERMINAL_COMMITMENT_PROGRESS = 0.9
LANE_CHANGE_CONNECTOR_MAX_PROGRESS = 0.88
LANE_CHANGE_PRECOMMITMENT_MIN_MARGIN_M = 2.0
LANE_TERMINAL_PROGRESS_PROOF_STEP = 0.001
LANE_EXACT_SUFFIX_PROOF_STEP_M = 0.1

def _speed_policy(project_root: Path) -> dict[str, float | str]:
    """Load the frozen CARLA/XODR speed feasibility policy.

    OpenDRIVE is the lane-level legal-limit authority, while CARLA's vehicle
    API may expose its more conservative 30 km/h default on custom packages.
    Counterfactual bands must fit below both; the live API is checked again by
    the privileged witness on every active-band tick.
    """

    defaults = json.loads((project_root / "config/defaults.json").read_text(encoding="utf-8"))
    policy = defaults.get("speed", {})
    cap = float(policy.get("carla_runtime_conservative_cap_mps", 0.0))
    margin = float(policy.get("control_margin_mps", 0.0))
    gap = float(policy.get("counterfactual_gap_mps", 0.0))
    method = policy.get("cap_method")
    if (not math.isfinite(cap) or cap <= 0 or not math.isfinite(margin) or margin <= 0
            or not math.isfinite(gap) or gap < 2.0
            or method != "carla_default_limit_conservative_cap_v1"):
        raise RuntimeError("invalid frozen speed feasibility policy")
    return {"cap_mps": cap, "control_margin_mps": margin,
            "counterfactual_gap_mps": gap, "method": method}


def _distance(points: Iterable[dict[str, Any]]) -> float:
    values = list(points)
    return sum(math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))) for a, b in zip(values, values[1:]))


def _point(value: dict[str, Any]) -> dict[str, float]:
    point = {key: round(float(value.get(key, 0)), 4) for key in ("x", "y", "z")}
    if value.get("yaw_deg") is not None:
        point["yaw_deg"] = round(float(value["yaw_deg"]), 3)
    return point


def _polyline_fraction_point(
    points: Iterable[dict[str, Any]], fraction: float,
) -> dict[str, float]:
    """Interpolate a point at a planar arclength fraction of a polyline."""

    values = list(points)
    if len(values) < 2 or not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("polyline fraction requires two points and a finite value in [0,1]")
    lengths = [
        math.hypot(float(end["x"]) - float(start["x"]),
                   float(end["y"]) - float(start["y"]))
        for start, end in zip(values, values[1:])
    ]
    total = sum(lengths)
    if not math.isfinite(total) or total <= 1e-9:
        raise ValueError("polyline fraction requires positive planar arclength")
    target = float(fraction) * total
    along = 0.0
    for index, length in enumerate(lengths):
        if target <= along + length + 1e-9 or index == len(lengths) - 1:
            ratio = 0.0 if length <= 1e-12 else min(1.0, max(0.0, (target - along) / length))
            start, end = values[index], values[index + 1]
            return {
                axis: float(start.get(axis, 0.0))
                + ratio * (float(end.get(axis, 0.0)) - float(start.get(axis, 0.0)))
                for axis in ("x", "y", "z")
            }
        along += length
    raise AssertionError("polyline fraction interpolation fell through")


def _segment_yaw_deg(start: dict[str, Any], end: dict[str, Any]) -> float:
    dx = float(end["x"]) - float(start["x"])
    dy = float(end["y"]) - float(start["y"])
    if math.hypot(dx, dy) <= 1e-9:
        raise ValueError("path segment has no planar direction")
    return math.degrees(math.atan2(dy, dx))


def _rectangle(center: dict[str, float], yaw_deg: float, length: float = 10.0, width: float = 6.0) -> list[list[float]]:
    yaw = math.radians(yaw_deg)
    forward, side = (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))
    coordinates = []
    for longitudinal, lateral in ((length/2, width/2), (length/2, -width/2), (-length/2, -width/2), (-length/2, width/2)):
        coordinates.append([center["x"] + forward[0]*longitudinal + side[0]*lateral,
                            center["y"] + forward[1]*longitudinal + side[1]*lateral])
    coordinates.append(coordinates[0])
    return coordinates


def _forward_envelope(start: dict[str, float], yaw_deg: float, distance: float = 8.0) -> list[dict[str, float]]:
    yaw = math.radians(yaw_deg)
    return [_point(start), {"x": round(float(start["x"]) + distance*math.cos(yaw), 4),
                             "y": round(float(start["y"]) + distance*math.sin(yaw), 4),
                             "z": round(float(start.get("z", 0)), 4)}]


def _gate(identifier: str) -> dict[str, Any]:
    return {"enter_gate": {"id": identifier}}


def _eventually(expression: dict[str, Any]) -> dict[str, Any]:
    return {"eventually": expression}


def _instructions(canonical: str, second: str, third: str) -> list[str]:
    return [canonical, second, third]


def _arm(identifier: str, instructions: list[str], contract: dict[str, Any], opposite: dict[str, Any],
         outcome: str, target: dict[str, Any], oracle_path: list[dict[str, Any]],
         actor_setup: dict[str, Any] | None = None, deadline: float = 90.0) -> dict[str, Any]:
    return {"id": identifier, "instructions": instructions, "contract": contract,
            "opposite_contract": opposite, "outcome_key": outcome, "target": target,
            "actor_setup": actor_setup or {}, "deadline_seconds": deadline,
            "oracle_path": [_point(point) for point in oracle_path], "instruction_timeline": []}


def _pcu(layer: str, index: int, category: str, town: str, family: str, envelope: list[dict[str, Any]],
         arms: list[dict[str, Any]], source_refs: list[str], *, ladder: int | None = None,
         metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    identifier = f"{layer}-{index:03d}"
    split = "dev" if index <= 6 else "test"
    group = f"ladder-{ladder:02d}" if ladder is not None else f"{layer.lower()}-{category.lower()}-{index:03d}"
    semantic_state = "pending_review" if category == "road_name" else ("existing_annotation" if layer in {"C", "D"} or category == "waiting_parking" else "not_applicable")
    return {"id": identifier, "layer": layer, "category": category, "split": split,
            "group_id": group, "town": town, "map_family": family,
            "coarse_envelope": [_point(point) for point in envelope], "arms": arms,
            "source_refs": source_refs, "ladder_id": f"L{ladder:02d}" if ladder is not None else None,
            "eligibility": {"static_map": "passed", "semantic_assets": semantic_state,
                            "carla_oracle": "pending_execution"}, "metadata": metadata or {}}


def _opposite_split_spawn_tokens(
    pcus: Iterable[dict[str, Any]], split: str, town: str,
) -> set[str]:
    """Reserve already-generated physical spawn cells from the other split."""

    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    opposite = "test" if split == "dev" else "dev"
    tokens: set[str] = set()
    for pcu in pcus:
        if pcu.get("split") != opposite or pcu.get("town") != town:
            continue
        arms = pcu.get("arms")
        if not isinstance(arms, list) or not arms:
            continue
        path = arms[0].get("oracle_path") if isinstance(arms[0], dict) else None
        if not isinstance(path, list) or not path or not isinstance(path[0], dict):
            continue
        start = path[0]
        tokens.add(topology_spawn_token(town, start.get("x"), start.get("y")))
    return tokens


def _contains_parking_target(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("type") == "parking_slot" or any(
            _contains_parking_target(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_parking_target(child) for child in value)
    return False


def _maneuver_pair(approach: dict[str, Any], choices: tuple[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    values = {item["maneuver"]: item for item in approach["maneuvers"]}
    return values[choices[0]], values[choices[1]]


def _gate_target(town: str, approach: dict[str, Any], maneuver: dict[str, Any]) -> dict[str, Any]:
    region = maneuver["region"]
    return {"type": "gate", "id": f"{town}:{approach['id']}:{maneuver['maneuver']}",
            "polygon": _rectangle({**region["center"], "z": 0}, float(region["yaw_deg"]),
                                  2*float(region["half_length_m"]), 2*float(region["half_width_m"])),
            "exit_road_id": maneuver["exit_road_id"], "maneuver": maneuver["maneuver"]}


MANEUVER_TEXT = {
    "left": ("Turn left at this intersection.", "Take the left turn at the junction.", "At this intersection, go left."),
    "right": ("Turn right at this intersection.", "Take the right turn at the junction.", "At this intersection, go right."),
    "straight": ("Go straight through this intersection.", "Continue straight across the junction.", "At this intersection, keep straight."),
}


def _changeable_adjacent_ids(lane: dict[str, Any]) -> list[str]:
    permission = str(lane.get("lane_change", "NONE")).lower()
    result = []
    if lane.get("left_adjacent") and permission in {"left", "both"}:
        result.append(lane["left_adjacent"])
    if lane.get("right_adjacent") and permission in {"right", "both"}:
        result.append(lane["right_adjacent"])
    return result


def _load_lane_candidates(map_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    selected = []
    preferred = ("Town01", "Town03", "Town04", "Town05", "Town06", "hanghailou3", "hubeiwenli4", "yangluogang1")
    families = {"Town01": "M-U", "Town03": "M-U", "Town04": "M-U", "Town05": "M-U", "Town06": "M-U",
                "hanghailou3": "M-H", "hubeiwenli4": "M-C", "yangluogang1": "M-P"}
    for town in preferred:
        graph = json.loads((map_dir / f"{town}.lane_graph.json").read_text(encoding="utf-8"))
        lane_by_id={lane["id"]:lane for lane in graph["lanes"]}
        def pair_is_close(lane: dict[str, Any]) -> bool:
            adjacent=[lane_by_id[value] for value in _changeable_adjacent_ids(lane) if value in lane_by_id]
            return bool(adjacent) and all(math.dist((lane["samples"][0]["x"],lane["samples"][0]["y"]),(other["samples"][0]["x"],other["samples"][0]["y"])) <= 8 for other in adjacent)
        candidates = [lane for lane in graph["lanes"] if len(lane["samples"]) >= 35 and len(_changeable_adjacent_ids(lane)) >= 2 and not lane["is_junction"] and pair_is_close(lane)]
        if not candidates:
            candidates = [lane for lane in graph["lanes"] if len(lane["samples"]) >= 35 and _changeable_adjacent_ids(lane) and not lane["is_junction"] and pair_is_close(lane)]
        if candidates:
            selected.append((town, families[town], candidates[len(candidates)//2]))
    # The layer quota needs eight lane programs.  Multiple well-separated
    # corridors in a town are valid independent topology groups, so fill any
    # gap deterministically from additional close adjacent-lane candidates.
    if len(selected) < 8:
        chosen={(town,lane["id"]) for town,_,lane in selected}
        for town in preferred:
            graph=json.loads((map_dir/f"{town}.lane_graph.json").read_text(encoding="utf-8"));lane_by_id={lane["id"]:lane for lane in graph["lanes"]}
            for lane in graph["lanes"]:
                adjacent=[lane_by_id[value] for value in _changeable_adjacent_ids(lane) if value in lane_by_id]
                if (town,lane["id"]) in chosen or len(lane["samples"])<35 or lane["is_junction"] or not adjacent:
                    continue
                if max(math.dist((lane["samples"][0]["x"],lane["samples"][0]["y"]),(other["samples"][0]["x"],other["samples"][0]["y"])) for other in adjacent) > 8:
                    continue
                selected.append((town,families[town],lane));chosen.add((town,lane["id"]))
                if len(selected)>=8:return selected
    return selected


def _load_signal_free_speed_corridors(
    project_root: Path, map_dir: Path,
) -> dict[str, list[tuple[str, str, dict[str, Any], dict[str, Any]]]]:
    """Load independently mined, exact execution slices for speed programs.

    The audit artifacts contain 60--61 m paths plus at least 20 m of proven
    signal/STOP/junction-free lane on both sides.  We deliberately consume the
    audited path itself rather than re-mining from a lane start: a continuous
    positive speed invariant is not compatible with an unexpected red light.
    """

    primary_path = project_root / "generated/signal_free_speed_corridor_audit.json"
    reserve_path = project_root / "generated/signal_free_ad_corridor_reserve_audit.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
    if primary.get("status") != "PASS" or reserve.get("status") != "PASS":
        raise RuntimeError("signal-free corridor audit is not PASS")

    families = {
        "Town01": "M-U", "Town02": "M-U", "Town03": "M-U",
        "Town04": "M-U", "Town05": "M-U", "Town06": "M-U",
        "hanghailou3": "M-H", "hubeiwenli4": "M-C",
        "yangluogang1": "M-P",
    }
    by_assignment = {
        str(value.get("pcu_id") or value.get("recommended_assignment")): value
        for value in [*primary.get("candidates", []), *reserve.get("candidates", [])]
    }
    maps_config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    xodr_root = Path(maps_config["xodr_root"])

    def load(identifier: str) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        value = by_assignment.get(identifier)
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise RuntimeError(f"{identifier}: missing PASS signal-free corridor audit")
        town, lane_id = str(value["town"]), str(value["lane_graph_lane_id"])
        graph_path = map_dir / f"{town}.lane_graph.json"
        xodr_path = xodr_root / f"{town}.xodr"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        lane = next((item for item in graph["lanes"] if item["id"] == lane_id), None)
        if lane is None:
            raise RuntimeError(f"{identifier}: audited lane {lane_id} is absent")
        if sha256_file(graph_path) != value["source"]["lane_graph_sha256"]:
            raise RuntimeError(f"{identifier}: audited lane graph hash is stale")
        if sha256_file(xodr_path) != value["source"]["xodr_sha256"]:
            raise RuntimeError(f"{identifier}: audited XODR hash is stale")
        selection = value["lane_sample_slice"]
        start, end = int(selection["path_start_index_inclusive"]), int(selection["path_end_index_inclusive"])
        buffer_start = int(selection["buffer_start_index_inclusive"])
        buffer_end = int(selection["buffer_end_index_inclusive"])
        samples = lane["samples"]
        raw_path = [_point(item) for item in samples[start:end + 1]]
        raw_buffer = [_point(item) for item in samples[buffer_start:buffer_end + 1]]
        expected_path = value["recommended_oracle_path"]
        expected_buffer = value["audited_safety_buffer_path"]
        if raw_path != expected_path or raw_buffer != expected_buffer:
            raise RuntimeError(f"{identifier}: lane graph no longer matches audited path")
        path_hash = stable_hash(raw_path)
        buffer_hash = stable_hash(raw_buffer)
        if path_hash != value["recommended_oracle_path_canonical_sha256"]:
            raise RuntimeError(f"{identifier}: audited execution path hash mismatch")
        if buffer_hash != value["audited_safety_buffer_path_canonical_sha256"]:
            raise RuntimeError(f"{identifier}: audited safety buffer hash mismatch")
        geometry = value["geometry"]
        exact = value["exact_lane_direction_proof"]
        control = value["xodr_signal_stop_junction_proof"]
        clearance = control["supplementary_map_signal_clearance"]
        for label, candidate in (
            ("geometry", geometry), ("exact lane", exact),
            ("signal/STOP/junction", control), ("signal clearance", clearance),
        ):
            if not isinstance(candidate, dict) or candidate.get("passed") is not True:
                raise RuntimeError(f"{identifier}: nested {label} audit is not PASS")
        # Re-run the authoritative exact-lane proof over the complete 20 m
        # pre/path/post buffer.  A top-level PASS bit or artifact hash alone is
        # not allowed to substitute for checking the actual frozen geometry.
        carla_map, _, _ = _directed_route_context(project_root, town)
        buffer_checked = _validate_signal_free_execution_path(
            carla_map, raw_buffer, {"lane_id": lane_id},
        )
        if buffer_checked != int(exact["samples_checked"]):
            raise RuntimeError(
                f"{identifier}: exact buffer sample count changed "
                f"({buffer_checked} != {exact['samples_checked']})"
            )
        nearest_signal = clearance.get("nearest_other_road_signal_proxy_distance_m")
        proof = {
            "method": "exact_signal_stop_free_lane_slice_v2",
            "lane_id": lane_id,
            "path_start_index_inclusive": start,
            "path_end_index_inclusive": end,
            "buffer_start_index_inclusive": buffer_start,
            "buffer_end_index_inclusive": buffer_end,
            "road_s_min": round(min(float(selection["path_start_s"]), float(selection["path_end_s"])), 4),
            "road_s_max": round(max(float(selection["path_start_s"]), float(selection["path_end_s"])), 4),
            "corridor_length_m": round(float(geometry["path_length_m"]), 6),
            "before_buffer_length_m": round(float(geometry["before_buffer_length_m"]), 6),
            "after_buffer_length_m": round(float(geometry["after_buffer_length_m"]), 6),
            "sample_spacing_max_m": float(exact["sample_spacing_max_m"]),
            "samples_checked": int(exact["samples_checked"]),
            "junction_sample_count": int(exact["junction_samples"]),
            "road_signal_count": int(control["candidate_road_signal_count"]),
            "road_signal_reference_count": int(control["candidate_road_signal_reference_count"]),
            "road_stop_control_count": int(control["candidate_road_stop_light_object_count"]),
            "map_road_signal_count": int(clearance["map_signal_count"]),
            "other_road_signal_clearance_m": (
                None if nearest_signal is None else round(float(nearest_signal), 4)
            ),
            "execution_path_sha256": path_hash,
            "safety_buffer_path_sha256": buffer_hash,
            "lane_graph_sha256": sha256_file(graph_path),
            "xodr_sha256": str(value["source"]["xodr_sha256"]),
            "audit_sha256": sha256_file(primary_path if identifier.startswith("B-") else reserve_path),
        }
        sliced_lane = dict(lane)
        sliced_lane["samples"] = [dict(item) for item in samples[start:end + 1]]
        return town, families[town], sliced_lane, proof

    return {
        "b_dev": [load(f"B-{index:03d}") for index in range(1, 7)],
        "b_test": [load(f"B-{index:03d}") for index in range(7, 9)],
        "a_test": [load(f"A-{index:03d}") for index in range(21, 27)],
        "d_test": [load(f"D-{index:03d}") for index in range(17, 25)],
    }


def _lane_by_id(map_dir: Path, town: str) -> dict[str, dict[str, Any]]:
    graph = json.loads((map_dir / f"{town}.lane_graph.json").read_text(encoding="utf-8"))
    return {lane["id"]: lane for lane in graph["lanes"]}


def _lane_path(
    lane: dict[str, Any], distance_m: float = 65.0, *, start_index: int = 0,
) -> list[dict[str, Any]]:
    """Return a bounded, direction-preserving behavior corridor."""

    values = lane["samples"][start_index:]
    if len(values) < 2:
        raise RuntimeError(f"lane {lane['id']} has no bounded corridor after index {start_index}")
    selected = [values[0]]
    accumulated = 0.0
    for previous, current in zip(values, values[1:]):
        accumulated += math.dist(
            (float(previous["x"]), float(previous["y"])),
            (float(current["x"]), float(current["y"])),
        )
        if accumulated >= distance_m:
            selected.append(current)
            break
        if accumulated >= 8.0 * len(selected):
            selected.append(current)
    if selected[-1] is values[0]:
        selected.append(values[-1])
    return [_point(value) for value in selected]


def _lane_speed_limit_mps(project_root: Path, town: str, lane: dict[str, Any]) -> float:
    """Read the frozen OpenDRIVE lane/road speed for one behavior corridor."""

    config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    root = ET.parse(Path(config["xodr_root"]) / f"{town}.xodr").getroot()
    road = root.find(f"road[@id='{int(lane['road_id'])}']")
    if road is None:
        raise RuntimeError(f"{town}: missing OpenDRIVE road {lane['road_id']} for speed proof")
    candidates: list[tuple[float, str]] = []
    for element in road.findall(f".//lane[@id='{int(lane['lane_id'])}']/speed"):
        candidates.append((float(element.attrib["max"]), element.attrib.get("unit", "m/s")))
    if not candidates:
        for element in road.findall("type/speed"):
            candidates.append((float(element.attrib["max"]), element.attrib.get("unit", "m/s")))
    if not candidates:
        raise RuntimeError(f"{town}: no frozen speed limit for road/lane {lane['road_id']}/{lane['lane_id']}")
    factors = {"m/s": 1.0, "mps": 1.0, "km/h": 1.0 / 3.6, "kph": 1.0 / 3.6,
               "mph": 0.44704}
    values = []
    for value, unit in candidates:
        if unit not in factors:
            raise RuntimeError(f"{town}: unsupported OpenDRIVE speed unit {unit!r}")
        values.append(value * factors[unit])
    return min(values)


def _disjoint_speed_bands(legal_speed_mps: float, *, control_margin_mps: float = 0.35,
                          gap_mps: float = 2.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return two legal bands with a frozen two-m/s counterfactual gap."""

    if gap_mps < 2.0:
        raise RuntimeError("counterfactual speed-band gap must be at least 2m/s")
    if legal_speed_mps >= 9.0 + control_margin_mps:
        return (3.0, 5.0), (7.0, 9.0)
    high_max = math.floor((float(legal_speed_mps) - control_margin_mps) * 10.0) / 10.0
    high_min = high_max - 2.0
    low_max = high_min - gap_mps
    low_min = max(0.3, low_max - 2.0)
    if low_max - low_min < 1.0 or high_min - low_max < gap_mps:
        raise RuntimeError(
            f"legal speed {legal_speed_mps:.3f}m/s cannot support two separated bands"
        )
    return (round(low_min, 1), round(low_max, 1)), (round(high_min, 1), round(high_max, 1))


def _paired_lane_paths(
    project_root: Path,
    town: str,
    lane: dict[str, Any],
    lanes: dict[str, dict[str, Any]],
    distance_m: float = 55.0,
    *,
    terminal_commitment: bool = False,
) -> tuple[
    str, str, list[dict[str, float]], list[dict[str, float]], dict[str, Any] | None,
]:
    """Return one exact-lane change arm and keep arm from one shared spawn.

    A straight chord from the initial current-lane point to an adjacent-lane
    endpoint cuts across the road shoulder on several maps.  Mine a short
    diagonal at an XODR-authorized lane-change corridor, then follow the target
    lane.  Every connector is checked at 0.1 m using the same exact-lane and
    direction rule as the serialized oracle path.
    """

    changeable = _changeable_adjacent_ids(lane)
    adjacent_id = changeable[0] if changeable else None
    if adjacent_id not in lanes:
        raise RuntimeError(f"lane {lane['id']} has no represented adjacent driving lane")
    adjacent = lanes[adjacent_id]
    samples = lane["samples"]
    carla_map, _, _ = _directed_route_context(project_root, town)
    direction = -1.0 if int(lane["lane_id"]) > 0 else 1.0
    candidates = []
    for index in range(5, min(len(samples) - 10, 45)):
        current = samples[index]
        for delta_s in range(2, 36):
            desired_s = float(current["s"]) + direction * delta_s
            target = min(adjacent["samples"], key=lambda item: abs(float(item["s"]) - desired_s))
            proof = _validate_directed_polyline(carla_map, [current, target])
            if proof["valid"]:
                candidates.append((index, delta_s, current, target, proof))
                break
    if not candidates:
        raise RuntimeError(f"{town}/{lane['id']}: no exact directed lane-change connector")
    # Prefer a connector after a meaningful current-lane prefix.  The caller's
    # outcome contract is a proven terminal commitment at 90% progress.
    change_index, _, change_start, change_end, _ = candidates[-1]
    terminal_s = float(change_end["s"]) + direction * max(10.0, distance_m - change_index)
    adjacent_end = min(adjacent["samples"], key=lambda item: abs(float(item["s"]) - terminal_s))
    current_end = min(samples, key=lambda item: abs(float(item["s"]) - float(adjacent_end["s"])))
    prefix = [_point(samples[0]), _point(change_start)]
    change_path = [*prefix, _point(change_end), _point(adjacent_end)]
    keep_path = [*prefix, _point(current_end)]
    for label, path in (("change", change_path), ("keep", keep_path)):
        proof = _validate_directed_polyline(carla_map, path)
        if not proof["valid"]:
            raise RuntimeError(f"{town}/{lane['id']}: {label} lane path failed exact proof: {proof}")

    if not terminal_commitment:
        return adjacent_id, lane["id"], change_path, keep_path, None

    # A correct change may still be classified in the source lane well after
    # the midpoint of the shared prefix.  Freeze the counterfactual outcome
    # only after the connector has ended, and prove the whole terminal window
    # against exact, unprojected XODR membership.  This prevents a slow but
    # correct change from irreversibly latching the keep-lane outcome.
    change_total = _distance(change_path)
    connector_length = _distance(change_path[:3])
    connector_end_progress = connector_length / change_total
    if connector_end_progress > LANE_CHANGE_CONNECTOR_MAX_PROGRESS + 1e-12:
        raise RuntimeError(
            f"{town}/{lane['id']}: lane-change connector ends at progress "
            f"{connector_end_progress:.9f}, after the frozen maximum "
            f"{LANE_CHANGE_CONNECTOR_MAX_PROGRESS:.2f}"
        )
    precommitment_margin = (
        LANE_TERMINAL_COMMITMENT_PROGRESS * change_total - connector_length
    )
    if precommitment_margin + 1e-12 < LANE_CHANGE_PRECOMMITMENT_MIN_MARGIN_M:
        raise RuntimeError(
            f"{town}/{lane['id']}: lane-change connector has only "
            f"{precommitment_margin:.6f}m before terminal commitment; require "
            f"{LANE_CHANGE_PRECOMMITMENT_MIN_MARGIN_M:.1f}m"
        )
    carla = _import_carla()
    terminal_checks = int(round(
        (1.0 - LANE_TERMINAL_COMMITMENT_PROGRESS)
        / LANE_TERMINAL_PROGRESS_PROOF_STEP
    ))
    exact_counts: dict[str, int] = {}
    source_key = (int(lane["road_id"]), int(lane["section_id"]), int(lane["lane_id"]))
    target_key = (
        int(adjacent["road_id"]), int(adjacent["section_id"]), int(adjacent["lane_id"]),
    )
    adjacent_side = None
    if lane.get("left_adjacent") == adjacent_id:
        adjacent_side = "left"
    elif lane.get("right_adjacent") == adjacent_id:
        adjacent_side = "right"
    connector_pose = change_path[1]
    connector_waypoint = carla_map.get_waypoint(
        carla.Location(
            x=float(connector_pose["x"]), y=float(connector_pose["y"]),
            z=float(connector_pose.get("z", 0.0)),
        ),
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    permission = "" if connector_waypoint is None else str(
        connector_waypoint.lane_change
    ).casefold()
    connector_adjacent = None if connector_waypoint is None or adjacent_side is None else getattr(
        connector_waypoint, f"get_{adjacent_side}_lane"
    )()
    connector_adjacent_key = None if connector_adjacent is None else (
        int(connector_adjacent.road_id), int(connector_adjacent.section_id),
        int(connector_adjacent.lane_id),
    )
    if (connector_waypoint is None or connector_waypoint.is_junction
            or adjacent_side is None
            or not (adjacent_side in permission or "both" in permission)
            or connector_adjacent is None
            or connector_adjacent.lane_type != carla.LaneType.Driving
            or connector_adjacent_key != target_key
            or int(connector_waypoint.lane_id) * int(connector_adjacent.lane_id) <= 0):
        raise RuntimeError(
            f"{town}/{lane['id']}: connector start lacks exact local "
            f"{adjacent_side or 'unknown'} lane-change permission"
        )
    raw_checks = 0
    for label, path, expected_keys in (
        ("change", change_path, (None, source_key, target_key, target_key)),
        ("keep", keep_path, (None, source_key, source_key)),
    ):
        for pose_index, (pose, expected_key) in enumerate(zip(path, expected_keys)):
            waypoint = carla_map.get_waypoint(
                carla.Location(
                    x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0)),
                ),
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            observed = None if waypoint is None else (
                int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id),
            )
            z_error = math.inf if waypoint is None else abs(
                float(pose.get("z", 0.0)) - float(waypoint.transform.location.z)
            )
            yaw_error = math.inf if waypoint is None else abs(
                (float(waypoint.transform.rotation.yaw) - float(pose["yaw_deg"]) + 180.0)
                % 360.0 - 180.0
            )
            if (waypoint is None or (expected_key is not None and observed != expected_key)
                    or z_error > 0.1 + 1e-9 or yaw_error > 3.0 + 1e-9):
                raise RuntimeError(
                    f"{town}/{lane['id']}: {label} raw pose {pose_index} differs: "
                    f"{observed} expected={expected_key}, z_error={z_error:.6f}m, "
                    f"yaw_error={yaw_error:.6f}deg"
                )
            raw_checks += 1
    for label, path, expected in (
        ("change", change_path, adjacent),
        ("keep", keep_path, lane),
    ):
        expected_key = (
            int(expected["road_id"]), int(expected["section_id"]), int(expected["lane_id"]),
        )
        terminal_yaw = _segment_yaw_deg(path[-2], path[-1])
        for sample_index in range(terminal_checks + 1):
            fraction = LANE_TERMINAL_COMMITMENT_PROGRESS + (
                sample_index * LANE_TERMINAL_PROGRESS_PROOF_STEP
            )
            point = _polyline_fraction_point(path, min(1.0, fraction))
            waypoint = carla_map.get_waypoint(
                carla.Location(x=point["x"], y=point["y"], z=point["z"]),
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            observed = None if waypoint is None else (
                int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id),
            )
            z_error = math.inf if waypoint is None else abs(
                float(point["z"]) - float(waypoint.transform.location.z)
            )
            yaw_error = math.inf if waypoint is None else abs(
                (float(waypoint.transform.rotation.yaw) - terminal_yaw + 180.0)
                % 360.0 - 180.0
            )
            if observed != expected_key or z_error > 0.1 + 1e-9 or yaw_error > 3.0 + 1e-9:
                raise RuntimeError(
                    f"{town}/{lane['id']}: {label} terminal lane proof differs "
                    f"at progress {fraction:.3f}: {observed} != {expected_key}, "
                    f"z_error={z_error:.6f}m, yaw_error={yaw_error:.6f}deg"
                )
        exact_counts[f"{label}_terminal_window"] = terminal_checks + 1
    exact_counts["raw_serialized_poses"] = raw_checks

    # The sparse serialized change path records its connector endpoint at
    # index 2.  Densely prove the full suffix from that endpoint to the final
    # target-lane pose, rather than inferring lane identity from its endpoints.
    change_suffix = change_path[2:]
    suffix_length = _distance(change_suffix)
    suffix_checks = max(1, int(math.ceil(suffix_length / LANE_EXACT_SUFFIX_PROOF_STEP_M)))
    adjacent_key = target_key
    suffix_yaw = _segment_yaw_deg(change_suffix[0], change_suffix[-1])
    for sample_index in range(suffix_checks + 1):
        point = _polyline_fraction_point(change_suffix, sample_index / suffix_checks)
        waypoint = carla_map.get_waypoint(
            carla.Location(x=point["x"], y=point["y"], z=point["z"]),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        observed = None if waypoint is None else (
            int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id),
        )
        z_error = math.inf if waypoint is None else abs(
            float(point["z"]) - float(waypoint.transform.location.z)
        )
        yaw_error = math.inf if waypoint is None else abs(
            (float(waypoint.transform.rotation.yaw) - suffix_yaw + 180.0)
            % 360.0 - 180.0
        )
        if observed != adjacent_key or z_error > 0.1 + 1e-9 or yaw_error > 3.0 + 1e-9:
            raise RuntimeError(
                f"{town}/{lane['id']}: lane-change suffix differs at sample "
                f"{sample_index}/{suffix_checks}: {observed} != {adjacent_key}, "
                f"z_error={z_error:.6f}m, yaw_error={yaw_error:.6f}deg"
            )
    exact_counts["change_connector_to_terminal"] = suffix_checks + 1
    proof = {
        "schema_version": "language2drive.lane_terminal_commitment.v1",
        "method": "serialized_xy_connector_terminal_exact_lane_v1",
        "commitment_progress": LANE_TERMINAL_COMMITMENT_PROGRESS,
        "connector_max_progress": LANE_CHANGE_CONNECTOR_MAX_PROGRESS,
        "minimum_precommitment_margin_m": LANE_CHANGE_PRECOMMITMENT_MIN_MARGIN_M,
        "exact_suffix_step_m": LANE_EXACT_SUFFIX_PROOF_STEP_M,
        "change_connector_end_progress": round(connector_end_progress, 9),
        "change_precommitment_margin_m": round(precommitment_margin, 6),
        "change_terminal_suffix_m": round(suffix_length, 6),
        "source_lane_id": lane["id"],
        "target_lane_id": adjacent_id,
        "exact_lane_sample_counts": exact_counts,
    }
    maps_config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    xodr_path = (Path(maps_config["xodr_root"]) / f"{town}.xodr").resolve()
    lane_graph_path = (
        project_root / "generated/maps/lane_graphs" / f"{town}.lane_graph.json"
    ).resolve()
    proof.update({
        "xodr_source_ref": str(xodr_path),
        "xodr_sha256": sha256_file(xodr_path),
        "lane_graph_source_ref": str(lane_graph_path),
        "lane_graph_sha256": sha256_file(lane_graph_path),
    })
    return adjacent_id, lane["id"], change_path, keep_path, proof


def _pass_lane_proof(project_root: Path, town: str, lane: dict[str, Any]) -> dict[str, Any] | None:
    """Mirror the oracle's locked-lane pass prerequisite at generation time."""

    carla = _import_carla()
    carla_map, _, _ = _directed_route_context(project_root, town)
    # Move six metres into the represented lane so next(5m) cannot cross a
    # boundary and silently change the oracle's locked lane key.
    sample = min(lane["samples"], key=lambda value: abs(float(value["s"]) - (
        float(lane["samples"][0]["s"]) + (6.0 if int(lane["lane_id"]) < 0 else -6.0)
    )))
    waypoint = carla_map.get_waypoint(
        carla.Location(x=float(sample["x"]), y=float(sample["y"]), z=float(sample.get("z", 0.0))),
        project_to_road=False, lane_type=carla.LaneType.Driving,
    )
    if waypoint is None or waypoint.is_junction or str(waypoint.lane_change).split(".")[-1].lower() == "none":
        return None
    key = (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))
    forward = waypoint.next(5.0)
    if len(forward) != 1 or (int(forward[0].road_id), int(forward[0].section_id), int(forward[0].lane_id)) != key:
        return None
    choices = []
    for side in ("get_left_lane", "get_right_lane"):
        side_name = side.removeprefix("get_").removesuffix("_lane")
        permission = str(waypoint.lane_change).casefold()
        if "both" not in permission and side_name not in permission:
            continue
        adjacent = getattr(waypoint, side)()
        if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
            continue
        if int(adjacent.lane_id) * int(waypoint.lane_id) <= 0:
            continue
        error = abs((float(adjacent.transform.rotation.yaw) - float(waypoint.transform.rotation.yaw) + 180.0) % 360.0 - 180.0)
        adjacent_forward = adjacent.next(5.0)
        adjacent_key = (int(adjacent.road_id), int(adjacent.section_id), int(adjacent.lane_id))
        if error <= 60.0 and len(adjacent_forward) == 1 and (
            int(adjacent_forward[0].road_id), int(adjacent_forward[0].section_id), int(adjacent_forward[0].lane_id)
        ) == adjacent_key:
            choices.append((side, adjacent, error))
    if not choices:
        return None
    side, adjacent, error = choices[0]
    return {
        "start": _point(sample),
        "lane_id": f"{town}:{key[0]}:{key[1]}:{key[2]}",
        "adjacent_lane_id": f"{town}:{int(adjacent.road_id)}:{int(adjacent.section_id)}:{int(adjacent.lane_id)}",
        "adjacent_lane_numeric_id": int(adjacent.lane_id),
        "adjacent_road_id": int(adjacent.road_id),
        "adjacent_section_id": int(adjacent.section_id),
        "side": side.removeprefix("get_").removesuffix("_lane"),
        "heading_error_deg": round(error, 3),
        "lookahead_m": 5.0,
    }


def _load_slots(review_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with (review_dir / "slot_registry.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row["polygon"] = json.loads(row.pop("polygon_json"))
            if row["status"].startswith("confirmed"):
                result[row["slot_number"]] = row
    return result


def _load_slot_collision_occupancy(
    project_root: Path,
    review_dir: Path,
    slots: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, bool], list[str]]:
    """Load the frozen CARLA terminal-collision classification for all slots."""

    path = (project_root / PARKING_SLOT_OCCUPANCY_ASSET).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    root_keys = {
        "schema_version", "method", "town", "map_name", "vehicle_blueprint",
        "spawn_z_m", "repeat_count", "slots", "slots_sha256", "summary",
        "source_files", "runtime_authority", "listener_process_receipt_sha256",
        "scope_statement",
    }
    if not isinstance(value, dict) or set(value) != root_keys:
        raise RuntimeError("parking slot collision occupancy root shape differs")
    if (
        value["schema_version"]
        != "language2drive.parking_slot_collision_occupancy.v1"
        or value["method"]
        != "carla_try_spawn_release_ego_at_xodr_slot_terminal_v1"
        or value["town"] != "parkinglotfish-new3"
        or not isinstance(value["map_name"], str)
        or value["map_name"].rsplit("/", 1)[-1] != value["town"]
        or value["vehicle_blueprint"] != "vehicle.lincoln.mkz_2020"
        or type(value["spawn_z_m"]) not in {int, float}
        or float(value["spawn_z_m"]) != 0.5
        or type(value["repeat_count"]) is not int
        or value["repeat_count"] != 2
        or not isinstance(value["scope_statement"], str)
        or not value["scope_statement"]
    ):
        raise RuntimeError("parking slot collision occupancy method differs")
    rows = value["slots"]
    if (
        not isinstance(rows, list) or not rows
        or value["slots_sha256"] != stable_hash(rows)
        or [row.get("slot_number") for row in rows] != sorted(slots)
    ):
        raise RuntimeError("parking slot collision occupancy inventory differs")
    occupancy: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "slot_number", "physical_space_id", "source_entity_id",
            "terminal_pose", "terminal_spawn_clear",
        }:
            raise RuntimeError("parking slot collision occupancy row shape differs")
        slot_number = row["slot_number"]
        slot = slots.get(slot_number)
        if slot is None or type(row["terminal_spawn_clear"]) is not bool:
            raise RuntimeError("parking slot collision occupancy row is unbound")
        polygon = slot["polygon"][:-1]
        expected_pose = {
            "x": sum(float(point[0]) for point in polygon) / len(polygon),
            "y": sum(float(point[1]) for point in polygon) / len(polygon),
            "z": 0.0,
            "yaw_deg": float(slot["heading_deg"]),
        }
        if (
            row["physical_space_id"] != slot["physical_space_id"]
            or row["source_entity_id"] != slot["entity_id"]
            or row["terminal_pose"] != expected_pose
        ):
            raise RuntimeError(
                f"parking slot collision occupancy geometry differs for {slot_number}"
            )
        occupancy[slot_number] = row["terminal_spawn_clear"]
    clear_slots = sorted(name for name, clear in occupancy.items() if clear)
    blocked_slots = sorted(name for name, clear in occupancy.items() if not clear)
    summary = value["summary"]
    if not isinstance(summary, dict) or summary != {
        "slot_count": len(rows),
        "clear_count": len(clear_slots),
        "blocked_count": len(blocked_slots),
        "clear_slots_sha256": stable_hash(clear_slots),
        "blocked_slots_sha256": stable_hash(blocked_slots),
    }:
        raise RuntimeError("parking slot collision occupancy summary differs")
    sources = value["source_files"]
    expected_source_names = {
        "slot_registry", "runtime_manifest", "runtime_provenance",
        "available_maps_witness", "capture_tool",
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_names:
        raise RuntimeError("parking slot collision occupancy sources differ")
    source_refs = [str(path)]
    for name in sorted(sources):
        source = sources[name]
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise RuntimeError("parking slot collision occupancy source shape differs")
        source_path = Path(source["path"])
        if (
            not source_path.is_absolute() or not source_path.is_file()
            or sha256_file(source_path) != source["sha256"]
        ):
            raise RuntimeError(
                f"parking slot collision occupancy source is stale: {name}"
            )
        source_refs.append(str(source_path))
    if Path(sources["slot_registry"]["path"]).resolve() != (
        review_dir / "slot_registry.csv"
    ).resolve():
        raise RuntimeError("parking slot collision occupancy binds another registry")
    authority = value["runtime_authority"]
    if not isinstance(authority, dict) or set(authority) != {
        "resolved_carla_root", "manifest_sha256", "inventory_sha256",
        "files_sha256", "file_count", "total_size_bytes",
        "shipping_binary_sha256", "provenance_sha256", "available_maps_sha256",
        "client_version", "server_version", "available_map_count", "server_uid",
        "server_gids",
    }:
        raise RuntimeError("parking slot collision runtime authority shape differs")
    if (
        authority["manifest_sha256"] != sources["runtime_manifest"]["sha256"]
        or authority["provenance_sha256"]
        != sources["runtime_provenance"]["sha256"]
        or not isinstance(value["listener_process_receipt_sha256"], str)
        or len(value["listener_process_receipt_sha256"]) != 64
    ):
        raise RuntimeError("parking slot collision runtime authority differs")
    return path, occupancy, source_refs


def _path_index(path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, str], dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_start: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[tuple[int, str], dict[str, Any]] = {}
    for item in data["paths"]:
        if not item.get("reachable", True):
            continue
        start = int(item["start_vertex"])
        by_start[start].append(item)
        by_target[(start, item["end_name"])] = item
    return by_start, by_target


def _parking_paths() -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, str], dict[str, Any]], dict[str, int]]:
    path = TRAJECTORIES / "parking_traj_graph_sequence_starts_150_151_158_159.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_start, by_target = _path_index(path)
    slot_vertex = {}
    for key, meta in data["vertex_meta"].items():
        import re
        match = re.search(r"([A-F]\d{3})", meta["primary_name"])
        if match and match.group(1) not in slot_vertex:
            slot_vertex[match.group(1)] = int(key)
    return by_start, by_target, slot_vertex


def _path_for_vertex(paths: dict[int, list[dict[str, Any]]], start: int, vertex: int) -> dict[str, Any]:
    return next(item for item in paths[start] if int(item["end_vertex"]) == vertex)


def _slot_target(slot: dict[str, Any]) -> dict[str, Any]:
    physical_id = str(slot.get("physical_space_id", "")).strip()
    if not physical_id:
        raise RuntimeError(f"slot {slot['slot_number']} has no one-to-one XODR parking-space binding")
    return {"type": "parking_slot", "id": physical_id, "source_entity_id": slot["entity_id"],
            "slot_number": slot["slot_number"],
            "region": slot["region"], "polygon": slot["polygon"], "heading_deg": float(slot["heading_deg"]),
            "speed_max_mps": 0.2, "dwell_seconds": 3.0, "heading_tolerance_deg": 15.0}


def _parking_contract(target: dict[str, Any]) -> dict[str, Any]:
    return {"sequence": [
        {"vehicle_footprint_inside_slot": {"polygon": target["polygon"], "tolerance_m": 0.2,
                                            "heading_deg": target["heading_deg"], "heading_tolerance_deg": 15.0}},
        {"dwell": {"key": target["id"], "seconds": 3.0, "max_speed_mps": 0.2}},
    ]}


def _region_polygon(slots: list[dict[str, Any]]) -> list[list[float]]:
    """Return a compact, non-overlapping semantic-zone terminal polygon.

    Letter regions in the legacy registry are discontinuous groups, so their
    global bounding boxes swallow unrelated aisles and other regions.  The
    core contract uses the robust median slot centre and a compact 12x8m
    terminal region; all source slots remain visible in the review overlay.
    """

    centers=[]
    for slot in slots:
        polygon=slot["polygon"][:-1] if slot["polygon"][0] == slot["polygon"][-1] else slot["polygon"]
        centers.append((sum(point[0] for point in polygon)/len(polygon),sum(point[1] for point in polygon)/len(polygon)))
    centers.sort();center_x=centers[len(centers)//2][0]
    center_y=sorted(value[1] for value in centers)[len(centers)//2]
    return [[center_x-6,center_y-4],[center_x+6,center_y-4],[center_x+6,center_y+4],[center_x-6,center_y+4],[center_x-6,center_y-4]]


def _terminal_region(slot: dict[str, Any], margin_m: float = 1.5) -> list[list[float]]:
    polygon=slot["polygon"][:-1] if slot["polygon"][0] == slot["polygon"][-1] else slot["polygon"]
    low_x,high_x=min(p[0] for p in polygon)-margin_m,max(p[0] for p in polygon)+margin_m
    low_y,high_y=min(p[1] for p in polygon)-margin_m,max(p[1] for p in polygon)+margin_m
    return [[low_x,low_y],[high_x,low_y],[high_x,high_y],[low_x,high_y],[low_x,low_y]]


def _polygon_centroid(polygon: list[list[float]]) -> dict[str, float]:
    vertices = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
    if not vertices:
        raise ValueError("empty polygon")
    return {"x": sum(float(point[0]) for point in vertices) / len(vertices),
            "y": sum(float(point[1]) for point in vertices) / len(vertices), "z": 0.0}


def _path_to_target(points: list[dict[str, Any]], target: dict[str, Any]) -> list[dict[str, Any]]:
    """End an oracle witness at the authoritative target geometry.

    Legacy parking trajectories terminate at hand-clicked label points,
    while scoring uses XODR polygons. Keeping the labelled approach but
    replacing its last point with the polygon centroid gives the private
    oracle a physically meaningful final manoeuvre.
    """

    polygon = target.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return list(points)
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    endpoint = {
        "x": sum(float(point[0]) for point in vertices) / len(vertices),
        "y": sum(float(point[1]) for point in vertices) / len(vertices),
        "z": float(points[-1].get("z", 0.0)) if points else 0.0,
    }
    if not points:
        return [endpoint]
    return [*points[:-1], endpoint]


def _terminal_region_contract(polygon: list[list[float]], seconds: float) -> dict[str, Any]:
    """Require stable terminal occupancy instead of poisoning transit paths."""

    return {
        "eventually": {
            "dwell": {
                "condition": {"inside_region": {"polygon": polygon}},
                "seconds": seconds,
            }
        }
    }


_REGION_TERMINAL_POSES: dict[str, dict[str, float]] = {
    # Deterministically mined against parkinglotfish-new3.xodr for the fixed
    # Lincoln MKZ 2020 evaluator ego.  Every pose is exact Driving/Parking-lane
    # legal and its full 2.446 x 0.918 m half-extent footprint lies inside the
    # confirmed compact semantic region with zero polygon tolerance.  The
    # independent audit and connector proof are frozen in
    # generated/dynamic_oracle_audit.json.
    # Keys are the confirmed representative XODR slot, not the public region
    # letter: the same letter legitimately has multiple disjoint polygons.
    "A804": {"x": -83.7334, "y": -47.1255, "z": 0.0, "yaw_deg": 356.977},
    "B207": {"x": 31.2260, "y": -53.0805, "z": 0.0, "yaw_deg": 356.977},
    "C106": {"x": 96.4425, "y": -33.5234, "z": 0.0, "yaw_deg": 356.756},
    "D507": {"x": -96.9000, "y": -46.3126, "z": 0.0, "yaw_deg": 356.977},
    "E101": {"x": 85.0226, "y": -32.8904, "z": 0.0, "yaw_deg": 356.756},
    "F103": {"x": -33.2783, "y": -26.2030, "z": 0.0, "yaw_deg": 356.756},
    "D406": {"x": -139.3290, "y": -44.0672, "z": 0.0, "yaw_deg": 356.977},
    "E703": {"x": 11.6681, "y": -28.8635, "z": 0.0, "yaw_deg": 356.756},
    "D501": {"x": -72.6295, "y": -47.6020, "z": 0.0, "yaw_deg": 356.977},
    "F409": {"x": -100.2161, "y": -22.3848, "z": 0.0, "yaw_deg": 356.756},
    "A208": {"x": 57.6759, "y": -54.5232, "z": 0.0, "yaw_deg": 356.977},
}


def _polygon_contains(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    vertices = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
    inside = False
    previous = len(vertices) - 1
    for index, vertex in enumerate(vertices):
        xi, yi = float(vertex[0]), float(vertex[1])
        xj, yj = float(vertices[previous][0]), float(vertices[previous][1])
        if ((yi > y) != (yj > y)
                and x <= (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        previous = index
    return inside


def _validate_region_terminal_pose(
    project_root: Path,
    identifier: str,
    pose: dict[str, float],
    polygon: list[list[float]],
) -> dict[str, Any]:
    """Fail closed unless a fixed semantic-region terminal is physically legal.

    The evaluator ego is frozen to ``vehicle.lincoln.mkz_2020``.  These CARLA
    0.9.15 blueprint extents/offsets are versioned in the independent dynamic
    audit; checking the complete oriented footprint here prevents a map or
    generated-polygon change from silently invalidating the witness pose.
    """

    carla = _import_carla()
    carla_map, _, _ = _directed_route_context(project_root, "parkinglotfish-new3")
    location = carla.Location(
        x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0))
    )
    waypoint = None
    for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
        candidate = carla_map.get_waypoint(
            location, project_to_road=False, lane_type=lane_type
        )
        if candidate is not None:
            waypoint = candidate
            break
    if waypoint is None:
        raise RuntimeError(f"region {identifier}: terminal pose is not exact-lane legal")

    yaw = math.radians(float(pose["yaw_deg"]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    offset_x, offset_y = -0.006293095648288727, 0.000002179145667469129
    center_x = float(pose["x"]) + offset_x * cosine - offset_y * sine
    center_y = float(pose["y"]) + offset_x * sine + offset_y * cosine
    half_length, half_width = 2.44619083404541, 0.9183566570281982
    corners = [
        (
            center_x + longitudinal * cosine - lateral * sine,
            center_y + longitudinal * sine + lateral * cosine,
        )
        for longitudinal, lateral in (
            (half_length, half_width), (half_length, -half_width),
            (-half_length, -half_width), (-half_length, half_width),
        )
    ]
    if not all(_polygon_contains(corner, polygon) for corner in corners):
        raise RuntimeError(
            f"region {identifier}: Lincoln MKZ footprint leaves terminal polygon"
        )
    lane_yaw = float(waypoint.transform.rotation.yaw)
    yaw_error = abs((float(pose["yaw_deg"]) - lane_yaw + 180.0) % 360.0 - 180.0)
    # The map-legality predicate permits any non-opposing orientation.  The
    # stricter dynamic constraint is imposed below against the final connector
    # tangent, because these semantic regions include manoeuvring space rather
    # than a single lane-aligned parking pose.
    if yaw_error > 120.0:
        raise RuntimeError(
            f"region {identifier}: terminal heading differs from lane by {yaw_error:.2f} degrees"
        )
    return {
        "method": "offline_carla_exact_lane_plus_full_ego_footprint_v1",
        "ego_blueprint": "vehicle.lincoln.mkz_2020",
        "road_id": int(waypoint.road_id),
        "section_id": int(waypoint.section_id),
        "lane_id": int(waypoint.lane_id),
        "lane_yaw_deg": round(lane_yaw, 3),
        "heading_error_deg": round(yaw_error, 3),
        "polygon_tolerance_m": 0.0,
    }


def _validate_region_terminal_connector(
    project_root: Path,
    identifier: str,
    pose: dict[str, float],
    path: list[dict[str, float]],
) -> dict[str, Any]:
    if len(path) < 2:
        raise RuntimeError(f"region {identifier}: terminal route has fewer than two points")
    start, end = path[-2], path[-1]
    dx, dy = float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"])
    distance = math.hypot(dx, dy)
    if distance <= 0.05:
        raise RuntimeError(f"region {identifier}: terminal connector collapsed")
    tangent = math.degrees(math.atan2(dy, dx))
    tangent_error = abs((float(pose["yaw_deg"]) - tangent + 180.0) % 360.0 - 180.0)
    if tangent_error > 15.0:
        raise RuntimeError(
            f"region {identifier}: terminal pose differs from connector by {tangent_error:.2f} degrees"
        )
    carla = _import_carla()
    carla_map, _, _ = _directed_route_context(project_root, "parkinglotfish-new3")
    samples = max(1, int(math.ceil(distance / 0.1)))
    for index in range(samples + 1):
        ratio = index / samples
        location = carla.Location(
            x=float(start["x"]) + ratio * dx,
            y=float(start["y"]) + ratio * dy,
            z=float(start.get("z", 0.0)) + ratio * (
                float(end.get("z", 0.0)) - float(start.get("z", 0.0))
            ),
        )
        if not any(
            carla_map.get_waypoint(
                location, project_to_road=False, lane_type=lane_type
            ) is not None
            for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking)
        ):
            raise RuntimeError(
                f"region {identifier}: terminal connector leaves exact legal lanes at sample {index}/{samples}"
            )
    return {
        "connector_distance_m": round(distance, 3),
        "connector_tangent_deg": round(tangent, 3),
        "connector_heading_error_deg": round(tangent_error, 3),
        "connector_sample_spacing_max_m": 0.1,
        "connector_samples": samples + 1,
        "connector_exact_lane_legal": True,
    }


def _common_directed_region_pair(
    project_root: Path,
    first_pose: dict[str, float],
    second_pose: dict[str, float],
    *,
    minimum_start_m: float = 20.0,
    maximum_route_m: float = 260.0,
) -> dict[str, Any]:
    """Mine a shared spawn whose final CARLA route tangent matches each pose."""

    carla = _import_carla()
    carla_map, planner, candidates = _directed_route_context(
        project_root, "parkinglotfish-new3"
    )
    raw_poses = (first_pose, second_pose)
    # Several confirmed region polygons sit laterally beside the OpenDRIVE
    # centerline.  Routing directly to the exact pose would append a 90-degree
    # lateral jump.  Build a short forward offset approach instead: project a
    # point four metres behind the terminal yaw to the centerline, then join an
    # exact stage three metres behind the pose and finally the pose itself.
    stages = []
    terminals = []
    for pose in raw_poses:
        yaw = math.radians(float(pose["yaw_deg"]))
        stage = {
            "x": float(pose["x"]) - 3.0 * math.cos(yaw),
            "y": float(pose["y"]) - 3.0 * math.sin(yaw),
            "z": float(pose.get("z", 0.0)),
        }
        projection_probe = carla.Location(
            x=float(pose["x"]) - 4.0 * math.cos(yaw),
            y=float(pose["y"]) - 4.0 * math.sin(yaw),
            z=float(pose.get("z", 0.0)),
        )
        terminal = carla_map.get_waypoint(
            projection_probe, project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        if terminal is None:
            terminal = carla_map.get_waypoint(
                projection_probe, project_to_road=True, lane_type=carla.LaneType.Parking,
            )
        stages.append(stage); terminals.append(terminal)
    if any(value is None for value in terminals):
        raise RuntimeError("region terminal pose has no exact Driving/Parking waypoint")
    ranked_candidates = []
    for candidate in candidates:
        distances = [candidate.transform.location.distance(value.transform.location) for value in terminals]
        if min(distances) < minimum_start_m or max(distances) > maximum_route_m:
            continue
        ranked_candidates.append((
            max(distances), sum(distances), int(candidate.road_id), int(candidate.section_id),
            int(candidate.lane_id), round(float(candidate.s), 3), candidate,
        ))
    ranked_candidates.sort(key=lambda row: row[:6])
    for _, _, _, _, _, _, candidate in ranked_candidates:
        paths, lengths, valid = [], [], True
        for pose, terminal in zip(raw_poses, terminals):
            direct = candidate.transform.location.distance(terminal.transform.location)
            if direct < minimum_start_m or direct > maximum_route_m:
                valid = False; break
            leg_result = _directed_leg(
                "parkinglotfish-new3", carla_map, planner, candidate, terminal,
            )
            if leg_result is None:
                valid = False; break
            locations = list(leg_result[0])
            stage = stages[len(paths)]
            exact_stage = carla.Location(x=stage["x"], y=stage["y"], z=stage["z"])
            exact = carla.Location(x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0)))
            if locations[-1].distance(exact_stage) > 0.05:
                locations.append(exact_stage)
            locations.append(exact)
            proof = _validate_directed_polyline(carla_map, locations)
            if not proof["valid"]:
                valid = False; break
            points = [_point({"x": item.x, "y": item.y, "z": item.z}) for item in locations]
            try:
                _validate_region_terminal_connector(
                    project_root, "candidate", pose, points,
                )
            except RuntimeError:
                valid = False; break
            length = _polyline_length(locations)
            if length > maximum_route_m or length > direct * 1.8 + 25.0:
                valid = False; break
            paths.append(points); lengths.append(length)
        if valid:
            return {
                "start": paths[0][0],
                "yaw_deg": round(float(candidate.transform.rotation.yaw), 3),
                "paths": paths,
                "canonical_lengths_m": [round(value, 3) for value in lengths],
                "candidate_policy": "first_complete_proof_after_stable_distance_lane_sort_v1",
            }
    raise RuntimeError(
        f"no shared directed region spawn for {first_pose} / {second_pose}"
    )


def _humanize(value: str) -> str:
    import re
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("Near To", " near ").replace("And", " and ")
    return " ".join(text.split())


def _port_points() -> dict[int, dict[str, float]]:
    """Load the existing hand-labelled port locations keyed by their V id."""

    import re
    raw_path = DATA_ROOT / "carla_point_jsons/port.json"
    result: dict[int, dict[str, float]] = {}
    for value in json.loads(raw_path.read_text(encoding="utf-8")):
        match = re.match(r"V(\d+)_", str(value.get("word", "")))
        if match:
            result[int(match.group(1))] = _point(value["location"])
    return result


def _angle_difference_degrees(first: float, second: float) -> float:
    """Smallest signed CARLA-yaw delta from ``first`` to ``second``."""

    return (float(second) - float(first) + 180.0) % 360.0 - 180.0


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float],
                            end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return math.dist(point, start)
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    projected = (start[0] + ratio * dx, start[1] + ratio * dy)
    return math.dist(point, projected)


def _urban_name_segments() -> list[dict[str, Any]]:
    """Bind the existing live-CARLA urban labels to geometric segments.

    The old manual graph occasionally stores an edge key in the reverse
    direction (for example ``7-14`` while the registry contains ``14-7``).
    Street names are undirected assets, so both spellings are resolved while
    route direction remains governed by CARLA topology.
    """

    graph_path = TRAJECTORIES / "urban_traj_graph_sequence_manual.json"
    point_path = DATA_ROOT / "carla_point_jsons/urban.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    points: dict[str, dict[str, Any]] = {}
    for row in json.loads(point_path.read_text(encoding="utf-8")):
        points[str(row["word"]).lower()] = row

    def resolve(identifier: str) -> dict[str, Any] | None:
        return points.get(identifier.lower()) or points.get(f"{identifier}_N".lower())

    result: list[dict[str, Any]] = []
    for path in graph["paths"]:
        for first, second, edge in zip(path["vertices_along_path"], path["vertices_along_path"][1:],
                                       path["edge_tokens_along_path"]):
            name = graph["road_names"].get(edge)
            if name is None:
                name = graph["road_names"].get("-".join(reversed(edge.split("-"))))
            a, b = resolve(first), resolve(second)
            if not name or a is None or b is None:
                continue
            start = (float(a["location"]["x"]), float(a["location"]["y"]))
            end = (float(b["location"]["x"]), float(b["location"]["y"]))
            heading = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
            result.append({"name": name, "start": start, "end": end, "heading_deg": heading,
                           "edge_token": edge, "source_path_id": path["path_id"]})
    return result


def _direction_word(yaw_deg: float) -> str:
    # CARLA yaw is expressed in its left-handed world coordinates.  The names
    # here are only disambiguators for two directions of the same named road.
    yaw = float(yaw_deg) % 360.0
    choices = ((0.0, "eastbound"), (90.0, "southbound"), (180.0, "westbound"), (270.0, "northbound"))
    return min(choices, key=lambda item: abs(_angle_difference_degrees(yaw, item[0])))[1]


def _mine_carla_junction_approaches(project_root: Path) -> list[dict[str, Any]]:
    """Mine directed, same-spawn counterfactual junction pairs from CARLA.

    This deliberately does not use the legacy OpenDRIVE-only benchmark poses:
    that file was generated before the OpenDRIVE-to-Unreal Y-axis conversion
    and several poses project onto unrelated lanes.  Junction lane pairs from
    ``carla.Junction`` are the reachability authority.  The first eight
    returned approaches additionally have two street-name bindings supported
    by existing live-CARLA annotations, allowing legal A/B/C matched ladders.
    """

    carla = _import_carla()
    maps_config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    xodr_path = Path(maps_config["xodr_root"]) / "American-city0428.xodr"
    carla_map = carla.Map("American-city0428", xodr_path.read_text(encoding="utf-8"))
    junctions = {waypoint.junction_id: waypoint.get_junction()
                 for waypoint in carla_map.generate_waypoints(2.0) if waypoint.is_junction}
    name_segments = _urban_name_segments()

    candidates: list[dict[str, Any]] = []
    for junction_id, junction in sorted(junctions.items()):
        grouped: dict[tuple[float, float, float], list[tuple[Any, Any]]] = defaultdict(list)
        for begin, end in junction.get_waypoints(carla.LaneType.Driving):
            transform = begin.transform
            key = (round(float(transform.location.x), 1), round(float(transform.location.y), 1),
                   round(float(transform.rotation.yaw) % 360.0, 1))
            grouped[key].append((begin, end))

        for group_key, pairs in sorted(grouped.items()):
            if len(pairs) < 2:
                continue
            branch_rows: list[dict[str, Any]] = []
            common_starts: list[Any] = []
            for begin, end in pairs:
                previous, following = begin.previous(15.0), end.next(18.0)
                if not previous or not following:
                    continue
                start, target = previous[0], following[0]
                common_starts.append(start)
                delta = _angle_difference_degrees(begin.transform.rotation.yaw, end.transform.rotation.yaw)
                maneuver = "straight" if abs(delta) <= 35.0 else ("left" if delta < 0 else "right")
                target_transform = target.transform
                target_xy = (float(target_transform.location.x), float(target_transform.location.y))
                target_heading = float(target_transform.rotation.yaw) % 180.0
                name_matches = []
                for segment in name_segments:
                    heading_error = abs(_angle_difference_degrees(target_heading, segment["heading_deg"]))
                    heading_error = min(heading_error, abs(180.0 - heading_error))
                    if heading_error > 20.0:
                        continue
                    distance = _point_segment_distance(target_xy, segment["start"], segment["end"])
                    name_matches.append((distance, heading_error, segment))
                name_matches.sort(key=lambda value: (value[0], value[1], value[2]["name"]))
                semantic = name_matches[0] if name_matches and name_matches[0][0] <= 15.0 else None
                begin_point, end_point, target_point = (_point({"x": value.transform.location.x,
                    "y": value.transform.location.y, "z": value.transform.location.z})
                    for value in (begin, end, target))
                canonical_length = (start.transform.location.distance(begin.transform.location)
                                    + abs(float(end.s) - float(begin.s))
                                    + end.transform.location.distance(target.transform.location))
                branch_rows.append({
                    "maneuver": maneuver,
                    "exit_road_id": int(target.road_id),
                    "connecting_road_id": int(begin.road_id),
                    "heading_delta_deg": round(delta, 3),
                    "region": {"center": target_point, "yaw_deg": round(float(target_transform.rotation.yaw), 3),
                               "half_length_m": 5.0, "half_width_m": 3.0},
                    "junction_path": [begin_point, end_point, target_point],
                    "canonical_path_length_m": round(canonical_length, 3),
                    "semantic_name": semantic[2]["name"] if semantic else None,
                    "semantic_distance_m": round(float(semantic[0]), 3) if semantic else None,
                    "semantic_heading_error_deg": round(float(semantic[1]), 3) if semantic else None,
                    "semantic_edge_token": semantic[2]["edge_token"] if semantic else None,
                    "direction": _direction_word(target_transform.rotation.yaw),
                })
            if len(branch_rows) < 2 or len({row["maneuver"] for row in branch_rows}) < 2:
                continue
            reference = common_starts[0].transform.location
            if any(reference.distance(item.transform.location) > 1.0 for item in common_starts[1:]):
                continue
            start_transform = common_starts[0].transform
            start = _point({"x": start_transform.location.x, "y": start_transform.location.y,
                            "z": start_transform.location.z})
            # Prefer two semantically bound branches for matched ladders.  For
            # A-only candidates, use the most distinct maneuver pair.
            named = [row for row in branch_rows if row["semantic_name"]]
            selected = named[:2] if len(named) >= 2 else branch_rows[:2]
            for row in selected:
                # Location-only replanning through an overlapping junction can
                # project an intermediate point onto a sibling connecting
                # lane.  A direct start-to-exit query is unambiguous; retain
                # the lane-pair samples as provenance and use their s-length
                # as the independent canonical distance check.
                row["junction_path_provenance"] = row.pop("junction_path")
                row["path"] = [start, row["region"]["center"]]
            incoming = common_starts[0]
            candidates.append({
                "id": f"carla_j{junction_id}_r{incoming.road_id}_l{incoming.lane_id}_s{int(round(incoming.s))}",
                "junction_id": int(junction_id),
                "incoming_road_id": int(incoming.road_id),
                "start": {**start, "yaw_deg": round(float(start_transform.rotation.yaw), 3)},
                "junction_center": {"x": group_key[0], "y": group_key[1]},
                "maneuvers": selected,
                "semantic_pair": len(named) >= 2,
            })

    semantic = [item for item in candidates if item["semantic_pair"]]
    if len(semantic) < 8:
        raise RuntimeError(f"only {len(semantic)} existing-label-backed CARLA semantic junction pairs; need 8")
    # Semantic entities are physical sign-to-exit bindings (approach gate plus
    # labelled edge), not the bare lexical road-name string: the same public
    # street name legitimately appears on multiple independent signs.
    selected = semantic[:8]
    selected_ids = {item["id"] for item in selected}
    used_junctions = {item["junction_id"] for item in selected}
    # Add four independent A-only approaches, preferring unseen junctions.
    remainder = [item for item in candidates if item["id"] not in selected_ids]
    remainder.sort(key=lambda item: (item["junction_id"] in used_junctions, item["junction_id"], item["id"]))
    selected.extend(remainder[:4])
    if len(selected) < 12:
        raise RuntimeError(f"only {len(selected)} directed CARLA junction pairs; need 12")
    return selected


_DIRECTED_ROUTE_CACHE: dict[str, tuple[Any, Any, list[Any]]] = {}
_DIRECTED_LEG_CACHE: dict[tuple[Any, ...], tuple[list[Any], float, dict[str, Any]] | None] = {}


def _directed_route_context(project_root: Path, town: str) -> tuple[Any, Any, list[Any]]:
    if town in _DIRECTED_ROUTE_CACHE:
        return _DIRECTED_ROUTE_CACHE[town]
    carla = _import_carla()
    python_api = (Path(__file__).resolve().parents[2] / "carla_runtime/PythonAPI/carla")
    if str(python_api) not in sys.path:
        sys.path.insert(0, str(python_api))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    maps_config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    xodr_path = Path(maps_config["xodr_root"]) / f"{town}.xodr"
    carla_map = carla.Map(town, xodr_path.read_text(encoding="utf-8"))
    context = (carla_map, GlobalRoutePlanner(carla_map, 1.0),
               [item for item in carla_map.generate_waypoints(5.0) if not item.is_junction])
    _DIRECTED_ROUTE_CACHE[town] = context
    return context


def _polyline_length(points: list[Any]) -> float:
    return sum(first.distance(second) for first, second in zip(points, points[1:]))


def _waypoint_key(waypoint: Any) -> tuple[int, int, int, float]:
    return (
        int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id),
        round(float(waypoint.s), 3),
    )


def _candidate_distance_key(candidate: Any, distances: list[float]) -> tuple[Any, ...]:
    """Stable, geometry-only ordering used before expensive route proofs."""

    return (
        round(max(distances), 6), round(sum(distances), 6),
        *_waypoint_key(candidate),
    )


def _directed_leg(
    town: str,
    carla_map: Any,
    planner: Any,
    start: Any,
    end: Any,
) -> tuple[list[Any], float, dict[str, Any]] | None:
    """Compile and prove one directed GRP leg with process-local caching.

    The cache stores only map-geometric facts for the current in-process XODR
    context.  Call-site policy (length limits, gate order, deadlines) is still
    evaluated on every use.  Failed/empty/reverse legs are negative-cached so
    repeated D programs cannot spend minutes proving the same impossibility.
    """

    start_location = start.transform.location
    end_location = end.transform.location
    key = (
        town, *_waypoint_key(start),
        round(float(start_location.x), 4), round(float(start_location.y), 4),
        round(float(start_location.z), 4),
        *_waypoint_key(end),
        round(float(end_location.x), 4), round(float(end_location.y), 4),
        round(float(end_location.z), 4),
    )
    if key in _DIRECTED_LEG_CACHE:
        return _DIRECTED_LEG_CACHE[key]
    try:
        route = planner.trace_route(start.transform.location, end_location)
    except Exception:
        _DIRECTED_LEG_CACHE[key] = None
        return None
    if not route:
        _DIRECTED_LEG_CACHE[key] = None
        return None
    locations = [waypoint.transform.location for waypoint, _ in route]
    # GRP samples on a one-metre grid and may return its first/last sample just
    # behind a requested projected waypoint.  Connecting that sample verbatim
    # creates a short reverse seam between otherwise forward legs.  Trim only
    # within the local 2.5 m sampling window, and require the replacement
    # connector itself to pass the same 0.1 m exact-lane/direction proof.
    start_index = next((
        index for index, value in enumerate(locations)
        if start_location.distance(value) <= 2.5
        and _validate_directed_polyline(carla_map, [start_location, value])["valid"]
    ), None)
    end_index = next((
        index for index in range(len(locations) - 1, -1, -1)
        if locations[index].distance(end_location) <= 2.5
        and _validate_directed_polyline(carla_map, [locations[index], end_location])["valid"]
    ), None)
    if start_index is None or end_index is None or start_index > end_index:
        _DIRECTED_LEG_CACHE[key] = None
        return None
    locations = locations[start_index:end_index + 1]
    if locations[0].distance(start_location) > 0.01:
        locations.insert(0, start_location)
    else:
        locations[0] = start_location
    if locations[-1].distance(end_location) > 0.01:
        locations.append(end_location)
    else:
        locations[-1] = end_location
    proof = _validate_directed_polyline(carla_map, locations)
    if not proof["valid"]:
        _DIRECTED_LEG_CACHE[key] = None
        return None
    result = (locations, _polyline_length(locations), proof)
    _DIRECTED_LEG_CACHE[key] = result
    return result


def _validate_directed_polyline(
    carla_map: Any,
    points: list[Any],
    *,
    sample_spacing_m: float = 0.1,
    maximum_heading_error_deg: float = 90.0,
) -> dict[str, Any]:
    """Prove that every sampled chord is on an exact, forward legal lane.

    CARLA's route planner can return a non-empty trace whose first chord runs
    backwards along a lane.  It can also expose a sparse chord which cuts off
    the represented lane.  Generation therefore uses the evaluator's exact
    waypoint precedence (Driving, then Parking) at 0.1 m spacing and a stricter
    90-degree direction margin before accepting a candidate.
    """

    carla = _import_carla()

    def location(value: Any) -> Any:
        if all(hasattr(value, axis) for axis in ("x", "y", "z")):
            return value
        return carla.Location(
            x=float(value["x"]), y=float(value["y"]),
            z=float(value.get("z", 0.0)),
        )

    if len(points) < 2:
        return {"valid": False, "failure": "fewer_than_two_points"}
    checked_samples = 0
    for segment_index, (raw_start, raw_end) in enumerate(zip(points, points[1:])):
        start, end = location(raw_start), location(raw_end)
        dx, dy, dz = (
            float(end.x) - float(start.x),
            float(end.y) - float(start.y),
            float(end.z) - float(start.z),
        )
        distance = math.hypot(dx, dy)
        if distance <= 0.05:
            continue
        tangent_yaw = math.degrees(math.atan2(dy, dx))
        sample_count = max(1, int(math.ceil(distance / sample_spacing_m)))
        for sample_index in range(sample_count + 1):
            ratio = sample_index / sample_count
            sample = carla.Location(
                x=float(start.x) + ratio * dx,
                y=float(start.y) + ratio * dy,
                z=float(start.z) + ratio * dz,
            )
            waypoint = None
            for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
                waypoint = carla_map.get_waypoint(
                    sample, project_to_road=False, lane_type=lane_type,
                )
                if waypoint is not None:
                    break
            checked_samples += 1
            if waypoint is None:
                return {
                    "valid": False,
                    "failure": "no_exact_driving_or_parking",
                    "segment_index": segment_index,
                    "sample_index": sample_index,
                }
            if not waypoint.is_junction:
                lane_yaw = float(waypoint.transform.rotation.yaw)
                heading_error = abs(
                    (tangent_yaw - lane_yaw + 180.0) % 360.0 - 180.0
                )
                if heading_error > maximum_heading_error_deg:
                    return {
                        "valid": False,
                        "failure": "wrong_direction",
                        "segment_index": segment_index,
                        "sample_index": sample_index,
                        "heading_error_deg": round(heading_error, 3),
                    }
    return {
        "valid": True,
        "sample_spacing_max_m": sample_spacing_m,
        "maximum_heading_error_deg": maximum_heading_error_deg,
        "checked_samples": checked_samples,
    }


def _validate_signal_free_execution_path(
    carla_map: Any, points: list[dict[str, Any]], proof: dict[str, Any],
) -> int:
    """Re-prove the serialized speed path against its exact frozen lane."""

    carla = _import_carla()
    try:
        _, road_text, section_text, lane_text = str(proof["lane_id"]).rsplit(":", 3)
        expected = (int(road_text), int(section_text), int(lane_text))
    except (ValueError, KeyError) as error:
        raise RuntimeError("invalid signal-free lane identity") from error
    checked = 0
    for segment_index, (start, end) in enumerate(zip(points, points[1:])):
        dx, dy, dz = (
            float(end["x"]) - float(start["x"]),
            float(end["y"]) - float(start["y"]),
            float(end.get("z", 0.0)) - float(start.get("z", 0.0)),
        )
        distance = math.hypot(dx, dy)
        if distance <= 0.05:
            continue
        tangent = math.degrees(math.atan2(dy, dx))
        count = max(1, int(math.ceil(distance / 0.1)))
        for sample_index in range(count + 1):
            ratio = sample_index / count
            location = carla.Location(
                x=float(start["x"]) + ratio * dx,
                y=float(start["y"]) + ratio * dy,
                z=float(start.get("z", 0.0)) + ratio * dz,
            )
            waypoint = None
            for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
                waypoint = carla_map.get_waypoint(location, False, lane_type)
                if waypoint is not None:
                    break
            checked += 1
            if waypoint is None:
                raise RuntimeError(
                    f"signal-free path left exact lane at {segment_index}/{sample_index}"
                )
            observed = (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))
            error = abs((tangent - float(waypoint.transform.rotation.yaw) + 180.0) % 360.0 - 180.0)
            if observed != expected or bool(waypoint.is_junction) or error > 90.0 + 1e-6:
                raise RuntimeError(
                    f"signal-free path proof mismatch at {segment_index}/{sample_index}: "
                    f"expected={expected}, observed={observed}, junction={waypoint.is_junction}, "
                    f"heading_error={error:.3f}"
                )
    return checked


def _waypoint_for_exact_location(carla_map: Any, value: dict[str, Any]) -> Any:
    carla = _import_carla()
    location = carla.Location(
        x=float(value["x"]), y=float(value["y"]), z=float(value.get("z", 0.0)),
    )
    for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
        waypoint = carla_map.get_waypoint(
            location, project_to_road=False, lane_type=lane_type,
        )
        if waypoint is not None:
            return waypoint
    return None


def _annotate_path_yaws(
    carla_map: Any,
    points: list[dict[str, Any]],
    *,
    terminal_yaw_deg: float | None = None,
) -> list[dict[str, float]]:
    """Attach executable ego yaw to every serialized oracle point."""

    result = []
    for index, raw in enumerate(points):
        point = _point(raw)
        waypoint = _waypoint_for_exact_location(carla_map, point)
        if waypoint is not None:
            yaw = float(waypoint.transform.rotation.yaw)
        elif index > 0:
            yaw = float(result[-1]["yaw_deg"])
        elif len(points) > 1:
            yaw = math.degrees(math.atan2(
                float(points[1]["y"]) - float(raw["y"]),
                float(points[1]["x"]) - float(raw["x"]),
            ))
        else:
            raise RuntimeError("cannot annotate yaw for a one-point oracle path")
        point["yaw_deg"] = round(yaw, 3)
        result.append(point)
    if terminal_yaw_deg is not None:
        result[-1]["yaw_deg"] = round(float(terminal_yaw_deg), 3)
    return result


def _contract_speed_bands(value: Any) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    if isinstance(value, dict):
        body = value.get("speed_band")
        if isinstance(body, dict) and isinstance(body.get("min_mps"), (int, float)) and isinstance(body.get("max_mps"), (int, float)):
            result.append((float(body["min_mps"]), float(body["max_mps"])))
        for child in value.values():
            result.extend(_contract_speed_bands(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_contract_speed_bands(child))
    return result


def _speed_to_mps(raw: str, unit: str) -> float:
    unit = (unit or "m/s").strip().casefold()
    factors = {
        "m/s": 1.0, "mps": 1.0, "ms": 1.0,
        "km/h": 1.0 / 3.6, "kph": 1.0 / 3.6, "kmh": 1.0 / 3.6,
        "mph": 0.44704,
    }
    if unit not in factors:
        raise RuntimeError(f"unsupported OpenDRIVE speed unit {unit!r}")
    try:
        value = float(raw) * factors[unit]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid OpenDRIVE speed {raw!r}") from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"non-positive OpenDRIVE speed {raw!r} {unit}")
    return value


def _xodr_speed_index(xodr_path: Path) -> dict[int, dict[str, Any]]:
    root = ET.parse(xodr_path).getroot()
    output: dict[int, dict[str, Any]] = {}
    for road in root.findall("road"):
        road_id = int(road.attrib["id"])
        road_speeds = []
        for kind in road.findall("type"):
            speed = kind.find("speed")
            if speed is not None:
                road_speeds.append((
                    float(kind.attrib.get("s", 0.0)),
                    _speed_to_mps(speed.attrib.get("max", ""), speed.attrib.get("unit", "m/s")),
                ))
        sections = []
        lanes = road.find("lanes")
        if lanes is not None:
            for section in lanes.findall("laneSection"):
                lane_speeds: dict[int, list[tuple[float, float]]] = {}
                for group in ("left", "center", "right"):
                    parent = section.find(group)
                    if parent is None:
                        continue
                    for lane in parent.findall("lane"):
                        values = []
                        for speed in lane.findall("speed"):
                            values.append((
                                float(speed.attrib.get("sOffset", 0.0)),
                                _speed_to_mps(speed.attrib.get("max", ""), speed.attrib.get("unit", "m/s")),
                            ))
                        if values:
                            lane_speeds[int(lane.attrib["id"])] = sorted(values)
                sections.append((float(section.attrib.get("s", 0.0)), lane_speeds))
        output[road_id] = {"road_speeds": sorted(road_speeds), "sections": sorted(sections)}
    return output


def _waypoint_speed_limit_mps(index: dict[int, dict[str, Any]], waypoint: Any) -> float:
    road_id, lane_id, s_value = int(waypoint.road_id), int(waypoint.lane_id), float(waypoint.s)
    if road_id not in index:
        raise RuntimeError(f"speed proof road {road_id} is absent from XODR")
    record = index[road_id]
    applicable_sections = [value for value in record["sections"] if value[0] <= s_value + 1e-6]
    if applicable_sections:
        section_s, lanes = applicable_sections[-1]
        lane_values = lanes.get(lane_id, [])
        applicable_lane = [value for value in lane_values if value[0] <= s_value - section_s + 1e-6]
        if applicable_lane:
            return float(applicable_lane[-1][1])
    applicable_road = [value for value in record["road_speeds"] if value[0] <= s_value + 1e-6]
    if applicable_road:
        return float(applicable_road[-1][1])
    raise RuntimeError(f"road/lane {road_id}/{lane_id} at s={s_value:.3f} has no legal speed")


def _attach_speed_proofs(project_root: Path, pcus: list[dict[str, Any]]) -> None:
    """Freeze private, path-wide OpenDRIVE speed evidence for speed programs."""

    config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    runtime_policy = _speed_policy(project_root)
    runtime_cap = float(runtime_policy["cap_mps"])
    control_margin = float(runtime_policy["control_margin_mps"])
    xodr_root = Path(config["xodr_root"])
    town_cache: dict[str, tuple[Any, dict[int, dict[str, Any]], Path]] = {}
    carla = _import_carla()
    for pcu in pcus:
        town = pcu["town"]
        for arm in pcu["arms"]:
            bands = _contract_speed_bands(arm.get("contract"))
            if not bands:
                continue
            if town == "road1":
                raise RuntimeError(f"{pcu['id']}/{arm['id']}: road1 cannot carry a core speed proof")
            if town not in town_cache:
                xodr_path = xodr_root / f"{town}.xodr"
                carla_map, _, _ = _directed_route_context(project_root, town)
                town_cache[town] = (carla_map, _xodr_speed_index(xodr_path), xodr_path)
            carla_map, speed_index, xodr_path = town_cache[town]
            limits: list[float] = []
            sample_count = 0
            path = arm["oracle_path"]
            for first, second in zip(path, path[1:]):
                dx, dy, dz = (
                    float(second["x"]) - float(first["x"]),
                    float(second["y"]) - float(first["y"]),
                    float(second.get("z", 0.0)) - float(first.get("z", 0.0)),
                )
                count = max(1, int(math.ceil(math.hypot(dx, dy) / 0.1)))
                for sample_index in range(count + 1):
                    ratio = sample_index / count
                    location = carla.Location(
                        x=float(first["x"]) + ratio * dx,
                        y=float(first["y"]) + ratio * dy,
                        z=float(first.get("z", 0.0)) + ratio * dz,
                    )
                    waypoint = None
                    for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
                        waypoint = carla_map.get_waypoint(location, False, lane_type)
                        if waypoint is not None:
                            break
                    if waypoint is None:
                        raise RuntimeError(
                            f"{pcu['id']}/{arm['id']}: speed proof has no exact lane at sample {sample_count}"
                        )
                    limits.append(_waypoint_speed_limit_mps(speed_index, waypoint))
                    sample_count += 1
            if not limits:
                raise RuntimeError(f"{pcu['id']}/{arm['id']}: empty speed proof")
            legal = min(limits)
            effective_cap = min(legal, runtime_cap)
            for low, high in bands:
                if low > high or high > effective_cap - control_margin + 1e-6:
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: band [{low},{high}] exceeds effective "
                        f"speed cap min(XODR={legal:.6f}, CARLA={runtime_cap:.6f})-{control_margin:.3f}"
                    )
            arm["legal_speed_mps"] = round(legal, 6)
            arm["legal_speed_proof"] = {
                "method": "oracle_path_exact_waypoint_min_v1",
                "samples": sample_count,
                "lane_types": ["Driving", "Parking"],
                "sample_spacing_max_m": 0.1,
                "minimum_positive_mps": round(legal, 6),
                "town": town,
                "xodr_sha256": sha256_file(xodr_path),
            }
            arm["speed_feasibility_cap_proof"] = {
                "method": runtime_policy["method"],
                "cap_mps": round(runtime_cap, 6),
                "control_margin_mps": round(control_margin, 6),
            }


def _inside_polygon_with_tolerance(
    point: tuple[float, float], polygon: list[list[float]], tolerance_m: float,
) -> bool:
    if _polygon_contains(point, polygon):
        return True
    vertices = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
    return any(
        _point_segment_distance(point, tuple(map(float, first[:2])), tuple(map(float, second[:2])))
        <= tolerance_m
        for first, second in zip(vertices, vertices[1:] + vertices[:1])
    )


def _parking_approach_target(
    project_root: Path,
    target: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Attach one deterministic full-footprint/static-clear axle-v4 manoeuvre."""
    carla_map, _, _ = _directed_route_context(project_root, "parkinglotfish-new3")
    approach, proof, manoeuvre = _mine_parking_manoeuvre(
        project_root, target, carla_map, _waypoint_for_exact_location,
    )
    target["reverse_manoeuvre"] = manoeuvre
    return approach, proof


def _prepare_parking_route(
    project_root: Path,
    target: dict[str, Any],
    path: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    approach, proof = _parking_approach_target(project_root, target)
    target["approach_pose"] = approach
    target["approach_pose_proof"] = proof
    if not path:
        raise RuntimeError(f"slot {target['slot_number']}: empty common route")
    return [*path[:-1], *_parking_manoeuvre_path(target)]


def _common_directed_parking_pair(
    project_root: Path,
    first_target: dict[str, Any],
    second_target: dict[str, Any],
    *,
    minimum_start_m: float = 20.0,
    maximum_route_m: float = 260.0,
) -> dict[str, Any]:
    """Compile one shared-start, stored-XODR parking-v5 route pair.

    The ordinary route terminates at manoeuvre pose zero (the lane-centre
    prelead), never at the physical slot-axis approach.  A 0.05 m GRP trace
    supplies the final >=7 m construction-authoritative tail; its XODR tuple
    and full-double s are frozen before serialization.  The prelead appears
    exactly once when the collapsed, gear-free manoeuvre suffix is spliced.
    """

    approach_a, proof_a = (
        (dict(first_target["approach_pose"]), dict(first_target["approach_pose_proof"]))
        if isinstance(first_target.get("approach_pose"), dict)
        else _parking_approach_target(project_root, first_target)
    )
    approach_b, proof_b = (
        (dict(second_target["approach_pose"]), dict(second_target["approach_pose_proof"]))
        if isinstance(second_target.get("approach_pose"), dict)
        else _parking_approach_target(project_root, second_target)
    )
    first_target["approach_pose"], first_target["approach_pose_proof"] = approach_a, proof_a
    second_target["approach_pose"], second_target["approach_pose_proof"] = approach_b, proof_b
    del approach_a, approach_b, proof_a, proof_b  # slot-axis values stay target-local
    return _parking_v5_route_pair(
        project_root, first_target, second_target,
        minimum_start_m=minimum_start_m, maximum_route_m=maximum_route_m,
    )


_PARKING_V5_COMMON_START_XODR_HINTS: dict[
    tuple[str, str], tuple[int, int, float]
] = {
    # The first otherwise-stable C-027 start is a dev-used D-004 cell.  This
    # split-safe XODR candidate is the first independently audited successor.
    ("B101", "B102"): (12, 1, 10.0),
    # Preserve the independently audited D-015 empty-program common start.
    ("D402", "D403"): (3, -1, 20.0),
}


def _parking_v5_canonical_yaw(value: Any) -> float:
    result = round((float(value) + 180.0) % 360.0 - 180.0, 3)
    return 0.0 if result == 0.0 else result


def _parking_v5_prelead_waypoint(carla_map: Any, target: dict[str, Any]) -> Any:
    topology = target["reverse_manoeuvre"]["proof"]["construction_topology_proof"]
    sample = topology["samples"][0]
    waypoint = carla_map.get_waypoint_xodr(
        int(sample["road_id"]), int(sample["lane_id"]), float(sample["s_m"]),
    )
    if waypoint is None or int(waypoint.section_id) != int(sample["section_id"]):
        raise RuntimeError(
            f"slot {target['slot_number']}: stored-XODR prelead cannot reconstruct"
        )
    expected = {
        "x": round(float(waypoint.transform.location.x), 4),
        "y": round(float(waypoint.transform.location.y), 4),
        "z": round(float(waypoint.transform.location.z), 4),
        "yaw_deg": _parking_v5_canonical_yaw(waypoint.transform.rotation.yaw),
    }
    first = {
        key: target["reverse_manoeuvre"]["poses"][0][key]
        for key in ("x", "y", "z", "yaw_deg")
    }
    if first != expected:
        raise RuntimeError(
            f"slot {target['slot_number']}: serialized prelead differs from stored XODR"
        )
    return waypoint


def _parking_v5_graph_index(project_root: Path) -> tuple[
    dict[tuple[int, int, int], dict[str, Any]], dict[str, dict[str, Any]]
]:
    graph = json.loads((
        project_root / "generated/maps/lane_graphs/parkinglotfish-new3.lane_graph.json"
    ).read_text(encoding="utf-8"))
    by_key = {
        (int(row["road_id"]), int(row["section_id"]), int(row["lane_id"])): row
        for row in graph["lanes"]
    }
    return by_key, {str(row["id"]): row for row in graph["lanes"]}


def _parking_v5_fine_leg(
    project_root: Path, carla_map: Any, planner: Any, start_waypoint: Any,
    target: dict[str, Any],
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Trace and qualify a construction-authoritative fine ordinary leg."""

    from . import parking as parking_module

    target_waypoint = _parking_v5_prelead_waypoint(carla_map, target)
    by_key, _ = _parking_v5_graph_index(project_root)

    def lane_key(waypoint: Any) -> tuple[int, int, int]:
        return (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))

    def direct_successor(first: tuple[int, int, int], second: tuple[int, int, int]) -> bool:
        return str(by_key[second]["id"]) in by_key[first].get("successors", [])

    def canonical_xodr(road_id: int, lane_id: int, s_m: float) -> Any:
        waypoint = carla_map.get_waypoint_xodr(int(road_id), int(lane_id), float(s_m))
        if waypoint is None:
            raise RuntimeError("non-reconstructible stored-XODR waypoint")
        return waypoint

    def row(waypoint: Any, *, s_override: float | None = None) -> dict[str, Any]:
        requested_s = float(waypoint.s if s_override is None else s_override)
        restored = canonical_xodr(int(waypoint.road_id), int(waypoint.lane_id), requested_s)
        if int(restored.section_id) != int(waypoint.section_id):
            raise RuntimeError("stored-XODR waypoint section differs")
        return {
            "x": round(float(restored.transform.location.x), 4),
            "y": round(float(restored.transform.location.y), 4),
            "z": round(float(restored.transform.location.z), 4),
            "yaw_deg": _parking_v5_canonical_yaw(restored.transform.rotation.yaw),
            "_lane": lane_key(restored), "_s_m": requested_s,
        }

    traced = planner.trace_route(
        start_waypoint.transform.location, target_waypoint.transform.location,
    )
    if not traced:
        raise RuntimeError(f"slot {target['slot_number']}: fine GRP route is empty")
    raw = [row(start_waypoint)]
    traced_waypoints = [waypoint for waypoint, _ in traced]
    index = 0
    while index < len(traced_waypoints):
        waypoint = traced_waypoints[index]
        try:
            raw.append(row(waypoint))
            index += 1
            continue
        except RuntimeError:
            pass
        # CARLA can emit an exact lane-boundary s which get_waypoint_xodr
        # refuses.  Drop that non-authoritative point; when the canonical
        # predecessor endpoint keeps both gaps <=0.1m, insert it instead.
        while index < len(traced_waypoints):
            candidate = traced_waypoints[index]
            try:
                following = row(candidate)
                break
            except RuntimeError:
                index += 1
        if index >= len(traced_waypoints):
            raise RuntimeError(f"slot {target['slot_number']}: fine route ends at invalid boundary")
        previous_key = tuple(raw[-1]["_lane"])
        following_key = tuple(following["_lane"])
        if previous_key == following_key or not direct_successor(previous_key, following_key):
            raise RuntimeError(f"slot {target['slot_number']}: invalid boundary transition")
        endpoint_s = float(by_key[previous_key]["samples"][-1]["s"])
        boundary = row(canonical_xodr(previous_key[0], previous_key[2], endpoint_s))
        if (math.hypot(boundary["x"] - raw[-1]["x"], boundary["y"] - raw[-1]["y"])
                <= 0.1 + 1e-9
                and math.hypot(following["x"] - boundary["x"], following["y"] - boundary["y"])
                <= 0.1 + 1e-9):
            raw.append(boundary)
        # Do not consume following: the next iteration appends it exactly.

    target_sample = target["reverse_manoeuvre"]["proof"]["construction_topology_proof"]["samples"][0]
    raw.append(row(target_waypoint, s_override=float(target_sample["s_m"])))
    rows: list[dict[str, Any]] = []
    for value in raw:
        if rows and all(value[name] == rows[-1][name] for name in ("x", "y", "z")):
            rows[-1] = value
        else:
            rows.append(value)
    expected_terminal = {
        key: target["reverse_manoeuvre"]["poses"][0][key]
        for key in ("x", "y", "z", "yaw_deg")
    }
    rows[-1].update(expected_terminal)
    public = [
        {key: value[key] for key in ("x", "y", "z", "yaw_deg")}
        for value in rows
    ]
    directed = _validate_directed_polyline(
        carla_map, public, sample_spacing_m=0.1, maximum_heading_error_deg=3.0,
    )
    if not directed["valid"]:
        raise RuntimeError(f"slot {target['slot_number']}: fine route is not directed: {directed}")

    tail_first = len(rows) - 1
    tail_length = 0.0
    while tail_first > 0 and tail_length < 7.0:
        tail_length += math.dist(
            tuple(float(rows[tail_first][name]) for name in ("x", "y", "z")),
            tuple(float(rows[tail_first - 1][name]) for name in ("x", "y", "z")),
        )
        tail_first -= 1
    if tail_length < 7.0 - 1e-9:
        raise RuntimeError(f"slot {target['slot_number']}: fine route lacks 7m dense tail")
    tail_rows = rows[tail_first:]
    for first, second in zip(tail_rows, tail_rows[1:]):
        gap = math.dist(
            tuple(float(first[name]) for name in ("x", "y", "z")),
            tuple(float(second[name]) for name in ("x", "y", "z")),
        )
        if not 1e-8 < gap <= 0.1 + 1e-9:
            raise RuntimeError(f"slot {target['slot_number']}: dense-tail spacing differs")
    topology = parking_module._construction_topology_proof(
        parking_module._geometry(project_root), carla_map,
        public[tail_first:-1],
        [
            {"road_id": int(value["_lane"][0]),
             "section_id": int(value["_lane"][1]),
             "lane_id": int(value["_lane"][2]), "s_m": float(value["_s_m"])}
            for value in tail_rows[:-1]
        ],
        pose_index_space="arm.oracle_path", first_pose_index=tail_first,
    )
    tail_poses = [dict(value, gear="forward") for value in public[tail_first:]]
    context = parking_module._geometry(project_root)
    remote_slot = [
        [-10000.0, -10000.0], [-9999.0, -10000.0],
        [-9999.0, -9999.0], [-10000.0, -9999.0], [-10000.0, -10000.0],
    ]
    if not parking_module._exact_map_footprint_audit(
        context, carla_map, tail_poses, remote_slot,
    ):
        raise RuntimeError(f"slot {target['slot_number']}: dense-tail exact OBB audit fails")
    overlap = parking_module._overlap_index_sets(context, tail_poses)
    for pose, indices in zip(tail_poses, overlap):
        footprint = parking_module._footprint(
            context, float(pose["x"]), float(pose["y"]),
            math.radians(float(pose["yaw_deg"])),
        )
        if any(footprint.distance(context["obstacles"][item]) < 0.2 - 1e-9
               for item in indices):
            raise RuntimeError(f"slot {target['slot_number']}: dense-tail static clearance fails")
    # Compact only the unprotected ordinary prefix.  Retain the common start
    # and choose the farthest subsequent point whose direct gap is <=4m;
    # append the proof-bound >=7m tail byte-for-byte.  Topology indices are
    # shifted only after this deterministic compaction; its stored samples and
    # transforms are unchanged except for their oracle_path index space.
    prefix = public[:tail_first]
    if not prefix:
        raise RuntimeError(f"slot {target['slot_number']}: ordinary prefix is empty")
    # Build a backwards-reachable DAG over the authoritative fine route.  An
    # edge is admitted only when it preserves the serialized source yaw and a
    # complete <=0.1 m directed/exact-lane replay.  Computing reachability
    # backwards prevents a locally farthest greedy choice from entering a
    # dead end.  The source-arclength window is deliberately no larger than
    # the 4 m serialized gap bound; we never rely on a bend-cutting shortcut.
    reachable = [False] * len(prefix)
    successor: list[int | None] = [None] * len(prefix)
    reachable[-1] = True
    for cursor in range(len(prefix) - 2, -1, -1):
        candidates: list[int] = []
        walked = 0.0
        following = cursor + 1
        while following < len(prefix):
            walked += math.dist(
                tuple(float(prefix[following - 1][name]) for name in ("x", "y", "z")),
                tuple(float(prefix[following][name]) for name in ("x", "y", "z")),
            )
            if walked > 4.0 + 1e-9:
                break
            if reachable[following]:
                candidates.append(following)
            following += 1
        for following in reversed(candidates):
            candidate = prefix[following]
            direct_gap = math.dist(
                tuple(float(prefix[cursor][name]) for name in ("x", "y", "z")),
                tuple(float(candidate[name]) for name in ("x", "y", "z")),
            )
            if not 0.01 + 1e-9 < direct_gap <= 4.0 + 1e-9:
                continue
            chord_yaw = math.degrees(math.atan2(
                float(candidate["y"]) - float(prefix[cursor]["y"]),
                float(candidate["x"]) - float(prefix[cursor]["x"]),
            ))
            yaw_error = abs((
                chord_yaw - float(prefix[cursor]["yaw_deg"]) + 180.0
            ) % 360.0 - 180.0)
            if (yaw_error <= 3.0 + 1e-9 and _validate_directed_polyline(
                    carla_map, [prefix[cursor], candidate],
                    sample_spacing_m=0.1, maximum_heading_error_deg=90.0,
            )["valid"]):
                reachable[cursor] = True
                successor[cursor] = following
                break
    if not reachable[0]:
        raise RuntimeError(
            f"slot {target['slot_number']}: ordinary prefix compaction DAG has no "
            "directed path from the common start"
        )
    compact_prefix = [prefix[0]]
    cursor = 0
    while cursor < len(prefix) - 1:
        following = successor[cursor]
        if following is None:
            raise RuntimeError(
                f"slot {target['slot_number']}: ordinary prefix reachability chain breaks "
                f"at source index {cursor}"
            )
        compact_prefix.append(prefix[following])
        cursor = following
    compacted = [*compact_prefix, *public[tail_first:]]
    compacted_tail_first = len(compact_prefix)
    index_delta = compacted_tail_first - tail_first
    topology = json.loads(json.dumps(topology))
    topology["first_pose_index"] += index_delta
    topology["last_pose_index"] += index_delta
    for sample in topology["samples"]:
        sample["pose_index"] += index_delta
    topology["samples_sha256"] = stable_hash(topology["samples"])
    transforms = []
    for sample in topology["samples"]:
        waypoint = carla_map.get_waypoint_xodr(
            int(sample["road_id"]), int(sample["lane_id"]), float(sample["s_m"]),
        )
        if waypoint is None:
            raise RuntimeError(f"slot {target['slot_number']}: shifted tail sample cannot reconstruct")
        transforms.append({
            "pose_index": int(sample["pose_index"]),
            "x": round(float(waypoint.transform.location.x), 4),
            "y": round(float(waypoint.transform.location.y), 4),
            "z": round(float(waypoint.transform.location.z), 4),
            "yaw_deg": _parking_v5_canonical_yaw(waypoint.transform.rotation.yaw),
        })
    topology["canonical_waypoint_transforms_sha256"] = stable_hash(transforms)
    for run in topology["runs"]:
        run["first_pose_index"] += index_delta
        run["last_pose_index"] += index_delta
    return compacted, {
        "tail_start_index": compacted_tail_first,
        "ordinary_tail_topology_proof": topology,
        "route_length_m": _distance(compacted),
    }


def _parking_v5_materialize_route(
    ordinary: list[dict[str, Any]], ordinary_proof: dict[str, Any],
    target: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    explicit = _parking_manoeuvre_path(target)
    if ordinary[-1] != explicit[0]:
        raise RuntimeError(f"slot {target['slot_number']}: ordinary route misses prelead")
    path = [*ordinary[:-1], *explicit]
    suffix_start = len(ordinary) - 1
    tail_start = int(ordinary_proof["tail_start_index"])
    topology = ordinary_proof["ordinary_tail_topology_proof"]
    if topology["last_pose_index"] != suffix_start - 1:
        raise RuntimeError(f"slot {target['slot_number']}: ordinary topology range differs")
    route_proof = {
        "schema_version": "language2drive.parking_v5_route_proof.v1",
        "method": "oracle_path_stored_xodr_tail_exact_suffix_v1",
        "oracle_path_sha256": stable_hash(path),
        "ordinary_tail_start_index": tail_start,
        "explicit_suffix_start_index": suffix_start,
        "ordinary_tail_sha256": stable_hash(path[tail_start:suffix_start + 1]),
        "explicit_suffix_sha256": stable_hash(path[suffix_start:]),
        "target_prelead_sample_sha256": stable_hash(
            target["reverse_manoeuvre"]["proof"]["construction_topology_proof"]["samples"][0]
        ),
        "ordinary_tail_topology_proof": topology,
    }
    return path, route_proof


def _parking_v5_path_length_xyz(path: list[dict[str, Any]]) -> float:
    """Return the once-counted three-dimensional serialized path length."""

    return sum(math.dist(
        tuple(float(first[name]) for name in ("x", "y", "z")),
        tuple(float(second[name]) for name in ("x", "y", "z")),
    ) for first, second in zip(path, path[1:]))


def _parking_v5_accept_materialized_route(
    project_root: Path, carla_map: Any, target: dict[str, Any],
    path: list[dict[str, Any]], proof: dict[str, Any],
) -> float:
    """Requalify every derived v5 invariant on the final serialized route.

    Fine-leg qualification happens before the explicit manoeuvre is spliced
    and before the ordinary prefix is compacted.  It therefore cannot by
    itself prove the two final chord seams or the final proof/hash index
    space.  This post-materialization gate intentionally duplicates those
    safety-critical derived checks inside the producer.  A rejected arm makes
    the pair search try its next common start; the independent schema remains
    the release authority and repeats the same checks.
    """

    from . import parking as parking_module

    if not isinstance(path, list) or not path:
        raise RuntimeError(f"slot {target['slot_number']}: materialized path is empty")
    pose_keys = {"x", "y", "z", "yaw_deg"}
    for index, pose in enumerate(path):
        if not isinstance(pose, dict) or set(pose) != pose_keys:
            raise RuntimeError(
                f"slot {target['slot_number']}: path pose {index} is not exact4"
            )
        for field in pose_keys:
            value = pose[field]
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(float(value))):
                raise RuntimeError(
                    f"slot {target['slot_number']}: path pose {index}.{field} "
                    "is not a finite JSON number"
                )
        if (float(pose["x"]) != round(float(pose["x"]), 4)
                or float(pose["y"]) != round(float(pose["y"]), 4)
                or float(pose["z"]) != round(float(pose["z"]), 4)
                or float(pose["yaw_deg"])
                != _parking_v5_canonical_yaw(pose["yaw_deg"])):
            raise RuntimeError(
                f"slot {target['slot_number']}: path pose {index} precision/yaw differs"
            )
    required = {
        "schema_version", "method", "oracle_path_sha256",
        "ordinary_tail_start_index", "explicit_suffix_start_index",
        "ordinary_tail_sha256", "explicit_suffix_sha256",
        "target_prelead_sample_sha256", "ordinary_tail_topology_proof",
    }
    if not isinstance(proof, dict) or set(proof) != required:
        raise RuntimeError(f"slot {target['slot_number']}: route proof exact9 differs")
    if (proof["schema_version"] != "language2drive.parking_v5_route_proof.v1"
            or proof["method"] != "oracle_path_stored_xodr_tail_exact_suffix_v1"):
        raise RuntimeError(f"slot {target['slot_number']}: route proof schema/method differs")
    for field in (
        "oracle_path_sha256", "ordinary_tail_sha256", "explicit_suffix_sha256",
        "target_prelead_sample_sha256",
    ):
        value = proof[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError(
                f"slot {target['slot_number']}: route proof {field} is not SHA256"
            )
    tail = proof["ordinary_tail_start_index"]
    suffix = proof["explicit_suffix_start_index"]
    if (not isinstance(tail, int) or isinstance(tail, bool)
            or not isinstance(suffix, int) or isinstance(suffix, bool)
            or not 0 < tail < suffix < len(path)):
        raise RuntimeError(f"slot {target['slot_number']}: route proof index order differs")
    explicit = _parking_manoeuvre_path(target)
    topology = proof["ordinary_tail_topology_proof"]
    topology_keys = {
        "schema_version", "method", "pose_index_space", "first_pose_index",
        "last_pose_index", "samples", "samples_sha256",
        "canonical_waypoint_transforms_sha256", "runs", "transitions",
    }
    sample_keys = {"pose_index", "road_id", "section_id", "lane_id", "s_m"}
    run_keys = {
        "first_pose_index", "last_pose_index", "lane_graph_id",
        "travel_s_direction",
    }
    transition_keys = {"from_run_index", "to_run_index"}

    def strict_index(value: Any, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"slot {target['slot_number']}: {label} is not a strict index")
        return value

    def strict_integer(value: Any, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise RuntimeError(f"slot {target['slot_number']}: {label} is not a strict integer")
        return value

    if (not isinstance(topology, dict) or set(topology) != topology_keys
            or topology["schema_version"]
            != "language2drive.parking_v5_stored_xodr_topology.v1"
            or topology["method"] != "stored_xodr_construction_waypoint_sequence_v1"
            or topology["pose_index_space"] != "arm.oracle_path"):
        raise RuntimeError(f"slot {target['slot_number']}: tail topology root differs")
    topology_first = strict_index(topology["first_pose_index"], "topology.first")
    topology_last = strict_index(topology["last_pose_index"], "topology.last")
    for field in ("samples_sha256", "canonical_waypoint_transforms_sha256"):
        value = topology[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise RuntimeError(
                f"slot {target['slot_number']}: tail topology {field} is not SHA256"
            )
    samples = topology["samples"]
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"slot {target['slot_number']}: tail topology samples are empty")
    for offset, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != sample_keys:
            raise RuntimeError(
                f"slot {target['slot_number']}: tail sample {offset} is not exact5"
            )
        pose_index = strict_index(sample["pose_index"], f"sample {offset}.pose_index")
        strict_index(sample["road_id"], f"sample {offset}.road_id")
        strict_index(sample["section_id"], f"sample {offset}.section_id")
        strict_integer(sample["lane_id"], f"sample {offset}.lane_id")
        if sample["lane_id"] == 0:
            raise RuntimeError(f"slot {target['slot_number']}: tail sample lane is zero")
        s_m = sample["s_m"]
        if (not isinstance(s_m, (int, float)) or isinstance(s_m, bool)
                or not math.isfinite(float(s_m))):
            raise RuntimeError(f"slot {target['slot_number']}: tail sample s is nonfinite")
        if pose_index != topology_first + offset:
            raise RuntimeError(f"slot {target['slot_number']}: tail sample indices differ")
    if topology["samples_sha256"] != stable_hash(samples):
        raise RuntimeError(f"slot {target['slot_number']}: tail samples hash differs")
    runs = topology["runs"]
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"slot {target['slot_number']}: tail topology runs are empty")
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != run_keys:
            raise RuntimeError(f"slot {target['slot_number']}: tail run {index} is not exact4")
        strict_index(run["first_pose_index"], f"run {index}.first")
        strict_index(run["last_pose_index"], f"run {index}.last")
        if (not isinstance(run["lane_graph_id"], str)
                or not isinstance(run["travel_s_direction"], int)
                or isinstance(run["travel_s_direction"], bool)
                or run["travel_s_direction"] not in (-1, 1)):
            raise RuntimeError(f"slot {target['slot_number']}: tail run {index} types differ")
    transitions = topology["transitions"]
    if not isinstance(transitions, list):
        raise RuntimeError(f"slot {target['slot_number']}: tail transitions are not a list")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict) or set(transition) != transition_keys:
            raise RuntimeError(
                f"slot {target['slot_number']}: tail transition {index} is not exact2"
            )
        strict_index(transition["from_run_index"], f"transition {index}.from")
        strict_index(transition["to_run_index"], f"transition {index}.to")
    target_topology = target["reverse_manoeuvre"]["proof"]["construction_topology_proof"]
    if (path[suffix:] != explicit
            or proof["oracle_path_sha256"] != stable_hash(path)
            or proof["ordinary_tail_sha256"] != stable_hash(path[tail:suffix + 1])
            or proof["explicit_suffix_sha256"] != stable_hash(path[suffix:])
            or proof["target_prelead_sample_sha256"]
            != stable_hash(target_topology["samples"][0])
            or topology_first != tail
            or topology_last != suffix - 1):
        raise RuntimeError(f"slot {target['slot_number']}: materialized route binding differs")

    context = parking_module._geometry(project_root)
    if len(samples) != suffix - tail:
        raise RuntimeError(f"slot {target['slot_number']}: tail topology inventory differs")
    reconstructed = parking_module._construction_topology_proof(
        context, carla_map, path[tail:suffix], samples,
        first_pose_index=tail, pose_index_space="arm.oracle_path",
        require_raw_transform=True,
    )
    if stable_hash(reconstructed) != stable_hash(topology):
        raise RuntimeError(f"slot {target['slot_number']}: tail stored-XODR replay differs")

    # The compact prefix has no duplicate XODR proof authority, but every
    # serialized point and every <=0.1 m interpolation sample must remain on
    # exact Driving/Parking surface and directed.  Its coarse serialization is
    # bounded to 4 m and preserves each source waypoint's forward yaw.
    import carla

    ordinary = path[:suffix + 1]
    for index, (first, second) in enumerate(zip(ordinary, ordinary[1:])):
        gap = math.dist(
            tuple(float(first[name]) for name in ("x", "y", "z")),
            tuple(float(second[name]) for name in ("x", "y", "z")),
        )
        if gap > 4.0 + 1e-9:
            raise RuntimeError(
                f"slot {target['slot_number']}: compact ordinary gap exceeds 4m at {index}"
            )
    for index, pose in enumerate(ordinary):
        location = carla.Location(
            x=float(pose["x"]), y=float(pose["y"]), z=float(pose["z"]),
        )
        if not any(carla_map.get_waypoint(location, False, lane_type) is not None
                   for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking)):
            raise RuntimeError(
                f"slot {target['slot_number']}: compact ordinary point {index} is off lane"
            )
    directed = _validate_directed_polyline(
        carla_map, ordinary, sample_spacing_m=0.1,
        maximum_heading_error_deg=90.0,
    )
    if not directed["valid"]:
        raise RuntimeError(
            f"slot {target['slot_number']}: compact ordinary directed replay fails: {directed}"
        )
    for index, pose in enumerate(ordinary[:-1]):
        following = next((candidate for candidate in ordinary[index + 1:] if math.dist(
            tuple(float(pose[name]) for name in ("x", "y", "z")),
            tuple(float(candidate[name]) for name in ("x", "y", "z")),
        ) > 0.01 + 1e-9), None)
        if following is None:
            continue
        chord = math.degrees(math.atan2(
            float(following["y"]) - float(pose["y"]),
            float(following["x"]) - float(pose["x"]),
        ))
        if abs(_angle_difference_degrees(float(pose["yaw_deg"]), chord)) > 3.0 + 1e-9:
            raise RuntimeError(
                f"slot {target['slot_number']}: compact ordinary serialized yaw differs "
                f"at {index}"
            )

    dense = path[tail:suffix + 1]
    dense_gaps = [math.dist(
        tuple(float(first[name]) for name in ("x", "y", "z")),
        tuple(float(second[name]) for name in ("x", "y", "z")),
    ) for first, second in zip(dense, dense[1:])]
    if (any(not 1e-8 < gap <= 0.1 + 1e-9 for gap in dense_gaps)
            or sum(dense_gaps) < 7.0 - 1e-9):
        raise RuntimeError(f"slot {target['slot_number']}: dense tail spacing/length differs")
    prefix_gap = math.dist(
        tuple(float(path[tail - 1][name]) for name in ("x", "y", "z")),
        tuple(float(path[tail][name]) for name in ("x", "y", "z")),
    )
    cross_gap = math.dist(
        tuple(float(path[suffix - 1][name]) for name in ("x", "y", "z")),
        tuple(float(path[suffix][name]) for name in ("x", "y", "z")),
    )
    if prefix_gap > 1.0 + 1e-9 or not 1e-8 < cross_gap <= 0.1 + 1e-9:
        raise RuntimeError(f"slot {target['slot_number']}: protected route splice gap differs")

    # Bind the ordinary tail's final construction lane to the target's sample
    # zero without using ambiguous inverse lookup at merges.
    ordinary_last = samples[-1]
    target_first = target_topology["samples"][0]
    by_key, by_id, directions = parking_module._lane_graph_index(context)
    last_key = tuple(int(ordinary_last[name]) for name in ("road_id", "section_id", "lane_id"))
    first_key = tuple(int(target_first[name]) for name in ("road_id", "section_id", "lane_id"))
    last_id, first_id = str(by_key[last_key]["id"]), str(by_key[first_key]["id"])
    if last_id == first_id:
        if ((float(target_first["s_m"]) - float(ordinary_last["s_m"]))
                * directions[last_id] <= 1e-8):
            raise RuntimeError(f"slot {target['slot_number']}: tail-to-prelead s reverses")
    elif first_id not in by_id[last_id].get("successors", []):
        raise RuntimeError(f"slot {target['slot_number']}: tail-to-prelead is not a successor")
    last_waypoint = carla_map.get_waypoint_xodr(
        int(ordinary_last["road_id"]), int(ordinary_last["lane_id"]),
        float(ordinary_last["s_m"]),
    )
    target_waypoint = carla_map.get_waypoint_xodr(
        int(target_first["road_id"]), int(target_first["lane_id"]),
        float(target_first["s_m"]),
    )
    cross_chord = math.degrees(math.atan2(
        float(path[suffix]["y"]) - float(path[suffix - 1]["y"]),
        float(path[suffix]["x"]) - float(path[suffix - 1]["x"]),
    ))
    if (last_waypoint is None or target_waypoint is None
            or max(
                abs(_angle_difference_degrees(
                    cross_chord, float(last_waypoint.transform.rotation.yaw),
                )),
                abs(_angle_difference_degrees(
                    cross_chord, float(target_waypoint.transform.rotation.yaw),
                )),
            ) > 3.0 + 1e-9):
        raise RuntimeError(f"slot {target['slot_number']}: tail-to-prelead XODR yaw differs")

    def nonzero_chord(values: list[dict[str, Any]], *, reverse: bool = False) -> float:
        pairs = list(zip(values, values[1:]))
        if reverse:
            pairs.reverse()
        for first, second in pairs:
            if math.dist(
                tuple(float(first[name]) for name in ("x", "y", "z")),
                tuple(float(second[name]) for name in ("x", "y", "z")),
            ) > 1e-6:
                return math.degrees(math.atan2(
                    float(second["y"]) - float(first["y"]),
                    float(second["x"]) - float(first["x"]),
                ))
        raise RuntimeError(f"slot {target['slot_number']}: protected seam has no chord")

    prefix_chord = nonzero_chord(path[:tail + 1], reverse=True)
    dense_chord = nonzero_chord(dense)
    explicit_chord = nonzero_chord(path[suffix:])
    if (abs(_angle_difference_degrees(prefix_chord, dense_chord)) > 3.0 + 1e-9
            or abs(_angle_difference_degrees(cross_chord, explicit_chord)) > 3.0 + 1e-9):
        raise RuntimeError(f"slot {target['slot_number']}: protected parking-v5 seam differs")

    dense_forward = [dict(pose, gear="forward") for pose in dense]
    remote_polygon = [
        [1_000_000.0, 1_000_000.0], [1_000_001.0, 1_000_000.0],
        [1_000_000.0, 1_000_001.0], [1_000_000.0, 1_000_000.0],
    ]
    if not parking_module._exact_map_footprint_audit(
        context, carla_map, dense_forward, remote_polygon,
    ):
        raise RuntimeError(f"slot {target['slot_number']}: dense tail exact OBB audit fails")
    overlap = parking_module._overlap_index_sets(context, dense_forward)
    for pose, indices in zip(dense_forward, overlap):
        footprint = parking_module._footprint(
            context, float(pose["x"]), float(pose["y"]),
            math.radians(float(pose["yaw_deg"])),
        )
        if any(footprint.distance(context["obstacles"][index]) < 0.2 - 1e-9
               for index in indices):
            raise RuntimeError(f"slot {target['slot_number']}: dense tail static clearance fails")

    actual_length = _parking_v5_path_length_xyz(path)
    if not math.isfinite(actual_length):
        raise RuntimeError(f"slot {target['slot_number']}: materialized route length is nonfinite")
    return actual_length


def _parking_v5_route_pair(
    project_root: Path, first_target: dict[str, Any], second_target: dict[str, Any],
    *, minimum_start_m: float, maximum_route_m: float,
) -> dict[str, Any]:
    if (not isinstance(minimum_start_m, (int, float))
            or isinstance(minimum_start_m, bool)
            or not math.isfinite(float(minimum_start_m))
            or not isinstance(maximum_route_m, (int, float))
            or isinstance(maximum_route_m, bool)
            or not math.isfinite(float(maximum_route_m))
            or not 0.0 < float(minimum_start_m) < float(maximum_route_m)):
        raise ValueError("parking-v5 route minimum/maximum bounds are invalid")
    carla_map, coarse_planner, candidates = _directed_route_context(
        project_root, "parkinglotfish-new3",
    )
    python_api = (Path(__file__).resolve().parents[2] / "carla_runtime/PythonAPI/carla")
    if str(python_api) not in sys.path:
        sys.path.insert(0, str(python_api))
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    targets = (first_target, second_target)
    target_waypoints = tuple(_parking_v5_prelead_waypoint(carla_map, value) for value in targets)
    pair_key = tuple(sorted(str(value["slot_number"]) for value in targets))
    ranked: list[Any] = []
    hint = _PARKING_V5_COMMON_START_XODR_HINTS.get(pair_key)
    if hint is not None:
        if (not isinstance(hint, tuple) or len(hint) != 3
                or not isinstance(hint[0], int) or isinstance(hint[0], bool)
                or not isinstance(hint[1], int) or isinstance(hint[1], bool)
                or hint[1] == 0
                or not isinstance(hint[2], (int, float)) or isinstance(hint[2], bool)
                or not math.isfinite(float(hint[2]))):
            raise RuntimeError(f"parking-v5 common-start hint {pair_key} is malformed")
        waypoint = carla_map.get_waypoint_xodr(hint[0], hint[1], hint[2])
        if waypoint is None:
            raise RuntimeError(
                f"parking-v5 common-start hint {pair_key} cannot reconstruct"
            )
        distances = [waypoint.transform.location.distance(item.transform.location)
                     for item in target_waypoints]
        if any(not math.isfinite(float(value)) for value in distances):
            raise RuntimeError(
                f"parking-v5 common-start hint {pair_key} has nonfinite distance"
            )
        # A geometrically valid hint is only a stable ordering preference.  If
        # the caller's stricter distance window or a junction excludes it,
        # skip it and continue through the ordinary nonjunction candidate
        # pool; do not turn an inadmissible hint into a global failure.
        if (not waypoint.is_junction and min(distances) >= minimum_start_m
                and max(distances) <= maximum_route_m):
            ranked.append(waypoint)
    remainder = []
    for candidate in candidates:
        distances = [candidate.transform.location.distance(item.transform.location)
                     for item in target_waypoints]
        if (candidate.is_junction
                or any(not math.isfinite(float(value)) for value in distances)
                or min(distances) < minimum_start_m or max(distances) > maximum_route_m):
            continue
        remainder.append((_candidate_distance_key(candidate, distances), candidate))
    for _, candidate in sorted(remainder, key=lambda value: value[0]):
        if all(_waypoint_key(candidate) != _waypoint_key(value) for value in ranked):
            ranked.append(candidate)

    fine_planner = GlobalRoutePlanner(carla_map, 0.05)
    failures = []
    for start in ranked:
        coarse = [
            _directed_leg("parkinglotfish-new3", carla_map, coarse_planner, start, target_wp)
            for target_wp in target_waypoints
        ]
        if any(value is None for value in coarse):
            continue
        if any(float(value[1]) > maximum_route_m for value in coarse if value is not None):
            continue
        try:
            compiled = [
                _parking_v5_fine_leg(project_root, carla_map, fine_planner, start, target)
                for target in targets
            ]
            materialized = [
                _parking_v5_materialize_route(ordinary, proof, target)
                for (ordinary, proof), target in zip(compiled, targets)
            ]
            paths = [value[0] for value in materialized]
            if paths[0][0] != paths[1][0]:
                raise RuntimeError(
                    f"parking-v5 {pair_key} lost common serialized start"
                )
            qualified_lengths = [
                _parking_v5_accept_materialized_route(
                    project_root, carla_map, target, path, proof,
                )
                for target, (path, proof) in zip(targets, materialized)
            ]
        except (RuntimeError, ValueError) as error:
            failures.append(str(error))
            continue
        return {
            "start": dict(paths[0][0]),
            "yaw_deg": float(paths[0][0]["yaw_deg"]),
            "paths": paths,
            "route_proofs": [value[1] for value in materialized],
            "canonical_lengths_m": [round(value, 3) for value in qualified_lengths],
            "candidate_policy": "stored_xodr_topk_first_joint_complete_v1",
        }
    raise RuntimeError(
        f"parking-v5 pair {pair_key} has no joint stored-XODR route; "
        f"failures={failures[-3:]}"
    )


def _common_directed_route_pair(project_root: Path, town: str,
                                first_target: dict[str, float], second_target: dict[str, float],
                                *, minimum_start_m: float = 20.0,
                                maximum_route_m: float = 260.0,
                                include_target_connector: bool = True) -> dict[str, Any]:
    """Find one directed spawn with local legal routes to two target centers.

    Parking-space centers are off the driving lane.  The returned path includes
    the projected terminal lane waypoint followed by the short parking
    connector, preventing a route interpolator from cutting diagonally across
    the lot.  Candidate selection is deterministic and rejects long loops.
    """

    carla = _import_carla()
    carla_map, planner, candidates = _directed_route_context(project_root, town)
    targets = [carla.Location(x=float(value["x"]), y=float(value["y"]), z=float(value.get("z", 0)))
               for value in (first_target, second_target)]
    projected = [carla_map.get_waypoint(value, project_to_road=True, lane_type=carla.LaneType.Driving)
                 for value in targets]
    if any(item is None for item in projected):
        raise RuntimeError(f"{town}: target cannot be projected to a driving lane")
    connectors = [item.transform.location.distance(target) for item, target in zip(projected, targets)]
    if max(connectors) > 12.0:
        raise RuntimeError(f"{town}: target-to-lane connector exceeds 12m: {connectors}")

    ranked_candidates = []
    for candidate in candidates:
        direct = [candidate.transform.location.distance(item.transform.location) for item in projected]
        if min(direct) < minimum_start_m or max(direct) > maximum_route_m:
            continue
        ranked_candidates.append((_candidate_distance_key(candidate, direct), candidate, direct))
    ranked_candidates.sort(key=lambda row: row[0])

    for _, candidate, direct in ranked_candidates:
        lengths, route_rows = [], []
        valid = True
        for target, target_waypoint, connector, euclidean in zip(targets, projected, connectors, direct):
            leg = _directed_leg(town, carla_map, planner, candidate, target_waypoint)
            if leg is None:
                valid = False; break
            locations, lane_length, _ = leg
            total = lane_length + (connector if include_target_connector else 0.0)
            if lane_length < minimum_start_m - 2.0 or total > maximum_route_m:
                valid = False; break
            if total > euclidean * 1.8 + 25.0:
                valid = False; break
            lengths.append(total)
            points = [_point({"x": value.x, "y": value.y, "z": value.z}) for value in locations]
            exact_target = _point({"x": target.x, "y": target.y, "z": target.z})
            if include_target_connector and math.dist((points[-1]["x"], points[-1]["y"]),
                                                      (exact_target["x"], exact_target["y"])) > 0.05:
                points.append(exact_target)
            route_rows.append(points)
        if valid:
            exact_start = _point({
                "x": candidate.transform.location.x, "y": candidate.transform.location.y,
                "z": candidate.transform.location.z,
            })
            for path in route_rows:
                path[0] = dict(exact_start)
            return {
                "start": exact_start,
                "yaw_deg": round(float(candidate.transform.rotation.yaw), 3),
                "paths": route_rows,
                "canonical_lengths_m": [round(float(value), 3) for value in lengths],
                "projected_terminal_lane": [
                    _point({"x": waypoint.transform.location.x,
                            "y": waypoint.transform.location.y,
                            "z": waypoint.transform.location.z})
                    for waypoint in projected
                ],
                "candidate_policy": "first_complete_proof_after_stable_distance_lane_sort_v1",
            }
    raise RuntimeError(f"{town}: no common local directed start for targets {first_target} / {second_target}")


def _common_directed_program_pair(project_root: Path, town: str,
                                  first_program: list[dict[str, float]],
                                  second_program: list[dict[str, float]],
                                  *, gate_radius_m: float = 5.0,
                                  minimum_start_m: float = 15.0,
                                  maximum_program_m: float = 900.0,
                                  maximum_leg_factor: float = 6.0,
                                  forbidden_spawn_tokens: set[str] | frozenset[str] | None = None,
                                  ) -> dict[str, Any]:
    """Compile two ordered gate programs from one directed CARLA spawn."""

    carla = _import_carla()
    forbidden = frozenset() if forbidden_spawn_tokens is None else forbidden_spawn_tokens
    if (not isinstance(forbidden, (set, frozenset))
            or any(not isinstance(value, str) for value in forbidden)):
        raise ValueError("forbidden spawn tokens must be a set of strings")
    carla_map, planner, candidates = _directed_route_context(project_root, town)
    raw_programs = [first_program, second_program]
    locations = [[carla.Location(x=float(value["x"]), y=float(value["y"]), z=float(value.get("z", 0)))
                  for value in program] for program in raw_programs]
    projected = [[carla_map.get_waypoint(value, project_to_road=True, lane_type=carla.LaneType.Driving)
                  for value in program] for program in locations]
    for raw, waypoints in zip(locations, projected):
        if any(item is None for item in waypoints):
            raise RuntimeError(f"{town}: program target cannot be projected")
        connector_distances = [item.transform.location.distance(target) for item, target in zip(waypoints, raw)]
        if max(connector_distances) > gate_radius_m:
            raise RuntimeError(f"{town}: gate outside radius after lane projection: {connector_distances}")

    # Inter-stage legs are independent of the shared spawn candidate.  Prove
    # them once before scanning hundreds of starts; an impossible A->B or
    # B->A suffix otherwise makes every candidate repeat the same expensive
    # negative result before the pair can be rejected.
    for program_index, program in enumerate(projected):
        for stage_index in range(1, len(program)):
            current, target = program[stage_index - 1], program[stage_index]
            leg_result = _directed_leg(town, carla_map, planner, current, target)
            if leg_result is None:
                raise RuntimeError(
                    f"{town}: no directed inter-stage leg {stage_index - 1}->{stage_index} "
                    f"for program {raw_programs[program_index]}"
                )
            leg, leg_length, _ = leg_result
            direct = current.transform.location.distance(target.transform.location)
            if leg_length > direct * maximum_leg_factor + 35.0:
                raise RuntimeError(
                    f"{town}: inter-stage leg exceeds detour policy for program "
                    f"{raw_programs[program_index]}"
                )
            later_raw = locations[program_index][stage_index + 1:]
            if any(any(value.distance(later) <= gate_radius_m for value in leg[:-1])
                   for later in later_raw):
                raise RuntimeError(
                    f"{town}: inter-stage leg enters a later gate early for program "
                    f"{raw_programs[program_index]}"
                )

    ranked_candidates = []
    for candidate in candidates:
        candidate_start = _point({
            "x": candidate.transform.location.x,
            "y": candidate.transform.location.y,
            "z": candidate.transform.location.z,
        })
        if topology_spawn_token(
            town, candidate_start["x"], candidate_start["y"],
        ) in forbidden:
            continue
        first_distances = [candidate.transform.location.distance(program[0].transform.location)
                           for program in projected]
        if (any(distance < minimum_start_m for distance in first_distances)
                or any(distance > maximum_program_m for distance in first_distances)):
            continue
        ranked_candidates.append((_candidate_distance_key(candidate, first_distances), candidate))
    ranked_candidates.sort(key=lambda row: row[0])

    for _, candidate in ranked_candidates:
        compiled, totals, valid = [], [], True
        for program_index, program in enumerate(projected):
            current = candidate
            points = [_point({"x": current.transform.location.x, "y": current.transform.location.y,
                              "z": current.transform.location.z})]
            total = 0.0
            for stage_index, target in enumerate(program):
                leg_result = _directed_leg(town, carla_map, planner, current, target)
                if leg_result is None:
                    valid = False; break
                leg, leg_length, _ = leg_result
                direct = current.transform.location.distance(target.transform.location)
                if leg_length > direct * maximum_leg_factor + 35.0:
                    valid = False; break
                # A strict program may not enter any later gate before its
                # declared stage.  Fail closed at generation time.
                later_raw = locations[program_index][stage_index + 1:]
                if any(any(value.distance(later) <= gate_radius_m for value in leg[:-1]) for later in later_raw):
                    valid = False; break
                total += leg_length
                leg_points = [_point({"x": value.x, "y": value.y, "z": value.z}) for value in leg]
                if points and leg_points and math.dist(
                    tuple(float(points[-1][axis]) for axis in ("x", "y", "z")),
                    tuple(float(leg_points[0][axis]) for axis in ("x", "y", "z")),
                ) <= 0.25:
                    # GRP may serialize the same projected waypoint a few
                    # centimetres differently at a leg boundary.  Retaining
                    # both creates a synthetic reverse seam even though each
                    # directed leg is valid.  Collapse only the documented
                    # waypoint-projection tolerance; larger gaps remain and
                    # are rejected by the complete-polyline proof below.
                    leg_points = leg_points[1:]
                points.extend(leg_points)
                current = target
            if not valid or total > maximum_program_m:
                valid = False; break
            compiled_proof = _validate_directed_polyline(carla_map, points)
            if not compiled_proof["valid"]:
                valid = False; break
            # Replay gate rising edges over the complete compiled path.  A
            # later stage entered during the spawn-to-first leg is an invalid
            # ordered witness even if each individual route leg is directed.
            inside = [False] * len(locations[program_index])
            entered: list[int] = []
            for point in points:
                for gate_index, gate in enumerate(locations[program_index]):
                    now = math.hypot(
                        float(point["x"]) - float(gate.x),
                        float(point["y"]) - float(gate.y),
                    ) <= gate_radius_m
                    if now and not inside[gate_index]:
                        entered.append(gate_index)
                    inside[gate_index] = now
            if entered != list(range(len(locations[program_index]))):
                valid = False; break
            compiled.append(points); totals.append(total)
        if valid:
            exact_start = _point({
                "x": candidate.transform.location.x, "y": candidate.transform.location.y,
                "z": candidate.transform.location.z,
            })
            for path in compiled:
                path[0] = dict(exact_start)
            return {
                "start": exact_start,
                "yaw_deg": round(float(candidate.transform.rotation.yaw), 3),
                "paths": compiled,
                "canonical_lengths_m": [round(float(value), 3) for value in totals],
                "candidate_policy": "first_complete_proof_after_stable_distance_lane_sort_v1",
            }
    raise RuntimeError(f"{town}: no common directed start for ordered programs {first_program} / {second_program}")


def _legalize_oracle_paths(project_root: Path, pcus: list[dict[str, Any]]) -> None:
    """Replace legacy hand-graph polylines with directed CARLA-map routes.

    The old graph contributes semantic waypoints only. Reachability and every
    path segment come from CARLA's directed topology. ``road1`` is the sole
    documented exception because its get_topology() is empty.
    """
    carla = _import_carla()
    python_api = (Path(__file__).resolve().parents[2] / "carla_runtime/PythonAPI/carla")
    if str(python_api) not in sys.path:
        sys.path.insert(0, str(python_api))
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    maps_config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    xodr_root = Path(maps_config["xodr_root"])
    planners: dict[str, Any] = {}
    carla_maps: dict[str, Any] = {}
    for town in sorted({pcu["town"] for pcu in pcus if pcu["town"] != "road1"}):
        xodr = (xodr_root / f"{town}.xodr").read_text(encoding="utf-8")
        carla_maps[town] = carla.Map(town, xodr)
        planners[town] = GlobalRoutePlanner(carla_maps[town], 1.0)

    for pcu in pcus:
        for arm in pcu["arms"]:
            source = arm["oracle_path"]
            intended_length = float(arm.get("intended_path_length_m", _distance(source)))
            signal_proof = arm.get("signal_free_corridor_proof")
            if signal_proof is not None:
                # This versioned proof binds the exact 1 m execution slice.
                # Generic 4 m compaction would detach the serialized path from
                # its audited 20 m pre/post buffer and is therefore forbidden.
                if not arm.get("oracle_route_prevalidated"):
                    raise RuntimeError(f"{pcu['id']}/{arm['id']}: signal-free route is not prevalidated")
                path_hash = stable_hash(source)
                if path_hash != signal_proof.get("execution_path_sha256"):
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: serialized signal-free path hash mismatch"
                    )
                if any("yaw_deg" not in point for point in source):
                    raise RuntimeError(f"{pcu['id']}/{arm['id']}: signal-free path lacks yaw")
                # Signal-free corridor audits and their schema bind planar
                # OpenDRIVE lane arclength.  Elevation must not silently turn
                # that frozen XY proof into a different 3D measurement.
                legal_length = _distance(source)
                if abs(legal_length - float(signal_proof["corridor_length_m"])) > 0.01:
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: signal-free path length differs from proof"
                    )
                rechecked = _validate_signal_free_execution_path(
                    carla_maps[pcu["town"]], source, signal_proof,
                )
                if rechecked <= 0:
                    raise RuntimeError(f"{pcu['id']}/{arm['id']}: empty signal-free exact proof")
                arm["oracle_path"] = [dict(point) for point in source]
                arm["canonical_legal_path_length_m"] = round(legal_length, 3)
                arm["intended_path_length_m"] = round(intended_length, 3)
                arm["intended_to_legal_ratio"] = round(
                    legal_length / max(intended_length, 1e-9), 5,
                )
                arm["route_length_ratio"] = 1.0
                arm["oracle_route_source"] = "audited_signal_free_lane_slice_v2"
                continue
            parking_v5_route_proof = arm.get("parking_v5_route_proof")
            if parking_v5_route_proof is not None:
                if (
                    not arm.get("oracle_route_prevalidated")
                    or stable_hash(source) != parking_v5_route_proof.get("oracle_path_sha256")
                ):
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: parking-v5 source/proof hash differs"
                    )
                # Parking-v5 exact9 binds the once-counted serialized path in
                # three dimensions (the same definition used by its schema
                # and post-materialization qualification gate).
                legal_length = _parking_v5_path_length_xyz(source)
                if round(legal_length, 3) != round(intended_length, 3):
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: parking-v5 intended length differs"
                    )
                # The exact9 proof protects its dense tail and complete
                # explicit suffix.  Generic compaction/yaw annotation is not
                # allowed to mutate either (and the already-directed prefix is
                # intentionally retained byte-for-byte here).
                arm["oracle_path"] = [dict(point) for point in source]
                arm["canonical_legal_path_length_m"] = round(legal_length, 3)
                arm["intended_path_length_m"] = round(legal_length, 3)
                arm["intended_to_legal_ratio"] = 1.0
                arm["route_length_ratio"] = 1.0
                arm["oracle_route_source"] = "stored_xodr_parking_v5_prevalidated"
                continue
            if arm.get("oracle_route_prevalidated"):
                legal_length = _distance(source)
                intended_ratio = legal_length / max(intended_length, 1e-9)
                if not 0.8 <= intended_ratio <= 1.25:
                    raise RuntimeError(f"{pcu['id']}/{arm['id']}: prevalidated CARLA route ratio {intended_ratio:.3f}")
                target = arm.get("target", {})
                selected_target = target
                if target.get("type") == "if_else":
                    condition = str(target.get("condition", "")).split(":", 1)[-1]
                    occupied = bool(arm.get("actor_setup", {}).get("slot_occupancy", {}).get(condition))
                    selected_target = target["then" if occupied else "else"]
                manoeuvre = (
                    _parking_manoeuvre_path(selected_target)
                    if selected_target.get("type") == "parking_slot" else []
                )
                if manoeuvre:
                    if len(source) < len(manoeuvre) or source[-len(manoeuvre):] != manoeuvre:
                        raise RuntimeError(
                            f"{pcu['id']}/{arm['id']}: parking v4 source does not end in exact "
                            "collapsed manoeuvre"
                        )
                    # Compact only the ordinary forward route through the
                    # audited approach.  Every <=0.1 m manoeuvre pose is an
                    # immutable proof input and must survive serialization.
                    compact_source = source[:len(source) - len(manoeuvre) + 1]
                else:
                    compact_source = source
                compact = [compact_source[0]]; accumulated = 0.0
                for previous, current in zip(compact_source, compact_source[1:]):
                    accumulated += math.dist((previous["x"], previous["y"]), (current["x"], current["y"]))
                    if accumulated >= 4.0:
                        compact.append(current); accumulated = 0.0
                if compact[-1] != compact_source[-1]: compact.append(compact_source[-1])
                if selected_target.get("type") == "region":
                    # The audited region terminal pose is intentionally offset
                    # from the lane centre.  Preserve its short, directed
                    # pre-terminal stage when compacting; otherwise the 4 m
                    # sampler can replace it with a lateral connector whose
                    # tangent is incompatible with both the lane and terminal
                    # yaw.
                    stage = source[-2]
                    compact = [point for point in compact[:-1]
                               if math.dist((point["x"], point["y"]),
                                            (stage["x"], stage["y"])) > 0.05]
                    compact.extend([_point(stage), _point(source[-1])])
                terminal_yaw = None
                if selected_target.get("type") == "parking_slot":
                    terminal_yaw = float(selected_target["approach_pose"]["yaw_deg"])
                elif selected_target.get("type") == "region":
                    terminal_yaw = float(selected_target["terminal_pose"]["yaw_deg"])
                compact = _annotate_path_yaws(carla_maps[pcu["town"]], compact,
                                              terminal_yaw_deg=terminal_yaw)
                if selected_target.get("type") == "parking_slot":
                    compact[-1]["yaw_deg"] = round(float(selected_target["approach_pose"]["yaw_deg"]), 3)
                    if compact[-1] != manoeuvre[0]:
                        raise RuntimeError(
                            f"{pcu['id']}/{arm['id']}: compacted route disagrees with parking approach"
                        )
                    compact = [*compact[:-1], *manoeuvre]
                arm["oracle_path"] = compact
                arm["canonical_legal_path_length_m"] = round(legal_length, 3)
                arm["intended_path_length_m"] = round(intended_length, 3)
                arm["intended_to_legal_ratio"] = round(intended_ratio, 5)
                arm["route_length_ratio"] = round(_distance(compact) / max(legal_length, 1e-9), 5)
                arm["oracle_route_source"] = "offline_carla.GlobalRoutePlanner_prevalidated"
                continue
            if pcu["town"] == "road1" or len(source) < 2:
                arm["canonical_legal_path_length_m"] = round(intended_length, 3)
                arm["intended_path_length_m"] = round(intended_length, 3)
                arm["intended_to_legal_ratio"] = 1.0
                arm["route_length_ratio"] = 1.0
                arm["oracle_route_source"] = "road1_monotonic_s_fallback" if pcu["town"] == "road1" else "stationary"
                if pcu["town"] != "road1":
                    arm["oracle_path"] = _annotate_path_yaws(carla_maps[pcu["town"]], source)
                continue
            route: list[dict[str, float]] = []
            segments = list(zip(source, source[1:]))
            for segment_index, (start, end) in enumerate(segments):
                origin = carla.Location(x=float(start["x"]), y=float(start["y"]), z=float(start.get("z", 0)))
                destination = carla.Location(x=float(end["x"]), y=float(end["y"]), z=float(end.get("z", 0)))
                if arm.get("offroad_final_connector") and segment_index == len(segments) - 1:
                    points = [_point(start), _point(end)]
                else:
                    try:
                        segment = planners[pcu["town"]].trace_route(origin, destination)
                    except Exception as error:
                        raise RuntimeError(f"directed routing failed for {pcu['id']}/{arm['id']} from {start} to {end}: {error}") from error
                    if not segment:
                        raise RuntimeError(f"no directed CARLA route for {pcu['id']}/{arm['id']} from {start} to {end}")
                    points = [_point({"x": wp.transform.location.x, "y": wp.transform.location.y,
                                      "z": wp.transform.location.z}) for wp, _ in segment]
                exact_segment_start, exact_segment_end = _point(start), _point(end)
                if points and math.dist((points[0]["x"], points[0]["y"]),
                                        (exact_segment_start["x"], exact_segment_start["y"])) > 0.25:
                    points.insert(0, exact_segment_start)
                if points and math.dist((points[-1]["x"], points[-1]["y"]),
                                        (exact_segment_end["x"], exact_segment_end["y"])) > 0.25:
                    points.append(exact_segment_end)
                if route and points and math.dist((route[-1]["x"], route[-1]["y"]), (points[0]["x"], points[0]["y"])) < 0.25:
                    points = points[1:]
                route.extend(points)
            # CARLA projects both endpoints to a driving waypoint.  Preserve
            # the exact common source and annotated destination so paired arms
            # really share their initial state and parking targets retain the
            # final low-speed manoeuvre.  These short connectors are included
            # in both the canonical and compact lengths.
            exact_start, exact_end = _point(source[0]), _point(source[-1])
            full_route = list(route)
            if math.dist((full_route[0]["x"], full_route[0]["y"]),
                         (exact_start["x"], exact_start["y"])) > 0.25:
                full_route.insert(0, exact_start)
            if math.dist((full_route[-1]["x"], full_route[-1]["y"]),
                         (exact_end["x"], exact_end["y"])) > 0.25:
                full_route.append(exact_end)
            compact = [full_route[0]]
            accumulated = 0.0
            for previous, current in zip(full_route, full_route[1:]):
                accumulated += math.dist((previous["x"], previous["y"]), (current["x"], current["y"]))
                if accumulated >= 4.0:
                    compact.append(current); accumulated = 0.0
            if compact[-1] != full_route[-1]: compact.append(full_route[-1])
            compact = _annotate_path_yaws(carla_maps[pcu["town"]], compact)
            legal_length = _distance(full_route)
            intended_ratio = legal_length / max(intended_length, 1e-9)
            if not 0.8 <= intended_ratio <= 1.25:
                raise RuntimeError(
                    f"{pcu['id']}/{arm['id']}: directed legal route length {legal_length:.3f}m "
                    f"is {intended_ratio:.3f}x the intended annotated path {intended_length:.3f}m"
                )
            arm["oracle_path"] = compact
            arm["canonical_legal_path_length_m"] = round(legal_length, 3)
            arm["intended_path_length_m"] = round(intended_length, 3)
            arm["intended_to_legal_ratio"] = round(intended_ratio, 5)
            arm["oracle_route_source"] = "offline_carla.GlobalRoutePlanner"
            arm["route_length_ratio"] = round(_distance(compact) / max(legal_length, 1e-9), 5)

    # A paired counterfactual has one exact execution transform, including yaw.
    for pcu in pcus:
        first_yaw = float(pcu["arms"][0]["oracle_path"][0]["yaw_deg"])
        for arm in pcu["arms"]:
            arm["oracle_path"][0]["yaw_deg"] = round(first_yaw, 3)
            signal_proof = arm.get("signal_free_corridor_proof")
            if signal_proof is not None:
                if stable_hash(arm["oracle_path"]) != signal_proof["execution_path_sha256"]:
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: normalized signal-free path hash mismatch"
                    )
                _validate_signal_free_execution_path(
                    carla_maps[pcu["town"]], arm["oracle_path"], signal_proof,
                )
            target = arm.get("target", {})
            selected_target = target
            if target.get("type") == "if_else":
                condition = str(target.get("condition", "")).split(":", 1)[-1]
                occupied = bool(arm.get("actor_setup", {}).get("slot_occupancy", {}).get(condition))
                selected_target = target["then" if occupied else "else"]
            # Parking's immutable, gear-labelled actor-transform suffix is
            # checked by its axle-correct v4 proof.  Only the ordinary forward prefix through the
            # exact approach belongs to the lane-direction proof.
            if selected_target.get("type") == "parking_slot":
                manoeuvre = _parking_manoeuvre_path(selected_target)
                if (len(arm["oracle_path"]) < len(manoeuvre)
                        or arm["oracle_path"][-len(manoeuvre):] != manoeuvre):
                    raise RuntimeError(
                        f"{pcu['id']}/{arm['id']}: serialized parking suffix lost exact closure"
                    )
                parking_v5_route_proof = arm.get("parking_v5_route_proof")
                if parking_v5_route_proof is not None:
                    if stable_hash(arm["oracle_path"]) != parking_v5_route_proof.get(
                        "oracle_path_sha256"
                    ):
                        raise RuntimeError(
                            f"{pcu['id']}/{arm['id']}: paired yaw normalization detached "
                            "parking-v5 route proof"
                        )
                    # Full directed/spacing/topology/OBB replay is performed by
                    # the exact source schema.  Never reinterpret pose0 as the
                    # physical slot-axis approach here.
                    continue
                directed_path = arm["oracle_path"][:-len(manoeuvre) + 1]
            else:
                directed_path = arm["oracle_path"]
            proof = _validate_directed_polyline(
                carla_maps[pcu["town"]],
                directed_path,
                maximum_heading_error_deg=90.0,
            )
            if not proof["valid"]:
                raise RuntimeError(f"{pcu['id']}/{arm['id']}: serialized oracle path failed: {proof}")
            if selected_target.get("type") == "parking_slot":
                approach = selected_target.get("approach_pose")
                if not isinstance(approach, dict) or len(arm["oracle_path"]) < 3:
                    raise RuntimeError(f"{pcu['id']}/{arm['id']}: parking path lacks audited approach")
                if math.dist(
                    tuple(float(directed_path[-1].get(axis, 0.0)) for axis in ("x", "y", "z")),
                    tuple(float(approach.get(axis, 0.0)) for axis in ("x", "y", "z")),
                ) > 1e-6:
                    raise RuntimeError(f"{pcu['id']}/{arm['id']}: forward prefix does not end at approach_pose")


def _validate_signal_free_split_isolation(
    project_root: Path, pcus: list[dict[str, Any]],
) -> None:
    """Reject physical-road reuse by any final opposite-split route.

    Lane predicates alone do not cover object/headway/passing programs.  This
    qualification projects every serialized chord at <=1 m onto the offline
    CARLA map, then compares physical road IDs for the same town.  It runs only
    after route legalization, so the audit covers exactly what the evaluator
    will execute rather than a pre-compaction candidate.
    """

    carla = _import_carla()
    maps = {
        town: _directed_route_context(project_root, town)[0]
        for town in sorted({pcu["town"] for pcu in pcus})
    }
    roads: dict[tuple[str, str], set[int]] = defaultdict(set)
    sample_counts: dict[tuple[str, str], int] = defaultdict(int)
    for pcu in pcus:
        key = (pcu["town"], pcu["split"])
        carla_map = maps[pcu["town"]]
        for arm in pcu["arms"]:
            for first, second in zip(arm["oracle_path"], arm["oracle_path"][1:]):
                dx, dy, dz = (
                    float(second["x"]) - float(first["x"]),
                    float(second["y"]) - float(first["y"]),
                    float(second.get("z", 0.0)) - float(first.get("z", 0.0)),
                )
                count = max(1, int(math.ceil(math.hypot(dx, dy) / 1.0)))
                for index in range(count + 1):
                    ratio = index / count
                    location = carla.Location(
                        x=float(first["x"]) + ratio * dx,
                        y=float(first["y"]) + ratio * dy,
                        z=float(first.get("z", 0.0)) + ratio * dz,
                    )
                    waypoint = None
                    for lane_type in (carla.LaneType.Driving, carla.LaneType.Parking):
                        waypoint = carla_map.get_waypoint(location, False, lane_type)
                        if waypoint is not None:
                            break
                    sample_counts[key] += 1
                    if waypoint is not None:
                        roads[key].add(int(waypoint.road_id))

    for pcu in pcus:
        proof = pcu["arms"][0].get("signal_free_corridor_proof")
        if not isinstance(proof, dict):
            continue
        parts = str(proof["lane_id"]).rsplit(":", 3)
        road_id = int(parts[1])
        opposite = "test" if pcu["split"] == "dev" else "dev"
        opposite_key = (pcu["town"], opposite)
        if road_id in roads[opposite_key]:
            raise RuntimeError(
                f"{pcu['id']}: signal-free road {pcu['town']}:{road_id} "
                f"is reused by a final {opposite} route"
            )
        pcu.setdefault("metadata", {})["signal_free_split_isolation"] = {
            "method": "final_oracle_path_exact_road_projection_v1",
            "candidate_road_id": road_id,
            "opposite_split": opposite,
            "opposite_split_route_samples_checked": sample_counts[opposite_key],
            "physical_road_overlap_count": 0,
            "sample_spacing_max_m": 1.0,
        }


def _directed_feasible_approaches(project_root: Path, approaches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    carla = _import_carla()
    python_api = (Path(__file__).resolve().parents[2] / "carla_runtime/PythonAPI/carla")
    if str(python_api) not in sys.path: sys.path.insert(0, str(python_api))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    config = json.loads((project_root / "config/maps.json").read_text(encoding="utf-8"))
    path = Path(config["xodr_root"]) / "American-city0428.xodr"
    planner = GlobalRoutePlanner(carla.Map("American-city0428", path.read_text(encoding="utf-8")), 1.0)
    result = []
    for approach in approaches:
        start = approach["start"]
        origin = carla.Location(x=float(start["x"]), y=float(start["y"]), z=float(start.get("z", 0)))
        valid = True
        for maneuver in approach["maneuvers"]:
            center = maneuver["region"]["center"]
            destination = carla.Location(x=float(center["x"]), y=float(center["y"]), z=0)
            try:
                valid = valid and bool(planner.trace_route(origin, destination))
            except Exception:
                valid = False
        if valid: result.append(approach)
    return result


def generate_core_sources(project_root: Path, output: Path | None = None) -> dict[str, Any]:
    output = output or project_root / "sources/core_pcus.json"
    map_dir, review_dir = project_root / "generated/maps/lane_graphs", project_root / "assets/review"
    if not map_dir.is_dir():
        raise FileNotFoundError("run `l2d build-maps` before generating the suite")
    pcus: list[dict[str, Any]] = []
    lane_terminal_source_paths: set[str] = set()
    usable = _mine_carla_junction_approaches(project_root)
    speed_policy = _speed_policy(project_root)
    runtime_speed_cap = float(speed_policy["cap_mps"])
    speed_margin = float(speed_policy["control_margin_mps"])
    speed_gap = float(speed_policy["counterfactual_gap_mps"])
    signal_free_speed_corridors = _load_signal_free_speed_corridors(
        project_root, map_dir,
    )
    primary_corridor_audit = str(
        project_root / "generated/signal_free_speed_corridor_audit.json"
    )
    reserve_corridor_audit = str(
        project_root / "generated/signal_free_ad_corridor_reserve_audit.json"
    )

    # A: 12 junction/fork, 8 lane, 6 speed/start, 6 waiting/adjacent slots.
    for i, approach in enumerate(usable[:12], 1):
        first, second = approach["maneuvers"][:2]
        ta, tb = _gate_target("American-city0428", approach, first), _gate_target("American-city0428", approach, second)
        start = approach["start"]
        arms = [_arm(first["maneuver"], _instructions(*MANEUVER_TEXT[first["maneuver"]]), _eventually(_gate(ta["id"])), _eventually(_gate(tb["id"])), ta["id"], ta, first["path"], deadline=45),
                _arm(second["maneuver"], _instructions(*MANEUVER_TEXT[second["maneuver"]]), _eventually(_gate(tb["id"])), _eventually(_gate(ta["id"])), tb["id"], tb, second["path"], deadline=45)]
        for arm, maneuver in zip(arms, (first, second)):
            arm["intended_path_length_m"] = maneuver["canonical_path_length_m"]
            arm["junction_path_provenance"] = maneuver["junction_path_provenance"]
        pcus.append(_pcu("A", i, "junction", "American-city0428", "M-X", _forward_envelope(start, start["yaw_deg"]), arms, [str(INTERSECTIONS)], ladder=i if i <= 8 else None))

    lane_candidates = _load_lane_candidates(map_dir)
    for offset, (town, family, lane) in enumerate(lane_candidates[:8], 13):
        lanes = _lane_by_id(map_dir, town)
        a_id, b_id, path_a, path_b, terminal_proof = _paired_lane_paths(
            project_root, town, lane, lanes, terminal_commitment=True,
        )
        if terminal_proof is None:
            raise AssertionError("A/lane terminal proof was not generated")
        lane_terminal_source_paths.update((
            terminal_proof["xodr_source_ref"],
            terminal_proof["lane_graph_source_ref"],
        ))
        a_word = "left" if lane.get("left_adjacent") == a_id else "right"
        b_word = "current"
        ta, tb = {"type": "lane", "id": a_id}, {"type": "lane", "id": b_id}
        # Both arms share a long physical prefix in the current lane.  A lane
        # predicate activated at the midpoint can therefore latch the current
        # outcome during a correct (especially slow) lane change.  The paired
        # miner proves that its connector ends by progress .88 and that both
        # paths remain in their respective terminal lanes throughout .9..1;
        # only then activate the symmetric two-second outcome commitment.
        ca = {"sequence": [
            {"progress": {"min": LANE_TERMINAL_COMMITMENT_PROGRESS}},
            {"dwell": {
                "condition": {"lane_relation": {"value": a_id}},
                "seconds": 2.0,
            }},
        ]}
        cb = {"sequence": [
            {"progress": {"min": LANE_TERMINAL_COMMITMENT_PROGRESS}},
            {"dwell": {
                "condition": {"lane_relation": {"value": b_id}},
                "seconds": 2.0,
            }},
        ]}
        text_a = _instructions(f"Move one lane to the {a_word}.", f"Change into the lane on the {a_word}.", f"Take the adjacent {a_word} lane.")
        text_b = (_instructions("Keep the current lane.", "Stay in this lane.", "Continue without changing lanes.") if b_word == "current" else
                  _instructions("Move one lane to the right.", "Change into the lane on the right.", "Take the adjacent right lane."))
        arms = [_arm(a_word, text_a, ca, cb, a_id, ta, path_a, deadline=40),
                _arm(b_word, text_b, cb, ca, b_id, tb, path_b, deadline=40)]
        for arm in arms:
            arm["intended_path_length_m"] = _distance(arm["oracle_path"])
            arm["oracle_route_prevalidated"] = True
        pcus.append(_pcu(
            "A", offset, "lane", town, family,
            [_point(lane["samples"][0]), _point(lane["samples"][-1])],
            arms, [
                terminal_proof["lane_graph_source_ref"],
                terminal_proof["xodr_source_ref"],
            ],
            metadata={"lane_terminal_commitment_proof": terminal_proof},
        ))

    # ``road1`` remains an offline topology-fallback fixture only.  Positive
    # speed invariants also cannot cross a stop sign or traffic light.  Use a
    # frozen, buffered, signal-free custom-map corridor for all A speed/start
    # programs so both arms remain feasible independently of signal phase.
    a_speed_corridors = signal_free_speed_corridors["a_test"]
    for j in range(4):
        index = 21+j
        control_town, control_family, control_lane, control_signal_proof = a_speed_corridors[j]
        control_samples = control_lane["samples"]
        path = [_point(value) for value in control_samples]
        control_source = str(map_dir / f"{control_town}.lane_graph.json")
        control_legal = min(
            _lane_speed_limit_mps(project_root, control_town, control_lane),
            runtime_speed_cap,
        )
        low, high = _disjoint_speed_bands(
            control_legal, control_margin_mps=speed_margin, gap_mps=speed_gap,
        )
        low_band={"speed_band":{"min_mps":low[0],"max_mps":low[1]}}
        high_band={"speed_band":{"min_mps":high[0],"max_mps":high[1]}}
        # A high-speed arm necessarily accelerates through the lower numeric
        # band.  Treat the requested band as a stable terminal commitment, not
        # an incidental value observed once during the transient.
        ca={"sequence":[
            {"dwell":{"condition":low_band,"seconds":2.0}},
            {"until":{"invariant":low_band,"condition":{"progress":{"min":0.9}}}},
        ]}
        cb={"sequence":[
            {"dwell":{"condition":high_band,"seconds":2.0}},
            {"until":{"invariant":high_band,"condition":{"progress":{"min":0.9}}}},
        ]}
        arms = [_arm("low", _instructions(f"Slow down to between {low[0]:.0f} and {low[1]:.0f} meters per second.", "Reduce to the lower marked speed band.", "Drive in the low speed band."), ca, cb, "low", {"type":"speed_band","min_mps":low[0],"max_mps":low[1]}, path, deadline=35),
                _arm("high", _instructions(f"Speed up to between {high[0]:.0f} and {high[1]:.0f} meters per second.", "Increase to the higher marked speed band.", "Drive in the high speed band."), cb, ca, "high", {"type":"speed_band","min_mps":high[0],"max_mps":high[1]}, path, deadline=35)]
        for arm in arms:
            arm["signal_free_corridor_proof"] = control_signal_proof
            arm["oracle_route_prevalidated"] = True
        pcus.append(_pcu("A", index, "speed_start", control_town, control_family,
                         _forward_envelope(path[0], float(control_samples[0]["yaw_deg"])),
                         arms, [control_source, reserve_corridor_audit]))

    parking_data = json.loads((TRAJECTORIES / "parking_traj_graph_sequence_starts_150_151_158_159.json").read_text(encoding="utf-8"))
    starts = {int(k): v for k, v in parking_data["vertex_meta"].items() if int(k) >= 150}
    point_lookup = {int(k): v for k, v in parking_data["vertex_meta"].items()}
    parking_paths, _, slot_vertices = _parking_paths(); slots = _load_slots(review_dir)
    (
        slot_occupancy_path,
        slot_terminal_clear,
        slot_occupancy_source_refs,
    ) = _load_slot_collision_occupancy(project_root, review_dir, slots)
    clear_slots = {
        name: slot for name, slot in slots.items() if slot_terminal_clear[name]
    }
    # Pure wait/start controls use two further independently audited corridors;
    # only the temporal program differs between arms of the same PCU.
    for j in range(25, 27):
        control_town, control_family, control_lane, control_signal_proof = a_speed_corridors[j - 21]
        control_samples = control_lane["samples"]
        route_points = [_point(value) for value in control_samples]
        control_source = str(map_dir / f"{control_town}.lane_graph.json")
        start = route_points[0]
        # "Start now" and "wait, then start" must remain true semantic
        # counterfactuals under the global NOT-opposite success rule.  A loose
        # eventual-progress goal would also become true after the delayed arm
        # legitimately starts, making that arm logically impossible.  Eight
        # seconds gives a legal controller ample time to establish 15% route
        # progress while remaining disjoint from the ten-second hold.
        move = {"within": {
            "condition": {"progress": {"min": 0.15}},
            "seconds": 8,
        }}
        wait_then_move = {"sequence": [
            {"dwell": {"condition": {"speed_band": {"min_mps": 0, "max_mps": .2}}, "seconds": 10}},
            {"progress": {"min": .15}},
        ]}
        arms = [_arm("start", _instructions("Start driving now.", "Set off now.", "Begin moving immediately."), move, wait_then_move, "start_now", {"type":"progress","min":.15}, route_points, deadline=20),
                _arm("wait_then_start", _instructions("Wait here for ten seconds, then start driving.", "Remain stopped for ten seconds before setting off.", "Hold position for ten seconds and then begin moving."), wait_then_move, move, "delayed_start", {"type":"delayed_progress","wait_seconds":10,"min":.15}, route_points, deadline=35)]
        for arm in arms:
            arm["signal_free_corridor_proof"] = control_signal_proof
            arm["oracle_route_prevalidated"] = True
        pcus.append(_pcu("A", j, "speed_start", control_town, control_family,
                         _forward_envelope(route_points[0], float(control_samples[0]["yaw_deg"])),
                         arms, [control_source, reserve_corridor_audit]))

    slot_names = sorted(clear_slots)
    adjacent_pairs = []
    for a, b in zip(slot_names, slot_names[1:]):
        if a[0] == b[0] and int(b[1:]) - int(a[1:]) == 1:
            adjacent_pairs.append((a, b))
    # Preserve literal adjacent-space semantics while excluding every baked
    # collision occupant frozen by the CARLA terminal-spawn asset.
    waiting_pairs = [
        ("B204", "B205"), ("B205", "B206"), ("B206", "B207"),
        ("B207", "B208"), ("D402", "D403"), ("D403", "D404"),
    ]
    if any(pair not in adjacent_pairs for pair in waiting_pairs):
        raise RuntimeError("collision-clear waiting pair is not adjacent")
    for index, (a_name, b_name) in enumerate(waiting_pairs, 27):
        ta, tb = _slot_target(clear_slots[a_name]), _slot_target(clear_slots[b_name]); ca, cb = _parking_contract(ta), _parking_contract(tb)
        route_pair = _common_directed_parking_pair(project_root, ta, tb)
        arms = [_arm("left", _instructions("Park in the empty space on the left.", "Use the adjacent left-hand parking space.", "Pull into the open bay to the left."), ca, cb, ta["id"], ta, route_pair["paths"][0], deadline=120),
                _arm("right", _instructions("Park in the empty space on the right.", "Use the adjacent right-hand parking space.", "Pull into the open bay to the right."), cb, ca, tb["id"], tb, route_pair["paths"][1], deadline=120)]
        for arm, length, route_proof in zip(
            arms, route_pair["canonical_lengths_m"], route_pair["route_proofs"],
        ):
            arm["intended_path_length_m"] = length
            arm["offroad_final_connector"] = True
            arm["oracle_route_prevalidated"] = True
            arm["parking_v5_route_proof"] = route_proof
        pcus.append(_pcu("A", index, "waiting_parking", "parkinglotfish-new3", "M-G", _forward_envelope(route_pair["start"], route_pair["yaw_deg"]), arms, [str(review_dir / "slot_registry.csv")]))

    # B: speed corridor (8), lane corridor (8), object/headway (8), pass/order (8).
    behavior_routes = lane_candidates * 4
    corridor_low, corridor_high = _disjoint_speed_bands(
        runtime_speed_cap, control_margin_mps=speed_margin, gap_mps=speed_gap,
    )
    b_speed_corridors = (
        signal_free_speed_corridors["b_dev"]
        + signal_free_speed_corridors["b_test"]
    )
    for index, (town, family, lane, signal_proof) in enumerate(
        b_speed_corridors, 1,
    ):
        path = [_point(value) for value in lane["samples"]]
        low, high = _disjoint_speed_bands(
            min(_lane_speed_limit_mps(project_root, town, lane), runtime_speed_cap),
            control_margin_mps=speed_margin, gap_mps=speed_gap,
        )
        end={"type":"gate","id":f"scope:{town}:speed:{index}","center":path[-1],"radius_m":4}
        ca={"sequence":[{"dwell":{"condition":{"speed_band":{"min_mps":low[0],"max_mps":low[1]}},"seconds":2.0}},{"until":{"invariant":{"speed_band":{"min_mps":low[0],"max_mps":low[1]}},"condition":_gate(end["id"])}}]}
        cb={"sequence":[{"dwell":{"condition":{"speed_band":{"min_mps":high[0],"max_mps":high[1]}},"seconds":2.0}},{"until":{"invariant":{"speed_band":{"min_mps":high[0],"max_mps":high[1]}},"condition":_gate(end["id"])}}]}
        arms=[_arm("low",_instructions("Keep a low legal speed throughout this corridor.","Maintain the lower speed band along this section.","Drive this entire stretch in the low speed range."),ca,cb,"low_corridor",{"type":"until","constraint":{"type":"speed_band","min_mps":low[0],"max_mps":low[1]},"end":end},path,deadline=80),
              _arm("high",_instructions("Keep a higher legal speed throughout this corridor.","Maintain the higher speed band along this section.","Drive this entire stretch in the high speed range."),cb,ca,"high_corridor",{"type":"until","constraint":{"type":"speed_band","min_mps":high[0],"max_mps":high[1]},"end":end},path,deadline=80)]
        for arm in arms:
            arm["intended_path_length_m"] = _distance(path)
            arm["oracle_route_prevalidated"] = True
            arm["signal_free_corridor_proof"] = signal_proof
        pcus.append(_pcu(
            "B", index, "speed_constraint", town, family,
            _forward_envelope(path[0], float(path[0]["yaw_deg"])), arms,
            [str(map_dir/f"{town}.lane_graph.json"), primary_corridor_audit], ladder=index,
        ))
    for index in range(9,17):
        town,family,lane=behavior_routes[index-9]; lanes=_lane_by_id(map_dir,town)
        a_id,b_id,pa,pb,_terminal_proof=_paired_lane_paths(project_root,town,lane,lanes)
        if _terminal_proof is not None:
            raise AssertionError("B/lane path unexpectedly generated an A terminal proof")
        gate_center={axis:round((float(pa[-1][axis])+float(pb[-1][axis]))/2,4) for axis in ("x","y","z")}
        end={"type":"gate","id":f"scope:{town}:lane:{index}","center":gate_center,"radius_m":6}
        ca={"sequence":[{"dwell":{"condition":{"lane_relation":{"value":a_id}},"seconds":2.0}},{"until":{"invariant":{"lane_relation":{"value":a_id}},"condition":_gate(end["id"])}}]}
        cb={"sequence":[{"dwell":{"condition":{"lane_relation":{"value":b_id}},"seconds":2.0}},{"until":{"invariant":{"lane_relation":{"value":b_id}},"condition":_gate(end["id"])}}]}
        arms=[_arm("lane_a",_instructions("Stay in the left-side lane for this whole section.","Keep to the left lane along the corridor.","Use the left-side lane until this section ends."),ca,cb,a_id,{"type":"until","constraint":{"type":"lane","id":a_id},"end":end},pa,deadline=80),
              _arm("lane_b",_instructions("Stay in the right-side lane for this whole section.","Keep to the right lane along the corridor.","Use the right-side lane until this section ends."),cb,ca,b_id,{"type":"until","constraint":{"type":"lane","id":b_id},"end":end},pb,deadline=80)]
        for arm in arms:
            arm["intended_path_length_m"] = _distance(arm["oracle_path"])
            arm["oracle_route_prevalidated"] = True
        pcus.append(_pcu("B",index,"lane_constraint",town,family,[pa[0],pa[-1]],arms,[str(map_dir/f"{town}.lane_graph.json")]))
    behavior_speed_routes = list(lane_candidates)
    # Town05 road37 and Town03 road0 traverse junction connector roads whose
    # OpenDRIVE has no declared speed.  They remain valid lane-change tasks,
    # but speed/headway/until programs require path-wide positive speed
    # provenance.  Substitute two audited corridors with explicit limits.
    replacement_behavior_lanes = [
        ("Town01", "M-U", _lane_by_id(map_dir, "Town01")["Town01:12:0:-1"]),
        ("hanghailou3", "M-H", _lane_by_id(map_dir, "hanghailou3")["hanghailou3:2:0:1"]),
    ]
    replacement_iter = iter(replacement_behavior_lanes)
    for position, (town, family, lane) in enumerate(behavior_speed_routes):
        if (town, lane["id"]) in {
            ("Town05", "Town05:37:0:3"), ("Town03", "Town03:0:0:-2"),
        }:
            behavior_speed_routes[position] = next(replacement_iter)
    for index in range(17,25):
        town,family,lane=behavior_speed_routes[index-17]; path=_lane_path(lane); actor=f"lead-{index:02d}"
        normal_band={"headway_band":{"actor_id":actor,"min_seconds":1.5,"max_seconds":2.5}}
        large_band={"headway_band":{"actor_id":actor,"min_seconds":3.5,"max_seconds":5.0}}
        # Permit a deterministic acquisition transient, then require the
        # selected gap continuously through 90% task progress.  A loose
        # eventual dwell could latch one band and later finish in the other,
        # making both counterfactual outcomes true in one rollout.
        normal={"sequence":[
            {"dwell":{"condition":normal_band,"seconds":5.0}},
            {"until":{"invariant":normal_band,"condition":{"progress":{"min":0.9}}}},
        ]}
        large={"sequence":[
            {"dwell":{"condition":large_band,"seconds":5.0}},
            {"until":{"invariant":large_band,"condition":{"progress":{"min":0.9}}}},
        ]}
        setup={"actors":[{"id":actor,"blueprint":"vehicle.tesla.model3","color":"0,0,255","relative_spawn_m":25,"program":"constant_speed"}]}
        arms=[_arm("normal",_instructions("Follow the blue car at a normal safe distance.","Keep the usual gap behind the blue vehicle.","Maintain a standard following distance from the blue car."),normal,large,"normal_headway",{"type":"headway","actor_id":actor,"min_seconds":1.5,"max_seconds":2.5},path,setup,90),
              _arm("large",_instructions("Leave a larger gap behind the blue car.","Keep extra following distance from the blue vehicle.","Stay farther back from the blue car."),large,normal,"large_headway",{"type":"headway","actor_id":actor,"min_seconds":3.5,"max_seconds":5},path,setup,90)]
        pcus.append(_pcu("B",index,"object_relation",town,family,[path[0],path[-1]],arms,[str(map_dir/f"{town}.lane_graph.json")]))
    passing_routes: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    passing_seen: set[tuple[str, str]] = set()
    family_by_town = {
        "Town01": "M-U", "Town03": "M-U", "Town04": "M-U", "Town05": "M-U",
        "Town06": "M-U", "hanghailou3": "M-H", "hubeiwenli4": "M-C",
        "yangluogang1": "M-P",
    }
    pass_pool = list(lane_candidates)
    for town, family in family_by_town.items():
        for candidate in _lane_by_id(map_dir, town).values():
            pass_pool.append((town, family, candidate))
    for town, family, lane in pass_pool:
        identity = (town, lane["id"])
        if identity in passing_seen or lane.get("is_junction") or len(lane.get("samples", [])) < 72:
            continue
        passing_seen.add(identity)
        proof = _pass_lane_proof(project_root, town, lane)
        if proof is None:
            continue
        passing_routes.append((town, family, lane, proof))
        if len(passing_routes) == 8:
            break
    if len(passing_routes) != 8:
        raise RuntimeError(f"need eight audited passing corridors, found {len(passing_routes)}")

    for index in range(25,33):
        town,family,lane,pass_proof=passing_routes[index-25]
        start_index=min(
            range(len(lane["samples"])),
            key=lambda position: math.dist(
                (float(lane["samples"][position]["x"]), float(lane["samples"][position]["y"])),
                (float(pass_proof["start"]["x"]), float(pass_proof["start"]["y"])),
            ),
        )
        path=_lane_path(lane,start_index=start_index); path[0]=_point(pass_proof["start"])
        actor=f"slow-{index:02d}"
        pass_c={"eventually":{"actor_order":{"id":f"ego_ahead:{actor}"}}}
        # Every overtake begins behind the truck, so a five-second eventual
        # behind-state is not the opposite outcome: it would latch before a
        # perfectly valid pass completes.  The non-overtake arm instead stays
        # behind until 90% path progress.  A pass must occur before that
        # terminal commitment to avoid satisfying both counterfactuals.
        follow_c={"until":{
            "invariant":{"actor_order":{"id":f"ego_behind:{actor}"}},
            "condition":{"progress":{"min":0.9}},
        }}
        setup={"actors":[{"id":actor,"blueprint":"vehicle.carlamotors.carlacola","relative_spawn_m":18,"program":"slow_constant_speed"}]}
        pass_target={"type":"actor_order","actor_id":actor,"order":"ego_ahead",
                     "pass_lane_id":int(pass_proof["adjacent_lane_numeric_id"]),
                     "pass_lane_proof":pass_proof}
        arms=[_arm("pass",_instructions("Safely pass the slow truck.","Overtake the slow vehicle when safe.","Move ahead of the slow truck safely."),pass_c,follow_c,"ego_ahead",pass_target,path,setup,100),
              _arm("follow",_instructions("Do not overtake; remain behind the slow truck.","Keep following the truck without passing it.","Stay behind the slow vehicle."),follow_c,pass_c,"ego_behind",{"type":"actor_order","actor_id":actor,"order":"ego_behind"},path,setup,100)]
        pcus.append(_pcu("B",index,"passing_order",town,family,[path[0],path[-1]],arms,[str(map_dir/f"{town}.lane_graph.json")]))

    # C: 8 M-X road-name branches backed by existing live-CARLA annotations,
    # plus port targets, parking regions, and exact slots.
    for index,approach in enumerate(usable[:8],1):
        ma,mb=approach["maneuvers"][:2];ta,tb=_gate_target("American-city0428",approach,ma),_gate_target("American-city0428",approach,mb)
        name_a,name_b=ma["semantic_name"],mb["semantic_name"]
        label_a = f"{ma['direction']} {name_a}" if name_a == name_b else name_a
        label_b = f"{mb['direction']} {name_b}" if name_a == name_b else name_b
        start=approach["start"]
        arms=[_arm("road_a",_instructions(f"Take {label_a}.",f"Drive onto {label_a}.",f"At the sign, continue toward {label_a}."),_eventually(_gate(ta["id"])),_eventually(_gate(tb["id"])),ta["id"],{**ta,"semantic_name":name_a,"semantic_label":label_a},ma["path"],deadline=55),
              _arm("road_b",_instructions(f"Take {label_b}.",f"Drive onto {label_b}.",f"At the sign, continue toward {label_b}."),_eventually(_gate(tb["id"])),_eventually(_gate(ta["id"])),tb["id"],{**tb,"semantic_name":name_b,"semantic_label":label_b},mb["path"],deadline=55)]
        for arm, maneuver in zip(arms, (ma, mb)):
            arm["intended_path_length_m"] = maneuver["canonical_path_length_m"]
            arm["junction_path_provenance"] = maneuver["junction_path_provenance"]
        evidence = {label_a:{"gate":ta["id"],"edge_token":ma["semantic_edge_token"],"distance_m":ma["semantic_distance_m"],"heading_error_deg":ma["semantic_heading_error_deg"]},
                    label_b:{"gate":tb["id"],"edge_token":mb["semantic_edge_token"],"distance_m":mb["semantic_distance_m"],"heading_error_deg":mb["semantic_heading_error_deg"]}}
        pcus.append(_pcu("C",index,"road_name","American-city0428","M-X",_forward_envelope(start,start["yaw_deg"]),arms,[str(DATA_ROOT/"carla_point_jsons/urban.json"),str(DATA_ROOT/"trj_seq_jsons/urban_traj_graph_sequence_manual.json")],ladder=index,metadata={"fixed_semantic_mapping":evidence,"semantic_review":"pending_visibility_witness"}))

    port_file=TRAJECTORIES/"port_traj_graph_sequence_all_start.json";port=json.loads(port_file.read_text(encoding="utf-8"));port_by_start,_=_path_index(port_file);port_points=_port_points()
    # Port targets are final destinations rather than one-shot branch gates.
    # A two-second terminal dwell makes the outcomes mutually exclusive even
    # when a legal access road passes close to another labelled destination.
    # C-009..016 are all test PCUs.  Restrict them to V7--V12 so the V0--V6
    # port entities used by D-001..006 remain dev-only across *all* layers.
    # These are the eight V7--V12 pairs for which both targets have a common
    # local directed CARLA spawn.  The legacy port graph supplies names only.
    port_programs=((7,11),(7,12),(8,9),(8,10),(8,11),(9,10),(9,11),(11,12))
    for index,(target_a_vertex,target_b_vertex) in enumerate(port_programs,9):
        name_a=_humanize(port["vertex_meta"][str(target_a_vertex)]["primary_name"])
        name_b=_humanize(port["vertex_meta"][str(target_b_vertex)]["primary_name"])
        center_a,center_b=port_points[target_a_vertex],port_points[target_b_vertex]
        poly_a,poly_b=_rectangle(center_a,0,10,10),_rectangle(center_b,0,10,10)
        ta={"type":"gate","id":f"port:v{target_a_vertex}","center":center_a,"radius_m":5,"semantic_name":name_a}
        tb={"type":"gate","id":f"port:v{target_b_vertex}","center":center_b,"radius_m":5,"semantic_name":name_b}
        ca,cb=_terminal_region_contract(poly_a,2.0),_terminal_region_contract(poly_b,2.0)
        # Each confirmed port label lies within 3.8 m of its legal lane and
        # therefore inside the frozen 10x10 terminal polygon.  End at that
        # exact projected lane pose; appending the physical label centre would
        # create a lateral 90-degree connector without changing the semantic
        # destination contract.
        route_pair=_common_directed_route_pair(
            project_root,"yangluogang1",center_a,center_b,
            minimum_start_m=15.0,maximum_route_m=300.0,
            include_target_connector=False,
        )
        arms=[_arm("target_a",_instructions(f"Drive to {name_a}.",f"Follow the signs for {name_a}.",f"Navigate toward {name_a}."),ca,cb,ta["id"],ta,route_pair["paths"][0],deadline=180),
              _arm("target_b",_instructions(f"Drive to {name_b}.",f"Follow the signs for {name_b}.",f"Navigate toward {name_b}."),cb,ca,tb["id"],tb,route_pair["paths"][1],deadline=180)]
        for arm,length in zip(arms,route_pair["canonical_lengths_m"]):
            arm["intended_path_length_m"]=length;arm["oracle_route_prevalidated"]=True
        pcus.append(_pcu("C",index,"port_target","yangluogang1","M-P",_forward_envelope(route_pair["start"],route_pair["yaw_deg"]),arms,[str(port_file),str(DATA_ROOT/"carla_point_jsons/port.json")],metadata={"excluded_unreviewed_vertices":[13,14],"legacy_graph_role":"semantic_names_only"}))

    by_region:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for slot in slots.values():by_region[slot["region"]].append(slot)
    # Existing confirmed XODR-bound representative slots.  Their compact
    # terminal regions are disjoint and preserve the A--F semantic zone label
    # without using overlapping whole-zone bounding boxes.
    region_programs=(("A","A804","B","B207"),("C","C106","D","D507"),("E","E101","F","F103"),
                     ("A","A804","C","C106"),("B","B207","D","D406"),("C","C106","E","E703"),
                     ("D","D501","F","F409"),("A","A208","F","F409"))
    for index,(ra,slot_a_name,rb,slot_b_name) in enumerate(region_programs,17):
        sa,sb=slots[slot_a_name],slots[slot_b_name];poly_a,poly_b=_terminal_region(sa),_terminal_region(sb)
        # Region is a terminal-state contract: incidental transit through the
        # other named zone must not poison the run.  Five seconds of stable
        # occupancy makes the two final outcomes mutually exclusive.
        ca,cb=_terminal_region_contract(poly_a,5.0),_terminal_region_contract(poly_b,5.0)
        pose_a,pose_b=(
            dict(_REGION_TERMINAL_POSES[slot_a_name]),
            dict(_REGION_TERMINAL_POSES[slot_b_name]),
        )
        proof_a=_validate_region_terminal_pose(project_root,ra,pose_a,poly_a)
        proof_b=_validate_region_terminal_pose(project_root,rb,pose_b,poly_b)
        target_a,target_b=(
            {"type":"region","id":ra,"polygon":poly_a,"terminal_pose":pose_a,
             "terminal_pose_proof":proof_a},
            {"type":"region","id":rb,"polygon":poly_b,"terminal_pose":pose_b,
             "terminal_pose_proof":proof_b},
        )
        route_pair=_common_directed_region_pair(project_root,pose_a,pose_b)
        proof_a.update(_validate_region_terminal_connector(
            project_root,ra,pose_a,route_pair["paths"][0]
        ))
        proof_b.update(_validate_region_terminal_connector(
            project_root,rb,pose_b,route_pair["paths"][1]
        ))
        arms=[_arm("region_a",_instructions(f"Drive to parking zone {ra}.",f"Follow the signs to zone {ra}.",f"Navigate into area {ra}."),ca,cb,f"region:{ra}",target_a,route_pair["paths"][0],deadline=180),
              _arm("region_b",_instructions(f"Drive to parking zone {rb}.",f"Follow the signs to zone {rb}.",f"Navigate into area {rb}."),cb,ca,f"region:{rb}",target_b,route_pair["paths"][1],deadline=180)]
        for arm,length in zip(arms,route_pair["canonical_lengths_m"]):
            arm["intended_path_length_m"]=length;arm["offroad_final_connector"]=True;arm["oracle_route_prevalidated"]=True
        pcus.append(_pcu("C",index,"parking_region","parkinglotfish-new3","M-G",_forward_envelope(route_pair["start"],route_pair["yaw_deg"]),arms,[str(review_dir/"slot_registry.csv")]))

    exact_pairs = [
        ("B204", "B205"), ("E101", "E102"),
        ("E601", "E602"), ("E602", "E603"),
        ("F303", "F304"), ("C505", "C506"),
        ("B403", "B404"), ("B502", "B503"),
    ]
    if any(pair not in adjacent_pairs for pair in exact_pairs):
        raise RuntimeError("collision-clear exact-parking pair is not adjacent")
    for index,(a_name,b_name) in enumerate(exact_pairs,25):
        ta,tb=_slot_target(clear_slots[a_name]),_slot_target(clear_slots[b_name]);ca,cb=_parking_contract(ta),_parking_contract(tb)
        route_pair=_common_directed_parking_pair(project_root,ta,tb)
        arms=[_arm("slot_a",_instructions(f"Park in space {a_name}.",f"Navigate to and park at {a_name}.",f"Use the parking space numbered {a_name}."),ca,cb,ta["id"],ta,route_pair["paths"][0],deadline=200),
              _arm("slot_b",_instructions(f"Park in space {b_name}.",f"Navigate to and park at {b_name}.",f"Use the parking space numbered {b_name}."),cb,ca,tb["id"],tb,route_pair["paths"][1],deadline=200)]
        for arm,length,route_proof in zip(
            arms,route_pair["canonical_lengths_m"],route_pair["route_proofs"]
        ):
            arm["intended_path_length_m"]=length;arm["offroad_final_connector"]=True;arm["oracle_route_prevalidated"]=True
            arm["parking_v5_route_proof"]=route_proof
        pcus.append(_pcu("C",index,"exact_parking","parkinglotfish-new3","M-G",_forward_envelope(route_pair["start"],route_pair["yaw_deg"]),arms,[str(review_dir/"slot_registry.csv")]))

    # D: eight strict orders, eight conditionals, eight until constraints, and
    # eight delayed references.  Exact direction replay shows that the legacy
    # V0--V6 port labels provide only one bidirectional order pair, so they
    # cannot honestly supply six dev programs.  Use confirmed one-to-one XODR
    # parking-space entities for seven programs and retain the independently
    # proven V11<->V12 port program for M-P coverage.  The first six parking
    # pairs are dev-only; the seventh and the port pair are test-only, and all
    # entities are disjoint across the split.
    strict_parking_pairs = [
        ("B208", "F206"), ("B401", "E303"), ("D511", "F204"),
        ("A113", "E401"), ("B301", "F301"), ("B307", "E203"),
        ("D402", "F303"),
    ]
    parking_gate_centers: dict[str, dict[str, float]] = {}
    strict_specs: list[dict[str, Any]] = []
    for first_slot, second_slot in strict_parking_pairs:
        first_center = _polygon_centroid(clear_slots[first_slot]["polygon"])
        second_center = _polygon_centroid(clear_slots[second_slot]["polygon"])
        first_gate, second_gate = f"parking-stage:{first_slot}", f"parking-stage:{second_slot}"
        parking_gate_centers[first_gate] = first_center
        parking_gate_centers[second_gate] = second_center
        strict_specs.append({
            "town": "parkinglotfish-new3", "family": "M-G",
            "names": (first_slot, second_slot), "gates": (first_gate, second_gate),
            "centers": (first_center, second_center),
            "sources": [str(review_dir / "slot_registry.csv")],
        })
    strict_specs.append({
        "town": "yangluogang1", "family": "M-P",
        "names": tuple(_humanize(port["vertex_meta"][str(value)]["primary_name"])
                       for value in (11, 12)),
        "gates": ("port:v11", "port:v12"),
        "centers": (port_points[11], port_points[12]),
        "sources": [str(port_file), str(DATA_ROOT / "carla_point_jsons/port.json")],
    })
    for index, spec in enumerate(strict_specs, 1):
        split = "dev" if index <= 6 else "test"
        first_gate, second_gate = spec["gates"]
        first_center, second_center = spec["centers"]
        first_name, second_name = spec["names"]
        ca = {"sequence": [_gate(first_gate), _gate(second_gate)]}
        cb = {"sequence": [_gate(second_gate), _gate(first_gate)]}
        route_pair = _common_directed_program_pair(
            project_root, spec["town"],
            [first_center, second_center], [second_center, first_center],
            forbidden_spawn_tokens=_opposite_split_spawn_tokens(
                pcus, split, spec["town"],
            ),
        )
        arms = [
            _arm("order_ab", _instructions(
                f"First visit {first_name}, then go to {second_name}.",
                f"Go to {first_name} before continuing to {second_name}.",
                f"Visit {first_name} first and {second_name} second.",
            ), ca, cb, "order_ab", {"type": "sequence", "stages": [first_gate, second_gate]},
                 route_pair["paths"][0], deadline=240),
            _arm("order_ba", _instructions(
                f"First visit {second_name}, then go to {first_name}.",
                f"Go to {second_name} before continuing to {first_name}.",
                f"Visit {second_name} first and {first_name} second.",
            ), cb, ca, "order_ba", {"type": "sequence", "stages": [second_gate, first_gate]},
                 route_pair["paths"][1], deadline=240),
        ]
        for arm, length in zip(arms, route_pair["canonical_lengths_m"]):
            arm["intended_path_length_m"] = length
            arm["oracle_route_prevalidated"] = True
        pcus.append(_pcu(
            "D", index, "strict_order", spec["town"], spec["family"],
            _forward_envelope(route_pair["start"], route_pair["yaw_deg"]),
            arms, spec["sources"],
        ))

    # Eight confirmed triples whose exact Lincoln terminal transforms are
    # collision-clear in the frozen cooked map.  The XODR registry also
    # contains bays occupied by unlabelled baked vehicles; those labels remain
    # visible to review but are never compiled as benchmark targets.
    conditional_triples = [
        ("E101", "E102", "C104"), ("B505", "B506", "A705"),
        ("F103", "F204", "F201"), ("D206", "D315", "C801"),
        ("C801", "F405", "F402"), ("E501", "E403", "E503"),
        ("F301", "F303", "F304"), ("C505", "C506", "E702"),
    ]
    if len(conditional_triples) != 8:
        raise RuntimeError(f"need eight adjacent parking triples for D conditionals, found {len(conditional_triples)}")
    for index,(a_name,b_name,c_name) in enumerate(conditional_triples,9):
        ta,tb,tc=(_slot_target(clear_slots[a_name]),_slot_target(clear_slots[b_name]),
                  _slot_target(clear_slots[c_name]))
        # Compile both static branches, not only the branch activated by this
        # PCU's actor setup.  This keeps the program self-contained and lets
        # the schema/audit reject an unaudited counterfactual target.
        for parking_target in (ta, tb, tc):
            approach, approach_proof = _parking_approach_target(project_root, parking_target)
            parking_target["approach_pose"] = approach
            parking_target["approach_pose_proof"] = approach_proof
        occupancy={"actor_order":{"id":f"occupied:{a_name}"}}
        ca={"if_else":{"condition":occupancy,"then":_parking_contract(tb),"else":_parking_contract(ta)}}
        cb={"if_else":{"condition":occupancy,"then":_parking_contract(tc),"else":_parking_contract(tb)}}
        # D-015 is deliberately empty.  Its occupied=true third-slot branch is
        # still compiled and fully validated, while no dynamic Audi collision
        # geometry is introduced in the selected runtime state.
        occupied=False if index == 15 else index % 2 == 1
        setup={"condition_cell":"occupied" if occupied else "empty","slot_occupancy":{a_name:occupied,b_name:False,c_name:False},"actors":([{"id":f"occupant:{a_name}","blueprint":"vehicle.audi.a2","slot":a_name}] if occupied else [])}
        near_target,far_target=(tb,tc) if occupied else (ta,tb)
        route_pair=_common_directed_parking_pair(project_root,near_target,far_target)
        arms=[_arm("fallback_near",_instructions(f"If {a_name} is occupied, park in {b_name}; otherwise use {a_name}.",f"Use {a_name} unless it is taken, in which case choose {b_name}.",f"Check {a_name}: if unavailable, park at {b_name}."),ca,cb,"conditional_near",{"type":"if_else","condition":f"occupied:{a_name}","then":tb,"else":ta},route_pair["paths"][0],setup,220),
              _arm("fallback_far",_instructions(f"If {a_name} is occupied, park in {c_name}; otherwise use {b_name}.",f"Use {b_name} unless {a_name} is taken, in which case choose {c_name}.",f"Check {a_name}: if unavailable, park at {c_name}; otherwise use {b_name}."),cb,ca,"conditional_far",{"type":"if_else","condition":f"occupied:{a_name}","then":tc,"else":tb},route_pair["paths"][1],setup,220)]
        for arm,length,route_proof in zip(
            arms,route_pair["canonical_lengths_m"],route_pair["route_proofs"]
        ):
            arm["intended_path_length_m"]=length;arm["offroad_final_connector"]=True;arm["oracle_route_prevalidated"]=True
            arm["parking_v5_route_proof"]=route_proof
        pcus.append(_pcu("D",index,"conditional","parkinglotfish-new3","M-G",_forward_envelope(route_pair["start"],route_pair["yaw_deg"]),arms,[str(review_dir/"slot_registry.csv")]))

    for index, (town, family, lane, signal_proof) in enumerate(
        signal_free_speed_corridors["d_test"], 17,
    ):
        path=[_point(value) for value in lane["samples"]];end=f"scope:{town}:{index}"
        low_values,high_values=_disjoint_speed_bands(
            min(_lane_speed_limit_mps(project_root,town,lane), runtime_speed_cap),
            control_margin_mps=speed_margin, gap_mps=speed_gap,
        )
        low_band={"speed_band":{"min_mps":low_values[0],"max_mps":low_values[1]}}
        high_band={"speed_band":{"min_mps":high_values[0],"max_mps":high_values[1]}}
        low={"sequence":[{"dwell":{"condition":low_band,"seconds":2.0}},{"until":{"invariant":low_band,"condition":_gate(end)}}]}
        high={"sequence":[{"dwell":{"condition":high_band,"seconds":2.0}},{"until":{"invariant":high_band,"condition":_gate(end)}}]}
        target={"type":"gate","id":end,"center":path[-1],"radius_m":4}
        arms=[_arm("low_until",_instructions(f"Keep between {low_values[0]:g} and {low_values[1]:g} meters per second until the end marker, then continue normally.","Maintain the low speed band until the marker is passed.","Stay in the lower speed range until reaching the end gate."),low,high,"low_until",{"type":"until","constraint":{"min_mps":low_values[0],"max_mps":low_values[1]},"end":target},path,deadline=100),
              _arm("high_until",_instructions(f"Keep between {high_values[0]:g} and {high_values[1]:g} meters per second until the end marker, then continue normally.","Maintain the high speed band until the marker is passed.","Stay in the higher speed range until reaching the end gate."),high,low,"high_until",{"type":"until","constraint":{"min_mps":high_values[0],"max_mps":high_values[1]},"end":target},path,deadline=100)]
        for arm in arms:
            arm["signal_free_corridor_proof"] = signal_proof
            arm["oracle_route_prevalidated"] = True
        pcus.append(_pcu(
            "D", index, "until_invariant", town, family,
            [path[0], path[-1]], arms,
            [str(map_dir/f"{town}.lane_graph.json"), reserve_corridor_audit],
        ))

    # The port has only four split-safe bidirectional pairs for the test-side
    # entity set.  Memory tasks therefore use confirmed, globally one-to-one
    # XODR parking-space bindings.  This supplies eight distinct programs
    # without inventing semantics or admitting the unreviewed V13/V14 labels.
    # All eight pairs have both directed inter-stage legs, a shared spawn, and
    # exact gate-transition replay.  The earlier A804/C106, A804/D507, and
    # D406/E703 pairs are omitted because endpoint-canonical proof rejects one
    # direction.  Replacements reuse confirmed test-side entities only.
    memory_programs = [
        ("A804", "B207"), ("A804", "E101"), ("A209", "D406"),
        ("A209", "F409"), ("B207", "F103"), ("B506", "D315"),
        ("B502", "E602"), ("B406", "E503"),
    ]
    for index,(a,b) in enumerate(memory_programs,25):
        split = "dev" if index <= 6 else "test"
        name_a,name_b=a,b
        gate_a,gate_b=f"parking-stage:{a}",f"parking-stage:{b}"
        center_a,center_b=(_polygon_centroid(clear_slots[a]["polygon"]),
                           _polygon_centroid(clear_slots[b]["polygon"]))
        parking_gate_centers[gate_a]=center_a;parking_gate_centers[gate_b]=center_b
        ref_a=f"memory_a_{index:03d}";ref_b=f"memory_b_{index:03d}"
        ca={"bind_reference":{"reference":ref_a,"value":gate_a,"body":{"sequence":[_gate(gate_b),{"enter_gate":{"id":{"$ref":ref_a}}}]}}}
        cb={"bind_reference":{"reference":ref_b,"value":gate_b,"body":{"sequence":[_gate(gate_a),{"enter_gate":{"id":{"$ref":ref_b}}}]}}}
        route_pair=_common_directed_program_pair(project_root,"parkinglotfish-new3",
                                                 [center_b,center_a], [center_a,center_b],
                                                 forbidden_spawn_tokens=_opposite_split_spawn_tokens(
                                                     pcus, split, "parkinglotfish-new3",
                                                 ))
        arms=[_arm("remember_a",_instructions(f"Remember {name_a}; visit {name_b}, then return to that remembered place.",f"Keep {name_a} in mind, go to {name_b}, and finally go back there.",f"After visiting {name_b}, return to the place I named first: {name_a}."),ca,cb,"memory_a",{"type":"memory","bind":name_a,"stages":[name_b,name_a]},route_pair["paths"][0],deadline=260),
              _arm("remember_b",_instructions(f"Remember {name_b}; visit {name_a}, then return to that remembered place.",f"Keep {name_b} in mind, go to {name_a}, and finally go back there.",f"After visiting {name_a}, return to the place I named first: {name_b}."),cb,ca,"memory_b",{"type":"memory","bind":name_b,"stages":[name_a,name_b]},route_pair["paths"][1],deadline=260)]
        for arm,length in zip(arms,route_pair["canonical_lengths_m"]):
            arm["intended_path_length_m"]=length;arm["oracle_route_prevalidated"]=True
        pcus.append(_pcu("D",index,"memory_reference","parkinglotfish-new3","M-G",_forward_envelope(route_pair["start"],route_pair["yaw_deg"]),arms,[str(review_dir/"slot_registry.csv")]))

    if len(pcus) != 128:
        raise RuntimeError(f"generator bug: expected 128 PCUs, produced {len(pcus)}")
    parking_registry = _parking_registry(project_root)
    parking_town_registry = parking_registry["towns"]["parkinglotfish-new3"]
    parking_vehicle_asset = (project_root / PARKING_VEHICLE_BBOX_ASSET).resolve()
    parking_vehicle_evidence = json.loads(
        parking_vehicle_asset.read_text(encoding="utf-8")
    )["source_files"]
    parking_dynamics_asset = (project_root / PARKING_VEHICLE_DYNAMICS_ASSET).resolve()
    parking_dynamics_evidence = json.loads(
        parking_dynamics_asset.read_text(encoding="utf-8")
    )["source_files"]
    parking_source_refs = [
        parking_town_registry["bbox_asset"]["source_ref"],
        parking_town_registry["xodr"]["source_ref"],
        parking_town_registry["lane_graph"]["source_ref"],
        parking_town_registry["control_model"]["vehicle_dynamics_source_ref"],
        str(parking_vehicle_asset),
        *(value["path"] for _, value in sorted(parking_vehicle_evidence.items())),
        *(path for path, _ in sorted(parking_dynamics_evidence.items())),
    ]
    slot_registry_source = str((review_dir / "slot_registry.csv").resolve())
    for pcu in pcus:
        if slot_registry_source in pcu["source_refs"]:
            for source_ref in slot_occupancy_source_refs:
                if source_ref not in pcu["source_refs"]:
                    pcu["source_refs"].append(source_ref)
    for pcu in pcus:
        if any(_contains_parking_target(arm.get("target")) for arm in pcu["arms"]):
            for source_ref in parking_source_refs:
                if source_ref not in pcu["source_refs"]:
                    pcu["source_refs"].append(source_ref)
    _legalize_oracle_paths(project_root, pcus)
    _validate_signal_free_split_isolation(project_root, pcus)
    _attach_speed_proofs(project_root, pcus)
    port_source=DATA_ROOT/"carla_point_jsons/port.json"
    geometry_registry={
        f"port:v{vertex}": {
            "type":"gate", "id":f"port:v{vertex}", "center":point,
            "radius_m":5.0, "geometry_version":"existing-port-labels-v1",
            "source_ref":str(port_source), "source_sha256":sha256_file(port_source),
        }
        for vertex,point in sorted(port_points.items()) if vertex <= 12
    }
    slot_registry_source=review_dir/"slot_registry.csv"
    for identifier,center in sorted(parking_gate_centers.items()):
        slot_number=identifier.rsplit(":",1)[-1]
        geometry_registry[identifier]={
            "type":"gate", "id":identifier, "center":center, "radius_m":5.0,
            "geometry_version":"confirmed-xodr-slot-binding-v1",
            "physical_space_id":slots[slot_number]["physical_space_id"],
            "source_ref":str(slot_registry_source), "source_sha256":sha256_file(slot_registry_source),
        }
    source_files = {
        str(path): sha256_file(path) for path in (
            INTERSECTIONS,
            TRAJECTORIES / "parking_traj_graph_sequence_starts_150_151_158_159.json",
            port_file, review_dir / "slot_registry.csv",
            *(Path(value) for value in sorted(lane_terminal_source_paths)),
            *(Path(value) for value in parking_source_refs),
            *slot_occupancy_source_refs,
        )
    }
    payload={"schema_version":"1.0","generator":"language2drive.generator",
             "geometry_registry":geometry_registry,
             "parking_manoeuvre_registry":parking_registry,
             "source_files":source_files,"pcus":pcus}
    atomic_json(output,payload)
    return {"pcu_count":len(pcus),"arm_count":sum(len(p["arms"]) for p in pcus),"output":str(output)}
