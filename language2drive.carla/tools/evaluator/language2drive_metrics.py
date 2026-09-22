"""
Calculate Evaluation Metrics Calculator

** Only for one route **

This version combine the original Language2Drive metrics with the new metrics from the new version.

Metrics Layers
--------------

Table 1 =====

Layer A: Same with standard bench2drive metrics - Merging, Overtaking, Emergency Brake, Give Way

Layer B: Increase speed, decrease speed, keep speed

Layer C: SR (within 5 meters)

Layer D: Sequentials (follow our predefined waypoints order) 暂时不考虑这个

Table 2 ====

1. IFS

Layer B: difference between final speed and speed mentioned in instruction

Layer C, D: NE, SPL, TL

2. DPS

Layer A: driving score, efficiency, comfortable

Layer C: efficiency, comfortable

"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import subprocess
import time
import numpy as np
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "language2drive", "tools"))


########## Layer A ##########
# Copied from Bench2Drive/tools/ability_benchmark.py
Ability = {
    "Overtaking": [
        "Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
        "HazardAtSideLaneTwoWays", "HazardAtSideLane", "ParkedObstacleTwoWays",
        "ParkedObstacle", "VehicleOpensDoorTwoWays",
    ],
    "Merging": [
        "CrossingBicycleFlow", "EnterActorFlow", "HighwayExit", "InterurbanActorFlow",
        "HighwayCutIn", "InterurbanAdvancedActorFlow", "MergerIntoSlowTrafficV2",
        "MergerIntoSlowTraffic", "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow", "ParkingExit", "SequentialLaneChange",
        "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
    ],
    "Emergency_Brake": [
        "BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute",
        "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight", "ParkingCutIn",
        "PedestrianCrossing", "ParkingCrossingPedestrian", "StaticCutIn", "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian", "ControlLoss",
    ],
    "Give_Way": ["InvadingTurn", "YieldToEmergencyVehicle"],
    # "Traffic_Signs": [
    #     "BlockedIntersection", "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight",
    #     "PedestrianCrossing", "VehicleTurningRoute", "VehicleTurningRoutePedestrian",
    #     "EnterActorFlow", "CrossingBicycleFlow", "NonSignalizedJunctionLeftTurn",
    #     "NonSignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurnEnterFlow",
    #     "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn",
    #     "SignalizedJunctionLeftTurnEnterFlow", "T_Junction", "VanillaNonSignalizedTurn",
    #     "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
    #     "VanillaNonSignalizedTurnEncounterStopsign",
    # ],
}

ABILITY_DISPLAY = {
    "Overtaking": "overtaking",
    "Merging": "merging",
    "Emergency_Brake": "emergency_brake",
    "Give_Way": "give_way",
    # "Traffic_Signs": "traffic_sign",
}


def load_result(path: str) -> Any:
    """Return saved result content (JSON or JSONL)."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)
    

def get_record(data: Dict[str, any]) -> Dict[str, any]:
    """Return useful info from result file."""
    records = data["_checkpoint"]["records"]
    if not records:
        raise ValueError("No route records found in Result JSON File.")
    if len(records) > 1:
        # 暂时一条一条评估，后续可以修改
        print(f"Warning: found {len(records)} records, using the first one.", file=sys.stderr)
    return records[0]


def extract_route_id(record: Dict[str, any]) -> str:
    """Get route id of a sample."""
    return record["route_id"].split("_")[1]


def get_scenario_name(
    record: Dict[str, any],
    routes_xml: str,
    route_id: str,
) -> str:
    """Get scenario name."""
    scenario_name = record.get("scenario_name") or ""
    if scenario_name:
        return scenario_name

    if routes_xml is None:
        return scenario_name

    tree = ET.parse(routes_xml)
    for route in tree.getroot().findall("route"):
        if route.get("id") == route_id:
            scenario = route.find("scenarios/scenario")
            if scenario is not None:
                return scenario.get("type") or ""
    return scenario_name


def normalize_scenario_name(scenario_name: str) -> str:
    """Strip Bench2Drive repetition suffix (_1, _2, ...) for Ability lookup."""
    if not scenario_name:
        return scenario_name
    base, _, suffix = scenario_name.rpartition("_")
    if suffix.isdigit():
        return base
    return scenario_name


def get_infraction_status(record: Dict[str, any]) -> bool:
    """Get infraction status of a sample not include min_speed_infraction."""
    for infraction, value in record["infractions"].items():
        if infraction == "min_speed_infractions":
            continue
        if len(value) > 0:
            return True
    return False


def get_record_success_status(record: Dict[str, any]) -> bool:
    """Return the success status of a sample."""
    if record["status"] in ("Completed", "Perfect"):
        return not get_infraction_status(record)
    return False
    
    
def update_ability(
    scenario_name: str, 
    ability_statistic: Dict[str, any], 
    status: bool
) -> None:
    """Update ability statistic."""
    normalized_name = normalize_scenario_name(scenario_name)
    for ability, scenarios in Ability.items():
        if normalized_name in scenarios:
            ability_statistic[ability][1] += 1
            if status:
                ability_statistic[ability][0] += 1
    
    
def ability_categories_for_scenario(scenario_name: str) -> List[str]:
    """Return a list that contains all abilities for scenario."""
    normalized_name = normalize_scenario_name(scenario_name)
    return [ability for ability, scenarios in Ability.items() if normalized_name in scenarios]


def compute_layer_A_basic_metrics(
    record: Dict[str, any],
    scenario_name: str,
    routes_xml: str
) -> Dict[str, any]:
    """Calculate bascie metrics for Layer A - merging, overtaking, emergency_brake, give_way"""
    record_success_status = get_record_success_status(record)
    ability_statistic = {key: [0, 0.0] for key in Ability} # 前面对应总数量，后面对应成功数量
    
    update_ability(
        scenario_name=scenario_name,
        ability_statistic=ability_statistic,
        status=record_success_status,
    )
    
    # Get all abilities that this scenario should be considered
    applicable = ability_categories_for_scenario(scenario_name)
    # Calculate results for layerA
    ability_scores = {}
    for ability in Ability:
        success_count, total_count = ability_statistic[ability]
        display_name = ABILITY_DISPLAY[ability]
        if ability in applicable and total_count > 0:
            score = float(success_count) / float(total_count)
            ability_scores[display_name] = {
                "applicable": True,
                "success_count": int(success_count),
                "total_count": int(total_count),
                "score": score,
            }
        else:
            ability_scores[display_name] = {
                "applicable": False,
                "success_count": 0,
                "total_count": 0,
                "score": None,
            }

    scored_values = [v["score"] for v in ability_scores.values() if v["score"] is not None]
    official_style_mean = (
        sum(
            ability_scores[name]["score"] if ability_scores[name]["score"] is not None else 0.0
            for name in ABILITY_DISPLAY.values()
        )
        / 5.0
        if scored_values
        else None
    )

    return {
        "route_id": record["route_id"],
        "scenario_name": scenario_name,
        "record_success_status": record_success_status,
        "applicable_abilities": applicable,
        "abilities": ability_scores,
        "mean_applicable_only": (
            sum(scored_values) / len(scored_values) if scored_values else None
        ),
        "mean_official_style": official_style_mean,
    }


def find_metric_info(
    records_dir: str,
    save_name: Optional[str] = None,
    route_id: Optional[str] = None,
) -> Optional[str]:
    """Locate metric_info.json saved alongside frames.jsonl and route_meta.json."""
    candidates = []
    if save_name:
        candidates.append(os.path.join(records_dir, save_name, "metric_info.json"))
    if route_id:
        candidates.append(os.path.join(records_dir, route_id, "metric_info.json"))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def is_success(record: Dict[str, any]) -> bool:
    """Same rule as Bench2Drive merge_route_json.py."""
    if record["status"] not in ("Completed", "Perfect"):
        return False
    for key, value in record["infractions"].items():
        if key == "min_speed_infractions":
            continue
        if len(value) > 0:
            return False
    return True


def compute_efficiency(record: Dict[str, any]) -> float | None:
    """Driving efficiency from min_speed_infractions messages."""
    messages = record["infractions"].get("min_speed_infractions", [])
    if not messages:
        return None

    values = []
    for msg in messages:
        match = re.search(r"\b\d+\.?\d*%", msg)
        if match is None:
            continue
        pct = float(match.group().rstrip("%"))
        if pct > 1000:
            continue
        values.append(pct)

    if not values:
        return None
    return sum(values) / len(values)


def summarize_infractions(record: Dict[str, any]) -> Dict[str, any]:
    """Summarize all infractions, same as Bench2Drive merge_route_json.py."""
    summary = {}
    for key, events in record["infractions"].items():
        summary[key] = len(events)
    return summary


def load_metric_arrays(metric_info_path: str) -> Dict[str, np.ndarray]:
    """Load the metric in the array format."""
    with open(metric_info_path, encoding="utf-8") as f:
        data = json.load(f)

    arrays = {
        "acceleration": [],
        "angular_velocity": [],
        "forward_vector": [],
        "right_vector": [],
        "location": [],
        "rotation": [],
    }
    for step in sorted(data.keys(), key=int):
        frame = data[step]
        for key in arrays:
            arrays[key].append(frame[key])

    for key in arrays:
        arrays[key] = np.array(arrays[key])
    return arrays


def compute_smoothness(metric_info_path: str) -> float:
    """Calculate the smoothness score for Layer A."""
    # Keep scipy optional for Layer B/C/D-only evaluation.
    from efficiency_smoothness_benchmark import (  # noqa: E402
        seg_compute_comfort_metric,
    )

    arrays = load_metric_arrays(metric_info_path)
    return seg_compute_comfort_metric(**arrays)


def computer_layer_A_advanced_metrics(
    result_path: str, 
    record: Dict[str, any], 
    metric_info_path=None
) -> Dict[str, any]:
    """Calculate driving score, efficiency, comfortable for Layer A."""
    scores = record["scores"]
    driving_score = scores["score_composed"]
    route_completion = scores["score_route"]
    infraction_penalty = scores["score_penalty"]
    
    report = {
        "result_file": os.path.abspath(result_path),
        "route_id": record["route_id"],
        "save_name": record.get("save_name"),
        "status": record["status"],
        "town_name": record.get("town_name"),
        "driving_score": driving_score,
        "route_completion": route_completion,
        "infraction_penalty": infraction_penalty,
        "success": is_success(record),
        "success_rate": 1.0 if is_success(record) else 0.0,
        "efficiency": compute_efficiency(record),
        "smoothness": None,
        "metric_info_path": metric_info_path,
        "route_length_m": record["meta"]["route_length"],
        "duration_game_s": record["meta"]["duration_game"],
        "infraction_counts": summarize_infractions(record),
        "infraction_messages": {
            k: v for k, v in record["infractions"].items() if len(v) > 0
        },
    }
    if metric_info_path:
        report["smoothness"] = compute_smoothness(metric_info_path)
    return report


########## Layer B ##########
MIN_SPEED_FRAMES = 6
SEGMENT_DELTA_THRESHOLD_MPS = 0.5
SEGMENT_RANGE_THRESHOLD_MPS = 0.5
MONOTONIC_MARGIN_MPS = 0.15


def extract_speed_series(frames: List[Dict]) -> tuple[List[float], List[float]]:
    """Get times and speeds for one sample."""
    times = []
    speeds = []
    for frame in frames:
        ego = frame.get("ego")
        if ego is None:
            continue
        times.append(float(frame["sim_time"]))
        speeds.append(float(ego["speed"]))
    return times, speeds


def segment_means(values: List[float], n_segments: int = 3) -> List[float]:
    """Segment in 3 parts, and calculate mean speed of each part."""
    n = len(values)
    seg_len = n // n_segments
    means = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else n
        chunk = values[start:end]
        if chunk:
            means.append(sum(chunk) / len(chunk))
    return means


def speed_trend_slope(times: List[float], speeds: List[float]) -> float:
    """Least-squares slope of speed (m/s) vs sim_time (s)."""
    if len(times) < 2:
        return 0.0
    duration = times[-1] - times[0]
    if duration <= 1e-6:
        return 0.0
    slope, _ = np.polyfit(np.array(times), np.array(speeds), 1)
    return float(slope)


def is_monotonic_increasing(values: List[float], margin: float) -> bool:
    """Is monotonic increasing or not."""
    return all(values[i] + margin < values[i + 1] for i in range(len(values) - 1))


def is_monotonic_decreasing(values: List[float], margin: float) -> bool:
    """Is monotonic decreasing or not."""
    return all(values[i] > values[i + 1] + margin for i in range(len(values) - 1))


def classify_speed_behavior(
    times: List[float],
    speeds: List[float],
    seg_means: List[float],
) -> tuple[str, Dict[str, Any]]:
    """
    Classify speed trend without instruction.

    1. 严格判断
    keep speed - 1.第一段和最后一段的平均速度差小于threshold 2.最大最小段的平均速度差小于一个threshold 3.斜率小于threshold 4.不是递增或者递减
    accelerate - 1.第一段和最后一段的平均速度之差大于threshold 2.斜率大于threshold 3.单调递增
    deaccelerate - 1.第一段和最后一段的平均速度之差小于-threshold 2.斜率小于-threshold 3. 单调递减
    """
    mean_start = seg_means[0]
    mean_end = seg_means[-1]
    delta = mean_end - mean_start
    seg_range = max(seg_means) - min(seg_means)
    slope = speed_trend_slope(times, speeds) # 通过最小而成线性拟合，计算斜率，如果斜率为0则表示keep speed

    duration = max(times[-1] - times[0], 0.1)
    # 在这个时间内判断slope的threshold应该设置为多少来判断是increase还是decrease
    slope_threshold = SEGMENT_DELTA_THRESHOLD_MPS / duration

    mono_up = is_monotonic_increasing(seg_means, MONOTONIC_MARGIN_MPS)
    mono_down = is_monotonic_decreasing(seg_means, MONOTONIC_MARGIN_MPS)

    evidence = {
        "mean_speed_start_mps": round(mean_start, 4),
        "mean_speed_end_mps": round(mean_end, 4),
        "speed_delta_mps": round(delta, 4),
        "segment_means_mps": [round(m, 4) for m in seg_means],
        "segment_range_mps": round(seg_range, 4),
        "speed_trend_slope_mps2": round(slope, 4),
        "slope_threshold_mps2": round(slope_threshold, 4),
        "monotonic_increasing": mono_up,
        "monotonic_decreasing": mono_down,
    }

    # 严格判断
    keep_speed = (
        abs(delta) < SEGMENT_DELTA_THRESHOLD_MPS
        and seg_range < SEGMENT_RANGE_THRESHOLD_MPS
        and abs(slope) < slope_threshold
        and not mono_up
        and not mono_down
    )
    accelerate = (
        delta >= SEGMENT_DELTA_THRESHOLD_MPS
        and slope >= slope_threshold
        and mono_up
    )
    decelerate = (
        delta <= -SEGMENT_DELTA_THRESHOLD_MPS
        and slope <= -slope_threshold
        and mono_down
    )

    # 宽松判断
    if keep_speed:
        behavior = "keep_speed"
    elif accelerate:
        behavior = "accelerate"
    elif decelerate:
        behavior = "decelerate"
    elif abs(delta) < SEGMENT_DELTA_THRESHOLD_MPS and seg_range < SEGMENT_RANGE_THRESHOLD_MPS:
        behavior = "keep_speed"
    elif delta > 0 and slope > 0:
        behavior = "accelerate"
    elif delta < 0 and slope < 0:
        behavior = "decelerate"
    else:
        behavior = "mixed"

    return behavior, evidence


def compute_layer_B_metrics(
    frames: List[Dict],
    route_meta: Optional[Dict[str, any]] = None,
) -> Dict[str, any]:
    """Calculate increase speed, decrease speed, keep speed."""
    _ = route_meta

    times, speeds = extract_speed_series(frames)
    if not speeds:
        raise ValueError("No speed values.")

    n = len(speeds)
    result = {
        "n_frames": n,
        "mean_speed_mps": round(sum(speeds) / n, 4),
        "speed_std_mps": round(float(np.std(speeds)), 4),
        # "speed_behavior": "insufficient_data",
        "accelerate_success": None,
        "decelerate_success": None,
        "keep_speed_success": None,
    }

    # 至少6帧我们才开始判断加速，减速，速度保持
    if n < MIN_SPEED_FRAMES:
        return result

    seg_means = segment_means(speeds, 3)
    if len(seg_means) < 3:
        return result

    behavior, evidence = classify_speed_behavior(times, speeds, seg_means)
    result.update(evidence)
    # result["speed_behavior"] = behavior
    result["accelerate_success"] = behavior == "accelerate"
    result["decelerate_success"] = behavior == "decelerate"
    result["keep_speed_success"] = behavior == "keep_speed"
    return result


def aggregate_layer_B_results(
    records_dir: str,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Layer B metrics for every route directory and aggregate summary stats."""
    manifest_by_name = load_manifest(manifest_path)
    route_dirs = sorted(
        d for d in glob.glob(os.path.join(records_dir, "*"))
        if os.path.isdir(d)
    )

    per_route = []
    accelerate_flags = []
    decelerate_flags = []
    keep_speed_flags = []

    for route_dir in route_dirs:
        save_name = os.path.basename(route_dir)
        frames_path = os.path.join(route_dir, "frames.jsonl")
        meta_path = os.path.join(route_dir, "route_meta.json")

        if not os.path.exists(frames_path):
            print(f"{save_name}: no frames.jsonl", file=sys.stderr)
            continue

        frames = load_result(frames_path)
        route_meta = load_result(meta_path) if os.path.exists(meta_path) else {}
        metrics = compute_layer_B_metrics(frames, route_meta)

        route_name = save_name.rsplit("_rep", 1)[0]
        manifest_entry = manifest_by_name.get(route_name, {})

        per_route.append({
            "route_dir": save_name,
            "route_name": route_name,
            "merged_id": manifest_entry.get("merged_id"),
            "source_xml": manifest_entry.get("source_xml"),
            "town": manifest_entry.get("town") or route_meta.get("town_name"),
            "instruction": manifest_entry.get("instruction"),
            "metrics": metrics,
        })

        if metrics.get("accelerate_success") is not None:
            accelerate_flags.append(1.0 if metrics["accelerate_success"] else 0.0)
        if metrics.get("decelerate_success") is not None:
            decelerate_flags.append(1.0 if metrics["decelerate_success"] else 0.0)
        if metrics.get("keep_speed_success") is not None:
            keep_speed_flags.append(1.0 if metrics["keep_speed_success"] else 0.0)

    return {
        "Layer_Type": "Layer_B",
        "records_dir": os.path.abspath(records_dir),
        "manifest": os.path.abspath(manifest_path) if manifest_path else None,
        "total_route_dirs": len(route_dirs),
        "evaluated_routes": len(per_route),
        "summary": {
            "accelerate_rate": _mean(accelerate_flags),
            "decelerate_rate": _mean(decelerate_flags),
            "keep_speed_rate": _mean(keep_speed_flags),
        },
        "per_route": per_route,
    }


########## Layer C ##########
def get_navigation_error(frames: List[Dict], expected_goal: Optional[Dict]) -> float:
    """Calculate the navigation error in meters."""
    if not expected_goal or not frames:
        return None
    loc = frames[-1]['ego']['location']
    return math.sqrt(
        (loc['x'] - expected_goal['x']) ** 2
        + (loc['y'] - expected_goal['y']) ** 2
        + (loc.get('z', 0) - expected_goal.get('z', 0)) ** 2
    )


def get_trajectory_length(frames: List[Dict]) -> float:
    """Calculate the length of the trajectory in meters."""
    total = 0.0
    for i in range(1, len(frames)):
        a = frames[i - 1]['ego']['location']
        b = frames[i]['ego']['location']
        total += math.sqrt(
            (b['x'] - a['x']) ** 2 + (b['y'] - a['y']) ** 2 + (b['z'] - a['z']) ** 2
        )
    return total


def get_shortest_length(route_meta: Dict[str, any]) -> Optional[float]:
    """Calculate GT shortes path from route meta json which is from xml file."""
    waypoints = route_meta.get("waypoints") or []
    if len(waypoints) < 2:
        return None
    total = 0.0
    for i in range(1, len(waypoints)):
        a = waypoints[i - 1]
        b = waypoints[i]
        total += math.sqrt(
            (b["x"] - a["x"]) ** 2
            + (b["y"] - a["y"]) ** 2
            + (b.get("z", 0) - a.get("z", 0)) ** 2
        )
    return total


def _frame_location_2d(frame: Dict[str, Any]) -> Optional[tuple[float, float]]:
    try:
        location = frame["ego"]["location"]
        return float(location["x"]), float(location["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _segment_circle_first_t(
    start: tuple[float, float],
    end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> Optional[float]:
    """Return the first segment parameter intersecting a circle, if any."""
    start_x, start_y = start
    delta_x = end[0] - start_x
    delta_y = end[1] - start_y
    offset_x = start_x - center[0]
    offset_y = start_y - center[1]

    if offset_x * offset_x + offset_y * offset_y <= radius * radius:
        return 0.0

    quadratic_a = delta_x * delta_x + delta_y * delta_y
    if quadratic_a <= 1e-12:
        return None
    quadratic_b = 2.0 * (offset_x * delta_x + offset_y * delta_y)
    quadratic_c = (
        offset_x * offset_x + offset_y * offset_y - radius * radius
    )
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
    if discriminant < 0.0:
        return None
    sqrt_discriminant = math.sqrt(discriminant)
    roots = (
        (-quadratic_b - sqrt_discriminant) / (2.0 * quadratic_a),
        (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a),
    )
    valid_roots = [root for root in roots if 0.0 <= root <= 1.0]
    return min(valid_roots) if valid_roots else None


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / denominator
    ratio = max(0.0, min(1.0, ratio))
    projection = (start[0] + ratio * delta_x, start[1] + ratio * delta_y)
    return math.hypot(point[0] - projection[0], point[1] - projection[1])


def compute_decision_metrics(
    frames: List[Dict[str, Any]],
    decision: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Classify the first committed branch after a key decision center."""
    if not decision:
        return None

    center = decision.get("center") or {}
    candidates = decision.get("candidates") or []
    expected_exit = decision.get("expected_exit")
    result: Dict[str, Any] = {
        "Decision_ID": decision.get("id"),
        "Decision_Pair_ID": decision.get("pair_id"),
        "Expected_Exit": expected_exit,
        "Actual_Exit": "unresolved",
        "Decision_Reached": False,
        "Decision_Covered": False,
        "Decision_Success": False,
        "Decision_Hit_Time": None,
        "Decision_Failure_Reason": "decision_not_reached",
        "Candidate_Min_Distance": {},
    }
    if not center or not candidates:
        result["Decision_Failure_Reason"] = "invalid_decision_schema"
        return result

    samples = []
    for frame_index, frame in enumerate(frames):
        location = _frame_location_2d(frame)
        if location is None:
            continue
        samples.append(
            (
                frame_index,
                float(frame.get("sim_time", frame_index)),
                location,
            )
        )
    if not samples:
        result["Decision_Failure_Reason"] = "no_valid_frames"
        return result

    center_xy = (float(center["x"]), float(center["y"]))
    center_radius = float(center.get("radius_m", 4.0))
    reached_segment = None
    if math.dist(samples[0][2], center_xy) <= center_radius:
        reached_segment = 0
    else:
        for segment_index in range(1, len(samples)):
            if _segment_circle_first_t(
                samples[segment_index - 1][2],
                samples[segment_index][2],
                center_xy,
                center_radius,
            ) is not None:
                reached_segment = segment_index
                break
    if reached_segment is None:
        return result
    result["Decision_Reached"] = True

    candidate_hits = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id"))
        candidate_xy = (float(candidate["x"]), float(candidate["y"]))
        candidate_radius = float(candidate.get("radius_m", 3.0))
        min_distance = math.inf
        first_hit = None

        start_index = max(1, reached_segment)
        for segment_index in range(start_index, len(samples)):
            previous = samples[segment_index - 1]
            current = samples[segment_index]
            min_distance = min(
                min_distance,
                _point_segment_distance(
                    candidate_xy, previous[2], current[2]
                ),
            )
            hit_t = _segment_circle_first_t(
                previous[2],
                current[2],
                candidate_xy,
                candidate_radius,
            )
            if hit_t is not None and first_hit is None:
                hit_time = previous[1] + hit_t * (current[1] - previous[1])
                first_hit = {
                    "candidate_id": candidate_id,
                    "time": hit_time,
                    "frame_index": current[0],
                }
        if reached_segment == 0:
            min_distance = min(
                min_distance, math.dist(samples[0][2], candidate_xy)
            )
        result["Candidate_Min_Distance"][candidate_id] = (
            round(min_distance, 3) if math.isfinite(min_distance) else None
        )
        if first_hit is not None:
            candidate_hits.append(first_hit)

    candidate_hits.sort(key=lambda hit: (hit["time"], hit["candidate_id"]))
    if not candidate_hits:
        result["Decision_Failure_Reason"] = "exit_not_committed"
        return result

    first_hit = candidate_hits[0]
    simultaneous_hits = [
        hit
        for hit in candidate_hits[1:]
        if hit["time"] - first_hit["time"] <= 1.0
    ]
    if simultaneous_hits:
        result["Actual_Exit"] = "ambiguous"
        result["Decision_Failure_Reason"] = "multiple_exit_hits"
        result["Decision_Hit_Time"] = round(first_hit["time"], 3)
        return result

    actual_exit = first_hit["candidate_id"]
    result["Actual_Exit"] = actual_exit
    result["Decision_Covered"] = True
    result["Decision_Success"] = actual_exit == expected_exit
    result["Decision_Hit_Time"] = round(first_hit["time"], 3)
    result["Decision_Failure_Reason"] = (
        None if result["Decision_Success"] else "wrong_exit"
    )
    return result


def compute_layer_C_metrics(
    frames: List[Dict], 
    route_meta: Dict[str, any],
) -> Dict[str, any]:
    """Calculate VLN mertics for layer C."""
    results = {}
    
    # Calculate TL
    trajectory_length = get_trajectory_length(frames)
    results["Trajectory_Length"] = round(trajectory_length, 3)
    
    expected_goal = route_meta.get('expected_goal')
    # Calculate NE
    navigation_error = get_navigation_error(frames, expected_goal)
    results["Navigation_Error"] = navigation_error
    
    # Calculate SR
    if navigation_error is not None:
        goal_success = navigation_error < 5.0
    else:
        goal_success = False
    results["Success"] = goal_success
    
    # Calculate SPL
    l_shortest = get_shortest_length(route_meta)
    if l_shortest is not None and l_shortest > 0:
        spl = goal_success * (l_shortest / max(trajectory_length, l_shortest))
    else:
        spl = None
    results["SPL"] = spl

    decision_metrics = compute_decision_metrics(
        frames,
        route_meta.get("decision_point"),
    )
    if decision_metrics is not None:
        results.update(decision_metrics)
    
    return results


def load_manifest(manifest_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Build lookup tables from a merge_l2d_layer_routes manifest."""
    if not manifest_path:
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    by_route_name: Dict[str, Dict[str, Any]] = {}
    for entry in manifest.get("routes", []):
        route_name = entry.get("route_name")
        if route_name:
            by_route_name[route_name] = entry
    return by_route_name


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def aggregate_layer_C_results(
    records_dir: str,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Layer C metrics for every route directory and aggregate summary stats."""
    manifest_by_name = load_manifest(manifest_path)
    route_dirs = sorted(
        d for d in glob.glob(os.path.join(records_dir, "*"))
        if os.path.isdir(d)
    )

    per_route = []
    success_flags = []
    navigation_errors = []
    spl_values = []
    trajectory_lengths = []
    decision_success_flags = []
    decision_coverage_flags = []
    decision_resolved_success = []
    decision_confusion = {}
    decision_pairs = {}

    for route_dir in route_dirs:
        save_name = os.path.basename(route_dir)
        frames_path = os.path.join(route_dir, "frames.jsonl")
        meta_path = os.path.join(route_dir, "route_meta.json")

        if not os.path.exists(frames_path):
            print(f"{save_name}: no frames.jsonl", file=sys.stderr)
            frames = []
        else:
            frames = load_result(frames_path)
        route_meta = load_result(meta_path) if os.path.exists(meta_path) else {}
        metrics = compute_layer_C_metrics(frames, route_meta)

        route_name = save_name.rsplit("_rep", 1)[0]
        manifest_entry = manifest_by_name.get(route_name, {})

        route_result = {
            "route_dir": save_name,
            "route_name": route_name,
            "merged_id": manifest_entry.get("merged_id"),
            "source_xml": manifest_entry.get("source_xml"),
            "town": manifest_entry.get("town") or route_meta.get("town_name"),
            "instruction": (
                manifest_entry.get("instruction")
                or route_meta.get("instruction")
            ),
            "metrics": metrics,
        }
        per_route.append(route_result)

        if metrics.get("Success") is not None:
            success_flags.append(bool(metrics["Success"]))
        if metrics.get("Navigation_Error") is not None:
            navigation_errors.append(float(metrics["Navigation_Error"]))
        if metrics.get("SPL") is not None:
            spl_values.append(float(metrics["SPL"]))
        if metrics.get("Trajectory_Length") is not None:
            trajectory_lengths.append(float(metrics["Trajectory_Length"]))
        if "Decision_Success" in metrics:
            success = bool(metrics["Decision_Success"])
            covered = bool(metrics["Decision_Covered"])
            decision_success_flags.append(success)
            decision_coverage_flags.append(covered)
            if covered:
                decision_resolved_success.append(success)
            expected = str(metrics.get("Expected_Exit") or "unknown")
            actual = str(metrics.get("Actual_Exit") or "unresolved")
            decision_confusion.setdefault(expected, {})
            decision_confusion[expected][actual] = (
                decision_confusion[expected].get(actual, 0) + 1
            )
            pair_id = metrics.get("Decision_Pair_ID")
            if pair_id:
                decision_pairs.setdefault(str(pair_id), []).append(metrics)

    pair_results = {}
    for pair_id, pair_metrics in sorted(decision_pairs.items()):
        expected_exits = {
            metric.get("Expected_Exit")
            for metric in pair_metrics
            if metric.get("Expected_Exit")
        }
        complete = len(pair_metrics) >= 2 and len(expected_exits) >= 2
        pair_results[pair_id] = {
            "route_count": len(pair_metrics),
            "expected_exits": sorted(expected_exits),
            "complete": complete,
            "success": complete and all(
                bool(metric.get("Decision_Success"))
                for metric in pair_metrics
            ),
        }
    pair_success_flags = [
        bool(pair_result["success"])
        for pair_result in pair_results.values()
    ]

    return {
        "Layer_Type": "Layer_C",
        "records_dir": os.path.abspath(records_dir),
        "manifest": os.path.abspath(manifest_path) if manifest_path else None,
        "total_route_dirs": len(route_dirs),
        "evaluated_routes": len(per_route),
        "summary": {
            "success_rate": round(sum(success_flags) / len(success_flags), 4) if success_flags else None,
            "mean_navigation_error": _mean(navigation_errors),
            "mean_spl": _mean(spl_values),
            "mean_trajectory_length": _mean(trajectory_lengths),
            "decision_routes": len(decision_success_flags),
            "decision_success_rate": _mean(
                [float(flag) for flag in decision_success_flags]
            ),
            "decision_coverage": _mean(
                [float(flag) for flag in decision_coverage_flags]
            ),
            "decision_conditional_accuracy": _mean(
                [float(flag) for flag in decision_resolved_success]
            ),
            "decision_confusion_matrix": decision_confusion,
            "decision_pair_success_rate": _mean(
                [float(flag) for flag in pair_success_flags]
            ),
            "decision_pairs": pair_results,
        },
        "per_route": per_route,
    }


########## Layer D ##########
def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    seq_a, seq_b: shape (N, 2) or (N, 3) arrays of x,y(,z)
    Returns the DTW distance.
    """
    n, m = len(seq_a), len(seq_b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )
    return dtw[n, m]


def compute_layer_D_metrics(
    route_meta: Dict[str, any],
    frames: List[Dict],
) -> Dict[str, Any]:
    """Calculate sequential score by using dtw score."""
    gt_waypoints = route_meta.get("waypoints", [])
    actual_locations = [f["ego"]["location"] for f in frames if "ego" in f]

    seq_a = np.array([[p["x"], p["y"]] for p in gt_waypoints])
    seq_b = np.array([[loc["x"], loc["y"]] for loc in actual_locations])
    
    dtw = dtw_distance(seq_a, seq_b)
    normalized_dtw = dtw / (len(seq_a) + len(seq_b))
    return normalized_dtw


def aggregate_layer_D_results(
    records_dir: str,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Layer D metrics for every route directory and aggregate summary stats."""
    manifest_by_name = load_manifest(manifest_path)
    route_dirs = sorted(
        d for d in glob.glob(os.path.join(records_dir, "*"))
        if os.path.isdir(d)
    )

    per_route = []
    dtw_values = []

    for route_dir in route_dirs:
        save_name = os.path.basename(route_dir)
        frames_path = os.path.join(route_dir, "frames.jsonl")
        meta_path = os.path.join(route_dir, "route_meta.json")

        if not os.path.exists(frames_path):
            print(f"{save_name}: no frames.jsonl", file=sys.stderr)
            continue

        frames = load_result(frames_path)
        route_meta = load_result(meta_path) if os.path.exists(meta_path) else {}
        dtw_score = compute_layer_D_metrics(route_meta, frames)

        route_name = save_name.rsplit("_rep", 1)[0]
        manifest_entry = manifest_by_name.get(route_name, {})

        per_route.append({
            "route_dir": save_name,
            "route_name": route_name,
            "merged_id": manifest_entry.get("merged_id"),
            "source_xml": manifest_entry.get("source_xml"),
            "town": manifest_entry.get("town") or route_meta.get("town_name"),
            "instruction": manifest_entry.get("instruction"),
            "metrics": {"Sequential_Score": dtw_score},
        })

        if dtw_score is not None:
            dtw_values.append(float(dtw_score))

    return {
        "Layer_Type": "Layer_D",
        "records_dir": os.path.abspath(records_dir),
        "manifest": os.path.abspath(manifest_path) if manifest_path else None,
        "total_route_dirs": len(route_dirs),
        "evaluated_routes": len(per_route),
        "summary": {
            "mean_sequential_score": _mean(dtw_values),
        },
        "per_route": per_route,
    }


def aggregate_layer_A_results(
    result_path: str,
    records_dir: str,
    routes_xml: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Layer A metrics for all records in a result file and aggregate."""
    data = load_result(result_path)
    records = data["_checkpoint"]["records"]
    if not records:
        raise ValueError("No route records found in Result JSON File.")

    per_route = []
    driving_scores = []
    route_completions = []
    infraction_penalties = []
    success_flags = []
    smoothness_values = []

    # Shared ability accumulator across all routes
    all_ability_stats = {
        name: {"success_count": 0, "total_count": 0}
        for name in ABILITY_DISPLAY.values()
    }

    for record in records:
        route_id = extract_route_id(record)
        scenario_name = get_scenario_name(record, routes_xml, route_id)

        basic_metrics = compute_layer_A_basic_metrics(record, scenario_name, routes_xml)

        metric_info_path = None
        if record.get("save_name") or record.get("route_id"):
            metric_info_path = find_metric_info(
                records_dir,
                save_name=record.get("save_name"),
                route_id=record.get("route_id"),
            )
        advanced_metrics = computer_layer_A_advanced_metrics(result_path, record, metric_info_path)

        # Accumulate ability counts from this route
        for display_name, stats in basic_metrics["abilities"].items():
            if stats["applicable"]:
                all_ability_stats[display_name]["success_count"] += stats["success_count"]
                all_ability_stats[display_name]["total_count"] += stats["total_count"]

        per_route.append({
            "route_id": record["route_id"],
            "scenario_name": scenario_name,
            "basic_metrics": basic_metrics,
            "advanced_metrics": advanced_metrics,
        })

        ds = record["scores"].get("score_composed")
        if ds is not None:
            driving_scores.append(float(ds))
        rc = record["scores"].get("score_route")
        if rc is not None:
            route_completions.append(float(rc))
        ip = record["scores"].get("score_penalty")
        if ip is not None:
            infraction_penalties.append(float(ip))
        success_flags.append(1.0 if is_success(record) else 0.0)
        sm = advanced_metrics.get("smoothness")
        if sm is not None:
            smoothness_values.append(float(sm))

    # Build aggregated ability scores
    aggregated_abilities = {}
    for display_name, counts in all_ability_stats.items():
        total = counts["total_count"]
        success = counts["success_count"]
        aggregated_abilities[display_name] = {
            "success_count": success,
            "total_count": total,
            "score": round(float(success) / float(total), 4) if total > 0 else None,
        }

    applicable_scores = [
        v["score"] for v in aggregated_abilities.values() if v["score"] is not None
    ]

    return {
        "Layer_Type": "Layer_A",
        "result_file": os.path.abspath(result_path),
        "total_records": len(records),
        "evaluated_routes": len(per_route),
        "summary": {
            "abilities": aggregated_abilities,
            "mean_applicable_ability": _mean(applicable_scores),
            "mean_driving_score": _mean(driving_scores),
            "mean_route_completion": _mean(route_completions),
            "mean_infraction_penalty": _mean(infraction_penalties),
            "success_rate": _mean(success_flags),
            "mean_smoothness": _mean(smoothness_values),
        },
        "per_route": per_route,
    }


def main():
    """Main Eval."""
    parser = argparse.ArgumentParser(description="Compute Language2Drive ability scores")
    
    parser.add_argument("--result", type=str, default=None, help="Leaderboard result.")
    parser.add_argument("--routes-xml", type=str, default=None, help="Route XML.")
    parser.add_argument('--records-dir', default='l2d_records', help='Root directory containing per-route log folders')
    parser.add_argument('--manifest', type=str, default=None, help='Manifest JSON from merge_l2d_layer_routes.py')
    
    parser.add_argument("--layer_A_output", type=str, default=None,)
    parser.add_argument("--layer_B_output", type=str, default=None,)
    parser.add_argument("--layer_C_output", type=str, default=None,)
    parser.add_argument("--layer_D_output", type=str, default=None,)
    args = parser.parse_args()
    
    
    ########## Layer A ##########
    if args.layer_A_output:
        if not args.result:
            raise ValueError("--result is required for Layer A metrics.")
        layer_A_results = aggregate_layer_A_results(
            result_path=args.result,
            records_dir=args.records_dir,
            routes_xml=args.routes_xml,
            manifest_path=args.manifest,
        )
        with open(args.layer_A_output, "w", encoding="utf-8") as f:
            json.dump(layer_A_results, f, indent=2, ensure_ascii=False)
        summary = layer_A_results.get("summary", {})
        print(
            f"Saved Layer A evaluation results to {args.layer_A_output} "
            f"({layer_A_results['evaluated_routes']} routes)"
        )
    
    ########## Layer C ##########
    if args.layer_C_output:
        layer_C_results = aggregate_layer_C_results(
            records_dir=args.records_dir,
            manifest_path=args.manifest,
        )
        with open(args.layer_C_output, "w", encoding="utf-8") as f:
            json.dump(layer_C_results, f, indent=2, ensure_ascii=False)
        summary = layer_C_results.get("summary", {})
        print(
            f"Saved Layer C evaluation results to {args.layer_C_output} "
            f"({layer_C_results['evaluated_routes']} routes)"
        )

    ########## Layer B ##########
    if args.layer_B_output:
        layer_B_results = aggregate_layer_B_results(
            records_dir=args.records_dir,
            manifest_path=args.manifest,
        )
        with open(args.layer_B_output, "w", encoding="utf-8") as f:
            json.dump(layer_B_results, f, indent=2, ensure_ascii=False)
        summary = layer_B_results.get("summary", {})
        print(
            f"Saved Layer B evaluation results to {args.layer_B_output} "
            f"({layer_B_results['evaluated_routes']} routes, "
            f"accelerate_rate={summary.get('accelerate_rate')}, "
            f"decelerate_rate={summary.get('decelerate_rate')}, "
            f"keep_speed_rate={summary.get('keep_speed_rate')})"
        )

    ########## Layer D ##########
    if args.layer_D_output:
        layer_D_results = aggregate_layer_D_results(
            records_dir=args.records_dir,
            manifest_path=args.manifest,
        )
        with open(args.layer_D_output, "w", encoding="utf-8") as f:
            json.dump(layer_D_results, f, indent=2, ensure_ascii=False)
        summary = layer_D_results.get("summary", {})
        print(
            f"Saved Layer D evaluation results to {args.layer_D_output} "
            f"({layer_D_results['evaluated_routes']} routes, "
            f"mean_sequential_score={summary.get('mean_sequential_score')})"
        )

    
if __name__ == "__main__":
    main()
