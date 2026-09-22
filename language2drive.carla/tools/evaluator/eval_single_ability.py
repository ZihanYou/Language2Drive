#!/usr/bin/env python3
"""
Compute Bench2Drive multi-ability scores for a single-route result JSON.

Logic follows Bench2Drive/tools/ability_benchmark.py:
  - Overtaking, Merging, Emergency_Brake, Give_Way: general success rule
  - Traffic_Signs: general rule + optional CARLA junction check (same as official)

Usage:
    python tools/eval_single_ability.py --result result_0.json
    python tools/eval_single_ability.py --result result_0.json \\
        --routes-xml language2drive/leaderboard/data/bench2drive220.xml \\
        --carla --carla-root /path/to/carla
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "Traffic_Signs": [
        "BlockedIntersection", "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight",
        "PedestrianCrossing", "VehicleTurningRoute", "VehicleTurningRoutePedestrian",
        "EnterActorFlow", "CrossingBicycleFlow", "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurnEnterFlow",
        "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow", "T_Junction", "VanillaNonSignalizedTurn",
        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurnEncounterStopsign",
    ],
}

ABILITY_DISPLAY = {
    "Overtaking": "overtaking",
    "Merging": "merging",
    "Emergency_Brake": "emergency_brake",
    "Give_Way": "give_way",
    "Traffic_Signs": "traffic_sign",
}


def get_infraction_status(record):
    for infraction, value in record["infractions"].items():
        if infraction == "min_speed_infractions":
            continue
        if len(value) > 0:
            return True
    return False


def update_ability(scenario_name, ability_statistic, status):
    for ability, scenarios in Ability.items():
        if scenario_name in scenarios:
            ability_statistic[ability][1] += 1
            if status:
                ability_statistic[ability][0] += 1


def get_position(xml_route):
    import carla

    waypoints_elem = xml_route.find("waypoints")
    keypoints = waypoints_elem.findall("position")
    return [
        carla.Location(float(pos.get("x")), float(pos.get("y")), float(pos.get("z")))
        for pos in keypoints
    ]


def get_waypoint_route(locs, grp):
    route = []
    for i in range(len(locs) - 1):
        interpolated_trace = grp.trace_route(locs[i], locs[i + 1])
        for wp, _ in interpolated_trace:
            route.append(wp)
    return route


def load_result(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_record(data):
    records = data["_checkpoint"]["records"]
    if not records:
        raise ValueError("No route records found in result JSON.")
    if len(records) > 1:
        print(f"Warning: found {len(records)} records, using the first one.", file=sys.stderr)
    return records[0]


def get_record_success_status(record):
    if record["status"] in ("Completed", "Perfect"):
        return not get_infraction_status(record)
    return False


def find_scenario_name(record, routes_xml, route_id):
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


def extract_route_id(record):
    return record["route_id"].split("_")[1]


def ability_categories_for_scenario(scenario_name):
    return [ability for ability, scenarios in Ability.items() if scenario_name in scenarios]


def find_route_xml(routes_xml, route_id):
    tree = ET.parse(routes_xml)
    for route in tree.getroot().findall("route"):
        if route.get("id") == route_id:
            return route
    return None


def start_carla(carla_root, port):
    carla_sh = os.path.join(carla_root, "CarlaUE4.sh")
    if not os.path.isfile(carla_sh):
        raise FileNotFoundError(f"CARLA not found at {carla_sh}")

    cmd = f"{carla_sh} -RenderOffScreen -nosound -carla-rpc-port={port}"
    server = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    print(f"Starting CARLA: {cmd}", flush=True)
    time.sleep(60)
    return server


def apply_traffic_signs_override(ability_statistic, route_record, route_xml, carla_client):
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    town = route_xml.get("town")
    world = carla_client.load_world(town)
    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, 1.0)

    waypoint_route = get_waypoint_route(get_position(route_xml), grp)

    count = 0
    for wp in waypoint_route:
        count += 1
        if wp.is_junction:
            break
    if not wp.is_junction:
        raise RuntimeError("This route does not contain any junction-waypoint!")

    junction_completion = float(count + 8) / float(len(waypoint_route))
    record_completion = route_record["scores"]["score_route"] / 100.0
    stop_infraction = route_record["infractions"]["stop_infraction"]
    red_light_infraction = route_record["infractions"]["red_light"]

    if (
        record_completion > junction_completion
        and not stop_infraction
        and not red_light_infraction
    ):
        ability_statistic["Traffic_Signs"][0] += 1
        ability_statistic["Traffic_Signs"][1] += 1
    else:
        ability_statistic["Traffic_Signs"][1] += 1


def compute_abilities(
    record,
    scenario_name,
    routes_xml=None,
    use_carla=False,
    carla_root=None,
    carla_host="localhost",
    carla_port=4000,
):
    record_success_status = get_record_success_status(record)
    ability_statistic = {key: [0, 0.0] for key in Ability}

    update_ability(scenario_name, ability_statistic, record_success_status)

    traffic_signs_note = None
    if scenario_name in Ability["Traffic_Signs"]:
        if use_carla and routes_xml is not None:
            route_id = extract_route_id(record)
            route_xml = find_route_xml(routes_xml, route_id)
            if route_xml is None:
                traffic_signs_note = (
                    f"Route id {route_id} not found in {routes_xml}; "
                    "Traffic_Signs junction override skipped."
                )
            else:
                import carla

                server = start_carla(carla_root, carla_port)
                try:
                    client = carla.Client(carla_host, carla_port)
                    client.set_timeout(300)
                    apply_traffic_signs_override(
                        ability_statistic, record, route_xml, client
                    )
                finally:
                    server.terminate()
        else:
            traffic_signs_note = (
                "Traffic_Signs junction override skipped. "
                "Pass --routes-xml and --carla to match official evaluation."
            )

    applicable = ability_categories_for_scenario(scenario_name)
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
        "traffic_signs_note": traffic_signs_note,
    }


def print_report(report):
    print("=" * 60)
    print("Single Route Ability Evaluation")
    print("=" * 60)
    print(f"Route ID         : {report['route_id']}")
    print(f"Scenario name    : {report['scenario_name'] or '(empty)'}")
    print(f"General success  : {report['record_success_status']}")
    print(f"Applicable       : {report['applicable_abilities'] or 'none'}")
    print("-" * 60)

    for ability in [
        "overtaking",
        "merging",
        "emergency_brake",
        "give_way",
        "traffic_sign",
    ]:
        item = report["abilities"][ability]
        if item["applicable"]:
            print(
                f"{ability:16s}: {item['score']:.4f}  "
                f"({item['success_count']}/{item['total_count']})"
            )
        else:
            print(f"{ability:16s}: N/A (scenario not in this category)")

    print("-" * 60)
    if report["mean_applicable_only"] is not None:
        print(f"Mean (applicable only) : {report['mean_applicable_only']:.4f}")
    else:
        print("Mean (applicable only) : N/A")

    if report["mean_official_style"] is not None:
        print(f"Mean (official /5)     : {report['mean_official_style']:.4f}")

    if report["traffic_signs_note"]:
        print(f"Note: {report['traffic_signs_note']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Bench2Drive ability scores for one result JSON."
    )
    parser.add_argument(
        "--result",
        type=str,
        default=os.path.join(_REPO_ROOT, "result_0.json"),
        help="Single-route leaderboard result JSON.",
    )
    parser.add_argument(
        "--routes-xml",
        type=str,
        default=None,
        help="Route XML used to read scenario type and Traffic_Signs junction check.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Override scenario type when result/json or routes xml has empty type.",
    )
    parser.add_argument(
        "--carla",
        action="store_true",
        help="Start CARLA for Traffic_Signs junction override (official behavior).",
    )
    parser.add_argument(
        "--carla-root",
        type=str,
        default=os.environ.get("CARLA_ROOT"),
        help="CARLA root directory.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON report.",
    )
    args = parser.parse_args()

    data = load_result(args.result)
    record = get_record(data)
    route_id = extract_route_id(record)

    scenario_name = args.scenario_name
    if not scenario_name:
        scenario_name = find_scenario_name(record, args.routes_xml, route_id)

    if args.carla and not args.carla_root:
        parser.error("--carla requires CARLA_ROOT or --carla-root.")

    report = compute_abilities(
        record=record,
        scenario_name=scenario_name,
        routes_xml=args.routes_xml,
        use_carla=args.carla,
        carla_root=args.carla_root,
        carla_host=args.host,
        carla_port=args.port,
    )
    report["result_file"] = os.path.abspath(args.result)
    report["routes_xml"] = os.path.abspath(args.routes_xml) if args.routes_xml else None

    print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
