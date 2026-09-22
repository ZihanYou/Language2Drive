from __future__ import annotations

import csv
import ctypes
import glob
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_json, sha256_file, stable_hash


ANNOTATION_ROOT = (Path(__file__).resolve().parents[2] / "assets/annotations")
POINT_FILES = {
    "parkinglotfish-new3": ANNOTATION_ROOT / "carla_point_jsons/parking.json",
    "yangluogang1": ANNOTATION_ROOT / "carla_point_jsons/port.json",
    "American-city0428": ANNOTATION_ROOT / "carla_point_jsons/urban.json",
}


def _import_carla() -> Any:
    """Import CARLA even in the Pillow-wheel environment used on this host."""
    prefix = Path(sys.prefix)
    # libcarla wheels on this machine link to hashed Pillow libraries. Preload
    # the dependency bundle before the first import; retrying a failed extension
    # import in the same interpreter is unreliable.
    for library in glob.glob(str(prefix / "lib/python*/site-packages/pillow.libs/*.so*")):
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    import carla
    return carla


def _lane_id(town: str, waypoint: Any) -> str:
    return f"{town}:{waypoint.road_id}:{waypoint.section_id}:{waypoint.lane_id}"


def _sample(waypoint: Any) -> dict[str, Any]:
    transform = waypoint.transform
    return {
        "s": round(float(waypoint.s), 3),
        "x": round(float(transform.location.x), 4),
        "y": round(float(transform.location.y), 4),
        "z": round(float(transform.location.z), 4),
        "yaw_deg": round(float(transform.rotation.yaw), 3),
        "junction": bool(waypoint.is_junction),
        "lane_width_m": round(float(waypoint.lane_width), 3),
    }


def _unique_samples(samples: Iterable[Any]) -> list[Any]:
    result: dict[tuple[int, int, int, int], Any] = {}
    for waypoint in samples:
        key = (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id), round(float(waypoint.s) * 10))
        result[key] = waypoint
    return list(result.values())


def _build_lane_graph(town: str, xodr: str, fine_everywhere: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    carla = _import_carla()
    carla_map = carla.Map(town, xodr)
    topology = list(carla_map.get_topology())
    base = list(carla_map.generate_waypoints(0.5 if fine_everywhere else 1.0))
    if not fine_everywhere:
        base.extend(wp for wp in carla_map.generate_waypoints(0.5) if wp.is_junction)
    # CARLA may omit very short lane sections from generate_waypoints while
    # still returning those sections as topology endpoints (Town06 is a known
    # example).  Retain both endpoints so every emitted successor has a stable
    # lane node instead of a dangling reference.
    base.extend(endpoint for edge in topology for endpoint in edge)
    waypoints = _unique_samples(wp for wp in base if wp.lane_type == carla.LaneType.Driving)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for waypoint in waypoints:
        grouped[_lane_id(town, waypoint)].append(waypoint)

    successors: dict[str, set[str]] = defaultdict(set)
    for start, end in topology:
        successors[_lane_id(town, start)].add(_lane_id(town, end))
    fallback = not topology
    if fallback:
        # road1 is a valid straight control map although CARLA reports no
        # topology. Samples are retained as monotonic directed lane chains.
        for lane, values in grouped.items():
            if len(values) > 1:
                successors[lane].add(lane)

    lanes, features = [], []
    for stable_id, values in sorted(grouped.items()):
        values.sort(key=lambda wp: float(wp.s), reverse=int(values[0].lane_id) > 0)
        representative = values[len(values) // 2]
        left, right = representative.get_left_lane(), representative.get_right_lane()
        left_id = _lane_id(town, left) if left is not None and left.lane_type == carla.LaneType.Driving else None
        right_id = _lane_id(town, right) if right is not None and right.lane_type == carla.LaneType.Driving else None
        # Adjacency is a graph edge, not a free-form hint; fail closed when the
        # neighboring section is not represented in the sampled lane registry.
        left_id = left_id if left_id in grouped else None
        right_id = right_id if right_id in grouped else None
        samples = [_sample(waypoint) for waypoint in values]
        lane = {
            "id": stable_id,
            "road_id": int(representative.road_id),
            "section_id": int(representative.section_id),
            "lane_id": int(representative.lane_id),
            "samples": samples,
            "successors": sorted(successors.get(stable_id, set())),
            "left_adjacent": left_id,
            "right_adjacent": right_id,
            "lane_change": str(representative.lane_change).split(".")[-1],
            "is_junction": any(sample["junction"] for sample in samples),
        }
        lanes.append(lane)
        if len(samples) >= 2:
            features.append({
                "type": "Feature",
                "properties": {key: lane[key] for key in ("id", "road_id", "section_id", "lane_id", "is_junction")},
                "geometry": {"type": "LineString", "coordinates": [[p["x"], p["y"], p["z"]] for p in samples]},
            })
    graph = {
        "schema_version": "1.0",
        "town": town,
        "stable_identity": "(town,road_id,section_id,lane_id,s)",
        "sample_resolution_m": 0.5 if fine_everywhere else 1.0,
        "junction_resolution_m": 0.5,
        "fallback": "monotonic_s_lane_chain" if fallback else None,
        "lane_count": len(lanes),
        "sample_count": sum(len(lane["samples"]) for lane in lanes),
        "lanes": lanes,
    }
    return graph, {"type": "FeatureCollection", "features": features}

def _xodr_assets(root: ET.Element) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract versioned traffic-control and roadside-object metadata."""

    signals: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for road in root.findall("road"):
        road_id = int(road.get("id", "0"))
        for signal in road.findall("./signals/signal"):
            signals.append({
                "road_id": road_id, "id": signal.get("id"),
                "s": float(signal.get("s", "0")), "t": float(signal.get("t", "0")),
                "type": signal.get("type"), "subtype": signal.get("subtype"),
                "dynamic": signal.get("dynamic"), "orientation": signal.get("orientation"),
                "value": signal.get("value"), "unit": signal.get("unit"),
            })
        for obj in road.findall("./objects/object"):
            objects.append({
                "road_id": road_id, "id": obj.get("id"), "type": obj.get("type"),
                "name": obj.get("name"), "s": float(obj.get("s", "0")),
                "t": float(obj.get("t", "0")), "hdg": float(obj.get("hdg", "0")),
                "width": float(obj.get("width", "0")), "length": float(obj.get("length", "0")),
            })
    return signals, objects



def _parking_object_counts(root: ET.Element) -> tuple[int, int]:
    values = []
    for road in root.findall("road"):
        for obj in road.findall(".//object[@type='parkingSpace']"):
            values.append((road.get("id"),) + tuple(round(float(obj.get(key, "0")), 3) for key in ("s", "t", "hdg", "width", "length")))
    return len(values), len(set(values))


def _reference_pose(road: ET.Element, s_value: float) -> tuple[float, float, float]:
    """Evaluate an OpenDRIVE road reference line, including clothoids."""

    geometries = road.findall("./planView/geometry")
    geometry = max((value for value in geometries if float(value.get("s", "0")) <= s_value + 1e-8),
                   key=lambda value: float(value.get("s", "0")))
    local = max(0.0, min(s_value - float(geometry.get("s", "0")), float(geometry.get("length", "0"))))
    x, y, heading = (float(geometry.get(key, "0")) for key in ("x", "y", "hdg"))
    child = list(geometry)[0]
    if child.tag == "line":
        return x + local*math.cos(heading), y + local*math.sin(heading), heading
    if child.tag == "arc":
        curvature = float(child.get("curvature", "0"))
        if abs(curvature) < 1e-12:
            return x + local*math.cos(heading), y + local*math.sin(heading), heading
        end_heading = heading + curvature*local
        return (x + (math.sin(end_heading)-math.sin(heading))/curvature,
                y - (math.cos(end_heading)-math.cos(heading))/curvature, end_heading)
    if child.tag == "spiral":
        k0, k1 = float(child.get("curvStart", "0")), float(child.get("curvEnd", "0"))
        length = max(float(geometry.get("length", "0")), 1e-12)
        steps = max(1, int(math.ceil(local/0.05))); step = local/steps
        for index in range(steps):
            midpoint = (index+0.5)*step
            angle = heading + k0*midpoint + 0.5*(k1-k0)*midpoint*midpoint/length
            x += step*math.cos(angle); y += step*math.sin(angle)
        return x, y, heading + k0*local + 0.5*(k1-k0)*local*local/length
    raise ValueError(f"unsupported OpenDRIVE geometry {child.tag}")


def _parking_objects_geojson(root: ET.Element, town: str) -> dict[str, Any]:
    """Convert de-duplicated XODR parkingSpace objects to world polygons."""

    features, seen = [], set()
    for road in root.findall("road"):
        for obj in road.findall(".//object[@type='parkingSpace']"):
            identity = (road.get("id"),) + tuple(round(float(obj.get(key, "0")), 3)
                                                     for key in ("s", "t", "hdg", "width", "length"))
            if identity in seen:
                continue
            seen.add(identity)
            s_value, t_value = float(obj.get("s", "0")), float(obj.get("t", "0"))
            ref_x, ref_y, ref_heading = _reference_pose(road, s_value)
            # CARLA/OpenDRIVE maps use a left-handed world Y axis relative to
            # the standard OpenDRIVE inertial frame.  In this exported map the
            # object s/t anchor is a corner; the parking-space center is one
            # half length along object heading from that anchor.
            anchor = [ref_x-math.sin(ref_heading)*t_value, -(ref_y+math.cos(ref_heading)*t_value)]
            heading = ref_heading+float(obj.get("hdg", "0"))
            if obj.get("orientation") == "-":
                heading += math.pi
            world_heading = -heading
            half_length = float(obj.get("length", "0"))/2.0
            center = [anchor[0]+half_length*math.cos(world_heading), anchor[1]+half_length*math.sin(world_heading)]
            polygon = _rectangle(center, math.degrees(world_heading), float(obj.get("length", "0")), float(obj.get("width", "0")))
            features.append({"type":"Feature","properties":{"entity_id":f"{town}:xodr-parking:{len(features):03d}",
                              "town":town,"road_id":int(road.get("id","0")),"object_id":obj.get("id"),
                              "s":s_value,"t":t_value,"heading_deg":math.degrees(world_heading),
                              "width_m":float(obj.get("width","0")),"length_m":float(obj.get("length","0")),
                              "status":"provisional_xodr"},
                             "geometry":{"type":"Polygon","coordinates":[polygon]}})
    return {"type":"FeatureCollection","features":features}


def _project_annotations(town: str, carla_map: Any, point_path: Path) -> list[dict[str, Any]]:
    carla = _import_carla()
    points = json.loads(point_path.read_text(encoding="utf-8"))
    features = []
    for point in points:
        location = point["location"]
        source_name = str(point["word"])
        name = re.sub(r"^V\d+_", "", source_name)
        query = carla.Location(x=float(location["x"]), y=float(location["y"]), z=float(location.get("z", 0)))
        waypoint = carla_map.get_waypoint(query, project_to_road=True, lane_type=carla.LaneType.Driving)
        projection_m = query.distance(waypoint.transform.location) if waypoint is not None else math.inf
        status = "confirmed_existing_annotation" if projection_m <= 5.0 else "needs_fix"
        properties = {
            "entity_id": f"{town}:{source_name}", "town": town, "canonical_text": name,
            "aliases": [], "category": "semantic_waypoint", "source_label": source_name,
            "yaw_deg": float(point.get("rotation", {}).get("yaw", 0)),
            "lane_id": _lane_id(town, waypoint) if waypoint is not None else None,
            "lane_s": round(float(waypoint.s), 3) if waypoint is not None else None,
            "projection_distance_m": round(float(projection_m), 3) if math.isfinite(projection_m) else None,
            "status": status, "source_path": str(point_path),
        }
        if town == "parkinglotfish-new3" and re.search(r"[A-F]\d{3}", name):
            properties["category"] = "parking_slot"
        elif town == "yangluogang1":
            properties["category"] = "port_destination"
        elif town == "American-city0428":
            properties["category"] = "junction_waypoint"
        features.append({"type": "Feature", "properties": properties,
                         "geometry": {"type": "Point", "coordinates": [float(location["x"]), float(location["y"]), float(location.get("z", 0))]}})
    return features


def _rectangle(center: list[float], yaw_deg: float, length: float = 5.5, width: float = 3.0) -> list[list[float]]:
    yaw = math.radians(yaw_deg)
    forward, side = (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))
    result = []
    for longitudinal, lateral in ((length/2, width/2), (length/2, -width/2), (-length/2, -width/2), (-length/2, width/2)):
        result.append([center[0] + forward[0]*longitudinal + side[0]*lateral,
                       center[1] + forward[1]*longitudinal + side[1]*lateral])
    result.append(result[0])
    return result


def _polygon_center(polygon: list[list[float]]) -> list[float]:
    points = polygon[:-1] if polygon and polygon[0] == polygon[-1] else polygon
    return [sum(point[0] for point in points)/len(points), sum(point[1] for point in points)/len(points)]


def _write_review_assets(features: list[dict[str, Any]], review_dir: Path,
                         parking_spaces: dict[str, Any] | None = None) -> dict[str, int]:
    review_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(review_dir / "semantic_entities.geojson", {"type": "FeatureCollection", "features": features})
    with (review_dir / "asset_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("entity_id", "town", "status", "reviewer", "reviewed_at", "notes"))
        writer.writeheader()
        for feature in features:
            p = feature["properties"]
            writer.writerow({"entity_id": p["entity_id"], "town": p["town"], "status": p["status"],
                             "reviewer": "migrated_existing_annotation", "reviewed_at": "2026-08-12",
                             "notes": f"projection_distance_m={p.get('projection_distance_m')}"})

    slot_features: dict[str, dict[str, Any]] = {}
    for feature in features:
        p = feature["properties"]
        if p.get("category") != "parking_slot":
            continue
        match = re.search(r"([A-F]\d{3})", p["canonical_text"])
        if not match:
            continue
        slot = match.group(1)
        if slot in slot_features:
            continue
        slot_features[slot] = feature

    # A nearest-neighbour lookup independently performed for every label can
    # bind adjacent labels to the same anonymous XODR parkingSpace.  Solve one
    # global rectangular assignment so each confirmed slot has unique physical
    # geometry.  A deterministic greedy fallback keeps map building usable in
    # minimal environments without SciPy.
    candidates = parking_spaces.get("features", []) if parking_spaces else []
    slot_names = sorted(slot_features)
    assignments: dict[str, tuple[dict[str, Any], float]] = {}
    if slot_names and candidates:
        distances = [[math.dist(slot_features[name]["geometry"]["coordinates"][:2],
                               _polygon_center(item["geometry"]["coordinates"][0]))
                      for item in candidates] for name in slot_names]
        try:
            from scipy.optimize import linear_sum_assignment
            rows, columns = linear_sum_assignment(distances)
            pairs = [(int(row), int(column), float(distances[int(row)][int(column)]))
                     for row, column in zip(rows, columns)]
        except ImportError:
            all_pairs = sorted((float(distance), row, column)
                               for row, values in enumerate(distances)
                               for column, distance in enumerate(values))
            used_rows, used_columns, pairs = set(), set(), []
            for distance, row, column in all_pairs:
                if row in used_rows or column in used_columns:
                    continue
                used_rows.add(row); used_columns.add(column)
                pairs.append((row, column, distance))
                if len(used_rows) == len(slot_names):
                    break
        for row, column, distance in pairs:
            if distance <= 6.0:
                assignments[slot_names[row]] = (candidates[column], distance)

    slots: dict[str, dict[str, Any]] = {}
    for slot in slot_names:
        feature = slot_features[slot]
        p = feature["properties"]
        center = feature["geometry"]["coordinates"]
        binding, binding_distance = assignments.get(slot, (None, math.inf))
        polygon = (binding["geometry"]["coordinates"][0] if binding else _rectangle(center, float(p["yaw_deg"])))
        heading = float(binding["properties"]["heading_deg"] if binding else p["yaw_deg"])
        effective_status = p["status"] if binding else "provisional_unbound_polygon"
        slots[slot] = {"entity_id": p["entity_id"],
                       "physical_space_id": binding["properties"]["entity_id"] if binding else "",
                       "town": p["town"], "slot_number": slot,
                       "region": slot[0], "polygon": polygon, "heading_deg": heading,
                       "status": effective_status, "notes": (f"one-to-one bound to {binding['properties']['entity_id']} from XODR parkingSpace; distance_m={binding_distance:.3f}"
                           if binding else "unbound: 5.5m x 3.0m fallback from existing target pose")}
    with (review_dir / "slot_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("entity_id", "physical_space_id", "town", "slot_number", "region", "polygon_json", "heading_deg", "status", "notes")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for slot in sorted(slots.values(), key=lambda row: row["slot_number"]):
            row = dict(slot); row["polygon_json"] = json.dumps(row.pop("polygon"), separators=(",", ":")); writer.writerow(row)

    port_graph = json.loads((ANNOTATION_ROOT / "trj_seq_jsons/port_traj_graph_sequence_all_start.json").read_text(encoding="utf-8"))
    with (review_dir / "sign_routes.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("entity_id", "town", "canonical_text", "aliases", "arrow", "outgoing_lane", "target_region", "status", "notes")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for path in port_graph["paths"]:
            if not path.get("reachable") or len(path.get("vertex_indices_along_path", [])) < 2:
                continue
            writer.writerow({"entity_id": f"yangluogang1:path:{path['path_id']}", "town": "yangluogang1",
                             "canonical_text": path["end_name"], "aliases": "", "arrow": "graph_next",
                             "outgoing_lane": path["vertex_indices_along_path"][1], "target_region": path["end_name"],
                             "status": "confirmed_existing_annotation", "notes": "migrated directed instruction chain"})
    return {"features": len(features), "slots": len(slots),
            "confirmed": sum(f["properties"]["status"].startswith("confirmed") for f in features),
            "needs_fix": sum(f["properties"]["status"] == "needs_fix" for f in features)}


def build_registry(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    xodr_root, graph_dir = Path(config["xodr_root"]), output.parent / "lane_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    records, errors, semantic_features = [], [], []
    parking_spaces: dict[str, Any] | None = None
    carla = _import_carla()
    for item in config["maps"]:
        path = xodr_root / f"{item['town']}.xodr"
        if not path.is_file():
            errors.append(f"missing XODR: {path}"); continue
        try:
            xodr = path.read_text(encoding="utf-8")
            root = ET.fromstring(xodr)
            carla_map = carla.Map(item["town"], xodr)
            graph, geojson = _build_lane_graph(item["town"], xodr, fine_everywhere=item["town"] == "parkinglotfish-new3")
            signals, objects = _xodr_assets(root)
            graph["signal_count"], graph["object_count"] = len(signals), len(objects)
            graph["signals"], graph["objects"] = signals, objects
            atomic_json(graph_dir / f"{item['town']}.lane_graph.json", graph)
            atomic_json(graph_dir / f"{item['town']}.lanes.geojson", geojson)
            if item["town"] == "parkinglotfish-new3":
                parking_spaces = _parking_objects_geojson(root, item["town"])
                atomic_json(output.parent / "parking_spaces.geojson", parking_spaces)
            if item["town"] in POINT_FILES:
                semantic_features.extend(_project_annotations(item["town"], carla_map, POINT_FILES[item["town"]]))
        except Exception as error:
            errors.append(f"invalid map {path}: {error}"); continue
        raw_parking, unique_parking = _parking_object_counts(root)
        records.append({**item, "xodr_path": str(path), "sha256": sha256_file(path),
                        "coordinate_system": "OpenDRIVE local ENU", "road_count": len(root.findall("road")),
                        "junction_count": len(root.findall("junction")), "parking_space_objects": raw_parking,
                        "parking_space_unique": unique_parking, "lane_count": graph["lane_count"],
                        "lane_sample_count": graph["sample_count"], "signal_count": graph["signal_count"],
                        "object_count": graph["object_count"], "topology_fallback": graph["fallback"],
                        "lane_graph": str(graph_dir / f"{item['town']}.lane_graph.json"),
                        "lane_geojson": str(graph_dir / f"{item['town']}.lanes.geojson"), "status": "parsed"})

    review_dir = config_path.parent.parent / "assets/review"
    review = _write_review_assets(semantic_features, review_dir, parking_spaces)
    result = {"schema_version": "1.0", "maps": records, "errors": errors, "semantic_review": review,
              "source_hash": stable_hash({"config": json.loads(config_path.read_text()), "maps": [r["sha256"] for r in records]}),
              "note": "Lane connectivity comes from offline carla.Map; legacy *_road_graph.json files are visualization-only."}
    atomic_json(output, result)
    return result


def review_gaps(review_dir: Path) -> dict[str, Any]:
    entities = json.loads((review_dir / "semantic_entities.geojson").read_text(encoding="utf-8"))
    statuses: dict[str, int] = defaultdict(int)
    with (review_dir / "asset_review.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            statuses[row.get("status", "").strip() or "unreviewed"] += 1
    confirmed = {f.get("properties", {}).get("entity_id") for f in entities.get("features", [])
                 if str(f.get("properties", {}).get("status", "")).startswith("confirmed")}
    return {"geojson_features": len(entities.get("features", [])), "confirmed_entities": len(confirmed),
            "review_statuses": dict(statuses), "ready_for_semantic_suite": bool(confirmed),
            "minimum_human_work": "review only rows with status=needs_fix"}
