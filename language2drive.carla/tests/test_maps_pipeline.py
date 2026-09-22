import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative):
    return json.loads((ROOT / relative).read_text())


class MapPipelineReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_json("generated/maps/registry.json")
        cls.maps = {entry["town"]: entry for entry in cls.registry["maps"]}

    def test_registry_is_complete_and_source_hashes_match(self):
        expected = {
            "Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07",
            "Town10HD", "yangluogang1", "parkinglotfish-new3", "American-city0428",
            "hanghailou3", "hubeiwenli4", "road1",
        }
        self.assertEqual(set(self.maps), expected)
        self.assertEqual(self.registry["errors"], [])
        for entry in self.maps.values():
            source = Path(entry["xodr_path"])
            self.assertTrue(source.is_file(), source)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry["sha256"])
            self.assertEqual(entry["status"], "parsed")
            self.assertGreater(entry["lane_count"], 0)

    def test_lane_graph_references_and_sampling_are_consistent(self):
        for entry in self.maps.values():
            graph = json.loads(Path(entry["lane_graph"]).read_text())
            lanes = {lane["id"]: lane for lane in graph["lanes"]}
            self.assertEqual(len(lanes), entry["lane_count"], entry["town"])
            self.assertEqual(sum(len(x["samples"]) for x in lanes.values()), entry["lane_sample_count"])
            self.assertEqual(len(graph.get("signals", [])), entry["signal_count"])
            self.assertEqual(len(graph.get("objects", [])), entry["object_count"])
            for lane_id, lane in lanes.items():
                town, road, section, signed_lane = lane_id.rsplit(":", 3)
                self.assertEqual(town, entry["town"])
                self.assertEqual(int(road), lane["road_id"])
                self.assertEqual(int(signed_lane), lane["lane_id"])
                samples = lane["samples"]
                self.assertTrue(samples, lane_id)
                for sample in samples:
                    for field in ("s", "x", "y", "z", "yaw_deg", "lane_width_m"):
                        self.assertTrue(math.isfinite(float(sample[field])), (lane_id, field))
                s_values = [float(sample["s"]) for sample in samples]
                if lane["lane_id"] < 0:
                    self.assertEqual(s_values, sorted(s_values), lane_id)
                else:
                    self.assertEqual(s_values, sorted(s_values, reverse=True), lane_id)
                for successor in lane.get("successors", []):
                    self.assertIn(successor, lanes, (lane_id, successor))
                for adjacent in (lane.get("left_adjacent"), lane.get("right_adjacent")):
                    if adjacent is not None:
                        self.assertIn(adjacent, lanes, (lane_id, adjacent))

    def test_road1_uses_explicit_monotonic_fallback(self):
        entry = self.maps["road1"]
        graph = json.loads(Path(entry["lane_graph"]).read_text())
        self.assertEqual(entry["topology_fallback"], "monotonic_s_lane_chain")
        self.assertEqual(graph["fallback"], "monotonic_s_lane_chain")
        self.assertEqual(graph["lane_count"], 4)
        for lane in graph["lanes"]:
            self.assertEqual(lane["successors"], [lane["id"]])
            self.assertEqual(len(lane["samples"]), 2000)

    def test_parking_objects_are_deduplicated_and_bound_core_slots_are_real(self):
        entry = self.maps["parkinglotfish-new3"]
        self.assertEqual(entry["parking_space_objects"], 255)
        self.assertEqual(entry["parking_space_unique"], 249)
        geojson = load_json("generated/maps/parking_spaces.geojson")
        self.assertEqual(len(geojson["features"]), 249)
        rings = []
        for feature in geojson["features"]:
            ring = feature["geometry"]["coordinates"][0]
            self.assertEqual(ring[0], ring[-1])
            self.assertGreaterEqual(len(ring), 5)
            rings.append(tuple((round(float(x), 6), round(float(y), 6)) for x, y, *_ in ring))
        self.assertEqual(len(set(rings)), 249)

        with (ROOT / "assets/review/slot_registry.csv").open(newline="") as handle:
            slots = list(csv.DictReader(handle))
        by_physical_id = {row["physical_space_id"]: row for row in slots if row["physical_space_id"]}
        suite = load_json("sources/core_pcus.json")["pcus"]
        referenced = []
        for pcu in suite:
            for arm in pcu["arms"]:
                target = arm.get("target", {})
                if target.get("type") == "parking_slot":
                    referenced.append(target)
        self.assertTrue(referenced)
        for target in referenced:
            slot_id = target["id"]
            self.assertIn(slot_id, by_physical_id)
            row = by_physical_id[slot_id]
            self.assertEqual(target["slot_number"], row["slot_number"])
            self.assertEqual(target["source_entity_id"], row["entity_id"])
            self.assertTrue(row["status"].startswith("confirmed"), slot_id)
            self.assertIn("bound to parkinglotfish-new3:xodr-parking:", row["notes"])
            self.assertEqual(target["polygon"], json.loads(row["polygon_json"]))
        referenced_numbers = {target["slot_number"] for target in referenced}
        self.assertFalse({"A111", "A112"} & referenced_numbers)
        physical_ids = [target["id"] for target in referenced]
        self.assertEqual(len(set(physical_ids)), len(set(referenced_numbers)))

    def test_known_port_outliers_are_not_used_by_core(self):
        review = load_json("assets/review/existing_label_inventory.json")
        port = json.dumps(review.get("port", review), sort_keys=True)
        self.assertIn("V13", port)
        self.assertIn("V14", port)
        core = json.dumps(load_json("sources/core_pcus.json"), sort_keys=True)
        self.assertNotIn('"port:v13"', core.lower())
        self.assertNotIn('"port:v14"', core.lower())


if __name__ == "__main__":
    unittest.main()
