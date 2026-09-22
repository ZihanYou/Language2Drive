import copy
import json
import math
import unittest
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from language2drive.suite import (
    _actor_program_tokens,
    _semantic_entity_tokens,
    _topology_tokens,
    load_sources,
    validate_suite,
)


ROOT = Path(__file__).resolve().parents[1]


class ReleaseDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pcus = load_sources(ROOT / "sources")
        cls.defaults = json.loads((ROOT / "config/defaults.json").read_text())

    def test_real_payload_split_keys_have_no_dev_test_overlap(self):
        result = validate_suite(self.pcus, self.defaults, True)
        self.assertTrue(result["valid"], result["errors"])
        for kind in ("semantic_entity", "topology", "actor_program"):
            self.assertEqual(result["split_audit"][kind]["leaks"], [], kind)
            self.assertGreater(result["split_audit"][kind]["unique_keys"], 0, kind)

        # Independently recompute each audit, including bound-reference gates
        # used by D memory programs.  This prevents a group-id-only check from
        # silently passing when a physical entity is reused under a new PCU.
        for extractor in (_semantic_entity_tokens, _topology_tokens, _actor_program_tokens):
            seen = defaultdict(set)
            for pcu in self.pcus:
                for token in extractor(pcu):
                    seen[token].add(pcu["split"])
            self.assertFalse({token: splits for token, splits in seen.items() if len(splits) > 1})

    def test_lane_outcomes_commit_only_after_the_proven_terminal_window(self):
        lane_pcus = [pcu for pcu in self.pcus if pcu.get("category") == "lane"]
        self.assertEqual(len(lane_pcus), 8)
        for pcu in lane_pcus:
            self.assertEqual(len(pcu["arms"]), 2)
            first, second = pcu["arms"]
            self.assertEqual(first["contract"], second["opposite_contract"])
            self.assertEqual(second["contract"], first["opposite_contract"])
            for arm in (first, second):
                stages = arm["contract"]["sequence"]
                self.assertEqual(stages[0], {"progress": {"min": 0.9}})
                self.assertEqual(stages[1]["dwell"]["seconds"], 2.0)
                self.assertEqual(
                    stages[1]["dwell"]["condition"]["lane_relation"]["value"],
                    arm["target"]["id"],
                )
            change = next(arm for arm in (first, second) if arm["id"] != "current")
            path = change["oracle_path"]
            self.assertEqual(len(path), 4)
            lengths = [
                math.hypot(end["x"] - start["x"], end["y"] - start["y"])
                for start, end in zip(path, path[1:])
            ]
            self.assertLessEqual(sum(lengths[:2]) / sum(lengths), 0.88)

        baseline = lane_pcus[0]
        for label, mutate in (
            ("old symmetric midpoint gate", lambda value: [
                stage["progress"].__setitem__("min", 0.5)
                for arm in value["arms"]
                for expression in (arm["contract"], arm["opposite_contract"])
                for stage in expression["sequence"][:1]
            ]),
            ("boolean proof number", lambda value:
                value["metadata"]["lane_terminal_commitment_proof"].__setitem__(
                    "commitment_progress", True
                )),
            ("forged connector margin", lambda value:
                value["metadata"]["lane_terminal_commitment_proof"].__setitem__(
                    "change_precommitment_margin_m", 1.0
                )),
        ):
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "terminal lane commitment|finite number|proof.*disagrees"
            ):
                from language2drive.schema import validate_pcu
                validate_pcu(candidate)

        def shift_path(value, axis, amount):
            for arm in value["arms"]:
                for point in arm["oracle_path"]:
                    point[axis] += amount

        def replace_string(value, replacements):
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, str) and child in replacements:
                        value[key] = replacements[child]
                    else:
                        replace_string(child, replacements)
            elif isinstance(value, list):
                for child in value:
                    replace_string(child, replacements)

        source_lane = baseline["metadata"]["lane_terminal_commitment_proof"]["source_lane_id"]
        target_lane = baseline["metadata"]["lane_terminal_commitment_proof"]["target_lane_id"]
        geometry_mutations = (
            ("translated xy", lambda value: (
                shift_path(value, "x", 1000.0), shift_path(value, "y", 1000.0)
            )),
            ("translated z", lambda value: shift_path(value, "z", 1000.0)),
            ("forged raw yaw", lambda value: [
                point.__setitem__("yaw_deg", 123.0)
                for arm in value["arms"] for point in arm["oracle_path"]
            ]),
            ("nonexistent lane identities", lambda value: replace_string(value, {
                source_lane: f"{value['town']}:999:0:3",
                target_lane: f"{value['town']}:999:0:2",
            })),
        )
        for label, mutate in geometry_mutations:
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                "exact terminal lane proof|lane identities are absent|"
                "connector start lacks exact local lane-change permission",
            ):
                from language2drive.schema import validate_pcu
                validate_pcu(candidate)

        def refresh_connector_proof(value):
            change = next(arm for arm in value["arms"] if arm["id"] != "current")
            points = change["oracle_path"]
            segments = [
                math.hypot(end["x"] - start["x"], end["y"] - start["y"])
                for start, end in zip(points, points[1:])
            ]
            total = sum(segments)
            connector = sum(segments[:2])
            suffix = sum(segments[2:])
            proof = value["metadata"]["lane_terminal_commitment_proof"]
            proof["change_connector_end_progress"] = round(connector / total, 9)
            proof["change_precommitment_margin_m"] = round(0.9 * total - connector, 6)
            proof["change_terminal_suffix_m"] = round(suffix, 6)
            proof["exact_lane_sample_counts"]["change_connector_to_terminal"] = (
                max(1, int(math.ceil(suffix / 0.1))) + 1
            )

        # Moving the common source-lane connector point beyond the target-lane
        # endpoint leaves every raw point on a real lane and can preserve all
        # terminal-window scalars.  The complete directed replay must still
        # reject the resulting backwards path[1] -> path[2] chord.
        backwards = copy.deepcopy(baseline)
        proof = backwards["metadata"]["lane_terminal_commitment_proof"]
        graph = json.loads(Path(proof["lane_graph_source_ref"]).read_text())
        source_graph_lane = next(
            lane for lane in graph["lanes"] if lane["id"] == proof["source_lane_id"]
        )
        source_sample = source_graph_lane["samples"][50]
        connector_pose = {
            "x": round(source_sample["x"], 4),
            "y": round(source_sample["y"], 4),
            "z": round(source_sample["z"], 4),
            "yaw_deg": round(source_sample["yaw_deg"], 3),
        }
        for arm in backwards["arms"]:
            arm["oracle_path"][1] = copy.deepcopy(connector_pose)
        refresh_connector_proof(backwards)
        with self.assertRaisesRegex(ValueError, "directed path runs backwards"):
            from language2drive.schema import validate_pcu
            validate_pcu(backwards)

        # A real, same-direction lane on the same road is not sufficient: the
        # target must be the locally permitted adjacent lane.  A-014 exposes a
        # third same-direction lane, whereas A-013 does not.
        nonadjacent = copy.deepcopy(next(pcu for pcu in lane_pcus if pcu["id"] == "A-014"))
        nonadjacent_proof = nonadjacent["metadata"]["lane_terminal_commitment_proof"]
        old_target = nonadjacent_proof["target_lane_id"]
        town, road, section, _lane = old_target.rsplit(":", 3)
        nonadjacent_target = f"{town}:{road}:{section}:3"
        replace_string(nonadjacent, {old_target: nonadjacent_target})
        with self.assertRaisesRegex(ValueError, "not a permitted adjacent lane"):
            from language2drive.schema import validate_pcu
            validate_pcu(nonadjacent)

        # Both the proof and the top-level recursive source map are authority.
        # Removing one side or forging both sides to the same stale digest must
        # fail before any XODR-derived geometry is trusted.
        missing_source = copy.deepcopy(baseline)
        missing_proof = missing_source["metadata"]["lane_terminal_commitment_proof"]
        missing_source["_source_files"].pop(missing_proof["xodr_source_ref"])
        with self.assertRaisesRegex(ValueError, "xodr is absent from source_files"):
            from language2drive.schema import validate_pcu
            validate_pcu(missing_source)

        stale_source = copy.deepcopy(baseline)
        stale_proof = stale_source["metadata"]["lane_terminal_commitment_proof"]
        stale_path = stale_proof["lane_graph_source_ref"]
        stale_proof["lane_graph_sha256"] = "0" * 64
        stale_source["_source_files"][stale_path] = "0" * 64
        with self.assertRaisesRegex(ValueError, "lane_graph source hash is stale"):
            from language2drive.schema import validate_pcu
            validate_pcu(stale_source)

    def test_directed_route_evidence_and_programs_are_frozen(self):
        strict = [pcu for pcu in self.pcus if pcu.get("category") == "strict_order"]
        memory = [pcu for pcu in self.pcus if pcu.get("category") == "memory_reference"]
        self.assertEqual((len(strict), len(memory)), (8, 8))
        for pcu in strict + memory:
            self.assertEqual(pcu["arms"][0]["oracle_path"][0], pcu["arms"][1]["oracle_path"][0])
            for arm in pcu["arms"]:
                self.assertEqual(arm["oracle_route_source"], "offline_carla.GlobalRoutePlanner_prevalidated")
                self.assertTrue(0.8 <= arm["intended_to_legal_ratio"] <= 1.25)
        # The directed port topology contains too few independently feasible
        # pairs to fill six dev programs without reusing semantic entities.
        # Dev strict-order programs therefore use confirmed parking-stage
        # gates, while the confirmed port V11/V12 pair remains in test.  Keep
        # this split explicit so a future generator cannot silently reintroduce
        # the entity leakage that the generic audit above is meant to prevent.
        dev_parking = {token for pcu in strict if pcu["split"] == "dev"
                       for token in _semantic_entity_tokens(pcu)
                       if token.startswith("geometry:parking-stage:")}
        test_port = {token for pcu in strict if pcu["split"] == "test"
                     for token in _semantic_entity_tokens(pcu) if token.startswith("geometry:port:v")}
        self.assertTrue(dev_parking)
        self.assertTrue(test_port)
        self.assertEqual({pcu["town"] for pcu in strict}, {"parkinglotfish-new3", "yangluogang1"})

        source = json.loads((ROOT / "sources/core_pcus.json").read_text())
        geometry = source["geometry_registry"]

        def ordered_gates(expression):
            bindings = {}
            if "bind_reference" in expression:
                binding = expression["bind_reference"]
                bindings[binding["reference"]] = binding["value"]
                expression = binding["body"]
            result = []
            for stage in expression.get("sequence", []):
                identifier = stage.get("enter_gate", {}).get("id")
                if isinstance(identifier, dict):
                    identifier = bindings[identifier["$ref"]]
                if isinstance(identifier, str):
                    result.append(identifier)
            return result

        for pcu in strict + memory:
            for arm in pcu["arms"]:
                expected = ordered_gates(arm["contract"])
                inside = {identifier: False for identifier in expected}
                entered = []
                for point in arm["oracle_path"]:
                    for identifier in expected:
                        gate = geometry[identifier]
                        center = gate["center"]
                        now = math.hypot(point["x"] - center["x"], point["y"] - center["y"]) <= gate["radius_m"]
                        if now and not inside[identifier]:
                            entered.append(identifier)
                        inside[identifier] = now
                self.assertEqual(entered, expected, f"{pcu['id']}/{arm['id']}")

    def test_every_arm_has_independent_route_length_evidence(self):
        arms = [arm for pcu in self.pcus for arm in pcu["arms"]]
        self.assertEqual(len(arms), 256)
        for arm in arms:
            self.assertGreater(arm["canonical_legal_path_length_m"], 0)
            self.assertGreater(arm["intended_path_length_m"], 0)
            self.assertTrue(0.8 <= arm["intended_to_legal_ratio"] <= 1.25)
            self.assertTrue(0.8 <= arm["route_length_ratio"] <= 1.25)
            self.assertIn(arm["oracle_route_source"], {
                "offline_carla.GlobalRoutePlanner",
                "offline_carla.GlobalRoutePlanner_prevalidated",
                "audited_signal_free_lane_slice_v2",
                "road1_monotonic_s_fallback",
                "stored_xodr_parking_v5_prevalidated",
            })

    def test_parking_v5_private_routes_preserve_exact9_and_once_length(self):
        source = json.loads((ROOT / "sources/core_pcus.json").read_text())
        sidecar = json.loads((ROOT / "build/suite/sidecar.private.json").read_text())
        source_arms = {
            (pcu["id"], arm["id"]): arm
            for pcu in source["pcus"] for arm in pcu["arms"]
            if "parking_v5_route_proof" in arm
        }
        routes = [
            route for route in sidecar["routes"]
            if "parking_v5_route_proof" in route
        ]
        self.assertEqual((len(source_arms), len(routes)), (44, 132))
        for route in routes:
            arm = source_arms[(route["pcu_id"], route["arm_id"])]
            self.assertEqual(route["parking_v5_route_proof"], arm["parking_v5_route_proof"])
            self.assertEqual(route["intended_path_length_m"], arm["intended_path_length_m"])

    def test_region_terminal_poses_are_frozen_inside_contract_geometry(self):
        regions = [pcu for pcu in self.pcus if pcu.get("category") == "parking_region"]
        self.assertEqual(len(regions), 8)

        def inside(point, polygon):
            x, y = point
            vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
            result = False
            previous = len(vertices) - 1
            for index, vertex in enumerate(vertices):
                xi, yi = map(float, vertex[:2])
                xj, yj = map(float, vertices[previous][:2])
                if ((yi > y) != (yj > y)
                        and x <= (xj - xi) * (y - yi) / (yj - yi) + xi):
                    result = not result
                previous = index
            return result

        for pcu in regions:
            for arm in pcu["arms"]:
                pose = arm["target"].get("terminal_pose")
                self.assertIsInstance(pose, dict, f"{pcu['id']}/{arm['id']}")
                self.assertTrue(
                    inside((pose["x"], pose["y"]), arm["target"]["polygon"]),
                    f"{pcu['id']}/{arm['id']}",
                )
                self.assertAlmostEqual(arm["oracle_path"][-1]["x"], pose["x"], places=3)
                self.assertAlmostEqual(arm["oracle_path"][-1]["y"], pose["y"], places=3)

    def test_standardized_public_artifacts_are_identity_opaque(self):
        build = ROOT / "build/suite"
        manifest = json.loads((build / "manifest.json").read_text())
        context = json.loads((build / "public_context.json").read_text())
        forbidden = {"pcu_id", "arm_id", "paraphrase", "seed", "l2d_ref", "layer", "split"}
        self.assertNotIn("pcus", manifest)
        self.assertEqual(manifest["routes"], context["routes"])
        self.assertEqual(len(manifest["routes"]), 768)
        for route in manifest["routes"]:
            self.assertTrue(forbidden.isdisjoint(route), route)
            self.assertEqual(set(route), {"route_entry_id", "instruction", "track"})

        root = ET.parse(build / "routes_standardized.xml").getroot()
        self.assertEqual(len(root.findall("route")), 768)
        groups = defaultdict(list)
        sidecar = json.loads((build / "sidecar.private.json").read_text())
        private_by_id = {row["route_entry_id"]: row for row in sidecar["routes"]}
        for route in root.findall("route"):
            route_id = int(route.attrib["id"])
            metadata = route.find("simlingo_metadata")
            self.assertIsNotNone(metadata)
            self.assertTrue(forbidden.isdisjoint(metadata.attrib), metadata.attrib)
            self.assertIsNone(route.find("l2d_ref"))
            private = private_by_id[route_id]
            groups[private["pcu_id"]].append(route)

        # Apart from route id and instruction text, all six public payloads of
        # one PCU (two arms x three expressions) are byte-structurally equal.
        def non_language_shape(route):
            clone = ET.fromstring(ET.tostring(route))
            clone.attrib.pop("id", None)
            instruction = clone.find("instruction")
            if instruction is not None: instruction.text = "<instruction>"
            metadata = clone.find("simlingo_metadata")
            if metadata is not None: metadata.attrib["instruction"] = "<instruction>"
            return ET.tostring(clone)

        self.assertEqual(set(map(len, groups.values())), {6})
        for routes in groups.values():
            self.assertEqual(len({non_language_shape(route) for route in routes}), 1)


if __name__ == "__main__":
    unittest.main()
