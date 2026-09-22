import json
import tempfile
import unittest
from pathlib import Path

from language2drive.contracts import ContractEngine, Verdict, score_rollout
from language2drive.generator import _opposite_split_spawn_tokens
from language2drive.io import topology_spawn_token
from language2drive.metrics import aggregate
from language2drive.schema import ArmIdentity, SchemaError, validate_pcu
from language2drive.suite import _topology_tokens, validate_suite


def pcu(identifier="A-001", split="dev", group="g1"):
    predicate = {"enter_gate": {"id": "left"}}
    opposite = {"enter_gate": {"id": "right"}}
    return {"id": identifier, "layer": identifier[0], "split": split, "group_id": group,
            "town": "Town01", "coarse_envelope": [{"x": 0, "y": 0}, {"x": 10, "y": 0}],
            "arms": [{"id": "left", "instructions": ["Turn left.", "Take the left turn.", "Go left."],
                      "contract": predicate, "opposite_contract": opposite, "outcome_key": "left",
                      "oracle_path": [{"x": 0, "y": 0, "z": 0, "yaw_deg": 0},
                                      {"x": 10, "y": 1, "z": 0, "yaw_deg": 6}]},
                     {"id": "right", "instructions": ["Turn right.", "Take the right turn.", "Go right."],
                      "contract": opposite, "opposite_contract": predicate, "outcome_key": "right",
                      "oracle_path": [{"x": 0, "y": 0, "z": 0, "yaw_deg": 0},
                                      {"x": 10, "y": -1, "z": 0, "yaw_deg": -6}]}]}


class SchemaTests(unittest.TestCase):
    def test_spawn_token_uses_shared_bankers_rounding(self):
        self.assertEqual(
            topology_spawn_token("parkinglotfish-new3", -5.5, -28.5),
            "spawn:parkinglotfish-new3:-6:-28",
        )
        self.assertEqual(
            topology_spawn_token("parkinglotfish-new3", 4.5, -29.5),
            "spawn:parkinglotfish-new3:4:-30",
        )

    def test_opposite_split_spawn_reservation_allows_same_split(self):
        same = pcu("D-001", "dev", "dev-program")
        other = pcu("C-025", "test", "test-program")
        same["town"] = other["town"] = "parkinglotfish-new3"
        same["arms"][0]["oracle_path"][0].update({"x": -5.5, "y": -28.5})
        other["arms"][0]["oracle_path"][0].update({"x": 4.5, "y": -29.5})
        self.assertEqual(
            _opposite_split_spawn_tokens([same, other], "dev", "parkinglotfish-new3"),
            {"spawn:parkinglotfish-new3:4:-30"},
        )

    def test_valid_pair(self):
        validate_pcu(pcu())

    def test_rejects_non_mutually_exclusive_outcome(self):
        value = pcu(); value["arms"][1]["outcome_key"] = "left"
        with self.assertRaises(SchemaError): validate_pcu(value)

    def test_identity_is_stable(self):
        self.assertEqual(ArmIdentity("A-001", "left", 2, 3, "standardized").run_id,
                         "standardized__A-001__left__p2__s3")

    def test_split_leakage(self):
        defaults = {"quota": {x: 0 for x in "ABCD"}, "dev_per_layer": 0, "paraphrases_per_arm": 3}
        result = validate_suite([pcu("A-001", "dev", "same"), pcu("A-002", "test", "same")], defaults, False)
        self.assertFalse(result["valid"])

    def test_speed_and_lane_targets_share_physical_road_topology_token(self):
        speed = pcu()
        speed["arms"][0]["signal_free_corridor_proof"] = {
            "lane_id": "Town01:7:0:-1",
        }
        lane = pcu()
        lane["arms"][0]["target"] = {"type": "lane", "id": "Town01:7:0:1"}
        self.assertIn("road:Town01:7", _topology_tokens(speed))
        self.assertIn("road:Town01:7", _topology_tokens(lane))

    def test_public_envelope_does_not_require_private_yaw(self):
        defaults = {"quota": {x: 0 for x in "ABCD"}, "dev_per_layer": 0, "paraphrases_per_arm": 3}
        value = pcu()
        result = validate_suite([value], defaults, False)
        self.assertFalse(any("invalid coarse envelope" in error for error in result["errors"]))

        del value["arms"][0]["oracle_path"][1]["yaw_deg"]
        result = validate_suite([value], defaults, False)
        self.assertTrue(any("missing finite oracle path" in error for error in result["errors"]))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.trace = [
            {"sim_time": 0.0, "ego": {"speed_mps": 2}, "events": [], "progress": 0},
            {"sim_time": 0.7, "ego": {"speed_mps": 3}, "events": ["left"], "progress": .5},
            {"sim_time": 1.9, "ego": {"speed_mps": 0}, "events": [], "progress": 1},
        ]

    def test_sequence_and_until(self):
        engine = ContractEngine(self.trace)
        sequence = {"sequence": [{"enter_gate": {"id": "left"}}, {"progress": {"min": 1}}]}
        self.assertTrue(engine.evaluate(sequence).passed)
        until = {"until": {"invariant": {"safe_legal": {}}, "condition": {"enter_gate": {"id": "left"}}}}
        self.assertTrue(engine.evaluate(until).passed)

    def test_wrong_order_and_opposite_fail(self):
        expression = {"sequence": [{"progress": {"min": 1}}, {"enter_gate": {"id": "left"}}]}
        self.assertFalse(ContractEngine(self.trace).evaluate(expression).passed)
        result = score_rollout(Verdict(True), Verdict(True), Verdict(True), Verdict(True), 2, 10, Verdict(True))
        self.assertFalse(result["success"])

    def test_collision_after_completion_is_unsafe(self):
        trace = self.trace + [{"sim_time": 2.1, "events": ["collision"], "ego": {}, "progress": 1}]
        # always expresses full-rollout safety; its final mask must remain true.
        mask = ContractEngine(trace)._mask({"always": {"safe_legal": {}}})
        self.assertFalse(mask[-1])


class MetricsTests(unittest.TestCase):
    def test_missing_stays_in_denominator_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results = root / "results"; results.mkdir()
            runs = []
            for arm in ("left", "right"):
                identity = ArmIdentity("A-001", arm, 0, 1, "standardized")
                runs.append({**identity.__dict__, "run_id": identity.run_id})
            (root / "ledger.json").write_text(json.dumps({"runs": runs}))
            run_dir = results / runs[0]["run_id"]; run_dir.mkdir()
            (run_dir / "contract_result.json").write_text(json.dumps({"success": True}))
            value = aggregate(root / "ledger.json", results)
            self.assertEqual(value["missing_runs"], 1)
            self.assertEqual(value["CFPS"]["A"], 0)
            self.assertEqual(value["FCS-2x2"]["reason"], "fixed_layout_only")


if __name__ == "__main__":
    unittest.main()
