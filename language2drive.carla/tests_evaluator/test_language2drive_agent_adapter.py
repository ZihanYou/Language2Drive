from __future__ import annotations

import ast
from contextlib import redirect_stderr
import io
import json
import os
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np

from team_code.language2drive_context import (
    CheckpointCompatibilityError,
    L2D_TRACK_LEGACY,
    L2D_TRACK_NATIVE,
    L2D_TRACK_STANDARDIZED,
    PRIVILEGED_ORACLE_CAPABILITY,
    PublicContextError,
    PublicContextState,
    bind_privileged_oracle_context,
    make_public_context,
    normalize_l2d_track,
    seed_everything,
    validate_checkpoint_coverage,
)
from team_code.language2drive_external_carla import (
    ExternalCarlaConnectionError,
    connect_existing_carla,
)
from leaderboard.utils.closed_loop_logger import (
    ClosedLoopLogger,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    RunningRedLightTest,
)
from srunner.scenariomanager.scenarioatomics import atomic_criteria


REPO_ROOT = Path(__file__).resolve().parents[1]


class _TensorShape:
    def __init__(self, elements: int):
        self.elements = elements

    def numel(self) -> int:
        return self.elements


class PublicContextTest(unittest.TestCase):
    def test_strict_legacy_alias_and_explicit_track(self) -> None:
        self.assertEqual(
            L2D_TRACK_STANDARDIZED, normalize_l2d_track(None, strict_vln=True)
        )
        self.assertEqual(L2D_TRACK_LEGACY, normalize_l2d_track("off"))
        self.assertEqual("native", normalize_l2d_track("native-system"))

    def test_standardized_context_is_minimal(self) -> None:
        context = make_public_context(
            instruction="Park in A103.",
            route_name="RouteScenario_7_rep0",
            seed=2,
            track="standardized",
        )
        self.assertEqual(
            {
                "instruction",
                "track",
                "instruction_version",
                "schema_version",
            },
            set(context),
        )
        self.assertFalse(
            {
                "route_name", "seed", "arm_id", "route_entry_id", "target",
                "goal", "gate", "polygon", "lane", "waypoints",
            }
            & set(context)
        )

    def test_standardized_mapping_rejects_evaluator_owned_identity(self) -> None:
        base = {
            "instruction": "Take the left fork.",
            "track": "standardized",
            "instruction_version": 0,
            "schema_version": "language2drive.public_context.v1",
        }
        for key, value in (
            ("route_name", "RouteScenario_0_rep0"),
            ("seed", 1),
        ):
            with self.subTest(key=key), self.assertRaises(PublicContextError):
                PublicContextState().bind({**base, key: value})

    def test_privileged_oracle_binding_is_native_only_and_capability_gated(self) -> None:
        private = {"route": {"route_entry_id": 7, "oracle_path": [1, 2]}}

        class OrdinaryAgent:
            pass

        self.assertFalse(
            bind_privileged_oracle_context(OrdinaryAgent(), private, L2D_TRACK_NATIVE)
        )

        class OracleAgent:
            language2drive_private_context_capability = PRIVILEGED_ORACLE_CAPABILITY

            def __init__(self):
                self.bound = None

            def set_language2drive_private_context(self, value):
                self.bound = value

        agent = OracleAgent()
        with self.assertRaisesRegex(PublicContextError, "native track"):
            bind_privileged_oracle_context(
                agent, private, L2D_TRACK_STANDARDIZED
            )
        self.assertIsNone(agent.bound)
        self.assertTrue(
            bind_privileged_oracle_context(agent, private, L2D_TRACK_NATIVE)
        )
        self.assertEqual(7, agent.bound["route"]["route_entry_id"])
        self.assertIsNot(agent.bound, private)

    def test_standardized_instruction_is_required(self) -> None:
        with self.assertRaises(PublicContextError):
            make_public_context(
                instruction=None,
                route_name="RouteScenario_0_rep0",
                seed=1,
                track="standardized",
            )

    def test_non_public_fields_are_rejected(self) -> None:
        context = make_public_context(
            instruction="Take the left fork.",
            route_name="RouteScenario_0_rep0",
            seed=1,
            track="standardized",
        )
        context["target_gate"] = {"x": 1.0, "y": 2.0}
        with self.assertRaises(PublicContextError):
            PublicContextState().bind(context)

    def test_rebinding_replaces_instruction_without_cross_route_state(self) -> None:
        state = PublicContextState()
        first = make_public_context(
            instruction="Use the west entrance.",
            route_name="RouteScenario_0_rep0",
            seed=1,
            track="standardized",
        )
        second = make_public_context(
            instruction="Use the east entrance.",
            route_name="RouteScenario_1_rep0",
            seed=3,
            track="standardized",
        )
        state.bind(first)
        state.bind(second)
        self.assertEqual("Use the east entrance.", state.context.instruction)
        self.assertIsNone(state.context.route_name)
        self.assertIsNone(state.context.seed)
        self.assertEqual(
            {key: value for key, value in first.items() if key != "instruction"},
            {key: value for key, value in second.items() if key != "instruction"},
        )

    def test_real_pcu_six_entries_have_instruction_only_payload_difference(self) -> None:
        xml_path = REPO_ROOT / "build/suite/routes_standardized.xml"
        routes = ET.parse(xml_path).getroot().findall("route")
        self.assertEqual(768, len(routes))
        sidecar = json.loads(
            (
                REPO_ROOT / "build/suite/sidecar.private.json"
            ).read_text(encoding="utf-8")
        )
        private_pcu_by_entry = {
            int(route["route_entry_id"]): route["pcu_id"]
            for route in sidecar["routes"]
        }
        routes_by_pcu = {}
        for route in routes:
            pcu_id = private_pcu_by_entry[int(route.attrib["id"])]
            routes_by_pcu.setdefault(pcu_id, []).append(route)
            metadata = route.find("simlingo_metadata").attrib
            self.assertFalse(
                {
                    "pcu_id", "arm_id", "paraphrase", "route_entry_id",
                    "seed", "l2d_ref",
                }
                & set(metadata)
            )
        self.assertEqual(128, len(routes_by_pcu))
        for pcu_id, members in sorted(routes_by_pcu.items()):
            with self.subTest(pcu_id=pcu_id):
                self.assertEqual(6, len(members))
                payloads = [
                    make_public_context(
                        instruction=route.findtext("instruction"),
                        route_name="RouteScenario_{}_rep0".format(
                            route.attrib["id"]
                        ),
                        seed=(index % 3) + 1,
                        track="standardized",
                    )
                    for index, route in enumerate(members)
                ]
                non_instruction = [
                    {
                        key: value for key, value in payload.items()
                        if key != "instruction"
                    }
                    for payload in payloads
                ]
                self.assertTrue(
                    all(value == non_instruction[0] for value in non_instruction)
                )
                serialized = json.dumps(non_instruction, sort_keys=True)
                for forbidden in (
                    "RouteScenario", "route_name", "route_entry", "arm_id",
                    "seed",
                ):
                    self.assertNotIn(forbidden, serialized)

    def test_seed_everything_repeats_python_and_numpy(self) -> None:
        seed_everything(3, include_torch=False)
        first = (random.random(), np.random.random())
        seed_everything(3, include_torch=False)
        second = (random.random(), np.random.random())
        self.assertEqual(first, second)
        self.assertEqual("0", os.environ["PYTHONHASHSEED"])


class CheckpointCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_state = {
            "vlm.backbone.weight": _TensorShape(1000),
            "vlm.model.embed_tokens.weight": _TensorShape(100),
            "vlm.lm_head.weight": _TensorShape(100),
        }

    def test_complete_checkpoint_reports_element_coverage(self) -> None:
        report = validate_checkpoint_coverage(
            self.model_state,
            dict(self.model_state),
            [],
            [],
            minimum_coverage=0.99,
        )
        self.assertEqual(1.0, report["coverage"])
        self.assertEqual(1200, report["loaded_parameter_elements"])
        self.assertEqual([], report["missing_critical_keys"])

    def test_missing_action_head_is_always_fatal(self) -> None:
        checkpoint = dict(self.model_state)
        checkpoint.pop("vlm.lm_head.weight")
        with self.assertRaisesRegex(
            CheckpointCompatibilityError, "critical planner/action"
        ):
            validate_checkpoint_coverage(
                self.model_state,
                checkpoint,
                ["vlm.lm_head.weight"],
                [],
                minimum_coverage=0.5,
            )

    def test_low_parameter_coverage_is_fatal(self) -> None:
        checkpoint = {
            key: value
            for key, value in self.model_state.items()
            if key != "vlm.backbone.weight"
        }
        with self.assertRaisesRegex(
            CheckpointCompatibilityError, "parameter coverage"
        ):
            validate_checkpoint_coverage(
                self.model_state,
                checkpoint,
                ["vlm.backbone.weight"],
                [],
                minimum_coverage=0.98,
            )


class DeterministicInferenceSourceTest(unittest.TestCase):

    def test_evaluator_has_explicit_context_and_standardized_plan_gate(self) -> None:
        source = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/leaderboard_evaluator.py"
        ).read_text(encoding="utf-8")
        self.assertIn("set_language2drive_context", source)
        self.assertIn(
            "language2drive_track != L2D_TRACK_STANDARDIZED", source
        )
        self.assertIn("set_context(public_context)", source)
        self.assertIn("os.environ.pop('L2D_ROUTE_NAME', None)", source)
        self.assertIn("os.environ.pop('L2D_SEED', None)", source)
        self.assertIn("os.environ.pop('L2D_ROUTES_XML', None)", source)
        self.assertIn("os.environ.pop('L2D_PRIVATE_SIDECAR', None)", source)
        self.assertIn("os.environ.pop('L2D_ROUTE_ENTRY_ID', None)", source)
        self.assertIn("os.environ.pop('L2D_ROUTE_INSTRUCTION', None)", source)
        self.assertLess(
            source.index("os.environ.pop('L2D_ROUTES_XML', None)"),
            source.index("self.module_agent = importlib.import_module"),
        )
        self.assertGreaterEqual(
            source.count("os.environ.pop('L2D_ROUTES_XML', None)"), 2
        )
        self.assertIn("secrets.token_hex(16)", source)
        self.assertIn("'language2drive-agent-io'", source)
        self.assertNotIn("_record_dir, '.agent_io'", source)
        self.assertIn("seed_everything(0)", source)
        self.assertIn("Standardized/Native Track requires", source)
        self.assertIn("--record_frame_interval=1", source)
        self.assertIn("private-sidecar evaluation requires an", source)
        self.assertIn("20 Hz contract/deadline runtime", source)
        self.assertIn("route artifacts never fall back", source)
        self.assertIn("os.environ.get('TMPDIR', '/tmp')", source)
        self.assertIn("else base_agent_config + '+' + save_name", source)
        self.assertLess(
            source.index("os.environ.pop('L2D_SEED', None)"),
            source.index("self.agent_instance = agent_class_obj"),
        )
        self.assertLess(
            source.index("set_context(public_context)"),
            source.index("self.agent_instance.setup(route_agent_config)"),
        )

    def test_external_carla_branch_never_owns_server_lifecycle(self) -> None:
        source = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/leaderboard_evaluator.py"
        ).read_text(encoding="utf-8")
        setup_start = source.index("    def _setup_simulation(self, args):")
        setup_end = source.index("    def _reset_world_settings(self):", setup_start)
        setup_source = source[setup_start:setup_end]
        external_start = setup_source.index("if self._external_carla:")
        legacy_start = setup_source.index("self.carla_path = os.environ", external_start)
        external_source = setup_source[external_start:legacy_start]
        legacy_source = setup_source[legacy_start:]
        for forbidden in (
            "find_free_port", "subprocess.Popen", "time.sleep", "atexit.register"
        ):
            self.assertNotIn(forbidden, external_source)
        self.assertIn("return connect_existing_carla", external_source)
        self.assertIn("--l2d-no-rendering", source)
        self.assertIn("validate_privileged_no_rendering_request", source)
        self.assertIn("settings.no_rendering_mode = False", source)
        self.assertIn("did not retain the required synchronous 20 Hz", source)
        # The flag is opt-in: legacy evaluator-owned startup is unchanged.
        self.assertIn("args.port = find_free_port(args.port)", legacy_source)
        self.assertIn("subprocess.Popen", legacy_source)
        self.assertIn("atexit.register", legacy_source)
        self.assertIn("time.sleep(60)", legacy_source)
        self.assertIn("if crashed and not self._external_carla", source)
        self.assertIn("--l2d-external-carla", source)

    def test_v1_scenario_uses_private_path_without_second_grp_or_agent_leak(self):
        evaluator = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/leaderboard_evaluator.py"
        ).read_text(encoding="utf-8")
        scenario = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/scenarios/route_scenario.py"
        ).read_text(encoding="utf-8")
        manipulation = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/utils/route_manipulation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("config.language2drive_private_route", evaluator)
        self.assertIn("direct_trajectory_from_oracle_path", scenario)
        self.assertIn(
            "respect_green_junction_entry=private_termination", scenario
        )
        self.assertIn("language2drive_private_termination", scenario)
        self.assertIn("BackgroundBehavior", scenario)
        self.assertIn("Legacy parked meshes", scenario)
        self.assertIn("self.available_parking_locations = []", scenario)
        self.assertIn("if debug_mode > 0 and not private_l2d", scenario)
        self.assertIn("[] if private_l2d", scenario)
        self.assertIn("require one frozen weather state", scenario)
        self.assertGreaterEqual(
            scenario.count("language2drive_private_termination"), 5
        )
        self.assertIn("private_sidecar.oracle_path.direct_v1", scenario)
        direct_start = manipulation.index("def direct_trajectory_from_oracle_path")
        direct_source = manipulation[direct_start:]
        self.assertNotIn("GlobalRoutePlanner(", direct_source)
        self.assertNotIn("trace_route(", direct_source)
        standardized_guard = evaluator.index(
            "if language2drive_track != L2D_TRACK_STANDARDIZED:"
        )
        plan_call = evaluator.index(
            "self.agent_instance.set_global_plan", standardized_guard
        )
        self.assertLess(standardized_guard, plan_call)

    def test_private_route_disables_legacy_route_dependent_parked_meshes(self):
        from leaderboard.scenarios.route_scenario import RouteScenario

        scenario = RouteScenario.__new__(RouteScenario)
        scenario.config = SimpleNamespace(
            language2drive_private_termination=True
        )
        scenario.client = SimpleNamespace(apply_batch=lambda _commands: None)
        scenario._parked_ids = []
        scenario.remove_all_actors = lambda: None
        scenario.available_parking_locations = [{"hidden_arm_hint": True}]
        scenario._get_parking_slots()
        self.assertEqual([], scenario.available_parking_locations)
        # The runtime call must return before reading ego/world/CARLA state.
        self.assertIsNone(scenario.spawn_parked_vehicles(None))

    def test_direct_private_road1_path_preserves_coordinates_without_topology(self):
        import leaderboard.utils.route_manipulation as route_manipulation

        class MapWithoutTopology:
            @staticmethod
            def to_opendrive():
                return "<OpenDRIVE><header/></OpenDRIVE>"

        class WorldWithoutTopology:
            @staticmethod
            def get_map():
                return MapWithoutTopology()

        class ProviderStub:
            @staticmethod
            def get_world():
                return WorldWithoutTopology()

        original_provider = route_manipulation.CarlaDataProvider
        route_manipulation.CarlaDataProvider = ProviderStub
        points = [
            {"x": -1000.0, "y": 1.75, "z": 0.0, "yaw_deg": 37.0},
            {"x": -990.0, "y": 1.75, "z": 0.0, "yaw_deg": 38.0},
            {"x": -980.0, "y": 1.75, "z": 0.0, "yaw_deg": 39.0},
        ]
        try:
            gps, route = route_manipulation.direct_trajectory_from_oracle_path(
                points
            )
        finally:
            route_manipulation.CarlaDataProvider = original_provider
        self.assertEqual(3, len(gps))
        self.assertEqual(3, len(route))
        self.assertEqual(
            [(-1000.0, 1.75), (-990.0, 1.75), (-980.0, 1.75)],
            [
                (item[0].location.x, item[0].location.y)
                for item in route
            ],
        )
        self.assertTrue(all(item[1].name == "LANEFOLLOW" for item in route))
        route_manipulation.CarlaDataProvider = ProviderStub
        try:
            _, yaw_route = route_manipulation.direct_trajectory_from_oracle_path(
                points
            )
            self.assertEqual(
                [37.0, 38.0, 39.0],
                [item[0].rotation.yaw for item in yaw_route],
            )
            with self.assertRaisesRegex(ValueError, "requires yaw_deg"):
                route_manipulation.direct_trajectory_from_oracle_path(
                    [{key: value for key, value in points[0].items()
                      if key != "yaw_deg"}, *points[1:]]
                )
        finally:
            route_manipulation.CarlaDataProvider = original_provider


class ExternalCarlaTest(unittest.TestCase):
    class Settings:
        synchronous_mode = False
        fixed_delta_seconds = None
        deterministic_ragdolls = False
        spectator_as_ego = True

    class TrafficManager:
        def __init__(self):
            self.calls = []

        def set_synchronous_mode(self, value):
            self.calls.append(("sync", value))

        def set_hybrid_physics_mode(self, value):
            self.calls.append(("hybrid", value))

    class World:
        def __init__(self):
            self.settings = ExternalCarlaTest.Settings()
            self.applied = []

        @staticmethod
        def get_map():
            return object()

        def get_settings(self):
            return self.settings

        def apply_settings(self, settings):
            self.applied.append(settings)

    class Client:
        def __init__(self, host, port):
            self.endpoint = (host, port)
            self.timeouts = []
            self.world = ExternalCarlaTest.World()
            self.tm = ExternalCarlaTest.TrafficManager()
            self.tm_ports = []

        def set_timeout(self, value):
            self.timeouts.append(value)

        def get_world(self):
            return self.world

        def get_trafficmanager(self, port):
            self.tm_ports.append(port)
            return self.tm

    class Carla:
        clients = []

        @classmethod
        def Client(cls, host, port):
            client = ExternalCarlaTest.Client(host, port)
            cls.clients.append(client)
            return client

    def setUp(self) -> None:
        self.Carla.clients = []

    def test_exact_ports_fail_fast_probe_and_synchronous_configuration(self):
        client, timeout, manager = connect_existing_carla(
            self.Carla,
            host="127.0.0.1",
            port=2345,
            traffic_manager_port=8765,
            timeout_seconds=600.0,
            frame_rate=20.0,
        )
        self.assertEqual(("127.0.0.1", 2345), client.endpoint)
        self.assertEqual([8765], client.tm_ports)
        self.assertEqual([10.0, 600.0], client.timeouts)
        self.assertEqual(600.0, timeout)
        self.assertEqual([("sync", True), ("hybrid", True)], manager.calls)
        settings = client.world.applied[0]
        self.assertTrue(settings.synchronous_mode)
        self.assertEqual(0.05, settings.fixed_delta_seconds)
        self.assertTrue(settings.deterministic_ragdolls)
        self.assertFalse(settings.spectator_as_ego)

    def test_missing_external_world_raises_without_retry(self):
        class BrokenClient(self.Client):
            def get_world(self):
                raise OSError("connection refused")

        class BrokenCarla:
            calls = 0

            @classmethod
            def Client(cls, host, port):
                cls.calls += 1
                return BrokenClient(host, port)

        with self.assertRaisesRegex(
            ExternalCarlaConnectionError, "connection refused"
        ):
            connect_existing_carla(
                BrokenCarla,
                host="127.0.0.1",
                port=2345,
                traffic_manager_port=8765,
                timeout_seconds=600.0,
                frame_rate=20.0,
            )
        self.assertEqual(1, BrokenCarla.calls)

    def test_partial_external_configuration_is_rolled_back(self):
        class BrokenTrafficManager(self.TrafficManager):
            def set_hybrid_physics_mode(self, value):
                self.calls.append(("hybrid", value))
                if value:
                    raise RuntimeError("hybrid setup failed")

        class Client(self.Client):
            def __init__(self, host, port):
                super().__init__(host, port)
                self.tm = BrokenTrafficManager()

        client = Client("127.0.0.1", 2345)

        class Carla:
            @staticmethod
            def Client(_host, _port):
                return client

        with self.assertRaisesRegex(
            ExternalCarlaConnectionError, "hybrid setup failed"
        ):
            connect_existing_carla(
                Carla,
                host="127.0.0.1",
                port=2345,
                traffic_manager_port=8765,
                timeout_seconds=600.0,
                frame_rate=20.0,
            )
        self.assertFalse(client.world.settings.synchronous_mode)
        self.assertIsNone(client.world.settings.fixed_delta_seconds)
        self.assertEqual(
            [
                ("sync", True), ("hybrid", True),
                ("sync", False), ("hybrid", False),
            ],
            client.tm.calls,
        )

    def test_no_rendering_is_capability_gated_and_reset_is_explicit(self):
        # Import the evaluator function through its active package layout so
        # this test exercises the same names as the production CLI process.
        from leaderboard import leaderboard_evaluator

        validate = leaderboard_evaluator.validate_privileged_no_rendering_request

        class Oracle:
            language2drive_private_context_capability = PRIVILEGED_ORACLE_CAPABILITY

        class Ordinary:
            pass

        private = {"route": {"route_entry_id": 1}}
        self.assertTrue(validate(
            True,
            external_carla=True,
            track=L2D_TRACK_NATIVE,
            private_context=private,
            agent_class=Oracle,
        ))
        self.assertFalse(validate(
            False,
            external_carla=False,
            track=L2D_TRACK_STANDARDIZED,
            private_context=None,
            agent_class=Ordinary,
        ))
        cases = (
            ({"external_carla": False, "track": L2D_TRACK_NATIVE,
              "private_context": private, "agent_class": Oracle},
             "external-carla"),
            ({"external_carla": True, "track": L2D_TRACK_STANDARDIZED,
              "private_context": private, "agent_class": Oracle},
             "native only"),
            ({"external_carla": True, "track": L2D_TRACK_NATIVE,
              "private_context": None, "agent_class": Oracle},
             "private L2D route"),
            ({"external_carla": True, "track": L2D_TRACK_NATIVE,
              "private_context": private, "agent_class": Ordinary},
             "privileged oracle capability"),
        )
        for arguments, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                validate(True, **arguments)

    def test_cleanup_attempts_every_stage_before_raising(self):
        from leaderboard import leaderboard_evaluator

        calls = []

        class Provider:
            @staticmethod
            def cleanup():
                calls.append("provider")
                raise RuntimeError("provider failed")

        class Manager:
            def close_language2drive_runtime(self):
                calls.append("runtime")
                raise RuntimeError("runtime failed")

            def get_running_status(self):
                calls.append("status")
                return True

            def cleanup(self):
                calls.append("manager")

        class Agent:
            def destroy(self):
                calls.append("agent")
                raise RuntimeError("agent failed")

        class RouteScenario:
            def remove_all_actors(self):
                calls.append("route")

        class Statistics:
            def remove_scenario(self):
                calls.append("statistics")

        class Sensor:
            id = 17

            def stop(self):
                calls.append("sensor_stop")
                raise RuntimeError("sensor stop failed")

            def destroy(self):
                calls.append("sensor_destroy")

        class Actors(list):
            def filter(self, _pattern):
                calls.append("sensor_enumeration")
                return self

        class World:
            def get_actors(self):
                return Actors([Sensor()])

        evaluator = leaderboard_evaluator.LeaderboardEvaluator.__new__(
            leaderboard_evaluator.LeaderboardEvaluator
        )
        evaluator.manager = Manager()
        evaluator._agent_watchdog = None
        evaluator.agent_instance = Agent()
        evaluator.route_scenario = RouteScenario()
        evaluator.statistics_manager = Statistics()
        evaluator.world = World()
        evaluator._client_timed_out = False
        old_provider = leaderboard_evaluator.CarlaDataProvider
        leaderboard_evaluator.CarlaDataProvider = Provider
        try:
            with self.assertRaisesRegex(
                RuntimeError, "private_runtime.*carla_data_provider.*agent.*sensor_stop"
            ):
                evaluator._cleanup()
        finally:
            leaderboard_evaluator.CarlaDataProvider = old_provider
        self.assertEqual(
            [
                "runtime", "provider", "agent", "route", "statistics",
                "status", "manager", "sensor_enumeration", "sensor_stop",
                "sensor_destroy",
            ],
            calls,
        )
        self.assertIsNone(evaluator.agent_instance)
        self.assertIsNone(evaluator.route_scenario)

    def test_external_world_reset_is_attempted_even_after_timeout(self):
        from leaderboard import leaderboard_evaluator

        class Settings:
            synchronous_mode = True
            fixed_delta_seconds = 0.05
            deterministic_ragdolls = True
            spectator_as_ego = False
            no_rendering_mode = True

        class World:
            def __init__(self):
                self.settings = Settings()
                self.ticks = 0

            def tick(self):
                self.ticks += 1

            def get_settings(self):
                return self.settings

            def apply_settings(self, settings):
                self.settings = settings

        class TrafficManager:
            def __init__(self):
                self.calls = []

            def set_synchronous_mode(self, value):
                self.calls.append(("sync", value))

            def set_hybrid_physics_mode(self, value):
                self.calls.append(("hybrid", value))

        evaluator = leaderboard_evaluator.LeaderboardEvaluator.__new__(
            leaderboard_evaluator.LeaderboardEvaluator
        )
        evaluator.world = World()
        evaluator.manager = object()
        evaluator.traffic_manager = TrafficManager()
        evaluator._external_carla = True
        evaluator._client_timed_out = True
        timeouts = []
        evaluator.client = SimpleNamespace(
            set_timeout=lambda value: timeouts.append(value),
            get_world=lambda: evaluator.world,
        )
        evaluator._reset_world_settings()
        self.assertEqual(1, evaluator.world.ticks)
        self.assertEqual([10.0], timeouts)
        self.assertFalse(evaluator.world.settings.synchronous_mode)
        self.assertIsNone(evaluator.world.settings.fixed_delta_seconds)
        self.assertFalse(evaluator.world.settings.no_rendering_mode)
        self.assertEqual(
            [("sync", False), ("hybrid", False)],
            evaluator.traffic_manager.calls,
        )

    def test_external_reset_continues_after_final_tick_failure(self):
        from leaderboard import leaderboard_evaluator

        class Settings:
            synchronous_mode = True
            fixed_delta_seconds = 0.05
            deterministic_ragdolls = True
            spectator_as_ego = False
            no_rendering_mode = True

        class World:
            def __init__(self):
                self.settings = Settings()
                self.applied = False

            @staticmethod
            def tick():
                raise RuntimeError("tick timed out")

            def get_settings(self):
                return self.settings

            def apply_settings(self, settings):
                self.settings = settings
                self.applied = True

        class TrafficManager:
            def __init__(self):
                self.calls = []

            def set_synchronous_mode(self, value):
                self.calls.append(("sync", value))

            def set_hybrid_physics_mode(self, value):
                self.calls.append(("hybrid", value))

        evaluator = leaderboard_evaluator.LeaderboardEvaluator.__new__(
            leaderboard_evaluator.LeaderboardEvaluator
        )
        evaluator.world = World()
        evaluator.manager = object()
        evaluator.traffic_manager = TrafficManager()
        evaluator._external_carla = True
        evaluator._client_timed_out = True
        evaluator.client = None

        with self.assertRaisesRegex(RuntimeError, "world_tick:tick timed out"):
            evaluator._reset_world_settings()

        self.assertTrue(evaluator.world.applied)
        self.assertFalse(evaluator.world.settings.synchronous_mode)
        self.assertIsNone(evaluator.world.settings.fixed_delta_seconds)
        self.assertFalse(evaluator.world.settings.no_rendering_mode)
        self.assertEqual(
            [("sync", False), ("hybrid", False)],
            evaluator.traffic_manager.calls,
        )

    def test_external_world_is_reset_when_agent_import_fails_before_run(self):
        from leaderboard import leaderboard_evaluator

        class World(self.World):
            def __init__(self):
                super().__init__()
                self.ticks = 0

            def tick(self):
                self.ticks += 1

        world = World()
        manager = self.TrafficManager()

        class Client:
            @staticmethod
            def get_world():
                return world

        args = SimpleNamespace(
            language2drive_external_carla=True,
            language2drive_track="standardized",
            agent="/root/autodl-tmp/does_not_exist/l2d_missing_agent.py",
            timeout=10.0,
            debug=0,
        )
        with mock.patch.object(
            leaderboard_evaluator.LeaderboardEvaluator,
            "_setup_simulation",
            return_value=(Client(), 10.0, manager),
        ):
            with self.assertRaises(ModuleNotFoundError):
                leaderboard_evaluator.LeaderboardEvaluator(args, object())
        self.assertEqual(1, world.ticks)
        self.assertFalse(world.settings.synchronous_mode)
        self.assertIsNone(world.settings.fixed_delta_seconds)
        self.assertFalse(world.settings.no_rendering_mode)
        self.assertEqual(
            [("sync", False), ("hybrid", False)], manager.calls
        )

    def test_private_termination_keeps_stats_node_and_cleanup_precedes_score(self):
        route_source = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/scenarios/route_scenario.py"
        ).read_text(encoding="utf-8")
        self.assertIn('Idle(name="Language2DrivePrivateTermination")', route_source)
        self.assertIn("timeout_node.timeout = False", route_source)

        manager_tree = ast.parse(
            (
                REPO_ROOT
                / "language2drive/leaderboard/leaderboard/scenarios/scenario_manager.py"
            ).read_text(encoding="utf-8")
        )
        manager = next(
            node
            for node in manager_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ScenarioManager"
        )
        for method_name in ("cleanup", "stop_scenario"):
            method = next(
                node
                for node in manager.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            )
            calls = [
                node.func.attr
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"close_language2drive_runtime", "close_logger"}
            ]
            self.assertLess(
                calls.index("close_language2drive_runtime"),
                calls.index("close_logger"),
            )

        evaluator_source = (
            REPO_ROOT
            / "language2drive/leaderboard/leaderboard/leaderboard_evaluator.py"
        ).read_text(encoding="utf-8")
        run_start = evaluator_source.index("    def run(self, args):")
        run_end = evaluator_source.index("\ndef main():", run_start)
        run_source = evaluator_source[run_start:run_end]
        self.assertIn("finally:", run_source)
        self.assertLess(
            run_source.rindex("self._cleanup()"),
            run_source.rindex("self._reset_world_settings()"),
        )
        route_start = evaluator_source.index(
            "    def _load_and_run_scenario(self, args, config):"
        )
        route_end = evaluator_source.index("    def run(self, args):", route_start)
        route_source = evaluator_source[route_start:route_end]
        interrupt_start = route_source.index("except KeyboardInterrupt as exc:")
        interrupt_end = route_source.index("except TickRuntimeError as exc:", interrupt_start)
        self.assertNotIn("return True", route_source[interrupt_start:interrupt_end])


class PrivateRedLightCorrectionTest(unittest.TestCase):
    class Light:
        def __init__(self, identifier, group=()):
            self.id = identifier
            self._group = list(group)

        def get_group_traffic_lights(self):
            return self._group

    class Waypoint:
        def __init__(self, junction_id=None, next_waypoints=()):
            self.is_junction = junction_id is not None
            self.is_intersection = self.is_junction
            self.junction_id = -1 if junction_id is None else junction_id
            self._next_waypoints = list(next_waypoints)

        def next(self, _distance):
            return self._next_waypoints

    @staticmethod
    def criterion():
        criterion = RunningRedLightTest.__new__(RunningRedLightTest)
        criterion._inside_green_entered_junction = False
        criterion._green_entry_junction_id = 71
        criterion._green_entry_light_ids = {10, 11}
        criterion._green_junction_exit_grace_until = 6.0
        criterion._traffic_light_junction_ids = {
            10: {71},
            11: {71},
            20: {92},
            21: {71},
        }
        return criterion

    def test_green_entry_group_and_same_junction_are_narrowly_suppressed(self):
        grouped = self.Light(10)
        cross_flow = self.Light(11)
        grouped._group = [grouped, cross_flow]
        self.assertEqual({10, 11}, RunningRedLightTest._light_group_ids(grouped))

        criterion = self.criterion()
        self.assertTrue(criterion._suppress_green_entry_signal(
            cross_flow, [], now=5.5
        ))
        # A custom-map cross-flow light can be absent from the CARLA group, but
        # its stop line still proves that it controls the same junction.
        self.assertTrue(criterion._suppress_green_entry_signal(
            self.Light(21), [], now=5.5
        ))
        self.assertEqual(
            71, criterion._confirmed_green_entry_junction(71, {10, 11})
        )
        self.assertIsNone(
            criterion._confirmed_green_entry_junction(92, {10, 11})
        )

    def test_exit_grace_never_suppresses_a_downstream_red_light(self):
        criterion = self.criterion()
        downstream_stop_line = self.Waypoint(
            next_waypoints=[self.Waypoint(junction_id=92)]
        )
        downstream = self.Light(20)
        self.assertFalse(criterion._suppress_green_entry_signal(
            downstream, [downstream_stop_line], now=5.5
        ))
        # Expiry also removes the exemption even for the original light.
        self.assertFalse(criterion._suppress_green_entry_signal(
            self.Light(10), [], now=6.01
        ))

    def test_update_still_records_downstream_red_during_exit_grace(self):
        carla = atomic_criteria.carla
        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.0),
            carla.Rotation(yaw=0.0),
        )
        road_waypoint = self.Waypoint()
        road_waypoint.road_id = 1
        road_waypoint.lane_id = -1

        stop_line = self.Waypoint(
            next_waypoints=[self.Waypoint(junction_id=92)]
        )
        stop_line.road_id = road_waypoint.road_id
        stop_line.lane_id = road_waypoint.lane_id
        stop_line.transform = transform
        stop_line.lane_width = 3.5

        downstream = self.Light(20)
        downstream.state = carla.TrafficLightState.Red
        downstream.get_transform = lambda: transform
        actor = SimpleNamespace(
            bounding_box=SimpleNamespace(extent=SimpleNamespace(x=1.0)),
            is_at_traffic_light=lambda: False,
        )

        criterion = self.criterion()
        criterion.actor = actor
        criterion._map = SimpleNamespace(get_waypoint=lambda _location: road_waypoint)
        criterion._list_traffic_lights = [
            (downstream, carla.Location(x=0.0, y=0.0, z=0.0), [stop_line])
        ]
        criterion._last_red_light_id = None
        criterion._respect_green_junction_entry = True
        criterion.debug = False
        criterion.is_vehicle_crossing_line = lambda _ego, _stop: True
        criterion.test_status = "INIT"
        criterion.actual_value = 0
        criterion.events = []
        criterion._terminate_on_failure = False
        criterion.status = atomic_criteria.py_trees.common.Status.RUNNING
        criterion.logger = SimpleNamespace(debug=lambda _message: None)

        with mock.patch.object(
            atomic_criteria.CarlaDataProvider,
            "get_transform",
            return_value=transform,
        ), mock.patch.object(
            atomic_criteria.GameTime, "get_time", return_value=5.5
        ), mock.patch.object(
            atomic_criteria.GameTime, "get_frame", return_value=110
        ):
            criterion.update()

        self.assertEqual("FAILURE", criterion.test_status)
        self.assertEqual(1, criterion.actual_value)
        self.assertEqual(20, criterion._last_red_light_id)

    def test_stop_line_junction_resolution_is_tightly_bounded(self):
        criterion = self.criterion()
        entry = self.Waypoint(junction_id=71)
        stop_line = self.Waypoint(next_waypoints=[entry])
        self.assertEqual(
            {71}, criterion._controlled_junction_ids([stop_line])
        )


class CanonicalLoggerTest(unittest.TestCase):
    def test_canonical_files_events_and_idempotent_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = make_public_context(
                instruction="Wait at the pickup point.",
                route_name="RouteScenario_2_rep0",
                seed=3,
                track="standardized",
            )
            logger = ClosedLoopLogger(
                temporary,
                {
                    "route_id": "RouteScenario_2",
                    "language2drive_public_context": context,
                },
            )
            logger.start_route()
            vector = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            rotation = SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0)
            transform = SimpleNamespace(
                location=vector,
                rotation=rotation,
                get_forward_vector=lambda: SimpleNamespace(
                    x=1.0, y=0.0, z=0.0
                ),
                get_right_vector=lambda: SimpleNamespace(
                    x=0.0, y=1.0, z=0.0
                ),
            )
            ego = SimpleNamespace(
                get_transform=lambda: transform,
                get_velocity=lambda: vector,
                get_acceleration=lambda: vector,
                get_angular_velocity=lambda: vector,
            )
            action = SimpleNamespace(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                reverse=False,
                hand_brake=False,
                gear=0,
            )
            logger.log_frame(
                0,
                SimpleNamespace(elapsed_seconds=0.0),
                ego,
                action,
            )
            logger.log_event({"event": "stage_transition", "stage": 1})
            logger.close_route({"termination": "completed"})
            logger.close_route({"termination": "must_not_overwrite"})

            root = Path(temporary)
            self.assertTrue((root / "run_meta.json").is_file())
            self.assertTrue((root / "route_meta.json").is_file())
            self.assertTrue((root / "trace.jsonl").is_file())
            self.assertTrue((root / "frames.jsonl").is_file())
            self.assertEqual(
                (root / "trace.jsonl").stat().st_ino,
                (root / "frames.jsonl").stat().st_ino,
            )
            self.assertEqual(
                (root / "trace.jsonl").read_bytes(),
                (root / "frames.jsonl").read_bytes(),
            )
            trace = json.loads((root / "trace.jsonl").read_text())
            self.assertEqual("Wait at the pickup point.", trace["instruction"])
            self.assertEqual(0, trace["instruction_version"])
            event = json.loads((root / "events.jsonl").read_text())
            self.assertEqual("stage_transition", event["event"])
            summary = json.loads((root / "route_summary.json").read_text())
            self.assertEqual("completed", summary["termination"])
            self.assertEqual(0, summary["logger_error_count"])

    def test_l2d_frames_hardlink_failure_is_capture_failure(self) -> None:
        class RuntimeProbe:
            failure = None

            def record_capture_failure(self, operation, exc):
                self.failure = (operation, type(exc).__name__)

            @staticmethod
            def drain_final_events():
                return []

            @staticmethod
            def report():
                return {"termination_reason": "evaluator_abort"}

        with tempfile.TemporaryDirectory() as temporary:
            runtime = RuntimeProbe()
            logger = ClosedLoopLogger(
                temporary,
                {
                    "route_id": "RouteScenario_2",
                    "language2drive_public_context": make_public_context(
                        instruction="Wait.",
                        route_name="RouteScenario_2_rep0",
                        seed=1,
                        track="standardized",
                    ),
                },
                private_context={"runtime_required": True},
                private_runtime=runtime,
            )
            logger.start_route()
            with mock.patch(
                "leaderboard.utils.closed_loop_logger.os.link",
                side_effect=OSError("injected hardlink failure"),
            ):
                logger.close_route()
            logger.close_route()
            root = Path(temporary)
            self.assertTrue((root / "trace.jsonl").is_file())
            self.assertFalse((root / "frames.jsonl").exists())
            self.assertEqual(
                ("publish_frames_hardlink", "OSError"), runtime.failure
            )
            summary = json.loads((root / "route_summary.json").read_text())
            self.assertEqual(1, summary["logger_error_count"])
            errors = (root / "logger_errors.jsonl").read_text()
            self.assertIn("publish_frames_hardlink", errors)

    def test_legacy_logger_keeps_independent_frames_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(
                temporary,
                {"route_id": "legacy"},
            )
            logger.start_route()
            logger.close_route()
            root = Path(temporary)
            self.assertNotEqual(
                (root / "trace.jsonl").stat().st_ino,
                (root / "frames.jsonl").stat().st_ino,
            )
            self.assertEqual(
                (root / "trace.jsonl").read_bytes(),
                (root / "frames.jsonl").read_bytes(),
            )

    def test_logger_errors_are_persisted_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(temporary, {"route_id": "demo"})
            logger.start_route()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                logger.log_event({"not_json": object()})
            logger.close_route()
            errors = (Path(temporary) / "logger_errors.jsonl").read_text()
            self.assertIn("log_event", errors)
            self.assertIn("log_event failed", stderr.getvalue())
            summary = json.loads(
                (Path(temporary) / "route_summary.json").read_text()
            )
            self.assertEqual(1, summary["logger_error_count"])

    def test_logger_error_marks_private_runtime_failed(self) -> None:
        class RuntimeProbe:
            failure = None

            def record_capture_failure(self, operation, exc):
                self.failure = (operation, type(exc).__name__)

            @staticmethod
            def drain_final_events():
                return []

            def report(self):
                return {"failure_reason": self.failure}

        runtime = RuntimeProbe()
        with tempfile.TemporaryDirectory() as temporary:
            logger = ClosedLoopLogger(
                temporary, {"route_id": "demo"}, private_runtime=runtime
            )
            logger.start_route()
            with redirect_stderr(io.StringIO()):
                logger.log_event({"not_json": object()})
            logger.close_route()
        self.assertEqual(("log_event", "TypeError"), runtime.failure)

    def test_opaque_agent_artifacts_are_adopted_by_rollout_logger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".agent_io" / "opaque-token"
            staging.mkdir(parents=True)
            (staging / "agent_plans.jsonl").write_text(
                '{"plan": 1}\n', encoding="utf-8"
            )
            rollout = root / "semantic-private-run-id"
            logger = ClosedLoopLogger(
                str(rollout),
                {"route_id": "not-agent-visible"},
                agent_artifact_dir=str(staging),
            )
            logger.start_route()
            logger.close_route()
            self.assertEqual(
                '{"plan": 1}\n',
                (rollout / "agent_plans.jsonl").read_text(encoding="utf-8"),
            )
            self.assertFalse(staging.exists())


if __name__ == "__main__":
    unittest.main()
