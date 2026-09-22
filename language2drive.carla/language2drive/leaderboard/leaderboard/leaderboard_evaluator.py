#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse

from argparse import RawTextHelpFormatter
# from distutils.version import LooseVersion
import importlib
import copy
import json
import os
import secrets

# import pkg_resources
import sys

import carla
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, TickRuntimeError
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer
import atexit
import subprocess
import time
import random
from datetime import datetime

from team_code.language2drive_context import (
    L2D_TRACK_LEGACY,
    L2D_TRACK_NATIVE,
    L2D_TRACK_STANDARDIZED,
    PRIVILEGED_ORACLE_CAPABILITY,
    bind_privileged_oracle_context,
    make_public_context,
    normalize_l2d_track,
    seed_everything,
)
from team_code.language2drive_trace import (
    Language2DriveTracePostprocessor,
    expected_run_id,
    load_private_route_context,
    private_public_start_consistency_errors,
    public_private_consistency_errors,
)
from team_code.language2drive_runtime import Language2DriveEvaluatorRuntime
from team_code.language2drive_external_carla import connect_existing_carla

sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}

import socket

def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1

def get_weather_id(weather_conditions):
    from xml.etree import ElementTree as ET
    WORK_DIR = os.environ.get("WORK_DIR", "")
    if WORK_DIR != "":
        WORK_DIR += "/"
    tree = ET.parse(WORK_DIR + 'leaderboard/data/weather.xml')
    root = tree.getroot()
    def conditions_match(weather, conditions):
        for (key, value) in weather:
            if key == 'route_percentage' : continue
            if str(getattr(conditions, key))!= value:
                return False
        return True
    for case in root.findall('case'):
        weather = case[0].items()
        if conditions_match(weather, weather_conditions):
            return case.items()[0][1]
    return None


def waypoints_from_route_config(config):
    """Build waypoints and expected_goal from XML keypoints in route config."""
    keypoints = getattr(config, 'keypoints', None) or []
    waypoints = [
        {
            'x': round(float(loc.x), 4),
            'y': round(float(loc.y), 4),
            'z': round(float(loc.z), 4),
        }
        for loc in keypoints
    ]
    expected_goal = waypoints[-1] if waypoints else None
    return waypoints, expected_goal


def env_flag(name):
    """Return True for common truthy environment-variable values."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def language2drive_track_from_args(args):
    """Resolve explicit CLI track, with the v0 strict env flag as an alias."""
    return normalize_l2d_track(
        getattr(args, 'language2drive_track', None) or os.environ.get('L2D_TRACK'),
        strict_vln=env_flag('L2D_STRICT_VLN'),
    )


def language2drive_seed_from_args(args):
    """Use the L2D seed when supplied, otherwise share the frozen TM seed."""
    explicit_seed = getattr(args, 'language2drive_seed', None)
    return int(
        args.traffic_manager_seed if explicit_seed is None else explicit_seed
    )


def route_entry_id_from_config(config):
    """Resolve the XML route id, with an optional single-route assertion."""
    name = str(getattr(config, 'name', ''))
    prefix = 'RouteScenario_'
    if not name.startswith(prefix):
        raise ValueError("cannot derive route_entry_id from {!r}".format(name))
    return int(name[len(prefix):])


def language2drive_v1_root_from_args(args):
    explicit = getattr(args, 'language2drive_v1_root', None)
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    # evaluator.py -> leaderboard -> leaderboard -> language2drive -> project
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..'
    ))


def validate_privileged_no_rendering_request(
    requested, *, external_carla, track, private_context, agent_class
):
    """Fail closed unless no-rendering is requested by the trusted oracle.

    Rendering is an agent-observable capability: disabling it would invalidate
    camera agents. The optimization therefore requires an existing external
    server, Native-System private evaluation and the exact versioned oracle
    class capability. Merely choosing ``--track SENSORS`` is insufficient.
    """

    if not requested:
        return False
    if not external_carla:
        raise ValueError("--l2d-no-rendering requires --l2d-external-carla")
    if track != L2D_TRACK_NATIVE:
        raise ValueError("--l2d-no-rendering is privileged-oracle/native only")
    if not isinstance(private_context, dict) or not isinstance(
        private_context.get('route'), dict
    ):
        raise ValueError("--l2d-no-rendering requires one private L2D route")
    if getattr(
        agent_class, 'language2drive_private_context_capability', None
    ) != PRIVILEGED_ORACLE_CAPABILITY:
        raise ValueError(
            "--l2d-no-rendering requires the versioned privileged oracle capability"
        )
    return True


class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    # Tunable parameters
    client_timeout = 300.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None
        self._external_carla = bool(
            getattr(args, 'language2drive_external_carla', False)
        )
        self.server = None

        self.statistics_manager = statistics_manager

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None

        # Setup the simulation.  Retain the probed external world immediately
        # so a later agent import/manager-construction failure can restore the
        # caller-owned server even though ``run()`` was never entered.
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(args)
        if self._external_carla:
            self.world = self.client.get_world()

        # dist = pkg_resources.get_distribution("carla")
        # if dist.version != 'leaderboard':
        #     if LooseVersion(dist.version) < LooseVersion('0.9.10'):
        #         raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        try:
            # Load agent
            # Agent modules execute arbitrary import-time code.  Remove legacy
            # route lookup capabilities before import, then repeat this per route
            # below to clear any state left by a preceding Native/legacy rollout.
            if language2drive_track_from_args(args) == L2D_TRACK_STANDARDIZED:
                os.environ.pop('L2D_ROUTE_NAME', None)
                os.environ.pop('L2D_SEED', None)
                os.environ.pop('L2D_ROUTES_XML', None)
                os.environ.pop('L2D_PRIVATE_SIDECAR', None)
                os.environ.pop('L2D_ROUTE_ENTRY_ID', None)
                os.environ.pop('L2D_ROUTE_INSTRUCTION', None)
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

            # Create the ScenarioManager
            self.manager = ScenarioManager(args.timeout, self.statistics_manager, args.debug)

            # Time control for summary purposes
            self._start_time = GameTime.get_time()
            self._end_time = None

            # Prepare the agent timer
            self._agent_watchdog = None
            signal.signal(signal.SIGINT, self._signal_handler)

            self._client_timed_out = False
        except BaseException:
            if self._external_carla:
                try:
                    self._reset_world_settings()
                except Exception:
                    print(
                        "Language2Drive external CARLA init-failure reset failed:\n{}".format(
                            traceback.format_exc()
                        ),
                        flush=True,
                    )
            raise

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took longer than {}s to setup".format(self.client_timeout))
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self):
        """
        Remove and destroy all actors, attempting every cleanup stage.

        Cleanup is part of the v1 result contract: a logger close failure or a
        leaked evaluator-owned actor must never be hidden by an earlier cleanup
        exception.  We therefore run all independent stages and raise one
        aggregate error only after the best-effort sweep has completed.
        """
        failures = []

        def attempt(label, operation):
            try:
                operation()
            except Exception as exc:
                failures.append((label, exc, traceback.format_exc()))

        # Semantic actors are owned by the evaluator-private runtime.  Destroy
        # them while CarlaDataProvider still has a live world/client.
        if self.manager:
            fail_runtime = getattr(
                self.manager, 'fail_language2drive_runtime', None
            )
            if callable(fail_runtime):
                attempt(
                    'private_runtime_abort_mark',
                    lambda: fail_runtime(
                        'evaluator_cleanup_without_private_termination'
                    ),
                )
            attempt(
                'private_runtime', self.manager.close_language2drive_runtime
            )
        attempt('carla_data_provider', CarlaDataProvider.cleanup)

        if self._agent_watchdog:
            attempt('agent_watchdog', self._agent_watchdog.stop)

        if self.agent_instance:
            agent_instance = self.agent_instance
            self.agent_instance = None

            def destroy_agent():
                agent_instance.destroy()

            attempt('agent', destroy_agent)

        if self.route_scenario:
            route_scenario = self.route_scenario
            self.route_scenario = None
            attempt('route_scenario', route_scenario.remove_all_actors)
            if self.statistics_manager:
                attempt(
                    'statistics_scenario',
                    self.statistics_manager.remove_scenario,
                )

        if self.manager:
            try:
                self._client_timed_out = not self.manager.get_running_status()
            except Exception as exc:
                failures.append(
                    ('manager_status', exc, traceback.format_exc())
                )
                self._client_timed_out = True
            attempt('scenario_manager', self.manager.cleanup)

        # Make sure no sensors are left streaming.  Resolve the collection once
        # so one bad sensor cannot prevent the others from being destroyed.
        if self.world is not None:
            sensors = []
            try:
                sensors = list(self.world.get_actors().filter('*sensor*'))
            except Exception as exc:
                failures.append(('sensor_enumeration', exc, traceback.format_exc()))
            for sensor in sensors:
                sensor_id = getattr(sensor, 'id', 'unknown')
                attempt(
                    'sensor_stop:{}'.format(sensor_id), sensor.stop
                )
                attempt(
                    'sensor_destroy:{}'.format(sensor_id), sensor.destroy
                )

        if failures:
            for label, _, formatted in failures:
                print(
                    "\n\033[91mLanguage2Drive cleanup stage {} failed:\n{}\033[0m".format(
                        label, formatted
                    ),
                    flush=True,
                )
            raise RuntimeError(
                "Language2Drive cleanup failed in stage(s): {}".format(
                    ", ".join(label for label, _, _ in failures)
                )
            )

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        if self._external_carla:
            print(
                "Using external CARLA server at {}:{} (TrafficManager {})".format(
                    args.host, args.port, args.traffic_manager_port
                ),
                flush=True,
            )
            # Exact caller-owned ports: this branch never selects ports, owns a
            # simulator process, waits for startup, or registers a kill hook.
            return connect_existing_carla(
                carla,
                host=args.host,
                port=args.port,
                traffic_manager_port=args.traffic_manager_port,
                timeout_seconds=args.timeout or self.client_timeout,
                frame_rate=self.frame_rate,
            )

        # Legacy default: retain the historical evaluator-owned CARLA lifecycle.
        self.carla_path = os.environ["CARLA_ROOT"]
        args.port = find_free_port(args.port)
        cmd1 = f"{os.path.join(self.carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port} -graphicsadapter={args.gpu_rank}"
        self.server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
        print(cmd1, self.server.returncode, flush=True)
        atexit.register(os.killpg, self.server.pid, signal.SIGKILL)
        print(f"carla server started, port={args.port}", flush=True)
        time.sleep(60)

        attempts = 0
        num_max_restarts = 20
        while attempts < num_max_restarts:
            try:
                print("Try to connect to carla server", flush=True)
                client = carla.Client(args.host, args.port)
                if args.timeout:
                    client_timeout = args.timeout
                client.set_timeout(client_timeout)

                print("Try to apply Carla world settings", flush=True)
                settings = carla.WorldSettings(
                    synchronous_mode = True,
                    fixed_delta_seconds = 1.0 / self.frame_rate,
                    deterministic_ragdolls = True,
                    spectator_as_ego = False
                )
                print("Finish apply Carla world settings", flush=True)
                client.get_world().apply_settings(settings)
                print(f"load_world success , attempts={attempts}", flush=True)
                break
            except Exception as e:
                print(f"load_world failed , attempts={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                time.sleep(5)
        attempts = 0
        num_max_restarts = 40
        while attempts < num_max_restarts:
            try:
                args.traffic_manager_port = find_free_port(args.traffic_manager_port)
                traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                print(f"traffic_manager init success, try_time={attempts}", flush=True)
                break
            except Exception as e:
                print(f"traffic_manager init fail, try_time={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                time.sleep(5)
        return client, client_timeout, traffic_manager

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # ``load_world`` can fail after CARLA has already switched maps. Query
        # the caller-owned server again so reset never targets a stale pre-load
        # World proxy retained by the evaluator.
        if self._external_carla and getattr(self, 'client', None) is not None:
            # Normal custom-map evaluation restores the user timeout (often
            # 600s) after the ten-second readiness probe.  Reapply the bounded
            # timeout for finalization so a server that has already crashed
            # cannot strand the evaluator for the full route timeout.
            try:
                self.client.set_timeout(10.0)
            except Exception:
                pass
            try:
                refreshed_world = self.client.get_world()
                if refreshed_world is not None:
                    self.world = refreshed_world
            except Exception:
                # Fall through to the last known World proxy; the surrounding
                # finalizer will report if applying reset there also fails.
                pass
        # Has simulation failed?
        if self.world and (
            self._external_carla
            or (self.manager and not self._client_timed_out)
        ):
            # Reset every caller-owned subsystem even when a final tick or one
            # of the reset operations fails.  In particular, a timed-out tick
            # must not strand an external CARLA server in synchronous or
            # no-rendering mode for the next evaluator process.
            reset_errors = []
            try:
                self.world.tick()
            except Exception as exc:
                reset_errors.append("world_tick:{}".format(exc))
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.deterministic_ragdolls = False
                settings.spectator_as_ego = True
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)
            except Exception as exc:
                reset_errors.append("world_settings:{}".format(exc))
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                reset_errors.append("traffic_manager_sync:{}".format(exc))
            try:
                self.traffic_manager.set_hybrid_physics_mode(False)
            except Exception as exc:
                reset_errors.append("traffic_manager_hybrid:{}".format(exc))
            if reset_errors:
                raise RuntimeError(
                    "CARLA reset incomplete: {}".format("; ".join(reset_errors))
                )

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """
        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.deterministic_ragdolls = True
        settings.spectator_as_ego = False
        # Explicitly overwrite the external server setting on every route.
        # This guarantees that an AutoVLA/FDQ evaluator following an oracle
        # process gets rendered camera frames even if the prior process died
        # before its normal reset path.
        settings.no_rendering_mode = bool(
            getattr(args, 'language2drive_no_rendering', False)
        )
        self.world.apply_settings(settings)
        applied_settings = self.world.get_settings()
        if (
            not bool(applied_settings.synchronous_mode)
            or applied_settings.fixed_delta_seconds is None
            or abs(
                float(applied_settings.fixed_delta_seconds)
                - 1.0 / self.frame_rate
            ) > 1e-9
        ):
            raise RuntimeError(
                "CARLA world did not retain the required synchronous 20 Hz settings"
            )

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(
            language2drive_seed_from_args(args)
        )

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

    def _register_statistics(self, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m", flush=True)
        self.statistics_manager.save_entry_status(entry_status)
        self.statistics_manager.compute_route_statistics(
            route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
        )

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(config.name, config.repetition_index), flush=True)

        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        language2drive_track = language2drive_track_from_args(args)
        language2drive_workload = getattr(args, 'language2drive_workload', 'core')
        language2drive_seed = language2drive_seed_from_args(args)
        seed_everything(language2drive_seed)
        route_entry_id = route_entry_id_from_config(config)
        asserted_route_entry_id = getattr(args, 'language2drive_route_entry_id', None)
        if (
            asserted_route_entry_id is not None
            and int(asserted_route_entry_id) != route_entry_id
        ):
            raise ValueError(
                "--l2d-route-entry-id={} does not match current XML route {}".format(
                    asserted_route_entry_id, route_entry_id
                )
            )
        public_context = make_public_context(
            instruction=getattr(config, 'instruction', None),
            route_name=route_name,
            seed=language2drive_seed,
            track=language2drive_track,
        )
        route_metadata = dict(getattr(config, 'simlingo_metadata', None) or {})
        private_context = None
        capture_provenance = {}
        sidecar_path = getattr(args, 'language2drive_private_sidecar', None)
        if sidecar_path:
            private_context = load_private_route_context(
                sidecar_path, route_entry_id
            )
            private_consistency_metadata = dict(route_metadata)
            private_consistency_metadata.update(
                {
                    'route_entry_id': route_entry_id,
                    'town': str(config.town),
                    'instruction': getattr(config, 'instruction', None),
                }
            )
            identity_errors = public_private_consistency_errors(
                private_context, private_consistency_metadata
            )
            if identity_errors:
                raise ValueError(
                    "public XML/private sidecar identity mismatch: {}".format(
                        "; ".join(identity_errors)
                    )
                )
            xml_waypoints, _ = waypoints_from_route_config(config)
            spawn_errors = private_public_start_consistency_errors(
                private_context, xml_waypoints
            )
            if spawn_errors:
                raise ValueError(
                    "public XML/private sidecar spawn mismatch: {}".format(
                        "; ".join(spawn_errors)
                    )
                )
            capture_provenance = {
                'route_entry_id': route_entry_id,
                'private_sidecar_sha256': private_context['sidecar_sha256'],
            }
        elif (
            language2drive_track in {
                L2D_TRACK_STANDARDIZED, L2D_TRACK_NATIVE
            }
            and str(route_metadata.get('benchmark_type', '')).startswith(
                'language2drive_v1'
            )
        ):
            raise ValueError(
                "Language2Drive v1 Standardized/Native Track requires "
                "--l2d-private-sidecar for evaluator-only scoring"
            )
        if private_context is not None and not getattr(
            args, 'record_eval_data', False
        ):
            raise ValueError(
                "Language2Drive v1 private-sidecar evaluation requires "
                "--record_eval_data so its 20 Hz contract/deadline runtime "
                "receives authoritative frames"
            )
        if private_context is not None and not getattr(args, 'record_dir', None):
            raise ValueError(
                "Language2Drive v1 private-sidecar evaluation requires an "
                "explicit --record_dir; route artifacts never fall back to "
                "the evaluator working directory or a user home"
            )
        if (
            private_context is not None
            and int(getattr(args, 'record_frame_interval', 1)) != 1
        ):
            raise ValueError(
                "Language2Drive v1 private contract monitoring requires "
                "--record_frame_interval=1 (the authoritative 20 Hz trace)"
            )
        l2d_root = language2drive_v1_root_from_args(args)
        l2d_runtime = None
        if private_context is not None:
            l2d_runtime = Language2DriveEvaluatorRuntime(
                private_context,
                l2d_root,
                provider=CarlaDataProvider,
                carla_module=carla,
                traffic_manager=self.traffic_manager,
                traffic_manager_port=args.traffic_manager_port,
                stability_seconds=getattr(
                    args, 'language2drive_stability_seconds', 2.0
                ),
                progress_minimum=getattr(
                    args, 'language2drive_progress_minimum', 0.05
                ),
            )
        self.manager.set_language2drive_runtime(l2d_runtime)
        record_identity = (
            private_context['route']
            if private_context is not None
            else route_metadata
        )
        record_name = expected_run_id(
            language2drive_track,
            record_identity,
            language2drive_seed,
        ) or route_name
        scenario_name = (
            config.scenario_configs[0].name
            if config.scenario_configs
            else 'Language2DriveContract'
        )
        town_name = str(config.town)
        weather_id = get_weather_id(config.weather[0][1])
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
        save_name = f"{route_name}_{town_name}_{scenario_name}_{weather_id}_{currentTime}"
        self.statistics_manager.create_route_data(route_name, scenario_name, weather_id, save_name, town_name, config.index)

        # Language2Drive evaluation logger, only when --record_eval_data set
        if getattr(args, 'record_eval_data', False):
            from leaderboard.utils.closed_loop_logger import ClosedLoopLogger

            route_meta = {
                'route_id': config.name,
                'route_entry_id': route_entry_id,
                'run_id': record_name,
                'scenario_name': scenario_name,
                'weather_id': weather_id,
                'save_name': route_name,
                'town_name': config.town,
                'scenario_route_geometry_source': (
                    'private_sidecar.oracle_path.direct_v1'
                    if private_context is not None
                    else 'xml_keypoints.global_route_planner'
                ),
                'no_rendering_mode': bool(
                    getattr(args, 'language2drive_no_rendering', False)
                ),
                'language2drive_workload': language2drive_workload,
            }

            xml_waypoints, xml_expected_goal = waypoints_from_route_config(config)
            if xml_waypoints:
                route_meta['waypoints'] = xml_waypoints
            if xml_expected_goal is not None:
                route_meta['expected_goal'] = xml_expected_goal
            instruction = getattr(config, 'instruction', None)
            if instruction:
                route_meta['instruction'] = instruction
            simlingo_metadata = getattr(config, 'simlingo_metadata', None)
            if simlingo_metadata:
                route_meta['simlingo_metadata'] = simlingo_metadata
            decision_point = getattr(config, 'decision_point', None)
            if decision_point:
                route_meta['decision_point'] = decision_point
            route_meta['language2drive_public_context'] = public_context

            _record_dir = getattr(args, 'record_dir', None) or os.path.join(
                os.environ.get('TMPDIR', '/tmp'), 'language2drive-records'
            )
            rollout_record_dir = os.path.join(_record_dir, record_name)
            agent_artifact_dir = None
            if language2drive_track == L2D_TRACK_STANDARDIZED:
                # A semantic expected-run id contains PCU/arm/paraphrase/seed.
                # Never expose that identity through an otherwise innocent
                # output-path capability. The logger adopts these artifacts
                # after the agent has been torn down.
                agent_artifact_dir = os.path.join(
                    os.environ.get('TMPDIR', '/tmp'),
                    'language2drive-agent-io', secrets.token_hex(16)
                )
                os.environ['L2D_RECORD_DIR'] = agent_artifact_dir
            else:
                os.environ['L2D_RECORD_DIR'] = rollout_record_dir
            postprocessor = None
            if private_context is not None:
                postprocessor = Language2DriveTracePostprocessor(
                    private_context,
                    l2d_root,
                    stability_seconds=getattr(
                        args, 'language2drive_stability_seconds', 2.0
                    ),
                    progress_minimum=getattr(
                        args, 'language2drive_progress_minimum', 0.05
                    ),
                )
            _l2d_logger = ClosedLoopLogger(
                save_dir=rollout_record_dir,
                route_meta=route_meta,
                frame_interval=getattr(args, 'record_frame_interval', 1),
                private_context=private_context,
                postprocessor=postprocessor,
                capture_provenance=capture_provenance,
                private_runtime=l2d_runtime,
                agent_artifact_dir=agent_artifact_dir,
                workload=language2drive_workload,
            )
            self.manager.set_logger(_l2d_logger)
        else:
            self.manager.set_logger(None)


        print("\033[1m> Loading the world\033[0m", flush=True)

        # Load the world and the scenario
        try:
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            agent_class_obj = getattr(self.module_agent, agent_class_name)
            validate_privileged_no_rendering_request(
                bool(getattr(args, 'language2drive_no_rendering', False)),
                external_carla=self._external_carla,
                track=language2drive_track,
                private_context=private_context,
                agent_class=agent_class_obj,
            )
            config.language2drive_private_termination = l2d_runtime is not None
            # Scenario construction consumes the already validated private
            # witness directly. Standardized agents still receive neither this
            # attribute nor set_global_plan; Native agents may receive it as
            # their explicitly arm-specific full route.
            config.language2drive_private_route = (
                copy.deepcopy(private_context['route'])
                if private_context is not None
                else None
            )
            self._load_and_wait_for_world(args, config.town)
            self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(self.route_scenario)
            if l2d_runtime is not None:
                l2d_runtime.setup(
                    self.route_scenario.ego_vehicles[0], self.route_scenario
                )

        except Exception as exc:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self.manager.fail_language2drive_runtime("scenario_setup", exc)
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Setting up the agent\033[0m", flush=True)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog = Watchdog(args.timeout)
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            agent_class_obj = getattr(self.module_agent, agent_class_name)

            # Start the ROS1 bridge server only for ROS1 based agents.
            if getattr(agent_class_obj, 'get_ros_version')() == 1 and self._ros1_server is None:
                from leaderboard.autoagents.ros1_agent import ROS1Server
                self._ros1_server = ROS1Server()
                self._ros1_server.start()

            route_instruction = getattr(config, 'instruction', None)
            if language2drive_track == L2D_TRACK_LEGACY and route_instruction:
                # v0-only compatibility for agents that predate the explicit API.
                os.environ['L2D_ROUTE_INSTRUCTION'] = route_instruction
            else:
                os.environ.pop('L2D_ROUTE_INSTRUCTION', None)
            if language2drive_track == L2D_TRACK_STANDARDIZED:
                os.environ.pop('L2D_ROUTE_NAME', None)
                os.environ.pop('L2D_SEED', None)
                # Legacy launch scripts may expose the complete XML path.
                # Joining its unique instruction back to the current payload
                # would reveal the XML entry and coarse route envelope.
                os.environ.pop('L2D_ROUTES_XML', None)
                os.environ.pop('L2D_PRIVATE_SIDECAR', None)
                os.environ.pop('L2D_ROUTE_ENTRY_ID', None)
                # CARLA/TM retain the private rollout seed. The in-process
                # model is deterministic/greedy and receives a constant RNG
                # state: using seeds 1/2/3 here would itself reveal the private
                # seed through the first Python/NumPy/Torch random draw.
                seed_everything(0)
            else:
                os.environ['L2D_ROUTE_NAME'] = route_name
                os.environ['L2D_SEED'] = str(language2drive_seed)

            # Sanitize inherited per-route identity before executing any agent
            # constructor code. A previous Native/legacy route may otherwise
            # leave its route name or seed visible during a Standardized run.
            self.agent_instance = agent_class_obj(args.host, args.port, args.debug)

            set_context = getattr(
                self.agent_instance, 'set_language2drive_context', None
            )
            if set_context is None:
                if language2drive_track == L2D_TRACK_STANDARDIZED:
                    raise RuntimeError(
                        "Standardized Track requires agent method "
                        "set_language2drive_context(public_context)"
                    )
            else:
                set_context(public_context)

            # A versioned, trusted witness oracle may consume the hidden route
            # program on Native-System only. Standardized agents never reach a
            # private setter, even if they declare the oracle capability.
            bind_privileged_oracle_context(
                self.agent_instance,
                private_context,
                language2drive_track,
            )

            if language2drive_track != L2D_TRACK_STANDARDIZED:
                self.agent_instance.set_global_plan(
                    self.route_scenario.gps_route,
                    self.route_scenario.route,
                )

            # Keep args.agent_config immutable. Mutating it here used to produce
            # checkpoint+route0+route1 and bind route 1 to route 0's instruction.
            base_agent_config = args.agent_config.split('+', 1)[0]
            # The legacy suffix contains RouteScenario_<entry_id>; it is an
            # XML-joinable arm identity and is not a Standardized agent input.
            route_agent_config = (
                base_agent_config
                if language2drive_track == L2D_TRACK_STANDARDIZED
                else base_agent_config + '+' + save_name
            )
            self.agent_instance.setup(route_agent_config)

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                self.sensors_initialized = True

            self._agent_watchdog.stop()
            self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:", flush=True)
            print(f"{e}\033[0m\n", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self.manager.fail_language2drive_runtime(
                "sensor_configuration", e
            )
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)
            print(f"{e}\033[0m\n", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            self.manager.fail_language2drive_runtime("agent_setup", e)
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Running the route\033[0m", flush=True)

        # Run the scenario and Carla simulator
        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(self.route_scenario, self.agent_instance, config.index, config.repetition_index) # 出现segmentation fault 如果挂载-nullrhi
            self.manager.tick_count = 0
            self.manager.run_scenario()

        except AgentError as exc:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
            self.manager.fail_language2drive_runtime("agent_runtime", exc)

        except KeyboardInterrupt as exc:
            # Finalize through the common stop/cleanup path. An interrupted
            # rollout is a failed simulation, never a partially scored success.
            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self.manager.fail_language2drive_runtime(
                "keyboard_interrupt", exc
            )

        except TickRuntimeError as exc:
            entry_status, crash_message = "Started", "TickRuntime"
            self.manager.fail_language2drive_runtime("tick_runtime", exc)

        except Exception as exc:
            print("\n\033[91mError during the simulation:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self.manager.fail_language2drive_runtime("simulation", exc)

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m", flush=True)
            self.manager.stop_scenario()
            self._register_statistics(config.index, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()

        except Exception as exc:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            self.manager.fail_language2drive_runtime("scenario_stop", exc)
            self.manager.close_language2drive_runtime()
            self.manager.close_logger(summary={"termination": "stop_failed"})
            try:
                self._cleanup()
            except Exception:
                print(
                    "Language2Drive emergency cleanup failed:\n{}".format(
                        traceback.format_exc()
                    ),
                    flush=True,
                )
            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
        """
        # Load xml files to route indexer
        route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

        if args.resume:
            resume = route_indexer.validate_and_resume(args.checkpoint)
        else:
            resume = False

        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)
        else:
            self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

        crashed = False
        t1 = time.time()

        try:
            while route_indexer.peek() and not crashed:
                config = route_indexer.get_next_config()
                crashed = self._load_and_run_scenario(args, config)
                print(crashed, flush=True)
                self.statistics_manager.save_progress(
                    route_indexer.index, route_indexer.total
                )
                self.statistics_manager.write_statistics()
                if crashed:
                    print(f'{route_indexer.index} crash, [{route_indexer.index}/{route_indexer.total}], please restart', flush=True)
                    break
        finally:
            # External CARLA is caller-owned, but every evaluator process must
            # restore settings and destroy private actors on every exit path.
            if self._ros1_server is not None:
                self._ros1_server.shutdown()
            try:
                self._cleanup()
            except Exception:
                print(
                    "Language2Drive final cleanup failed:\n{}".format(
                        traceback.format_exc()
                    ),
                    flush=True,
                )
            try:
                self._reset_world_settings()
            except Exception:
                print(
                    "Language2Drive world-settings reset failed:\n{}".format(
                        traceback.format_exc()
                    ),
                    flush=True,
                )

        if not crashed:
            # Save global statistics
            print(f"cost time={time.time()-t1}", flush=True)
            print("\033[1m> Registering the global statistics\033[0m", flush=True)
            self.statistics_manager.compute_global_statistics()
            self.statistics_manager.validate_and_write_statistics(self.sensors_initialized, crashed)

        if crashed and not self._external_carla:
            cmd2 = "ps -ef | grep '-graphicsadapter="+ str(args.gpu_rank) + "' | grep -v grep | awk '{print $2}' | xargs -r kill -9"
            server = subprocess.Popen(cmd2, shell=True, preexec_fn=os.setsid)
            atexit.register(os.killpg, server.pid, signal.SIGKILL)

        return crashed

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=0, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int,
                        help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=600.0, type=float,
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes', required=True,
                        help='Name of the routes file to be executed.')
    parser.add_argument('--routes-subset', default='', type=str,
                        help='Execute a specific set of routes')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str,
                        help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")
    parser.add_argument("--gpu-rank", type=int, default=0)

    # Languag2Drive evaluation logging (disabled by default)
    parser.add_argument('--record_eval_data', action='store_true', default=False,
                        help='Enable Language2Drive per-frame JSONL evaluation logging')
    parser.add_argument('--record_dir', type=str, default=None,
                        help='Root directory for L2D evaluation log files (required for v1 Standardized Track)')
    parser.add_argument('--record_frame_interval', type=int, default=1,
                        help='Log one frame every N ticks (default: 1 = every tick)')
    parser.add_argument('--l2d-track', '--language2drive-track',
                        dest='language2drive_track',
                        choices=('legacy', 'standardized', 'native'),
                        default=None,
                        help='Language2Drive information track. Standardized never passes the global/dense route.')
    parser.add_argument('--l2d-workload', dest='language2drive_workload',
                        choices=('core', 'fdq', 'safety_priority'), default='core',
                        help='Versioned logger/scoring workload capability (default: core)')
    parser.add_argument('--l2d-external-carla',
                        dest='language2drive_external_carla', action='store_true', default=False,
                        help='Use the existing --host/--port CARLA server; never launch or kill CARLA')
    parser.add_argument('--l2d-no-rendering',
                        dest='language2drive_no_rendering', action='store_true', default=False,
                        help='Privileged oracle/native only: disable rendering while preserving 20 Hz physics/logging')
    parser.add_argument('--l2d-seed', '--language2drive-seed',
                        dest='language2drive_seed', type=int, default=None,
                        help='Seed Python/NumPy/Torch for this run (defaults to --traffic-manager-seed)')
    parser.add_argument('--l2d-private-sidecar',
                        dest='language2drive_private_sidecar', default=None,
                        help='Evaluator-only v1 sidecar; never forwarded to the agent')
    parser.add_argument('--l2d-route-entry-id',
                        dest='language2drive_route_entry_id', type=int, default=None,
                        help='Assert that a single-run process is evaluating this XML route id')
    parser.add_argument('--l2d-v1-root',
                        dest='language2drive_v1_root', default=None,
                        help='Root of language2drive.v1 used by the offline contract scorer')
    parser.add_argument('--l2d-stability-seconds',
                        dest='language2drive_stability_seconds', type=float, default=2.0,
                        help='Post-completion stability window (default: 2 seconds)')
    parser.add_argument('--l2d-progress-minimum',
                        dest='language2drive_progress_minimum', type=float, default=0.05,
                        help='Minimum hidden-path normalized progress for global Progress (default: 0.05)')

    arguments = parser.parse_args()

    # Agent modules are imported in-process.  Remove raw CLI values (especially
    # the evaluator-only private sidecar path) before importing agent code.
    sys.argv[:] = [sys.argv[0]]

    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    crashed = leaderboard_evaluator.run(arguments)

    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
