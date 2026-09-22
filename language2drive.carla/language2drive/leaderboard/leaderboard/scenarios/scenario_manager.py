#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import os
import signal
import sys
import time
import traceback

import py_trees
import carla
import threading

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError, TickRuntimeError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._watchdog = None
        self._agent_watchdog = None
        self._scenario_thread = None

        self._statistics_manager = statistics_manager

        self.tick_count = 0

        # Language2Drive evaluation logger (None = disabled)
        self._logger = None
        self._language2drive_runtime = None
        self._l2d_stopped_seconds = 0.0

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def set_logger(self, logger) -> None:
        """Attach (or detach) a ClosedLoopLogger for Language2Drive evaluation."""
        self._logger = logger

    def set_language2drive_runtime(self, runtime) -> None:
        """Attach evaluator-private actor/termination runtime for one route."""
        if self._language2drive_runtime is not None and runtime is not None:
            raise RuntimeError("Language2Drive runtime already attached")
        self._language2drive_runtime = runtime

    def close_language2drive_runtime(self) -> None:
        """Idempotently destroy evaluator-private semantic actors."""
        runtime = self._language2drive_runtime
        self._language2drive_runtime = None
        if runtime is not None:
            runtime.cleanup()

    def fail_language2drive_runtime(self, operation, exc=None) -> None:
        """Record an evaluator-owned abort before closing the private trace."""

        runtime = self._language2drive_runtime
        if runtime is not None:
            runtime.record_evaluator_failure(operation, exc)

    def close_logger(self, summary=None) -> None:
        """Idempotently flush and detach the Language2Drive route logger."""
        logger = self._logger
        self._logger = None
        if logger is not None:
            logger.close_route(summary=summary)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Agent took longer than {}s to send its command".format(self._timeout))
        elif self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self.fail_language2drive_runtime(
            "manager_cleanup_without_private_termination"
        )
        self.close_language2drive_runtime()
        self.close_logger(summary={"termination": "manager_cleanup"})
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None
        self._watchdog = None
        self._agent_watchdog = None

    def load_scenario(self, scenario, agent, route_index, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

        # Language2Drive per-route logging
        if self._logger is not None:
            try:
                self._logger.start_route()
            except Exception:
                print(
                    "Language2Drive logger failed to start:\n{}".format(
                        traceback.format_exc()
                    ),
                    flush=True,
                )
                raise

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            time.sleep(1)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()
        self._l2d_stopped_seconds = 0.0

        # Detects if the simulation is down
        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        # Stop the agent from freezing the simulation
        self._agent_watchdog = Watchdog(self._timeout)
        self._agent_watchdog.start()

        self._running = True

        # Thread for build_scenarios
        self._scenario_thread = threading.Thread(target=self.build_scenarios_loop, args=(self._debug_mode > 0, ))
        self._scenario_thread.start()

        while self._running:
            self._tick_scenario()

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            if self._language2drive_runtime is not None:
                self._language2drive_runtime.tick_world()
            self.tick_count += 1
            self._watchdog.pause()

            # if self.tick_count > 4000:
            #     raise TickRuntimeError("RuntimeError, tick_count > 4000")

            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                ego_action = self._agent_wrapper()
                self._agent_watchdog.pause()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self._watchdog.resume()
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()

            # Log after the criteria tree tick so collision/off-road/red-light
            # TrafficEvents for this snapshot are available authoritatively.
            if self._logger is not None:
                agent = getattr(self._agent_wrapper, '_agent', None)
                self._logger.log_frame(
                    timestamp.frame,
                    timestamp,
                    self.ego_vehicles[0],
                    ego_action,
                    self.scenario,
                    self._statistics_manager,
                    controller_source=getattr(
                        agent, 'controller_source', 'unknown'
                    ),
                )

            if (
                self._language2drive_runtime is not None
                and self._language2drive_runtime.should_terminate
            ):
                reason = (
                    self._language2drive_runtime.termination_reason
                    or self._language2drive_runtime.failure_reason
                    or "unknown"
                )
                print(
                    "Language2Drive private termination: {}".format(reason),
                    flush=True,
                )
                self._running = False

            # Legacy demo caps remain available without a private v1 runtime.
            # A v1 Standardized rollout is owned exclusively by the private
            # contract/deadline monitor, so a public coarse endpoint or stale
            # environment variable cannot truncate it.
            if self._language2drive_runtime is None:
                elapsed_game_time = GameTime.get_time() - self.start_game_time
                max_game_seconds = float(
                    os.environ.get("L2D_MAX_GAME_SECONDS", "0") or 0
                )
                stopped_early_seconds = float(
                    os.environ.get("L2D_STOPPED_EARLY_SECONDS", "0") or 0
                )
                stopped_grace_seconds = float(
                    os.environ.get("L2D_STOPPED_GRACE_SECONDS", "8") or 8
                )
                if (
                    stopped_early_seconds > 0
                    and elapsed_game_time >= stopped_grace_seconds
                ):
                    speed = self.ego_vehicles[0].get_velocity().length()
                    if speed < 0.05:
                        self._l2d_stopped_seconds += float(
                            getattr(timestamp, "delta_seconds", 0.05)
                        )
                    else:
                        self._l2d_stopped_seconds = 0.0
                    if self._l2d_stopped_seconds >= stopped_early_seconds:
                        print(
                            "L2D demo stopped after the ego remained stationary "
                            f"for {self._l2d_stopped_seconds:.1f} game seconds",
                            flush=True,
                        )
                        self._running = False
                if max_game_seconds > 0 and elapsed_game_time >= max_game_seconds:
                    print(
                        f"L2D demo reached {max_game_seconds:.1f} game-second limit",
                        flush=True,
                    )
                    self._running = False

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if (
                self._language2drive_runtime is None
                and self.scenario_tree.status != py_trees.common.Status.RUNNING
            ):
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()
            self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                          carla.Rotation(pitch=-90)))

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog:
            self._watchdog.stop()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        self.compute_duration_time()

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                self._agent_wrapper.cleanup()
                self._agent_wrapper = None

            # self.analyze_scenario()

        # Make sure the scenario thread finishes to avoid blocks
        self._running = False
        self._scenario_thread.join()
        self._scenario_thread = None
        self.fail_language2drive_runtime(
            "scenario_stopped_without_private_termination"
        )
        self.close_language2drive_runtime()
        self.close_logger(
            summary={
                "termination": "scenario_stopped",
                "scenario_duration_system": self.scenario_duration_system,
                "scenario_duration_game": self.scenario_duration_game,
            }
        )

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)
