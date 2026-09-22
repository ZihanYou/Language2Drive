"""Fail-fast connection setup for an evaluator-external CARLA server.

This module deliberately has no CARLA import.  The leaderboard injects its
already imported CARLA module, which keeps the behavior unit-testable without
starting a simulator or loading libcarla in the test process.
"""

from __future__ import annotations

import math
from typing import Any, Tuple


class ExternalCarlaConnectionError(RuntimeError):
    """The explicitly requested external CARLA endpoint was not ready."""


def _port(value: Any, label: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be an integer".format(label)) from exc
    if not 1 <= port <= 65535:
        raise ValueError("{} must be in [1, 65535]".format(label))
    return port


def connect_existing_carla(
    carla_module,
    *,
    host: str,
    port: int,
    traffic_manager_port: int,
    timeout_seconds: float,
    frame_rate: float,
) -> Tuple[Any, float, Any]:
    """Connect to one existing server without selecting or launching a port.

    The initial RPC probe is capped at ten seconds so a missing external server
    fails quickly.  Once the world and TrafficManager are confirmed and placed
    in synchronous mode, the caller-requested timeout is restored for custom-map
    loading and normal evaluation.
    """

    endpoint_port = _port(port, "CARLA port")
    tm_port = _port(traffic_manager_port, "TrafficManager port")
    configured_timeout = float(timeout_seconds)
    frequency = float(frame_rate)
    if not math.isfinite(configured_timeout) or configured_timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("frame_rate must be finite and positive")
    probe_timeout = min(configured_timeout, 10.0)

    client = None
    world = None
    traffic_manager = None
    original_settings = None
    try:
        client = carla_module.Client(str(host), endpoint_port)
        client.set_timeout(probe_timeout)
        world = client.get_world()
        if world is None or world.get_map() is None:
            raise RuntimeError("get_world/get_map returned no live world")

        traffic_manager = client.get_trafficmanager(tm_port)
        if traffic_manager is None:
            raise RuntimeError("get_trafficmanager returned no manager")

        settings = world.get_settings()
        original_settings = {
            name: getattr(settings, name)
            for name in (
                "synchronous_mode", "fixed_delta_seconds",
                "deterministic_ragdolls", "spectator_as_ego",
                "no_rendering_mode",
            )
            if hasattr(settings, name)
        }
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / frequency
        settings.deterministic_ragdolls = True
        settings.spectator_as_ego = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        # Map loading can legitimately exceed the short readiness probe.
        client.set_timeout(configured_timeout)
        return client, configured_timeout, traffic_manager
    except Exception as exc:
        # A half-configured caller-owned server is worse than a failed probe.
        # Roll back every world field changed above and explicitly release the
        # TM from synchronous/hybrid mode before surfacing the connection error.
        if world is not None and original_settings is not None:
            try:
                settings = world.get_settings()
                for name, value in original_settings.items():
                    setattr(settings, name, value)
                world.apply_settings(settings)
            except Exception:
                pass
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            try:
                traffic_manager.set_hybrid_physics_mode(False)
            except Exception:
                pass
        raise ExternalCarlaConnectionError(
            "external CARLA at {}:{} (TrafficManager {}) is not ready: {}: {}".format(
                host, endpoint_port, tm_port, type(exc).__name__, exc
            )
        ) from exc
