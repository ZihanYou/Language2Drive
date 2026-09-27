"""Opt-in observed-velocity handoff profile for the AlpaSim host runtime."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import importlib
import json
import logging
import threading
from typing import Any

import numpy as np

from .observed_motion_diagnostic import velocity_from_observed_poses


PROFILE = "handoff-observed-velocity-20260917"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _HandoffContext:
    driver_service: Any
    timestamp_us: int
    poses: tuple[dict[str, Any], dict[str, Any]]
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class _Installation:
    policy_class: type[Any]
    driver_class: type[Any]
    original_policy_run: Any
    original_submit_trajectory: Any
    wrapped_policy_run: Any
    wrapped_submit_trajectory: Any


_ACTIVE: ContextVar[_HandoffContext | None] = ContextVar(
    "l2d_handoff_observed_velocity", default=None
)
_LOCK = threading.Lock()
_INSTALLATION: _Installation | None = None
_APPLIED: set[tuple[int, str, int]] = set()
_AUDIT: list[dict[str, Any]] = []


def _latest_observed_poses(trajectory: Any, timestamp_us: int) -> list[dict[str, Any]] | None:
    timestamps = np.asarray(trajectory.timestamps_us)
    eligible = np.flatnonzero(timestamps <= timestamp_us)
    if len(eligible) < 2 or int(timestamps[eligible[-1]]) != timestamp_us:
        return None
    records = []
    for index in eligible[-2:]:
        pose = trajectory.get_pose(int(index))
        records.append(
            {
                "timestamp_us": int(timestamps[index]),
                "position_xyz": [float(value) for value in pose.vec3],
                "quaternion_xyzw": [float(value) for value in pose.quat],
            }
        )
    return records


def _context_for_policy(event: Any, state: Any) -> _HandoffContext | None:
    unbound = state.unbound
    timestamp_us = int(event.timestamp_us)
    target_time_us = timestamp_us + int(event.interval_us)
    last_update_us = state.last_egopose_update_us
    if (
        int(unbound.force_gt_duration_us) <= 0
        or timestamp_us != int(unbound.closed_loop_start_us)
        or last_update_us is None
        or int(last_update_us) < 0
        or int(last_update_us) >= timestamp_us
        or target_time_us in unbound.force_gt_period
    ):
        return None
    records = _latest_observed_poses(state.ego_trajectory_estimate, timestamp_us)
    if records is None:
        return None
    velocity = velocity_from_observed_poses(records, timestamp_us)
    if all(component == 0.0 for component in velocity):
        return None
    return _HandoffContext(
        driver_service=event.services.driver,
        timestamp_us=timestamp_us,
        poses=(records[0], records[1]),
        velocity=tuple(float(component) for component in velocity),
    )


def _state_values(state: Any) -> tuple[float, ...]:
    values = []
    for field in (
        "linear_velocity",
        "angular_velocity",
        "linear_acceleration",
        "angular_acceleration",
    ):
        vector = getattr(state, field)
        values.extend((float(vector.x), float(vector.y), float(vector.z)))
    return tuple(values)


def _copy_states(states: list[Any]) -> list[Any]:
    copies = []
    for state in states:
        copied = type(state)()
        copied.CopyFrom(state)
        copies.append(copied)
    return copies


def _resolve_runtime_classes() -> tuple[type[Any], type[Any]]:
    policy = importlib.import_module("alpasim_runtime.events.policy")
    driver = importlib.import_module("alpasim_runtime.services.driver_service")
    return policy.PolicyEvent, driver.DriverService


def install() -> None:
    """Install the profile on the process-local official runtime classes."""

    global _INSTALLATION
    with _LOCK:
        if _INSTALLATION is not None:
            raise RuntimeError(f"{PROFILE} is already installed")
        policy_class, driver_class = _resolve_runtime_classes()
        original_policy_run = policy_class.run
        original_submit = driver_class.submit_trajectory

        @wraps(original_policy_run)
        async def policy_run(event: Any, state: Any, queue: Any) -> Any:
            token = _ACTIVE.set(_context_for_policy(event, state))
            try:
                return await original_policy_run(event, state, queue)
            finally:
                _ACTIVE.reset(token)

        @wraps(original_submit)
        async def submit_trajectory(
            service: Any, trajectory: Any, dynamic_states_in_rig: list[Any]
        ) -> Any:
            context = _ACTIVE.get()
            states = dynamic_states_in_rig
            if (
                context is not None
                and context.driver_service is service
                and isinstance(states, list)
                and len(states) > 0
                and len(states) == len(trajectory.timestamps_us)
                and int(trajectory.timestamps_us[-1]) == context.timestamp_us
                and all(value == 0.0 for value in _state_values(states[-1]))
            ):
                session_uuid = str(service._require_session_info().uuid)
                key = (id(service), session_uuid, context.timestamp_us)
                with _LOCK:
                    first_submission = key not in _APPLIED
                    if first_submission:
                        _APPLIED.add(key)
                if first_submission:
                    states = _copy_states(states)
                    old_velocity = [
                        float(states[-1].linear_velocity.x),
                        float(states[-1].linear_velocity.y),
                        float(states[-1].linear_velocity.z),
                    ]
                    new_velocity = list(context.velocity)
                    states[-1].linear_velocity.x = new_velocity[0]
                    states[-1].linear_velocity.y = new_velocity[1]
                    states[-1].linear_velocity.z = new_velocity[2]
                    record = {
                        "profile": PROFILE,
                        "audit_stage": "prepared_for_submission",
                        "session_uuid": session_uuid,
                        "timestamp_us": context.timestamp_us,
                        "old_linear_velocity_m_s": old_velocity,
                        "new_linear_velocity_m_s": new_velocity,
                        "observed_poses": [dict(pose) for pose in context.poses],
                        "approximation": "two-pose backward finite difference in current rig axes",
                        "ground_truth_used": False,
                        "future_used": False,
                        "route_used": False,
                    }
                    with _LOCK:
                        _AUDIT.append(record)
                    logger.info("l2d_handoff_observed_velocity %s", json.dumps(record, sort_keys=True))
            return await original_submit(service, trajectory, states)

        policy_class.run = policy_run
        driver_class.submit_trajectory = submit_trajectory
        _APPLIED.clear()
        _AUDIT.clear()
        _INSTALLATION = _Installation(
            policy_class=policy_class,
            driver_class=driver_class,
            original_policy_run=original_policy_run,
            original_submit_trajectory=original_submit,
            wrapped_policy_run=policy_run,
            wrapped_submit_trajectory=submit_trajectory,
        )


def uninstall() -> None:
    """Remove this profile without overwriting another runtime patch."""

    global _INSTALLATION
    with _LOCK:
        installation = _INSTALLATION
        if installation is None:
            raise RuntimeError(f"{PROFILE} is not installed")
        if (
            installation.policy_class.run is not installation.wrapped_policy_run
            or installation.driver_class.submit_trajectory
            is not installation.wrapped_submit_trajectory
        ):
            raise RuntimeError("runtime methods changed after profile installation")
        installation.policy_class.run = installation.original_policy_run
        installation.driver_class.submit_trajectory = (
            installation.original_submit_trajectory
        )
        _INSTALLATION = None
        _APPLIED.clear()


def audit_records() -> tuple[dict[str, Any], ...]:
    """Return detached structured records for applied handoff substitutions."""

    with _LOCK:
        return tuple(json.loads(json.dumps(record)) for record in _AUDIT)
