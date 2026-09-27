"""Privileged NRE-local path follower used only to qualify oracle witnesses."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .constants import ORACLE_CONTROLLER_ID
from .trajectory import sample_plan

try:
    from alpasim_driver.models import DriveCommand
    from alpasim_driver.models.base import BaseTrajectoryModel, ModelPrediction
    from alpasim_utils.geometry import Pose
except ImportError:  # Keep source inspection usable outside the pinned live env.
    DriveCommand = Any  # type: ignore
    BaseTrajectoryModel = object  # type: ignore
    ModelPrediction = Any  # type: ignore
    Pose = Any  # type: ignore


def _pose_matrix(position: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    matrix[:3, 3] = position
    return matrix


class PrivilegedOraclePathModel(BaseTrajectoryModel):  # type: ignore[misc]
    """Return a receding horizon from a curator-frozen privileged path.

    This model is deliberately registered under a distinct AlpaSim model type.
    It never serves benchmark-model runs and consumes no camera, language, route,
    or learned-model input when selecting the trajectory.
    """

    HORIZON_S = 6.0

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: Any,
        camera_ids: List[str],
        context_length: int,
        output_frequency_hz: int,
    ) -> "PrivilegedOraclePathModel":
        del model_cfg, device
        return cls(camera_ids, context_length or 1, output_frequency_hz)

    def __init__(
        self, camera_ids: List[str], context_length: int, output_frequency_hz: int
    ) -> None:
        if not camera_ids:
            raise ValueError("privileged oracle requires at least one runtime camera")
        if output_frequency_hz < 1:
            raise ValueError("oracle output frequency must be positive")
        self._camera_ids = list(camera_ids)
        self._context_length = int(context_length)
        self._output_frequency_hz = int(output_frequency_hz)

    @property
    def camera_ids(self) -> List[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> int:
        return int(command)

    def predict(self, prediction_input: Any) -> Any:
        plan = getattr(prediction_input, "oracle_plan", None)
        current_pose = getattr(
            prediction_input, "oracle_pose_local_to_rig", None
        )
        planning_time_us = getattr(prediction_input, "oracle_time_us", None)
        start_timestamp_us = getattr(
            prediction_input, "oracle_start_timestamp_us", None
        )
        if (
            not isinstance(plan, dict)
            or current_pose is None
            or planning_time_us is None
            or start_timestamp_us is None
        ):
            raise ValueError("privileged oracle input is incomplete")
        if plan.get("controller") != ORACLE_CONTROLLER_ID:
            raise ValueError("unrecognized privileged oracle controller")

        step_s = 1.0 / self._output_frequency_hz
        count = max(1, int(round(self.HORIZON_S * self._output_frequency_hz)))
        elapsed_s = (int(planning_time_us) - int(start_timestamp_us)) / 1_000_000.0
        relative_times = np.arange(1, count + 1, dtype=float) * step_s
        sample_times = elapsed_s + relative_times
        positions_local, yaws_local = sample_plan(plan, sample_times)

        pose_rig_to_local = current_pose.inverse()
        positions_rig = []
        rotations_rig = []
        for position, yaw in zip(positions_local, yaws_local):
            target_local = Pose.from_se3(_pose_matrix(position, float(yaw)))
            target_rig = pose_rig_to_local @ target_local
            target_matrix = target_rig.as_se3()
            positions_rig.append(target_matrix[:3, 3])
            rotations_rig.append(target_matrix[:3, :3])

        waypoint_timestamps_us = int(planning_time_us) + np.rint(
            relative_times * 1_000_000.0
        ).astype(np.uint64)
        return ModelPrediction(
            candidate_positions=np.asarray([positions_rig], dtype=float),
            candidate_rotations=np.asarray([rotations_rig], dtype=float),
            model_t0_us=int(planning_time_us),
            pose_local_to_rig_t0=current_pose,
            waypoint_timestamps_us=waypoint_timestamps_us,
        )
