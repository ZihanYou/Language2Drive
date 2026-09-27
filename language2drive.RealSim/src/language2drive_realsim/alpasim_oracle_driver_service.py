"""AlpaSim service for curator-frozen privileged oracle trajectories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from .io import load_json
from .oracle_model import ORACLE_CONTROLLER_ID


def _upstream_types() -> Any:
    try:
        from alpasim_driver.main import EgoDriverService
        from alpasim_driver.models import DriveCommand
        from alpasim_driver.models.base import PredictionInput
        from alpasim_utils.geometry import pose_from_grpc
    except ImportError as error:
        raise RuntimeError(
            "l2d-alpasim-oracle-driver must run inside the pinned AlpaSim environment"
        ) from error
    return EgoDriverService, DriveCommand, PredictionInput, pose_from_grpc


def service_class() -> Any:
    EgoDriverService, DriveCommand, PredictionInput, pose_from_grpc = (
        _upstream_types()
    )

    class PrivilegedOracleEgoDriverService(EgoDriverService):
        def __init__(self, cfg: Any) -> None:
            if cfg.model.model_type != "l2d_privileged_oracle":
                raise ValueError(
                    "oracle service requires model type l2d_privileged_oracle"
                )
            sidecar_path = os.environ.get("L2D_REALSIM_SIDECAR")
            if not sidecar_path:
                raise ValueError("L2D_REALSIM_SIDECAR is required")
            sidecar = load_json(Path(sidecar_path))
            if sidecar.get("schema_version") != "l2d.realsim.sidecar.v0.1":
                raise ValueError("invalid RealSim sidecar schema")
            self._private_sessions = sidecar.get("sessions", {})
            super().__init__(cfg)

        def start_session(self, request: Any, context: Any) -> Any:
            private = self._private_sessions.get(request.session_uuid)
            if private is None:
                raise ValueError("oracle received an unknown opaque session")
            plan = private.get("oracle_plan")
            if not isinstance(plan, dict) or plan.get("controller") != ORACLE_CONTROLLER_ID:
                raise ValueError("session has no frozen privileged oracle plan")
            response = super().start_session(request, context)
            self._sessions[request.session_uuid].current_command = DriveCommand.UNKNOWN
            return response

        def _run_batch(self, batch: List[Any]) -> List[Any]:
            inputs = []
            for job in batch:
                if job.pose is None:
                    raise ValueError("oracle session has no current ego pose")
                private = self._private_sessions[job.session.uuid]
                pose = pose_from_grpc(job.pose.pose)
                prediction_input = PredictionInput(
                    camera_images={},
                    command=DriveCommand.UNKNOWN,
                    speed=0.0,
                    acceleration=0.0,
                    ego_pose_history=job.session.poses,
                    inference_seed=job.session.seed + job.session.inference_count,
                    previous_plan=None,
                    route=None,
                )
                job.session.inference_count += 1
                prediction_input.oracle_plan = private["oracle_plan"]
                prediction_input.oracle_pose_local_to_rig = pose
                prediction_input.oracle_time_us = int(job.pose.timestamp_us)
                prediction_input.oracle_start_timestamp_us = int(
                    private["start_timestamp_us"]
                )
                inputs.append(prediction_input)
            predictions = self._model.predict_batch(inputs)
            if len(predictions) != len(batch):
                raise RuntimeError("oracle model returned an invalid batch size")
            return predictions

    return PrivilegedOracleEgoDriverService


def main() -> int:
    try:
        import alpasim_driver.main as upstream
    except ImportError as error:
        raise RuntimeError(
            "l2d-alpasim-oracle-driver must run inside the pinned AlpaSim environment"
        ) from error
    upstream.EgoDriverService = service_class()
    upstream.main()
    return 0
