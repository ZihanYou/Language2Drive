"""Sidecar-aware AlpaSim driver service without changing upstream protobufs."""

import os
from pathlib import Path
from typing import Any, List

from .language import SidecarLanguageConditionProvider


def _upstream_types() -> Any:
    try:
        from alpasim_driver.main import EgoDriverService
        from alpasim_driver.models import DriveCommand
        from alpasim_driver.models.base import PredictionInput
    except ImportError as error:
        raise RuntimeError(
            "l2d-alpasim-driver must run inside the pinned AlpaSim Python environment"
        ) from error
    return EgoDriverService, DriveCommand, PredictionInput


def service_class() -> Any:
    EgoDriverService, DriveCommand, PredictionInput = _upstream_types()

    class LanguageConditionedEgoDriverService(EgoDriverService):
        def __init__(self, cfg: Any) -> None:
            if cfg.route is None or cfg.route.use_waypoint_commands:
                raise ValueError("RealSim requires route.use_waypoint_commands=false")
            supported_models = {
                "l2d_alpamayo1_5",
                "l2d_alpamayo2_super",
                "l2d_simlingo",
                "l2d_unified",
            }
            if cfg.model.model_type not in supported_models:
                raise ValueError(
                    "RealSim v0.1 requires a language-conditioned benchmark "
                    "model plugin, got %s" % cfg.model.model_type
                )
            sidecar_path = os.environ.get("L2D_REALSIM_SIDECAR")
            if not sidecar_path:
                raise ValueError("L2D_REALSIM_SIDECAR is required")
            self._language_provider = SidecarLanguageConditionProvider.from_path(
                Path(sidecar_path)
            )
            super().__init__(cfg)

        def start_session(self, request: Any, context: Any) -> Any:
            response = super().start_session(request, context)
            # Upstream defaults the command to STRAIGHT even when waypoint
            # command inference is disabled. Use the explicit neutral enum.
            self._sessions[request.session_uuid].current_command = DriveCommand.UNKNOWN
            return response

        def submit_image_observation(self, request: Any, context: Any) -> Any:
            """Acknowledge frozen-rig cameras that this model does not consume.

            RealSim records the complete, reviewed four-camera rig in the ASL.
            A model is still allowed to declare a strict subset through
            ``inference.use_cameras``.  Upstream rejects every image outside
            that subset, which makes single-camera official models unusable on
            the common benchmark envelope.  Acknowledge those images without
            decoding or caching them; desired cameras retain the exact
            upstream path and are the only pixels exposed to the model.
            """
            session = self._sessions[request.session_uuid]
            if request.camera_image.logical_id not in session.frame_caches:
                from alpasim_grpc.v0.common_pb2 import Empty

                return Empty()
            return super().submit_image_observation(request, context)

        def _run_batch(self, batch: List[Any]) -> List[Any]:
            inputs = []
            for job in batch:
                speed, acceleration = self._get_speed_and_acceleration(job.session)
                inference_seed = job.session.seed + job.session.inference_count
                job.session.inference_count += 1
                camera_images = self._prepare_camera_images(job.session)
                self._warn_once_if_frames_trail_request(job, camera_images)
                prediction_input = PredictionInput(
                    camera_images=camera_images,
                    command=DriveCommand.UNKNOWN,
                    speed=speed,
                    acceleration=acceleration,
                    ego_pose_history=job.session.poses,
                    inference_seed=inference_seed,
                    previous_plan=job.session.last_selected_plan,
                    # The runtime may retain a common envelope for progress and
                    # topology, but no route tensor/polyline crosses into the
                    # tested model. This closes both command and A1.5 nav-text
                    # leakage paths at the boundary.
                    route=None,
                )
                snapshot = self._language_provider.get_instruction(
                    job.session.uuid, job.timestamp_us
                )
                # PredictionInput is a non-slotted dataclass at the pinned
                # AlpaSim commit. Only this string crosses into the model.
                prediction_input.language_instruction = snapshot.text
                if type(self._model).__name__ == "UnifiedTrajectoryModel":
                    prediction_input.l2d_session_uuid = job.session.uuid
                    prediction_input.l2d_timestamp_us = job.timestamp_us
                    from omegaconf import OmegaConf
                    prediction_input.l2d_camera_calibration = OmegaConf.to_container(self._cfg.rectification, resolve=True)
                inputs.append(prediction_input)

            predictions = self._model.predict_batch(inputs)
            if len(predictions) != len(batch):
                raise RuntimeError(
                    "AlpaSim model returned %d predictions for a batch of %d"
                    % (len(predictions), len(batch))
                )
            for job, prediction in zip(batch, predictions):
                job.session.last_selected_plan = prediction.selected_plan
            return predictions

    return LanguageConditionedEgoDriverService


def main() -> int:
    try:
        import alpasim_driver.main as upstream
    except ImportError as error:
        raise RuntimeError(
            "l2d-alpasim-driver must run inside the pinned AlpaSim Python environment"
        ) from error
    upstream.EgoDriverService = service_class()
    upstream.main()
    return 0
