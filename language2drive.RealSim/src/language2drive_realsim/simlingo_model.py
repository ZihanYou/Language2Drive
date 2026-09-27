"""AlpaSim adapter for the official SimLingo release.

The released model depends on Python 3.8-era packages while the pinned
AlpaSim driver runs on Python 3.12.  A persistent local worker keeps those
environments isolated.  The protocol carries one RGB frame and JSON metadata;
it deliberately does not use pickle or expose route/command geometry.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
from typing import Any, BinaryIO

import numpy as np

try:
    import torch
    from alpasim_driver.models.base import (
        BaseTrajectoryModel,
        DriveCommand,
        ModelPrediction,
    )
    from alpasim_utils.geometry import Pose, Trajectory, pose_from_grpc
except ImportError:  # Keep source inspection usable outside the AlpaSim env.
    torch = None  # type: ignore
    BaseTrajectoryModel = object  # type: ignore
    DriveCommand = Any  # type: ignore
    ModelPrediction = Any  # type: ignore
    Pose = Any  # type: ignore
    Trajectory = Any  # type: ignore
    pose_from_grpc = None


_HEADER = struct.Struct("!I")
_MAX_JSON_BYTES = 1 << 20
_MAX_IMAGE_BYTES = 32 << 20
_OFFICIAL_CAMERA_ID = "camera_front_wide_120fov"
_OFFICIAL_CONTEXT_LENGTH = 1
_OFFICIAL_OUTPUT_HZ = 4
_OFFICIAL_FRAME_HW = (512, 1024)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"worker stream ended with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_request(
    image: np.ndarray,
    *,
    speed_m_s: float,
    instruction: str,
    inference_seed: int,
) -> bytes:
    """Encode one inference request using length-framed JSON plus raw RGB."""

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("SimLingo input must be an HWC uint8 RGB image")
    image_bytes = array.tobytes(order="C")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError("SimLingo image exceeds protocol size limit")
    text = str(instruction).strip()
    if not text and os.environ.get("L2D_DIAGNOSTIC_ALLOW_EMPTY") != "1":
        raise ValueError("Language2Drive sidecar instruction must be non-empty")
    metadata = json.dumps(
        {
            "op": "infer",
            "shape": list(array.shape),
            "dtype": "uint8",
            "image_nbytes": len(image_bytes),
            "speed_m_s": float(speed_m_s),
            "instruction": text,
            "inference_seed": int(inference_seed),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(metadata) > _MAX_JSON_BYTES:
        raise ValueError("SimLingo request metadata exceeds protocol size limit")
    return _HEADER.pack(len(metadata)) + metadata + image_bytes


def encode_response(payload: dict[str, Any]) -> bytes:
    """Encode a worker response as one bounded JSON packet."""

    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(body) > _MAX_JSON_BYTES:
        raise ValueError("SimLingo response exceeds protocol size limit")
    return _HEADER.pack(len(body)) + body


def read_response(stream: BinaryIO) -> dict[str, Any]:
    """Read and validate one response packet."""

    (size,) = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    if size < 2 or size > _MAX_JSON_BYTES:
        raise ValueError(f"invalid SimLingo response length {size}")
    payload = json.loads(_read_exact(stream, size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SimLingo response must be a JSON object")
    return payload


def read_request(stream: BinaryIO) -> tuple[dict[str, Any], np.ndarray]:
    """Read one worker request and reconstruct its bounded RGB array."""

    (size,) = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    if size < 2 or size > _MAX_JSON_BYTES:
        raise ValueError(f"invalid SimLingo request metadata length {size}")
    metadata = json.loads(_read_exact(stream, size).decode("utf-8"))
    if not isinstance(metadata, dict) or metadata.get("op") != "infer":
        raise ValueError("invalid SimLingo request")
    shape = metadata.get("shape")
    if (
        metadata.get("dtype") != "uint8"
        or not isinstance(shape, list)
        or len(shape) != 3
        or shape[2] != 3
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError("invalid SimLingo image metadata")
    expected = int(np.prod(shape, dtype=np.int64))
    declared = metadata.get("image_nbytes")
    if declared != expected or expected > _MAX_IMAGE_BYTES:
        raise ValueError("SimLingo image byte count does not match its shape")
    image = np.frombuffer(_read_exact(stream, expected), dtype=np.uint8)
    return metadata, image.reshape(tuple(shape))


def _polyline_distances(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


def fuse_official_trajectories(
    route_xy_carla: Any, speed_waypoints_xy_carla: Any
) -> np.ndarray:
    """Apply SimLingo's official lateral/longitudinal division of labor.

    The CARLA agent steers from ``pred_route`` and derives desired speed from
    ``pred_speed_wps``.  For a trajectory-only controller, cumulative distance
    from the speed waypoints is sampled along the predicted route.  The final
    sign flip converts CARLA's y-right convention to AlpaSim's y-left frame.
    """

    route = np.asarray(route_xy_carla, dtype=np.float64)
    speed_waypoints = np.asarray(speed_waypoints_xy_carla, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 1:
        raise ValueError(f"unexpected SimLingo route shape {route.shape}")
    if (
        speed_waypoints.ndim != 2
        or speed_waypoints.shape[1] != 2
        or len(speed_waypoints) < 1
    ):
        raise ValueError(
            f"unexpected SimLingo speed-waypoint shape {speed_waypoints.shape}"
        )
    if not np.isfinite(route).all() or not np.isfinite(speed_waypoints).all():
        raise ValueError("SimLingo returned non-finite trajectory values")

    origin = np.zeros((1, 2), dtype=np.float64)
    route_path = route if np.linalg.norm(route[0]) <= 1e-7 else np.vstack((origin, route))
    speed_path = np.vstack((origin, speed_waypoints))
    requested_distances = _polyline_distances(speed_path)[1:]
    route_distances = _polyline_distances(route_path)

    # Remove repeated path points so interpolation has a strictly increasing axis.
    keep = np.concatenate(([True], np.diff(route_distances) > 1e-7))
    route_path = route_path[keep]
    route_distances = route_distances[keep]
    if len(route_path) == 1:
        sampled = np.repeat(route_path, len(requested_distances), axis=0)
    else:
        clipped = np.minimum(requested_distances, route_distances[-1])
        sampled = np.column_stack(
            [np.interp(clipped, route_distances, route_path[:, axis]) for axis in (0, 1)]
        )
        beyond = requested_distances > route_distances[-1]
        if beyond.any():
            tangent = route_path[-1] - route_path[-2]
            tangent /= np.linalg.norm(tangent)
            sampled[beyond] = route_path[-1] + (
                requested_distances[beyond] - route_distances[-1]
            )[:, None] * tangent

    sampled[:, 1] *= -1.0
    return sampled.astype(np.float32)


class _WorkerClient:
    def __init__(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        code_root: Path,
        checkpoint_path: Path,
        config_path: Path,
        base_model_path: Path,
        device: str,
        startup_timeout_s: float = 300.0,
        inference_timeout_s: float = 300.0,
    ) -> None:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(code_root) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        self._process = subprocess.Popen(
            [
                str(python_executable),
                str(worker_script),
                "--code-root",
                str(code_root),
                "--checkpoint",
                str(checkpoint_path),
                "--config",
                str(config_path),
                "--base-model",
                str(base_model_path),
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
            bufsize=0,
        )
        self._inference_timeout_s = inference_timeout_s
        atexit.register(self.close)
        ready = self._read_with_timeout(startup_timeout_s)
        if not ready.get("ok") or not ready.get("ready"):
            self.close()
            raise RuntimeError(f"SimLingo worker failed to initialize: {ready}")

    def _read_with_timeout(self, timeout_s: float) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("SimLingo worker stdout is unavailable")
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout_s):
                self.close()
                raise TimeoutError(f"SimLingo worker timed out after {timeout_s:.0f}s")
        return read_response(self._process.stdout)

    def infer(
        self,
        image: np.ndarray,
        *,
        speed_m_s: float,
        instruction: str,
        inference_seed: int,
    ) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"SimLingo worker exited with code {self._process.returncode}"
            )
        if self._process.stdin is None:
            raise RuntimeError("SimLingo worker stdin is unavailable")
        self._process.stdin.write(
            encode_request(
                image,
                speed_m_s=speed_m_s,
                instruction=instruction,
                inference_seed=inference_seed,
            )
        )
        self._process.stdin.flush()
        payload = self._read_with_timeout(self._inference_timeout_s)
        if not payload.get("ok"):
            raise RuntimeError(f"SimLingo inference failed: {payload.get('error')}")
        return payload

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _absolute_local_path(value: str | os.PathLike[str], project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


class LanguageConditionedSimLingoModel(BaseTrajectoryModel):  # type: ignore[misc]
    """Official SimLingo model exposed through AlpaSim's trajectory API."""

    NUM_CAMERAS = 1
    EXPECTED_HEIGHT, EXPECTED_WIDTH = _OFFICIAL_FRAME_HW
    MIN_FRAME_HW = _OFFICIAL_FRAME_HW

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: Any,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "LanguageConditionedSimLingoModel":
        project_root = Path(__file__).resolve().parents[2]
        checkpoint = _absolute_local_path(model_cfg.checkpoint_path, project_root)
        code_root = _absolute_local_path(
            os.environ.get(
                "L2D_SIMLINGO_CODE_ROOT",
                "model_zoo/simlingo-official-743b243",
            ),
            project_root,
        )
        python_executable = _absolute_local_path(
            os.environ.get(
                "L2D_SIMLINGO_WORKER_PYTHON",
                str(project_root.parent / ".venvs/simlingo/bin/python"),
            ),
            project_root,
        )
        base_setting = model_cfg.tokenizer_path or os.environ.get(
            "L2D_SIMLINGO_BASE_MODEL",
            str(
                project_root.parent
                / ".cache/huggingface/hub/models--OpenGVLab--InternVL2-1B"
                / "snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187"
            ),
        )
        base_model = _absolute_local_path(base_setting, project_root)
        config = checkpoint.parents[2] / ".hydra/config.yaml"
        required = {
            "official checkpoint": checkpoint,
            "official Hydra config": config,
            "official source checkout": code_root,
            "official base model": base_model,
            "worker Python": python_executable,
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing SimLingo assets: " + "; ".join(missing))
        return cls(
            checkpoint_path=checkpoint,
            config_path=config,
            code_root=code_root,
            base_model_path=base_model,
            python_executable=python_executable,
            device=str(device),
            camera_ids=camera_ids,
            context_length=context_length or _OFFICIAL_CONTEXT_LENGTH,
            output_frequency_hz=output_frequency_hz,
        )

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        config_path: Path,
        code_root: Path,
        base_model_path: Path,
        python_executable: Path,
        device: str,
        camera_ids: list[str],
        context_length: int,
        output_frequency_hz: int,
        worker: Any | None = None,
    ) -> None:
        if camera_ids != [_OFFICIAL_CAMERA_ID]:
            raise ValueError(
                "official SimLingo requires only camera_front_wide_120fov"
            )
        if context_length != _OFFICIAL_CONTEXT_LENGTH:
            raise ValueError("official SimLingo requires context_length=1")
        if output_frequency_hz != _OFFICIAL_OUTPUT_HZ:
            raise ValueError("official SimLingo waypoints require output_frequency_hz=4")
        self._camera_ids = list(camera_ids)
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._worker = worker or _WorkerClient(
            python_executable=python_executable,
            worker_script=Path(__file__).with_name("simlingo_worker.py"),
            code_root=code_root,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            base_model_path=base_model_path,
            device=device,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> int:
        del command
        return 0

    def predict(self, prediction_input: Any) -> Any:
        self._validate_cameras(prediction_input.camera_images)
        frames = prediction_input.camera_images[_OFFICIAL_CAMERA_ID]
        if len(frames) != 1:
            raise ValueError(f"SimLingo expects 1 frame, got {len(frames)}")
        timestamp_us, image = frames[0]
        if torch is not None and isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.shape != (self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH, 3):
            raise ValueError(
                "SimLingo expects rectified RGB frame shape "
                f"{(self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH, 3)}, got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"SimLingo expects uint8 frames, got {image.dtype}")
        instruction = str(
            getattr(prediction_input, "language_instruction", "")
        ).strip()
        if not instruction and os.environ.get("L2D_DIAGNOSTIC_ALLOW_EMPTY") != "1":
            raise ValueError("Language2Drive sidecar instruction must be non-empty")

        result = self._worker.infer(
            image,
            speed_m_s=float(prediction_input.speed),
            instruction=instruction,
            inference_seed=int(prediction_input.inference_seed),
        )
        trajectory_xy = fuse_official_trajectories(
            result.get("route"), result.get("speed_waypoints")
        )
        headings = self._compute_headings_from_trajectory(trajectory_xy)
        prediction = ModelPrediction.from_planar(trajectory_xy, headings)
        language = result.get("language")
        if isinstance(language, list):
            language = language[0] if language else None
        prediction.reasoning_text = None if language is None else str(language)

        # Anchor the image-derived trajectory at the matching local ego pose
        # when the driver history covers the frame timestamp.
        poses = sorted(
            prediction_input.ego_pose_history or [], key=lambda item: item.timestamp_us
        )
        if poses and poses[0].timestamp_us <= timestamp_us <= poses[-1].timestamp_us:
            timestamps = np.asarray([item.timestamp_us for item in poses], dtype=np.uint64)
            geometry_poses = [pose_from_grpc(item.pose) for item in poses]
            pose_t0 = Trajectory.from_poses(timestamps, geometry_poses).interpolate_pose(
                int(timestamp_us)
            )
            prediction.model_t0_us = int(timestamp_us)
            prediction.pose_local_to_rig_t0 = pose_t0
            step_us = 1_000_000 // self.output_frequency_hz
            prediction.waypoint_timestamps_us = int(timestamp_us) + np.arange(
                1, len(trajectory_xy) + 1, dtype=np.uint64
            ) * step_us
        return prediction

    def close(self) -> None:
        self._worker.close()
