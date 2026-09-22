"""Qwen-Drive-1.0-4B native Planning Expert bridge for Language2Drive."""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ROOT = Path(__file__).resolve().parent
# Qwen-Drive needs newer Torch/Transformers than simlingo-py311 provides. Put
# only this model's isolated vendor and pinned source ahead of the untouched env
# before importing the shared agent (which itself imports torch).
ordered_paths = (
    MODEL_ROOT / "vendor",
    MODEL_ROOT / "code/src",
    MODEL_ROOT.parent,
    MODEL_ROOT.parent / "language2drive_adapters/vendor",
)
for item in ordered_paths:
    while str(item) in sys.path:
        sys.path.remove(str(item))
sys.path[:0] = [str(item) for item in ordered_paths]
os.environ.setdefault("TRITON_CACHE_DIR", str(MODEL_ROOT / "cache/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(MODEL_ROOT / "cache/torchinductor"))

from language2drive_adapters import Language2DriveTrajectoryAgent  # noqa: E402


def get_entry_point():
    return "QwenDriveLanguage2DriveAgent"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class QwenDriveLanguage2DriveAgent(Language2DriveTrajectoryAgent):
    model_root = MODEL_ROOT

    def setup(self, path_to_conf_file: str, route_index=None) -> None:
        super().setup(path_to_conf_file, route_index)
        self._qwen_frames: dict[str, deque[np.ndarray]] = {
            name: deque(maxlen=31) for name in ("front", "left", "right")
        }
        self._qwen_motion: deque[tuple[float, float, float, float, float]] = deque(
            maxlen=31
        )
        self._qwen_x = 0.0
        self._qwen_y = 0.0
        self._qwen_yaw = 0.0
        self._qwen_last_speed = 0.0
        self._qwen_last_timestamp: float | None = None

    @staticmethod
    def _sensor_value(value: Any) -> Any:
        return value[1] if isinstance(value, (tuple, list)) and len(value) == 2 else value

    def _append_scene_context(
        self, input_data: dict[str, Any], timestamp: float, speed: float
    ) -> None:
        for name in self._qwen_frames:
            key = f"rgb_{name}"
            if key in input_data:
                self._qwen_frames[name].append(self._rgb(input_data[key]))

        imu = self._sensor_value(input_data.get("imu", np.zeros(7, dtype=np.float32)))
        imu = np.asarray(imu).reshape(-1)
        compass = float(imu[-1]) if len(imu) else self._qwen_yaw
        if not math.isfinite(compass):
            compass = self._qwen_yaw
        now = float(timestamp)
        dt = 0.05 if self._qwen_last_timestamp is None else now - self._qwen_last_timestamp
        if not math.isfinite(dt) or dt <= 0.0 or dt > 0.25:
            dt = 0.05
        acceleration = (speed - self._qwen_last_speed) / dt
        self._qwen_x += speed * math.cos(compass) * dt
        self._qwen_y += speed * math.sin(compass) * dt
        self._qwen_yaw = compass
        self._qwen_motion.append(
            (self._qwen_x, self._qwen_y, compass, float(speed), float(acceleration))
        )
        self._qwen_last_speed = float(speed)
        self._qwen_last_timestamp = now

    @staticmethod
    def _sample(values: list[Any], positions: list[int]) -> list[Any]:
        if not values:
            raise RuntimeError("Qwen-Drive temporal input is empty")
        return [values[max(0, len(values) + position)] for position in positions]

    def _predict(self, input_data: dict[str, Any], speed: float) -> None:
        temporal_images = {
            name: self._sample(list(frames), [-31, -21, -11, -1])
            for name, frames in self._qwen_frames.items()
        }
        motion = self._sample(list(self._qwen_motion), list(range(-31, 0, 2)))
        if len(motion) != 16:
            raise RuntimeError(f"expected 16 Qwen-Drive history states, got {len(motion)}")
        current_x, current_y, current_yaw, _speed, _accel = motion[-1]
        history = np.empty((16, 3), dtype=np.float64)
        history_velocity = np.zeros((16, 2), dtype=np.float32)
        history_acceleration = np.zeros((16, 2), dtype=np.float32)
        cosine, sine = math.cos(current_yaw), math.sin(current_yaw)
        for index, (x, y, yaw, old_speed, acceleration) in enumerate(motion):
            dx, dy = x - current_x, y - current_y
            history[index] = (
                cosine * dx + sine * dy,
                sine * dx - cosine * dy,
                _wrap(current_yaw - yaw),
            )
            history_velocity[index, 0] = old_speed
            history_acceleration[index, 0] = acceleration
        self.backend.set_scene_context(
            temporal_images=temporal_images,
            history=history,
            history_velocity=history_velocity,
            history_acceleration=history_acceleration,
            ego_acceleration=(float(motion[-1][4]), 0.0),
        )
        super()._predict(input_data, speed)

    def run_step(self, input_data, timestamp, sensors=None):
        raw_speed = self._sensor_value(input_data.get("speed", {"speed": 0.0}))
        speed = float(raw_speed.get("speed", 0.0) if isinstance(raw_speed, dict) else raw_speed)
        self._append_scene_context(input_data, timestamp, speed)
        return super().run_step(input_data, timestamp, sensors)
