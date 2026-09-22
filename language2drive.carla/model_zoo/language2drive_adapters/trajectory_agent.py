"""Shared closed-loop bridge for trajectory-producing vision-language models.

This module is deliberately self-contained under ``model_zoo``.  It consumes
only the Language2Drive public context, camera pixels and ego measurements;
the standardized track never receives or reconstructs the evaluator route.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import carla
import numpy as np
import torch
from leaderboard.autoagents import autonomous_agent

if not hasattr(torch.compiler, "is_compiling"):
    # Accessing torch._dynamo here eagerly imports the environment's global
    # Transformers before the selected model-local vendor is on sys.path.
    # These adapters never torch.compile their models, so the compatibility
    # value is deterministically false and avoids cross-version module mixing.
    torch.compiler.is_compiling = lambda: False


MODEL_ZOO = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_ZOO.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from team_code.language2drive_context import (  # noqa: E402
    L2D_TRACK_STANDARDIZED,
    PublicContextState,
    normalize_l2d_track,
)


def _path(value: str, base: Path) -> Path:
    result = Path(value).expanduser()
    return result.resolve() if result.is_absolute() else (base / result).resolve()


def _check_assets(config: dict[str, Any], base: Path) -> None:
    missing: list[str] = []
    for raw in config.get("required_files", []):
        target = _path(str(raw), base)
        if not target.is_file():
            missing.append(str(target))
    for raw in config.get("required_dirs", []):
        target = _path(str(raw), base)
        if not target.is_dir() or not any(target.iterdir()):
            missing.append(str(target))
        else:
            for index_name in (
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ):
                index_path = target / index_name
                if not index_path.is_file():
                    continue
                index = json.loads(index_path.read_text(encoding="utf-8"))
                for shard in sorted(set(index.get("weight_map", {}).values())):
                    if not (target / shard).is_file():
                        missing.append(str(target / shard))
    if missing:
        lines = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"{config.get('model_name', 'model')} assets are incomplete:\n  - {lines}\n"
            f"Run: {base / 'download.sh'}"
        )


def _camera(spec: dict[str, Any], sensor_id: str, yaw: float) -> dict[str, Any]:
    return {
        "type": "sensor.camera.rgb",
        "x": float(spec.get("x", 1.3)),
        "y": float(spec.get("y", 0.0)),
        "z": float(spec.get("z", 2.3)),
        "roll": 0.0,
        "pitch": float(spec.get("pitch", 0.0)),
        "yaw": float(yaw),
        "width": int(spec.get("width", 896)),
        "height": int(spec.get("height", 512)),
        "fov": float(spec.get("fov", 100.0)),
        "id": sensor_id,
    }


class Language2DriveTrajectoryAgent(autonomous_agent.AutonomousAgent):
    """Base agent: model trajectory -> guarded pure-pursuit CARLA control."""

    model_root: Path | None = None

    def set_language2drive_context(self, public_context: dict[str, Any]) -> dict[str, Any]:
        state = getattr(self, "_l2d_context_state", None)
        if state is None:
            state = PublicContextState()
            self._l2d_context_state = state
        context = state.bind(public_context)
        self.evaluation_track = context.track
        self.route_instruction = context.instruction
        self.instruction_version = context.instruction_version
        if context.track == L2D_TRACK_STANDARDIZED:
            self._global_plan = None
            self._global_plan_world_coord = None
        return context.to_dict()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        if getattr(self, "evaluation_track", None) == L2D_TRACK_STANDARDIZED:
            raise RuntimeError(
                "set_global_plan() is forbidden on Language2Drive Standardized Track"
            )
        return super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def setup(self, path_to_conf_file: str, route_index=None) -> None:  # noqa: ARG002
        if not torch.cuda.is_available():
            raise RuntimeError("This model adapter requires an NVIDIA CUDA GPU")
        config_arg = str(path_to_conf_file).split("+", 1)[0]
        config_path = Path(config_arg).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"adapter config not found: {config_path}")
        self.config_path = config_path
        self.cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if self.cfg.get("schema_version") != "language2drive.model_adapter.v1":
            raise ValueError("unsupported adapter schema_version")
        _check_assets(self.cfg, config_path.parent)

        context = getattr(getattr(self, "_l2d_context_state", None), "context", None)
        self.evaluation_track = (
            context.track if context is not None else normalize_l2d_track(os.getenv("L2D_TRACK"))
        )
        self.route_instruction = context.instruction if context is not None else ""
        if self.evaluation_track == L2D_TRACK_STANDARDIZED:
            if context is None:
                raise RuntimeError("standardized track requires set_language2drive_context()")
            if not self.route_instruction.strip():
                raise RuntimeError("standardized instruction is empty")

        self.track = autonomous_agent.Track.SENSORS
        self.device = torch.device("cuda")
        self.step = -1
        self.last_inference_step = -10**9
        self.last_plan_step = -10**9
        self.last_control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.controller_source = "model_trajectory:setup"
        self.waypoints: list[tuple[float, float]] = []
        self.last_model_text = ""
        self.last_error: str | None = None
        self.speed_history: deque[float] = deque(maxlen=80)
        self.runtime = self.cfg.get("runtime", {})
        self.control_cfg = self.cfg.get("control", {})
        self.inference_interval = max(1, int(self.runtime.get("inference_interval_frames", 20)))
        self.hold_frames = max(self.inference_interval, int(self.runtime.get("plan_hold_frames", 60)))
        self.max_consecutive_errors = max(1, int(self.runtime.get("max_consecutive_errors", 3)))
        self.consecutive_errors = 0
        self.record_dir = Path(os.environ["L2D_RECORD_DIR"]).resolve() if os.getenv("L2D_RECORD_DIR") else None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(config_path.parent / "hf_cache"))
        shared_vendor = MODEL_ZOO / "language2drive_adapters" / "vendor"
        vendor = config_path.parent / "vendor"
        source = _path(str(self.cfg.get("source_dir", "code")), config_path.parent)
        extra_sources = [
            _path(str(value), config_path.parent)
            for value in self.cfg.get("extra_source_dirs", [])
        ]
        for item in (shared_vendor, vendor, source, *extra_sources):
            if item.is_dir() and str(item) not in sys.path:
                sys.path.insert(0, str(item))

        from .vlm_backends import create_backend

        self.backend = create_backend(self.cfg, config_path.parent, self.device)
        torch.manual_seed(int(self.runtime.get("sampling_seed", 0)))
        torch.cuda.manual_seed_all(int(self.runtime.get("sampling_seed", 0)))

    def sensors(self) -> list[dict[str, Any]]:
        cfg = self.cfg.get("camera", {}) if hasattr(self, "cfg") else {}
        views = cfg.get("views", ["front", "left", "right"])
        yaw = {
            "front": 0.0,
            "left": -60.0,
            "right": 60.0,
            "back_left": -120.0,
            "back_right": 120.0,
            "rear": 180.0,
        }
        result = [_camera(cfg, f"rgb_{name}", yaw[name]) for name in views]
        result.extend(
            [
                {
                    "type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                    "sensor_tick": 0.05, "id": "imu",
                },
                {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
            ]
        )
        return result

    @staticmethod
    def _rgb(value: Any) -> np.ndarray:
        raw = value[1] if isinstance(value, (tuple, list)) and len(value) == 2 else value
        if not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[2] < 3:
            raise ValueError("camera sensor did not return an HxWx3/4 array")
        return np.ascontiguousarray(raw[:, :, :3][:, :, ::-1])

    def _history_xy(self) -> list[tuple[float, float]]:
        # Ego-only dead reckoning, used as motion context.  No route/target data.
        speeds = list(self.speed_history)[-30:]
        distance = 0.0
        points: list[tuple[float, float]] = []
        for speed in reversed(speeds[::5] or [0.0]):
            points.append((-distance, 0.0))
            distance += max(0.0, float(speed)) * 0.25
        return list(reversed(points))[-6:]

    def _predict(self, input_data: dict[str, Any], speed: float) -> None:
        images = {
            key.removeprefix("rgb_"): self._rgb(value)
            for key, value in input_data.items()
            if key.startswith("rgb_")
        }
        started = time.monotonic()
        result = self.backend.predict(
            images=images,
            instruction=self.route_instruction,
            speed=speed,
            history_xy=self._history_xy(),
        )
        points = result.waypoints
        self.last_model_text = result.text
        if len(points) < 2:
            raise ValueError(
                "model returned fewer than two waypoint pairs; output="
                + repr(result.text[:500])
            )
        checked: list[tuple[float, float]] = []
        max_range = float(self.runtime.get("max_waypoint_range_m", 80.0))
        for x, y in points[: int(self.runtime.get("max_waypoints", 12))]:
            x, y = float(x), float(y)
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if math.hypot(x, y) > max_range:
                continue
            checked.append((x, y))
        if len(checked) < 2:
            raise ValueError(
                f"model returned fewer than two waypoints inside the {max_range:g} m "
                "safety guard; output=" + repr(result.text[:500])
            )
        self.waypoints = checked
        self.last_plan_step = self.step
        self.consecutive_errors = 0
        self.last_error = None
        if self.record_dir is not None:
            payload = {
                "step": self.step,
                "latency_s": time.monotonic() - started,
                "instruction": self.route_instruction,
                "waypoints": checked,
                "model_text": result.text,
            }
            with (self.record_dir / "model_plans.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _control(self, speed: float) -> carla.VehicleControl:
        if len(self.waypoints) < 2 or self.step - self.last_plan_step > self.hold_frames:
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        forward = [(x, y) for x, y in self.waypoints if x > 0.15]
        if not forward:
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        lookahead = float(self.control_cfg.get("lookahead_m", 3.5)) + speed * float(
            self.control_cfg.get("lookahead_speed_gain", 0.35)
        )
        aim = min(forward, key=lambda p: abs(math.hypot(*p) - lookahead))
        steer = math.atan2(2.0 * float(self.control_cfg.get("wheelbase_m", 2.8)) * aim[1], max(0.25, aim[0] ** 2 + aim[1] ** 2))
        steer /= float(self.control_cfg.get("max_steer_angle_rad", 0.7))
        steer = float(np.clip(steer, -1.0, 1.0))

        dt = float(self.runtime.get("waypoint_dt_s", 0.5))
        segment_speeds = [
            math.dist(forward[i - 1], forward[i]) / dt for i in range(1, min(len(forward), 5))
        ]
        target = min(float(self.control_cfg.get("max_target_speed_mps", 8.0)), float(np.median(segment_speeds)) if segment_speeds else 0.0)
        error = target - speed
        brake = float(np.clip(-error * float(self.control_cfg.get("brake_kp", 0.35)), 0.0, 1.0))
        throttle = 0.0 if brake > 0.05 else float(np.clip(error * float(self.control_cfg.get("throttle_kp", 0.3)), 0.0, float(self.control_cfg.get("max_throttle", 0.7))))
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):  # noqa: ARG002
        self.step += 1
        raw_speed = input_data.get("speed", (None, {"speed": 0.0}))[1]
        speed = float(raw_speed.get("speed", 0.0) if isinstance(raw_speed, dict) else raw_speed)
        self.speed_history.append(speed)
        if (
            self.step == 0
            or self.step - self.last_inference_step >= self.inference_interval
        ):
            self.last_inference_step = self.step
            try:
                self._predict(input_data, speed)
            except Exception as exc:  # fail-safe braking; repeated failures are surfaced
                self.consecutive_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.waypoints = []
                if self.consecutive_errors >= self.max_consecutive_errors:
                    raise RuntimeError(
                        f"{self.cfg.get('model_name')} inference failed "
                        f"{self.consecutive_errors} consecutive times: {self.last_error}"
                    ) from exc
        self.last_control = self._control(speed)
        self.controller_source = (
            "model_trajectory:pure_pursuit"
            if len(self.waypoints) >= 2
            else "model_trajectory:inference_fail_safe"
        )
        return self.last_control

    def destroy(self, results=None):  # noqa: ARG002
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.close()
        self.backend = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
