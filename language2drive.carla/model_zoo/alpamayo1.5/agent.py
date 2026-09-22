"""Language2Drive adapter for NVIDIA Alpamayo 1.5.

The adapter intentionally lives with the model bundle so Language2Drive can
load it through ``--agent-path`` without changing benchmark code or the
``simlingo-py311`` conda environment.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import zipfile
from collections import deque
from pathlib import Path
from typing import Any


MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODEL_ROOT.parents[1]
VENDOR_ROOT = MODEL_ROOT / "vendor"
SOURCE_ROOT = MODEL_ROOT / "code" / "src"

# The base conda environment remains untouched.  Newer Transformers/tokenizers
# are vendored with this model and take precedence only in the evaluator
# process that imports this agent.
for _path in (VENDOR_ROOT, SOURCE_ROOT, REPO_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf_cache"))

import carla  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from leaderboard.autoagents import autonomous_agent  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

# Transformers 4.57's fast image utilities probe this public helper, which was
# added after Torch 2.2.  The equivalent Torch 2.2 implementation already
# exists under torch._dynamo.
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = getattr(torch._dynamo, "is_compiling", lambda: False)

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.config import Alpamayo1_5Config  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from team_code.language2drive_context import (  # noqa: E402
    L2D_TRACK_STANDARDIZED,
    PublicContextState,
    normalize_l2d_track,
)


def get_entry_point() -> str:
    """Return the class name expected by the CARLA leaderboard loader."""

    return "Alpamayo1_5Agent"


def _resolve_model_path(raw: str, *, config_path: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Adapter paths are model-bundle relative, not evaluator-CWD relative.
    return (config_path.parent / path).resolve()


def _bgrx_to_chw_rgb(array: np.ndarray) -> torch.Tensor:
    """Convert CARLA BGRA uint8 to the model's CHW RGB uint8 tensor."""

    rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _angle_delta_degrees(value: float, reference: float) -> float:
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


class Alpamayo1_5Agent(autonomous_agent.AutonomousAgent):
    """Closed-loop CARLA agent backed by Alpamayo 1.5 trajectories."""

    def set_language2drive_context(self, public_context: dict[str, Any]) -> dict[str, Any]:
        state = getattr(self, "_l2d_context_state", None)
        if state is None:
            state = PublicContextState()
            self._l2d_context_state = state
        context = state.bind(public_context)
        self.evaluation_track = context.track
        self.strict_vln = context.track == L2D_TRACK_STANDARDIZED
        self.route_instruction = context.instruction
        self.instruction_version = context.instruction_version
        if self.strict_vln:
            # Never retain an answer-bearing route if an evaluator bound the
            # public context after constructing the agent.
            self._global_plan = None
            self._global_plan_world_coord = None
            self.org_dense_route_world_coord = None
            self._plan_gps_HACK = None
        return context.to_dict()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        if getattr(self, "evaluation_track", None) == L2D_TRACK_STANDARDIZED:
            raise RuntimeError(
                "set_global_plan() is forbidden on Language2Drive Standardized Track"
            )
        return super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def setup(self, path_to_conf_file: str, route_index=None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Alpamayo 1.5 requires an NVIDIA CUDA GPU")

        config_arg = str(path_to_conf_file).rsplit("+", 1)[0]
        config_path = Path(config_arg).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Alpamayo adapter config not found: {config_path}")
        self.adapter_config_path = config_path
        self.adapter_cfg = json.loads(config_path.read_text(encoding="utf-8"))

        bound_context = getattr(
            getattr(self, "_l2d_context_state", None), "context", None
        )
        if bound_context is not None:
            self.evaluation_track = bound_context.track
            self.route_instruction = bound_context.instruction
        else:
            self.evaluation_track = normalize_l2d_track(os.environ.get("L2D_TRACK"))
            self.route_instruction = ""
        self.strict_vln = self.evaluation_track == L2D_TRACK_STANDARDIZED
        if self.strict_vln and bound_context is None:
            raise RuntimeError(
                "Language2Drive Standardized Track requires "
                "set_language2drive_context(public_context)"
            )
        if self.strict_vln and not self.route_instruction:
            raise RuntimeError("Language2Drive Standardized instruction is empty")

        self.track = autonomous_agent.Track.SENSORS
        self.device = torch.device("cuda")
        self.step = -1
        self.initialized = False
        self.last_inference_time: float | None = None
        self.latest_plan_world: list[tuple[float, float, float]] = []
        self.latest_reasoning = ""
        self.plan_progress_idx = 0
        self.strict_launch_frames = 0
        self.model_target_speed_mps = 0.0
        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.controller_source = "setup"

        runtime = self.adapter_cfg["runtime"]
        self.history_steps = int(runtime["history_steps"])
        self.history_dt = float(runtime["history_dt"])
        self.camera_frames = int(runtime["camera_frames"])
        self.replan_interval = float(runtime["replan_interval"])
        self.max_generation_length = int(runtime["max_generation_length"])
        self.diffusion_steps = int(runtime["diffusion_steps"])
        navigation_cfg = runtime.get("navigation_cfg", {})
        if not isinstance(navigation_cfg, dict):
            raise ValueError("runtime.navigation_cfg must be an object")
        self.navigation_cfg_enabled = bool(navigation_cfg.get("enabled", False))
        self.navigation_cfg_weight = float(
            navigation_cfg.get("guidance_weight", 1.5)
        )
        self.navigation_cfg_temperature = float(
            navigation_cfg.get("temperature", 0.6)
        )
        if not math.isfinite(self.navigation_cfg_weight) or self.navigation_cfg_weight < 0.0:
            raise ValueError(
                "runtime.navigation_cfg.guidance_weight must be finite and non-negative"
            )
        if (
            not math.isfinite(self.navigation_cfg_temperature)
            or self.navigation_cfg_temperature <= 0.0
        ):
            raise ValueError(
                "runtime.navigation_cfg.temperature must be finite and positive"
            )
        front_capture = runtime.get("front_frame_capture", {})
        if not isinstance(front_capture, dict):
            raise ValueError("runtime.front_frame_capture must be an object")
        self.front_capture_enabled = bool(front_capture.get("enabled", False))
        self.front_capture_interval = float(front_capture.get("interval_seconds", 2.0))
        self.front_capture_quality = int(front_capture.get("jpeg_quality", 85))
        self.front_capture_max_frames = int(front_capture.get("max_frames", 24))
        if not math.isfinite(self.front_capture_interval) or self.front_capture_interval <= 0:
            raise ValueError(
                "runtime.front_frame_capture.interval_seconds must be finite and positive"
            )
        if not 1 <= self.front_capture_quality <= 100:
            raise ValueError("runtime.front_frame_capture.jpeg_quality must be in [1, 100]")
        if self.front_capture_max_frames < 1:
            raise ValueError("runtime.front_frame_capture.max_frames must be positive")
        self.front_capture_count = 0
        self.front_capture_last_time: float | None = None
        self.front_capture_error: str | None = None
        self.history_buffer: deque[dict[str, Any]] = deque(
            maxlen=max(128, int(self.history_steps / self.history_dt))
        )

        checkpoint_dir = _resolve_model_path(
            self.adapter_cfg["checkpoint_dir"], config_path=config_path
        )
        processor_dir = _resolve_model_path(
            self.adapter_cfg["processor_dir"], config_path=config_path
        )
        if not (checkpoint_dir / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"Alpamayo checkpoint is incomplete: {checkpoint_dir}")
        if not (processor_dir / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Alpamayo processor assets are incomplete: {processor_dir}")

        model_config_dict = json.loads(
            (checkpoint_dir / "config.json").read_text(encoding="utf-8")
        )
        attention = str(runtime.get("attention_implementation", "flash_attention_2"))
        model_config_dict["vlm_name_or_path"] = str(processor_dir)
        model_config_dict["attn_implementation"] = attention
        model_config_dict["model_dtype"] = "bfloat16"
        model_config = Alpamayo1_5Config(**model_config_dict)

        self.model = Alpamayo1_5.from_pretrained(
            checkpoint_dir,
            config=model_config,
            dtype=torch.bfloat16,
            attn_implementation=attention,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device=self.device, dtype=torch.bfloat16).eval()
        self.processor = AutoProcessor.from_pretrained(
            processor_dir,
            min_pixels=model_config.min_pixels,
            max_pixels=model_config.max_pixels,
            local_files_only=True,
            use_fast=False,
        )
        # The checkpoint model builds the exact extended 155697-token
        # Alpamayo tokenizer.  Reuse it in the visual processor.
        self.processor.tokenizer = self.model.tokenizer

        # Fixed agent-side sampling keeps Standardized inference repeatable
        # without consuming the evaluator-owned route seed.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

        self.record_dir = (
            Path(os.environ["L2D_RECORD_DIR"]).resolve()
            if os.environ.get("L2D_RECORD_DIR")
            else None
        )
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            load_report = {
                "schema_version": "language2drive.alpamayo1_5_load.v1",
                "checkpoint_dir": str(checkpoint_dir),
                "processor_dir": str(processor_dir),
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "dtype": str(next(self.model.parameters()).dtype),
                "attention_implementation": attention,
                "cuda_device": torch.cuda.get_device_name(self.device),
                "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 4),
                "navigation_cfg": {
                    "enabled": self.navigation_cfg_enabled,
                    "guidance_weight": self.navigation_cfg_weight,
                    "temperature": self.navigation_cfg_temperature,
                },
                "front_frame_capture": {
                    "enabled": self.front_capture_enabled,
                    "interval_seconds": self.front_capture_interval,
                    "jpeg_quality": self.front_capture_quality,
                    "max_frames": self.front_capture_max_frames,
                },
            }
            (self.record_dir / "alpamayo_load_report.json").write_text(
                json.dumps(load_report, indent=2) + "\n", encoding="utf-8"
            )

        print(
            "[Alpamayo1.5] loaded {} parameters on {} ({:.2f} GiB), instruction={!r}".format(
                sum(p.numel() for p in self.model.parameters()),
                torch.cuda.get_device_name(self.device),
                torch.cuda.memory_allocated() / 1024**3,
                self.route_instruction,
            ),
            flush=True,
        )

    def sensors(self) -> list[dict[str, Any]]:
        camera = self.adapter_cfg["camera"]
        common = {
            "type": "sensor.camera.rgb",
            "x": float(camera["x"]),
            "z": float(camera["z"]),
            "roll": 0.0,
            "pitch": float(camera.get("pitch", 0.0)),
            "width": int(camera["width"]),
            "height": int(camera["height"]),
        }
        return [
            {
                **common,
                "id": "CAM_CROSS_LEFT",
                "y": -float(camera["lateral_offset"]),
                "yaw": -float(camera["cross_yaw"]),
                "fov": float(camera["wide_fov"]),
            },
            {
                **common,
                "id": "CAM_FRONT_WIDE",
                "y": 0.0,
                "yaw": 0.0,
                "fov": float(camera["wide_fov"]),
            },
            {
                **common,
                "id": "CAM_CROSS_RIGHT",
                "y": float(camera["lateral_offset"]),
                "yaw": float(camera["cross_yaw"]),
                "fov": float(camera["wide_fov"]),
            },
            {
                **common,
                "id": "CAM_FRONT_TELE",
                "y": 0.0,
                "yaw": 0.0,
                "fov": float(camera["tele_fov"]),
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "SPEED",
            },
        ]

    def _snapshot(self, input_data, timestamp: float) -> dict[str, Any] | None:
        if self.hero_actor is None:
            self.get_hero()
        if self.hero_actor is None:
            return None
        transform = self.hero_actor.get_transform()
        return {
            "timestamp": float(timestamp),
            "images": {
                0: _bgrx_to_chw_rgb(input_data["CAM_CROSS_LEFT"][1]),
                1: _bgrx_to_chw_rgb(input_data["CAM_FRONT_WIDE"][1]),
                2: _bgrx_to_chw_rgb(input_data["CAM_CROSS_RIGHT"][1]),
                6: _bgrx_to_chw_rgb(input_data["CAM_FRONT_TELE"][1]),
            },
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw": float(transform.rotation.yaw),
            "speed": float(input_data["SPEED"][1]["speed"]),
        }

    def _capture_front_frame(self, input_data, timestamp: float) -> None:
        """Persist sparse public front-camera evidence as a bounded ZIP archive."""

        if (
            not self.front_capture_enabled
            or self.record_dir is None
            or self.front_capture_count >= self.front_capture_max_frames
            or (
                self.front_capture_last_time is not None
                and timestamp - self.front_capture_last_time
                < self.front_capture_interval - 1e-3
            )
        ):
            return
        try:
            bgra = input_data["CAM_FRONT_WIDE"][1]
            rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
            encoded = io.BytesIO()
            Image.fromarray(rgb, mode="RGB").save(
                encoded,
                format="JPEG",
                quality=self.front_capture_quality,
            )
            archive_path = self.record_dir / "front_frames.zip"
            with zipfile.ZipFile(
                archive_path, "a", compression=zipfile.ZIP_STORED
            ) as archive:
                if self.front_capture_count == 0:
                    archive.writestr(
                        "manifest.json",
                        json.dumps(
                            {
                                "schema_version": "language2drive.front_frames.v1",
                                "camera": "CAM_FRONT_WIDE",
                                "instruction": self.route_instruction,
                                "interval_seconds": self.front_capture_interval,
                                "jpeg_quality": self.front_capture_quality,
                                "max_frames": self.front_capture_max_frames,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                    )
                archive.writestr(
                    "front_wide/{:04d}_t{:08.2f}.jpg".format(
                        self.front_capture_count, timestamp
                    ),
                    encoded.getvalue(),
                )
            self.front_capture_count += 1
            self.front_capture_last_time = timestamp
        except Exception as error:  # Diagnostic capture must not alter driving.
            if self.front_capture_error is None:
                self.front_capture_error = "{}: {}".format(type(error).__name__, error)
                print(
                    "[Alpamayo1.5] front frame capture disabled: {}".format(
                        self.front_capture_error
                    ),
                    flush=True,
                )
            self.front_capture_enabled = False

    def _sample_at(self, target_time: float) -> dict[str, Any]:
        values = list(self.history_buffer)
        chosen = values[0]
        for candidate in reversed(values):
            if candidate["timestamp"] <= target_time + 1e-3:
                chosen = candidate
                break
        return chosen

    def _model_inputs(self, timestamp: float) -> dict[str, Any]:
        camera_samples = [
            self._sample_at(timestamp - (self.camera_frames - 1 - i) * 0.1)
            for i in range(self.camera_frames)
        ]
        frames = torch.stack(
            [
                sample["images"][camera_id]
                for camera_id in (0, 1, 2, 6)
                for sample in camera_samples
            ],
            dim=0,
        )
        messages = helper.create_message(
            frames=frames,
            camera_indices=torch.tensor([0, 1, 2, 6], dtype=torch.int64),
            num_frames_per_camera=self.camera_frames,
            nav_text=self.route_instruction,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        history = [
            self._sample_at(timestamp - (self.history_steps - 1 - i) * self.history_dt)
            for i in range(self.history_steps)
        ]
        current = history[-1]
        yaw = math.radians(current["yaw"])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        xyz = np.zeros((self.history_steps, 3), dtype=np.float32)
        rotations = np.zeros((self.history_steps, 3, 3), dtype=np.float32)
        for index, sample in enumerate(history):
            dx = sample["x"] - current["x"]
            dy = sample["y"] - current["y"]
            # CARLA is forward-right-up; Alpamayo uses forward-left-up.
            xyz[index, 0] = cos_yaw * dx + sin_yaw * dy
            xyz[index, 1] = sin_yaw * dx - cos_yaw * dy
            xyz[index, 2] = sample["z"] - current["z"]
            model_yaw = math.radians(-_angle_delta_degrees(sample["yaw"], current["yaw"]))
            c, s = math.cos(model_yaw), math.sin(model_yaw)
            rotations[index] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

        payload = {
            "tokenized_data": tokenized,
            "ego_history_xyz": torch.from_numpy(xyz).unsqueeze(0).unsqueeze(0),
            "ego_history_rot": torch.from_numpy(rotations).unsqueeze(0).unsqueeze(0),
        }
        return helper.to_device(payload, self.device)

    def _refresh_plan(self, timestamp: float) -> None:
        model_inputs = self._model_inputs(timestamp)
        # Alpamayo's public rollout method hard-codes do_sample=True.  Reset a
        # fixed agent-side RNG before every receding-horizon inference so
        # repeated replans do not alternate unpredictably between moving and
        # near-zero trajectories.  This constant is independent of the
        # evaluator-owned route seed.
        sampling_seed = int(self.adapter_cfg["runtime"].get("sampling_seed", 0))
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        inference_method = self.model.sample_trajectories_from_data_with_vlm_rollout
        diffusion_kwargs = {"inference_step": self.diffusion_steps}
        if self.navigation_cfg_enabled:
            inference_method = (
                self.model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav
            )
            diffusion_kwargs.update(
                {
                    "use_classifier_free_guidance": True,
                    "inference_guidance_weight": self.navigation_cfg_weight,
                    "temperature": self.navigation_cfg_temperature,
                }
            )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _pred_rot, extra = inference_method(
                data=model_inputs,
                top_p=float(self.adapter_cfg["runtime"]["top_p"]),
                temperature=float(self.adapter_cfg["runtime"]["temperature"]),
                num_traj_samples=1,
                max_generation_length=self.max_generation_length,
                diffusion_kwargs=diffusion_kwargs,
                return_extra=True,
            )
        trajectory = pred_xyz[0, 0, 0].float().cpu().numpy()
        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            raise RuntimeError(f"Alpamayo returned invalid trajectory shape {trajectory.shape}")

        transform = self.hero_actor.get_transform()
        origin = transform.location
        yaw = math.radians(transform.rotation.yaw)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        self.latest_plan_world = []
        for point in trajectory:
            local_x = float(point[0])
            local_y_right = -float(point[1])
            world_x = origin.x + local_x * cos_yaw - local_y_right * sin_yaw
            world_y = origin.y + local_x * sin_yaw + local_y_right * cos_yaw
            self.latest_plan_world.append((world_x, world_y, float(point[2])))
        self.plan_progress_idx = 0
        self.last_inference_time = float(timestamp)
        cot = extra.get("cot")
        self.latest_reasoning = str(cot[0, 0, 0]) if cot is not None else ""

        step_speeds = np.linalg.norm(
            np.diff(
                np.concatenate(
                    [np.zeros((1, 2), dtype=np.float32), trajectory[:, :2]],
                    axis=0,
                ),
                axis=0,
            ),
            axis=1,
        ) / 0.1
        hist_xyz = model_inputs["ego_history_xyz"][:, -1]
        hist_rot = model_inputs["ego_history_rot"][:, -1]
        estimated_v0 = float(
            self.model.action_space.estimate_t0_states(hist_xyz, hist_rot)["v"]
            .float()
            .cpu()
            .reshape(-1)[0]
        )

        if self.record_dir is not None:
            record = {
                "timestamp": float(timestamp),
                "instruction": self.route_instruction,
                "navigation_cfg_enabled": self.navigation_cfg_enabled,
                "navigation_cfg_weight": self.navigation_cfg_weight,
                "navigation_cfg_temperature": self.navigation_cfg_temperature,
                "reasoning": self.latest_reasoning,
                "estimated_history_v0_mps": estimated_v0,
                "prefix_speed_mps": float(step_speeds[:5].mean()) if len(step_speeds) else 0.0,
                "tail_speed_mps": float(step_speeds[-32:].mean()) if len(step_speeds) else 0.0,
                "final_displacement_m": float(np.linalg.norm(trajectory[-1, :2])),
                "trajectory_model_xyz": trajectory.tolist(),
            }
            with (self.record_dir / "agent_plans.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _trajectory_total_distance(self) -> float:
        if not self.latest_plan_world:
            return 0.0
        transform = self.hero_actor.get_transform()
        points = [(transform.location.x, transform.location.y)] + [
            (point[0], point[1]) for point in self.latest_plan_world
        ]
        return sum(
            math.hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(points, points[1:])
        )

    def _control_from_plan(self, current_speed: float) -> carla.VehicleControl:
        control_cfg = self.adapter_cfg["control"]
        control = carla.VehicleControl()
        if not self.latest_plan_world:
            self.model_target_speed_mps = 0.0
            self.controller_source = "safe_stop_no_model_plan"
            control.brake = 1.0
            return control

        transform = self.hero_actor.get_transform()
        location = transform.location
        yaw = math.radians(transform.rotation.yaw)
        search_start = max(0, self.plan_progress_idx - 2)
        nearest = min(
            range(search_start, len(self.latest_plan_world)),
            key=lambda i: math.hypot(
                self.latest_plan_world[i][0] - location.x,
                self.latest_plan_world[i][1] - location.y,
            ),
        )
        self.plan_progress_idx = max(self.plan_progress_idx, nearest)

        lookahead = max(
            float(control_cfg["lookahead_base_m"]),
            float(control_cfg["lookahead_base_m"])
            + current_speed * float(control_cfg["lookahead_speed_gain"]),
        )
        target_index = self.plan_progress_idx
        covered = 0.0
        for index in range(self.plan_progress_idx, len(self.latest_plan_world) - 1):
            a, b = self.latest_plan_world[index], self.latest_plan_world[index + 1]
            covered += math.hypot(b[0] - a[0], b[1] - a[1])
            target_index = index + 1
            if covered >= lookahead:
                break
        target = self.latest_plan_world[target_index]
        dx, dy = target[0] - location.x, target[1] - location.y
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y_right = -dx * math.sin(yaw) + dy * math.cos(yaw)
        denominator = max(local_x * local_x + local_y_right * local_y_right, 1e-3)
        steer_angle = math.atan2(
            2.0 * float(control_cfg["wheelbase_m"]) * local_y_right,
            denominator,
        )
        control.steer = float(
            np.clip(
                steer_angle / float(control_cfg["max_steer_angle_rad"]),
                -1.0,
                1.0,
            )
        )

        # Alpamayo emits points at 10 Hz.  The unicycle decoder integrates
        # from history v0, so a standing-start plan stays near zero for the
        # first seconds even when later waypoints accelerate.  Tracking only
        # the 0.5 s prefix therefore treats a "go later" plan as a stop and
        # freezes the history.  When the CoT/instruction is not a hold or
        # hazard, follow the faster of the prefix and the trajectory tail.
        points = self.latest_plan_world
        deltas = [
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points[:-1], points[1:])
        ]
        speeds = [delta / 0.1 for delta in deltas]
        prefix_n = min(5, len(speeds))
        tail_n = min(
            max(1, int(float(control_cfg.get("speed_tail_seconds", 3.2)) / 0.1)),
            len(speeds),
        )
        prefix_speed = sum(speeds[:prefix_n]) / prefix_n if prefix_n else 0.0
        tail_speed = sum(speeds[-tail_n:]) / tail_n if tail_n else 0.0
        target_speed = (
            max(prefix_speed, tail_speed)
            if self._straight_launch_allowed()
            else prefix_speed
        )
        target_speed = min(target_speed, float(control_cfg["max_target_speed_mps"]))
        self.model_target_speed_mps = target_speed

        if (
            self._trajectory_total_distance() < float(control_cfg["stop_plan_distance_m"])
            or target_speed < float(control_cfg["stop_speed_mps"])
        ):
            self.controller_source = "alpamayo_model_stop"
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = min(1.0, 0.55 + 0.15 * current_speed)
        else:
            self.controller_source = "alpamayo_trajectory_tracker"
            error = target_speed - current_speed
            if error >= 0.0:
                control.throttle = float(
                    np.clip(
                        float(control_cfg["throttle_kp"]) * error,
                        0.0,
                        float(control_cfg["max_throttle"]),
                    )
                )
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(
                    np.clip(float(control_cfg["brake_kp"]) * -error, 0.0, 1.0)
                )
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def _straight_launch_allowed(self) -> bool:
        """Allow a bounded cold start without overriding stop/hazard semantics."""

        instruction = self.route_instruction.casefold()
        reasoning = self.latest_reasoning.casefold()
        hold_phrases = (
            "wait here",
            "remain stopped",
            "hold position",
            "stay stopped",
            "do not move",
            "don't move",
        )
        hazard_phrases = (
            "red light",
            "red traffic light",
            "stop sign",
            "pedestrian",
            "yield",
            "blocked",
            "obstacle",
            "collision",
        )
        return not any(phrase in instruction for phrase in hold_phrases) and not any(
            phrase in reasoning for phrase in hazard_phrases
        )

    def run_step(self, input_data, timestamp, sensors=None):  # noqa: ARG002
        self.step += 1
        snapshot = self._snapshot(input_data, float(timestamp))
        if snapshot is None:
            self.controller_source = "safe_stop_no_hero"
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self._capture_front_frame(input_data, float(timestamp))
        self.history_buffer.append(snapshot)

        if not self.initialized:
            self.initialized = True
            self.controller_source = "spawn_initialization_stop"
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            return self.control

        if self.last_inference_time is None or (
            float(timestamp) - self.last_inference_time >= self.replan_interval - 1e-3
        ):
            self._refresh_plan(float(timestamp))

        current_speed = float(input_data["SPEED"][1]["speed"])
        control = self._control_from_plan(current_speed)
        # Alpamayo's action space conditions on the velocity reconstructed from
        # ego history.  At a CARLA spawn that history is static, so the first
        # predicted half-second can be nearly motionless even when the full
        # trajectory eventually accelerates.  Warm the model into its training
        # distribution with a bounded straight launch from the common approach.
        # This never consults the hidden/dense route and never overrides an
        # explicit hold instruction or a model-observed stop hazard.
        control_cfg = self.adapter_cfg["control"]
        if (
            current_speed < float(control_cfg["launch_until_speed_mps"])
            and self.model_target_speed_mps
            < float(control_cfg["launch_plan_speed_threshold_mps"])
            and self.strict_launch_frames
            < int(control_cfg["straight_launch_frames"])
            and self._straight_launch_allowed()
        ):
            self.controller_source = "standardized_straight_launch"
            control.steer = 0.0
            control.throttle = float(control_cfg["launch_throttle"])
            control.brake = 0.0
            self.strict_launch_frames += 1
        self.control = control
        return control

    def destroy(self, results=None) -> None:  # noqa: ARG002
        if self.record_dir is not None:
            archive_path = self.record_dir / "front_frames.zip"
            if archive_path.is_file():
                with zipfile.ZipFile(
                    archive_path, "a", compression=zipfile.ZIP_STORED
                ) as archive:
                    archive.writestr(
                        "summary.json",
                        json.dumps(
                            {
                                "schema_version": "language2drive.front_frames_summary.v1",
                                "saved_frames": self.front_capture_count,
                                "capture_error": self.front_capture_error,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                    )
            summary = {
                "schema_version": "language2drive.alpamayo1_5_summary.v1",
                "instruction": self.route_instruction,
                "last_reasoning": self.latest_reasoning,
                "steps": self.step + 1,
            }
            (self.record_dir / "alpamayo_agent_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
