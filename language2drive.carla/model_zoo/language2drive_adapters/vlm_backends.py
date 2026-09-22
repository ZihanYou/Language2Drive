"""Inference backends used by the model-local closed-loop bridge."""

from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass
class Prediction:
    waypoints: list[tuple[float, float]]
    text: str


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def parse_waypoints(text: str, expected: int = 8) -> list[tuple[float, float]]:
    """Parse nested pairs or a flat comma-separated trajectory robustly."""

    clean = text.replace("−", "-")
    # Prefer the explicitly requested answer section so coordinates mentioned
    # during model reasoning cannot be mistaken for control waypoints.
    answer = clean.rsplit("<answer>", 1)[-1]
    number = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    labeled = re.findall(
        rf"[\[\(]\s*x\s*[:=]\s*{number}\s*[,;]\s*y\s*[:=]\s*{number}",
        answer,
        flags=re.IGNORECASE,
    )
    if labeled:
        return [(float(x), float(y)) for x, y in labeled[:expected]]
    triples = re.findall(
        rf"[\[\(]\s*{number}\s*[,;]\s*{number}\s*[,;]\s*{number}\s*[\]\)]",
        answer,
    )
    # A malformed final point may contain an accidental heading while all
    # preceding points are ordinary pairs.  Do not let one such triple hide a
    # complete pair trajectory; prefer triples only when they form a usable
    # trajectory on their own.
    if len(triples) >= 2:
        return [(float(x), float(y)) for x, y, _heading in triples[:expected]]
    candidates = re.findall(
        r"[\[\(]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[,;]\s*"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,?\s*[\]\)]",
        answer,
    )
    if candidates:
        return [(float(x), float(y)) for x, y in candidates[:expected]]
    values = [
        float(x)
        for x in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", answer)
    ]
    return [(values[i], values[i + 1]) for i in range(0, min(len(values) - 1, expected * 2), 2)]


def _prompt(instruction: str, speed: float, history_xy: list[tuple[float, float]]) -> str:
    history = ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in history_xy)
    return (
        "You are the trajectory planner controlling an autonomous car. Use the camera "
        "views and obey this exact navigation instruction: "
        f"{instruction!r}. Current ego speed is {speed:.2f} m/s. Recent ego-local "
        f"history (x forward, y left) is [{history}]. Predict exactly 8 future "
        "ego-local waypoints at 0.5-second intervals. Every x must be between 0 and "
        "40 metres, every y must be between -10 and 10 metres, and forward x should "
        "increase along the trajectory. Positive y means left. You must return a "
        "non-empty trajectory even when the scene is uncertain; in that case choose a "
        "slow, conservative trajectory rather than [] or a refusal. Finish with exactly "
        "this machine-readable form: <answer>[[x1,y1],[x2,y2],[x3,y3],[x4,y4],"
        "[x5,y5],[x6,y6],[x7,y7],[x8,y8]]</answer>. Do not put any other coordinate "
        "pairs inside the answer tags."
    )


def _onevl_prompt(
    instruction: str, speed: float, history_xy: list[tuple[float, float]]
) -> str:
    """Match the released OneVL-NAVSIM single-camera training prompt."""

    lower = instruction.lower()
    if "left" in lower or "左" in instruction:
        command = "TURN LEFT"
    elif "right" in lower or "右" in instruction:
        command = "TURN RIGHT"
    else:
        command = "MOVE FORWARD"
    history = list(history_xy[-3:])
    history = [(0.0, 0.0)] * (3 - len(history)) + history
    history_text = ", ".join(f"[{x:.2f}, {y:.2f}, 0.00]" for x, y in history)
    return (
        f"is the front view. Command: {command}. Velocity: [{speed:.2f}, "
        f"0.00]. Acceleration: [0.00, 0.00]. Historical trajectory: "
        f"[{history_text}]. Output the reasoning in <think></think> and the "
        "predicted trajectory in <answer></answer>. For the content in "
        "<answer></answer>, Only generate the predicted future waypoints in pure "
        "text format: [x_1, y_1, heading_1], [x_2, y_2, heading_2], ..., "
        "[x_8, y_8, heading_8]. Output exactly 8 waypoints in the format "
        "[x, y, heading], with numbers in decimal form and exactly 2 digits after "
        "the decimal point. Separate waypoints using comma. Do not include extra "
        "text, brackets, or invalid values. The output must strictly follow this "
        "structure."
    )






class TransformersTrajectoryBackend:
    def __init__(self, cfg: dict[str, Any], base: Path, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.cfg = cfg
        self.device = device
        backend = cfg["backend"]
        self.model_path = _resolve(backend["model_path"], base)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True, local_files_only=True
            )
        except ValueError:
            # The released Impromptu-VLA checkpoint was saved by a 4.49 dev
            # build with ``Qwen2_5_VLImageProcessor`` in its metadata.  The
            # benchmark's stable 4.49 build provides the same implementation
            # as ``Qwen2VLImageProcessor`` but cannot resolve the dev-only
            # class name through AutoProcessor.
            with (self.model_path / "config.json").open(encoding="utf-8") as stream:
                model_type = json.load(stream).get("model_type")
            if model_type != "qwen2_5_vl":
                raise
            from transformers import AutoTokenizer, Qwen2VLImageProcessor
            from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
                Qwen2_5_VLProcessor,
            )

            image_processor = Qwen2VLImageProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            with (self.model_path / "chat_template.json").open(encoding="utf-8") as stream:
                chat_template = json.load(stream)["chat_template"]
            self.processor = Qwen2_5_VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                chat_template=chat_template,
            )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        use_device_map = bool(backend.get("device_map_auto"))
        if use_device_map:
            load_kwargs.update(
                device_map="auto",
                max_memory={
                    0: f"{int(backend.get('max_gpu_memory_gib', 19))}GiB",
                    "cpu": f"{int(backend.get('max_cpu_memory_gib', 96))}GiB",
                },
            )
        model_cls: Any = AutoModelForCausalLM
        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq

                model_cls = AutoModelForVision2Seq
            except ImportError:
                pass
        try:
            self.model = model_cls.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
        except (ValueError, KeyError, TypeError):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
        if not use_device_map:
            self.model = self.model.to(device)
        self.model.eval()

    def predict(self, *, images, instruction, speed, history_xy) -> Prediction:
        # Preserve each released model's declared view order.  In particular,
        # Impromptu-VLA was trained on front, front-right, front-left rather
        # than the generic front, left, right ordering.
        text_only = bool(self.cfg["backend"].get("text_only"))
        configured_views = self.cfg.get("camera", {}).get("views", ["front", "left", "right", "rear"])
        order = [name for name in configured_views if name in images]
        pil = [Image.fromarray(images[name]) for name in order]
        content = [{"type": "image", "image": image} for image in pil]
        prompt = (_onevl_prompt(instruction, speed, history_xy)
                  if self.cfg["backend"].get("onevl_latent_prefix")
                  else _prompt(instruction, speed, history_xy))
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.cfg["backend"].get("onevl_latent_prefix"):
            n_text = int(self.cfg["backend"].get("num_latent", 2))
            n_visual = int(self.cfg["backend"].get("num_latent_vis", 4))
            answer_prefix = str(self.cfg["backend"].get("answer_prefix", "["))
            text += (
                "<|start-latent-vis|>" + "<|latent-vis|>" * n_visual
                + "<|end-latent-vis|><|start-latent|>" + "<|latent|>" * n_text
                + "<|end-latent|><answer>" + answer_prefix
            )
        if text_only:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(
                text=[text], images=pil, padding=True, return_tensors="pt"
            ).to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg["runtime"].get("max_new_tokens", 192)),
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[1]
        output = self.processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=False)[0]
        if self.cfg["backend"].get("onevl_latent_prefix"):
            # The official NAVSIM answer prefix is part of the prompt and is
            # therefore absent from the decoded continuation. Restore it
            # unconditionally; otherwise a valid 8-point answer silently loses
            # its first (unbracketed in the continuation) waypoint.
            answer_prefix = str(self.cfg["backend"].get("answer_prefix", "["))
            points = parse_waypoints(answer_prefix + output)
        else:
            points = parse_waypoints(output)
        if self.cfg["backend"].get("coordinate_convention") == "right_forward":
            points = [(float(forward), -float(right)) for right, forward in points]
        if self.cfg["backend"].get("auto_coordinate_convention"):
            forward_x = sum(float(x) > 0.15 for x, _y in points)
            forward_y = sum(float(y) > 0.15 for _x, y in points)
            if forward_x < 2 and forward_y >= 2:
                # Some general-purpose VLMs verbalize image-plane Cartesian
                # coordinates (x right, y forward) despite the requested ego
                # convention. Convert their model-produced trajectory without
                # inventing a route or replacement plan.
                points = [(float(y), -float(x)) for x, y in points]
        return Prediction(points, output)

    def close(self) -> None:
        self.model = None
        self.processor = None




class SennaBackend:
    """Official Senna VLM loader; high-level plans are converted to a trajectory."""

    def __init__(self, cfg: dict[str, Any], base: Path, device: torch.device):
        from llava.model.builder import load_senna_pretrained_model
        from transformers import AutoConfig

        self.cfg = cfg
        self.device = device
        path = str(_resolve(cfg["backend"]["model_path"], base))
        vision_tower = str(_resolve(cfg["backend"]["vision_tower_path"], base))
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        # The released checkpoint contains an absolute path from the authors'
        # training host.  Bind the same official CLIP tower to this model-local
        # checkpoint before the Senna class constructs its delayed vision tower.
        model_config.mm_vision_tower = vision_tower
        # The official Senna loader gates its custom class on the literal
        # ``llava`` model name (the released directory itself is named Senna).
        name = "llava"
        self.model_path = path
        self.tokenizer, self.model, self.image_processor, _ = load_senna_pretrained_model(
            path,
            None,
            name,
            device_map=0,
            config=model_config,
            local_files_only=True,
        )

    def predict(self, *, images, instruction, speed, history_xy) -> Prediction:  # noqa: ARG002
        import data_tools.senna_qa_utils as senna_qa

        # Match the six-token camera order used by Senna's released plan QA.
        order = ("front", "left", "right", "back_left", "back_right", "rear")
        pil = [Image.fromarray(images[name]) for name in order if name in images]
        image_prompt = (
            "<FRONT VIEW>:\n<image>\n"
            "<FRONT LEFT VIEW>:\n<image>\n"
            "<FRONT RIGHT VIEW>:\n<image>\n"
            "<BACK LEFT VIEW>:\n<image>\n"
            "<BACK RIGHT VIEW>:\n<image>\n"
            "<BACK VIEW>:\n<image>\n"
        )
        question = image_prompt + (
            f"Your current speed is {speed:.2f} m/s, the navigation command is "
            f"'{instruction}', based on the understanding of the driving scene "
            "and the navigation information, what is your plan for the next three "
            "seconds? Please answer your SPEED plan and your PATH plan. SPEED "
            "includes KEEP, ACCELERATE, DECELERATE, and STOP; PATH includes "
            "STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. For "
            "example, a correct answer format is 'KEEP, LEFT_CHANGE'."
        )
        # The released helper accepts file paths.  Keep CARLA frames in memory by
        # replacing only its image loader in this evaluator process.
        senna_qa.load_images = lambda _items: pil
        args = types.SimpleNamespace(
            model_path=self.model_path,
            model_base=None,
            query=question,
            conv_mode="llava_v1",
            image_file=[f"memory-{i}" for i in range(len(pil))],
            sep=",",
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
        )
        answer = senna_qa.eval_multi_img_model_wo_init(
            args, self.tokenizer, self.model, self.image_processor
        )
        lower = answer.lower()
        if any(word in lower for word in ("stop", "deceler", "slow")):
            target_speed = 0.0 if "stop" in lower else min(speed, 2.0)
        elif "acceler" in lower:
            target_speed = min(8.0, speed + 2.0)
        else:
            target_speed = max(3.0, min(7.0, speed))
        direction = -1.0 if "right" in lower else (1.0 if "left" in lower else 0.0)
        points = []
        for i in range(1, 9):
            t = 0.5 * i
            x = target_speed * t
            y = direction * 0.055 * x * x
            points.append((x, y))
        return Prediction(points, answer)

    def close(self) -> None:
        self.model = None




class QwenDriveBackend:
    """Official Qwen-Drive VLM plus its reward-optimized Planning Expert."""

    def __init__(self, cfg: dict[str, Any], base: Path, device: torch.device):
        from qwen_drive import InferenceMode, QwenDriveForPlanning

        self.cfg = cfg
        self.device = device
        self.InferenceMode = InferenceMode
        backend = cfg["backend"]
        self.model_path = _resolve(backend["model_path"], base)
        self.planner_path = _resolve(backend["planner_path"], base)
        self.model = QwenDriveForPlanning.from_pretrained(
            self.model_path,
            planner=self.planner_path,
            dtype=torch.bfloat16,
            attn_implementation=str(backend.get("attn_implementation", "sdpa")),
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self._scene_context: dict[str, Any] | None = None

    def set_scene_context(
        self,
        *,
        temporal_images: dict[str, list[np.ndarray]],
        history: np.ndarray,
        history_velocity: np.ndarray,
        history_acceleration: np.ndarray,
        ego_acceleration: tuple[float, float],
    ) -> None:
        self._scene_context = {
            "temporal_images": temporal_images,
            "history": history,
            "history_velocity": history_velocity,
            "history_acceleration": history_acceleration,
            "ego_acceleration": ego_acceleration,
        }

    @staticmethod
    def _navigation(instruction: str) -> tuple[int, list[float]]:
        lower = instruction.lower()
        if "left" in lower or "左" in instruction:
            return 1, [1.0, 0.0, 0.0, 0.0]
        if "right" in lower or "右" in instruction:
            return 2, [0.0, 0.0, 1.0, 0.0]
        return 0, [0.0, 1.0, 0.0, 0.0]

    def predict(self, *, images, instruction, speed, history_xy) -> Prediction:
        from qwen_drive import CameraFrame, DrivingScene

        context = self._scene_context
        if context is None:
            # This fallback is useful for the model-local preflight. Live closed-loop
            # evaluation supplies the full 1.5-second context through the agent.
            temporal_images = {name: [image] * 4 for name, image in images.items()}
            history = np.zeros((16, 3), dtype=np.float64)
            history_velocity = np.zeros((16, 2), dtype=np.float32)
            history_velocity[:, 0] = float(speed)
            history_acceleration = np.zeros((16, 2), dtype=np.float32)
            ego_acceleration = (0.0, 0.0)
        else:
            temporal_images = context["temporal_images"]
            history = context["history"]
            history_velocity = context["history_velocity"]
            history_acceleration = context["history_acceleration"]
            ego_acceleration = context["ego_acceleration"]
        self._scene_context = None

        view_names = {
            "<FRONT VIEW>": "front",
            "<FRONT LEFT VIEW>": "left",
            "<FRONT RIGHT VIEW>": "right",
        }
        views = {
            official: [
                CameraFrame(Image.fromarray(np.ascontiguousarray(frame)))
                for frame in temporal_images[local]
            ]
            for official, local in view_names.items()
        }
        nav_command, driving_command = self._navigation(instruction)
        scene = DrivingScene(
            views=views,
            history=history,
            history_velocity=history_velocity,
            history_acceleration=history_acceleration,
            ego_velocity=(float(speed), 0.0),
            ego_acceleration=ego_acceleration,
            driving_command=driving_command,
            nav_command=nav_command,
        )
        # Keep the official kinematics/navigation prompt, then expose the exact
        # Language2Drive instruction so language variants remain observable.
        scene.instruction_text = (
            scene.instruction()
            + "\n3. Language driving instruction: ["
            + instruction.strip()
            + "]"
        )
        runtime = self.cfg.get("runtime", {})
        output = self.model.run(
            self.InferenceMode.REASONING_PLANNING,
            scene=scene,
            num_samples=1,
            num_steps=int(runtime.get("planning_steps", 10)),
            seed=int(runtime.get("sampling_seed", 42)),
            max_new_tokens=int(runtime.get("max_reasoning_tokens", 64)),
        )
        trajectory = np.asarray(output.trajectory)
        # The native output is 50 points at 10 Hz. Language2Drive's controller
        # consumes eight points at 0.5-second intervals over the first 4 seconds.
        indices = np.arange(4, min(40, len(trajectory)), 5, dtype=np.int64)
        points = [(float(trajectory[i, 0]), float(trajectory[i, 1])) for i in indices]
        return Prediction(points, output.reasoning or "")

    def close(self) -> None:
        self.model = None
        self._scene_context = None


def create_backend(cfg: dict[str, Any], base: Path, device: torch.device):
    kind = cfg["backend"]["type"]
    if kind == "transformers_trajectory":
        return TransformersTrajectoryBackend(cfg, base, device)
    if kind == "senna_high_level":
        return SennaBackend(cfg, base, device)
    if kind == "qwen_drive":
        return QwenDriveBackend(cfg, base, device)
    raise ValueError(f"unsupported backend type: {kind}")
