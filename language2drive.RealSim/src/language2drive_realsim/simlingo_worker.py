"""Isolated official SimLingo inference worker.

Run this file with the Python environment prescribed by the SimLingo release.
Stdout is reserved for the length-framed protocol; model logs go to stderr.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
from pathlib import Path
import sys
import traceback
from typing import Any


# Official SimLingo/CARLA front-camera pose from
# ``team_code/simlingo_utils.py:get_camera_extrinsics``.  The rotation is the
# identity and the translation is expressed in the ego frame.
OFFICIAL_CAMERA_TRANSLATION_M = (-1.5, 0.0, 2.0)


def format_instruction_prompt(speed_m_s: float, instruction: str) -> str:
    """Match official interactive ``user_flag=2`` route-free prompting."""

    text = str(instruction).strip()
    if not text and os.environ.get("L2D_DIAGNOSTIC_ALLOW_EMPTY") != "1":
        raise ValueError("Language2Drive sidecar instruction must be non-empty")
    speed = round(float(speed_m_s), 1)
    return f"<INSTRUCTION_FOLLOWING> Current speed: {speed} m/s. {text}"


class OfficialSimLingo:
    def __init__(
        self,
        *,
        code_root: Path,
        checkpoint: Path,
        config: Path,
        base_model: Path,
        device: str,
    ) -> None:
        sys.path.insert(0, str(code_root))

        import cv2
        import hydra
        import numpy as np
        from omegaconf import OmegaConf
        from PIL import Image
        import torch
        from transformers import AutoConfig, AutoProcessor

        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
        from simlingo_training.utils.internvl2_utils import (
            build_transform,
            dynamic_preprocess,
        )

        self.cv2 = cv2
        self.np = np
        self.Image = Image
        self.torch = torch
        self.DrivingInput = DrivingInput
        self.LanguageLabel = LanguageLabel
        self.build_transform = build_transform
        self.dynamic_preprocess = dynamic_preprocess
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("official SimLingo inference requires CUDA")

        cfg = OmegaConf.load(config)
        cfg.model.vision_model.variant = str(base_model)
        cfg.model.language_model.variant = str(base_model)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
        processor = AutoProcessor.from_pretrained(
            str(base_model), trust_remote_code=True, local_files_only=True
        )
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<WAYPOINTS>",
                    "<WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS>",
                    "<WAYPOINT_LAST>",
                    "<ROUTE>",
                    "<ROUTE_DIFF>",
                    "<TARGET_POINT>",
                ]
            }
        )
        tokenizer.padding_side = "left"

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=str(base_model.parent),
                _recursive_=False,
            ).to(self.device)
        finally:
            torch.set_default_dtype(default_dtype)

        try:
            state = torch.load(
                str(checkpoint), map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            state = torch.load(
                str(checkpoint), map_location="cpu", weights_only=True
            )
        if not isinstance(state, dict) or not state:
            raise ValueError("official SimLingo weight file is not a state dictionary")
        model.load_state_dict(state, strict=True)
        del state
        gc.collect()
        model.eval()

        config_obj = AutoConfig.from_pretrained(
            str(base_model), trust_remote_code=True, local_files_only=True
        )
        image_size = config_obj.force_image_size or config_obj.vision_config.image_size
        patch_size = config_obj.vision_config.patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config_obj.downsample_ratio**2)
        )
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.conversation_module = self._load_conversation(base_model / "conversation.py")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def _load_conversation(path: Path) -> Any:
        spec = importlib.util.spec_from_file_location("simlingo_official_conversation", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official conversation template from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _preprocess(self, rgb: Any) -> Any:
        cv2 = self.cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("SimLingo JPEG preprocessing failed")
        bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height = rgb.shape[0]
        rgb = rgb[: int(height - (height * 4.8) // 16), :, :]

        image = self.Image.fromarray(rgb)
        images = self.dynamic_preprocess(
            image, image_size=448, use_thumbnail=False, max_num=2
        )
        transform = self.build_transform(input_size=448)
        patches = self.torch.stack([transform(item) for item in images])
        if patches.shape[0] != 2:
            raise ValueError(
                f"official 1024x512 SimLingo preprocessing expected 2 patches, got {patches.shape[0]}"
            )
        return patches.view(1, 1, patches.shape[0], 3, 448, 448)

    def _language_label(self, prompt: str, num_patches: int) -> Any:
        template = self.conversation_module.get_conv_template("internlm2-chat")
        question = "<image>\n" + prompt
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        system_prompt = (
            template.system_template.replace("{system_message}", template.system_message)
            + template.sep
        )
        query = query.replace(system_prompt, "")
        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * self.num_image_token * num_patches
            + "</img>"
        )
        query = query.replace("<image>", image_tokens, 1)
        tokenized = self.tokenizer(
            [query], padding=True, return_tensors="pt", add_special_tokens=False
        )
        valid = tokenized["input_ids"] != self.tokenizer.pad_token_id
        return self.LanguageLabel(
            phrase_ids=tokenized["input_ids"].to(self.device),
            phrase_valid=valid.to(self.device),
            phrase_mask=valid.to(self.device),
            placeholder_values=[{}],
            language_string=[query],
            loss_masking=None,
        )

    def infer(
        self,
        rgb: Any,
        *,
        speed_m_s: float,
        instruction: str,
        inference_seed: int,
    ) -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(int(inference_seed))
        torch.cuda.manual_seed_all(int(inference_seed))
        processed = self._preprocess(rgb).to(self.device).bfloat16()
        prompt = format_instruction_prompt(speed_m_s, instruction)
        label = self._language_label(prompt, processed.shape[2])

        camera_intrinsics = torch.tensor(
            [[[358.506, 0.0, 512.0], [0.0, 358.506, 256.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        camera_extrinsics = torch.eye(
            4, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        camera_extrinsics[:, :3, 3] = torch.tensor(
            OFFICIAL_CAMERA_TRANSLATION_M,
            dtype=torch.float32,
            device=self.device,
        )
        model_input = self.DrivingInput(
            camera_images=processed,
            image_sizes=None,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            vehicle_speed=torch.tensor(
                [[round(float(speed_m_s), 1)]],
                dtype=torch.float32,
                device=self.device,
            ),
            target_point=torch.zeros((1, 2), dtype=torch.float32, device=self.device),
            prompt=label,
            prompt_inference=label,
        )
        with torch.inference_mode():
            speed_waypoints, route, language = self.model(model_input)
        if speed_waypoints is None or route is None:
            raise RuntimeError("official SimLingo returned a missing trajectory head")
        return {
            "ok": True,
            "speed_waypoints": speed_waypoints[0].float().cpu().tolist(),
            "route": route[0].float().cpu().tolist(),
            "language": list(language),
            "prompt": prompt,
        }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> int:
    protocol_in = sys.stdin.buffer
    protocol_out = sys.stdout.buffer
    # Reserve stdout for framed data before importing modules that print.
    sys.stdout = sys.stderr
    from simlingo_model import encode_response, read_request

    args = _arguments()
    try:
        model = OfficialSimLingo(
            code_root=args.code_root,
            checkpoint=args.checkpoint,
            config=args.config,
            base_model=args.base_model,
            device=args.device,
        )
        protocol_out.write(encode_response({"ok": True, "ready": True}))
        protocol_out.flush()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        protocol_out.write(
            encode_response({"ok": False, "ready": False, "error": str(error)})
        )
        protocol_out.flush()
        return 1

    while True:
        try:
            metadata, image = read_request(protocol_in)
        except EOFError:
            return 0
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            protocol_out.write(encode_response({"ok": False, "error": str(error)}))
            protocol_out.flush()
            continue
        try:
            response = model.infer(
                image,
                speed_m_s=float(metadata["speed_m_s"]),
                instruction=str(metadata["instruction"]),
                inference_seed=int(metadata["inference_seed"]),
            )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": str(error)}
        protocol_out.write(encode_response(response))
        protocol_out.flush()


if __name__ == "__main__":
    raise SystemExit(main())
