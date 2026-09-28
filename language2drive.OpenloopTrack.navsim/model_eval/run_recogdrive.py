#!/usr/bin/env python3
import argparse
import gc
import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from common import (acceleration_history, append_progress, consolidate, ego_history,
                    image_path, iter_queries, load_progress, seed_everything,
                    submission_trajectory, velocity_history)


def _dummy_module(name, symbols):
    module = types.ModuleType(name)
    for symbol in symbols:
        setattr(module, symbol, type(symbol, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules[name] = module


# These modules are used only by ReCogDrive's optional GRPO scoring path.
_dummy_module("navsim.common.dataclasses", ["Trajectory"])
_dummy_module("navsim.common.dataloader", ["MetricCacheLoader"])
score_module = types.ModuleType("navsim.evaluate.pdm_score")
score_module.pdm_score = lambda *args, **kwargs: None
sys.modules[score_module.__name__] = score_module
_dummy_module(
    "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
    ["PDMScorer", "PDMScorerConfig"],
)
_dummy_module(
    "navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator",
    ["PDMSimulator"],
)
_dummy_module("nuplan.planning.simulation.trajectory.trajectory_sampling", ["TrajectorySampling"])

from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig)
from navsim.agents.recogdrive.recogdrive_backbone import (  # noqa: E402
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, system_message)
from navsim.agents.recogdrive.utils.conversation import get_conv_template  # noqa: E402
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image  # noqa: E402


def planner_config():
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8, "head_dim": 48, "num_layers": 16,
            "output_dim": 512, "dropout": 0.0, "attention_bias": True,
            "norm_eps": 1e-5, "interleave_attention": True,
        },
        action_dim=3, action_horizon=8, input_embedding_dim=384,
        sampling_method="ddim", num_inference_steps=5, grpo=False,
        model_dtype="float16", vlm_size="small",
    )


class Backbone:
    def __init__(self, checkpoint):
        self.model = AutoModel.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, use_flash_attn=False, device_map="cuda",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, use_fast=False)
        self.model.system_message = system_message
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.num_image_token = 256

    def encode(self, pixel_values, question):
        num_patches = pixel_values.shape[0]
        template = get_conv_template("internvl2_5")
        template.system_message = system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer([query], return_tensors="pt", padding="max_length",
                                max_length=2800).to("cuda")
        position_ids = inputs.attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(inputs.attention_mask == 0, 1)
        image_flags = torch.ones(num_patches, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            outputs = self.model(
                pixel_values=pixel_values.to("cuda", dtype=next(self.model.parameters()).dtype),
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                position_ids=position_ids, image_flags=image_flags,
                output_hidden_states=True, return_dict=True,
            )
        return outputs.hidden_states[-1]


def prompt(q, history):
    history_str = " ".join(
        f"   - t-{3-i}: ({pose[0]:+.2f}, {pose[1]:+.2f}, {pose[2]:+.2f})"
        for i, pose in enumerate(history)
    )
    return (
        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Language driving instruction: [{q.instruction}]\n"
        "Output requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", type=Path, default=Path("model_eval/weights/recogdrive-vlm-2b"))
    p.add_argument("--planner", type=Path, default=Path(
        "model_eval/weights/recogdrive-2b-rl/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"))
    p.add_argument("--output", type=Path, default=Path("model_eval/runs/recogdrive"))
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    args = p.parse_args()
    seed_everything(args.seed)
    backbone = Backbone(str(args.vlm))
    planner = ReCogDriveDiffusionPlanner(planner_config()).to("cuda").eval()
    checkpoint = torch.load(args.planner, map_location="cpu", weights_only=False, mmap=True)
    prefix = "agent.action_head."
    target = planner.state_dict()
    state = {}
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        if stripped in target and target[stripped].shape == value.shape:
            state[stripped] = value
    missing, unexpected = planner.load_state_dict(state, strict=False)
    print(f"planner missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del state, checkpoint
    gc.collect()
    progress_path = args.output / "progress.jsonl"
    done = load_progress(progress_path)
    count = 0
    for query_index, q in enumerate(iter_queries()):
        if query_index % args.world != args.rank:
            continue
        if q.key in done and "trajectory" in done[q.key]:
            continue
        if args.limit is not None and count >= args.limit:
            break
        seed_everything(args.seed)
        started = time.time()
        row = {"key": q.key, "pcu_id": q.pcu_id, "rewrite": q.rewrite, "arm": q.arm}
        try:
            history = torch.tensor(ego_history(q.public_input), dtype=torch.float32, device="cuda")
            pixels = load_image(str(image_path(q.public_input, 3, "CAM_F0")), max_num=12)
            hidden = backbone.encode(pixels, prompt(q, history.cpu().numpy()))
            velocity = torch.tensor(velocity_history(q.public_input)[-1], device="cuda")
            acceleration = torch.tensor(acceleration_history(q.public_input)[-1], device="cuda")
            status = torch.cat([torch.tensor([0, 1, 0, 0], device="cuda"),
                                velocity, acceleration]).unsqueeze(0)
            history_flat = history.reshape(1, -1)
            planner_dtype = next(planner.parameters()).dtype
            action_inputs = BatchFeature({
                "state": torch.cat([status, history_flat], dim=1).to(planner_dtype),
                "his_traj": history_flat.to(planner_dtype),
                "status_feature": status.to(planner_dtype),
            })
            with torch.inference_mode():
                result = planner.get_action(hidden.to(dtype=next(planner.parameters()).dtype),
                                            action_inputs)
            trajectory = result["pred_traj"].float().cpu().squeeze(0).numpy()
            row.update(trajectory=submission_trajectory(trajectory[:, :2], trajectory[:, 2]),
                       valid=True)
        except Exception as exc:
            row.update(trajectory={}, valid=False, error=f"{type(exc).__name__}: {exc}")
        row["latency"] = time.time() - started
        append_progress(progress_path, row)
        count += 1
        print(f"[{count}] {q.key} valid={row['valid']} {row['latency']:.2f}s", flush=True)
    if args.limit is None and args.world == 1:
        consolidate(progress_path, args.output)


if __name__ == "__main__":
    main()
