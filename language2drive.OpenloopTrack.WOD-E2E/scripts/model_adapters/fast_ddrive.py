#!/usr/bin/env python3
"""Fast-dDrive inference on WOD-E2E public prerelease inputs.

The checkpoint's published WOD prompt uses three front cameras, navigation
intent, and ego history. An English passenger instruction is appended here as
an experimental adaptation; the released checkpoint has no documented
passenger-instruction training. This runner reads no reference trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from drivema import (VIEWS, front_camera_history_complete, history,
                     interpolate_to_wod, read_jsonl, safe_image,
                     selected_requests)


PAIR = re.compile(r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]")


def native_history(row: dict) -> str:
    state = history(row)
    lines = []
    # Public history is sampled at 4 Hz over [-3.75, 0] seconds. Taking
    # indices 3, 5, ..., 15 reproduces the released model's 0.5 s schedule.
    for index in range(3, 16, 2):
        time_s = -3.75 + index * 0.25
        position = state["traj"][index]
        acceleration = state["acc"][index]
        velocity = state["vel"][index]
        lines.append(
            f"(t{time_s:+.1f}s) [{position[0]:.2f}, {position[1]:.2f}], "
            f"Acceleration: X {acceleration[0]:.2f}, Y {acceleration[1]:.2f} m/s², "
            f"Velocity: X {velocity[0]:.2f}, Y {velocity[1]:.2f} m/s"
        )
    return "; ".join(lines)


def prompt(row: dict, instruction: str) -> str:
    intent = row.get("route_command", "UNKNOWN")
    if intent not in ("GO_STRAIGHT", "GO_LEFT", "GO_RIGHT", "UNKNOWN"):
        raise ValueError(f"unsupported route command: {intent}")
    return (
        "You are an expert autonomous driving agent.\n"
        "Task 1: For every critical-object category, answer yes or no for "
        "whether it affects the ego trajectory: nearby_vehicle, pedestrian, "
        "cyclist, construction, traffic_element, weather_condition, road_hazard, "
        "emergency_vehicle, animal, special_vehicle, conflicting_vehicle, "
        "door_opening_vehicle.\n"
        "Task 2: Describe the relevant objects' likely motion and their effect "
        "on the next three seconds.\n"
        "Task 3: Predict speed in {keep, accelerate, decelerate, stop, other} "
        "and command in {straight, yield, left_turn, right_turn, lane_follow, "
        "lane_change_left, lane_change_right, reverse, overtake, other}.\n"
        "Task 4: Predict a five-second ego trajectory with five [x, y] waypoints "
        "at one-second intervals. Use JSON with keys critical_objects, "
        "explanation, future_meta_behavior, and trajectory.\n\n"
        "Input:\n"
        "- Three synchronized current front camera images: left, center, right.\n"
        f"- High-level navigation command: {intent}\n"
        "- Historical ego state from the last three seconds at 0.5-second intervals: "
        f"{native_history(row)}\n"
        f"- Passenger instruction: {instruction}\n"
        "- Coordinates: positive x forward, positive y left; meters relative to the current ego pose.\n"
        "Output:"
    )


def parse_trajectory(response: str) -> list[list[float]]:
    target = response
    try:
        document = json.loads(response)
        if isinstance(document, dict) and "trajectory" in document:
            field = document["trajectory"]
            target = field if isinstance(field, str) else json.dumps(field)
    except json.JSONDecodeError:
        match = re.search(r'"trajectory"\s*:', response)
        if match:
            target = response[match.end():]
    points = [[float(x), float(y)] for x, y in PAIR.findall(target)]
    if len(points) != 5 or not all(math.isfinite(n) for xy in points for n in xy):
        raise ValueError(f"expected five finite trajectory points; parsed {len(points)}")
    return points


def load_views(row: dict, root: Path):
    from PIL import Image

    images = []
    for view in VIEWS:
        with Image.open(safe_image(root, row["images"][view])) as image:
            images.append(image.convert("RGB"))
    return images


def run(args: argparse.Namespace) -> None:
    cache = Path(__file__).resolve().parents[2] / "models/cache/hf_home"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache / "modules"))
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    model_dir = args.model.resolve()
    for required in ("config.json", "model.safetensors.index.json"):
        if not (model_dir / required).is_file():
            raise FileNotFoundError(f"incomplete Fast-dDrive checkpoint: {required}")
    inputs = {row["input_id"]: row for row in read_jsonl(args.inputs)}
    requests = selected_requests(args.workload, args)
    device = f"cuda:{args.device_index}"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
        use_fast=False, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    processor.tokenizer = tokenizer
    mask_tokens = tokenizer.encode("||", add_special_tokens=False)
    if not mask_tokens:
        raise ValueError("Fast-dDrive mask token '||' missing")
    mask_id = int(mask_tokens[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.is_file() else "w"
    completed = set()
    if mode == "a":
        completed = {item["request_id"] for item in read_jsonl(args.output)}
    with args.output.open(mode, encoding="utf-8") as output:
        for request in requests:
            request_id = request["request_id"]
            if request_id in completed:
                continue
            row = inputs[request["input_id"]]
            instruction = str(request["instruction"])
            if request.get("instruction_language") != "en" or not instruction.strip():
                raise ValueError(f"{request_id}: expected nonempty English instruction")
            images = load_views(row, args.image_root)
            messages = [{"role": "user", "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": prompt(row, instruction)},
            ]}]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prepared = processor(
                text=[rendered], images=images, return_tensors="pt",
            ).to(device)
            kwargs = {
                "input_ids": prepared.input_ids,
                "tokenizer": tokenizer,
                "block_size": args.block_size,
                "max_tokens": args.max_tokens,
                "mask_id": mask_id,
                "threshold": 0.0,
            }
            for key in ("pixel_values", "image_grid_thw"):
                value = getattr(prepared, key, None)
                if value is not None:
                    kwargs[key] = value
            with torch.inference_mode():
                generated = model.scaffold_speculative_sample(**kwargs)
            response = tokenizer.decode(
                generated[0, prepared.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            result = {
                "request_id": request_id,
                "coordinate_frame": "ego_at_anchor",
                "reference_point": "rear_axle",
                "adapter": "fast_ddrive_passenger_prompt_experimental_v1",
                "used_views": list(VIEWS),
                "front_camera_history_complete": front_camera_history_complete(row),
                "raw_response": response,
            }
            try:
                native = parse_trajectory(response)
                result.update({
                    "status": "ok",
                    "native_timestamps_s": [1, 2, 3, 4, 5],
                    "native_xy_m": native,
                    "timestamps_s": [i / 4 for i in range(1, 21)],
                    "xy_m": interpolate_to_wod(native),
                })
            except ValueError as error:
                result.update({"status": "invalid_output", "error": str(error)})
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
