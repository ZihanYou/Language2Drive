#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from common import (acceleration_history, append_progress, consolidate, ego_history,
                    image_path, interpolate_history, iter_queries, load_progress,
                    parse_numeric_points, resample_trajectory, seed_everything,
                    submission_trajectory, velocity_history)


def fmt(point):
    return "[" + ", ".join(f"{float(x):.2f}" for x in point[:2]) + "]"


def first_prompt(q):
    row = q.public_input
    past = interpolate_history(ego_history(row)[:, :2])
    velocity = interpolate_history(velocity_history(row))
    acceleration = interpolate_history(acceleration_history(row))
    return (
        "You are an expert driver.\nInput:\n"
        "- 1 frame of multi-view images collected from the ego-vehicle at the present "
        "timestep: front_left_view: <image>; front_view:<image>; front_right_view:<image>\n"
        f"- Current high-level intent:{q.instruction}\n"
        f"- Current acceleration is {fmt(acceleration[-1])}\n"
        f"- Current velocity is {fmt(velocity[-1])}\n"
        f"- Observed 1.5-second past trajectory (16 interpolated steps at 10 Hz):"
        f"{', '.join(fmt(v) for v in past)}\n"
        f"- Observed 1.5-second past acceleration (16 interpolated steps at 10 Hz):"
        f"{', '.join(fmt(v) for v in acceleration)}\n"
        f"- Observed 1.5-second past velocity (16 interpolated steps at 10 Hz):"
        f"{', '.join(fmt(v) for v in velocity)}\n"
        "Coordinate System Definition: X-axis: positive forward, negative backward; "
        "Y-axis: positive left, negative right.\n"
        "Task: Inspect the input and make the decision.\nOutput format:\n"
        "longitudinal action: xx, lateral action: xx"
    )


SECOND_PROMPT = (
    "Task: Given the above information, predict the optimal 5-second future trajectory "
    "(5 steps at 1 Hz) of the ego vehicle.\nOutput format:\n"
    "[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5]"
)


def image_list(q):
    row = q.public_input
    return [Image.open(image_path(row, 3, camera)).convert("RGB")
            for camera in ("CAM_L0", "CAM_F0", "CAM_R0")]


def content_from_prompt(prompt):
    parts = prompt.split("<image>")
    content = []
    for index, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if index < len(parts) - 1:
            content.append({"type": "image"})
    return content


def generate(model, processor, messages, images, max_new_tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model_eval/weights/drivema-2b"))
    p.add_argument("--output", type=Path, default=Path("model_eval/runs/drivema"))
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    args = p.parse_args()
    seed_everything(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True).to("cuda").eval()
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
        images = image_list(q)
        try:
            prompt = first_prompt(q)
            turn1 = [{"role": "user", "content": content_from_prompt(prompt)}]
            decision = generate(model, processor, turn1, images, 96)
            turn2 = turn1 + [
                {"role": "assistant", "content": decision},
                {"role": "user", "content": SECOND_PROMPT},
            ]
            raw = generate(model, processor, turn2, images, 128)
            points = parse_numeric_points(raw, min_points=5)[-5:]
            xy = resample_trajectory([[0.0, 0.0]] + [v[:2] for v in points],
                                     [0, 1, 2, 3, 4, 5])
            row.update(trajectory=submission_trajectory(xy), decision=decision,
                       raw_output=raw, valid=True)
        except Exception as exc:
            row.update(trajectory={}, valid=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            for image in images:
                image.close()
        row["latency"] = time.time() - started
        append_progress(progress_path, row)
        count += 1
        print(f"[{count}] {q.key} valid={row['valid']} {row['latency']:.2f}s", flush=True)
    if args.limit is None and args.world == 1:
        consolidate(progress_path, args.output)


if __name__ == "__main__":
    main()
