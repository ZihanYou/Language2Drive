#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from qwen_drive import CameraFrame, DrivingScene, InferenceMode, QwenDriveForPlanning

from common import (acceleration_history, append_progress, consolidate, ego_history,
                    image_path, interpolate_history, iter_queries, load_progress,
                    seed_everything, submission_trajectory, velocity_history)


def instruction_text(history: np.ndarray, instruction: str) -> str:
    labels = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")
    observed = history[[0, 5, 10, 15]]
    lines = "".join(
        f" -{label}: ({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f});\n"
        for label, pose in zip(labels, observed)
    )
    return (
        "The input images are organized by camera view. Each view contains 4 temporal "
        "frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is the current "
        "frame at t=0s).\n1. Historical trajectories (x, y, heading) in the current "
        "frame's ego coordinate system. Positive x points forward, positive y points "
        "left, and a positive heading indicates a left turn:\n" + lines +
        f"2. Language driving instruction: [{instruction}]"
    )


def make_scene(q):
    row = q.public_input
    history = interpolate_history(ego_history(row))
    velocity = interpolate_history(velocity_history(row))
    acceleration = interpolate_history(acceleration_history(row))
    mapping = {
        "<FRONT VIEW>": "CAM_F0",
        "<FRONT LEFT VIEW>": "CAM_L0",
        "<FRONT RIGHT VIEW>": "CAM_R0",
    }
    views = {view: [CameraFrame(image_path(row, frame, camera)) for frame in range(4)]
             for view, camera in mapping.items()}
    return DrivingScene(
        views=views, history=history, history_velocity=velocity,
        history_acceleration=acceleration, ego_velocity=velocity[-1],
        ego_acceleration=acceleration[-1], driving_command=[0, 1, 0, 0],
        nav_command=0, instruction_text=instruction_text(history, q.instruction),
        token=q.key, metadata={"pcu_id": q.pcu_id, "rewrite": q.rewrite, "arm": q.arm},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model_eval/weights/qwen-drive-1.0-4b"))
    p.add_argument("--output", type=Path, default=Path("model_eval/runs/qwen_drive"))
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    progress_path = args.output / "progress.jsonl"
    done = load_progress(progress_path)
    seed_everything(args.seed)
    model = QwenDriveForPlanning.from_pretrained(
        str(args.model), planner=str(args.model / "planner-rl"),
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    count = 0
    for q in iter_queries():
        if q.key in done and "trajectory" in done[q.key]:
            continue
        if args.limit is not None and count >= args.limit:
            break
        seed_everything(args.seed)
        started = time.time()
        result = model.run(InferenceMode.REASONING_PLANNING, scene=make_scene(q),
                           num_samples=1, seed=args.seed, max_new_tokens=128)
        trajectory = np.asarray(result.trajectories[0], dtype=np.float64)
        selected = trajectory[[4, 9, 14, 19, 24, 29, 34, 39]]
        row = {
            "key": q.key, "pcu_id": q.pcu_id, "rewrite": q.rewrite, "arm": q.arm,
            "trajectory": submission_trajectory(selected[:, :2], selected[:, 2]),
            "reasoning": result.reasoning, "latency": time.time() - started,
            "valid": True,
        }
        append_progress(progress_path, row)
        count += 1
        print(f"[{count}] {q.key} {row['latency']:.2f}s", flush=True)
    if args.limit is None:
        consolidate(progress_path, args.output)


if __name__ == "__main__":
    main()
