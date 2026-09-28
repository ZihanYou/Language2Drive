#!/usr/bin/env python3
import argparse
import gc
import sys
import time
import types
from pathlib import Path

import torch
import yaml

# AutoVLA's inference class shares a file with training-only metric imports.
# Supply the three unused training symbols so inference has no NAVSIM devkit dependency.
score_stub = types.ModuleType("models.utils.score")
score_stub.PDM_Reward = type("PDM_Reward", (), {})
score_stub.TrajectorySampling = type("TrajectorySampling", (), {})
score_stub.Trajectory = type("Trajectory", (), {})
sys.modules["models.utils.score"] = score_stub

from models.autovla import AutoVLA  # noqa: E402

from common import (acceleration_history, append_progress, consolidate, image_path,
                    iter_queries, load_progress, seed_everything,
                    submission_trajectory, velocity_history)


def features(q):
    row = q.public_input
    return {
        "images": {
            "front_camera": [str(image_path(row, i, "CAM_F0")) for i in range(4)],
            "front_left_camera": [str(image_path(row, i, "CAM_L0")) for i in range(4)],
            "front_right_camera": [str(image_path(row, i, "CAM_R0")) for i in range(4)],
        },
        "sensor_data_path": None,
        "vehicle_velocity": velocity_history(row)[-1].tolist(),
        "vehicle_acceleration": acceleration_history(row)[-1].tolist(),
        "driving_command": q.instruction,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=Path,
                   default=Path("model_eval/weights/qwen2.5-vl-3b-instruct"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("model_eval/weights/autovla/AutoVLA_PDMS_89.ckpt"))
    p.add_argument("--config", type=Path, default=Path(
        "model_eval/repos/autovla/config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    p.add_argument("--codebook", type=Path,
                   default=Path("model_eval/repos/autovla/codebook_cache/agent_vocab.pkl"))
    p.add_argument("--output", type=Path, default=Path("model_eval/runs/autovla"))
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config["model"]["pretrained_model_path"] = str(args.base_model.resolve())
    config["model"]["codebook_cache_path"] = str(args.codebook.resolve())
    seed_everything(args.seed)
    model = AutoVLA(config, device="cuda").eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = {key.replace("autovla.", "", 1): value
             for key, value in checkpoint["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint missing={len(missing)} unexpected={len(unexpected)}", flush=True)
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
            trajectory, reasoning = model.predict(features(q))
            trajectory = torch.as_tensor(trajectory).float().cpu().numpy()
            if len(trajectory) < 8:
                raise ValueError(f"model produced {len(trajectory)} trajectory points")
            row.update(trajectory=submission_trajectory(trajectory[:8, :2], trajectory[:8, 2]),
                       reasoning=reasoning, valid=True)
        except Exception as exc:
            row.update(trajectory={}, valid=False,
                       error=f"{type(exc).__name__}: {exc}")
        row["latency"] = time.time() - started
        append_progress(progress_path, row)
        count += 1
        print(f"[{count}] {q.key} valid={row['valid']} {row['latency']:.2f}s", flush=True)
    if args.limit is None and args.world == 1:
        consolidate(progress_path, args.output)


if __name__ == "__main__":
    main()
