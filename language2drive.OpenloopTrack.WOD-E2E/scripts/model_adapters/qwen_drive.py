#!/usr/bin/env python3
"""Prepare public WOD-E2E language requests for the official Qwen-Drive runner.

This adapter reads only public inputs and workload. It never opens private targets.
It writes one official planning scene per request, with the exact passenger instruction
included in the model's text prompt. Inference remains a separate, explicit step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


VIEWS = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")
T_OFFSETS = (-1.5, -1.0, -0.5, 0.0)
INTENT_TO_NAV = {1: 0, 2: 1, 3: 2}  # Waymo: straight, left, right.
NAV_NAMES = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
HISTORY_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: all JSONL rows must be objects")
    return rows


def numeric_vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}: expected {length} numbers")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{label}: non-finite number")
    return result


def series(value: object, length: int, width: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}: expected [{length},{width}]")
    return [numeric_vector(row, width, f"{label}[{i}]") for i, row in enumerate(value)]


def get_images(row: dict, image_root: Path, allowed_jitter: float,
               allow_current_only: bool = False) -> tuple[dict[str, list[dict]], dict[str, list[float]]]:
    images = row.get("camera_history", row.get("images"))
    if not isinstance(images, dict):
        raise ValueError(f"{row.get('input_id')}: images must map camera names to 4 frames")
    if allow_current_only and all(images.get(view) == [] for view in VIEWS):
        current = row.get("images")
        if not isinstance(current, dict):
            raise ValueError(f"{row.get('input_id')}: current-only images are missing")
        result = {}
        for view in VIEWS:
            item = current.get(view)
            if not isinstance(item, dict):
                raise ValueError(f"{row.get('input_id')}: missing current {view} image")
            path = item.get("path")
            if not isinstance(path, str) or not path or Path(path).is_absolute():
                raise ValueError(f"{row.get('input_id')}: invalid current {view} path")
            absolute = (image_root / path).resolve()
            if not absolute.is_relative_to(image_root.resolve()) or not absolute.is_file():
                raise ValueError(f"{row.get('input_id')}: missing/unsafe current image {path}")
            if abs(float(item.get("offset_s", 0.0))) > 1e-6:
                raise ValueError(f"{row.get('input_id')}: current {view} is not the anchor")
            result[view] = [{"image": path, "timestamp_idx": 0,
                             "timestamp_offset_sec": 0.0,
                             "resized_width": 720, "resized_height": 799}]
        return result, {view: [0.0] for view in VIEWS}
    result, actual_offsets = {}, {}
    for view in VIEWS:
        items = images.get(view)
        if not isinstance(items, list) or len(items) != 4:
            raise ValueError(f"{row.get('input_id')}: {view} needs 4 frames at {T_OFFSETS} s")
        frames, offsets, seen_paths, seen_hashes = [], [], set(), set()
        for i, item in enumerate(items):
            path = item.get("path", item.get("image")) if isinstance(item, dict) else item
            if not isinstance(path, str) or not path or Path(path).is_absolute():
                raise ValueError(f"{row.get('input_id')}: {view}[{i}] needs a relative path")
            absolute = (image_root / path).resolve()
            if not absolute.is_relative_to(image_root.resolve()) or not absolute.is_file():
                raise ValueError(f"{row.get('input_id')}: missing/unsafe image {path}")
            if path in seen_paths:
                raise ValueError(f"{row.get('input_id')}: duplicate frame path in {view}: {path}")
            seen_paths.add(path)
            if isinstance(item, dict) and item.get("sha256"):
                if item["sha256"] in seen_hashes:
                    raise ValueError(f"{row.get('input_id')}: duplicate frame content in {view}")
                seen_hashes.add(item["sha256"])
            offset = (item.get("timestamp_offset_sec", item.get("offset_s"))
                      if isinstance(item, dict) else None)
            if offset is None and allowed_jitter > 0:
                raise ValueError(f"{row.get('input_id')}: {view}[{i}] lacks observed offset")
            offset = T_OFFSETS[i] if offset is None else float(offset)
            nominal = item.get("nominal_offset_s") if isinstance(item, dict) else None
            if nominal is not None and abs(float(nominal) - T_OFFSETS[i]) > 1e-6:
                raise ValueError(f"{row.get('input_id')}: {view}[{i}] wrong nominal offset {nominal}")
            if not math.isfinite(offset) or abs(offset - T_OFFSETS[i]) > allowed_jitter + 1e-8:
                raise ValueError(f"{row.get('input_id')}: {view}[{i}] observed offset {offset} exceeds jitter {allowed_jitter}")
            if offsets and offset <= offsets[-1]:
                raise ValueError(f"{row.get('input_id')}: {view} observed offsets are not increasing")
            if i == 3 and abs(offset) > 1e-6:
                raise ValueError(f"{row.get('input_id')}: {view} current frame must be the anchor")
            offsets.append(offset)
            size = (384, 416) if i < 3 else (720, 799)
            frames.append({
                "image": path,
                "timestamp_idx": i,
                "timestamp_offset_sec": offset,
                "resized_width": size[0],
                "resized_height": size[1],
            })
        result[view] = frames
        actual_offsets[view] = offsets
    return result, actual_offsets


def get_navigation(row: dict) -> tuple[int, list[int]]:
    command = row.get("nav_command")
    if command is None:
        intent = row.get("intent")
        if isinstance(intent, str):
            intent = {"GO_STRAIGHT": 1, "GO_LEFT": 2, "GO_RIGHT": 3}.get(intent)
        command = INTENT_TO_NAV.get(intent)
    if command is None:
        command = {"GO_STRAIGHT": 0, "GO_LEFT": 1, "GO_RIGHT": 2}.get(row.get("route_command"))
    if command is None or int(command) not in (0, 1, 2):
        raise ValueError(f"{row.get('input_id')}: missing valid Waymo intent/nav_command")
    command = int(command)
    onehot = [int(i == command) for i in range(3)] + [0]
    return command, onehot


def get_history(row: dict, official_repo: Path) -> tuple[list, list, list]:
    past = row.get("past_states")
    if not isinstance(past, dict):
        raise ValueError(f"{row.get('input_id')}: past_states is missing")
    if all(k in past for k in ("hist_traj_10hz", "hist_vel_10hz", "hist_acc_10hz")):
        return (
            series(past["hist_traj_10hz"], 16, 3, "hist_traj_10hz"),
            series(past["hist_vel_10hz"], 16, 2, "hist_vel_10hz"),
            series(past["hist_acc_10hz"], 16, 2, "hist_acc_10hz"),
        )
    aliases = {
        "pos_x": ("pos_x", "x"), "pos_y": ("pos_y", "y"),
        "vel_x": ("vel_x", "vx"), "vel_y": ("vel_y", "vy"),
        "accel_x": ("accel_x", "ax"), "accel_y": ("accel_y", "ay"),
    }
    raw = {}
    for official, options in aliases.items():
        value = next((past[k] for k in options if k in past), None)
        raw[official] = numeric_vector(value, 16, f"past_states.{official}")
    interpolation = official_repo / "scripts" / "interpolate_waymo_trajectory.py"
    if not interpolation.is_file():
        raise ValueError(f"official Qwen-Drive interpolation script missing: {interpolation}")
    spec = importlib.util.spec_from_file_location("qwen_drive_waymo_interpolation", interpolation)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    position, velocity, acceleration = module.interpolate_history(raw)
    return position.tolist(), velocity.tolist(), acceleration.tolist()


def prompt(history: list, nav: int, instruction: str, actual_offsets: dict[str, list[float]]) -> str:
    sampled = (0, 5, 10, 15)
    lines = "".join(
        f" -{HISTORY_LABELS[i]}: ({history[j][0]:.4f}, "
        f"{history[j][1]:.4f}, {history[j][2]:.4f});\n"
        for i, j in enumerate(sampled)
    )
    current_only = all(actual_offsets[view] == [0.0] for view in VIEWS)
    exact = not current_only and all(abs(actual_offsets[view][i] - T_OFFSETS[i]) < 1e-8 for view in VIEWS for i in range(4))
    if current_only:
        temporal_text = (
            "The input contains one current image at t=0s for each of FRONT, FRONT LEFT "
            "and FRONT RIGHT. No historical camera images are available. The ego-motion "
            "history below is available from vehicle states.\n"
        )
    elif exact:
        temporal_text = (
            "The input images are organized by camera view. Each view contains 4 temporal "
            "frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is the "
            "current frame at t=0s).\n"
        )
    else:
        details = "; ".join(
            f"{view}: {','.join(f'{value:.1f}' for value in actual_offsets[view])}"
            for view in VIEWS
        )
        temporal_text = (
            "The input images are organized by camera view, with each view ordered oldest "
            "to current. Nominal times are -1.5, -1.0, -0.5 and 0.0 seconds. "
            f"Actual observed time offsets in seconds are {details}.\n"
        )
    return (
        temporal_text +
        "1. Historical trajectories (x, y, heading) in the current frame's ego "
        "coordinate system. Positive x points forward, positive y points left, "
        "and a positive heading indicates a left turn：\n"
        f"{lines}2. Active navigation command: [{NAV_NAMES[nav]}]\n"
        f"3. Passenger instruction: {instruction}"
    )


def scene_record(row: dict, request: dict, image_root: Path, official_repo: Path,
                 allowed_jitter: float, allow_current_only: bool = False) -> dict:
    request_id = str(request.get("request_id") or "").strip()
    input_id = str(request.get("input_id") or "").strip()
    instruction = str(request.get("instruction") or "").strip()
    if not request_id or not input_id or not instruction:
        raise ValueError("request_id, input_id and instruction are required")
    if not any("A" <= ch <= "z" for ch in instruction) or any("\u4e00" <= ch <= "\u9fff" for ch in instruction):
        raise ValueError(f"{request_id}: passenger instruction must be English")
    images, actual_offsets = get_images(row, image_root, allowed_jitter, allow_current_only)
    history, velocity, acceleration = get_history(row, official_repo)
    nav, driving_command = get_navigation(row)
    if max(abs(v) for v in history[-1]) > 0.01:
        raise ValueError(f"{input_id}: last history pose must be current ego origin")
    content = []
    for view, tag in zip(VIEWS, ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")):
        content.append({"text": tag})
        for i, image in enumerate(images[view]):
            content.append({"text": f"frame: {i}"})
            content.append(image)
    content.append({"text": prompt(history, nav, instruction, actual_offsets)})
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    current_only = all(actual_offsets[view] == [0.0] for view in VIEWS)
    exact = not current_only and all(abs(actual_offsets[view][i] - T_OFFSETS[i]) < 1e-8 for view in VIEWS for i in range(4))
    return {
        "type": "chatml",
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": []}],
        "meta_info": {
            "dataset": "WOD_E2E_L2D_prerelease",
            "task_name": "instruction_conditioned_trajectory_prediction",
            "token": request_id,
            "scene_token": input_id,
            "frame_name": row.get("frame_name"),
            "cam_order": list(VIEWS),
            "instruction_sha256": digest,
            "temporal_sampling": ("current_only_3view" if current_only else
                                  "exact_0p5s" if exact else "nearest_observed_0p1s"),
            "observed_camera_offsets_s": actual_offsets,
            "allowed_temporal_jitter_s": allowed_jitter,
        },
        "trajectory": {
            "hist_traj_10hz": history,
            "hist_vel_10hz": velocity,
            "hist_acc_10hz": acceleration,
            "ego_status": {
                "ego_velocity": velocity[-1],
                "ego_acceleration": acceleration[-1],
                "driving_command": driving_command,
            },
            "nav_command": nav,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-temporal-jitter", type=float, default=0.0,
                        help="maximum absolute deviation from nominal camera times, in seconds")
    parser.add_argument("--skip-incomplete", action="store_true",
                        help="prepare only compatible public requests; write a per-request audit")
    parser.add_argument("--allow-current-only", action="store_true",
                        help="permit genuine current images only, as a separate fallback protocol")
    parser.add_argument("--current-only-only", action="store_true",
                        help="prepare only current-only fallback requests, skipping native 4-frame inputs")
    parser.add_argument("--audit", type=Path,
                        help="per-request status JSONL; required with --skip-incomplete")
    args = parser.parse_args()
    if not 0.0 <= args.allow_temporal_jitter <= 0.1:
        raise ValueError("--allow-temporal-jitter must be between 0 and 0.1 seconds")
    if args.skip_incomplete and args.audit is None:
        raise ValueError("--audit is required with --skip-incomplete")
    if args.current_only_only and not (args.allow_current_only and args.skip_incomplete):
        raise ValueError("--current-only-only requires --allow-current-only and --skip-incomplete")
    by_id = {}
    for row in load_jsonl(args.inputs):
        identifier = row.get("input_id")
        if not identifier or identifier in by_id:
            raise ValueError(f"duplicate/missing input_id: {identifier}")
        by_id[identifier] = row
    requests = load_jsonl(args.workload)
    seen = set()
    audit = []
    prepared = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with staged.open("w", encoding="utf-8") as handle:
            for request in requests:
                identifier = request.get("request_id")
                if identifier in seen:
                    raise ValueError(f"duplicate request_id: {identifier}")
                seen.add(identifier)
                input_id = request.get("input_id")
                if input_id not in by_id:
                    raise ValueError(f"request {identifier}: unknown input_id {input_id}")
                if args.current_only_only and by_id[input_id].get("history_complete"):
                    audit.append({"request_id": identifier, "input_id": input_id,
                                  "status": "skipped", "reason": "native_history_available"})
                    continue
                try:
                    record = scene_record(by_id[input_id], request, args.image_root,
                                          args.official_repo, args.allow_temporal_jitter,
                                          args.allow_current_only)
                except ValueError as exc:
                    if not args.skip_incomplete:
                        raise
                    audit.append({"request_id": identifier, "input_id": input_id,
                                  "status": "skipped", "reason": str(exc)})
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                audit.append({"request_id": identifier, "input_id": input_id,
                              "status": "prepared", "reason": None,
                              "temporal_sampling": record["meta_info"]["temporal_sampling"]})
                prepared += 1
        staged.replace(args.output)
    finally:
        staged.unlink(missing_ok=True)
    if args.audit is not None:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit),
                              encoding="utf-8")
    print(f"Prepared {prepared}/{len(requests)} Qwen-Drive scenes: {args.output}")
    print("Only public inputs were read; no trajectories or scores from private targets were used.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError) as exc:
        sys.exit(f"Qwen-Drive input incompatibility: {exc}")
