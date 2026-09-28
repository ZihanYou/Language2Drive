#!/usr/bin/env python3
"""CPU-only public-input compatibility audit for DriveMA and Fast-dDrive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from drivema import VIEWS, read_jsonl, safe_image


ROUTE_COMMANDS = {"GO_STRAIGHT", "GO_LEFT", "GO_RIGHT", "UNKNOWN"}
NOMINAL_CAMERA_OFFSETS = (-1.5, -1.0, -0.5, 0.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_input(row: dict, image_root: Path) -> dict:
    errors = []
    camera_history_issues = []
    route = row.get("route_command")
    if route not in ROUTE_COMMANDS:
        errors.append("unsupported_or_missing_route_command")
    past = row.get("past_states")
    if not isinstance(past, dict):
        errors.append("missing_past_states")
    else:
        for key in ("x", "y", "vx", "vy", "ax", "ay"):
            series = past.get(key)
            if not isinstance(series, list) or len(series) != 16:
                errors.append(f"past_states_{key}_not_16_points")
                continue
            try:
                finite = all(math.isfinite(float(number)) for number in series)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                errors.append(f"past_states_{key}_nonfinite")
    for view in VIEWS:
        item = row.get("images", {}).get(view)
        if item is None:
            errors.append(f"missing_{view}_image")
            continue
        try:
            path = safe_image(image_root, item)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    errors.append(f"invalid_{view}_dimensions")
        except (OSError, ValueError, TypeError):
            errors.append(f"unreadable_{view}_image")
        temporal = row.get("camera_history", {}).get(view, [])
        if len(temporal) != len(NOMINAL_CAMERA_OFFSETS):
            camera_history_issues.append(f"{view}_camera_history_not_4_frames")
            continue
        for item, nominal in zip(temporal, NOMINAL_CAMERA_OFFSETS):
            try:
                measured_nominal = float(item["nominal_offset_s"])
                actual = float(item["offset_s"])
                if abs(measured_nominal - nominal) > 1e-6:
                    camera_history_issues.append(f"{view}_nominal_time_mismatch")
                if abs(actual - nominal) > 0.100001:
                    camera_history_issues.append(f"{view}_time_jitter_over_0p1s")
                with Image.open(safe_image(image_root, item)) as image:
                    image.verify()
            except (KeyError, OSError, ValueError, TypeError):
                camera_history_issues.append(f"{view}_unreadable_history_frame")
    camera_complete = not camera_history_issues
    return {"errors": errors,
            "front_camera_history_complete": camera_complete,
            "front_camera_history_issues": camera_history_issues}


def audit(inputs_file: Path, workload_file: Path, image_root: Path) -> dict:
    inputs = read_jsonl(inputs_file)
    requests = read_jsonl(workload_file)
    by_id = {row["input_id"]: row for row in inputs}
    if len(by_id) != len(inputs):
        raise ValueError("duplicate input_id in public inputs")
    input_checks = {input_id: inspect_input(row, image_root)
                    for input_id, row in by_id.items()}
    records = []
    for request in requests:
        input_id = request.get("input_id")
        base = input_checks.get(input_id)
        errors = list(base["errors"]) if base else ["unknown_input_id"]
        instruction = request.get("instruction")
        if request.get("instruction_language") != "en":
            errors.append("instruction_language_not_en")
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append("empty_or_nonstring_instruction")
        runnable = not errors
        # Neither released model documents passenger-command conditioning.
        # The adapters pass the instruction in English, but that adaptation
        # remains experimental even when all sensor inputs are available.
        model_result = {
            "adapter_runnable": runnable,
            "native_sensor_input_contract_satisfied": runnable,
            "instruction_interface": "experimental_passenger_prompt",
            "rejection_reasons": errors,
        }
        records.append({
            "request_id": request.get("request_id"),
            "input_id": input_id,
            "frame_name": by_id[input_id]["frame_name"] if input_id in by_id else None,
            "front_camera_history_complete":
                base["front_camera_history_complete"] if base else False,
            "front_camera_history_issues":
                base["front_camera_history_issues"] if base else ["unknown_input_id"],
            "drivema_2b": model_result,
            "fast_ddrive": model_result.copy(),
        })
    if len({row["request_id"] for row in records}) != len(records):
        raise ValueError("duplicate request_id in public workload")
    summary = {}
    for model in ("drivema_2b", "fast_ddrive"):
        summary[model] = {
            "requests": len(records),
            "adapter_runnable": sum(row[model]["adapter_runnable"] for row in records),
            "adapter_rejected": sum(not row[model]["adapter_runnable"] for row in records),
            "native_sensor_input_contract_satisfied": sum(
                row[model]["native_sensor_input_contract_satisfied"] for row in records),
        }
    summary["front_camera_history"] = {
        "requests_complete": sum(row["front_camera_history_complete"] for row in records),
        "requests_incomplete": sum(not row["front_camera_history_complete"] for row in records),
        "relevance_to_these_models": "auxiliary_only; both released checkpoints use current front views",
    }
    return {
        "schema_version": "public_model_input_audit_v1",
        "audit_mode": "cpu_only_no_model_load",
        "inputs_path": str(inputs_file),
        "inputs_sha256": sha256(inputs_file),
        "workload_path": str(workload_file),
        "workload_sha256": sha256(workload_file),
        "input_count": len(inputs),
        "summary": summary,
        "requests": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.workload, args.image_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
