#!/usr/bin/env python3
"""Freeze a byte-reproducible vehicle local-BBox observation from an L2D trace."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from language2drive.io import atomic_json, sha256_file, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    evidence_dir = arguments.evidence_dir.resolve() if arguments.evidence_dir else None
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for name in ("trace.jsonl", "run_meta.json", "runner_command.json", "capture_capabilities.json"):
            destination = evidence_dir / name
            source = run_dir / name
            if destination.exists() and sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"immutable evidence differs from current source: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)
        run_dir = evidence_dir
    trace_path = run_dir / "trace.jsonl"
    boxes: list[dict] = []
    frames: list[int] = []
    reference_z: list[float] = []
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            ego = value["ego"]
            box = ego["bounding_box"]
            if set(box) != {"location", "rotation", "extent"}:
                raise RuntimeError("trace ego bounding_box keys are not canonical")
            if set(box["location"]) != {"x", "y", "z"} or set(box["extent"]) != {"x", "y", "z"}:
                raise RuntimeError("trace ego bounding_box vectors are incomplete")
            if set(box["rotation"]) != {"pitch", "yaw", "roll"}:
                raise RuntimeError("trace ego bounding_box rotation is incomplete")
            if not all(math.isfinite(float(number)) for body in box.values() for number in body.values()):
                raise RuntimeError("trace ego bounding_box is not finite")
            boxes.append(box)
            frames.append(int(value["frame"]))
            reference_z.append(float(ego["location"]["z"]))
    if not boxes or any(box != boxes[0] for box in boxes[1:]):
        raise RuntimeError("vehicle local bounding box is absent or inconsistent across trace")
    if any(float(box["extent"][axis]) <= 0 for axis in ("x", "y", "z") for box in boxes):
        raise RuntimeError("vehicle local bounding box extent must be positive")

    sources = {
        name: {"path": str(run_dir / name), "sha256": sha256_file(run_dir / name)}
        for name in ("trace.jsonl", "run_meta.json", "runner_command.json", "capture_capabilities.json")
    }
    payload = {
        "schema_version": "language2drive.vehicle_local_bbox_evidence.v1",
        "vehicle_blueprint": "vehicle.lincoln.mkz_2020",
        "local_bounding_box": boxes[0],
        # The registry and every manoeuvre proof bind the canonical local OBB
        # itself.  Blueprint identity is an adjacent, independently frozen
        # field rather than part of this digest.
        "local_bounding_box_sha256": stable_hash(boxes[0]),
        "observation": {
            "sample_count": len(boxes),
            "consistent_sample_count": len(boxes),
            "first_frame": min(frames),
            "last_frame": max(frames),
            "ego_reference_z_min": min(reference_z),
            "ego_reference_z_max": max(reference_z),
            "route_entry_id": 528,
            "pcu_id": "C-025",
            "arm_id": "slot_a",
            "paraphrase": 0,
            "seed": 1,
        },
        "source_files": sources,
        "scope_statement": (
            "This immutable asset freezes the CARLA vehicle.lincoln.mkz_2020 local "
            "bounding box observed identically in every recorded frame of one versioned "
            "privileged rollout. It is geometry evidence, not a claim that the failed "
            "rollout was a successful parking witness."
        ),
    }
    atomic_json(arguments.output.resolve(), payload)


if __name__ == "__main__":
    main()
