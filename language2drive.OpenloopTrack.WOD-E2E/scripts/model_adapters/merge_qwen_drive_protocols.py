#!/usr/bin/env python3
"""Merge disjoint Qwen native and current-only predictions in public workload order.

The sidecar records the image-input protocol of every predicted request. This script
reads public workload, scene files and prediction files only; it does not read targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_ids(path: Path, protocol: str) -> set[str]:
    identifiers = []
    for row in rows(path):
        meta = row["meta_info"]
        if protocol == "current_only_3view" and meta["temporal_sampling"] != protocol:
            raise ValueError(f"wrong temporal sampling in {path}: {meta['temporal_sampling']}")
        if protocol == "native_4frame" and meta["temporal_sampling"] not in (
            "exact_0p5s", "nearest_observed_0p1s"
        ):
            raise ValueError(f"wrong temporal sampling in {path}: {meta['temporal_sampling']}")
        identifiers.append(meta["token"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate scene token in {path}")
    return set(identifiers)


def predictions(path: Path, allowed: set[str]) -> dict[str, dict]:
    values = {}
    for row in rows(path):
        identifier = row.get("request_id")
        if identifier not in allowed:
            raise ValueError(f"{path}: unexpected request_id {identifier}")
        if identifier in values:
            raise ValueError(f"{path}: duplicate request_id {identifier}")
        if not isinstance(row.get("trajectory"), list) or len(row["trajectory"]) != 20:
            raise ValueError(f"{path}: invalid 20-point trajectory for {identifier}")
        values[identifier] = row
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--native-scenes", type=Path, required=True)
    parser.add_argument("--current-only-scenes", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--current-only-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    workload = [row["request_id"] for row in rows(args.workload)]
    if len(set(workload)) != len(workload):
        raise ValueError("duplicate public workload request_id")
    native_ids = scene_ids(args.native_scenes, "native_4frame")
    current_ids = scene_ids(args.current_only_scenes, "current_only_3view")
    if native_ids & current_ids:
        raise ValueError("native and current-only scenes overlap")
    if native_ids | current_ids != set(workload):
        raise ValueError("scene protocols do not partition the public workload")
    native = predictions(args.native_predictions, native_ids)
    current = predictions(args.current_only_predictions, current_ids)
    merged = {**native, **current}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(args.output.suffix + ".tmp")
    staged.write_text("".join(json.dumps(merged[key], allow_nan=False) + "\n"
                              for key in workload if key in merged), encoding="utf-8")
    staged.replace(args.output)
    report = {
        "workload_requests": len(workload),
        "native_4frame_requests": len(native_ids),
        "current_only_3view_requests": len(current_ids),
        "native_valid_predictions": len(native),
        "current_only_valid_predictions": len(current),
        "missing_predictions": len(workload) - len(merged),
        "protocols_are_not_directly_comparable": True,
        "public_workload_sha256": digest(args.workload),
        "native_scenes_sha256": digest(args.native_scenes),
        "current_only_scenes_sha256": digest(args.current_only_scenes),
        "native_predictions_sha256": digest(args.native_predictions),
        "current_only_predictions_sha256": digest(args.current_only_predictions),
        "merged_predictions_sha256": digest(args.output),
        "label_access": "none",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(merged)}/{len(workload)} public requests: "
          f"{len(native)} native, {len(current)} current-only")


if __name__ == "__main__":
    main()
