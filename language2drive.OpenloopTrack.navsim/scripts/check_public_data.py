#!/usr/bin/env python3
"""Validate the released public inputs and optionally the referenced sensors."""
import argparse
import json
import os
from pathlib import Path


TRACK = Path(__file__).resolve().parents[1]
PUBLIC = TRACK / "data/public"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((PUBLIC / "manifest.json").read_text())
    ids = manifest["pcu_ids"]
    assert len(ids) == 38 and len(set(ids)) == 38
    sensor_root = Path(os.environ.get(
        "L2D_NAVSIM_SENSOR_ROOT",
        str(TRACK / "assets/openscene-v1.1/sensor_blobs/test")))
    missing = []
    instructions = 0
    sensor_paths = set()
    for pcu_id in ids:
        item = json.loads((PUBLIC / "public_inputs" / (pcu_id + ".json")).read_text())
        assert item["pcu_id"] == pcu_id
        assert len(item["instructions"]) == 2
        assert len(item["paraphrases"]) == 2
        assert all(len(arm) == 3 for arm in item["paraphrases"])
        instructions += 8
        for frame in item["camera_paths_history"]:
            sensor_paths.update(frame.values())
        sensor_paths.update(item["lidar_paths_history"])
    if args.sensors:
        missing = [path for path in sorted(sensor_paths) if not (sensor_root / path).is_file()]
    result = {"pcus": len(ids), "instruction_strings": instructions,
              "queries_per_model": 304, "unique_sensor_files": len(sensor_paths),
              "missing_sensors": len(missing) if args.sensors else "not checked"}
    print(json.dumps(result, indent=2))
    if missing:
        print("first missing paths:", missing[:5])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
