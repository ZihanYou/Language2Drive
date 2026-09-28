#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from common import append_progress, parse_numeric_points, submission_trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chunk_dir", type=Path)
    p.add_argument("raw_dir", type=Path)
    p.add_argument("progress", type=Path)
    args = p.parse_args()
    parsed = invalid = 0
    for source in sorted(args.chunk_dir.glob("chunk_*.json")):
        raw_path = args.raw_dir / source.name
        if not raw_path.is_file():
            continue
        inputs = json.loads(source.read_text())
        outputs = json.loads(raw_path.read_text())
        if len(inputs) != len(outputs):
            raise RuntimeError(f"length mismatch: {source} {len(inputs)} vs {len(outputs)}")
        for item, output in zip(inputs, outputs):
            row = {
                "key": item["id"], "pcu_id": item["pcu_id"],
                "rewrite": item["rewrite"], "arm": item["arm"],
                "raw_output": output.get("output_text", ""),
                "latency": output.get("latency"),
            }
            try:
                points = parse_numeric_points("[" + row["raw_output"], min_points=8)[:8]
                headings = [point[2] for point in points] if all(len(point) >= 3 for point in points) else None
                row["trajectory"] = submission_trajectory([point[:2] for point in points], headings)
                row["valid"] = True
                parsed += 1
            except Exception as exc:
                row["valid"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
                invalid += 1
            append_progress(args.progress, row)
    print(json.dumps({"parsed": parsed, "invalid": invalid}))


if __name__ == "__main__":
    main()
