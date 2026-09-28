#!/usr/bin/env python3
"""Build OneVL's official JSON input using only frozen public fields."""
import argparse
import json
from pathlib import Path

from common import acceleration_history, ego_history, image_path, iter_queries, velocity_history


def f2(values):
    return "[" + ", ".join(f"{float(x):.2f}" for x in values) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for q in iter_queries():
        history = ego_history(q.public_input)[:3]
        velocity = velocity_history(q.public_input)[-1]
        acceleration = acceleration_history(q.public_input)[-1]
        history_text = "[" + ", ".join(f2(p) for p in history) + "]"
        prompt = (
            "<image> is the front view. "
            f"Command: {q.instruction} "
            f"Velocity: {f2(velocity)}. Acceleration: {f2(acceleration)}. "
            f"Historical trajectory: {history_text}. "
            "Predict the 4-second future trajectory at 0.5-second intervals. "
            "Output exactly 8 points in the form [x, y, heading]."
        )
        rows.append({
            "id": q.key,
            "pcu_id": q.pcu_id,
            "rewrite": q.rewrite,
            "arm": q.arm,
            "images": [str(image_path(q.public_input, 3, "CAM_F0"))],
            "messages": [{"role": "user", "content": prompt}],
            "GT": "",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(json.dumps({"samples": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
