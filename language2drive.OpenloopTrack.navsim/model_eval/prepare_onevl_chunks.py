#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--size", type=int, default=16)
    args = p.parse_args()
    rows = json.loads(args.dataset.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(rows), args.size)):
        path = args.output_dir / f"chunk_{index:03d}.json"
        path.write_text(json.dumps(rows[start:start + args.size], indent=2))
    print(json.dumps({"samples": len(rows), "chunks": (len(rows) + args.size - 1) // args.size}))


if __name__ == "__main__":
    main()
