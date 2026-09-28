#!/usr/bin/env python3
"""Convert resumable JSONL outputs to the canonical and four-wording submissions."""
import argparse
from pathlib import Path

from common import consolidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    canonical, rewrites = consolidate(args.run_dir / "progress.jsonl", args.run_dir)
    print(canonical)
    print(rewrites)


if __name__ == "__main__":
    main()
