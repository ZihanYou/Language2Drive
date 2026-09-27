#!/usr/bin/env sh
# Build an isolated AlpaSim + RealSim Python environment; no sudo required.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${L2D_PYTHON:-python3}" "$script_dir/setup_runtime.py" "$@"
