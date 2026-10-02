#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${L2D_PYTHON:-python3}" -m l2d_benchmark.cli setup --workspace "$root" "$@"
