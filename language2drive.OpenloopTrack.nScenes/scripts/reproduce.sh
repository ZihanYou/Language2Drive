#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRACK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
L2D_PYTHON=${L2D_PYTHON:-"${L2D_ENV_PREFIX:-$TRACK_ROOT/.conda}/bin/python"}
cd "$TRACK_ROOT"
"$L2D_PYTHON" scripts/download_assets.py
"$L2D_PYTHON" scripts/reproduce_paper.py --check-paper "$@"
