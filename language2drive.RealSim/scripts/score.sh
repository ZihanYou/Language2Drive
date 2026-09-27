#!/usr/bin/env sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
python=${L2D_SCORE_PYTHON:-"$project_dir/.venv/bin/python"}
if [ ! -x "$python" ]; then
    echo 'Run sh scripts/setup.sh first, or set L2D_SCORE_PYTHON to your evaluator Python.' >&2
    exit 1
fi
exec "$python" -m language2drive_realsim.paper_metrics "$@"
