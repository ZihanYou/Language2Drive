#!/bin/sh
set -eu
track=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv=${L2D_LEGACY_SCORER_VENV:-$track/navsim_offline/envs/paper-legacy}
python_bin=${PYTHON:-python3}
if [ ! -x "$venv/bin/python" ]; then
  "$python_bin" -m venv --without-pip "$venv"
fi
if "$venv/bin/python" -c 'import shapely' >/dev/null 2>&1; then
  echo 'Legacy scoring requires an isolated environment without Shapely.' >&2
  exit 2
fi
exec "$venv/bin/python" "$track/navsim_offline/score.py" "$@"
