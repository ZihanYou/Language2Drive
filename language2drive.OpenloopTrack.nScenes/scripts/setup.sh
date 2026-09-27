#!/bin/sh
set -eu
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
case "${1:-}" in
  ''|--with-assets) ;;
  *) echo 'Usage: sh scripts/setup.sh [--with-assets]' >&2; exit 2 ;;
esac
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRACK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
L2D_ENV_PREFIX=${L2D_ENV_PREFIX:-"$TRACK_ROOT/.conda"}
CONDA_EXE=${CONDA_EXE:-conda}
if ! command -v "$CONDA_EXE" >/dev/null 2>&1; then
  echo 'Conda is required. Install Miniconda/Miniforge, then rerun this script.' >&2
  exit 1
fi
if [ ! -x "$L2D_ENV_PREFIX/bin/python" ]; then
  "$CONDA_EXE" env create --prefix "$L2D_ENV_PREFIX" --file "$TRACK_ROOT/environment.yml"
fi
cd "$TRACK_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$L2D_ENV_PREFIX/bin/python" -m unittest discover -s tests -v
if [ "${1:-}" = '--with-assets' ]; then
  "$L2D_ENV_PREFIX/bin/python" scripts/download_assets.py
elif [ -n "${1:-}" ]; then
  echo 'Usage: sh scripts/setup.sh [--with-assets]' >&2
  exit 2
fi
printf 'Ready. Python: %s/bin/python\n' "$L2D_ENV_PREFIX"
