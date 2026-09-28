#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
environment="$root/.conda/wod-e2e-paper"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required" >&2
  exit 2
fi

if [ ! -x "$environment/bin/python" ]; then
  conda env create --prefix "$environment" --file "$root/environment.yml"
fi

exec conda run --no-capture-output --prefix "$environment" \
  python "$root/scripts/download_assets.py" "$@"
