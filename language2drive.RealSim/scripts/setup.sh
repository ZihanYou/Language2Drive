#!/usr/bin/env sh
# One-command, isolated deployment. No sudo or global Python changes.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${L2D_PYTHON:-python3}" "$script_dir/deploy.py" "$@"
