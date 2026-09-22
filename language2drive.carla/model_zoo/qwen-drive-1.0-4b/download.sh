#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "${L2D_PYTHON:-python}" "$ROOT/tools/download_models.py" qwen-drive-1.0-4b "$@"
