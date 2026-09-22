#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The reviewed sources are included in Git; verify them without overwriting adapters.
exec "${L2D_PYTHON:-python}" "$ROOT/preflight.py" --code-only
