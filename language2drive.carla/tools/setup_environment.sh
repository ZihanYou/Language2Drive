#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${L2D_PYTHON:-python}
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
"$PYTHON" -m pip install 'torch==2.2.0+cu121' 'torchvision==0.17.0+cu121' --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
WHEEL="$ROOT/carla_runtime/PythonAPI/carla/dist/carla-0.9.16-cp311-cp311-linux_x86_64.whl"
if [[ ! -f $WHEEL ]]; then echo 'Download the carla resource group first.' >&2; exit 2; fi
"$PYTHON" -m pip install "$WHEEL"
"$PYTHON" -m pip install -e "$ROOT[test]"
"$ROOT/model_zoo/install_shared_vendor.sh"
