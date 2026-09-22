#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${L2D_PYTHON:-python}
TARGET="$ROOT/language2drive_adapters/vendor"
mkdir -p "$TARGET"
if [ ! -d "$TARGET/huggingface_hub-0.36.0.dist-info" ]; then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    "$PYTHON" -m pip install --target "$TARGET" --upgrade --no-deps huggingface-hub==0.36.0
fi
if [ ! -d "$TARGET/hf_transfer-0.1.9.dist-info" ]; then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    "$PYTHON" -m pip install --target "$TARGET" --upgrade --no-deps hf-transfer==0.1.9
fi
