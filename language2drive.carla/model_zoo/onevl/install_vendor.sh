#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${L2D_PYTHON:-python}
mkdir -p "$ROOT/vendor"
if [ -d "$ROOT/vendor/transformers-4.57.1.dist-info" ] && \
   [ -d "$ROOT/vendor/tokenizers-0.22.1.dist-info" ]; then
  exit 0
fi
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  "$PYTHON" -m pip install --target "$ROOT/vendor" --upgrade \
    --no-deps transformers==4.57.1 tokenizers==0.22.1
