#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${L2D_PYTHON:-python}
mkdir -p "$ROOT/vendor"
if [ -d "$ROOT/vendor/sentencepiece-0.2.1.dist-info" ] && \
   [ -d "$ROOT/vendor/protobuf-4.25.8.dist-info" ]; then
  exit 0
fi
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u ALL_PROXY -u all_proxy \
  "$PYTHON" -m pip install --target "$ROOT/vendor" --upgrade --no-deps \
  sentencepiece==0.2.1 protobuf==4.25.8
