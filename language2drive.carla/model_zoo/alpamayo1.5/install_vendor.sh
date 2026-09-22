#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"${L2D_PYTHON:-python}" -m pip install --target "$ROOT/vendor" --upgrade --no-deps \
  transformers==4.57.1 tokenizers==0.22.1 huggingface-hub==0.36.0 einops==0.8.1
