#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${L2D_PYTHON:-python}
mkdir -p "$ROOT/vendor"
"$PYTHON" -m pip install --target "$ROOT/vendor" --upgrade --no-deps \
  decord==0.6.0 iopath==0.1.10 fvcore==0.1.5.post20221221 \
  webdataset==0.2.100 braceexpand==0.1.7 portalocker==3.1.1 easydict==1.13 \
  fairscale==0.4.4 diffusers==0.29.2 sentencepiece==0.2.1 protobuf==4.25.8
"$PYTHON" -m pip install --target "$ROOT/vendor" --upgrade --no-deps \
  torch-scatter==2.1.2 \
  -f 'https://data.pyg.org/whl/torch-2.2.0+cu121.html'
