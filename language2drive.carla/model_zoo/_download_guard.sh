#!/usr/bin/env bash
set -euo pipefail

if pgrep -af '[l]anguage2drive\.cli run.*--execute' >/dev/null && [ "${L2D_ALLOW_CONCURRENT_DOWNLOAD:-0}" != "1" ]; then
  echo "Refusing to download while a Language2Drive --execute experiment is active." >&2
  echo "Wait for it to finish, or explicitly set L2D_ALLOW_CONCURRENT_DOWNLOAD=1." >&2
  exit 75
fi

MODEL_ZOO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FREE_KB=$(df -Pk "$MODEL_ZOO_ROOT" | awk 'NR==2 {print $4}')
MIN_FREE_GB=${L2D_MIN_FREE_GB:-60}
if [ "$FREE_KB" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
  echo "Less than ${MIN_FREE_GB} GiB free; refusing a large model download." >&2
  exit 76
fi
