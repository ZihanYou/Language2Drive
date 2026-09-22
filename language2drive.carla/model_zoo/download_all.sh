#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for model in alpamayo1.5 simlingo lmdrive onevl impromptu-vla qwen-drive-1.0-4b senna; do
  "$ROOT/$model/download.sh"
done
