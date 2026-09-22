#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$ROOT/run_benchmark.sh" "${1:?Usage: run_base.sh MODEL}" base
