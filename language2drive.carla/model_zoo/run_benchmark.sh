#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?Usage: run_benchmark.sh MODEL small|base|large}
TIER=${2:-base}
case "$TIER" in small|base|large) ;; *) echo 'Tier must be small, base or large' >&2; exit 2;; esac
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${L2D_PYTHON:-python}
case "$MODEL" in alpamayo1.5|simlingo|lmdrive|onevl|impromptu-vla|qwen-drive-1.0-4b|senna) ;; *) echo "Unknown release model: $MODEL" >&2; exit 2;; esac
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/language2drive/scenario_runner:$ROOT/language2drive/leaderboard:$ROOT/carla_runtime/PythonAPI:$ROOT/carla_runtime/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
PY_PREFIX=$("$PYTHON" -c 'import sys; print(sys.prefix)')
export LD_LIBRARY_PATH="$PY_PREFIX/lib/python3.11/site-packages/pillow.libs:$PY_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ARGS=(-m language2drive.cli run --agent "$MODEL" --track standardized
  --split "${L2D_SPLIT:-test}" --tier "$TIER" --seeds "${L2D_SEEDS:-1}"
  --results "${L2D_RESULTS:-$ROOT/results/${MODEL}_${TIER}}"
  --plan-output "${L2D_PLAN:-$ROOT/generated/${MODEL}_${TIER}_plan.json}"
  --wall-timeout "${L2D_WALL_TIMEOUT:-1800}"
  --host "${L2D_HOST:-127.0.0.1}" --port "${L2D_PORT:-2000}"
  --traffic-manager-port "${L2D_TM_PORT:-8000}" --gpu-rank "${L2D_GPU_RANK:-0}")
if [[ -n ${L2D_MAX_RUNS:-} ]]; then ARGS+=(--max-runs "$L2D_MAX_RUNS"); fi
if [[ ${L2D_EXECUTE:-0} == 1 ]]; then ARGS+=(--execute); fi
if [[ ${L2D_PRINT_COMMAND:-0} == 1 ]]; then printf '%q ' "$PYTHON" "${ARGS[@]}"; printf '\n'; exit 0; fi
"$PYTHON" "$ROOT/model_zoo/preflight.py" --model "$MODEL"
exec "$PYTHON" "${ARGS[@]}"
