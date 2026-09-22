#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/root/miniconda3/envs/simlingo-py311/bin/python"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export LD_LIBRARY_PATH="/root/miniconda3/envs/simlingo-py311/lib/python3.11/site-packages/pillow.libs:/root/miniconda3/envs/simlingo-py311/lib"

exec "${PYTHON}" -m language2drive.cli run \
  --agent-path "${PROJECT_ROOT}/team_code/agent_internvl.py" \
  --agent-config "${PROJECT_ROOT}/model_zoo/internvl_model/runtime_config/config.yaml" \
  --track standardized \
  --split dev \
  --seeds 1 \
  --max-runs 1 \
  --plan-output "${PROJECT_ROOT}/generated/internvl_debug_plan.json" \
  "$@"
