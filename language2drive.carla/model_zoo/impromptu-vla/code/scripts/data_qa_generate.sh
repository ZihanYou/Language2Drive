#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/.."
PY_SCRIPT="${PROJECT_ROOT}/scripts/data_qa_generate.py"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PY_SCRIPT not found"
    exit 1
fi

python "$PY_SCRIPT"