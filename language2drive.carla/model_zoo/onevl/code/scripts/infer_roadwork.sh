#!/bin/bash
# OneVL inference on Roadwork test set
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# ---- Configurable ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-false}
OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/output/roadwork/roadwork_results.json"}

# ---- Fixed ----
MODEL_PATH=""
TEST_SET_PATH=${ROOT_DIR}/test_data/roadwork_test.json
IMAGE_BASE_PATH=""
PREFIX_K=5
ANSWER_PREFIX="[["

export MODEL_PATH TEST_SET_PATH IMAGE_BASE_PATH OUTPUT_PATH PREFIX_K
export DECODER_EXPLAIN VISUAL_DECODER_EXPLAIN

echo "=== Roadwork Inference ==="
echo "  DECODER_EXPLAIN:        ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN: ${VISUAL_DECODER_EXPLAIN}"
echo "  OUTPUT_PATH:            ${OUTPUT_PATH}"
exec bash "${ROOT_DIR}/run_infer.sh"
