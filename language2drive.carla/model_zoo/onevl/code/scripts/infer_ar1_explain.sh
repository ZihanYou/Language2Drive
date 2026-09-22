#!/bin/bash
# OneVL inference on NAVSIM test set
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# ---- Configurable ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-true}
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-true}
OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/output/ar1_explain/ar1_results_explain.json"}

# ---- Fixed ----
MODEL_PATH=""
TEST_SET_PATH=${ROOT_DIR}/test_data/ar1_test.jsonl
IMAGE_BASE_PATH=""
ANSWER_PREFIX="[["

export MODEL_PATH TEST_SET_PATH IMAGE_BASE_PATH OUTPUT_PATH ANSWER_PREFIX
export DECODER_EXPLAIN VISUAL_DECODER_EXPLAIN

echo "=== AR1 Inference ==="
echo "  DECODER_EXPLAIN:        ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN: ${VISUAL_DECODER_EXPLAIN}"
echo "  OUTPUT_PATH:            ${OUTPUT_PATH}"
exec bash "${ROOT_DIR}/run_infer.sh"
