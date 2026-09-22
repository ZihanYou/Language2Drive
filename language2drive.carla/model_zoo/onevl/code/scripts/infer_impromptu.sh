#!/bin/bash
# OneVL inference on Impromptu test set
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# ---- Configurable ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-false}
OUTPUT_PATH=${OUTPUT_PATH:-"${ROOT_DIR}/output/impromptu/impromptu_results.json"}

# ---- Fixed ----
MODEL_PATH=""
TEST_SET_PATH=${ROOT_DIR}/test_data/impromptu_test.jsonl
IMAGE_BASE_PATH=""
ANSWER_PREFIX=""

export MODEL_PATH TEST_SET_PATH IMAGE_BASE_PATH OUTPUT_PATH
export DECODER_EXPLAIN VISUAL_DECODER_EXPLAIN

echo "=== Impromptu Inference ==="
echo "  DECODER_EXPLAIN:        ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN: ${VISUAL_DECODER_EXPLAIN}"
echo "  OUTPUT_PATH:            ${OUTPUT_PATH}"
exec bash "${ROOT_DIR}/run_infer.sh"
