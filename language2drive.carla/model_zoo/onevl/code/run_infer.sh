#!/bin/bash
# ============================================================================
# OneVL Multi-GPU (optional multi-node) inference launcher
#
# Usage:
#   # Single-node, auto-detect GPUs:
#   bash run_infer.sh
#
#   # Multi-node (set MLP_WORKER_NUM, MLP_ROLE_INDEX, etc. via launcher):
#   bash run_infer.sh
#
# All paths below should be adjusted to your environment.
# ============================================================================
set -e

# ---- Python interpreter (override with PYTHON=/path/to/python if needed) ----
PYTHON=${PYTHON:-python3}

# ---- Paths (edit these) ----
MODEL_PATH=${MODEL_PATH:?  "Set MODEL_PATH to the OneVL checkpoint directory"}
TEST_SET_PATH=${TEST_SET_PATH:?  "Set TEST_SET_PATH to the test JSON/JSONL file"}
IMAGE_BASE_PATH=${IMAGE_BASE_PATH:-""}
OUTPUT_PATH=${OUTPUT_PATH:-"${MODEL_PATH}/infer_results/onevl_merged.json"}

# ---- OneVL latent-token hyperparameters ----
NUM_LATENT=${NUM_LATENT:-2}
NUM_LATENT_VIS=${NUM_LATENT_VIS:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
ANSWER_PREFIX=${ANSWER_PREFIX:-""}
PREFIX_K=${PREFIX_K:-0}

# ---- Aux text decoder (set DECODER_EXPLAIN=true to enable) ----
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
AUX_VISUAL_CONDITION=${AUX_VISUAL_CONDITION:-true}
C_THOUGHT=${C_THOUGHT:-2}
MAX_EXPLAIN_TOKENS=${MAX_EXPLAIN_TOKENS:-1024}

# ---- Aux visual decoder (set VISUAL_DECODER_EXPLAIN=true to enable) ----
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-false}
VISUAL_AUX_VISUAL_CONDITION=${VISUAL_AUX_VISUAL_CONDITION:-true}
C_THOUGHT_VISUAL=${C_THOUGHT_VISUAL:-4}
MAX_VISUAL_TOKENS=${MAX_VISUAL_TOKENS:-2560}

# ---- Locate inference script (same directory as this shell script) ----
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INFER_SCRIPT="${SCRIPT_DIR}/infer_onevl.py"

# ---- Build extra CLI flags ----
EXTRA_FLAGS=""
if [ "${DECODER_EXPLAIN}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --decoder_explain --c_thought ${C_THOUGHT} --max_explain_tokens ${MAX_EXPLAIN_TOKENS}"
    if [ "${AUX_VISUAL_CONDITION}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --aux_visual_condition"
    fi
fi
if [ "${VISUAL_DECODER_EXPLAIN}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --visual_decoder_explain --c_thought_visual ${C_THOUGHT_VISUAL} --max_visual_tokens ${MAX_VISUAL_TOKENS}"
    if [ "${VISUAL_AUX_VISUAL_CONDITION}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --visual_aux_visual_condition"
    fi
fi

IMAGE_BASE_FLAG=""
if [ -n "${IMAGE_BASE_PATH}" ]; then
    IMAGE_BASE_FLAG="--image_base_path ${IMAGE_BASE_PATH}"
fi

echo "=== OneVL Inference Configuration ==="
echo "  MODEL_PATH:               ${MODEL_PATH}"
echo "  TEST_SET_PATH:            ${TEST_SET_PATH}"
echo "  IMAGE_BASE_PATH:          ${IMAGE_BASE_PATH}"
echo "  OUTPUT_PATH:              ${OUTPUT_PATH}"
echo "  NUM_LATENT / VIS:         ${NUM_LATENT} / ${NUM_LATENT_VIS}"
echo "  DECODER_EXPLAIN:          ${DECODER_EXPLAIN}"
echo "  VISUAL_DECODER_EXPLAIN:   ${VISUAL_DECODER_EXPLAIN}"
echo "  ANSWER_PREFIX:            ${ANSWER_PREFIX}"
echo "======================================"

# ============================================================================
# Multi-node / multi-GPU orchestration
# ============================================================================
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
RUN_ID=${RUN_ID:-${MLP_JOB_ID:-${SLURM_JOB_ID:-$$}}}

OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
mkdir -p "${OUTPUT_DIR}"

if [ "${NNODES}" -gt 1 ] && [ "${RUN_ID}" = "$$" ]; then
    INFER_RUN_ID_FILE="${OUTPUT_DIR}/._infer_multinode_run_id"
    if [ "${NODE_RANK}" -eq 0 ]; then
        RUN_ID=$(date +%s)
        echo "${RUN_ID}" > "${INFER_RUN_ID_FILE}"
        sync
        echo "Rank 0: set RUN_ID=${RUN_ID}"
    else
        echo "[Node ${NODE_RANK}] Waiting for RUN_ID from rank 0 ..."
        while [ ! -f "${INFER_RUN_ID_FILE}" ]; do sleep 2; done
        RUN_ID=$(cat "${INFER_RUN_ID_FILE}")
        echo "[Node ${NODE_RANK}] Got RUN_ID=${RUN_ID}"
    fi
fi

# ---- Detect GPUs ----
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "[WARN] No GPUs detected, falling back to 1 worker on CPU"
    NUM_GPUS=1
fi
echo "[Node ${NODE_RANK}] Detected ${NUM_GPUS} GPUs"

TOTAL_WORKERS=$((NNODES * NUM_GPUS))
MY_SHARD_START=$((NODE_RANK * NUM_GPUS))
MY_SHARD_END=$(((NODE_RANK + 1) * NUM_GPUS))

SPLIT_DIR="${OUTPUT_DIR}/_splits_${RUN_ID}"
BARRIER_DIR="${SPLIT_DIR}"
DONE_FILE="${BARRIER_DIR}/done.${NODE_RANK}"
SPLIT_DONE_FILE="${BARRIER_DIR}/split_done"

# ---- Split data (rank 0 only) ----
if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${SPLIT_DIR}"
    echo "Splitting test set into ${TOTAL_WORKERS} shards ..."
    $PYTHON -c "
import json, math
path = '${TEST_SET_PATH}'
if path.endswith('.jsonl'):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
else:
    with open(path, 'r') as f:
        data = json.load(f)
n = ${TOTAL_WORKERS}
shard_size = math.ceil(len(data) / n) if n else 0
for i in range(n):
    shard = data[i*shard_size : (i+1)*shard_size]
    if not shard:
        continue
    with open(f'${SPLIT_DIR}/shard_{i}.json', 'w') as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f'  Shard {i}: {len(shard)} samples')
"
    touch "${SPLIT_DONE_FILE}"
else
    echo "[Node ${NODE_RANK}] Waiting for split ..."
    while [ ! -f "${SPLIT_DONE_FILE}" ]; do sleep 2; done
fi

# ---- Launch per-GPU inference ----
echo "[Node ${NODE_RANK}] Launching shards ${MY_SHARD_START}..$((MY_SHARD_END-1)) ..."
PIDS=()
LOCAL_GPU=0
for SHARD_ID in $(seq ${MY_SHARD_START} $((MY_SHARD_END - 1))); do
    SHARD_INPUT="${SPLIT_DIR}/shard_${SHARD_ID}.json"
    SHARD_OUTPUT="${SPLIT_DIR}/predict_${SHARD_ID}.json"

    if [ ! -f "${SHARD_INPUT}" ]; then
        LOCAL_GPU=$((LOCAL_GPU + 1))
        continue
    fi

    echo "  Shard ${SHARD_ID} → cuda:${LOCAL_GPU}"
    $PYTHON "${INFER_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --test_set_path "${SHARD_INPUT}" \
        --output_path "${SHARD_OUTPUT}" \
        --device "cuda:${LOCAL_GPU}" \
        --num_latent ${NUM_LATENT} \
        --num_latent_vis ${NUM_LATENT_VIS} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --answer_prefix "${ANSWER_PREFIX}" \
        --prefix_k ${PREFIX_K} \
        ${IMAGE_BASE_FLAG} \
        ${EXTRA_FLAGS} &

    PIDS+=($!)
    LOCAL_GPU=$((LOCAL_GPU + 1))
done

FAIL=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || FAIL=1
done
if [ ${FAIL} -ne 0 ]; then
    echo "[Node ${NODE_RANK}] ERROR: one or more processes failed"
    exit 1
fi

touch "${DONE_FILE}"
echo "[Node ${NODE_RANK}] Inference done."

# ---- Merge (rank 0 only) ----
if [ "${NODE_RANK}" -ne 0 ]; then
    exit 0
fi

echo "Waiting for all nodes ..."
for r in $(seq 0 $((NNODES - 1))); do
    while [ ! -f "${BARRIER_DIR}/done.${r}" ]; do sleep 3; done
done

echo "Merging results ..."
$PYTHON -c "
import json, glob
shards = sorted(
    glob.glob('${SPLIT_DIR}/predict_*.json'),
    key=lambda p: int(p.rsplit('_', 1)[1].replace('.json', '')))
merged = []
for path in shards:
    with open(path) as f:
        merged.extend(json.load(f))
with open('${OUTPUT_PATH}', 'w') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)
print(f'Merged {len(merged)} samples → ${OUTPUT_PATH}')
"

rm -rf "${SPLIT_DIR}"
echo "=== Done. Output: ${OUTPUT_PATH} ==="
