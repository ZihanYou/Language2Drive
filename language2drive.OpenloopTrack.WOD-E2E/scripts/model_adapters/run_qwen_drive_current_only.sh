#!/usr/bin/env bash
# Run the separately tagged 3-current-image fallback only after GPU assignment.
set -euo pipefail

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: QWEN_LIMIT=2 $0 <user-assigned GPU index>" >&2
  exit 2
fi
gpu_id="$1"
gpu_processes="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_processes" ]]; then
  echo "GPU $gpu_id has active compute processes; select an idle GPU" >&2
  exit 3
fi
free_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 22000 )); then
  echo "GPU $gpu_id has ${free_mib:-unknown} MiB free; need at least 22000 MiB" >&2
  exit 3
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code="${QWEN_REPO:-$root/models/cache/Qwen-Drive-1.0-code}"
weights="${QWEN_WEIGHTS:-$root/models/cache/Qwen-Drive-1.0-4B}"
run_dir="${QWEN_RUN_DIR:-$root/models/runs/qwen_drive}"
release="${QWEN_PUBLIC_DIR:-$root/prerelease/v0.1.0/public}"
scenes="$run_dir/current_only_unique_scenes.jsonl"
mapping="$run_dir/current_only_dedup_map.jsonl"
python_bin="${QWEN_PYTHON:-python}"
cache_dir="${QWEN_RUNTIME_CACHE:-$root/.cache/qwen_drive_runtime}"
mkdir -p "$cache_dir/triton" "$cache_dir/xdg" "$cache_dir/tmp"

[[ -s "$scenes" ]] || { echo "Missing current-only scene file: $scenes" >&2; exit 4; }
[[ -s "$mapping" ]] || { echo "Missing current-only dedup map: $mapping" >&2; exit 4; }
for part in "$weights/model.safetensors" "$weights/planner-sft/model.safetensors" "$weights/planner-rl/model.safetensors"; do
  [[ -s "$part" ]] || { echo "Missing model part: $part" >&2; exit 4; }
done
limit_args=()
if [[ -n "${QWEN_LIMIT:-}" ]]; then
  if [[ ! "$QWEN_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN_LIMIT must be a positive integer" >&2
    exit 2
  fi
  limit_args=(--limit "$QWEN_LIMIT")
fi

for planner in sft rl; do
  raw="$run_dir/current_only_${planner}_unique_predictions.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$code/src" \
    TRITON_CACHE_DIR="$cache_dir/triton" XDG_CACHE_HOME="$cache_dir/xdg" \
    TMPDIR="$cache_dir/tmp" TORCHINDUCTOR_CACHE_DIR="$cache_dir/torchinductor" \
    "$python_bin" "$code/scripts/run_planning.py" \
      --model "$weights" --planner "$weights/planner-$planner" \
      --scenes "$scenes" --image-root "$release" --output "$raw" \
      --mode reasoning_planning --num-samples 1 --seed 42 \
      --num-workers 2 --attn-implementation sdpa --device cuda:0 \
      "${limit_args[@]}"
  "$python_bin" "$root/scripts/model_adapters/qwen_drive_postprocess.py" \
    --raw "$raw" --workload "$release/workload.jsonl" \
    --output "$run_dir/current_only_${planner}_unique_predictions_4hz.jsonl" \
    --audit "$run_dir/current_only_${planner}_conversion_audit.jsonl" \
    --report "$run_dir/current_only_${planner}_conversion_report.json"
  "$python_bin" "$root/scripts/model_adapters/qwen_drive_dedup.py" fanout \
    --mapping "$mapping" --workload "$release/workload.jsonl" \
    --unique-predictions "$run_dir/current_only_${planner}_unique_predictions_4hz.jsonl" \
    --output "$run_dir/current_only_${planner}_predictions_4hz.jsonl" \
    --audit "$run_dir/current_only_${planner}_fanout_audit.jsonl" \
    --report "$run_dir/current_only_${planner}_fanout_report.json"
done
