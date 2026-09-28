#!/usr/bin/env bash
# Run only after the user has assigned this GPU. No private targets are read here.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: QWEN_LIMIT=2 $0 <user-assigned GPU index>" >&2
  exit 2
fi
gpu_id="$1"
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "GPU index must be a nonnegative integer" >&2
  exit 2
fi
gpu_processes="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_processes" ]]; then
  echo "GPU $gpu_id has active compute processes; select an idle GPU" >&2
  exit 3
fi
free_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 22000 )); then
  echo "GPU $gpu_id has ${free_mib:-unknown} MiB free; Qwen-Drive requires a nearly free 24 GB card" >&2
  exit 3
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code="${QWEN_REPO:-$root/models/cache/Qwen-Drive-1.0-code}"
weights="${QWEN_WEIGHTS:-$root/models/cache/Qwen-Drive-1.0-4B}"
run_dir="${QWEN_RUN_DIR:-$root/models/runs/qwen_drive}"
pilot="${QWEN_PUBLIC_DIR:-$root/prerelease/pilot-v0.1.0/public}"
python_bin="${QWEN_PYTHON:-python}"
scenes="$run_dir/pilot_scenes.jsonl"
cache_dir="${QWEN_RUNTIME_CACHE:-$root/.cache/qwen_drive_runtime}"
mkdir -p "$cache_dir/triton" "$cache_dir/xdg" "$cache_dir/tmp"

[[ -f "$scenes" ]] || { echo "Missing $scenes; run qwen_drive.py first" >&2; exit 4; }
[[ -f "$weights/model.safetensors" ]] || { echo "Missing VLM weights" >&2; exit 4; }
[[ -f "$weights/planner-sft/model.safetensors" ]] || { echo "Missing SFT weights" >&2; exit 4; }
[[ -f "$weights/planner-rl/model.safetensors" ]] || { echo "Missing RL weights" >&2; exit 4; }

limit_args=()
if [[ -n "${QWEN_LIMIT:-}" ]]; then
  if [[ ! "$QWEN_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "QWEN_LIMIT must be a positive integer" >&2
    exit 2
  fi
  limit_args=(--limit "$QWEN_LIMIT")
fi

for planner in sft rl; do
  raw="$run_dir/${planner}_predictions.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$code/src" \
    TRITON_CACHE_DIR="$cache_dir/triton" XDG_CACHE_HOME="$cache_dir/xdg" \
    TMPDIR="$cache_dir/tmp" TORCHINDUCTOR_CACHE_DIR="$cache_dir/torchinductor" \
    "$python_bin" "$code/scripts/run_planning.py" \
    --model "$weights" --planner "$weights/planner-$planner" \
    --scenes "$scenes" --image-root "$pilot" --output "$raw" \
    --mode reasoning_planning --num-samples 1 --seed 42 \
    --num-workers 2 --attn-implementation sdpa --device cuda:0 \
    "${limit_args[@]}"
  "$python_bin" "$root/scripts/model_adapters/qwen_drive_postprocess.py" \
    --raw "$raw" --workload "$pilot/workload.jsonl" \
    --output "$run_dir/${planner}_predictions_4hz.jsonl" \
    --audit "$run_dir/${planner}_conversion_audit.jsonl" \
    --report "$run_dir/${planner}_conversion_report.json"
done
