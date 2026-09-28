#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$base"
rank=${1:-0}
world=${2:-1}
onevl_python=${L2D_ONEVL_PYTHON:-$base/model_eval/envs/onevl/bin/python}
chunks="$base/model_eval/data/onevl_chunks"
raw="$base/model_eval/runs/onevl/raw_chunks"
mkdir -p "$raw" "$base/model_eval/tmp"
i=0
for source in "$chunks"/chunk_*.json; do
  if [ $((i % world)) -ne "$rank" ]; then
    i=$((i + 1))
    continue
  fi
  i=$((i + 1))
  name=$(basename "$source")
  output="$raw/$name"
  if [ -s "$output" ] && "$onevl_python" -c \
    'import json,sys; raise SystemExit(0 if len(json.load(open(sys.argv[1]))) == len(json.load(open(sys.argv[2]))) else 1)' \
    "$source" "$output"; then
    echo "skip $name"
    continue
  fi
  "$onevl_python" "$base/model_eval/repos/onevl/infer_onevl.py" \
    --model_path "$base/model_eval/weights/onevl-navsim" \
    --test_set_path "$source" --output_path "$output" --device cuda:0 \
    --answer_prefix '[' --max_new_tokens 256
done
PYTHONPATH="$base/model_eval" "$onevl_python" \
  "$base/model_eval/collect_onevl.py" "$chunks" "$raw" \
  "$base/model_eval/runs/onevl/progress.jsonl"
