#!/bin/sh
set -eu
track=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$track"
model=${1:-all}
qwen_python=${L2D_QWEN_PYTHON:-model_eval/envs/qwen/bin/python}
legacy_python=${L2D_LEGACY_PYTHON:-model_eval/envs/legacy/bin/python}
onevl_python=${L2D_ONEVL_PYTHON:-model_eval/envs/onevl/bin/python}
run_qwen() {
  PYTHONPATH="$track/model_eval:$track/model_eval/repos/qwen_drive" \
    "$qwen_python" model_eval/run_qwen_drive.py
}
run_autovla() {
  PYTHONPATH="$track/model_eval:$track/model_eval/repos/autovla" \
    "$legacy_python" model_eval/run_autovla.py
}
run_drivema() {
  PYTHONPATH="$track/model_eval" \
    "$qwen_python" model_eval/run_drivema.py
}
run_recogdrive() {
  PYTHONPATH="$track/model_eval:$track/model_eval/repos/recogdrive" \
    "$legacy_python" model_eval/run_recogdrive.py
}
run_onevl() {
  mkdir -p model_eval/data/onevl_chunks model_eval/runs/onevl
  PYTHONPATH="$track/model_eval" "$onevl_python" \
    model_eval/build_onevl_dataset.py model_eval/data/onevl_eval.json
  "$onevl_python" model_eval/prepare_onevl_chunks.py \
    model_eval/data/onevl_eval.json model_eval/data/onevl_chunks
  sh model_eval/run_onevl_chunks.sh
  PYTHONPATH="$track/model_eval" "$onevl_python" \
    model_eval/consolidate.py model_eval/runs/onevl
}
case "$model" in
  qwen_drive) run_qwen ;;
  autovla) run_autovla ;;
  drivema) run_drivema ;;
  recogdrive) run_recogdrive ;;
  onevl) run_onevl ;;
  all) run_qwen; run_autovla; run_drivema; run_recogdrive; run_onevl ;;
  *) echo "usage: $0 {qwen_drive|autovla|drivema|recogdrive|onevl|all}" >&2; exit 2 ;;
esac
