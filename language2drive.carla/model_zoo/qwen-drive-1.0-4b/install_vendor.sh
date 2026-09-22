#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${L2D_PYTHON:-python}"
VENDOR="${ROOT}/vendor"
WHEELS="${ROOT}/downloads"
TMP_ROOT="${ROOT}/.pip-tmp"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
mkdir -p "${VENDOR}" "${WHEELS}" "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export PIP_NO_CACHE_DIR=1

# Qwen-Drive's released implementation requires Torch >=2.8. Keep the CUDA
# runtime and all Python packages project-local; simlingo-py311 is untouched.
shopt -s nullglob
core_wheels=(
  "${WHEELS}"/torch-2.8.0+cu128-*.whl
  "${WHEELS}"/torchvision-0.23.0+cu128-*.whl
  "${WHEELS}"/torchaudio-2.8.0+cu128-*.whl
  "${WHEELS}"/triton-3.4.0-*.whl
  "${WHEELS}"/nvidia_*.whl
)
if (( ${#core_wheels[@]} < 18 )); then
  "${PYTHON}" -m pip download --dest "${WHEELS}" --find-links "${WHEELS}" \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128'
  core_wheels=(
    "${WHEELS}"/torch-2.8.0+cu128-*.whl
    "${WHEELS}"/torchvision-0.23.0+cu128-*.whl
    "${WHEELS}"/torchaudio-2.8.0+cu128-*.whl
    "${WHEELS}"/triton-3.4.0-*.whl
    "${WHEELS}"/nvidia_*.whl
  )
fi
if (( ${#core_wheels[@]} < 18 )); then
  printf 'Incomplete local Torch/CUDA wheel set (%d, expected at least 18)\n' \
    "${#core_wheels[@]}" >&2
  exit 2
fi
"${PYTHON}" -m pip install --target "${VENDOR}" --upgrade --no-deps \
  "${core_wheels[@]}"

"${PYTHON}" -m pip install --target "${VENDOR}" --upgrade --no-deps \
  'transformers==5.14.0' \
  'huggingface-hub==1.5.0' \
  'tokenizers==0.22.2' \
  'safetensors==0.8.0' \
  'regex==2026.8.31' \
  'flash-linear-attention==0.5.1' \
  'fla-core==0.5.1' \
  'einops==0.8.1'

# NumPy and other pure-Python utilities are read from simlingo-py311. Its
# existing NumPy 1.26.4 preserves the CARLA ABI; nothing is installed there.

printf 'Qwen-Drive vendor ready at %s\n' "${VENDOR}"
