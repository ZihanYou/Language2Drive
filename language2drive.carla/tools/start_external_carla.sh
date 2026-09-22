#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${PROJECT_ROOT}/carla_runtime"
RUNTIME_ROOT="${PROJECT_ROOT}/.runtime"
CARLA_PORT=2000
GPU_RANK=0
BOOTSTRAP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap) BOOTSTRAP=1; shift ;;
    --port)
      CARLA_PORT="${2:?--port requires a value}"
      shift 2
      ;;
    --gpu-rank)
      GPU_RANK="${2:?--gpu-rank requires a value}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if [[ ! "${CARLA_PORT}" =~ ^[0-9]+$ ]] || (( CARLA_PORT < 1 || CARLA_PORT > 65535 )); then
  echo "CARLA port must be in 1..65535: ${CARLA_PORT}" >&2
  exit 2
fi
if [[ ! "${GPU_RANK}" =~ ^[0-9]+$ ]]; then
  echo "GPU rank must be a non-negative integer: ${GPU_RANK}" >&2
  exit 2
fi
if (( CARLA_PORT == 2000 )); then
  SERVER_RUNTIME_ROOT="${RUNTIME_ROOT}/carla-server-release-v1"
else
  SERVER_RUNTIME_ROOT="${RUNTIME_ROOT}/carla-server-release-v1-port-${CARLA_PORT}"
fi
if (( BOOTSTRAP )); then SERVER_RUNTIME_ROOT="${RUNTIME_ROOT}/carla-bootstrap-port-${CARLA_PORT}"; fi
SERVER_PATH="/usr/local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/bin:/sbin:/bin"
SHIPPING_BINARY="${CARLA_ROOT}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
PYTHON="${L2D_PYTHON:-python}"

if [[ "$(id -u)" != "0" ]]; then
  echo "CARLA release launcher must run as root" >&2
  exit 2
fi

if [[ -e "${SERVER_RUNTIME_ROOT}" ]]; then
  if [[ -L "${SERVER_RUNTIME_ROOT}" || ! -d "${SERVER_RUNTIME_ROOT}" ]]; then
    echo "CARLA release runtime path must be a real directory: ${SERVER_RUNTIME_ROOT}" >&2
    exit 2
  fi
  if [[ -n "$(find "${SERVER_RUNTIME_ROOT}" -mindepth 1 -print -quit)" ]]; then
    echo "CARLA release runtime directory must be absent or empty before launch: ${SERVER_RUNTIME_ROOT}" >&2
    exit 2
  fi
fi

for directory in \
  "${SERVER_RUNTIME_ROOT}/tmp" \
  "${SERVER_RUNTIME_ROOT}/xdg-cache" \
  "${SERVER_RUNTIME_ROOT}/cuda-cache" \
  "${SERVER_RUNTIME_ROOT}/carla-cache" \
  "${SERVER_RUNTIME_ROOT}/user-dir" \
  "${SERVER_RUNTIME_ROOT}/home"; do
  mkdir -p "${directory}"
done
chmod 0755 "${SERVER_RUNTIME_ROOT}"

if (( ! BOOTSTRAP )); then
"${PYTHON}" - \
  "${SERVER_RUNTIME_ROOT}" \
  "${PROJECT_ROOT}/.runtime/carla_runtime_assets.current.json" \
  "${PROJECT_ROOT}/.runtime/carla_runtime_assets.current.provenance.json" \
  "${PROJECT_ROOT}/tools/start_external_carla.sh" \
  "${SHIPPING_BINARY}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

runtime_root, manifest, provenance, launcher, shipping = map(Path, sys.argv[1:])
for path in (manifest, provenance, launcher, shipping):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"release launch input is not a regular file: {path}")
hz = int(os.sysconf("SC_CLK_TCK"))
uptime = float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])
payload = {
    "schema_version": "language2drive.carla_server_launch_receipt.v1",
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip(),
    "prelaunch_monotonic_ticks": int(uptime * hz),
    "clock_ticks_per_second": hz,
    "server_runtime_root": str(runtime_root.resolve()),
    "runtime_asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "runtime_asset_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
    "shipping_binary_sha256": hashlib.sha256(shipping.read_bytes()).hexdigest(),
}
temporary = runtime_root / ".launch-receipt.json.tmp"
target = runtime_root / "launch-receipt.json"
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(temporary, 0o444)
os.replace(temporary, target)
directory = os.open(runtime_root, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
fi

cd "${CARLA_ROOT}"

chown root:root "${SERVER_RUNTIME_ROOT}"
chmod 0755 "${SERVER_RUNTIME_ROOT}"
if (( ! BOOTSTRAP )); then
  chown root:root "${SERVER_RUNTIME_ROOT}/launch-receipt.json"
  chmod 0444 "${SERVER_RUNTIME_ROOT}/launch-receipt.json"
fi
for directory in \
  "${SERVER_RUNTIME_ROOT}/tmp" \
  "${SERVER_RUNTIME_ROOT}/xdg-cache" \
  "${SERVER_RUNTIME_ROOT}/cuda-cache" \
  "${SERVER_RUNTIME_ROOT}/carla-cache" \
  "${SERVER_RUNTIME_ROOT}/user-dir" \
  "${SERVER_RUNTIME_ROOT}/home"; do
  chown -R sim:sim "${directory}"
  chmod 0700 "${directory}"
done

CARLA_ARGUMENTS=(
  CarlaUE4 -RenderOffScreen -nosound "-carla-port=${CARLA_PORT}"
)
if (( CARLA_PORT != 2000 || GPU_RANK != 0 )); then
  CARLA_ARGUMENTS+=("-graphicsadapter=${GPU_RANK}")
fi
CARLA_ARGUMENTS+=(-quality-level=Epic)

exec runuser -u sim -- env -i \
  PATH="${SERVER_PATH}" \
  HOME="${SERVER_RUNTIME_ROOT}/home" \
  TMPDIR="${SERVER_RUNTIME_ROOT}/tmp" \
  XDG_CACHE_HOME="${SERVER_RUNTIME_ROOT}/xdg-cache" \
  CUDA_CACHE_PATH="${SERVER_RUNTIME_ROOT}/cuda-cache" \
  CARLA_CACHE_DIR="${SERVER_RUNTIME_ROOT}/carla-cache" \
  UE-SharedDataCachePath="${SERVER_RUNTIME_ROOT}/user-dir" \
  USER="sim" LOGNAME="sim" \
  LANG="C.UTF-8" LC_ALL="C.UTF-8" \
  "${SHIPPING_BINARY}" "${CARLA_ARGUMENTS[@]}"
