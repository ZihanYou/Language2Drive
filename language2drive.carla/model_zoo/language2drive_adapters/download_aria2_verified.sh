#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 URL TARGET EXPECTED_SIZE EXPECTED_SHA256" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARIA="$ROOT/vendor/aria2/usr/bin/aria2c"
ARIA_LIB="$ROOT/vendor/aria2/usr/lib/x86_64-linux-gnu"
URL=$1
TARGET=$2
EXPECTED_SIZE=$3
EXPECTED_SHA256=${4,,}
PART="$TARGET.part"
CONNECTIONS=${L2D_ARIA2_CONNECTIONS:-16}
SPLIT=${L2D_ARIA2_SPLIT:-$CONNECTIONS}
LOWEST_SPEED=${L2D_ARIA2_LOWEST_SPEED:-0}
SESSION_SECONDS=${L2D_ARIA2_SESSION_SECONDS:-0}
CDN_IP=${L2D_CDN_IP:-}
CDN_IPS=${L2D_CDN_IPS:-}

case "$EXPECTED_SIZE" in
  ''|*[!0-9]*) echo "EXPECTED_SIZE must be a positive integer" >&2; exit 2 ;;
esac
if [ "$EXPECTED_SIZE" -le 0 ]; then
  echo "EXPECTED_SIZE must be positive" >&2
  exit 2
fi
if [[ ! "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_SHA256 must contain exactly 64 hexadecimal digits" >&2
  exit 2
fi
case "$CONNECTIONS" in
  ''|*[!0-9]*) echo "L2D_ARIA2_CONNECTIONS must be an integer from 1 to 16" >&2; exit 2 ;;
esac
if [ "$CONNECTIONS" -lt 1 ] || [ "$CONNECTIONS" -gt 16 ]; then
  echo "L2D_ARIA2_CONNECTIONS must be in [1, 16]" >&2
  exit 2
fi
case "$SPLIT" in
  ''|*[!0-9]*) echo "L2D_ARIA2_SPLIT must be an integer from 1 to 64" >&2; exit 2 ;;
esac
if [ "$SPLIT" -lt 1 ] || [ "$SPLIT" -gt 64 ]; then
  echo "L2D_ARIA2_SPLIT must be in [1, 64]" >&2
  exit 2
fi
case "$SESSION_SECONDS" in
  ''|*[!0-9]*) echo "L2D_ARIA2_SESSION_SECONDS must be a non-negative integer" >&2; exit 2 ;;
esac
if [ ! -x "$ARIA" ] || [ ! -d "$ARIA_LIB" ]; then
  echo "model-local aria2 runtime is missing under $ROOT/vendor/aria2" >&2
  exit 1
fi

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

is_verified() {
  local path=$1
  [ -f "$path" ] || return 1
  [ "$(stat -c %s "$path")" = "$EXPECTED_SIZE" ] || return 1
  [ "$(file_sha256 "$path")" = "$EXPECTED_SHA256" ]
}

mkdir -p "$(dirname "$TARGET")"
if is_verified "$TARGET"; then
  echo "reuse verified $TARGET"
  exit 0
fi
if [ -e "$TARGET" ]; then
  echo "refusing to overwrite invalid existing target: $TARGET" >&2
  exit 1
fi
if is_verified "$PART"; then
  mv -- "$PART" "$TARGET"
  echo "verified $TARGET"
  exit 0
fi

# Resolve the public mirror once.  This avoids repeating the redirect for every
# range and permits pinning a measured CDN IPv4 address without touching
# /etc/hosts.  The final exact LFS SHA-256 remains the content authority.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
HEADERS=$(env -u LD_LIBRARY_PATH /usr/bin/curl -4 --fail --silent --show-error \
  --connect-timeout 15 --max-time 60 -I "$URL" | tr -d '\r')
LOCATION=$(printf '%s\n' "$HEADERS" | sed -n 's/^[Ll]ocation: //p' | head -1)
if [ -z "$LOCATION" ]; then
  echo "mirror did not return a download redirect for $URL" >&2
  exit 1
fi

DOWNLOAD_URLS=("$LOCATION")
ARIA_EXTRA=()
if [ -n "$CDN_IPS" ] && [ -n "$CDN_IP" ]; then
  echo "set only one of L2D_CDN_IP or L2D_CDN_IPS" >&2
  exit 2
fi
if [ -n "$CDN_IPS" ] || [ -n "$CDN_IP" ]; then
  case "$LOCATION" in
    https://us.aws.cdn.hf.co/*)
      DOWNLOAD_URLS=()
      for ip in ${CDN_IPS:-$CDN_IP}; do
        if [[ ! "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
          echo "invalid IPv4 address in L2D_CDN_IPS/L2D_CDN_IP: $ip" >&2
          exit 2
        fi
        DOWNLOAD_URLS+=("https://$ip/${LOCATION#https://us.aws.cdn.hf.co/}")
      done
      ARIA_EXTRA+=(--check-certificate=false --header='Host: us.aws.cdn.hf.co')
      ;;
    *)
      echo "a CDN IP was set but redirect host is not us.aws.cdn.hf.co" >&2
      exit 1
      ;;
  esac
fi

ARIA_LOG_ARGS=()
if [ -n "${L2D_ARIA2_LOG:-}" ]; then
  mkdir -p "$(dirname "$L2D_ARIA2_LOG")"
  ARIA_LOG_ARGS+=(--log="$L2D_ARIA2_LOG" --log-level=notice)
fi

ARIA_COMMAND=(
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY
    -u ALL_PROXY -u all_proxy LD_LIBRARY_PATH="$ARIA_LIB"
    "$ARIA"
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --file-allocation=none \
    --max-connection-per-server="$CONNECTIONS" \
    --split="$SPLIT" \
    --min-split-size=20M \
    --max-tries=0 \
    --retry-wait=1 \
    --connect-timeout=15 \
    --timeout=60 \
    --lowest-speed-limit="$LOWEST_SPEED" \
    --uri-selector=adaptive \
    --summary-interval=10 \
    --console-log-level=warn \
    "${ARIA_LOG_ARGS[@]}" \
    "${ARIA_EXTRA[@]}" \
    --dir="$(dirname "$PART")" \
    --out="$(basename "$PART")"
    "${DOWNLOAD_URLS[@]}"
)

ACTIVE_PID=""
stop_active_download() {
  local child=${ACTIVE_PID:-}
  local pgid
  if [ -z "$child" ] || ! kill -0 "$child" 2>/dev/null; then
    return
  fi
  pgid=$(ps -o pgid= -p "$child" | tr -d ' ')
  if [ "$pgid" = "$child" ]; then
    kill -TERM -- "-$child" 2>/dev/null || true
  else
    kill -TERM "$child" 2>/dev/null || true
  fi
  wait "$child" 2>/dev/null || true
}
trap 'stop_active_download; exit 130' INT
trap 'stop_active_download; exit 143' TERM HUP

session=0
while true; do
  session=$((session + 1))
  if [ "$SESSION_SECONDS" -eq 0 ]; then
    set +e
    "${ARIA_COMMAND[@]}" &
    ACTIVE_PID=$!
    wait "$ACTIVE_PID"
    status=$?
    ACTIVE_PID=""
    set -e
    if [ "$status" -ne 0 ]; then
      echo "aria2 failed with status $status" >&2
      exit "$status"
    fi
    break
  fi
  echo "aria2 connection session $session (up to ${SESSION_SECONDS}s)"
  set +e
  timeout --signal=INT --kill-after=10s "${SESSION_SECONDS}s" \
    "${ARIA_COMMAND[@]}" &
  ACTIVE_PID=$!
  wait "$ACTIVE_PID"
  status=$?
  ACTIVE_PID=""
  set -e
  case "$status" in
    0) break ;;
    124) sleep 1 ;;
    *) echo "aria2 failed with status $status" >&2; exit "$status" ;;
  esac
done
trap - INT TERM HUP

if ! is_verified "$PART"; then
  actual_size=$(stat -c %s "$PART" 2>/dev/null || echo missing)
  echo "download verification failed for $PART (size=$actual_size)" >&2
  exit 1
fi
mv -- "$PART" "$TARGET"
echo "verified $TARGET"
