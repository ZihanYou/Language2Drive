#!/bin/sh
set -eu
track=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON:-python3}
venv="$track/model_eval/envs/base"
if [ ! -x "$venv/bin/python" ]; then
  "$python_bin" -m venv "$venv"
fi
if ! "$venv/bin/python" -c 'import huggingface_hub' >/dev/null 2>&1; then
  env -u http_proxy -u https_proxy -u all_proxy -u ftp_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u FTP_PROXY \
      "$venv/bin/python" -m pip install -i https://mirrors.aliyun.com/pypi/simple/ \
      'huggingface-hub==0.36.0'
fi
exec "$venv/bin/python" "$track/scripts/download_assets.py" --all "$@"
