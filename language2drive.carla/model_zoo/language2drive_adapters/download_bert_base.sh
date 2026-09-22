#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 TARGET_DIR" >&2
  exit 2
fi

TARGET=$1
BASE_URL=${L2D_BERT_MIRROR:-https://modelscope.cn/models/google-bert/bert-base-uncased/resolve/master}
mkdir -p "$TARGET"

download_file() {
  local name=$1
  local output="$TARGET/$name"
  local partial="$output.part"
  if [ -s "$output" ]; then
    return 0
  fi
  if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u ALL_PROXY -u all_proxy \
      curl --fail --location --retry 5 --continue-at - \
      --output "$partial" "$BASE_URL/$name"; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u ALL_PROXY -u all_proxy curl --fail --location --retry 5 \
      --output "$partial" "$BASE_URL/$name"
  fi
  mv "$partial" "$output"
}

for file in config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors; do
  download_file "$file"
done

# Published file digest returned by both the upstream repository and mirror.
EXPECTED=68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
printf '%s  %s\n' "$EXPECTED" "$TARGET/model.safetensors" | sha256sum --check --status
echo "BERT checkpoint ready: $TARGET"
