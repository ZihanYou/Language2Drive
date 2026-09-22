#!/usr/bin/env python3
"""Fetch pinned Qwen-Drive runtime sources through the GitHub blob API."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path


REVISION = "28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"
REPOSITORY = "QwenLM/Qwen-Drive-1.0"
ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "code"
RUNTIME_PREFIX = "src/qwen_drive/"
EXTRA_FILES = {"LICENSE", "README.md", "pyproject.toml"}


def request_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "language2drive-qwen-drive-adapter",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    for name in (
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"
    ):
        os.environ.pop(name, None)
    tree = request_json(
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
    )
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated repository tree")
    selected = [
        item
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and (
            str(item.get("path", "")).startswith(RUNTIME_PREFIX)
            or item.get("path") in EXTRA_FILES
        )
    ]
    if not selected:
        raise RuntimeError("no Qwen-Drive runtime files found at the pinned revision")
    for item in sorted(selected, key=lambda value: str(value["path"])):
        path = str(item["path"])
        blob = request_json(
            f"https://api.github.com/repos/{REPOSITORY}/git/blobs/{item['sha']}"
        )
        if blob.get("encoding") != "base64":
            raise RuntimeError(f"unexpected encoding for {path}: {blob.get('encoding')}")
        payload = base64.b64decode(str(blob["content"]), validate=False)
        target = DESTINATION / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f"fetched {path} ({len(payload)} bytes)", flush=True)
    required = {
        "src/qwen_drive/__init__.py",
        "src/qwen_drive/modeling_qwen_drive.py",
        "src/qwen_drive/planning_expert.py",
        "src/qwen_drive/scene.py",
    }
    missing = [path for path in sorted(required) if not (DESTINATION / path).is_file()]
    if missing:
        raise RuntimeError(f"incomplete runtime source checkout: {missing}")
    (DESTINATION / ".source_revision").write_text(REVISION + "\n", encoding="utf-8")
    print(f"Qwen-Drive runtime source ready at {DESTINATION} ({REVISION})")


if __name__ == "__main__":
    main()
