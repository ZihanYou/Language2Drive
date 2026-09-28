#!/usr/bin/env python3
"""Resumable checkpoint downloader with proxy variables intentionally disabled."""
import argparse
import json
import os
from pathlib import Path

for key in list(os.environ):
    if key.lower().endswith("proxy"):
        os.environ.pop(key, None)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("--revision", required=True,
                        help="Frozen model revision from docs/EXPERIMENT_PROTOCOL.md")
    parser.add_argument("--ignore", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ["HF_ENDPOINT"]
    api = HfApi(endpoint=endpoint)
    info = api.model_info(args.repo_id, revision=args.revision,
                          files_metadata=True, timeout=60)
    if info.sha != args.revision:
        raise RuntimeError(f"revision mismatch: {info.sha} != {args.revision}")
    selected = [f for f in info.siblings
                if not any(Path(f.rfilename).match(pattern) for pattern in args.ignore)]
    manifest = {
        "repo_id": args.repo_id,
        "revision": info.sha,
        "source": "official Hugging Face repository via hf-mirror.com",
        "endpoint": endpoint,
        "proxy": "disabled",
        "expected_bytes": sum((f.size or 0) for f in selected),
        "ignored_patterns": args.ignore,
        "files": [{"path": f.rfilename, "bytes": f.size} for f in selected],
    }
    (args.local_dir / "download_manifest.pending.json").write_text(
        json.dumps(manifest, indent=2))
    snapshot_download(
        repo_id=args.repo_id,
        revision=info.sha,
        local_dir=args.local_dir,
        ignore_patterns=args.ignore or None,
        max_workers=args.workers,
    )
    missing = [row["path"] for row in manifest["files"]
               if not (args.local_dir / row["path"]).is_file()]
    size_mismatch = [row["path"] for row in manifest["files"]
                     if row["bytes"] is not None and (args.local_dir / row["path"]).is_file()
                     and (args.local_dir / row["path"]).stat().st_size != row["bytes"]]
    manifest["missing_files"] = missing
    manifest["size_mismatch_files"] = size_mismatch
    manifest["status"] = "complete" if not missing and not size_mismatch else "incomplete"
    (args.local_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
                      ("repo_id", "revision", "expected_bytes", "status", "missing_files")}))
    if missing or size_mismatch:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
