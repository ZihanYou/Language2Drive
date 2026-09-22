#!/usr/bin/env python3
"""Download and verify Language2Drive CARLA resources from a pinned HF revision."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def extract(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith('.zst'):
        import zstandard
        with archive.open('rb') as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as bundle:
                    bundle.extractall(destination, filter='data')
        return
    with tarfile.open(archive, 'r:gz') as bundle:
        # Python's data filter also rejects links which escape destination and
        # ignores archived uid/gid information on extraction.
        bundle.extractall(destination, filter='data')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', default=['benchmark'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--endpoint', default=os.getenv('HF_ENDPOINT', 'https://hf-mirror.com'))
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.runtime/downloads')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'resources/manifest.json').read_text())
    available = {r['group']: r for r in spec['archives']}
    if args.list:
        for group, item in available.items():
            print(f"{group:32s} {item['size'] / 2**30:8.3f} GiB")
        return
    selected = list(available) if args.all else args.groups
    unknown = set(selected) - available.keys()
    if unknown:
        parser.error('Unknown resource group(s): ' + ', '.join(sorted(unknown)))
    if not spec.get('revision') or len(spec['revision']) != 40:
        parser.error('The resource manifest must contain a published immutable revision.')
    from huggingface_hub import hf_hub_download
    for group in selected:
        item = available[group]
        filename = 'language2drive.carla/' + item['file']
        kwargs = dict(repo_id=spec['repo_id'], repo_type='dataset', filename=filename,
                      revision=spec['revision'], cache_dir=str(args.cache_dir), token=False)
        try:
            path = Path(hf_hub_download(**kwargs, endpoint=args.endpoint))
        except Exception:
            if args.endpoint.rstrip('/') == 'https://huggingface.co':
                raise
            path = Path(hf_hub_download(**kwargs, endpoint='https://huggingface.co'))
        if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
            raise SystemExit(f'Checksum mismatch: {group}; remove the cached file and retry.')
        print(f'Verified {group}', flush=True)
        if not args.verify_only:
            needed = item.get('unpacked_size', item['size'] * 3) + 4 * 2**30
            if shutil.disk_usage(ROOT).free < needed:
                raise SystemExit(f'Need at least {needed / 2**30:.1f} GiB free to extract {group}.')
            extract(path, ROOT / item['extract_to'])
            print(f'Installed {group}', flush=True)

if __name__ == '__main__':
    main()
