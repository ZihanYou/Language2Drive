#!/usr/bin/env python3
"""Verify the seven released model adapters, sources, and local inference assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ('alpamayo1.5', 'simlingo', 'lmdrive', 'onevl', 'impromptu-vla', 'qwen-drive-1.0-4b', 'senna')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def shards(path):
    index = path / 'model.safetensors.index.json'
    if not index.is_file():
        index = path / 'pytorch_model.bin.index.json'
    if not index.is_file():
        return []
    return [path / name for name in sorted(set(json.loads(index.read_text())['weight_map'].values()))]

def audit(model, code_only=False):
    base = ROOT / model
    config = base / ('adapter_config.py' if model == 'lmdrive' else 'adapter_config.json')
    missing = []
    for path in [base / 'agent.py', config, base / 'SOURCE.json']:
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
    source = json.loads((ROOT / 'SOURCES.json').read_text())['sources'][model]
    for relative, expected in source.get('files_sha256', {}).items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            missing.append(relative)
    if not code_only:
        required = [ROOT / 'language2drive_adapters/vendor/huggingface_hub']
        if model == 'lmdrive':
            required += [base / 'checkpoints' / p for p in (
                'llava-v1.5-7b/config.json', 'llava-v1.5-7b/tokenizer.model',
                'lmdrive_llava.pth', 'vision_encoder.pth.tar',
                'bert-base-uncased/vocab.txt', 'bert-base-uncased/model.safetensors')]
            required += shards(base / 'checkpoints/llava-v1.5-7b')
            required += [base / 'vendor' / p for p in ('decord', 'iopath', 'webdataset')]
        else:
            cfg = json.loads(config.read_text())
            required += [base / p for p in cfg.get('required_files', []) + cfg.get('required_dirs', [])]
            if model == 'simlingo':
                required += [base / cfg['checkpoint'], base / cfg['internvl_backbone'] / 'model.safetensors']
            if model == 'alpamayo1.5':
                required += [base / cfg['checkpoint_dir'] / 'model.safetensors.index.json',
                             base / cfg['processor_dir'] / 'tokenizer.json', base / 'vendor/transformers']
            for directory in (base / 'checkpoints').glob('**/'):
                required += shards(directory)
        for path in required:
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                missing.append(str(path.relative_to(ROOT)))
    return {'model': model, 'status': 'ready' if not missing else 'missing_files',
            'source_commit': source['commit'], 'missing_or_modified': sorted(set(missing))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--code-only', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    rows = [audit(m, args.code_only) for m in ([args.model] if args.model else MODELS)]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(row['model'] + ': ' + row['status'])
            for name in row['missing_or_modified']:
                print('  ' + name)
    return 0 if all(row['status'] == 'ready' for row in rows) else 2

if __name__ == '__main__':
    raise SystemExit(main())
