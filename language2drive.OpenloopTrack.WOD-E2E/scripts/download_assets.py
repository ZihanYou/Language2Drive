"""Prepare the paper's WOD-E2E source subset and optionally fetch model assets.

Waymo data must first be obtained by the user through the official data portal.
This script selects the published shard inventory without duplicating raw data,
builds local evaluation assets, freezes contracts, and verifies their hashes.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'drivema': ('zwc2003/DriveMA-2B',
                 'd5f4905d2032da943a7ae8f22d5b2b8efbe4b0f7',
                 ROOT / 'models/cache/drivema-2b'),
    'fast-ddrive': ('Efficient-Large-Model/Fast-dDrive',
                    '0fda81009f4efa58a2debbb48c0c09818e45341f',
                    ROOT / 'models/cache/fast-ddrive'),
    'qwen-drive': ('Qwen/Qwen-Drive-1.0-4B', None,
                   ROOT / 'models/cache/Qwen-Drive-1.0-4B'),
}
QWEN_CODE_COMMIT = '28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0'


def call(*args):
    subprocess.run([str(value) for value in args], cwd=ROOT, check=True)


def source_subset(source_root):
    source_root = Path(source_root).resolve()
    source_spec = json.loads((ROOT / 'config/paper_source_shards.json').read_text())
    subset = ROOT / 'assets/source_shards'
    subset.mkdir(parents=True, exist_ok=True)
    required = source_spec['shards'] + [source_spec['scenario_clusters_file']]
    for item in required:
        name = item['name']
        source = (source_root / name).resolve()
        if not source.is_file() or source.stat().st_size != item['size_bytes']:
            raise FileNotFoundError(f'Missing or unexpected source shard: {name}')
        target = subset / name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f'Shard link points to another source: {name}')
        elif target.exists():
            raise ValueError(f'Expected a shard link, found an existing file: {target}')
        else:
            target.symlink_to(source)
    return subset


def build_local(source_root):
    subset = source_subset(source_root)
    release = ROOT / 'prerelease/v0.1.0'
    contracts = ROOT / 'contracts/wod-e2e-p0-v0.1.0'
    if not release.exists():
        call(sys.executable, ROOT / 'scripts/audit_local_data.py',
             '--root', subset, '--out', ROOT / 'artifacts')
        call(sys.executable, ROOT / 'scripts/summarize_candidates.py')
        call(sys.executable, ROOT / 'scripts/make_review_materials.py')
        call(sys.executable, ROOT / 'scripts/draft_semantic_tasks.py')
        call(sys.executable, ROOT / 'scripts/build_full_prerelease.py',
             '--source-root', subset, '--expected-candidates', 563,
             '--out', release)
    if not contracts.exists():
        call(sys.executable, ROOT / 'scripts/freeze_paper_contracts.py',
             '--release-dir', release, '--expected-pcus', 241,
             '--output-dir', contracts)
    call(sys.executable, ROOT / 'scripts/verify_assets.py',
         '--release-dir', release, '--contracts-dir', contracts)
    print(f'Local evaluation assets ready: {release}')


def fetch_model(name):
    repo, revision, output = MODELS[name]
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install requirements-download.txt in this conda environment') from exc
    endpoint = 'https://hf-mirror.com'
    os.environ['HF_ENDPOINT'] = endpoint
    api = HfApi(endpoint=endpoint)
    pinned = revision or api.model_info(repo).sha
    if not pinned or len(pinned) != 40:
        raise ValueError(f'Cannot resolve a pinned revision for {name}')
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=pinned, local_dir=output,
                      endpoint=endpoint, max_workers=4)
    receipt = {'repo': repo, 'revision': pinned, 'endpoint': endpoint,
               'local_dir': str(output)}
    (output / 'download_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Downloaded {repo} at {pinned}')
    if name == 'qwen-drive':
        code = ROOT / 'models/cache/Qwen-Drive-1.0-code'
        if not code.exists():
            code.parent.mkdir(parents=True, exist_ok=True)
            call('git', 'clone', '--filter=blob:none',
                 'https://github.com/QwenLM/Qwen-Drive-1.0.git', code)
            subprocess.run(['git', '-C', str(code), 'checkout', QWEN_CODE_COMMIT], check=True)
        else:
            current = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'],
                                              text=True).strip()
            if current != QWEN_CODE_COMMIT:
                raise ValueError('Existing Qwen source is not at the pinned commit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        help='directory of WOD-E2E validation shards obtained from Waymo')
    parser.add_argument('--models', default='none',
                        help='none, all, or comma-separated drivema,fast-ddrive,qwen-drive')
    parser.add_argument('--only-models', action='store_true',
                        help='download model assets without constructing local WOD assets')
    args = parser.parse_args()
    if not args.only_models:
        if args.source_root is None:
            parser.error('Provide --source-root after obtaining WOD-E2E from https://waymo.com/open/data/e2e/')
        build_local(args.source_root)
    selected = list(MODELS) if args.models == 'all' else (
        [] if args.models == 'none' else args.models.split(','))
    if len(selected) != len(set(selected)) or any(name not in MODELS for name in selected):
        parser.error('--models must be none, all, or a unique comma-separated model list')
    for name in selected:
        fetch_model(name)


if __name__ == '__main__':
    main()
