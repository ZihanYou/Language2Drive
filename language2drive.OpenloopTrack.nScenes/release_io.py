"""Verify and load a frozen Language2Drive Offline benchmark release."""

import hashlib
import json
from pathlib import Path

from build_catalog import canonical_hash


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def load_release(directory, include_private=False, split='all'):
    directory = Path(directory)
    manifest = json.loads((directory / 'release_manifest.json').read_text())
    if manifest.get('dataset_version') != 'v1.0-test' or manifest.get('status') != 'approved':
        raise ValueError('Not an approved nuScenes test release')
    public_path = directory / 'public_inputs.jsonl'
    if sha256(public_path) != manifest['public_sha256']:
        raise ValueError('Public release hash mismatch')
    records = jsonl(public_path)
    ids = [row['pcu_id'] for row in records]
    if len(ids) != len(set(ids)) or len(ids) != manifest['approved_pcus']:
        raise ValueError('Public PCU count or uniqueness mismatch')
    dev = set(manifest['dev_pcus'])
    heldout = set(manifest['heldout_pcus'])
    if dev & heldout or dev | heldout != set(ids):
        raise ValueError('Release split does not partition PCUs')
    for row in records:
        if row.get('status') != 'approved' or row.get('dataset_version') != 'v1.0-test':
            raise ValueError('Unapproved PCU in release')
        if 'private_sidecar' in row or 'contracts' in row:
            raise ValueError('Private data found in public release')
        shared = dict(row['public_input'])
        expected_hash = shared.pop('public_input_hash')
        if canonical_hash(shared) != expected_hash:
            raise ValueError('Public input hash mismatch: ' + row['pcu_id'])
        if set(row['arms']) != {'arm_0', 'arm_1'}:
            raise ValueError('Incorrect arm set: ' + row['pcu_id'])
    if include_private:
        private_path = directory / 'private_sidecars.jsonl'
        if sha256(private_path) != manifest['private_sha256']:
            raise ValueError('Private release hash mismatch')
        private_rows = jsonl(private_path)
        private = {row['pcu_id']: row for row in private_rows}
        if len(private) != len(private_rows) or set(private) != set(ids):
            raise ValueError('Private PCUs do not match public PCUs')
        for row in records:
            sidecar = private[row['pcu_id']]
            if sidecar.get('status') != 'approved' or sidecar.get('family') != row['family']:
                raise ValueError('Private metadata mismatch: ' + row['pcu_id'])
            row['private_sidecar'] = sidecar['private_sidecar']
            row['contracts'] = sidecar['contracts']
    if split not in ('all', 'dev', 'heldout'):
        raise ValueError('Unknown split: ' + split)
    selected = set(ids) if split == 'all' else (dev if split == 'dev' else heldout)
    records = [row for row in records if row['pcu_id'] in selected]
    return manifest, records
