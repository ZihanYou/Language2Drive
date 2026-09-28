"""Verify the locally built WOD-E2E release and optional frozen contracts."""

import argparse
import json
from pathlib import Path

from score_p0_kinematic_proxy import rows, sha256


def verify(release, contracts_dir=None):
    release = Path(release).resolve()
    manifest = json.loads((release / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['file_sha256'].items():
        path = (release / relative).resolve()
        if not path.is_relative_to(release) or not path.is_file():
            raise ValueError(f'Missing or unsafe release file: {relative}')
        if sha256(path) != expected:
            raise ValueError(f'Release file hash mismatch: {relative}')

    inputs = list(rows(release / 'public/inputs.jsonl'))
    workload = list(rows(release / 'public/workload.jsonl'))
    registry = list(rows(release / 'private/candidates.jsonl'))
    images = list(rows(release / 'public/image_manifest.jsonl'))
    input_ids = {item['input_id'] for item in inputs}
    request_ids = {item['request_id'] for item in workload}
    if len(input_ids) != len(inputs) or len(request_ids) != len(workload):
        raise ValueError('Duplicate public input or request ID')
    if len(inputs) != manifest['anchor_scenes'] or len(registry) != manifest['candidate_pairs']:
        raise ValueError('Source counts differ from release manifest')
    if len(workload) != manifest['english_p0_requests'] or len(images) != manifest['image_files']:
        raise ValueError('Request or image counts differ from release manifest')
    if any(item['input_id'] not in input_ids for item in workload):
        raise ValueError('Workload refers to an unknown input')

    image_hashes = {}
    public_root = (release / 'public').resolve()
    for item in images:
        relative = item['path']
        path = (public_root / relative).resolve()
        if not path.is_relative_to(public_root) or not path.is_file():
            raise ValueError(f'Missing or unsafe image: {relative}')
        if relative in image_hashes:
            raise ValueError(f'Duplicate image path: {relative}')
        if sha256(path) != item['sha256']:
            raise ValueError(f'Image hash mismatch: {relative}')
        if item['input_id'] not in input_ids:
            raise ValueError(f'Image refers to an unknown input: {relative}')
        image_hashes[relative] = item['sha256']
    for item in inputs:
        views = list(item['images'].values())
        history = [frame for frames in item['camera_history'].values() for frame in frames]
        for frame in views + history:
            if image_hashes.get(frame['path']) != frame['sha256']:
                raise ValueError(f'Input references an absent or changed image: {frame["path"]}')
    for item in registry:
        if item['input_id'] not in input_ids:
            raise ValueError(f'Candidate refers to unknown input: {item["candidate_id"]}')
        requested = set(item['request_ids'].values())
        if not requested.issubset(request_ids):
            raise ValueError(f'Candidate refers to unknown request: {item["candidate_id"]}')
        if bool(requested) and len(requested) != 2:
            raise ValueError(f'Candidate does not have two instruction arms: {item["candidate_id"]}')

    result = {'release_version': manifest['version'],
              'anchors': len(inputs), 'candidate_pairs': len(registry),
              'requests': len(workload), 'verified_images': len(images),
              'release_manifest_sha256': sha256(release / 'manifest.json')}
    if contracts_dir is not None:
        from evaluate_paper import load_contracts
        frozen, contracts, _, _ = load_contracts(release, contracts_dir)
        result['paper_contracts'] = len(contracts)
        result['contract_manifest_sha256'] = sha256(Path(contracts_dir) / 'manifest.json')
        if len(contracts) != frozen['pcus']:
            raise ValueError('Frozen contract count mismatch')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--contracts-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.release_dir, args.contracts_dir), indent=2))


if __name__ == '__main__':
    main()
