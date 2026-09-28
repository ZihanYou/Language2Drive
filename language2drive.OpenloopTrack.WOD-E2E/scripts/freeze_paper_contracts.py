"""Freeze the WOD-E2E P0 evaluation contracts before reading model outputs.

The resulting directory is a private evaluator asset. It contains reference
derived targets and must be kept separate from public model inputs.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from score_p0_kinematic_proxy import build_contract, rows, sha256


PROFILE = 'wod_e2e_p0_kinematic_proxy_v0.1.0'


def freeze(release, output, expected_pcus=None):
    release = Path(release)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Contract directory exists: {output}')
    manifest_path = release / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    registry_path = release / 'private/candidates.jsonl'
    workload_path = release / 'public/workload.jsonl'
    for path in (registry_path, workload_path):
        name = str(path.relative_to(release))
        if sha256(path) != manifest['file_sha256'][name]:
            raise ValueError(f'Release hash mismatch: {name}')
    candidates = list(rows(registry_path))
    workload = list(rows(workload_path))
    known_requests = {item['request_id'] for item in workload}
    if len(known_requests) != len(workload):
        raise ValueError('Duplicate release request ID')
    if len(candidates) != manifest['candidate_pairs']:
        raise ValueError('Candidate count differs from release manifest')

    contracts = []
    exclusions = Counter()
    families = Counter()
    seen_candidates = set()
    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        if candidate_id in seen_candidates:
            raise ValueError(f'Duplicate candidate ID: {candidate_id}')
        seen_candidates.add(candidate_id)
        if candidate['status'] != 'DRAFT_P0_UNREVIEWED':
            exclusions['no_p0_request'] += 1
            continue
        contract, reason = build_contract(candidate)
        if reason:
            exclusions[reason] += 1
            continue
        request_ids = candidate['request_ids']
        if set(request_ids) != {'a', 'b'} or any(
            request_ids[arm] not in known_requests for arm in ('a', 'b')
        ):
            raise ValueError(f'Invalid paired requests: {candidate_id}')
        contracts.append({
            'candidate_id': candidate_id,
            'expression_id': 'P0',
            'request_ids': request_ids,
            'contract': contract,
        })
        families[contract['family']] += 1
    if expected_pcus is not None and len(contracts) != expected_pcus:
        raise ValueError(f'Expected {expected_pcus} PCUs, found {len(contracts)}')

    output.mkdir(parents=True)
    contract_path = output / 'contracts.jsonl'
    with contract_path.open('w', encoding='utf-8') as stream:
        for item in contracts:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + '\n')
    frozen = {
        'profile': PROFILE,
        'source_release_manifest_sha256': sha256(manifest_path),
        'source_registry_sha256': sha256(registry_path),
        'source_workload_sha256': sha256(workload_path),
        'contracts_file': contract_path.name,
        'contracts_sha256': sha256(contract_path),
        'expression_ids': ['P0'],
        'pcus': len(contracts),
        'requests': 2 * len(contracts),
        'candidate_pairs': len(candidates),
        'family_counts': dict(families),
        'qualification_exclusions': dict(exclusions),
        'contract_construction': {
            'STOP_GO': 'q_stop is the rated stop endpoint',
            'SPEED_CHOICE': 'require n-s>=1 m/s; low=s+0.25*(n-s); high=s+0.75*(n-s)',
            'reference_check': 'each witness passes own binary predicate and fails opposite',
        },
    }
    (output / 'manifest.json').write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-pcus', type=int)
    args = parser.parse_args()
    result = freeze(args.release_dir, args.output_dir, args.expected_pcus)
    print(json.dumps({'profile': result['profile'], 'pcus': result['pcus'],
                      'requests': result['requests'], 'family_counts': result['family_counts']},
                     indent=2))


if __name__ == '__main__':
    main()
