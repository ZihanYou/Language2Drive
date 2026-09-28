"""Evaluate WOD-E2E paired instructions with frozen P0 motion contracts.

Implements the WOD-E2E four-metric profile in paper Appendix E.2/E.5.
One P0 wording is scheduled per PCU, so wording averaging is a single item.
"""

import argparse
import json
from pathlib import Path

from freeze_paper_contracts import PROFILE
from score_p0_kinematic_proxy import metric_summary, rows, score_arm, sha256


NAMES = {
    'SemSISR': 'SemSISR_P0_kinematic_proxy_0to100',
    'SemCFPS': 'SemCFPS_P0_kinematic_proxy_0to100',
    'SCS': 'SCS_P0_kinematic_proxy_0to100',
    'PCS': 'PCS_P0_kinematic_proxy_0to100',
}


def named_path(value):
    name, separator, raw_path = value.partition('=')
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError('Expected MODEL=predictions.jsonl')
    return name, Path(raw_path)


def aggregate(records):
    detail = metric_summary(records)
    if detail is None:
        return None
    return {
        'pcus': detail['pcus'],
        'requests': detail['requests'],
        'predicted': detail['predicted'],
        'valid': detail['valid'],
        'invalid': detail['invalid'],
        'exclusive_success_arms': detail['exclusive_success_arms'],
        'exclusive_success_pairs': detail['exclusive_success_pairs'],
        'metrics_0to100': {name: detail[key] for name, key in NAMES.items()},
    }


def load_contracts(release, contract_dir):
    release = Path(release)
    contract_dir = Path(contract_dir)
    release_manifest_path = release / 'manifest.json'
    release_manifest = json.loads(release_manifest_path.read_text(encoding='utf-8'))
    frozen = json.loads((contract_dir / 'manifest.json').read_text(encoding='utf-8'))
    if frozen['profile'] != PROFILE:
        raise ValueError('Unexpected contract scoring profile')
    if frozen['expression_ids'] != ['P0']:
        raise ValueError('This evaluator expects one P0 wording per PCU')
    registry = release / 'private/candidates.jsonl'
    workload = release / 'public/workload.jsonl'
    expected = {
        'source_release_manifest_sha256': sha256(release_manifest_path),
        'source_registry_sha256': sha256(registry),
        'source_workload_sha256': sha256(workload),
    }
    for key, digest in expected.items():
        if frozen[key] != digest:
            raise ValueError(f'Frozen contracts do not match release: {key}')
    for path in (registry, workload):
        name = str(path.relative_to(release))
        if release_manifest['file_sha256'][name] != sha256(path):
            raise ValueError(f'Release hash mismatch: {name}')
    contract_path = contract_dir / frozen['contracts_file']
    if sha256(contract_path) != frozen['contracts_sha256']:
        raise ValueError('Frozen contract file hash mismatch')
    contracts = list(rows(contract_path))
    if len(contracts) != frozen['pcus']:
        raise ValueError('Frozen PCU count mismatch')
    if len({row['candidate_id'] for row in contracts}) != len(contracts):
        raise ValueError('Duplicate frozen candidate ID')
    candidates = {row['candidate_id']: row for row in rows(registry)}
    requests = {row['request_id'] for row in rows(workload)}
    for row in contracts:
        candidate = candidates.get(row['candidate_id'])
        if candidate is None or row['expression_id'] != 'P0':
            raise ValueError(f'Unknown frozen PCU: {row["candidate_id"]}')
        if row['request_ids'] != candidate['request_ids']:
            raise ValueError(f'Request mapping changed: {row["candidate_id"]}')
        if any(value not in requests for value in row['request_ids'].values()):
            raise ValueError(f'Unknown frozen request: {row["candidate_id"]}')
    return frozen, contracts, candidates, requests


def evaluate(release, contract_dir, models, output):
    frozen, contracts, candidates, known = load_contracts(release, contract_dir)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'profile': PROFILE,
        'metric_scope': 'WOD-E2E P0 kinematic proxy',
        'aggregation': 'equal PCUs; mean two arms, then one P0 expression per PCU',
        'denominator_pcus': frozen['pcus'],
        'denominator_requests': frozen['requests'],
        'contract_manifest_sha256': sha256(Path(contract_dir) / 'manifest.json'),
        'models': {},
    }
    for name, prediction_path in models:
        if name in report['models']:
            raise ValueError(f'Duplicate model name: {name}')
        predictions = {}
        for row in rows(prediction_path):
            request_id = row.get('request_id')
            if request_id not in known or request_id in predictions:
                raise ValueError(f'Unknown or duplicate prediction request: {request_id}')
            predictions[request_id] = row
        scored = []
        for record in contracts:
            candidate = candidates[record['candidate_id']]
            contract = record['contract']
            scored.append({
                'candidate_id': record['candidate_id'],
                'family': contract['family'],
                'expression_id': record['expression_id'],
                'camera_protocol': 'four_frame_available' if candidate['history_complete']
                                   else 'current_only_available',
                'a': score_arm(candidate, 0, contract, predictions),
                'b': score_arm(candidate, 1, contract, predictions),
            })
        per_pcu = output / (name.replace('/', '_') + '_per_pcu.jsonl')
        with per_pcu.open('w', encoding='utf-8') as stream:
            for item in scored:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False,
                                        sort_keys=True) + '\n')
        families = sorted({item['family'] for item in scored})
        protocols = sorted({item['camera_protocol'] for item in scored})
        report['models'][name] = {
            'predictions_sha256': sha256(prediction_path),
            'per_pcu_sha256': sha256(per_pcu),
            'overall': aggregate(scored),
            'by_family': {family: aggregate([item for item in scored
                                            if item['family'] == family])
                          for family in families},
            'by_camera_history_availability': {
                protocol: aggregate([item for item in scored
                                     if item['camera_protocol'] == protocol])
                for protocol in protocols},
        }
    (output / 'summary.json').write_text(json.dumps(report, indent=2,
                                                   ensure_ascii=False,
                                                   allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--contracts-dir', type=Path, required=True)
    parser.add_argument('--model', action='append', type=named_path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.release_dir, args.contracts_dir, args.model, args.output_dir)
    print(json.dumps({name: item['overall']['metrics_0to100']
                      for name, item in report['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
