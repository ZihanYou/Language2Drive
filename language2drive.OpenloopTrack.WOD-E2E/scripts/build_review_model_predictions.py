"""Materialize private model trajectory overlays for the SSH review page.

Only valid 20-point trajectories are drawn. Missing/invalid predictions remain
explicit. Model diagnostics are never added to public benchmark inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from score_prerelease import normalize_trajectory


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def named_path(value):
    name, separator, raw_path = value.partition('=')
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError('Expected MODEL=path')
    return name, Path(raw_path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix='.model-predictions-', suffix='.json',
                                            dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw_temp, path)
    finally:
        if os.path.exists(raw_temp):
            os.unlink(raw_temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--model', action='append', type=named_path, required=True)
    parser.add_argument('--score', action='append', type=named_path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir
    manifest = json.loads((release / 'manifest.json').read_text())
    registry_path = release / 'private/candidates.jsonl'
    workload_path = release / 'public/workload.jsonl'
    for path in (registry_path, workload_path):
        key = str(path.relative_to(release))
        if sha256(path) != manifest['file_sha256'][key]:
            raise ValueError(f'Frozen release hash mismatch: {key}')
    candidates = list(rows(registry_path))
    known_ids = {row['request_id'] for row in rows(workload_path)}
    score_paths = dict(args.score)
    if len(score_paths) != len(args.score):
        raise ValueError('Duplicate score name')
    model_info, by_candidate = {}, {}
    for name, path in args.model:
        if name in model_info:
            raise ValueError(f'Duplicate model name: {name}')
        predictions = {}
        for row in rows(path):
            request_id = row.get('request_id')
            if request_id not in known_ids or request_id in predictions:
                raise ValueError(f'Unknown or duplicate request ID: {request_id}')
            predictions[request_id] = row
        diagnostics = {}
        score_path = score_paths.get(name)
        if score_path is not None:
            diagnostics = {row['candidate_id']: row for row in rows(score_path)}
            if len(diagnostics) != sum(c['status'] == 'DRAFT_P0_UNREVIEWED'
                                           for c in candidates):
                raise ValueError(f'Incomplete candidate diagnostics for {name}')
        valid = invalid = 0
        for candidate in candidates:
            requests = candidate['request_ids']
            if not requests:
                continue
            arms = {}
            for arm, request_id in requests.items():
                row = predictions.get(request_id)
                if row is None:
                    arms[arm] = {'status': 'missing'}
                    continue
                normalized, reason = normalize_trajectory(row.get('trajectory'))
                if reason:
                    invalid += 1
                    arms[arm] = {'status': 'invalid', 'reason': reason}
                    continue
                valid += 1
                points, _ = normalized
                arms[arm] = {'status': 'valid', 'trajectory': points}
                detail = diagnostics.get(candidate['candidate_id'], {}).get('requests', {}).get(arm)
                if detail and detail['status'] == 'valid':
                    arms[arm]['reference_margin_m'] = detail['reference_margin_m']
                    arms[arm]['nearest_reference_agreement'] = detail['nearest_reference_agreement']
            by_candidate.setdefault(candidate['candidate_id'], {})[name] = arms
        model_info[name] = {'prediction_path': str(path), 'prediction_sha256': sha256(path),
                            'score_path': str(score_path) if score_path else None,
                            'score_sha256': sha256(score_path) if score_path else None,
                            'predicted': len(predictions), 'valid': valid,
                            'invalid': invalid, 'missing': len(known_ids) - len(predictions)}
    payload = {'status': 'private_unreviewed_model_overlay',
               'release_manifest_sha256': sha256(release / 'manifest.json'),
               'models': model_info, 'candidates': by_candidate}
    atomic_json(args.output, payload)
    print(json.dumps({'models': model_info, 'candidates': len(by_candidate),
                      'output': str(args.output), 'sha256': sha256(args.output)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
