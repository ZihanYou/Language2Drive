#!/usr/bin/env python3
"""Validate actual campaign completeness; never create or substitute evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(output):
    root = Path(output)
    summary = json.loads((root / 'summary.json').read_text())
    failures, counts = [], {}
    for track in ('carla', 'realsim', 'nuscenes', 'navsim', 'wod'):
        row = summary.get('tracks', {}).get(track, {})
        if row.get('status') != 'scored':
            failures.append(track + ': no successful native scoring run')
    for track, minimum in [('nuscenes', 120), ('navsim', 304), ('wod', 482)]:
        path = root / track / 'inference.json'
        if not path.is_file():
            failures.append(track + ': missing inference inventory'); continue
        inventory = json.loads(path.read_text())
        counts[track] = inventory.get('scheduled_requests', 0)
        if counts[track] < minimum or inventory.get('failed_requests') != 0:
            failures.append(track + ': incomplete or failed full-cohort inference')
    real = root / 'realsim/runtime_return.json'
    experiment = root / 'realsim/experiment.json'
    if real.is_file() and experiment.is_file():
        outcomes = json.loads(real.read_text())['rollout_returns']
        records = json.loads(experiment.read_text())['results']
        counts['realsim'] = len(outcomes)
        if len(outcomes) != 60 or len({r['session_uuid'] for r in outcomes}) != 60 or not all(r['success'] is True for r in outcomes):
            failures.append('realsim: expected 60 successful unique real runtime outcomes')
        if len(records) != 60 or not all(r.get('execution_status') == 'completed' and r.get('evidence_valid') is True for r in records):
            failures.append('realsim: missing valid native trace evidence')
    else:
        failures.append('realsim: missing runtime/experiment evidence')
    scheduled = 0
    for split in ('dev', 'test'):
        path = root / 'carla' / split / 'run_plan.json'
        score = root / 'carla' / split / 'score.json'
        if not path.is_file() or not score.is_file():
            failures.append('carla: missing ' + split + ' native plan/score'); continue
        plan = json.loads(path.read_text())
        scope = plan.get('evaluation_scope', {}).get('run_ids', [])
        expected = len(scope)
        scheduled += expected
        result = json.loads(score.read_text())
        metrics = result.get('metrics', {})
        if (not expected or len(set(scope)) != expected
                or result.get('scope') != plan.get('evaluation_scope')
                or result.get('plan_file_sha256') != hashlib.sha256(path.read_bytes()).hexdigest()
                or any(metrics.get(key) != expected for key in ('expected_runs', 'valid_result_runs', 'finalized_runs'))
                or metrics.get('missing_runs') != 0):
            failures.append('carla: incomplete or unbound ' + split + ' native trace evidence')
    counts['carla'] = scheduled
    if scheduled != 384:
        failures.append('carla: expected 384 core scheduled rollouts, dev + test, seed 1')
    return dict(complete=not failures, counts=counts, failures=failures,
                formal_submission_approved=False,
                note='Artifact completeness check; native scores and receipts remain authoritative.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = validate(args.output)
    except (OSError, ValueError, KeyError) as error:
        result = dict(complete=False, failures=[str(error)], formal_submission_approved=False)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['complete'] else 1)
