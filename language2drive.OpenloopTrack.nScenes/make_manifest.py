#!/usr/bin/env python3
"""Write a provisional, log-grouped split and source hashes for the test pilot."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path


ROOT = PROJECT_ROOT
DATA = DATA_ROOT
ARTIFACTS = ARTIFACT_ROOT


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-only', action='store_true')
    args = parser.parse_args()
    candidates = [json.loads(x) for x in (ARTIFACTS/'candidates/eligible_review_queue.jsonl').open()]
    needed = {x['public_input']['sample_token'] for x in candidates}
    catalog = {}
    with (ARTIFACTS/'scene_catalog/anchors.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['sample_token'] in needed:
                catalog[row['sample_token']] = row
    if len(catalog) != len(candidates):
        raise ValueError('Catalog does not contain every candidate sample')
    groups = defaultdict(list)
    locations = {}
    for candidate in candidates:
        row = catalog[candidate['public_input']['sample_token']]
        groups[row['log_token']].append(candidate['pcu_id'])
        locations[row['log_token']] = row['location']
    city_logs = defaultdict(list)
    for log,location in locations.items():
        city_logs[location].append(log)
    dev_logs = set()
    for location,logs in city_logs.items():
        logs.sort(key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        dev_logs.update(logs[:max(1,round(len(logs)*0.25))])
    split = {'status':'provisional_needs_human_review','dataset_version':'v1.0-test',
             'grouping':'log_token',
             'dev':sorted(p for log,ids in groups.items() if log in dev_logs for p in ids),
             'heldout':sorted(p for log,ids in groups.items() if log not in dev_logs for p in ids),
             'dev_log_tokens':sorted(dev_logs),
             'heldout_log_tokens':sorted(set(groups)-dev_logs)}
    assert not (set(split['dev_log_tokens']) & set(split['heldout_log_tokens']))
    (ARTIFACTS/'candidates/provisional_split.json').write_text(json.dumps(split,indent=2))
    if args.split_only:
        return

    source_files = [DATA/'v1.0-test'/f'{name}.json' for name in
                    ('scene','sample','sample_data','ego_pose','log','sensor',
                     'calibrated_sensor','sample_annotation','instance')]
    source_files += [DATA/'maps/expansion'/f'{location}.json' for location in sorted(city_logs)]
    output_files = [ARTIFACTS/path for path in
                    ('scene_catalog/anchors.jsonl','candidates/review_queue.jsonl',
                     'candidates/speed_review_queue.jsonl','candidates/all_review_queue.jsonl',
                     'candidates/eligible_review_queue.jsonl','audit/auto_rejected.jsonl',
                     'public_inputs/review_queue.jsonl','private_sidecars/review_queue.jsonl',
                     'candidates/provisional_split.json','audit/lidar_diagnostics.jsonl',
                     'audit/footprint_diagnostics.jsonl','audit/footprint_swept_diagnostics.jsonl',
                     'audit/review_sheet.csv','audit/validation_report.json',
                     'model_outputs/constant_velocity.jsonl',
                     'scores/constant_velocity/summary.json')]
    manifest = {'dataset_version':'v1.0-test','status':'provisional_needs_human_review',
                'candidate_count':len(candidates),'scene_count':len({catalog[x['public_input']['sample_token']]['scene_token'] for x in candidates}),
                'log_count':len(groups),'location_counts':dict(Counter(x['public_input']['location'] for x in candidates)),
                'family_counts':dict(Counter(x['family'] for x in candidates)),
                'side_counts':dict(Counter(x['private_sidecar']['target_side'] for x in candidates
                                           if x['family']=='lane_change_keep')),
                'split_counts':{'dev':len(split['dev']),'heldout':len(split['heldout'])},
                'source_sha256':{str(p):sha256(p) for p in source_files},
                'artifact_sha256':{str(p.relative_to(ARTIFACTS)):sha256(p) for p in output_files},
                'code_sha256':{p.name:sha256(p) for p in sorted(ROOT.glob('*.py'))}}
    (ARTIFACTS/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('_sha256')},indent=2))


if __name__ == '__main__':
    main()
