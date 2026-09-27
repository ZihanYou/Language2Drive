#!/usr/bin/env python3
"""Freeze only human-approved nuScenes-test PCUs into a versioned release."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


ROOT = ARTIFACT_ROOT


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path,rows):
    with path.open('w') as output:
        for row in rows:
            output.write(json.dumps(row,separators=(',',':'))+'\n')


REVIEW_CRITERIA = ('observable', 'arm_0_feasible', 'arm_1_feasible',
                   'mutually_exclusive', 'traffic_clearance', 'map_binding')


def freeze(sheet,output,min_pcus,version,require_complete_review=False):
    sheet_hash = sha256(sheet)
    with sheet.open(newline='') as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        decisions = list(reader)
    if sha256(sheet) != sheet_hash:
        raise ValueError('Review sheet changed while reading; retry the freeze')
    if len({row['pcu_id'] for row in decisions})!=len(decisions):
        raise ValueError('Duplicate PCU ID in review sheet')
    eligible = read_jsonl(ROOT/'candidates/eligible_review_queue.jsonl')
    expected = {row['pcu_id'] for row in eligible}
    if not expected.issubset({row['pcu_id'] for row in decisions}):
        raise ValueError('Review sheet is missing eligible candidates')
    if require_complete_review:
        required = {'reviewer', 'submitted_at_utc', 'review_status'} | set(REVIEW_CRITERIA)
        if not required.issubset(columns):
            raise ValueError('Complete release requires submitted review fields')
        if any(not row['human_decision'].strip() for row in decisions):
            raise ValueError('Complete release requires a decision for every candidate')
        if any(not row['reviewer'].strip() or not row['submitted_at_utc'].strip()
               for row in decisions):
            raise ValueError('Complete release requires reviewer and submission time')
    approvals = set()
    for row in decisions:
        decision = row['human_decision'].strip().lower()
        if decision not in ('','approve','reject'):
            raise ValueError(f'Unknown decision for {row["pcu_id"]}: {decision}')
        if decision=='approve':
            if row['pcu_id'] not in expected or row['auto_status']!='needs_human_review':
                raise ValueError(f'Cannot approve auto-rejected PCU: {row["pcu_id"]}')
            if not row['human_notes'].strip():
                raise ValueError(f'Approval requires evidence notes: {row["pcu_id"]}')
            if require_complete_review and any(row[field] != 'yes' for field in REVIEW_CRITERIA):
                raise ValueError(f'Approval requires all review criteria: {row["pcu_id"]}')
            approvals.add(row['pcu_id'])
    if len(approvals)<min_pcus:
        raise ValueError(f'{len(approvals)} approved PCUs; at least {min_pcus} required')

    public = {row['pcu_id']:row for row in read_jsonl(ROOT/'public_inputs/review_queue.jsonl')}
    private = {row['pcu_id']:row for row in read_jsonl(ROOT/'private_sidecars/review_queue.jsonl')}
    if approvals-set(public) or approvals-set(private):
        raise ValueError('Approved PCU missing public or private record')
    split = json.loads((ROOT/'candidates/provisional_split.json').read_text())
    dev = sorted(approvals & set(split['dev']))
    heldout = sorted(approvals & set(split['heldout']))
    if len(dev)+len(heldout)!=len(approvals):
        raise ValueError('Approved PCU missing from log-grouped split')
    output.mkdir(parents=True,exist_ok=False)
    public_path = output/'public_inputs.jsonl'
    private_path = output/'private_sidecars.jsonl'
    write_jsonl(public_path,[dict(public[pcu_id],status='approved') for pcu_id in sorted(approvals)])
    write_jsonl(private_path,[dict(private[pcu_id],status='approved') for pcu_id in sorted(approvals)])
    review_path = output/'review_sheet.csv'
    shutil.copyfile(sheet,review_path)
    if sha256(review_path) != sheet_hash:
        raise ValueError('Review sheet changed during release copy')
    family_counts = {}
    location_counts = {}
    for pcu_id in approvals:
        family = public[pcu_id]['family']
        location = public[pcu_id]['public_input']['location']
        family_counts[family] = family_counts.get(family,0)+1
        location_counts[location] = location_counts.get(location,0)+1
    manifest = {'release_version':version,'dataset_version':'v1.0-test',
                'status':'approved', 'common_horizon_s':3.0,
                'coordinate_frame':'ego_at_anchor','reference_point':'ego_pose_origin',
                'approved_pcus':len(approvals),'reviewed_pcus':len(decisions),
                'rejected_pcus':sum(row['human_decision'].strip().lower()=='reject' for row in decisions),
                'families':family_counts,'locations':location_counts,
                'dev_pcus':dev,'heldout_pcus':heldout,
                'review_sheet_sha256':sheet_hash,
                'public_sha256':sha256(public_path),'private_sha256':sha256(private_path),
                'source_manifest_sha256':sha256(ROOT/'manifest.json'),
                'freeze_code_sha256':sha256(Path(__file__)),
                'verifier_code_sha256':sha256(Path(__file__).with_name('score_lane_pcus.py'))}
    (output/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'release_version':version,'approved_pcus':len(approvals),
            'dev':len(dev),'heldout':len(heldout),'output':str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-sheet',type=Path,default=ROOT/'audit/review_sheet.csv')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-pcus',type=int,default=40)
    parser.add_argument('--version',default='nuscenes-test-v0.1')
    parser.add_argument('--require-complete-review',action='store_true')
    args = parser.parse_args()
    print(json.dumps(freeze(args.review_sheet,args.output,args.min_pcus,args.version,
                            args.require_complete_review),indent=2))


if __name__ == '__main__':
    main()
