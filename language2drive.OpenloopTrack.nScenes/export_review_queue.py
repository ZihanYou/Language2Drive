#!/usr/bin/env python3
"""Separate public pilot inputs from private evaluation sidecars."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import hashlib
import json
from pathlib import Path

from build_catalog import canonical_hash


ROOT = ARTIFACT_ROOT
SOURCE = ROOT / 'candidates/eligible_review_queue.jsonl'
CATALOG = ROOT / 'scene_catalog/anchors.jsonl'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return 'sha256:' + h.hexdigest()


def main():
    with CATALOG.open() as catalog:
        timestamps = {row['sample_token']:row['timestamp_us']
                      for row in (json.loads(line) for line in catalog)}
    public_path = ROOT/'public_inputs/review_queue.jsonl'
    private_path = ROOT/'private_sidecars/review_queue.jsonl'
    public_path.parent.mkdir(parents=True,exist_ok=True)
    private_path.parent.mkdir(parents=True,exist_ok=True)
    count = 0
    with SOURCE.open() as source, public_path.open('w') as pub, private_path.open('w') as priv:
        for line in source:
            pcu = json.loads(line)
            if pcu['family']=='lane_change_keep':
                side = pcu['private_sidecar']['target_side']
                first_paraphrases = [
                    f'Move into the lane immediately to our {side} and remain in it.',
                    f'Take the adjacent {side} lane, then keep driving there.',
                    f'Shift {side} by one lane and hold that lane.']
                second_paraphrases = [
                    'Continue in this lane without changing lanes.',
                    'Keep to the lane we are in now.',
                    'Stay in the present lane for this stretch.']
            elif pcu['family']=='scoped_speed':
                first_paraphrases = [
                    'Drive at roughly fifteen kilometers per hour on this stretch.',
                    'Keep a slow speed of about 15 km/h ahead.',
                    'Travel through this section at around 15 km/h.']
                second_paraphrases = [
                    'Drive at roughly thirty kilometers per hour on this stretch.',
                    'Keep a moderate speed of about 30 km/h ahead.',
                    'Travel through this section at around 30 km/h.']
            else:
                raise ValueError(pcu['family'])
            shared = dict(pcu['public_input'])
            expected_hash = shared.pop('public_input_hash')
            shared['dataset_version'] = 'v1.0-test'
            shared['timestamp_us'] = timestamps[shared['sample_token']]
            if canonical_hash(shared) != expected_hash:
                raise ValueError(f'Public input hash mismatch: {pcu["pcu_id"]}')
            shared['public_input_hash'] = expected_hash
            public = {'pcu_id':pcu['pcu_id'], 'dataset_version':'v1.0-test',
                      'status':'needs_human_review', 'family':pcu['family'],
                      'public_input':shared,
                      'arms':{
                          'arm_0':{'instruction':pcu['arms']['arm_0']['instruction'],
                                   'paraphrases':first_paraphrases},
                          'arm_1':{'instruction':pcu['arms']['arm_1']['instruction'],
                                   'paraphrases':second_paraphrases}}}
            private = {'pcu_id':pcu['pcu_id'], 'dataset_version':'v1.0-test',
                       'status':'needs_human_review', 'family':pcu['family'],
                       'contracts':{arm:pcu['arms'][arm]['contract'] for arm in ('arm_0','arm_1')},
                       'private_sidecar':pcu['private_sidecar']}
            pub.write(json.dumps(public,separators=(',',':'))+'\n')
            priv.write(json.dumps(private,separators=(',',':'))+'\n')
            count += 1
    manifest = {'dataset_version':'v1.0-test','status':'needs_human_review',
                'pcus':count,'public_path':str(public_path),'private_path':str(private_path),
                'source_sha256':digest(SOURCE),'public_sha256':digest(public_path),
                'private_sha256':digest(private_path)}
    (ROOT/'candidates/review_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
