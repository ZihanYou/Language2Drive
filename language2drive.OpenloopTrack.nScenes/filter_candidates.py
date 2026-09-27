#!/usr/bin/env python3
"""Exclude candidates failing both swept-footprint-origin assumptions."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import json
from pathlib import Path


ROOT = ARTIFACT_ROOT


def main():
    footprint = {x['pcu_id']:x for x in
                 (json.loads(line) for line in (ROOT/'audit/footprint_swept_diagnostics.jsonl').open())}
    kept,rejected = [],[]
    with (ROOT/'candidates/all_review_queue.jsonl').open() as source:
        for line in source:
            pcu = json.loads(line)
            result = footprint[pcu['pcu_id']]
            failed = []
            for arm in ('arm_0','arm_1'):
                if not any(result[arm][origin]['all_covered'] for origin in
                           ('assumed_rearward_origin','assumed_center_origin')):
                    failed.append(arm)
            if failed:
                rejected.append({'pcu_id':pcu['pcu_id'],'reason':'swept_footprint_fails_both_origin_assumptions',
                                 'arms':failed})
            else:
                kept.append(pcu)
    with (ROOT/'candidates/eligible_review_queue.jsonl').open('w') as output:
        for pcu in kept:
            output.write(json.dumps(pcu,separators=(',',':'))+'\n')
    with (ROOT/'audit/auto_rejected.jsonl').open('w') as output:
        for item in rejected:
            output.write(json.dumps(item)+'\n')
    print(json.dumps({'raw_candidates':len(kept)+len(rejected),
                      'eligible_for_human_review':len(kept),'auto_rejected':len(rejected)},indent=2))


if __name__ == '__main__':
    main()
