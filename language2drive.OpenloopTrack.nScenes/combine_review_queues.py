#!/usr/bin/env python3
"""Combine test-only PCU families into one provisional review queue."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import json
from pathlib import Path


ROOT = (ARTIFACT_ROOT / 'candidates')
SOURCES = [ROOT/'review_queue.jsonl',ROOT/'speed_review_queue.jsonl']
OUTPUT = ROOT/'all_review_queue.jsonl'


def main():
    rows = []
    for path in SOURCES:
        with path.open() as source:
            rows.extend(json.loads(line) for line in source)
    ids = [row['pcu_id'] for row in rows]
    scenes = [row['public_input']['scene_token'] for row in rows]
    if len(set(ids))!=len(ids) or len(set(scenes))!=len(scenes):
        raise ValueError('Duplicate PCU ID or scene across families')
    if any(row['dataset_version']!='v1.0-test' for row in rows):
        raise ValueError('Only nuScenes test PCUs may enter the review queue')
    rows.sort(key=lambda row:row['pcu_id'])
    with OUTPUT.open('w') as output:
        for row in rows:
            output.write(json.dumps(row,separators=(',',':'))+'\n')
    print(f'Combined {len(rows)} test-set candidates -> {OUTPUT}')


if __name__ == '__main__':
    main()
