#!/usr/bin/env python3
"""Create a review sheet with evidence links and blank human-decision fields."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import csv
import json
from pathlib import Path


ROOT = ARTIFACT_ROOT


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def main():
    raw = read_jsonl(ROOT/'candidates/all_review_queue.jsonl')
    eligible = {x['pcu_id'] for x in read_jsonl(ROOT/'candidates/eligible_review_queue.jsonl')}
    lidar = {x['pcu_id']:x for x in read_jsonl(ROOT/'audit/lidar_diagnostics.jsonl')}
    footprint = {x['pcu_id']:x for x in read_jsonl(ROOT/'audit/footprint_swept_diagnostics.jsonl')}
    rejected = {x['pcu_id']:x for x in read_jsonl(ROOT/'audit/auto_rejected.jsonl')}
    needed = {x['public_input']['sample_token'] for x in raw}
    log_by_sample = {}
    with (ROOT/'scene_catalog/anchors.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['sample_token'] in needed:
                log_by_sample[row['sample_token']] = row['log_token']
    columns = ['pcu_id','family','location','scene_token','log_token','sample_token',
               'arm_0_instruction','arm_1_instruction','lidar_arm_0_points','lidar_arm_1_points',
               'footprint_arm_0_pass','footprint_arm_1_pass','auto_status','review_priority',
               'evidence_link','human_decision','human_notes']
    output = ROOT/'audit/review_sheet.csv'
    with output.open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        for index,pcu in enumerate(raw,1):
            pcu_id = pcu['pcu_id']
            sample = pcu['public_input']['sample_token']
            scan = lidar[pcu_id]
            body = footprint[pcu_id]
            arm0_pass = any(body['arm_0'][origin]['all_covered'] for origin in
                            ('assumed_rearward_origin','assumed_center_origin'))
            arm1_pass = any(body['arm_1'][origin]['all_covered'] for origin in
                            ('assumed_rearward_origin','assumed_center_origin'))
            priority = ('excluded' if pcu_id in rejected else
                        'high' if max(scan['lidar_points_in_arm_0_corridor'],
                                      scan['lidar_points_in_arm_1_corridor'])>20 else 'normal')
            writer.writerow({
                'pcu_id':pcu_id,'family':pcu['family'],
                'location':pcu['public_input']['location'],
                'scene_token':pcu['public_input']['scene_token'],
                'log_token':log_by_sample[sample],'sample_token':sample,
                'arm_0_instruction':pcu['arms']['arm_0']['instruction'],
                'arm_1_instruction':pcu['arms']['arm_1']['instruction'],
                'lidar_arm_0_points':scan['lidar_points_in_arm_0_corridor'],
                'lidar_arm_1_points':scan['lidar_points_in_arm_1_corridor'],
                'footprint_arm_0_pass':arm0_pass,'footprint_arm_1_pass':arm1_pass,
                'auto_status':'needs_human_review' if pcu_id in eligible else 'auto_rejected',
                'review_priority':priority,
                'evidence_link':f'index.html#candidate-{index:03d}',
                'human_decision':'','human_notes':''})
    print(f'Wrote {len(raw)} rows -> {output}')


if __name__ == '__main__':
    main()
