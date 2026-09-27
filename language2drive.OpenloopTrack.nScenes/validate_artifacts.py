#!/usr/bin/env python3
"""Independently validate the full nuScenes-test review artifact set."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

from collections import Counter
import json
import math
from pathlib import Path

from build_catalog import canonical_hash
from build_lane_pcus import MapGeometry
from score_lane_pcus import score_one, to_global


ROOT = ARTIFACT_ROOT
DATA = DATA_ROOT


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def validate():
    test_samples = {x['token'] for x in json.loads((DATA/'v1.0-test/sample.json').read_text())}
    catalog = read_jsonl(ROOT/'scene_catalog/anchors.jsonl')
    assert len(catalog)==len(test_samples)==6008
    assert {x['sample_token'] for x in catalog}==test_samples
    assert all(x['dataset_version']=='v1.0-test' for x in catalog)
    public = read_jsonl(ROOT/'public_inputs/review_queue.jsonl')
    private = read_jsonl(ROOT/'private_sidecars/review_queue.jsonl')
    eligible = read_jsonl(ROOT/'candidates/eligible_review_queue.jsonl')
    assert len(public)==len(private)==len(eligible)==82
    assert {x['pcu_id'] for x in public}=={x['pcu_id'] for x in private}=={x['pcu_id'] for x in eligible}
    assert len({x['public_input']['scene_token'] for x in public})==82
    private_by_id = {x['pcu_id']:x for x in private}
    candidate_by_id = {x['pcu_id']:x for x in eligible}
    swept = {x['pcu_id']:x for x in read_jsonl(ROOT/'audit/footprint_swept_diagnostics.jsonl')}
    maps = {}
    counts = Counter()
    for record in public:
        shared = dict(record['public_input'])
        digest = shared.pop('public_input_hash')
        assert canonical_hash(shared)==digest
        assert shared['dataset_version']=='v1.0-test'
        assert shared['sample_token'] in test_samples
        assert all(h['timestamp_us']<shared['timestamp_us'] for h in shared['history_refs'])
        assert [h['sample_token'] for h in shared['history_refs']]==shared['history_sample_tokens']
        refs = [shared['sensor_refs']]+[h['sensor_refs'] for h in shared['history_refs']]
        assert all((DATA/item['filename']).is_file() for group in refs for item in group.values())
        public_text = json.dumps(record)
        assert all(secret not in public_text for secret in
                   ('private_sidecar','witness_low','witness_high','witness_keep',
                    'witness_change','target_lane_token','speed_bands_mps'))
        pcu = candidate_by_id[record['pcu_id']]
        assert pcu['family']==record['family']==private_by_id[record['pcu_id']]['family']
        location = shared['location']
        if location not in maps:
            maps[location] = MapGeometry(location)
        keys = (('witness_change','witness_keep') if pcu['family']=='lane_change_keep'
                else ('witness_low','witness_high'))
        for arm,key in zip(('arm_0','arm_1'),keys):
            witness = pcu['private_sidecar'][key]
            projected = to_global(pcu,witness['xy_m'])
            assert all(math.dist(a,b)<1e-6 for a,b in zip(projected,witness['global_xy_m']))
            result = score_one(maps[location],pcu,arm,witness)
            assert result['semantic_success'] and not result['opposite_contract']
            assert result['center_on_drivable'] and result['trajectory_feasible']
            evidence = swept[record['pcu_id']][arm]
            assert all(evidence[origin]['all_covered'] for origin in
                       ('assumed_rearward_origin','assumed_center_origin'))
            counts['verified_arms'] += 1
        counts[pcu['family']] += 1
    split = json.loads((ROOT/'candidates/provisional_split.json').read_text())
    assert set(split['dev']).isdisjoint(split['heldout'])
    assert set(split['dev'])|set(split['heldout'])=={x['pcu_id'] for x in public}
    assert set(split['dev_log_tokens']).isdisjoint(split['heldout_log_tokens'])
    report = {'dataset_version':'v1.0-test','test_samples':len(test_samples),
              'review_pcus':len(public),'verified_arms':counts['verified_arms'],
              'families':{key:counts[key] for key in ('lane_change_keep','scoped_speed')},
              'test_only':True,'public_private_separated':True,
              'input_hashes_valid':True,'coordinate_roundtrips_valid':True,
              'own_opposite_contracts_valid':True,
              'swept_footprint_assumptions_valid':True,
              'split_logs_disjoint':True,'status':'review_candidates_not_approved'}
    (ROOT/'audit/validation_report.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
