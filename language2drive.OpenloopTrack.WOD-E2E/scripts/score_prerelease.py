"""Score model trajectories against the unreviewed stop/go motion proxy.

This scorer is intentionally separate from public model inputs. It reports
coverage, validity and deterministic motion-contract results only. Safety and
scene-semantic correctness remain unknown until human review.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_trajectory(value):
    if isinstance(value, dict):
        value = value.get('xy_m')
    if not isinstance(value, list) or len(value) != 20:
        return None, 'expected_20_xy_points_at_4hz'
    points = []
    for point in value:
        if not isinstance(point, (list,tuple)) or len(point) != 2 or any(
            not isinstance(v, (int,float)) or isinstance(v,bool) or not math.isfinite(v)
            for v in point
        ):
            return None, 'invalid_xy_point'
        points.append([float(point[0]),float(point[1])])
    previous = (0.0,0.0)
    speeds = []
    for point in points:
        speeds.append(math.dist(point, previous)*4)
        previous = point
    if max(speeds) > 50 or any(abs(b-a)*4 > 20 for a,b in zip(speeds,speeds[1:])):
        return None, 'implausible_speed_or_acceleration'
    return (points, speeds), None


def contract_result(trajectory, contract):
    points, speeds = trajectory
    tail = sum(speeds[-4:])/4
    x5,y5 = points[-1]
    if contract == 'STOP_TERMINAL':
        success = tail <= 0.6
    elif contract == 'GO_TERMINAL':
        success = tail >= 2.0 and x5 >= 8.0
    else:
        raise ValueError('Unknown contract: '+contract)
    return {'success':success,'tail_mean_speed_mps':tail,'x5_m':x5,'y5_m':y5}


def summarize(units, mapping, prediction_by_id):
    requested = 2*len(units)
    seen = valid = success = complete_pairs = paired_success = cross_success = 0
    detail = []
    for unit in units:
        per = {}
        for request_id,contract in unit['request_to_contract'].items():
            row = prediction_by_id.get(request_id)
            if row is None:
                per[contract] = {'request_id':request_id,'status':'missing'}
                continue
            seen += 1
            trajectory,reason = normalize_trajectory(row.get('trajectory'))
            if reason:
                per[contract] = {'request_id':request_id,'status':'invalid','reason':reason}
                continue
            valid += 1
            result = contract_result(trajectory,contract)
            success += int(result['success'])
            per[contract] = {'request_id':request_id,'status':'valid',**result,
                             'cross_success':contract_result(
                                 trajectory,'GO_TERMINAL' if contract=='STOP_TERMINAL' else 'STOP_TERMINAL')['success']}
        complete = all(per[name]['status']=='valid' for name in ('STOP_TERMINAL','GO_TERMINAL'))
        if complete:
            complete_pairs += 1
            paired_success += int(per['STOP_TERMINAL']['success'] and per['GO_TERMINAL']['success'])
            cross_success += int(per['STOP_TERMINAL']['cross_success'] and per['GO_TERMINAL']['cross_success'])
        detail.append({'unit_id':unit['unit_id'],'source_candidate_id':unit['source_candidate_id'],
                       'split':unit['split'],'requests':per,
                       'pair_complete':complete,
                       'paired_success':complete and per['STOP_TERMINAL']['success'] and per['GO_TERMINAL']['success']})
    return ({'units':len(units),'requested':requested,'predicted':seen,'valid':valid,
             'missing':requested-seen,'invalid':seen-valid,'request_coverage':seen/requested if requested else None,
             'valid_success_count':success,
             'valid_success_rate':success/valid if valid else None,
             'all_request_success_lower_bound':success/requested if requested else None,
             'complete_pairs':complete_pairs,'paired_success_count':paired_success,
             'paired_success_rate_on_complete_pairs':paired_success/complete_pairs if complete_pairs else None,
             'cross_success_count':cross_success,
             'cross_success_rate_on_complete_pairs':cross_success/complete_pairs if complete_pairs else None},detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    release = args.release_dir
    manifest = json.loads((release/'manifest.json').read_text())
    if manifest['release_status'] != 'unreviewed_pilot':
        raise ValueError('Expected unreviewed stop/go pilot')
    for name,sha in manifest['file_sha256'].items():
        if digest(release/name) != sha:
            raise ValueError('Release file hash mismatch: '+name)
    workload = list(rows(release/'public/workload.jsonl'))
    mapping = {row['request_id']:row for row in workload}
    if len(mapping) != len(workload):
        raise ValueError('Duplicate release request IDs')
    targets = list(rows(release/'private/targets.jsonl'))
    prediction_by_id = {}
    for row in rows(args.predictions):
        request_id = row.get('request_id')
        if request_id not in mapping:
            raise ValueError('Unknown request ID: '+str(request_id))
        if request_id in prediction_by_id:
            raise ValueError('Duplicate prediction: '+request_id)
        prediction_by_id[request_id] = row
    summary = {'model_id':args.model_id,'release_version':manifest['version'],
               'prediction_sha256':digest(args.predictions),
               'release_manifest_sha256':digest(release/'manifest.json'),
               'status':'preliminary_unreviewed_motion_proxy',
               'safety_and_semantic_validity':'unknown',
               'splits':{}}
    details = []
    for split in ('dev','pilot_eval','all'):
        chosen = [u for u in targets if split=='all' or u['split']==split]
        score,detail = summarize(chosen,mapping,prediction_by_id)
        summary['splits'][split] = score
        if split=='all':
            details = detail
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (args.output_dir/'per_unit.jsonl').open('w') as f:
        for row in details:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
