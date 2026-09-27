#!/usr/bin/env python3
"""Score 3 s PCU trajectories against own and opposite contracts.

Use --release-dir for the human-approved benchmark. The legacy candidate path
remains available for construction diagnostics, not published model scores.
"""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import argparse
import json
import math
from pathlib import Path
import random

from build_lane_pcus import MapGeometry, TIMES, angle_diff, contract_pass
from release_io import load_release


ROOT = PROJECT_ROOT
CANDIDATES = ARTIFACT_ROOT / 'candidates/eligible_review_queue.jsonl'
SCORES = ARTIFACT_ROOT / 'scores'


def load_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_trajectory(traj):
    if not isinstance(traj, dict):
        return 'trajectory_type'
    if traj.get('coordinate_frame') != 'ego_at_anchor':
        return 'coordinate_frame'
    if traj.get('reference_point') != 'ego_pose_origin':
        return 'reference_point'
    timestamps = traj.get('timestamps_s')
    xy = traj.get('xy_m')
    if not isinstance(timestamps, list) or len(timestamps) != len(TIMES):
        return 'timestamps'
    if any(not isinstance(t, (int,float)) or not math.isfinite(t) or abs(t-expected)>1e-5
           for t,expected in zip(timestamps,TIMES)):
        return 'timestamps'
    if not isinstance(xy, list) or len(xy) != len(TIMES):
        return 'xy_length'
    if any(not isinstance(p,list) or len(p)!=2 or
           any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p)
           for p in xy):
        return 'xy_values'
    return None


def to_global(pcu, xy):
    pose = pcu['public_input']['ego_pose']
    ox, oy = pose['xy_m']
    yaw = pose['yaw_rad']
    c, s = math.cos(yaw), math.sin(yaw)
    return [[ox+c*x-s*y, oy+s*x+c*y] for x,y in xy]


def bootstrap_by_scene(rows, iterations=1000, seed=2026):
    scenes = {}
    for row in rows:
        scenes.setdefault(row['scene_token'],[]).append(row)
    labels = sorted(scenes)
    if not labels:
        return None
    rng = random.Random(seed)
    sisr,cfps = [],[]
    for _ in range(iterations):
        sample = [row for _ in labels for row in scenes[rng.choice(labels)]]
        sisr.append(sum(x['semantic_success'] for row in sample for x in row['arms'].values())/(2*len(sample)))
        cfps.append(sum(row['paired_success'] for row in sample)/len(sample))
    def interval(values):
        values.sort()
        return [values[int(0.025*(len(values)-1))],values[int(0.975*(len(values)-1))]]
    return {'OL_SemSISR_95CI':interval(sisr),'OL_SemCFPS_95CI':interval(cfps),
            'clusters':len(labels),'iterations':iterations,'seed':seed}


def score_one(geom, pcu, arm, trajectory):
    problem = validate_trajectory(trajectory)
    if problem:
        return {'output_valid': False, 'own_contract': False,
                'opposite_contract': False, 'semantic_success': False,
                'center_on_drivable': False, 'trajectory_feasible': False,
                'failure_code': 'O0_INVALID_OUTPUT',
                'detail': problem}
    points = to_global(pcu, trajectory['xy_m'])
    sidecar = pcu['private_sidecar']
    actual_progress = math.dist(pcu['public_input']['ego_pose']['xy_m'], points[-1])
    if pcu['family']=='lane_change_keep':
        current = sidecar['current_lane_tokens']
        target = sidecar['target_lane_tokens']
        change_contract = contract_pass(geom, {'global_xy_m': points}, target, 2.25)
        keep_contract = contract_pass(geom, {'global_xy_m': points}, current, 0)
        own = (change_contract if arm=='arm_0' else keep_contract) and actual_progress >= 8
        opposite = (keep_contract if arm=='arm_0' else change_contract) and actual_progress >= 8
    elif pcu['family']=='scoped_speed':
        local_positions = [[0.0,0.0]]+trajectory['xy_m']
        local_speeds = [math.dist(a,b)/0.25 for a,b in zip(local_positions,local_positions[1:])]
        bands = sidecar['speed_bands_mps']
        low = all(bands['low'][0]<=v<=bands['low'][1] for v,t in zip(local_speeds,TIMES) if t>=1.25)
        high = all(bands['high'][0]<=v<=bands['high'][1] for v,t in zip(local_speeds,TIMES) if t>=1.25)
        own = (low if arm=='arm_0' else high) and actual_progress >= 8
        opposite = (high if arm=='arm_0' else low) and actual_progress >= 8
    else:
        raise ValueError(f'Unsupported PCU family: {pcu["family"]}')
    geometry = all(geom.drivable_at(p) for p in points)
    positions = [pcu['public_input']['ego_pose']['xy_m']]+points
    speeds = [math.dist(a,b)/0.25 for a,b in zip(positions,positions[1:])]
    accelerations = [abs(a-b)/0.25 for a,b in zip(speeds,speeds[1:])]
    headings = [math.atan2(b[1]-a[1],b[0]-a[0]) for a,b in zip(positions,positions[1:])]
    yaw_rates = [abs(angle_diff(a,b))/0.25 for a,b in zip(headings,headings[1:])]
    feasible = (max(speeds) <= 17 and abs(speeds[0]-sidecar['speed_mps']) <= 3 and
                max(accelerations,default=0) <= 5 and max(yaw_rates,default=0) <= 1.0)
    semantic = own and not opposite
    if semantic:
        failure = None
    elif own and opposite:
        failure = 'O3_AMBIGUOUS'
    elif opposite:
        failure = 'O2_WRONG_SEMANTIC'
    elif actual_progress < 8:
        failure = 'O1_NO_COMMIT'
    else:
        failure = 'O5_QUANTITATIVE' if pcu['family']=='scoped_speed' else 'O10_PREFIX_ONLY'
    return {'output_valid': True, 'own_contract': own, 'opposite_contract': opposite,
            'semantic_success': semantic, 'center_on_drivable': geometry,
            'trajectory_feasible': feasible,
            'quality_code': None if feasible and geometry else ('O7_INFEASIBLE' if not feasible else 'O8_STATIC_UNSAFE'),
            'failure_code': failure, 'progress_m': round(actual_progress, 3)}


def evaluate(candidates, outputs, output_dir, release_version=None, split='all', self_check=False):
    source = load_jsonl(candidates) if isinstance(candidates, Path) else candidates
    pcus = {p['pcu_id']: p for p in source}
    seen = {}
    for record in outputs:
        condition = record.get('condition','canonical')
        seed = record.get('seed',0)
        if not isinstance(condition,str) or not condition or not isinstance(seed,int):
            raise ValueError('Condition must be a nonempty string and seed an integer')
        key = (record.get('pcu_id'),record.get('arm'),condition,seed)
        if key[0] not in pcus or key[1] not in ('arm_0','arm_1') or key in seen:
            raise ValueError(f'Unknown or duplicate PCU arm/condition/seed: {key}')
        recorded_hash = record.get('public_input_hash')
        if recorded_hash is not None and recorded_hash != pcus[key[0]]['public_input']['public_input_hash']:
            raise ValueError(f'Public input hash mismatch in model output: {key}')
        seen[key] = record.get('trajectory', {})
    maps = {}
    results = []
    metrics = {}
    conditions = sorted({(key[2],key[3]) for key in seen})
    for condition,seed in conditions:
        group = []
        for pcu in pcus.values():
            location = pcu['public_input']['location']
            if location not in maps:
                maps[location] = MapGeometry(location)
            arm_scores = {}
            for arm in ('arm_0','arm_1'):
                trajectory = seen.get((pcu['pcu_id'],arm,condition,seed), {})
                arm_scores[arm] = score_one(maps[location],pcu,arm,trajectory)
            row = {'pcu_id':pcu['pcu_id'], 'scene_token':pcu['public_input']['scene_token'],
                   'family':pcu['family'],'location':location,
                   'condition':condition,'seed':seed,'arms':arm_scores,
                   'paired_success':all(x['semantic_success'] for x in arm_scores.values())}
            group.append(row)
            results.append(row)
        n = len(group)
        metrics[f'{condition}/seed_{seed}'] = {
            'OL_SemSISR':sum(x['semantic_success'] for r in group for x in r['arms'].values())/(2*n) if n else None,
            'OL_SemCFPS':sum(r['paired_success'] for r in group)/n if n else None,
            'center_on_drivable_rate':sum(x['center_on_drivable'] for r in group for x in r['arms'].values())/(2*n) if n else None,
            'trajectory_feasible_rate':sum(x['trajectory_feasible'] for r in group for x in r['arms'].values())/(2*n) if n else None,
            'missing_arm_outputs':2*n-sum((p['pcu_id'],a,condition,seed) in seen for p in pcus.values() for a in ('arm_0','arm_1')),
            'scene_cluster_bootstrap':bootstrap_by_scene(group),
            'per_family':{
                family:{'pcus':sum(r['family']==family for r in group),
                        'OL_SemSISR':sum(x['semantic_success'] for r in group if r['family']==family for x in r['arms'].values())/(2*sum(r['family']==family for r in group)),
                        'OL_SemCFPS':sum(r['paired_success'] for r in group if r['family']==family)/sum(r['family']==family for r in group)}
                for family in sorted({r['family'] for r in group})}}
    canonical = metrics.get('canonical/seed_0',{})
    summary = {'dataset_version':'v1.0-test',
               'candidate_status':'approved' if release_version else 'needs_human_review',
               'release_version':release_version, 'split':split,
               'run_type':'witness_self_check' if self_check else 'model_outputs',
               'dynamic_safety':'unknown_no_test_instance_annotations',
               'pcus':len(pcus),'OL_SemSISR':canonical.get('OL_SemSISR'),
               'OL_SemCFPS':canonical.get('OL_SemCFPS'),
               'per_condition':metrics}
    if 'masked/seed_0' in metrics and canonical:
        summary['masked_CFPS_gap'] = canonical['OL_SemCFPS']-metrics['masked/seed_0']['OL_SemCFPS']
    if 'shuffled/seed_0' in metrics and canonical:
        summary['shuffled_CFPS_gap'] = canonical['OL_SemCFPS']-metrics['shuffled/seed_0']['OL_SemCFPS']
    paraphrases = [f'paraphrase_{i}/seed_0' for i in (1,2,3)]
    if all(label in metrics for label in paraphrases):
        by_key = {(r['pcu_id'],r['condition']):r for r in results if r['seed']==0}
        summary['Paraphrase_Worst_PCU'] = sum(
            all(by_key[(pcu_id,f'paraphrase_{i}')]['paired_success'] for i in (1,2,3))
            for pcu_id in pcus)/len(pcus)
    output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/'per_pcu.jsonl').open('w') as file:
        for row in results:
            file.write(json.dumps(row)+'\n')
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,default=CANDIDATES)
    parser.add_argument('--release-dir',type=Path,help='frozen benchmark release directory')
    parser.add_argument('--split',choices=('all','dev','heldout'),default='all')
    parser.add_argument('--outputs',type=Path)
    parser.add_argument('--output-dir',type=Path,default=SCORES)
    parser.add_argument('--self-check',action='store_true',help='score both generated witnesses')
    args = parser.parse_args()
    if args.release_dir:
        manifest, candidates = load_release(args.release_dir, include_private=True, split=args.split)
        release_version = manifest['release_version']
    else:
        if args.split != 'all':
            parser.error('--split requires --release-dir')
        candidates = args.candidates
        release_version = None
    if args.self_check:
        model_outputs = []
        for pcu in (candidates if isinstance(candidates,list) else load_jsonl(candidates)):
            witness_keys = ([('arm_0','witness_change'),('arm_1','witness_keep')]
                            if pcu['family']=='lane_change_keep' else
                            [('arm_0','witness_low'),('arm_1','witness_high')])
            for arm,key in witness_keys:
                model_outputs.append({'pcu_id':pcu['pcu_id'],'arm':arm,
                                      'trajectory':pcu['private_sidecar'][key]})
    elif args.outputs:
        model_outputs = load_jsonl(args.outputs)
        if not model_outputs:
            parser.error('Model output file is empty')
        if args.release_dir and not any(row.get('condition','canonical')=='canonical' for row in model_outputs):
            parser.error('Released benchmark scoring requires canonical outputs')
    else:
        parser.error('Provide --outputs or --self-check')
    print(json.dumps(evaluate(candidates,model_outputs,args.output_dir,
                              release_version,args.split,args.self_check),indent=2))


if __name__ == '__main__':
    main()
