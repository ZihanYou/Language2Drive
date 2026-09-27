"""nuScenes paper profile: Eqs. (34)--(40), Tables 14--15, Appendix E.6.

Per-instance kernels return fractions; report metrics except ADE use 0--100.
"""
from collections import defaultdict
import random

from score_lane_pcus import score_one, validate_trajectory
from model_eval.semantic_metrics import aggregate, ade
from model_eval.unified_semantic_metrics import (
    compliance, trajectory_preference, normalized_behavior_difference, family_macro)

PROFILE = 'nuscenes-paper-1.0'
CONDITIONS = ('canonical', 'masked', 'shuffled', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3')
WORDINGS = ('canonical', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3')
METRICS = ('SemSISR', 'SemCFPS', 'SCS', 'PCS', 'SRG', 'NBD', 'paired_ADE_m',
           'center_on_drivable', 'trajectory_feasible')
MAIN_METRICS = ('SemSISR', 'SemCFPS', 'SCS', 'PCS')


def normalized_payload(value):
    """Represent unparseable payloads as invalid; never fill trajectory points."""
    return value if isinstance(value, dict) else {}


def paired_completion(c0, c1):
    return {'SCS': (c0 + c1) / 2., 'PCS': min(c0, c1)}


def score_pair(pcu, trajectories, maps, corridors):
    trajectories = [normalized_payload(t) for t in trajectories]
    valid = [validate_trajectory(t) is None for t in trajectories]
    scores = [score_one(maps[pcu['public_input']['location']], pcu, arm, t)
              for arm, t in zip(('arm_0', 'arm_1'), trajectories)]
    matrix = [[compliance(pcu, t, arm, corridors) for arm in ('arm_0', 'arm_1')]
              for t in trajectories]
    binary = [float(s['semantic_success']) for s in scores]
    values = dict(SemSISR=sum(binary)/2., SemCFPS=binary[0]*binary[1],
                  **paired_completion(matrix[0][0], matrix[1][1]))
    values.update(SRG=None, NBD=None, paired_ADE_m=None,
                  center_on_drivable=sum(s['center_on_drivable'] for s in scores)/2.,
                  trajectory_feasible=sum(s['trajectory_feasible'] for s in scores)/2.)
    preferences = None
    if all(valid):
        preferences = [trajectory_preference(pcu, t, corridors) for t in trajectories]
        a, b = [t['xy_m'] for t in trajectories]
        values.update(SRG=(preferences[0]-preferences[1])/2.,
                      NBD=normalized_behavior_difference(a,b), paired_ADE_m=ade(a,b))
    return dict(pcu_id=pcu['pcu_id'], scene_token=pcu['public_input']['scene_token'],
                family=pcu['family'], valid_arms=valid, binary_arms=binary,
                own_completion=[matrix[0][0], matrix[1][1]],
                score_matrix=matrix, distance_preferences=preferences,
                arm_diagnostics=scores, values=values)


def collapse_wordings(rows, wordings):
    """Pair arms first, mean declared wordings within PCU, then mean PCUs."""
    groups = defaultdict(dict)
    for row in rows:
        if row['condition'] in wordings:
            if row['condition'] in groups[row['pcu_id']]:
                raise ValueError('Duplicate PCU/wording')
            groups[row['pcu_id']][row['condition']] = row
    collapsed = []
    for pid, group in groups.items():
        if set(group) != set(wordings):
            raise ValueError('Missing declared wording for '+pid)
        source = [group[w] for w in wordings]
        values = {}
        for metric in METRICS:
            measured = [r['values'][metric] for r in source if r['values'][metric] is not None]
            values[metric] = sum(measured)/len(measured) if measured else None
        collapsed.append(dict(pcu_id=pid, family=source[0]['family'],
                              scene_token=source[0]['scene_token'], values=values))
    return collapsed


def binary_interval(rows, metric):
    """Frozen binary scene bootstrap with floor-index quantiles."""
    scenes = defaultdict(list)
    for row in rows:
        scenes[row['scene_token']].append(row['values'][metric])
    labels = sorted(scenes)
    if not labels:
        return [None, None]
    rng = random.Random(2026)
    draws = []
    for _ in range(1000):
        sample = [v for _ in labels for v in scenes[rng.choice(labels)]]
        draws.append(sum(sample)/len(sample))
    draws.sort()
    return [draws[int(q*(len(draws)-1))] for q in (.025,.975)]


def scaled(stat, metric):
    factor = 1. if metric == 'paired_ADE_m' else 100.
    return dict(mean=None if stat['mean'] is None else factor*stat['mean'],
                ci95=[None if v is None else factor*v for v in stat['ci95']],
                n=stat['n'], unit='m' if metric=='paired_ADE_m' else 'percent_points')


def summarize(rows, wordings):
    selected = [r for r in rows if r['condition'] in wordings]
    collapsed = collapse_wordings(selected, wordings)
    family_scenes = {f: sorted({r['scene_token'] for r in collapsed if r['family']==f})
                     for f in sorted({r['family'] for r in collapsed})}
    result = {}
    for group in ['all']+list(family_scenes):
        items = [r for r in collapsed if group=='all' or r['family']==group]
        originals = [r for r in selected if group=='all' or r['family']==group]
        stats = {}
        for metric in METRICS:
            values = [(r['scene_token'], r['values'][metric]) for r in items
                      if r['values'][metric] is not None]
            stat = aggregate(values, [r['scene_token'] for r in items])
            if metric in ('SemSISR','SemCFPS'):
                stat['ci95'] = binary_interval(items,metric)
            stats[metric] = scaled(stat,metric)
        result[group] = dict(metrics=stats, coverage=dict(
            pcus=len(items), scheduled_pairs=len(originals), scheduled_arms=2*len(originals),
            valid_pairs=sum(all(r['valid_arms']) for r in originals),
            valid_arms=sum(sum(r['valid_arms']) for r in originals)))
    macro = {}
    for metric in METRICS:
        values = [(r['family'],r['scene_token'],r['values'][metric]) for r in collapsed
                  if r['values'][metric] is not None]
        macro[metric] = scaled(family_macro(values,family_scenes),metric)
    result['family_macro'] = dict(metrics=macro, weighting='equal family',
                                  bootstrap='2000 within-family scene draws; seed 2026')
    return result


def language_controls(rows):
    """PCU-wise all-paraphrase success and canonical-minus-control gaps."""
    conditions={r['condition'] for r in rows}
    by_key={(r['pcu_id'],r['condition']):r for r in rows}
    ids=sorted({r['pcu_id'] for r in rows})
    result={}
    rewrites=WORDINGS[1:]
    if all(w in conditions for w in rewrites):
        result['Paraphrase_Worst_PCU']=100*sum(
            all(by_key[(pid,w)]['values']['SemCFPS']==1 for w in rewrites)
            for pid in ids)/len(ids)
    for condition in ('masked','shuffled'):
        if 'canonical' in conditions and condition in conditions:
            result[condition+'_CFPS_gap']=100*sum(
                by_key[(pid,'canonical')]['values']['SemCFPS']-
                by_key[(pid,condition)]['values']['SemCFPS'] for pid in ids)/len(ids)
    return result
