#!/usr/bin/env python3
"""Unified post-hoc PCU compliance and instruction response diagnostics."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_eval.semantic_metrics import (MODELS, CONDITIONS, RELEASE, ARTIFACTS,
    MapGeometry, DATA, read_predictions, corridor_stats, speed_stats, ade, aggregate)
from release_io import load_release, sha256
from score_lane_pcus import validate_trajectory, to_global

OUT = ROOT / 'model_eval/unified_semantic_diagnostics'


def progress_factor(xy):
    return min(1., math.hypot(*xy[-1]) / 8.)


def normalized_behavior_difference(a, b):
    numerator = sum(math.dist(x, y) for x, y in zip(a, b))
    denominator = sum(math.hypot(*x) + math.hypot(*y) for x, y in zip(a, b))
    return numerator / denominator if denominator > 0 else 0.


def coverage_matrix_metrics(matrix, valid_pair, response_matrix=None):
    """Coverage-only diagnostic; the paper SRG uses trajectory_preference."""
    own0, own1 = matrix[0][0], matrix[1][1]
    response = matrix if response_matrix is None else response_matrix
    return {
        'SCS': (own0 + own1) / 2.,
        'PCS': min(own0, own1),
        'SRG_coverage': (response[0][0] + response[1][1] - response[0][1] - response[1][0]) / 2. if valid_pair else None,
    }


def compliance(pcu, trajectory, contract_arm, corridors, shared_scope=False):
    if validate_trajectory(trajectory) is not None:
        return 0.
    xy = trajectory['xy_m']
    if pcu['family'] == 'scoped_speed':
        key = 'low' if contract_arm == 'arm_0' else 'high'
        coverage = speed_stats(xy, pcu['private_sidecar']['speed_bands_mps'][key],
                               15 if key == 'low' else 30)['speed_coverage']
    else:
        current, target = corridors[pcu['pcu_id']]
        coverage = corridor_stats(to_global(pcu, xy), target if contract_arm == 'arm_0' else current,
                                  2.25 if shared_scope or contract_arm == 'arm_0' else 0.)['lane_coverage']
    return progress_factor(xy) * coverage


def preference_from_distances(d0, d1):
    """Dimensionless preference for arm 0; overlapping targets are ambiguous."""
    return (d1-d0)/(d0+d1) if d0+d1 > 0 else 0.


def trajectory_preference(pcu, trajectory, corridors):
    """Relative contract distance on the shared scope, without progress gating."""
    xy, values = trajectory['xy_m'], []
    if pcu['family'] == 'scoped_speed':
        bands = pcu['private_sidecar']['speed_bands_mps']
        for i,(a,b) in enumerate(zip([[0.,0.]]+xy,xy)):
            if (i+1)*.25 < 1.25:
                continue
            v=math.dist(a,b)/.25
            ds=[max(bands[k][0]-v,0.,v-bands[k][1]) for k in ('low','high')]
            values.append(preference_from_distances(*ds))
    else:
        current,target=corridors[pcu['pcu_id']]
        for i,p in enumerate(to_global(pcu,xy)):
            if (i+1)*.25 < 2.25:
                continue
            point=Point(p)
            values.append(preference_from_distances(target.distance(point),current.distance(point)))
    return float(np.mean(values))


def make_corridors(public):
    maps, corridors = {}, {}
    for p in public:
        if p['family'] != 'lane_change_keep':
            continue
        loc = p['public_input']['location']
        if loc not in maps:
            maps[loc] = MapGeometry(loc)
        corridors[p['pcu_id']] = tuple(unary_union([
            maps[loc].lane_tolerances[t] for t in p['private_sidecar'][key]
            if t in maps[loc].lane_tolerances]) for key in ('current_lane_tokens', 'target_lane_tokens'))
        if any(c.is_empty for c in corridors[p['pcu_id']]):
            raise ValueError('Empty contract corridor')
    return maps, corridors


def family_macro(values, family_scenes):
    """Equal family weight, with independent scene-cluster bootstrap within each."""
    estimates, draws, count = [], [], 0
    rng = np.random.default_rng(2026)
    for family, scenes in family_scenes.items():
        totals, ns = defaultdict(float), defaultdict(int)
        for fam, scene, v in values:
            if fam == family:
                totals[scene] += v
                ns[scene] += 1
                count += 1
        if sum(ns.values()) == 0:
            return {'mean': None, 'n': count, 'ci95': [None, None]}
        sums = np.array([totals[s] for s in scenes])
        n = np.array([ns[s] for s in scenes])
        estimates.append(float(sums.sum()/n.sum()))
        picks = rng.integers(0,len(scenes),(2000,len(scenes)))
        denom = n[picks].sum(axis=1)
        sample = np.divide(sums[picks].sum(axis=1),denom,
                           out=np.full(2000,np.nan),where=denom>0)
        draws.append(sample)
    boots = np.mean(draws,axis=0)
    boots = boots[np.isfinite(boots)]
    return {'mean':float(np.mean(estimates)), 'n':count, 'ci95':np.quantile(boots,[.025,.975]).tolist()}
