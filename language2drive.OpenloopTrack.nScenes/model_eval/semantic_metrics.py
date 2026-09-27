#!/usr/bin/env python3
"""Post-hoc continuous semantic diagnostics from existing frozen predictions.

No model inference, witness fitting, or changes to the official v1.0.0 score.
"""
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

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT
from build_lane_pcus import MapGeometry, TIMES, DATA
from release_io import load_release, sha256
from score_lane_pcus import validate_trajectory, to_global

RELEASE = RELEASE_ROOT
ARTIFACTS = ARTIFACT_ROOT / 'model_eval'
OUT = ROOT / 'model_eval/semantic_diagnostics'
CONDITIONS = ('canonical', 'masked', 'shuffled', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3')
MODELS = ('Constant velocity', 'OpenDriveVLA-0.5B', 'Impromptu-VLA-3B-3view',
          'UniDriveVLA-Base-Stage3', 'OmniDrive')
SPEED_RESPONSE_KMH = 1.0
LANE_RESPONSE_M = 0.2


def speed_stats(xy, band, target_kmh):
    speeds = [math.dist(a, b) / 0.25 for a, b in zip([[0., 0.]] + xy, xy)]
    scoped = [v for t, v in zip(TIMES, speeds) if t >= 1.25]
    low, high = band
    return {
        'speed_mean_kmh': float(np.mean(scoped) * 3.6),
        'speed_target_mae_kmh': float(np.mean([abs(v * 3.6 - target_kmh) for v in scoped])),
        'speed_band_violation_kmh': float(np.mean([max(low-v, 0., v-high) for v in scoped]) * 3.6),
        'speed_coverage': float(np.mean([low <= v <= high for v in scoped])),
        'speed_robustness_kmh': min(min(v-low, high-v) for v in scoped) * 3.6,
    }


def corridor_stats(points, corridor, hold_from):
    pts = [Point(p) for t, p in zip(TIMES, points) if t >= hold_from]
    distances = [corridor.distance(p) for p in pts]
    margins = [p.distance(corridor.boundary) if corridor.covers(p) else -d
               for p, d in zip(pts, distances)]
    return {'lane_outside_m': float(np.mean(distances)),
            'lane_coverage': float(np.mean([corridor.covers(p) for p in pts])),
            'lane_robustness_m': min(margins)}


def lane_preference(points, current, target):
    """Positive means closer to target corridor; shared late window for both arms."""
    return float(np.mean([current.distance(Point(p)) - target.distance(Point(p))
                          for p, t in zip(points, TIMES) if t >= 2.25]))


def ade(a, b):
    return float(np.mean([math.dist(x, y) for x, y in zip(a, b)]))


def paired_response(family, arm0, arm1):
    if family == 'scoped_speed':
        return arm1['speed_mean_kmh'] - arm0['speed_mean_kmh']
    return arm0['lane_preference_m'] - arm1['lane_preference_m']


def aggregate(values, scenes):
    """Arm-weighted mean; bootstrap whole scenes, retaining within-scene pairing."""
    scenes = sorted(set(scenes))
    sums, counts = defaultdict(float), defaultdict(int)
    for scene, value in values:
        sums[scene] += value
        counts[scene] += 1
    if not values:
        return {'mean': None, 'n': 0, 'ci95': [None, None]}
    totals = np.array([sums[s] for s in scenes])
    ns = np.array([counts[s] for s in scenes])
    picks = np.random.default_rng(2026).integers(0, len(scenes), (2000, len(scenes)))
    denom = ns[picks].sum(axis=1)
    valid = denom > 0
    samples = totals[picks].sum(axis=1)[valid] / denom[valid]
    return {'mean': float(totals.sum()/ns.sum()), 'n': len(values),
            'ci95': np.quantile(samples, [.025, .975]).tolist()}


def read_predictions(path, pcus, check_snapshot=False):
    if check_snapshot:
        provenance = json.loads((path.parent / 'provenance.json').read_text())
        if sha256(path) != provenance['scored_predictions_sha256']:
            raise ValueError('Score snapshot hash mismatch: ' + str(path))
    records = {}
    for line in path.read_text().splitlines():
        x = json.loads(line)
        if x['pcu_id'] not in pcus:
            continue
        key = (x['pcu_id'], x['arm'], x['condition'])
        if key in records or x['seed'] != 0:
            raise ValueError('Duplicate or wrong seed: ' + str(key))
        if x['public_input_hash'] != pcus[x['pcu_id']]['public_input']['public_input_hash']:
            raise ValueError('Input hash mismatch: ' + str(key))
        records[key] = x['trajectory']
    expected = {(p, a, c) for p in pcus for a in ('arm_0', 'arm_1') for c in CONDITIONS}
    if set(records) != expected:
        raise ValueError('Incomplete or unexpected prediction set: ' + str(path))
    return records
