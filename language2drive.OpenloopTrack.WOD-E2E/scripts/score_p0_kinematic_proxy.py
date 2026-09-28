"""WOD-E2E P0 task predicates, graded completion, and paired metric primitives.

Use freeze_paper_contracts.py before scoring. This module's command-line entry
forwards to evaluate_paper.py, which reads the frozen private contracts.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from score_prerelease import normalize_trajectory


VERSION = 'p0_kinematic_proxy_v0.1.0'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def clamp(value):
    return max(0.0, min(1.0, value))


def ramp_up(value, low, high):
    return clamp((value - low) / (high - low))


def ramp_down(value, low, high):
    return 1.0 - ramp_up(value, low, high)


def features(points):
    previous = [0.0, 0.0]
    speeds = []
    for point in points:
        speeds.append(math.dist(previous, point) * 4.0)
        previous = point
    return {'tail_speed_mps': sum(speeds[-4:]) / 4.0,
            'late_speed_mps': sum(speeds[11:]) / 9.0,
            'path_m': sum(speeds) / 4.0,
            'x5_m': points[-1][0], 'xy5_m': points[-1]}


def instruction_arm(candidate, first, second):
    labels = [candidate['instruction_a_draft'], candidate['instruction_b_draft']]
    a = [i for i, value in enumerate(labels) if value.startswith(first)]
    b = [i for i, value in enumerate(labels) if value.startswith(second)]
    if len(a) != 1 or len(b) != 1 or a[0] == b[0]:
        return None
    return a[0], b[0]


def stop_go_contract(candidate):
    assignment = instruction_arm(candidate, 'Stop smoothly', 'Continue driving')
    if assignment is None:
        return None, 'unrecognized_stop_go_instruction'
    stop, go = assignment
    stop_ref = candidate['references'][stop]['xy_m']
    contract = {'family': 'STOP_GO', 'stop_arm': stop, 'go_arm': go,
                'stop_xy5_m': stop_ref[-1],
                'stop_tail_speed_max_mps': 0.6,
                'go_tail_speed_min_mps': 2.0,
                'go_x5_min_m': 8.0,
                'stop_position_full_radius_m': 2.0,
                'stop_position_zero_radius_m': 5.0}
    return contract, None


def speed_contract(candidate):
    assignment = instruction_arm(candidate, 'Slow down', 'Maintain a normal')
    if assignment is None:
        return None, 'unrecognized_speed_instruction'
    slow, normal = assignment
    references = [features(value['xy_m']) for value in candidate['references']]
    low = references[slow]['late_speed_mps']
    high = references[normal]['late_speed_mps']
    gap = high - low
    if gap < 1.0:
        return None, 'reference_late_speed_gap_below_1mps'
    contract = {'family': 'SPEED_CHOICE', 'slow_arm': slow,
                'normal_arm': normal, 'slow_reference_late_speed_mps': low,
                'normal_reference_late_speed_mps': high,
                'slow_accept_max_mps': low + 0.25 * gap,
                'normal_accept_min_mps': high - 0.25 * gap,
                'minimum_path_m': 2.0}
    return contract, None


def primitive(feature, target, contract):
    if contract['family'] == 'STOP_GO':
        if target == 'stop':
            distance = math.dist(feature['xy5_m'], contract['stop_xy5_m'])
            speed = ramp_down(feature['tail_speed_mps'], 0.6, 2.0)
            position = ramp_down(distance, 2.0, 5.0)
            return (feature['tail_speed_mps'] <= 0.6 and distance <= 2.0,
                    min(speed, position),
                    {'tail_speed_mps': feature['tail_speed_mps'],
                     'stop_xy5_distance_m': distance})
        speed = ramp_up(feature['tail_speed_mps'], 0.6, 2.0)
        progress = ramp_up(feature['x5_m'], 0.0, 8.0)
        return (feature['tail_speed_mps'] >= 2.0 and feature['x5_m'] >= 8.0,
                min(speed, progress),
                {'tail_speed_mps': feature['tail_speed_mps'],
                 'x5_m': feature['x5_m']})
    slow_max = contract['slow_accept_max_mps']
    normal_min = contract['normal_accept_min_mps']
    velocity = feature['late_speed_mps']
    progress = ramp_up(feature['path_m'], 0.0, 2.0)
    if target == 'slow':
        speed = ramp_down(velocity, slow_max, normal_min)
        return (velocity <= slow_max and feature['path_m'] >= 2.0,
                min(speed, progress),
                {'late_speed_mps': velocity, 'path_m': feature['path_m']})
    speed = ramp_up(velocity, slow_max, normal_min)
    return (velocity >= normal_min and feature['path_m'] >= 2.0,
            min(speed, progress),
            {'late_speed_mps': velocity, 'path_m': feature['path_m']})


def targets(contract):
    if contract['family'] == 'STOP_GO':
        names = ['go', 'go']
        names[contract['stop_arm']] = 'stop'
    else:
        names = ['normal', 'normal']
        names[contract['slow_arm']] = 'slow'
    return names


def build_contract(candidate):
    family = candidate['task_family_draft']
    if family == 'STOP_GO':
        contract, reason = stop_go_contract(candidate)
    elif family == 'SPEED_CHOICE':
        contract, reason = speed_contract(candidate)
    else:
        return None, 'family_needs_scene_geometry'
    if reason:
        return None, reason
    names = targets(contract)
    for arm in (0, 1):
        feature = features(candidate['references'][arm]['xy_m'])
        own, _, _ = primitive(feature, names[arm], contract)
        opposite, _, _ = primitive(feature, names[1 - arm], contract)
        if not own or opposite:
            return None, 'reference_witness_fails_exclusive_proxy'
    return contract, None


def score_arm(candidate, arm, contract, predictions):
    key = 'a' if arm == 0 else 'b'
    request_id = candidate['request_ids'][key]
    row = predictions.get(request_id)
    if row is None:
        return {'request_id': request_id, 'status': 'missing',
                'exclusive_success': False, 'own_completion': 0.0}
    normalized, reason = normalize_trajectory(row.get('trajectory'))
    if reason:
        return {'request_id': request_id, 'status': 'invalid', 'reason': reason,
                'exclusive_success': False, 'own_completion': 0.0}
    feature = features(normalized[0])
    names = targets(contract)
    own, completion, detail = primitive(feature, names[arm], contract)
    opposite, _, _ = primitive(feature, names[1 - arm], contract)
    return {'request_id': request_id, 'status': 'valid',
            'own_pass': own, 'opposite_pass': opposite,
            'exclusive_success': own and not opposite,
            'own_completion': completion, 'feature': detail}


def model_arg(value):
    name, marker, path = value.partition('=')
    if not marker or not name or not path:
        raise argparse.ArgumentTypeError('Expected MODEL=predictions.jsonl')
    return name, Path(path)


def metric_summary(records):
    count = len(records)
    if not count:
        return None
    arms = [item[arm] for item in records for arm in ('a', 'b')]
    return {'pcus': count, 'requests': count * 2,
            'predicted': sum(x['status'] != 'missing' for x in arms),
            'valid': sum(x['status'] == 'valid' for x in arms),
            'invalid': sum(x['status'] == 'invalid' for x in arms),
            'exclusive_success_arms': sum(x['exclusive_success'] for x in arms),
            'exclusive_success_pairs': sum(x['a']['exclusive_success'] and
                                           x['b']['exclusive_success'] for x in records),
            'SemSISR_P0_kinematic_proxy_0to100':
                100.0 * sum(x['exclusive_success'] for x in arms) / (2 * count),
            'SemCFPS_P0_kinematic_proxy_0to100':
                100.0 * sum(x['a']['exclusive_success'] and
                            x['b']['exclusive_success'] for x in records) / count,
            'SCS_P0_kinematic_proxy_0to100':
                100.0 * sum(x['own_completion'] for x in arms) / (2 * count),
            'PCS_P0_kinematic_proxy_0to100':
                100.0 * sum(min(x['a']['own_completion'], x['b']['own_completion'])
                            for x in records) / count}


def main():
    from evaluate_paper import main as paper_main
    paper_main()


if __name__ == '__main__':
    main()
