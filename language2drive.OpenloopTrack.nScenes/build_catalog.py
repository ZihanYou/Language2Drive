#!/usr/bin/env python3
"""Create a reproducible anchor catalog from nuScenes v1.0-test only."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


DATA_ROOT = DATA_ROOT
OUTPUT_ROOT = (ARTIFACT_ROOT / 'scene_catalog')
CHANNELS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP')


def load_table(metadata, name):
    return json.loads((metadata / f'{name}.json').read_text())


def yaw_from_quaternion(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def canonical_hash(value):
    packed = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return 'sha256:' + hashlib.sha256(packed).hexdigest()


def build(data_root, output_root):
    metadata = data_root / 'v1.0-test'
    if not metadata.is_dir():
        raise FileNotFoundError(metadata)
    scenes = {x['token']: x for x in load_table(metadata, 'scene')}
    logs = {x['token']: x for x in load_table(metadata, 'log')}
    samples = {x['token']: x for x in load_table(metadata, 'sample')}
    if load_table(metadata, 'sample_annotation') or load_table(metadata, 'instance'):
        raise ValueError('Unexpected public object annotations in v1.0-test; inspect data version')

    sensors = {x['token']: x['channel'] for x in load_table(metadata, 'sensor')}
    calibrated = {x['token']: x for x in load_table(metadata, 'calibrated_sensor')}
    keyframes = defaultdict(dict)
    pose_tokens = set()
    for item in load_table(metadata, 'sample_data'):
        if not item['is_key_frame']:
            continue
        channel = sensors[calibrated[item['calibrated_sensor_token']]['sensor_token']]
        if channel not in CHANNELS:
            continue
        keyframes[item['sample_token']][channel] = {
            'token': item['token'],
            'timestamp_us': item['timestamp'],
            'filename': item['filename'],
            'ego_pose_token': item['ego_pose_token'],
            'calibrated_sensor_token': item['calibrated_sensor_token'],
        }
        if channel == 'LIDAR_TOP':
            pose_tokens.add(item['ego_pose_token'])
    poses = {}
    for pose in load_table(metadata, 'ego_pose'):
        if pose['token'] in pose_tokens:
            poses[pose['token']] = {
                'xy_m': pose['translation'][:2],
                'z_m': pose['translation'][2],
                'yaw_rad': yaw_from_quaternion(pose['rotation']),
                'timestamp_us': pose['timestamp'],
                'quaternion_wxyz': pose['rotation'],
            }
    if len(poses) != len(pose_tokens):
        raise ValueError('A keyframe ego pose is missing')

    output_root.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    missing = Counter()
    with (output_root / 'anchors.jsonl').open('w') as out:
        for scene in sorted(scenes.values(), key=lambda x: x['name']):
            location = logs[scene['log_token']]['location']
            if not (data_root / 'maps' / 'expansion' / f'{location}.json').exists():
                raise FileNotFoundError(f'map for {location}')
            token = scene['first_sample_token']
            history = []
            visited = set()
            while token:
                if token in visited:
                    raise ValueError(f'Cycle in sample chain for {scene["name"]}')
                visited.add(token)
                sample = samples[token]
                refs = keyframes[token]
                absent = [c for c in CHANNELS if c not in refs]
                if absent:
                    missing['channels'] += 1
                for channel, ref in refs.items():
                    if not (data_root / ref['filename']).is_file():
                        missing['sensor_files'] += 1
                lidar = refs.get('LIDAR_TOP')
                if lidar is None:
                    missing['lidar_pose'] += 1
                    token = sample['next']
                    continue
                pose = poses[lidar['ego_pose_token']]
                history_refs = []
                for old_token in history[-4:]:
                    old_refs = keyframes[old_token]
                    old_lidar = old_refs.get('LIDAR_TOP')
                    if old_lidar is None:
                        continue
                    history_refs.append({
                        'sample_token': old_token,
                        'timestamp_us': samples[old_token]['timestamp'],
                        'ego_pose': poses[old_lidar['ego_pose_token']],
                        'sensor_refs': old_refs,
                    })
                shared = {
                    'dataset_version': 'v1.0-test',
                    'scene_token': scene['token'],
                    'sample_token': token,
                    'timestamp_us': sample['timestamp'],
                    'location': location,
                    'ego_pose': pose,
                    'sensor_refs': refs,
                    'history_sample_tokens': history[-4:],
                    'history_refs': history_refs,
                }
                row = dict(shared, scene_name=scene['name'],
                           log_token=scene['log_token'],
                           next_sample_token=sample['next'],
                           public_input_hash=canonical_hash(shared))
                out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                counts['anchors'] += 1
                counts[location] += 1
                history.append(token)
                token = sample['next']
            if len(visited) != scene['nbr_samples']:
                raise ValueError(f'Scene sample count differs for {scene["name"]}')
            counts['scenes'] += 1
    summary = {'dataset_version': 'v1.0-test', 'data_root': str(data_root),
               'counts': dict(counts), 'missing': dict(missing),
               'annotation_records': 0, 'instance_records': 0,
               'hash_algorithm': 'sha256/canonical-json'}
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.data_root, args.output_root), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
