"""Public-only requests and native open-loop submission serialization."""
from pathlib import Path
import hashlib
import json
import sys
import uuid
from .adapter import Request, SPECS
from .service import Client, image_payload


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def image(path, entry=None):
    from PIL import Image
    import numpy as np
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Observation image not found: {path}')
    if entry and entry.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'Image checksum mismatch: {path}')
    with Image.open(path) as im:
        return image_payload(np.asarray(im.convert('RGB')))


def queries(track, backend, cfg):
    """Yield evaluator identities separately from the exact model request."""
    backend = Path(backend)
    seed = int(cfg.get('seed', 0))
    if track == 'nuscenes':
        sys.path.insert(0, str(backend))
        from release_io import load_release
        from run_model import instruction_for
        release = Path(cfg.get('release_dir', backend / 'releases/nuscenes-test-v1.0.0'))
        _, records = load_release(release, split=cfg.get('split', 'heldout'))
        conditions = cfg.get('conditions', ['canonical'])
        if any(c not in ('canonical', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3') for c in conditions):
            raise ValueError('Unified run supports canonical and three paraphrases; use native ablation runner for masked/shuffled')
        sensor_root = Path(cfg['sensor_root'])
        calibration_file = sensor_root / cfg.get('metadata_version', 'v1.0-test') / 'calibrated_sensor.json'
        calibrations = {r['token']: {k: r[k] for k in ('translation', 'rotation', 'camera_intrinsic') if k in r} for r in json.loads(calibration_file.read_text())} if calibration_file.is_file() else {}
        for row in records:
            public = row['public_input']
            cameras = {}
            for frame in [*public.get('history_refs', []), public]:
                for name, ref in frame['sensor_refs'].items():
                    if name.startswith('CAM'):
                        cameras.setdefault(name, []).append({'timestamp_us': ref.get('timestamp_us'), 'calibration': calibrations.get(ref.get('calibrated_sensor_token')), 'image': image(sensor_root / ref['filename'])})
            observation = {'cameras': cameras, 'ego_history': [r['ego_pose'] for r in public.get('history_refs', [])],
                           'ego_pose': public.get('ego_pose')}
            # History references are public sensor/ego metadata; never copy the enclosing row.
            for condition in conditions:
                for arm in ('arm_0', 'arm_1'):
                    meta = dict(pcu_id=row['pcu_id'], arm=arm, condition=condition, seed=seed,
                                public_input_hash=public['public_input_hash'])
                    yield meta, Request(track, instruction_for(row, arm, condition, {}), observation, SPECS[track], seed)
    elif track == 'navsim':
        public_root = Path(cfg.get('public_dir', backend / 'data/public'))
        sensor_root = Path(cfg['sensor_root'])
        manifest = json.loads((public_root / 'manifest.json').read_text())
        for pcu in manifest['pcu_ids']:
            row = json.loads((public_root / 'public_inputs' / (pcu + '.json')).read_text())
            cameras = {}
            for frame in row['camera_paths_history']:
                for name, path in frame.items():
                    cameras.setdefault(name, []).append({'image': image(sensor_root / path)})
            observation = {'cameras': cameras, 'ego_history': row['ego_history']}
            for condition in cfg.get('conditions', ['canonical', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3']):
                if condition not in ('canonical', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3'):
                    raise ValueError('Invalid NAVSIM wording')
                for arm in range(2):
                    text = row['instructions'][arm] if condition == 'canonical' else row['paraphrases'][arm][int(condition[-1])-1]
                    yield dict(pcu_id=pcu, arm=str(arm), condition=condition, seed=seed), Request(track, text, observation, SPECS[track], seed)
    elif track == 'wod':
        release = Path(cfg.get('release_dir', backend / 'prerelease/v0.1.0'))
        rows = {r['input_id']: r for r in read_jsonl(release / 'public/inputs.jsonl')}
        for item in read_jsonl(release / 'public/workload.jsonl'):
            row = rows[item['input_id']]
            cameras = {name: [{'offset_s': e.get('offset_s'), 'image': image(release / 'public' / e['path'], e)} for e in frames]
                       for name, frames in row['camera_history'].items()}
            observation = {'cameras': cameras, 'past_states': row['past_states'], 'calibration': row['calibration'], 'time_axis': row['time_axis']}
            yield {'request_id': item['request_id']}, Request(track, item['instruction'], observation, SPECS[track], seed)
    else:
        raise ValueError(f'Not an open-loop track: {track}')


def run_openloop(track, backend, cfg, output, endpoint, model_identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / 'predictions.ledger.jsonl'
    client = Client(endpoint)
    binding = {'track': track, 'config': cfg, 'model': model_identity, 'service': client.identity()}
    binding_path = output / 'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Output belongs to a different model/configuration. Choose a new output directory.')
    binding_path.write_text(json.dumps(binding, indent=2))
    completed = {}
    if ledger.exists():
        for row in read_jsonl(ledger):
            if row['key'] in completed:
                raise ValueError('Duplicate inference ledger entry')
            completed[row['key']] = row
    client = Client(endpoint)
    expected, failures = [], 0
    with ledger.open('a', encoding='utf-8') as handle:
        for meta, request in queries(track, backend, cfg):
            key = digest({'identity': meta, 'request': request.to_dict()})
            expected.append(key)
            if key not in completed:
                session = uuid.uuid4().hex
                try:
                    client.reset(session)
                    trajectory = client.predict(request, session)
                    row = {'key': key, 'meta': meta, 'instruction': request.instruction, 'trajectory': trajectory, 'error': None}
                except Exception as error:
                    row = {'key': key, 'meta': meta, 'instruction': request.instruction, 'trajectory': None, 'error': str(error)}
                finally:
                    try:
                        client.call('close', {'session': session})
                    except Exception:
                        pass
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                handle.flush()
                completed[key] = row
            failures += completed[key]['error'] is not None
    if not expected or set(completed) != set(expected):
        raise ValueError('Empty workload or ledger contains entries outside the scheduled inventory')
    rows = [completed[key] for key in expected]
    export(track, rows, output)
    report = {'scheduled_requests': len(rows), 'failed_requests': failures, 'predictions': str(output), 'status': 'inference_complete' if not failures else 'inference_failures'}
    (output / 'inference.json').write_text(json.dumps(report, indent=2))
    return report


def export(track, rows, output):
    output = Path(output)
    if track == 'nuscenes':
        with (output / 'predictions.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(dict(row['meta'], instruction=row['instruction'], trajectory=row['trajectory'])) + '\n')
    elif track == 'wod':
        with (output / 'predictions.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps({'request_id': row['meta']['request_id'], 'trajectory': row['trajectory']['xy_m'] if row['trajectory'] else []}) + '\n')
    else:
        wordings = {}
        for row in rows:
            meta, trajectory = row['meta'], row['trajectory']
            native = {} if trajectory is None else {'coordinate_frame': trajectory['coordinate_frame'], 'reference_point': trajectory['reference_point'], 'timestamps': trajectory['timestamps_s'], 'positions_xy': trajectory['xy_m']}
            wordings.setdefault(meta['condition'], {}).setdefault(meta['pcu_id'], {})[meta['arm']] = native
        (output / 'canonical.json').write_text(json.dumps(wordings.get('canonical', {}), indent=2))
        (output / 'rewrites.json').write_text(json.dumps(wordings, indent=2))
