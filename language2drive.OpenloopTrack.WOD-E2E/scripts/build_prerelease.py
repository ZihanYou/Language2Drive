"""Build an unreviewed WOD-E2E stop/proceed pilot from local rated validation frames.

Public files contain only causal observations and English requests. Rated
references, selection evidence and request labels stay in the private sidecar.
This is a proxy pilot; it does not certify scene semantics or driving safety.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

from audit_local_data import fields


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT',
           5: 'SIDE_RIGHT', 6: 'REAR_LEFT', 7: 'REAR', 8: 'REAR_RIGHT'}
FRONT = ('FRONT', 'FRONT_LEFT', 'FRONT_RIGHT')
OFFSETS = (-15, -10, -5, 0)  # WOD 10 Hz: -1.5, -1.0, -0.5, 0 s
STOP_TEXT = 'Stop the vehicle smoothly and remain stopped.'
GO_TEXT = 'Proceed forward along the current drivable route and keep moving.'
VERSION = 'wod-e2e-stop-go-pilot-v0.1.0'


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def movement(points):
    previous = (0.0, 0.0)
    speeds = []
    for point in points:
        speeds.append(math.dist(point, previous) * 4)
        previous = point
    return {'path_m': sum(speeds) / 4, 'tail_mps': sum(speeds[-4:]) / 4,
            'x5_m': points[19][0], 'y5_m': points[19][1],
            'initial_interval_mps': speeds[0],
            'max_abs_accel_mps2': max(abs(b-a)*4 for a,b in zip(speeds,speeds[1:])),
            'min_dx_m': min(points[i][0] - (points[i-1][0] if i else 0)
                            for i in range(20)),
            'heading_deg': math.degrees(math.atan2(points[19][1]-points[15][1],
                                                   points[19][0]-points[15][0]))}


def select_pairs(queue, rated, normalized):
    eligible = []
    rejection = {}
    for row in queue:
        key = row['frame_name']
        source = rated[key]
        references = {str(t['reference_index']): t for t in normalized[key]['preferences']}
        a, b = references[row['reference_a']], references[row['reference_b']]
        reason = None
        if source['intent'] != 1:
            reason = 'non_straight_shared_route'
        elif min(a['score'], b['score']) < 6:
            reason = 'rated_score_below_6'
        elif len(a['xy_m']) != 20 or len(b['xy_m']) != 20:
            reason = 'incomplete_5s_witness'
        if reason:
            rejection[reason] = rejection.get(reason, 0) + 1
            continue
        ma, mb = movement(a['xy_m']), movement(b['xy_m'])
        stop, go = (0, 1) if ma['tail_mps'] <= 0.6 and mb['tail_mps'] >= 2 else (1, 0) if mb['tail_mps'] <= 0.6 and ma['tail_mps'] >= 2 else (None, None)
        if stop is None:
            reason = 'no_disjoint_stop_go_witness'
        else:
            ms, mg = (ma, mb) if stop == 0 else (mb, ma)
            initial = math.hypot(source['past']['vx'][-1], source['past']['vy'][-1])
            if mg['path_m'] - ms['path_m'] < 8 or mg['x5_m'] < 8:
                reason = 'insufficient_forward_separation'
            elif abs(mg['y5_m']) >= 3 or abs(mg['heading_deg']) >= 15:
                reason = 'go_witness_not_straight'
            elif any(m['max_abs_accel_mps2'] > 8 or abs(m['initial_interval_mps']-initial) > 2 or m['min_dx_m'] < -0.5 for m in (ma, mb)):
                reason = 'continuity_or_acceleration_filter'
        if reason:
            rejection[reason] = rejection.get(reason, 0) + 1
            continue
        rank = (min(a['score'], b['score']), mg['path_m']-ms['path_m'], row['candidate_id'])
        eligible.append((rank, row, (a,b), (ma,mb), stop, go))
    # Pick at most one candidate per sequence. Highest min score, then largest
    # action separation; the ID is a deterministic final tie-breaker.
    chosen = {}
    for item in sorted(eligible, key=lambda x: (-x[0][0], -x[0][1], x[0][2])):
        sequence = item[1]['sequence_id']
        chosen.setdefault(sequence, item)
    result = sorted(chosen.values(), key=lambda x: x[1]['candidate_id'])
    rejection['duplicate_sequence'] = len(eligible) - len(result)
    return result, rejection


def index_selected_records(selected, source_root):
    sequences = set()
    for _, row, *_ in selected:
        sequences.add(row['sequence_id'])
    locations = {}
    # TFRecord shards distribute frames from a sequence across files. Scan all
    # locally available shards; the anchor's shard is not a history index.
    for source in sorted(source_root.glob('*.tfrecord-*')):
        filename = str(source)
        with open(filename, 'rb') as f:
            end_file = os.fstat(f.fileno()).st_size
            while f.tell() < end_file:
                offset = f.tell()
                header = f.read(12)
                if len(header) != 12:
                    raise EOFError(f'truncated TFRecord header: {filename}')
                length = struct.unpack('<Q', header[:8])[0]
                end = f.tell() + length
                name = None
                for n,w,start,size in fields(f,end):
                    if n == 1 and w == 2:
                        for fn,fw,fs,fz in fields(f,start+size):
                            if fn == 1 and fw == 2:
                                for cn,cw,cs,cz in fields(f,fs+fz):
                                    if cn == 1 and cw == 2:
                                        name = f.read(cz).decode('utf-8')
                                        break
                                break
                        break
                if name and name.rsplit('-',1)[0] in sequences:
                    locations[name] = (filename, offset)
                f.seek(end+4)
        print(f'Indexed {source.name}: {len(locations)} frames in selected sequences', flush=True)
    return locations


def choose_temporal_names(anchor, locations):
    seq, suffix = anchor.rsplit('-',1)
    index = int(suffix)
    available = {int(name.rsplit('-',1)[1]):name for name in locations
                 if name.startswith(seq+'-')}
    picked = []
    for delta in OFFSETS:
        expected = index+delta
        candidates = [k for k in available if abs(k-expected) <= 1]
        if not candidates:
            return None
        actual = min(candidates,key=lambda k:(abs(k-expected), k))
        picked.append((actual-index,available[actual],delta))
    if len({name for _,name,_ in picked}) != 4 or picked[-1][1] != anchor:
        return None
    return picked


def unpack_doubles(f, wire, start, size):
    if wire == 1:
        f.seek(start)
        return [struct.unpack('<d', f.read(8))[0]]
    if wire == 2 and size % 8 == 0:
        f.seek(start)
        return list(struct.unpack('<'+'d'*(size//8), f.read(size)))
    return []


def read_frame(filename, offset, desired):
    images, calibration = {}, {}
    with open(filename, 'rb') as f:
        f.seek(offset)
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(4)
        end = f.tell() + length
        for n,w,start,size in fields(f,end):
            if n != 1 or w != 2:
                continue
            for fn,fw,fs,fz in fields(f,start+size):
                if fn == 1 and fw == 2:
                    for cn,cw,cs,cz in fields(f,fs+fz):
                        if cn != 2 or cw != 2:
                            continue
                        item = {'intrinsic': [], 'extrinsic': []}
                        camera = None
                        for kn,kw,ks,kz in fields(f,cs+cz):
                            if kn == 1 and kw == 0:
                                camera = CAMERAS.get(ks)
                            elif kn == 2:
                                item['intrinsic'].extend(unpack_doubles(f,kw,ks,kz))
                            elif kn == 3 and kw == 2:
                                for en,ew,es,ez in fields(f,ks+kz):
                                    if en == 1:
                                        item['extrinsic'].extend(unpack_doubles(f,ew,es,ez))
                            elif kn == 4 and kw == 0:
                                item['width'] = ks
                            elif kn == 5 and kw == 0:
                                item['height'] = ks
                        if camera:
                            calibration[camera] = item
                elif fn == 4 and fw == 2:
                    camera, jpeg = None, None
                    for imn,imw,ims,imz in fields(f,fs+fz):
                        if imn == 1 and imw == 0:
                            camera = CAMERAS.get(ims)
                        elif imn == 2 and imw == 2 and camera in desired:
                            f.seek(ims)
                            jpeg = f.read(imz)
                    if camera in desired and jpeg is not None:
                        images[camera] = jpeg
    return images, calibration


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(canonical(row).decode('utf-8') + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'prerelease/pilot-v0.1.0')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f'Prerelase output already exists: {out}; choose a new version')
    queue = list(csv.DictReader((ROOT/'review/review_queue.csv').open(newline='', encoding='utf-8')))
    global RATED_ROWS
    RATED_ROWS = read_jsonl(ROOT/'artifacts/rated_frames.jsonl')
    rated = {r['frame_name']:r for r in RATED_ROWS}
    normalized = {r['frame_name']:r for r in read_jsonl(ROOT/'artifacts/normalized_rated_frames.jsonl')}
    chosen, rejected = select_pairs(queue, rated, normalized)
    locations = index_selected_records(chosen, args.source_root)
    selected = []
    for item in chosen:
        frame = item[1]['frame_name']
        temporal = choose_temporal_names(frame,locations)
        if temporal is not None:
            selected.append((item,temporal))
        else:
            rejected['missing_temporal_frame_within_0p1s'] = rejected.get('missing_temporal_frame_within_0p1s',0)+1
    # Stable split: 20% dev by hash, the rest pilot_eval. Sequence is the unit.
    ordered = sorted(selected, key=lambda z: digest(z[0][1]['sequence_id'].encode()))
    dev_sequences = {x[0][1]['sequence_id'] for x in ordered[:max(1, round(len(ordered)*0.2))]}
    staging = out.with_name(out.name+'.building')
    if staging.exists():
        raise FileExistsError(f'Previous incomplete build: {staging}')
    (staging/'public/images').mkdir(parents=True)
    (staging/'private').mkdir()
    public_inputs, workload, targets, assets = [], [], [], []
    for number, (item,temporal) in enumerate(selected,1):
        _, row, references, motions, stop_idx, go_idx = item
        candidate = row['candidate_id']
        frame = row['frame_name']
        input_id = 'I-' + digest(frame.encode())[:12]
        unit_id = f'PRE-{number:04d}'
        split = 'dev' if row['sequence_id'] in dev_sequences else 'pilot_eval'
        current_images, calibration = read_frame(*locations[frame], set(CAMERAS.values()))
        if set(current_images) != set(CAMERAS.values()):
            raise ValueError(f'{frame}: incomplete current 8-camera set')
        if any(len(calibration.get(cam,{}).get('intrinsic',[])) < 4 or
               len(calibration.get(cam,{}).get('extrinsic',[])) != 16
               for cam in CAMERAS.values()):
            raise ValueError(f'{frame}: incomplete calibration')
        image_map = {}
        history = {camera: [] for camera in FRONT}
        for actual_delta,name,nominal_delta in temporal:
            images = current_images if actual_delta == 0 else read_frame(*locations[name], set(FRONT))[0]
            needed = set(CAMERAS.values()) if actual_delta == 0 else set(FRONT)
            if set(images) != needed:
                raise ValueError(f'{name}: incomplete camera set')
            for camera,blob in sorted(images.items()):
                if not blob.startswith(b'\xff\xd8'):
                    raise ValueError(f'{name}/{camera}: invalid JPEG header')
                rel = f'images/{input_id}/{actual_delta:+04d}_{camera}.jpg'
                target = staging/'public'/rel
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(blob)
                entry = {'path':rel, 'sha256':digest(blob),
                         'offset_s':actual_delta/10,
                         'nominal_offset_s':nominal_delta/10}
                assets.append({'input_id':input_id,'camera':camera,**entry})
                if actual_delta == 0:
                    image_map[camera] = entry
                if camera in FRONT:
                    history[camera].append(entry)
        shared = {'input_id':input_id,'frame_name':frame,'sequence_id':row['sequence_id'],
                  'split':split,'route_command':'GO_STRAIGHT','images':image_map,
                  'camera_history':history,'calibration':calibration,
                  'past_states':rated[frame]['past'],
                  'time_axis':{'camera_hz':10,
                               'camera_temporal_selection':'nearest_observed_within_0p1s',
                               'max_temporal_jitter_s':0.1,'past_states_hz':4,
                               'trajectory_hz':4,'trajectory_seconds':5}}
        shared['shared_input_hash'] = digest(canonical(shared))
        public_inputs.append(shared)
        request_ids = {}
        for label, instruction in (('stop',STOP_TEXT),('go',GO_TEXT)):
            request_id = 'R-' + digest(f'{unit_id}:{label}'.encode())[:12]
            request_ids[label] = request_id
            workload.append({'request_id':request_id,'input_id':input_id,
                             'instruction':instruction,'instruction_language':'en'})
        targets.append({'unit_id':unit_id,'source_candidate_id':candidate,
                        'frame_name':frame,'sequence_id':row['sequence_id'],
                        'scenario_cluster':row['scenario_cluster'],'split':split,
                        'input_id':input_id,'shared_input_hash':shared['shared_input_hash'],
                        'request_to_contract':{request_ids['stop']:'STOP_TERMINAL',request_ids['go']:'GO_TERMINAL'},
                        'contracts':{'STOP_TERMINAL':{'tail_mean_speed_mps_max':0.6},
                                     'GO_TERMINAL':{'tail_mean_speed_mps_min':2.0,'x5_m_min':8.0}},
                        'references':[{'source_reference_index':int(row['reference_a'] if i==0 else row['reference_b']),
                                       'score':ref['score'],'contract':'STOP_TERMINAL' if i==stop_idx else 'GO_TERMINAL',
                                       'xy_m':ref['xy_m'],'motion':motions[i]}
                                      for i,ref in enumerate(references)],
                        'status':'unreviewed_motion_proxy','safety_status':'unknown',
                        'semantic_status':'unknown'})
    write_jsonl(staging/'public/inputs.jsonl', public_inputs)
    write_jsonl(staging/'public/workload.jsonl', workload)
    write_jsonl(staging/'private/targets.jsonl', targets)
    write_jsonl(staging/'public/image_manifest.jsonl', assets)
    sources = {name:digest((ROOT/name).read_bytes()) for name in
               ('review/review_queue.csv','artifacts/rated_frames.jsonl',
                'artifacts/normalized_rated_frames.jsonl')}
    files = [staging/name for name in ('public/inputs.jsonl','public/workload.jsonl',
                                       'public/image_manifest.jsonl','private/targets.jsonl')]
    manifest = {'version':VERSION,'release_status':'unreviewed_pilot',
                'source':'WOD-E2E validation rated anchors','source_sha256':sources,
                'selection_policy':'automatic_stop_go_v1','rejection_counts':rejected,
                'units':len(targets),'requests':len(workload),
                'split_counts':{'dev':sum(r['split']=='dev' for r in targets),
                                'pilot_eval':sum(r['split']=='pilot_eval' for r in targets)},
                'images':len(assets),'nominal_camera_offsets_s':[d/10 for d in OFFSETS],
                'camera_temporal_selection':'nearest_observed_within_0p1s',
                'file_sha256':{str(p.relative_to(staging)):digest(p.read_bytes()) for p in files},
                'limitations':['no human semantic or safety review',
                               'proxy stop/go contracts only',
                               'validation training overlap possible',
                               'P0 English instructions only',
                               'some camera history frames jitter by at most 0.1 s']}
    (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    staging.rename(out)
    print(json.dumps({'out':str(out),'units':len(targets),'requests':len(workload),
                      'images':len(assets),'split_counts':manifest['split_counts'],
                      'rejections':rejected},indent=2,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
