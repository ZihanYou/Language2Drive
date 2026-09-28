"""Materialize every unreviewed WOD-E2E candidate into an internal prerelease.

The registry includes all candidate pairs, including those without a defensible
automatic task. Model requests are created only for pairs with nonempty
English P0 drafts; unresolved pairs remain present in the private registry.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import build_prerelease as pilot


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'wod-e2e-unreviewed-prerelease-v0.1.0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--expected-candidates', type=int)
    parser.add_argument('--out', type=Path, default=ROOT/'prerelease/v0.1.0')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}; use a new version')
    queue = list(csv.DictReader((ROOT/'review/review_queue.csv').open(newline='',encoding='utf-8')))
    drafts = json.loads((ROOT/'review/instruction_drafts.json').read_text(encoding='utf-8'))
    rated_rows = pilot.read_jsonl(ROOT/'artifacts/rated_frames.jsonl')
    rated = {r['frame_name']:r for r in rated_rows}
    normalized = {r['frame_name']:r for r in pilot.read_jsonl(ROOT/'artifacts/normalized_rated_frames.jsonl')}
    if len(drafts) != len(queue) or len(rated) != len(rated_rows):
        raise ValueError('Draft/queue mismatch or duplicate rated anchor')
    if args.expected_candidates is not None and len(queue) != args.expected_candidates:
        raise ValueError('Candidate count differs from --expected-candidates')
    locations = pilot.index_selected_records([(None, {'sequence_id':r['sequence_id']}) for r in rated_rows], args.source_root)
    staging = out.with_name(out.name+'.building')
    if staging.exists():
        raise FileExistsError(f'Incomplete prior build: {staging}')
    (staging/'public/images').mkdir(parents=True)
    (staging/'private').mkdir()
    inputs, workload, registry, image_manifest = [], [], [], []
    input_ids = {}
    history_complete = {}
    for i, source in enumerate(sorted(rated_rows,key=lambda r:r['frame_name']),1):
        frame = source['frame_name']
        if frame not in locations:
            raise ValueError(f'Missing rated anchor from local shards: {frame}')
        input_id = 'I-' + pilot.digest(frame.encode())[:12]
        input_ids[frame] = input_id
        images, calibration = pilot.read_frame(*locations[frame],set(pilot.CAMERAS.values()))
        if set(images) != set(pilot.CAMERAS.values()):
            raise ValueError(f'{frame}: expected all 8 current cameras')
        if any(len(calibration.get(cam,{}).get('intrinsic',[])) < 4 or
               len(calibration.get(cam,{}).get('extrinsic',[])) != 16
               for cam in pilot.CAMERAS.values()):
            raise ValueError(f'{frame}: calibration incomplete')
        current = {}
        temporal = {cam:[] for cam in pilot.FRONT}
        saved_paths = set()

        def save(camera, delta, nominal, blob):
            rel = f'images/{input_id}/{delta:+04d}_{camera}.jpg'
            if rel not in saved_paths:
                if not blob.startswith(b'\xff\xd8'):
                    raise ValueError(f'{frame}: invalid JPEG for {camera} at {delta}')
                target = staging/'public'/rel
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(blob)
                saved_paths.add(rel)
                image_manifest.append({'input_id':input_id,'camera':camera,
                                       'offset_s':delta/10,'path':rel,
                                       'sha256':pilot.digest(blob)})
            return {'path':rel,'sha256':pilot.digest(blob),
                    'offset_s':delta/10,'nominal_offset_s':nominal/10}

        for camera,blob in sorted(images.items()):
            current[camera] = save(camera,0,0,blob)
        selected = pilot.choose_temporal_names(frame,locations)
        complete = selected is not None
        history_complete[frame] = complete
        if complete:
            for delta,name,nominal in selected:
                picture = images if delta == 0 else pilot.read_frame(*locations[name],set(pilot.FRONT))[0]
                if not set(pilot.FRONT).issubset(picture):
                    raise ValueError(f'{name}: missing front history camera')
                for camera in pilot.FRONT:
                    temporal[camera].append(save(camera,delta,nominal,picture[camera]))
        shared = {'input_id':input_id,'frame_name':frame,
                  'sequence_id':source['sequence_id'],
                  'route_command':{1:'GO_STRAIGHT',2:'GO_LEFT',3:'GO_RIGHT'}.get(source['intent'],'UNKNOWN'),
                  'images':current,'camera_history':temporal,
                  'history_complete':complete,'calibration':calibration,
                  'past_states':source['past'],
                  'time_axis':{'camera_hz':10,'past_states_hz':4,
                               'trajectory_hz':4,'trajectory_seconds':5,
                               'camera_temporal_selection':'nearest_observed_within_0p1s' if complete else 'unavailable',
                               'max_temporal_jitter_s':0.1}}
        shared['shared_input_hash'] = pilot.digest(pilot.canonical(shared))
        inputs.append(shared)
        if i % 25 == 0:
            print(f'Materialized {i}/{len(rated_rows)} anchors',flush=True)
    for row in queue:
        candidate = row['candidate_id']
        draft = drafts[candidate]
        frame = row['frame_name']
        refs = {str(t['reference_index']):t for t in normalized[frame]['preferences']}
        task_ready = (draft['task_family'] != 'UNRESOLVED' and
                      bool(draft['instruction_a']) and bool(draft['instruction_b']) and
                      draft['instruction_a'] != draft['instruction_b'] and
                      all(draft[key].isascii() for key in ('instruction_a','instruction_b')))
        request_ids = {}
        if task_ready:
            for arm in ('a','b'):
                request_id = 'R-' + pilot.digest(f'{candidate}:{arm}'.encode())[:12]
                request_ids[arm] = request_id
                workload.append({'request_id':request_id,'input_id':input_ids[frame],
                                 'instruction':draft[f'instruction_{arm}'],
                                 'instruction_language':'en'})
        registry.append({'candidate_id':candidate,'frame_name':frame,
                         'sequence_id':row['sequence_id'],'scenario_cluster':row['scenario_cluster'],
                         'input_id':input_ids[frame],
                         'status':'DRAFT_P0_UNREVIEWED' if task_ready else 'UNRESOLVED_UNREVIEWED',
                         'task_family_draft':draft['task_family'],
                         'task_summary_draft':draft['task_summary'],
                         'review_checks':draft['review_checks'],
                         'instruction_language':'en',
                         'instruction_a_draft':draft['instruction_a'],
                         'instruction_b_draft':draft['instruction_b'],
                         'request_ids':request_ids,
                         'history_complete':history_complete[frame],
                         'references':[{'reference_index':int(row[key]),
                                        'score':refs[row[key]]['score'],
                                        'xy_m':refs[row[key]]['xy_m']}
                                       for key in ('reference_a','reference_b')],
                         'semantic_validity':'unknown','safety_validity':'unknown',
                         'contract_status':'not_annotated'})
    pilot.write_jsonl(staging/'public/inputs.jsonl',inputs)
    pilot.write_jsonl(staging/'public/workload.jsonl',workload)
    pilot.write_jsonl(staging/'public/image_manifest.jsonl',image_manifest)
    pilot.write_jsonl(staging/'private/candidates.jsonl',registry)
    files = [staging/p for p in ('public/inputs.jsonl','public/workload.jsonl',
                                 'public/image_manifest.jsonl','private/candidates.jsonl')]
    sources = {name:pilot.digest((ROOT/name).read_bytes()) for name in
               ('review/review_queue.csv','review/instruction_drafts.json',
                'artifacts/rated_frames.jsonl','artifacts/normalized_rated_frames.jsonl')}
    counts = Counter(row['status'] for row in registry)
    manifest = {'version':VERSION,'release_status':'all_pending_unreviewed',
                'source':'WOD-E2E validation rated anchors',
                'candidate_pairs':len(registry),'anchor_scenes':len(inputs),
                'english_p0_requests':len(workload),
                'status_counts':dict(counts),
                'history_complete_anchors':sum(history_complete.values()),
                'history_complete_draft_pairs':sum(r['history_complete'] and r['status']=='DRAFT_P0_UNREVIEWED' for r in registry),
                'image_files':len(image_manifest),
                'source_sha256':sources,
                'file_sha256':{str(p.relative_to(staging)):pilot.digest(p.read_bytes()) for p in files},
                'image_hashes_in':'public/image_manifest.jsonl',
                'limitations':['all candidate semantics and safety remain unreviewed',
                               f"{counts.get('UNRESOLVED_UNREVIEWED', 0)} candidates have no English P0 request",
                               'camera history incomplete for some anchors',
                               'draft instructions are inferred from private rated motion',
                               'model training overlap must be checked']}
    (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    staging.rename(out)
    print(json.dumps({'out':str(out),'anchors':len(inputs),'pairs':len(registry),
                      'requests':len(workload),'status_counts':dict(counts),
                      'history_complete_anchors':manifest['history_complete_anchors'],
                      'images':len(image_manifest)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
