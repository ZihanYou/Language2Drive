#!/usr/bin/env python3
"""Build the local case gallery from author-provided evidence archives.

No inference or rescoring is performed. Only selected viewing copies and an
allowlisted display manifest are exported; raw private contracts stay in input
archives. Requires ffmpeg/ffprobe. Re-run after media edits to refresh hashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def read_json(z, path):
    return json.loads(z.read(path))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def collect(root):
    cases = []
    def add(archive, member, **data):
        data.update(archive=archive.name, member=member)
        cases.append(data)

    archive = root / 'Language2Drive_PCU_Paired_Videos.zip'
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0].split('/')[0]
        manifest = read_json(z, prefix + '/manifest.json')
        # Fixed first representative of each task family, plus two contrasting
        # speed cases from different scenes. Selection does not use scores.
        selected = {'P01', 'P02', 'P09', 'P11', 'P12', 'P16', 'P04', 'P08'}
        for case in manifest['cases']:
            if case['pcu'] not in selected:
                continue
            if case['mode'] == 'oracle' and case['pcu'] in {'P04', 'P08'}:
                continue
            pcu = read_json(z, prefix + '/pcus/' + case['pcu'] + '.json')
            instructions = [pcu['arms'][arm]['instruction_schedule'][0]['utterances'][0] for arm in ['R0', 'R1']]
            family = {'A2': 'Entrance choice', 'A4': 'Left or right lane', 'A5': 'Keep or change lane', 'B1': 'Speed band', 'B2': 'Persistent lane choice', 'B3': 'Speed at a landmark'}[case['family']]
            oracle = case['mode'] == 'oracle'
            add(archive, f"{prefix}/cases/{case['case_id']}/paired.mp4",
                id='realsim-' + slug(case['case_id']), pcu=case['pcu'], track='realsim',
                title=family, family=case['family'], scope=case['family'][0], scene=case['scene'],
                model='Privileged oracle' if oracle else 'Alpamayo 1.5',
                kind='oracle' if oracle else 'closed-loop', instructions=instructions,
                note=('Privileged oracle execution illustrates task feasibility; it is not a VLA baseline. ' if oracle else 'Actual Alpamayo 1.5 closed-loop execution; recorded model text is shown as logged. ') + 'NuRec-rendered cameras, matched seed 1701, canonical wording. The final 3 seconds hold the last frame. Trajectory differences alone do not establish semantic success.')

    archive = root / 'nuscenes_pcu_video_bundle_20261002_videos.zip'
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0].split('/')[0]
        for key in ['P01_scene-0077_speed', 'P06_scene-0112_lane', 'P09_scene-0142_lane', 'P55_scene-0836_speed']:
            data = read_json(z, f'{prefix}/cases/{key}/data.json')
            for model in [*data['models'], 'reference']:
                add(archive, f'{prefix}/cases/{key}/{model}/paired.mp4',
                    id='nuscenes-' + slug(key + '-' + model), pcu=key, track='nuscenes',
                    title='15 or 30 km/h' if data['family'] == 'scoped_speed' else 'Change lane or stay',
                    family=data['family'], scope=data['task_layer'], scene=data['location'], split=data['split'],
                    model='Reference witness' if model == 'reference' else model,
                    kind='reference' if model == 'reference' else 'open-loop',
                    instructions=[data['public']['arms'][arm]['instruction'] for arm in ['arm_0','arm_1']],
                    note='Recorded cameras are shared by both arms. Colored paths are ' + ('constructed reference witnesses' if model == 'reference' else 'saved, single-anchor model forecasts') + ', not executed rollouts. Future log frames are visualization context, never additional model input. Geometric separation is descriptive, not a success score.')

    archive = root / 'l2d_pcu_videos_20261002.zip'
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0].split('/')[0]
        manifest = read_json(z, prefix + '/manifest.json')
        families = sorted({c['family'] for c in manifest['cases']})
        for family in families:
            options = [c for c in manifest['cases'] if c['family'] == family]
            case = max(options, key=lambda c: c['pair_mean_separation_m'])
            key = case['pcu_id']
            data = read_json(z, f'{prefix}/cases/{key}.json')
            comparison = {
                'frame': 'rear axle; +X forward, +Y left; meters',
                'timestamps_s': [0.5 * i for i in range(1, 9)],
                'models': {manifest['models'][k]: v for k,v in data['models'].items()},
                'reference': data['reference_witnesses'],
            }
            add(archive, f'{prefix}/videos/paired/{key}.mp4',
                id='navsim-' + slug(key), pcu=key, track='navsim',
                title={'branch_choice':'Left turn or straight', 'landmark_stop':'Stop at a named landmark', 'lane_choice':'Lane choice', 'local_goal':'Scene-grounded destination', 'object_speed':'Speed relative to a vehicle', 'order':'Same actions, different order', 'pass_follow':'Pass or follow', 'speed_band':'Maintain the requested speed'}[family],
                family=family, scope=data['tier'], scene=data['map_location'],
                model='Reference witness', kind='reference', instructions=data['instructions'], comparison=comparison,
                note='The video displays approved reference witnesses over shared recorded footage. Use the separate trajectory inspector to compare five models’ saved predictions. The 12-second presentation includes reading holds; source cameras are sampled at 2 Hz. No closed-loop execution or new semantic scoring is shown.')

    archive = root / 'WOD-E2E_PCU_Video_Cases_2026-10-02.zip'
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0].split('/')[0]
        for case in read_json(z, prefix + '/cases/manifest.json'):
            key = case['candidate_id']
            data = read_json(z, f'{prefix}/cases/{key}/candidate_record.json')
            add(archive, f"{prefix}/cases/{case['video']}",
                id='wod-' + slug(key), pcu=key, track='wod',
                title=case['task_family'].replace('_', ' ').title(), family=case['task_family'],
                scope=None, scene=case['scene_cluster'], model='Four-model comparison',
                models=case['available_models_by_arm']['a'], kind='candidate',
                instructions=[data['instruction_a_draft'], data['instruction_b_draft']],
                note='Unreviewed prerelease candidate (contract_status=not_annotated), outside the scored P0 leaderboard cohort. Four recorded input-history frames are held for 1 second each. Colored paths are model predictions; gray candidates are not approved instruction targets. No semantic pass/fail claim is made.')
    return cases


def collect_carla(archive):
    """Cover every formal category and map; keep debug cohorts separate."""
    with zipfile.ZipFile(archive) as z:
        prefix = 'pcu_video_evidence_20261002/'
        manifest = read_json(z, prefix + 'manifest.json')
        formal = [c for c in manifest if c['case_id'].startswith('formal_')]
        selected = {next(c['case_id'] for c in formal if c['category'] == family)
                    for family in sorted({c['category'] for c in formal})}
        selected.update('formal_' + key for key in ['A-014','B-025','C-001','D-001','D-009','D-026'])
        for town in sorted({c['town'] for c in formal}):
            if not any(c['case_id'] in selected and c['town'] == town for c in formal):
                selected.add(next(c['case_id'] for c in formal if c['town'] == town))
        result = []
        for c in formal:
            if c['case_id'] not in selected:
                continue
            data = read_json(z, prefix + c['data'])
            result.append(dict(
                id='carla-' + slug(c['case_id']), pcu=c['pcu_id'],track='carla',
                title=c['category'].replace('_',' ').title(), family=c['category'],
                scope=c['layer'],scene=c['town'],model='Alpamayo 1.5',kind='closed-loop',
                visualization='Trajectory re-render',cohort=c['cohort'],
                instructions=[arm['instruction'] for arm in data['arms']],
                archive='pcu_video_evidence_20261002.zip',member=prefix+c['video'],
                note='Saved closed-loop execution, seed 1 / wording 0. Camera views are newly rendered pose replays, NOT the original model inputs. No model inference was rerun. Plans use ego coordinates; executed paths use world coordinates; gray paths are references. Long episodes are time-compressed as labeled, with ended arms held. Text/camera alignment is approximate. This evidence archive spans 10 maps and must not be substituted for a manuscript aggregate cohort.'
            ))
        return result


PRESENTATION_VERSION = 2
from render_case import render as render_case, layout, role, role_name, mode


def build_media(case, root, work, dest, refresh_posters=False, refresh_video=False):
    archive = root / case['archive']
    if case.get('archive') == 'pcu_video_evidence_20261002.zip':
        archive = work / 'carla.zip'
    src = work / 'source' / (case['id'] + '.mp4')
    src.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        with zipfile.ZipFile(archive) as z, z.open(case['member']) as stream, src.open('wb') as target:
            shutil.copyfileobj(stream, target)
    video = dest / (case['id'] + '.mp4')
    poster = dest / (case['id'] + '.webp')
    if refresh_video or not video.exists():
        tmp = work / (case['id'] + '-encode.mp4')
        render_case(case, archive, src, tmp, work)
        shutil.copyfile(tmp, video)
        tmp.unlink()
    else:
        layout(case, archive, work / (case['id'] + '-layout.png'))
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(video)]))
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    duration = float(probe['format']['duration'])
    if refresh_video or refresh_posters or not poster.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(min(3, duration/2)),
            '-i',str(video),'-frames:v','1','-vf',"scale='min(1280,iw)':-2",'-threads','1','-c:v','libwebp','-quality','85',str(poster)],check=True)
    case.update(video='assets/cases/' + video.name, poster='assets/cases/' + poster.name,
        duration_s=round(duration,3), width=stream['width'], height=stream['height'],
        source_sha256=sha(src), video_sha256=sha(video), bytes=video.stat().st_size,
        presentation_version=PRESENTATION_VERSION,
        evidence_role=role(case), evidence_label=role_name(case).title(), display_mode=mode(case),
        presentation='Unified 1080p layout; camera-only crops retain source timing; full trajectories redrawn from native numeric evidence at equal XY scale. Blue/orange identify instruction arms. Dashed gray denotes benchmark targets.')
    print('READY',case['id'],f'{duration:.1f}s',flush=True)
    return case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--carla-catalog',type=Path)
    parser.add_argument('--include-carla',action='store_true',help='Read work-dir/carla.zip')
    parser.add_argument('--refresh-posters',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    docs=Path(__file__).resolve().parents[1]/'docs'
    dest=docs/'assets/cases';dest.mkdir(parents=True,exist_ok=True)
    cases=collect(args.source_root)
    if args.include_carla:
        cases=collect_carla(args.work_dir/'carla.zip')+cases
    if args.carla_catalog:
        cases=json.loads(args.carla_catalog.read_text())+cases
    prior_path=docs/'data/cases.json'
    prior={c['id']:c for c in json.loads(prior_path.read_text())['cases']} if prior_path.exists() else {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cases=list(pool.map(lambda c:build_media(c,args.source_root,args.work_dir,dest,args.refresh_posters,
            prior.get(c['id'],{}).get('presentation_version')!=PRESENTATION_VERSION),cases))
    data={'schema':'language2drive.showcase.v1','selection':'Task-family, scene and model coverage; descriptive examples, not an estimate of aggregate success. Oracle and reference examples are labeled separately. WOD candidates are unreviewed.',
        'media_policy':'Author-provided viewing copies for the local project preview. Source datasets and model assets retain their own terms. No hidden raw contracts or sensor bundles are exported.',
        'cases':cases}
    (docs/'data/cases.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('EXPORTED',len(cases),'clips',sum(c['bytes'] for c in cases)//(1024*1024),'MiB')


if __name__ == '__main__':
    main()
