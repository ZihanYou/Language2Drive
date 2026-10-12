#!/usr/bin/env python3
"""Export recorded Alpamayo CoT and its native video clock from evidence ZIPs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def export_reasoning(case, archive, docs):
    if case['model'] != 'Alpamayo 1.5' or case['track'] not in ('carla', 'realsim'):
        return
    with zipfile.ZipFile(archive) as z:
        if case['track'] == 'carla':
            base = 'pcu_video_evidence_20261002/cases/' + case['member'].split('/')[-1].removesuffix('.mp4')
            data = json.loads(z.read(base + '/data.json'))
            scale = max(1, max(a['duration_s'] for a in data['arms']) / data['max_video_duration_s'])
            clocks = [(a['arm'], a['agent_anchor_s'], a['duration_s']) for a in data['arms']]
            field, clock_field, filename = 'reasoning', 'agent_time_s', 'reasoning.csv'
            alignment = data['alignment']
        else:
            base = case['member'].rsplit('/', 1)[0]
            data = json.loads(z.read(base + '/case.json'))
            start = max(a['first_frame_s'] for a in data['arms'])
            end = min(a['last_frame_s'] for a in data['arms'])
            scale = 1
            clocks = [(a['arm'], start, end - start) for a in data['arms']]
            field, clock_field, filename = 'reasoning_text', 'time_s', 'reasoning_transcript.csv'
            alignment = 'Video zero is the first shared camera timestamp; the final reading hold retains the last recorded output.'
        arms = []
        for index, (arm, anchor, duration) in enumerate(clocks):
            member = f'{base}/{arm}/{filename}'
            raw = z.read(member)
            rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            entries = []
            for record_index, row in enumerate(rows):
                if not row[field] or not row[field].strip():
                    continue
                source_time = float(row[clock_field])
                entries.append(dict(record_index=record_index, source_time_s=source_time,
                    video_time_s=max(0, (source_time - anchor) / scale), text=row[field],
                    instruction=row.get('instruction') or case['instructions'][index]))
            assert entries, f'Missing recorded CoT: {case["id"]}/{arm}'
            assert all(a['video_time_s'] <= b['video_time_s'] for a, b in zip(entries, entries[1:]))
            arms.append(dict(label='AB'[index], source_arm=arm, source_member=member,
                source_sha256=hashlib.sha256(raw).hexdigest(), clock_origin_s=anchor,
                video_end_s=duration / scale, entries=entries))
    payload = dict(schema='language2drive.reasoning.v1', case_id=case['id'], model=case['model'],
        track=case['track'], playback_scale=scale, alignment=alignment, arms=arms)
    path = docs / 'data/reasoning' / (case['id'] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    case['reasoning'] = path.relative_to(docs).as_posix()
    return sum(len(a['entries']) for a in arms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--carla-archive', type=Path, required=True)
    args = p.parse_args()
    docs = ROOT / 'docs'
    manifest = docs / 'data/cases.json'
    data = json.loads(manifest.read_text())
    count = records = 0
    for case in data['cases']:
        archive = args.carla_archive if case['track'] == 'carla' else args.source_root / case['archive']
        n = export_reasoning(case, archive, docs)
        if n:
            count += 1
            records += n
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f'EXPORTED: {count} paired cases; {records} original CoT records')


if __name__ == '__main__':
    main()
