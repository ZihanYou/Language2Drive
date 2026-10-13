#!/usr/bin/env python3
"""Validate local page references, anchors and manuscript-result inventory."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.refs = []; self.ids = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        for name in ('src', 'href'):
            if name in attrs: self.refs.append(attrs[name])

for html in DOCS.glob('*.html'):
    assert not re.search(r'</[a-zA-Z]+\s+[a-zA-Z]', html.read_text()), f'Malformed closing tag: {html.name}'
    p = Page(); p.feed(html.read_text())
    assert len(p.ids) == len(set(p.ids)), f'Duplicate HTML id in {html.name}'
    for ref in p.refs:
        if ref in ('', '#') or re.match(r'^(https?:|data:|mailto:)', ref): continue
        parsed=urlsplit(ref);path,anchor=parsed.path,parsed.fragment
        if path: assert (DOCS / path).is_file(), f'Missing page asset: {ref}'
        elif anchor: assert anchor in p.ids, f'Missing page anchor: {anchor}'
for css in DOCS.glob('*.css'):
    for ref in re.findall(r'url\([\'\"]?([^\)\'\"]+)', css.read_text()):
        assert (DOCS / ref).is_file(), f'Missing CSS asset: {ref}'
cases=json.loads((DOCS/'data/cases.json').read_text())['cases']
assert len(cases)==len({c['id'] for c in cases})==129
assert {c['id'] for c in cases if c['track']=='wod'}=={'wod-cand-0040','wod-cand-0038'}
assert not any(c['track']=='navsim' and c['family']=='branch_choice' for c in cases)
assert {c['track'] for c in cases}=={'carla','realsim','nuscenes','navsim','wod'}
for c in cases:
    for key in ['video','poster']:
        assert (DOCS/c[key]).is_file(), f'Missing media: {c[key]}'
        assert (DOCS/c[key]).resolve().is_relative_to(DOCS.resolve())
    assert (c['width'],c['height'])==(1920,1080) and c['presentation_version']==2
    assert c['evidence_role']==('benchmark-target' if c['kind'] in ('reference','oracle','target') else 'model-result')
    assert c['trajectory_source'] and c['trajectory_panel']
    assert len(c['instructions'])==2 and all(c['instructions'])
    assert re.fullmatch('[a-f0-9]{64}',c['source_sha256'])
    assert re.fullmatch('[a-f0-9]{64}',c['video_sha256'])
    assert (DOCS/c['video']).stat().st_size==c['bytes']
    if c['model']=='Alpamayo 1.5' and c['track'] in ('carla','realsim'):
        assert (DOCS/c['reasoning']).is_file()
        cot=json.loads((DOCS/c['reasoning']).read_text())
        assert cot['schema']=='language2drive.reasoning.v1' and cot['case_id']==c['id']
        assert cot['model']==c['model'] and cot['track']==c['track']
        assert [a['label'] for a in cot['arms']]==['A','B'] and cot['playback_scale']>=1
        for arm in cot['arms']:
            assert re.fullmatch('[a-f0-9]{64}',arm['source_sha256'])
            assert arm['source_member'].endswith('.csv') and arm['entries']
            assert 0<arm['video_end_s']<=c['duration_s']+.1
            for entry in arm['entries']:
                assert entry['text'] and entry['instruction']
                expected=max(0,(entry['source_time_s']-arm['clock_origin_s'])/cot['playback_scale'])
                assert abs(entry['video_time_s']-expected)<1e-8
            assert all(a['video_time_s']<=b['video_time_s'] for a,b in zip(arm['entries'],arm['entries'][1:]))
    if c['track']=='carla' and c['evidence_role']=='model-result':assert c['visualization']=='Trajectory re-render' and 'NOT the original model inputs' in c['note']
    if c['track']=='wod':
        assert c['kind']=='target' and c['evidence_role']=='benchmark-target'
        assert c['archive']=='WOD-E2E_Benchmark_Targets_2026-10-12.zip'
        assert not c.get('models') and not c.get('comparison')
        assert c['source_sha256']==c['source_video_sha256']==c['video_sha256']
    if c['track']=='navsim':
        assert c['kind']=='reference' and len(c['comparison']['models'])==5
        assert c['comparison']['timestamps_s']==[.5,1,1.5,2,2.5,3,3.5,4]
        assert all(len(v[arm])==8 for v in c['comparison']['models'].values() for arm in ['0','1'])
targets=[c for c in cases if c['track']=='carla' and c['kind']=='target']
assert len(targets)==66 and len({c['pcu'] for c in targets})==33
assert {c['scope'] for c in targets}==set('ABCD') and len({c['family'] for c in targets})==16
for c in targets:
    assert c['camera_view'] in ('front','exterior') and c['playback_speed']>=1
    assert len(c['sources'])==2 and len(c['trajectory_source'])==2
    assert c['sources'][0]['member']==c['member'] and c['sources'][1]['member']==c['partner_member']
    assert c['sources'][0]['sha256']==c['source_sha256']
    assert all(re.fullmatch('[a-f0-9]{64}',s['sha256']) for s in c['sources'])
    assert not c.get('reasoning') and not c.get('visualization')
atlas=json.loads((DOCS/'data/maps.json').read_text())
assert atlas['schema']=='language2drive.simulator.v1'
assert {m['id'] for m in atlas['maps']}=={'American-city0428','yangluogang1','parkinglotfish-new3','hanghailou3','hubeiwenli4','road1'}
assert sum(len(m['locations']) for m in atlas['maps'])==54
for m in atlas['maps']:
    assert (DOCS/m['thumbnail']).is_file() and m['lanes'] and len(m['tours'])==4
    assert re.fullmatch('[a-f0-9]{64}',m['geometry_sha256'])
    for l in m['lanes']:
        assert len(l['points'])>=2 and l['width_m']>0
    media=[]
    for p in m['locations']:
        assert set(p['views'])=={'overview','forward','reverse'}
        media+=list(p['views'].values())
    for t in m['tours']:
        assert t['duration_s']==45 and len(t['route'])>=200
        assert all(a[0]<b[0] for a,b in zip(t['route'],t['route'][1:]))
        assert (DOCS/t['media']['poster']).is_file()
        assert all(0<cut<t['duration_s'] for cut in t['cuts_s'])
        media.append(t['media'])
    for a in media:
        assert (DOCS/a['path']).is_file() and (DOCS/a['path']).stat().st_size==a['bytes']
        assert (DOCS/a['path']).resolve().is_relative_to(DOCS.resolve())
        assert re.fullmatch('[a-f0-9]{64}',a['sha256']) and re.fullmatch('[a-f0-9]{64}',a['source_sha256'])
scopes=json.loads((DOCS/'data/scope-cases.json').read_text())['cases']
assert [c['scope'] for c in scopes]==list('ABCD')
for c in scopes:
    assert c['evidence_role']=='benchmark-target' and c['case_id'] in {x['id'] for x in cases}
    assert c['paths'][0][0]==c['paths'][1][0]==[0,0]
    assert c['lanes'] and len(c['goals'])==2
provenance=json.loads((DOCS/'assets/framework/provenance.json').read_text())
assert 'front_wide_original_jpeg' in provenance['assets']['realsim.webp']['member']
story=json.loads((DOCS/'assets/demo/storyboard.json').read_text())
assert story['duration_s']==120 and story['resolution']==[1920,1080]
assert sum(s['duration_s'] for s in story['segments'])==120
by_id={c['id']:c for c in cases}
for segment in story['segments']:
    if segment['kind']=='case':
        assert by_id[segment['case_id']]['video_sha256']==segment['source_sha256']
        assert by_id[segment['case_id']]['evidence_role']==segment['evidence_role']
results = json.loads((DOCS / 'data/results.json').read_text())
for track, count in {'carla': 6, 'realsim': 8, 'nuscenes': 4, 'navsim': 5, 'wod': 4}.items():
    table = results['tracks'][track]
    assert len(table['rows']) == count, f'Wrong {track} result count'
    assert len({r['model'] for r in table['rows']}) == count
    for row in table['rows']:
        assert all(type(row[m]) in (int, float) and 0 <= row[m] <= 100 for m in table['metrics'])
        assert not row['model'].startswith('%')
assert results['tracks']['carla']['rows'][0]['L2D'] == 12.20
assert results['tracks']['realsim']['rows'][1]['DrivingScore'] == 97.17
models = json.loads((DOCS / 'data/models.json').read_text())['models']
references = {name: model for name, model in models.items()}
for model in models.values():
    assert re.fullmatch(r'https://arxiv.org/abs/\d{4}\.\d{4,5}', model['paper'])
    assert model['title']
    for alias in model.get('aliases', []):
        assert alias not in references, f'Duplicate model alias: {alias}'
        references[alias] = model
for table in results['tracks'].values():
    for row in table['rows']:
        assert row['model'] in references, f'Missing paper for {row["model"]}'
        assert references[row['model']]['paper'] in (ROOT / 'README.md').read_text()
for case in cases:
    names = [*case.get('models', []), *case.get('comparison', {}).get('models', {})]
    if case['evidence_role'] == 'model-result' and case['model'] != 'Four-model comparison':
        names.append(case['model'])
    assert all(name in references for name in names), f'Missing case paper: {case["id"]}'
assert sum(bool(c.get('reasoning')) for c in cases)==28
paper=json.loads((DOCS/'data/paper.json').read_text())
assert [a['id'] for a in paper['affiliations']]==list(range(1,20))
assert all(i in range(1,20) for a in paper['authors'] for i in a['affiliations'])
for path in [DOCS/'index.html', ROOT/'README.md', ROOT/'huggingface/README.md']:
    assert 'Project Lead' in path.read_text() and 'ByteDance' not in path.read_text()
print('PASS: all HTML/CSS references; 129 case clips; 66 CARLA target views; six maps, 54 locations and 24 tours; 28 paired CoT transcripts; film provenance; 27 manuscript rows; all model papers')
