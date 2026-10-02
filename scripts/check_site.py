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
assert len(cases)==len({c['id'] for c in cases})==73
assert {c['track'] for c in cases}=={'carla','realsim','nuscenes','navsim','wod'}
for c in cases:
    for key in ['video','poster']:
        assert (DOCS/c[key]).is_file(), f'Missing media: {c[key]}'
        assert (DOCS/c[key]).resolve().is_relative_to(DOCS.resolve())
    assert (c['width'],c['height'])==(1920,1080) and c['presentation_version']==2
    assert c['evidence_role']==('benchmark-target' if c['kind'] in ('reference','oracle') else 'model-result')
    assert c['trajectory_source'] and c['trajectory_panel']
    assert len(c['instructions'])==2 and all(c['instructions'])
    assert re.fullmatch('[a-f0-9]{64}',c['source_sha256'])
    assert re.fullmatch('[a-f0-9]{64}',c['video_sha256'])
    assert (DOCS/c['video']).stat().st_size==c['bytes']
    if c['track']=='carla':assert c['visualization']=='Trajectory re-render' and 'NOT the original model inputs' in c['note']
    if c['track']=='wod':assert c['kind']=='candidate' and 'not_annotated' in c['note']
    if c['track']=='navsim':
        assert c['kind']=='reference' and len(c['comparison']['models'])==5
        assert c['comparison']['timestamps_s']==[.5,1,1.5,2,2.5,3,3.5,4]
        assert all(len(v[arm])==8 for v in c['comparison']['models'].values() for arm in ['0','1'])
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
print('PASS: all HTML/CSS references; 73 evidence clips; film provenance; 27 manuscript baseline rows')
