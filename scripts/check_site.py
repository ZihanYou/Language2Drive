#!/usr/bin/env python3
"""Validate local page references, anchors and manuscript-result inventory."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re

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

p = Page(); p.feed((DOCS / 'index.html').read_text())
assert len(p.ids) == len(set(p.ids)), 'Duplicate HTML id'
for ref in p.refs:
    if ref in ('', '#') or re.match(r'^(https?:|data:|mailto:)', ref): continue
    path, _, anchor = ref.partition('#')
    if path: assert (DOCS / path).is_file(), f'Missing page asset: {ref}'
    elif anchor: assert anchor in p.ids, f'Missing page anchor: {anchor}'
for ref in re.findall(r'url\([\'\"]?([^\)\'\"]+)', (DOCS / 'style.css').read_text()):
    assert (DOCS / ref).is_file(), f'Missing CSS asset: {ref}'
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
print(f'PASS: {len(p.ids)} page anchors; all local assets; 27 manuscript baseline rows')
