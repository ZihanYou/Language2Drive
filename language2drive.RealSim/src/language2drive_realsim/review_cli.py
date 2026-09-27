"""Portable CPU examples, paper scoring and supplemental diagnostics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .paired_semantic_metrics_all30 import score_pair
from .paper_metrics import score_document as paper_score
from .supplemental_diagnostics import classify, summarize
from .physical_only_metrics import validated_trace_rows


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def portable_document(path):
    """Resolve trace references relative to the supplied experiment JSON."""
    path = Path(path).resolve()
    document = read(path)
    if document.get('seed') != 1701:
        raise ValueError('This release scoring entry point uses seed 1701')
    for row in document.get('results', []):
        if row.get('seed') != document['seed']:
            raise ValueError('Slot and experiment seed disagree')
        trace = row.get('origin', {}).get('trace')
        if trace:
            trace_path = Path(trace)
            if not trace_path.is_absolute():
                row['origin']['trace'] = str(path.parent / trace_path)
    return document


def toy_pairs():
    """Three synthetic geometry examples, not benchmark scenes or model results."""
    def polygon(y0, y1):
        return {'type': 'Polygon', 'coordinates': [
            [[-1, y0], [30, y0], [30, y1], [-1, y1], [-1, y0]]
        ]}

    def trace(speed, y):
        return [dict(time_s=t, position=[speed*t, y*t/4], speed_mps=speed,
                     progress_m=speed*t, collision=False, offroad=False, render_valid=True)
                for t in (0., 1., 2., 3., 4.)]

    base = dict(pcu_id='synthetic-speed', task_family='B1_segment_speed_band',
                decision_window=dict(start_s=1., deadline_s=4.),
                private_annotations=dict(scopes={'scope': polygon(-10, 10)}, lanes={}),
                arms={arm: {'semantic_contract': dict(kind='speed_band', scope='scope',
                      speed_min_mps=lo, speed_max_mps=hi, deadline_s=4.)}
                      for arm, lo, hi in [('R0', 1., 3.), ('R1', 4., 6.)]})
    speed = (base, trace(2., 0.), trace(5., 0.))
    lane = deepcopy(base)
    lane.update(pcu_id='synthetic-lane', task_family='A4_lane_change_left_right')
    lane['private_annotations']['lanes'] = {'lower': polygon(-3, -.01), 'upper': polygon(.01, 3)}
    for arm, target in [('R0', 'lower'), ('R1', 'upper')]:
        lane['arms'][arm]['semantic_contract'] = dict(kind='lane', target_lane=target,
                                                      scope='scope', deadline_s=4.)
    gate = deepcopy(base)
    gate.update(pcu_id='synthetic-route', task_family='A2_straight_turn')
    gate['decision_window']['commitment_gates'] = ['lower', 'upper']
    gate['private_annotations']['gates'] = {
        'lower': [[4., -3.], [4., -.01]], 'upper': [[4., .01], [4., 3.]]}
    for arm, target, other in [('R0', 'lower', 'upper'), ('R1', 'upper', 'lower')]:
        gate['arms'][arm]['semantic_contract'] = dict(kind='gate', target_gate=target,
                                                      opposing_gate=other, deadline_s=4.)
    return [speed, (lane, trace(2., -2.), trace(2., 2.)),
            (gate, trace(2., -2.), trace(2., 2.))]


def demo():
    return {'data_kind': 'synthetic_software_examples_not_experimental_results',
            'pairs': [dict(pcu_id=p['pcu_id'], **score_pair(p, a, b))
                      for p, a, b in toy_pairs()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output file; existing files are never overwritten')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('demo', help='Score synthetic speed, lane and route pairs without GPU or downloads')
    score = commands.add_parser('score', help='Score a complete 30-PCU component-record inventory')
    score.add_argument('--pcus', required=True, type=Path)
    score.add_argument('--experiment', required=True, type=Path)
    diag = commands.add_parser('diagnose', help='Separate technical, semantic and safety failure labels')
    diag.add_argument('--pcus', required=True, type=Path)
    diag.add_argument('--experiment', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'demo':
        result = demo()
    else:
        pcus = read(args.pcus)['pcus']
        by_id = {p['pcu_id']: p for p in pcus}
        if len(by_id) != len(pcus):
            raise ValueError('Duplicate PCU identifiers')
        document = portable_document(args.experiment)
        if args.command == 'score':
            result = paper_score(document, pcus, args.experiment.name)
        else:
            records = []
            for row in document['results']:
                usable = row.get('execution_status') == 'completed' and row.get('evidence_valid') is True
                rows = validated_trace_rows(Path(row['origin']['trace'])) if usable else None
                records.append(classify(row, by_id[row['pcu_id']], rows,
                                        technical_failure=row.get('technical_failure')))
            result = dict(records=records, summary=summarize(records))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
