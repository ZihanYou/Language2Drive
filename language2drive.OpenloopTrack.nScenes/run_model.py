#!/usr/bin/env python3
"""Run a trajectory adapter using public inputs only.

Adapter file must expose predict(public_input, instruction, seed) -> trajectory.
The adapter receives no arm label, target lane, contract, or private sidecar.
"""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

from build_catalog import canonical_hash
from release_io import load_release


ROOT = ARTIFACT_ROOT
DEFAULT_PUBLIC = ROOT/'public_inputs/review_queue.jsonl'


def load_adapter(path):
    spec = importlib.util.spec_from_file_location('l2d_user_adapter',path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load adapter file: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module,'predict',None)):
        raise ValueError('Adapter must define predict(public_input, instruction, seed)')
    return module.predict


def instruction_for(record, arm, condition, donors):
    instructions = record['arms'][arm]
    if condition == 'canonical':
        return instructions['instruction']
    if condition == 'masked':
        return ''
    if condition.startswith('paraphrase_'):
        index = int(condition.split('_')[1])-1
        if index not in range(3):
            raise ValueError(condition)
        return instructions['paraphrases'][index]
    if condition == 'shuffled':
        # A different PCU supplies the opposite-arm instruction. If its wording
        # matches, fall back to this PCU's opposite arm to ensure a changed input.
        donor = donors[record['pcu_id']]
        opposite = 'arm_1' if arm=='arm_0' else 'arm_0'
        text = donor['arms'][opposite]['instruction']
        return text if text != instructions['instruction'] else record['arms'][opposite]['instruction']
    raise ValueError(f'Unknown condition: {condition}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',type=Path)
    parser.add_argument('--release-dir',type=Path,help='frozen benchmark release directory')
    parser.add_argument('--split',choices=('all','dev','heldout'),default='all')
    parser.add_argument('--adapter-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conditions',default='canonical',
                        help='Comma-separated: canonical, masked, shuffled, paraphrase_1..3')
    parser.add_argument('--seeds',default='0',help='Comma-separated integer seeds')
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    if args.release_dir and args.public:
        parser.error('Use either --release-dir or --public')
    if args.release_dir:
        _, records = load_release(args.release_dir, split=args.split)
    else:
        if args.split != 'all':
            parser.error('--split requires --release-dir')
        with (args.public or DEFAULT_PUBLIC).open() as source:
            records = [json.loads(line) for line in source]
    records.sort(key=lambda x:x['pcu_id'])
    if args.limit:
        records = records[:args.limit]
    groups = defaultdict(list)
    for record in records:
        shared = dict(record['public_input'])
        expected = shared.pop('public_input_hash')
        if canonical_hash(shared)!=expected:
            raise ValueError(f'Public input hash mismatch: {record["pcu_id"]}')
        groups[record['family']].append(record)
    donors = {}
    for group in groups.values():
        for i,record in enumerate(group):
            donors[record['pcu_id']] = group[(i+1)%len(group)]
    predict = load_adapter(args.adapter_file)
    conditions = [x.strip() for x in args.conditions.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count = 0
    with args.output.open('w') as output:
        for record in records:
            for condition in conditions:
                for seed in seeds:
                    for arm in ('arm_0','arm_1'):
                        text = instruction_for(record,arm,condition,donors)
                        trajectory = predict(record['public_input'],text,seed)
                        row = {'pcu_id':record['pcu_id'],'arm':arm,
                               'condition':condition,'seed':seed,
                               'public_input_hash':record['public_input']['public_input_hash'],
                               'instruction':text,'trajectory':trajectory}
                        output.write(json.dumps(row,separators=(',',':'))+'\n')
                        count += 1
    print(f'Wrote {count} predictions -> {args.output}')


if __name__ == '__main__':
    main()
