#!/usr/bin/env python3
"""Resumeable, append-only model inference on a frozen PCU release."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_io import load_release  # noqa: E402
from run_model import instruction_for, load_adapter  # noqa: E402
from build_catalog import canonical_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--adapter-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('all', 'dev', 'heldout'), default='all')
    parser.add_argument('--conditions', default='canonical,masked,shuffled,paraphrase_1,paraphrase_2,paraphrase_3')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--max-new', type=int, default=0, help='Stop after this many newly written outputs')
    args = parser.parse_args()
    _, records = load_release(args.release_dir, split=args.split)
    records.sort(key=lambda row: row['pcu_id'])
    for row in records:
        shared = dict(row['public_input'])
        expected = shared.pop('public_input_hash')
        if canonical_hash(shared) != expected:
            raise ValueError('Public input hash mismatch: ' + row['pcu_id'])
    groups = defaultdict(list)
    for row in records:
        groups[row['family']].append(row)
    donors = {row['pcu_id']: group[(i + 1) % len(group)]
              for group in groups.values() for i, row in enumerate(group)}
    conditions = [item.strip() for item in args.conditions.split(',')]
    seeds = [int(item) for item in args.seeds.split(',')]
    existing = set()
    if args.output.exists():
        with args.output.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row['pcu_id'], row['arm'], row['condition'], row['seed'])
                if key in existing:
                    raise ValueError('Duplicate existing output: ' + str(key))
                existing.add(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predict = load_adapter(args.adapter_file)
    written = 0
    with args.output.open('a') as output:
        for row in records:
            for condition in conditions:
                for seed in seeds:
                    for arm in ('arm_0', 'arm_1'):
                        key = (row['pcu_id'], arm, condition, seed)
                        if key in existing:
                            continue
                        instruction = instruction_for(row, arm, condition, donors)
                        trajectory = predict(row['public_input'], instruction, seed)
                        payload = {'pcu_id': row['pcu_id'], 'arm': arm,
                                   'condition': condition, 'seed': seed,
                                   'public_input_hash': row['public_input']['public_input_hash'],
                                   'instruction': instruction, 'trajectory': trajectory}
                        output.write(json.dumps(payload, separators=(',', ':')) + '\n')
                        output.flush()
                        written += 1
                        if written % 10 == 0:
                            print('Saved {} new predictions'.format(written), flush=True)
                        if args.max_new and written >= args.max_new:
                            print('Stopped after {} new predictions'.format(written), flush=True)
                            return
    print('Complete: {} new, {} reused -> {}'.format(written, len(existing), args.output), flush=True)


if __name__ == '__main__':
    main()
