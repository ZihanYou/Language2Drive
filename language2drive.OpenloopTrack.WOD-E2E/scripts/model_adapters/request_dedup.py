"""Prepare exact public request deduplication and fan out deterministic predictions.

Use only for a fixed deterministic decoding protocol; sampling runs should keep
all requests independent. Inputs and model settings must be fixed across calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--workload', type=Path, required=True)
    prepare.add_argument('--unique-workload', type=Path, required=True)
    prepare.add_argument('--mapping', type=Path, required=True)
    expand = sub.add_parser('expand')
    expand.add_argument('--workload', type=Path, required=True)
    expand.add_argument('--mapping', type=Path, required=True)
    expand.add_argument('--predictions', type=Path, action='append', required=True)
    expand.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    workload = read(args.workload)
    ids = [r['request_id'] for r in workload]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate request ID')
    digest = hashlib.sha256(args.workload.read_bytes()).hexdigest()
    canonical, unique, mapping = {}, [], []
    for row in workload:
        key = (row['input_id'], row['instruction'])
        if key not in canonical:
            canonical[key] = row['request_id']
            unique.append(row)
        mapping.append({'request_id': row['request_id'], 'canonical_request_id': canonical[key]})
    if args.command == 'prepare':
        write(args.unique_workload, unique)
        args.mapping.parent.mkdir(parents=True, exist_ok=True)
        args.mapping.write_text(json.dumps({'source_sha256': digest, 'mapping': mapping}, indent=2) + '\n')
        print(f'{len(workload)} requests -> {len(unique)} deterministic calls')
        return
    saved = json.loads(args.mapping.read_text())
    if saved['source_sha256'] != digest or saved['mapping'] != mapping:
        raise ValueError('Mapping does not match workload')
    known = {r['request_id'] for r in unique}
    predictions = {}
    for path in args.predictions:
        for row in read(path):
            key = row.get('request_id')
            if key not in known or key in predictions:
                raise ValueError('Unknown or duplicate canonical prediction')
            predictions[key] = row
    outputs = []
    for item in mapping:
        source = item['canonical_request_id']
        if source not in predictions:
            continue
        result = dict(predictions[source], request_id=item['request_id'])
        if source != item['request_id']:
            result['reused_from_request_id'] = source
        outputs.append(result)
    write(args.output, outputs)
    print(f'Expanded {len(predictions)} calls -> {len(outputs)}/{len(workload)} requests')

if __name__ == '__main__':
    main()
