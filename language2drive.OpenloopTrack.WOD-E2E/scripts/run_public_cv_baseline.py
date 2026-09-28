"""CPU-only no-language constant-velocity diagnostic on the stop/go pilot.

Reads public inputs and requests only. The same trajectory is returned for
both instructions at a scene, so paired success must remain zero when the
two proxy contracts are mutually exclusive.
"""

import argparse
import json
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    inputs = {r['input_id']:r for r in rows(args.release_dir/'public/inputs.jsonl')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count = 0
    with args.output.open('w',encoding='utf-8') as f:
        for request in rows(args.release_dir/'public/workload.jsonl'):
            source = inputs[request['input_id']]
            vx = source['past_states']['vx'][-1]
            vy = source['past_states']['vy'][-1]
            trajectory = [[vx*(i+1)/4,vy*(i+1)/4] for i in range(20)]
            f.write(json.dumps({'request_id':request['request_id'],
                                'trajectory':trajectory},separators=(',',':'))+'\n')
            count += 1
    print(f'Wrote {count} public-only constant-velocity predictions to {args.output}')


if __name__=='__main__':
    main()
