"""Convert adapter xy_m output to the benchmark's 20-point trajectory JSONL.

The original model output remains untouched. Status and geometry failures are
retained as null trajectories so private scoring can count them as invalid.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    valid = invalid = 0
    reasons = {}
    with args.raw.open(encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = row.get('request_id')
            if not isinstance(request_id, str) or request_id in seen:
                raise ValueError(f'Missing or duplicate request ID: {request_id}')
            seen.add(request_id)
            points = row.get('xy_m')
            reason = None
            if row.get('status') != 'ok':
                reason = f"adapter_status_{row.get('status')}"
            elif row.get('coordinate_frame') != 'ego_at_anchor' or row.get('reference_point') != 'rear_axle':
                reason = 'unexpected_coordinate_frame'
            elif not isinstance(points, list) or len(points) != 20 or any(
                not isinstance(point, list) or len(point) != 2 or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                    for value in point) for point in points):
                reason = 'invalid_20_point_xy_m'
            if reason:
                invalid += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                result = {'request_id': request_id, 'trajectory': None,
                          'conversion_error': reason}
            else:
                valid += 1
                result = {'request_id': request_id, 'trajectory': points}
            target.write(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    report = {'raw_path': str(args.raw), 'raw_sha256': sha256(args.raw),
              'output_path': str(args.output), 'output_sha256': sha256(args.output),
              'rows': len(seen), 'conversion_valid': valid,
              'conversion_invalid': invalid, 'invalid_reasons': reasons,
              'conversion_version': 'adapter_xy_m_to_wod_4hz_v1'}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
