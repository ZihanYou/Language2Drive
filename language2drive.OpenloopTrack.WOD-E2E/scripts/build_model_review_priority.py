"""Rank all pending candidates for human review from model geometry diagnostics.

The ranking is an audit worklist, never a semantic or safety success label.
Every candidate remains present, including those with no drafted instruction.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import quote


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score_arg(value):
    name, separator, raw_path = value.partition('=')
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError('Expected MODEL=per_candidate.jsonl')
    return name, Path(raw_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--model-score', action='append', type=score_arg, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.release_dir / 'manifest.json').read_text())
    if manifest['release_status'] != 'all_pending_unreviewed':
        raise ValueError('Expected all-pending prerelease')
    candidate_path = args.release_dir / 'private/candidates.jsonl'
    if sha256(candidate_path) != manifest['file_sha256']['private/candidates.jsonl']:
        raise ValueError('Candidate registry differs from frozen release')
    candidates = list(rows(candidate_path))
    if len(candidates) != manifest['candidate_pairs']:
        raise ValueError('Candidate count differs from manifest')
    drafted_ids = {c['candidate_id'] for c in candidates
                   if c['status'] == 'DRAFT_P0_UNREVIEWED'}
    scores = {}
    source_hashes = {}
    for name, path in args.model_score:
        if name in scores:
            raise ValueError(f'Duplicate model name: {name}')
        selected = {row['candidate_id']: row for row in rows(path)}
        if set(selected) != drafted_ids:
            raise ValueError(f'{name} score candidate IDs differ from this release')
        summary_path = path.with_name('summary.json')
        if summary_path.is_file():
            score_summary = json.loads(summary_path.read_text())
            if score_summary.get('release_manifest_sha256') != sha256(args.release_dir / 'manifest.json'):
                raise ValueError(f'{name} score was built for a different release')
        scores[name] = selected
        source_hashes[name] = {'path': str(path), 'sha256': sha256(path)}
    output = []
    for candidate in candidates:
        key = candidate['candidate_id']
        paired_models = []
        aligned_arms = 0
        complete_models = 0
        model_detail = {}
        for name, items in scores.items():
            item = items.get(key)
            if item is None:
                continue
            if item['pair_complete']:
                complete_models += 1
            if item['paired_reference_alignment']:
                paired_models.append(name)
            aligned_arms += sum(bool(item['requests'][arm].get('nearest_reference_agreement'))
                                for arm in ('a', 'b'))
            model_detail[name] = {
                'pair_complete': item['pair_complete'],
                'paired_reference_alignment': item['paired_reference_alignment'],
                'a': item['requests']['a']['status'],
                'b': item['requests']['b']['status'],
                'a_margin_m': item['requests']['a'].get('reference_margin_m'),
                'b_margin_m': item['requests']['b'].get('reference_margin_m'),
            }
        if candidate['status'] == 'UNRESOLVED_UNREVIEWED':
            tier = 'needs_instruction_draft'
        elif paired_models:
            tier = 'paired_geometry_candidate'
        elif aligned_arms:
            tier = 'single_arm_geometry_candidate'
        elif complete_models:
            tier = 'complete_but_no_alignment'
        else:
            tier = 'missing_model_prediction'
        output.append({
            'candidate_id': key,
            'review_tier': tier,
            'paired_geometry_models': ';'.join(paired_models),
            'paired_geometry_model_count': len(paired_models),
            'aligned_arms_across_models': aligned_arms,
            'complete_models': complete_models,
            'task_family_draft': candidate['task_family_draft'],
            'scenario_cluster': candidate['scenario_cluster'],
            'frame_name': candidate['frame_name'],
            'history_complete': candidate['history_complete'],
            'instruction_a_draft': candidate['instruction_a_draft'],
            'instruction_b_draft': candidate['instruction_b_draft'],
            'model_detail_json': json.dumps(model_detail, ensure_ascii=False, sort_keys=True),
            'review_url': 'http://127.0.0.1:8787/?candidate=' + quote(key) +
                          ('&model=' + quote(paired_models[0]) if paired_models else ''),
        })
    order = {'paired_geometry_candidate': 0, 'single_arm_geometry_candidate': 1,
             'complete_but_no_alignment': 2, 'missing_model_prediction': 3,
             'needs_instruction_draft': 4}
    output.sort(key=lambda row: (order[row['review_tier']],
                                 -row['paired_geometry_model_count'],
                                 -row['aligned_arms_across_models'],
                                 -row['complete_models'], row['candidate_id']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    count = {name: sum(row['review_tier'] == name for row in output) for name in order}
    report = {'status': 'unreviewed_geometry_review_worklist',
              'all_pending_candidates': len(candidates),
              'source_release_manifest_sha256': sha256(args.release_dir / 'manifest.json'),
              'model_score_sources': source_hashes,
              'review_tier_counts': count,
              'output_sha256': sha256(args.output)}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2,
                                                               ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
