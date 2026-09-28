"""Diagnostic reference alignment for every drafted pair in the full prerelease.

Closer-to-reference is a motion comparison, not a semantic or safety success.
The output labels are therefore deliberately named reference_alignment.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from score_prerelease import normalize_trajectory, rows, digest


def ade(points, reference):
    return sum(math.dist(a,b) for a,b in zip(points,reference))/20


def evaluate(candidate, predictions):
    ids = candidate['request_ids']
    references = [ref['xy_m'] for ref in candidate['references']]
    details = {}
    for index,arm in enumerate(('a','b')):
        request_id = ids.get(arm)
        row = predictions.get(request_id)
        if row is None:
            details[arm] = {'status':'missing','request_id':request_id}
            continue
        normalized,reason = normalize_trajectory(row.get('trajectory'))
        if reason:
            details[arm] = {'status':'invalid','request_id':request_id,'reason':reason}
            continue
        points,_ = normalized
        own,other = ade(points,references[index]),ade(points,references[1-index])
        margin = other-own
        details[arm] = {'status':'valid','request_id':request_id,
                        'own_reference_ade_m':own,'other_reference_ade_m':other,
                        'reference_margin_m':margin,
                        'nearest_reference_agreement':margin>0.5,
                        'ambiguous_reference_distance':abs(margin)<=0.5,
                        'x5_m':points[-1][0],'y5_m':points[-1][1]}
    complete = all(details[arm]['status']=='valid' for arm in ('a','b'))
    return {'candidate_id':candidate['candidate_id'],
            'task_family_draft':candidate['task_family_draft'],
            'scenario_cluster':candidate['scenario_cluster'],
            'history_complete':candidate['history_complete'],
            'requests':details,'pair_complete':complete,
            'paired_reference_alignment':complete and all(details[arm]['nearest_reference_agreement'] for arm in ('a','b')),
            'paired_mean_margin_m':sum(details[arm]['reference_margin_m'] for arm in ('a','b'))/2 if complete else None}


def summary(items, total_pairs):
    requested = 2*total_pairs
    seen = valid = agreement = ambiguous = 0
    margins = []
    complete = paired = 0
    for item in items:
        for arm in ('a','b'):
            r = item['requests'][arm]
            if r['status'] != 'missing':
                seen += 1
            if r['status'] == 'valid':
                valid += 1
                agreement += int(r['nearest_reference_agreement'])
                ambiguous += int(r['ambiguous_reference_distance'])
                margins.append(r['reference_margin_m'])
        if item['pair_complete']:
            complete += 1
            paired += int(item['paired_reference_alignment'])
    return {'candidate_pairs':total_pairs,'requested':requested,
            'predicted':seen,'valid':valid,'missing':requested-seen,
            'invalid':seen-valid,'request_coverage':seen/requested if requested else None,
            'nearest_reference_agreement_count':agreement,
            'nearest_reference_agreement_rate_on_valid':agreement/valid if valid else None,
            'ambiguous_distance_count':ambiguous,
            'mean_reference_margin_m_on_valid':sum(margins)/len(margins) if margins else None,
            'complete_pairs':complete,'paired_reference_alignment_count':paired,
            'paired_reference_alignment_rate_on_complete':paired/complete if complete else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--model-id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    release = args.release_dir
    manifest = json.loads((release/'manifest.json').read_text())
    if manifest['release_status'] != 'all_pending_unreviewed':
        raise ValueError('Expected full unreviewed candidate prerelease')
    for name,expected in manifest['file_sha256'].items():
        if digest(release/name) != expected:
            raise ValueError('Release file hash mismatch: '+name)
    workload = list(rows(release/'public/workload.jsonl'))
    known = {r['request_id'] for r in workload}
    if len(known) != len(workload):
        raise ValueError('Duplicate request ID in release')
    predictions = {}
    for row in rows(args.predictions):
        key = row.get('request_id')
        if key not in known:
            raise ValueError('Unknown request ID: '+str(key))
        if key in predictions:
            raise ValueError('Duplicate prediction: '+key)
        predictions[key] = row
    registry = list(rows(release/'private/candidates.jsonl'))
    if len(registry) != manifest['candidate_pairs']:
        raise ValueError('Candidate count does not match release manifest')
    drafted = [r for r in registry if r['status']=='DRAFT_P0_UNREVIEWED']
    if len(drafted)*2 != len(workload):
        raise ValueError('Draft pair/request count mismatch')
    details = [evaluate(candidate,predictions) for candidate in drafted]
    groups = defaultdict(list)
    for item in details:
        groups[item['task_family_draft']].append(item)
    overall = summary(details,len(drafted))
    overall['all_pending_candidate_pair_coverage'] = overall['complete_pairs']/len(registry)
    report = {'model_id':args.model_id,'release_version':manifest['version'],
              'release_manifest_sha256':digest(release/'manifest.json'),
              'prediction_sha256':digest(args.predictions),
              'status':'unreviewed_reference_alignment_diagnostic',
              'all_pending_candidates':manifest['candidate_pairs'],
              'unresolved_without_request':manifest['status_counts'].get('UNRESOLVED_UNREVIEWED',0),
              'overall':overall,
              'by_draft_family':{name:summary(items,len(items)) for name,items in sorted(groups.items())}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (args.output_dir/'per_candidate.jsonl').open('w') as f:
        for item in details:
            f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
