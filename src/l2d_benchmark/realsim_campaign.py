"""Execute custom-model RealSim sessions and score receipt-bound native ASLs."""
import json
from pathlib import Path
from .runners import read_jsonl


def campaign(cfg, output):
    from language2drive_realsim.runtime_client import build_run_plan, submit_run_plan
    from language2drive_realsim.alpasim_bridge import export_asl, ALPASIM_COMMIT
    from language2drive_realsim.component_scoring import score_trace_components
    from language2drive_realsim.paper_metrics import score_document
    from language2drive_realsim.physical_only_metrics import STRICT
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    public = json.loads(Path(cfg['public_workload']).read_text())
    sidecar = json.loads(Path(cfg['sidecar']).read_text())
    pcus = json.loads(Path(cfg['pcus']).read_text())['pcus']
    ids = {p['pcu_id'] for p in pcus}
    plan = build_run_plan(public, sidecar)
    plan['runs'] = [r for r in plan['runs'] if r['random_seed'] == 1701 and r['pcu_id'] in ids]
    if len(ids) != 30 or len(plan['runs']) != 60:
        raise ValueError('RealSim paper protocol requires frozen 30 PCUs and 60 slots at seed 1701')
    plan['run_count'] = len(plan['runs'])
    (output / 'run_plan.json').write_text(json.dumps(plan, indent=2))
    returned = submit_run_plan(plan, cfg['runtime_address'], cfg['driver_address'], int(cfg.get('concurrency', 1)))
    (output / 'runtime_return.json').write_text(json.dumps(returned, indent=2, allow_nan=False))
    if not returned['driver_version'].startswith('l2d_unified-driver-') or returned['runtime_git_hash'] != ALPASIM_COMMIT:
        raise ValueError('Runtime must attest the pinned commit and l2d_unified driver identity')
    sessions, errors, source_hashes = {}, [], {}
    invalid_sessions = set()
    for index, asl in enumerate(sorted(Path(cfg['asl_root']).rglob('*.asl'))):
        trace = output / 'traces' / f'{index:05d}.jsonl'
        try:
            export_asl(asl, Path(cfg['eval_config']), cfg['usdz_glob'], Path(cfg['sidecar']), trace, ALPASIM_COMMIT)
            receipt = json.loads(trace.with_suffix('.receipt.json').read_text())
            session = receipt.get('session_uuid')
            if session not in {r['session_uuid'] for r in plan['runs']}:
                continue
            if session in sessions or receipt['source_asl_sha256'] in source_hashes:
                invalid_sessions.add(session)
                if receipt['source_asl_sha256'] in source_hashes:
                    invalid_sessions.add(source_hashes[receipt['source_asl_sha256']])
                raise ValueError('Duplicate ASL/session evidence')
            if not receipt['egodriver_version_id'].startswith('l2d_unified-driver-'):
                raise ValueError('ASL driver identity mismatch')
            source_hashes[receipt['source_asl_sha256']] = session
            sessions[session] = (trace, receipt)
        except Exception as error:
            errors.append({'asl': str(asl), 'error': str(error)})
    for session in invalid_sessions:
        sessions.pop(session, None)
    outcomes = {r['session_uuid']: r for r in returned['rollout_returns']}
    records, pair_hashes = [], {}
    for run in plan['runs']:
        row = dict(pcu_id=run['pcu_id'], arm_id=run['arm_id'], seed=1701,
                   execution_status='unavailable', evidence_valid=False)
        evidence = sessions.get(run['session_uuid'])
        if evidence and outcomes.get(run['session_uuid'], {}).get('success') is True:
            trace, receipt = evidence
            pcu = next(p for p in pcus if p['pcu_id'] == run['pcu_id'])
            try:
                verdict = score_trace_components(pcu, run['arm_id'], read_jsonl(trace), annotations=sidecar['sessions'][run['session_uuid']]['annotations'])
                components = verdict['components']
                if any(type(components.get(key)) is not bool for key in STRICT):
                    raise ValueError('Trace lacks binary strict component evidence')
                row.update(components=components, origin={'trace': str(trace.resolve())},
                           execution_status='completed', evidence_valid=verdict['trace_usable'])
                pair_hashes.setdefault(run['pcu_id'], []).append(receipt['preintervention_non_language_sha256'])
            except Exception as error:
                row['error'] = str(error)
        records.append(row)
    # Do not award either arm if actual shared-input evidence differs.
    for pcu_id, hashes in pair_hashes.items():
        if len(hashes) == 2 and hashes[0] != hashes[1]:
            for row in records:
                if row['pcu_id'] == pcu_id:
                    row.update(evidence_valid=False, error='Paired pre-intervention evidence mismatch')
    document = dict(model_name=cfg.get('model_name', 'custom'), configuration_id='unified-sdk-v0.1', seed=1701, expected_runs=60, results=records)
    (output / 'experiment.json').write_text(json.dumps(document, indent=2))
    score = score_document(document, pcus, 'unified SDK custom model; native paper protocol')
    score.update(formal_submission_approved=False, conversion_errors=errors)
    (output / 'score.json').write_text(json.dumps(score, indent=2, allow_nan=False))
    return score
