import json
from copy import deepcopy

import pytest

from language2drive_realsim.review_cli import demo, main, portable_document, toy_pairs
from language2drive_realsim.physical_only_metrics import physical_safety_from_trace
from language2drive_realsim.supplemental_diagnostics import classify, summarize, make_runs


def test_all_three_synthetic_task_groups():
    result = demo()
    assert result['data_kind'].startswith('synthetic_')
    assert {p['task_group'] for p in result['pairs']} == {'speed', 'lane', 'route'}
    for pair in result['pairs']:
        assert pair['SCS'] == pair['PCS'] == 1
        assert 0 <= pair['NBD'] <= 1
        assert pair['SRG'] > 0
        assert pair['identity_residual'] == pytest.approx(0)


def test_relative_trace_and_seed_contract(tmp_path):
    document = dict(seed=1701, results=[dict(seed=1701, origin=dict(trace='traces/a.jsonl'))])
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(document))
    loaded = portable_document(path)
    assert loaded['results'][0]['origin']['trace'] == str(tmp_path / 'traces/a.jsonl')
    document['results'][0]['seed'] = 1702
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='seed disagree'):
        portable_document(path)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / 'result.json'
    path.write_text('keep')
    monkeypatch.setattr('sys.argv', ['l2d-review', '--output', str(path), 'demo'])
    with pytest.raises(FileExistsError):
        main()
    assert path.read_text() == 'keep'


def test_physical_safety_uses_collision_and_offroad_only(tmp_path):
    rows = deepcopy(toy_pairs()[0][1])
    path = tmp_path / 'trace.jsonl'
    for row in rows:
        row.update(red_light_violation=True, speeding=True, wrong_way=True)
    def write():
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write()
    assert physical_safety_from_trace(path) is True
    rows[-1]['collision'] = True
    write()
    assert physical_safety_from_trace(path) is False
    rows[-1]['render_valid'] = False
    write()
    with pytest.raises(ValueError, match='invalid simulation'):
        physical_safety_from_trace(path)


def test_missing_evidence_not_technical_failure():
    result = classify({'arm_id': 'R0'}, {'pcu_id': 'toy'})
    assert result['primary'] == 'evidence_unknown'
    assert result['labels'] == []
    explicit = classify({'arm_id': 'R0'}, {'pcu_id': 'toy'}, technical_failure='invalid_waypoints')
    assert explicit['labels'] == ['output_control_failure']


def test_failure_overlaps_are_not_summed_as_unique_failures():
    rows = [dict(primary='safety_violation', labels=['wrong_behavior', 'safety_violation'],
                 choice='wrong', physical_safe=False),
            dict(primary='correct_incomplete', labels=['correct_incomplete'],
                 choice='correct', physical_safe=True)]
    result = summarize(rows)
    assert sum(result['primary_counts'].values()) == 2
    assert sum(result['multilabel_counts_do_not_sum'].values()) == 3
    assert result['unique_any_labeled_failure'] == 2
    assert result['pairwise_overlaps']['wrong_behavior&safety_violation'] == 1


def test_control_generation_preserves_seed_and_timing():
    pcus = [p for p, _, _ in toy_pairs()]
    runs = [dict(pcu_id=p['pcu_id'], arm_id=arm, session_uuid=p['pcu_id']+arm,
                 random_seed=1701, scenario_id=p['pcu_id'], start_time_offset_us=20)
            for p in pcus for arm in ('R0', 'R1')]
    sidecar = dict(schema_version='l2d.realsim.sidecar.v0.1', sessions={
        r['session_uuid']: dict(instruction_schedule=[dict(text=r['session_uuid'], issued_at_us=100, version=1)],
                                annotations={'fixed': True}) for r in runs})
    plan, changed = make_runs({'runs': runs}, sidecar, pcus, 'synthetic')
    assert len(plan['runs']) == len(changed['sessions']) == 21
    assert len({r['session_uuid'] for r in plan['runs']}) == 21
    for row in plan['runs']:
        session = changed['sessions'][row['session_uuid']]
        assert row['random_seed'] == 1701
        assert session['instruction_schedule'][0]['issued_at_us'] == 100
        assert session['annotations'] == {'fixed': True}


def test_complete_synthetic_cohort_through_cli(tmp_path, monkeypatch):
    pcus, records = [], []
    for index in range(30):
        pcu, first, second = deepcopy(toy_pairs()[index % 3])
        pcu['pcu_id'] = 'synthetic-%02d' % index
        pcu['oracle_plans_by_seed'] = {'1701': {}}
        for arm, other, rows in [('R0', 'R1', first), ('R1', 'R0', second)]:
            pcu['arms'][arm]['opposing_contract'] = deepcopy(pcu['arms'][other]['semantic_contract'])
            pcu['oracle_plans_by_seed']['1701'][arm] = {'waypoints': [
                dict(time_s=r['time_s'], x=r['position'][0], y=r['position'][1]) for r in rows]}
            trace = tmp_path / (pcu['pcu_id'] + arm + '.jsonl')
            trace.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            records.append(dict(pcu_id=pcu['pcu_id'], arm_id=arm, seed=1701,
                                execution_status='completed', evidence_valid=True,
                                origin=dict(trace=trace.name),
                                components=dict(own_contract=True, opposing_contract=False,
                                                progress=True, deadline=True, stability=True,
                                                semantic_success=True)))
        pcus.append(pcu)
    source = tmp_path / 'pcus.json'
    experiment = tmp_path / 'experiment.json'
    output = tmp_path / 'scores.json'
    source.write_text(json.dumps({'pcus': pcus}))
    experiment.write_text(json.dumps(dict(model_name='synthetic-not-a-model-result',
                                          configuration_id='synthetic', seed=1701,
                                          expected_runs=60, results=records)))
    monkeypatch.setattr('sys.argv', ['l2d-review', '--output', str(output), 'score',
                                    '--pcus', str(source), '--experiment', str(experiment)])
    main()
    result = json.loads(output.read_text())
    assert result['coverage']['valid_slots'] == 60
    assert result['coverage']['scheduled_pairs'] == 30
    for metric in ('SISR', 'CFPS', 'SemISR', 'Sem-CFPS', 'SCS', 'PCS', 'Safety', 'DrivingScore', 'L2D'):
        assert result['metrics'][metric] == pytest.approx(100)
