from copy import deepcopy
import json
import math

import pytest

from language2drive_realsim.paper_metrics import (
    ARM_METRICS, aggregate_pairs, l2d_summary, score_document, task_deadline_s,
)
from language2drive_realsim.physical_only_metrics import route_completion_percent


def pair(pcu, rewrite, first, second):
    return dict(pcu_id=pcu, rewrite_id=rewrite, arms={
        'R0': dict.fromkeys(ARM_METRICS, first), 'R1': dict.fromkeys(ARM_METRICS, second)})


def test_pcu_macro_not_pooled_and_exact_identities():
    rows = [pair('a', '0', 1., 0.), pair('a', '1', 1., 0.), pair('a', '2', 1., 0.),
            pair('b', '0', 1., 1.)]
    result = aggregate_pairs(rows)
    assert result['metrics']['SISR'] == 75
    assert result['metrics']['CFPS'] == 50
    assert result['metrics']['SCS'] == 75
    assert result['metrics']['PCS'] == 50
    assert result['identities']['one_sided_success_percent'] == 50
    assert result['identities']['strict_residual'] == pytest.approx(0)
    assert result['identities']['graded_residual'] == pytest.approx(0)


def test_graded_is_minimum_not_product():
    row = pair('a', '0', 0., 0.)
    row['arms']['R0']['SCS'], row['arms']['R1']['SCS'] = .8, .2
    result = aggregate_pairs([row])
    assert result['metrics']['SCS'] == 50
    assert result['metrics']['PCS'] == 20


def test_l2d_zero_language_floor_and_full_scale():
    result = aggregate_pairs([pair('a', '0', 1., 1.)])
    assert result['metrics']['L2D'] == pytest.approx(100)
    scores = dict.fromkeys(result['metrics'], 0.)
    scores.update(Safety=100., DrivingScore=100.)
    assert l2d_summary(scores)['L2D'] == 1
    scores.update(Safety=0., DrivingScore=0.)
    assert l2d_summary(scores)['L2D'] == 0


@pytest.mark.parametrize('values,expected', [
    ([13.33, 0., 18.33, 0., 21.43, 0., 93.33, 97.10], 9.22),
    ([11.67, 0., 15., 0., 18.12, 1.11, 100., 97.17], 8.38),
    ([6.67, 3.33, 13.33, 6.67, 16.67, 10., 68.33, 83.09], 7.58),
    ([1.67, 0., 8.33, 6.67, 9.73, 6.67, 41.67, 76.94], 3.36),
    ([0., 0., 1.67, 0., 4.44, 0., 43.33, 69.16], .92),
    ([0., 0., 1.67, 0., 3.71, 0., 30., 68.85], .78),
    ([0., 0., 0., 0., 0., 0., 43.33, 43.82], .44),
    ([0., 0., 0., 0., 0., 0., 33.33, 43.57], .38),
])
def test_published_table_rounding_crosscheck(values, expected):
    # Algebra check against displayed (already rounded) Table 2, not a rerun.
    names = ('SISR', 'CFPS', 'SemISR', 'Sem-CFPS', 'SCS', 'PCS', 'Safety', 'DrivingScore')
    assert l2d_summary(dict(zip(names, values)))['L2D'] == pytest.approx(expected, abs=.012)


@pytest.mark.parametrize('value', [None, True, math.nan, math.inf, -1., 1.1])
def test_invalid_grades_rejected(value):
    row = pair('a', '0', 0., 0.)
    row['arms']['R0']['SCS'] = value
    with pytest.raises(ValueError):
        aggregate_pairs([row])


def test_duplicate_empty_and_missing_arm_rejected():
    row = pair('a', '0', 0., 0.)
    with pytest.raises(ValueError):
        aggregate_pairs([row, row])
    with pytest.raises(ValueError):
        aggregate_pairs([])
    del row['arms']['R1']
    with pytest.raises(ValueError):
        aggregate_pairs([row])


def test_route_uses_own_deadline_not_opposing_tail():
    pcu = dict(decision_window=dict(deadline_s=4.), arms={'R0': dict(
        semantic_contract=dict(deadline_s=2.), opposing_contract=dict(deadline_s=6.))},
        oracle_plans_by_seed={'1701': {'R0': {'waypoints': [
            dict(time_s=t, x=t, y=0.) for t in (0., 2., 4., 6.)]}}})
    rows = [dict(time_s=t, position=[t, 0.]) for t in (0., 1., 2.)]
    assert task_deadline_s(pcu, 'R0') == 2.
    assert route_completion_percent(rows, pcu, 'R0', horizon_s=2.) == 100.
    rows[1]['position'] = [1., 6.]
    rows[2]['position'] = [2., 6.]
    assert route_completion_percent(rows, pcu, 'R0', horizon_s=2.) == 0.


def cohort(tmp_path):
    polygon = {'type': 'Polygon', 'coordinates': [[[-1., -2.], [30., -2.],
                [30., 2.], [-1., 2.], [-1., -2.]]]}
    rows = [dict(time_s=t, position=[3*t, 0.], speed_mps=3., progress_m=3*t,
                 collision=False, offroad=False, render_valid=True) for t in (0., 1., 2., 3., 4.)]
    path = tmp_path / 'trajectory.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    contract = dict(kind='speed_band', scope='s', speed_min_mps=2., speed_max_mps=4., deadline_s=4.)
    pcus, records = [], []
    for i in range(30):
        pcu = dict(pcu_id=f'p{i}', task_family='B1_segment_speed_band',
                   decision_window=dict(start_s=1., deadline_s=4.),
                   private_annotations=dict(scopes={'s': polygon}), arms={},
                   oracle_plans_by_seed={'1701': {}})
        for arm in ('R0', 'R1'):
            pcu['arms'][arm] = dict(semantic_contract=deepcopy(contract), opposing_contract=deepcopy(contract))
            pcu['oracle_plans_by_seed']['1701'][arm] = dict(waypoints=[
                dict(time_s=row['time_s'], x=row['position'][0], y=0.) for row in rows])
            records.append(dict(pcu_id=pcu['pcu_id'], arm_id=arm, seed=1701,
                                execution_status='completed', evidence_valid=True,
                                origin={'trace': str(path)}, components=dict(own_contract=True,
                                opposing_contract=False, progress=True, deadline=True, stability=True)))
        pcus.append(pcu)
    return pcus, dict(seed=1701, expected_runs=60, results=records)


def test_invalid_scheduled_arm_is_zero_with_fixed_denominator(tmp_path):
    pcus, document = cohort(tmp_path)
    document['results'][0].update(execution_status='failed', evidence_valid=False, origin={})
    result = score_document(document, pcus)
    for name in ARM_METRICS:
        assert result['metrics'][name] == pytest.approx(100*59/60)
    for name in ('CFPS', 'Sem-CFPS', 'PCS'):
        assert result['metrics'][name] == pytest.approx(100*29/30)
    assert result['coverage']['zero_credit_slots'] == 1
    assert result['identities']['graded_residual'] == pytest.approx(0)


def test_all_unavailable_is_numeric_zero_not_null(tmp_path):
    pcus, document = cohort(tmp_path)
    for row in document['results']:
        row.update(execution_status='unavailable', evidence_valid=False, origin={})
    result = score_document(document, pcus)
    assert set(result['metrics'].values()) == {0.}
    assert result['coverage']['scheduled_slots'] == 60


def test_valid_claims_need_components_and_trace(tmp_path):
    pcus, document = cohort(tmp_path)
    document['results'][0]['components'].pop('progress')
    with pytest.raises(ValueError, match='component'):
        score_document(document, pcus)


def test_slot_inventory_and_seed_are_enforced(tmp_path):
    pcus, document = cohort(tmp_path)
    document['results'][0]['seed'] = 99
    with pytest.raises(ValueError, match='seed'):
        score_document(document, pcus)
    document['results'].pop()
    with pytest.raises(ValueError, match='60'):
        score_document(document, pcus)
