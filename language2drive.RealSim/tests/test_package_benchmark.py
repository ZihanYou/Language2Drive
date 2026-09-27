"""The export must not change benchmark contracts or silently delete provenance."""
from copy import deepcopy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'package_benchmark', Path(__file__).parents[1] / 'scripts/package_benchmark.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_portable_cohort_preserves_semantics_and_source(tmp_path):
    root = str(tmp_path)
    source = dict(schema_version='test', release_profile={'reason': 'internal', 'coverage_limitations': ['kept']},
                  pcus=[dict(pcu_id='example', arms={'R0': {'target_speed': [1, 2]}},
                             private_annotations={
                                 'lanes': [{'polygon': [[1, 2], [3, 4]]}],
                                 'render_validity_protocol': {'pilot_evidence_refs': [root + '/review.json']},
                                 'trigger_referent': {'frame_refs': [root + '/frame.png']},
                                 'shared_lane_scope_basis': {'source_scope': root + '/scope.json'}})])
    before = deepcopy(source)
    exported = package.portable_cohort(source, tmp_path)
    assert source == before
    assert exported['pcus'][0]['arms'] == source['pcus'][0]['arms']
    ann = exported['pcus'][0]['private_annotations']
    assert ann['lanes'] == source['pcus'][0]['private_annotations']['lanes']
    assert ann['trigger_referent']['frame_refs'] == ['${AUTHORING_ROOT}/frame.png']
    assert ann['render_validity_protocol']['pilot_evidence_refs'] == ['${AUTHORING_ROOT}/review.json']
    assert ann['shared_lane_scope_basis']['source_scope'] == '${AUTHORING_ROOT}/scope.json'
    assert exported['release_profile'] == source['release_profile']
