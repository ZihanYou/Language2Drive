import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from language2drive import frozen_paths
from language2drive.io import sha256_file, stable_hash
from language2drive.release_scoring import stamp_scope
from language2drive.release_report import render
from language2drive.runner import _agent_paths, _validate_plan_sha256, RELEASE_AGENTS

ROOT = Path(__file__).resolve().parents[1]

def test_frozen_identity_reads_only_packaged_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen_paths, 'ROOT', tmp_path)
    destination = tmp_path / 'assets/provenance/runtime/witness.json'
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'packaged witness')
    name = '/root/autodl-tmp/lwq/language2drive.v1/.runtime/witness.json'
    path = frozen_paths.ProvenancePath(name)
    assert str(path) == name
    assert str(path.resolve(strict=True)) == name
    assert path.read_bytes() == b'packaged witness'
    assert sha256_file(path) == hashlib.sha256(b'packaged witness').hexdigest()
    destination.unlink()
    with pytest.raises(FileNotFoundError):
        path.read_bytes()

def test_scope_keeps_full_denominator_when_one_pending_job_is_selected(tmp_path):
    ledger = tmp_path / 'ledger.json'
    runs = [dict(run_id=f'r{i}', seed=1, split='test', track='standardized', route_entry_id=i)
            for i in range(4)]
    ledger.write_text(json.dumps({'runs': runs}))
    args = SimpleNamespace(command='run', seeds=[1], split='test', tier='base',
                           max_runs=1, route_entry_ids=None)
    plan = {'track': 'standardized', 'jobs': [{'run': runs[0]}]}
    stamp_scope(plan, args, ledger, tmp_path)
    assert plan['evaluation_scope']['run_ids'] == ['r0', 'r1', 'r2', 'r3']
    _validate_plan_sha256(plan)
    plan['evaluation_scope']['run_ids'].pop()
    with pytest.raises(ValueError, match='plan_sha256'):
        _validate_plan_sha256(plan)

def test_exact_route_scope_and_seed_selection(tmp_path):
    ledger = tmp_path / 'ledger.json'
    ledger.write_text(json.dumps({'runs': [
        dict(run_id=f'r{i}s{seed}', seed=seed, split='dev', track='standardized', route_entry_id=i)
        for seed in [1,2] for i in range(4)]}))
    args = SimpleNamespace(command='run', seeds=[2], split='dev', tier='small', route_entry_ids=[1,3])
    plan = {'track': 'standardized', 'jobs': []}
    stamp_scope(plan, args, ledger, tmp_path)
    assert plan['evaluation_scope']['run_ids'] == ['r1s2', 'r3s2']

def test_only_requested_models_are_registered():
    expected = {'alpamayo1.5','simlingo','lmdrive','onevl','impromptu-vla','qwen-drive-1.0-4b','senna'}
    assert RELEASE_AGENTS == expected
    assert set(json.loads((ROOT/'model_zoo/SOURCES.json').read_text())['sources']) == expected
    for model in expected:
        agent, config = _agent_paths(model, None, None)
        assert agent.is_file() and config.is_file()
        assert agent.parent.name == model

def test_report_exposes_scope_and_expected_denominator(tmp_path):
    score = {'schema_version':'language2drive.release_score.v1', 'model':'onevl',
             'track':'standardized', 'workload':'core', 'plan_sha256':'a'*64,
             'scope':{'run_ids':['a','b']},
             'metrics':{'expected_runs':2,'valid_result_runs':1,
                        'SISR_by_layer':{'A':0.5},'SemISR_by_layer':{'A':0.5},
                        'CFPS':{'A':0},'Sem-CFPS':{'A':0}}}
    source=tmp_path/'score.json';source.write_text(json.dumps(score))
    output=tmp_path/'report.md'
    render(source,output,json_output=tmp_path/'report.json')
    text=output.read_text()
    assert 'Expected runs: **2**' in text and 'Valid results: **1**' in text
    assert '50.00%' in text and 'denominator' in text
    assert json.loads((tmp_path/'report.json').read_text()) == score

def test_resource_archive_rejects_path_escape(tmp_path):
    import io,tarfile
    spec=importlib.util.spec_from_file_location('l2d_download_resources',ROOT/'tools/download_resources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        item=tarfile.TarInfo('../escape');item.size=1;bundle.addfile(item,io.BytesIO(b'x'))
    with pytest.raises(tarfile.FilterError):
        module.extract(archive,tmp_path/'destination')
    assert not (tmp_path/'escape').exists()
