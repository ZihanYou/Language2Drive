"""Environment orchestration regression tests: no network or GPU required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('setup_runtime', ROOT / 'scripts/setup_runtime.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_plan_is_non_mutating(tmp_path):
    target = tmp_path / 'not-created'
    output = subprocess.run([sys.executable, str(ROOT / 'scripts/setup_runtime.py'),
                             '--plan', '--work-dir', str(target)], capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    assert not target.exists()
    result = json.loads(output.stdout)
    sync = next(row['command'] for row in result['steps'] if row['name'] == 'sync-frozen-upstream')
    assert '--frozen' in sync and '--extra' in sync and 'all' in sync
    mirror = next(row['command'] for row in result['steps'] if row['name'] == 'download-hash-bound-mirror')
    assert '--require-hashes' in mirror and '--no-deps' in mirror
    assert result['proxy_enabled'] is False
    assert result['environment_archive_uploaded'] is False
    assert result['alpasim_commit'] == setup.PIN


def test_refuse_foreign_workdir(tmp_path):
    with pytest.raises(ValueError, match='ownership marker'):
        setup.validate_workdir(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_refuse_changed_revision(tmp_path):
    (tmp_path / setup.MARKER).write_text(json.dumps({'alpasim_commit': 'wrong'}))
    with pytest.raises(ValueError, match='another AlpaSim'):
        setup.validate_workdir(tmp_path)


def test_managed_workdir_can_resume(tmp_path):
    (tmp_path / setup.MARKER).write_text(json.dumps({'alpasim_commit': setup.PIN}))
    setup.validate_workdir(tmp_path)


def test_environment_removes_overrides_and_credentials(monkeypatch, tmp_path):
    for key in ('https_proxy', 'ALL_PROXY', 'UV_PROJECT_ENVIRONMENT', 'PYTHONPATH', 'HF_TOKEN'):
        monkeypatch.setenv(key, 'not-forwarded')
    env = setup.make_environment(tmp_path)
    assert all(k not in env for k in ('https_proxy', 'ALL_PROXY', 'PYTHONPATH', 'HF_TOKEN'))
    assert env['UV_PROJECT_ENVIRONMENT'] == str(tmp_path / 'venv')
    assert env['HF_HUB_OFFLINE'] == '1'
    assert env['GIT_LFS_SKIP_SMUDGE'] == '1'
    assert env['MPLCONFIGDIR'] == str(tmp_path / 'matplotlib-cache')
    assert setup.make_environment(tmp_path, keep_proxy=True)['https_proxy'] == 'not-forwarded'


def test_runtime_shell_syntax_and_help():
    subprocess.run(['sh', '-n', str(ROOT / 'scripts/setup_runtime.sh')], check=True)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/doctor_runtime.py'), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert '--require-gpu' in result.stdout


def test_release_supplies_upstream_lock():
    tomllib = pytest.importorskip('tomllib')
    lock = tomllib.loads((ROOT / 'config/alpasim-runtime.uv.lock').read_text())
    assert lock['version'] == 1
    assert lock['requires-python'] == '==3.12.*'
    assert any(p['name'] == 'alpasim-runtime' for p in lock['package'])
