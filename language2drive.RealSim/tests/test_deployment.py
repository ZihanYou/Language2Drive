import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_setup_help_and_shell_syntax():
    for name in ('setup.sh', 'score.sh'):
        subprocess.run(['sh', '-n', str(ROOT / 'scripts' / name)], check=True)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/deploy.py'), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert '--offline' in result.stdout
    assert '--keep-proxy' in result.stdout


def test_setup_refuses_non_environment_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/deploy.py'),
                             '--venv', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'refusing to modify' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_doctor_cpu_mode():
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/doctor.py')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert '"passed": true' in result.stdout
