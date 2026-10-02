#!/usr/bin/env python3
"""Install the actual wheel in an empty venv and extract all five backends."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

wheel = Path(sys.argv[1]).resolve()
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    assert 'l2d_benchmark/resources/backends/manifest.json' in names, 'Build with scripts/build_release.py'
    assert not any(n.endswith(('.pt', '.pth', '.ckpt', '.safetensors')) for n in names)
    assert any(n.endswith('THIRD_PARTY_NOTICES') for n in names)
with tempfile.TemporaryDirectory(prefix='l2d-install-') as temp:
    root = Path(temp)
    subprocess.run([sys.executable, '-m', 'venv', str(root / 'env')], check=True)
    python = str(root / 'env/bin/python')
    cli = str(root / 'env/bin/language2drive')
    subprocess.run([python, '-m', 'pip', 'install', '--no-deps', '--no-index', str(wheel)], check=True)
    workspace = root / 'workspace'
    subprocess.run([cli, 'init', '--workspace', str(workspace)], check=True, cwd=root)
    subprocess.run([cli, 'doctor', '--workspace', str(workspace)], check=True, cwd=root)
    for name in ['language2drive.carla', 'language2drive.RealSim', 'language2drive.OpenloopTrack.nScenes', 'language2drive.OpenloopTrack.navsim', 'language2drive.OpenloopTrack.WOD-E2E']:
        assert (workspace / name / 'README.md').is_file()
    subprocess.run([cli, 'adapter-check', '--adapter', str(workspace / 'examples/constant_velocity.py')], check=True, cwd=root)
    subprocess.run([cli, 'setup', '--workspace', str(workspace), '--tracks', 'navsim', '--network', 'official', '--package', str(wheel), '--dry-run'], check=True, cwd=root)
    # No PyPI lookup or original wheel path is needed after the initial install.
    planned = subprocess.check_output([cli, 'setup', '--workspace', str(workspace), '--tracks', 'openloop', '--network', 'official', '--dry-run'], cwd=root)
    assert all('language2drive==' not in ' '.join(step['argv']) for step in json.loads(planned)['commands'])
    code = 'from l2d_benchmark.installed_wheel import cached_wheel; print(cached_wheel(' + repr(str(workspace)) + '))'
    cached = Path(subprocess.check_output([python, '-c', code], cwd=root, text=True).strip())
    with zipfile.ZipFile(cached) as archive:
        assert 'l2d_benchmark/resources/backends/manifest.json' in archive.namelist()
    subprocess.run([python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', '--no-index', str(cached)], check=True, cwd=root)
    configuration = workspace / '.l2d/evaluation.json'
    subprocess.run([cli, 'configure', '--workspace', str(workspace), '--output', str(configuration)], check=True, cwd=root)
    subprocess.run([cli, 'run', '--config', str(configuration), '--adapter', str(workspace / 'examples/constant_velocity.py'), '--tracks', 'carla,realsim,openloop', '--dry-run'], check=True, cwd=root)
    subprocess.run([cli, 'evaluate', '--config', str(workspace / 'examples/evaluation.json'), '--tracks', 'all', '--output', str(root / 'outputs'), '--dry-run'], check=True, cwd=root)
    assert not (root / 'outputs').exists()
    # Extraction refuses differing files rather than overwriting users' work.
    target = workspace / 'language2drive.RealSim/README.md'
    target.write_text('modified by the user')
    assert subprocess.run([cli, 'init', '--workspace', str(workspace)], cwd=root).returncode != 0
print('PASS: clean wheel install, all five backends, CLI and immutable extraction')
