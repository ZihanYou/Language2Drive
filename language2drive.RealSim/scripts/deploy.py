"""Create an isolated evaluator environment, test it, and run the CPU demo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=root / '.venv')
    parser.add_argument('--index-url', default='https://pypi.tuna.tsinghua.edu.cn/simple')
    parser.add_argument('--offline', action='store_true', help='Use only locally available wheels')
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--keep-proxy', action='store_true', help='Explicitly opt into inherited proxy settings')
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('Python 3.10+ is required; select it with L2D_PYTHON')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    environment.pop('PYTHONPATH', None)
    if not args.keep_proxy:
        for name in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
            environment.pop(name, None)
    # The selected index must not be overridden by a machine-wide pip configuration.
    for name in tuple(environment):
        if name.startswith('PIP_'):
            environment.pop(name)
    environment['PIP_CONFIG_FILE'] = os.devnull
    target = args.venv.resolve()
    if target.exists() and not (target / 'pyvenv.cfg').is_file():
        parser.error('--venv exists but is not a virtual environment; refusing to modify it')
    if not target.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True, env=environment)
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    index = ['--no-index'] if args.offline else ['--index-url', args.index_url]
    if args.wheelhouse:
        if not args.wheelhouse.is_dir():
            parser.error('--wheelhouse must be a directory')
        index += ['--find-links', str(args.wheelhouse.resolve())]
    def run(arguments):
        subprocess.run([str(python), *arguments], cwd=root, env=environment, check=True)
    run(['-m', 'pip', 'install', '--disable-pip-version-check', *index, 'setuptools>=68', 'wheel'])
    run(['-m', 'pip', 'install', '--disable-pip-version-check', '--no-build-isolation',
         *index, '-e', str(root) + '[test]'])
    if not args.skip_tests:
        run(['-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'])
    artifacts = root / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='deployment-', dir=artifacts))
    run(['-m', 'language2drive_realsim.review_cli', '--output', str(output / 'demo.json'), 'demo'])
    run([str(root / 'scripts/doctor.py')])
    print(json.dumps(dict(status='offline_evaluator_ready', environment=str(target),
                          demo=str(output / 'demo.json'),
                          scoring_command=f'{python} -m language2drive_realsim.paper_metrics --help'), indent=2))


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print('Deployment stopped at a failed step; no success has been claimed. '
              'Resolve the reported dependency/test error, then rerun.', file=sys.stderr)
        raise SystemExit(exc.returncode)
