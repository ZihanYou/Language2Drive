"""Check runtime Python imports and optional host GPU/container prerequisites."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'config/alpasim_realsim.json').read_text())['alpasim_commit']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--require-docker', action='store_true')
    a = p.parse_args()
    checks = {}

    def check(name, fn):
        try:
            result = fn()
            checks[name] = dict(passed=bool(result))
        except Exception as error:
            checks[name] = dict(passed=False, error=f'{type(error).__name__}: {error}')

    def command(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=20).strip()

    check('python_3_12', lambda: sys.version_info[:2] == (3, 12))
    check('alpasim_pin', lambda: command(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD']) == PIN)
    for name in ('alpasim_runtime.simulate', 'alpasim_driver.main', 'alpasim_controller.server',
                 'alpasim_physics.server', 'alpasim_grpc.v0.runtime_pb2', 'utils_rs',
                 'language2drive_realsim.paper_metrics', 'language2drive_realsim.host_runtime'):
        check('import:' + name, lambda name=name: importlib.import_module(name) is not None)
    check('casadi_overlay', lambda: importlib.metadata.version('casadi') == '3.7.2')
    check('casadi_legacy_symbols', lambda: all(hasattr(importlib.import_module('casadi.tools'), k) for k in ('SX', 'MX')))
    required = {'l2d_alpamayo1_5', 'l2d_alpamayo2_super', 'l2d_simlingo', 'l2d_privileged_oracle'}
    check('driver_registration', lambda: required <= {e.name for e in importlib.metadata.entry_points(group='alpasim.models')})
    check('alpamayo2_import', lambda: importlib.import_module('language2drive_realsim.alpasim_model')._alpamayo2_helper is not None)
    check('alpamayo15_import', lambda: importlib.import_module('language2drive_realsim.alpasim_model').Alpamayo15Model is not object)
    if a.require_gpu:
        check('nvidia_driver', lambda: all(int(v.split('.')[0]) >= 570 for v in
            command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).splitlines()))
        check('torch_cuda', lambda: importlib.import_module('torch').cuda.is_available())
    if a.require_docker:
        check('docker_daemon', lambda: bool(command(['docker', 'info', '--format', '{{.ServerVersion}}'])))
        check('docker_compose', lambda: bool(command(['docker', 'compose', 'version'])))
        check('docker_nvidia_runtime', lambda: 'nvidia' in command(['docker', 'info', '--format', '{{json .Runtimes}}']))
    passed = all(x['passed'] for x in checks.values())
    print(json.dumps(dict(scope='runtime_prerequisites', checks=checks, passed=passed,
                         gpu_closed_loop_verified=False), indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
