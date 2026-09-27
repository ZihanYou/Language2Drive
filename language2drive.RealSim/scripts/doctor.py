"""Offline evaluator check; optionally require closed-loop deployment inputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-runtime', action='store_true')
    args = parser.parse_args()
    from language2drive_realsim.paper_metrics import PROFILE
    checks = dict(python=sys.version_info >= (3, 10), evaluator_import=True)
    if args.require_runtime:
        for executable, command in [('docker', ['docker', 'info', '--format', '{{.ServerVersion}}']),
                                    ('nvidia-smi', ['nvidia-smi', '-L'])]:
            if not shutil.which(executable):
                checks[executable] = False
                continue
            try:
                checks[executable] = subprocess.run(command, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=15).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                checks[executable] = False
        for name in ('L2D_PCUS_PATH', 'L2D_SCENE_ROOT', 'L2D_CHECKPOINT_ROOT'):
            value = os.environ.get(name)
            checks[name] = bool(value and (Path(value).is_file() if name == 'L2D_PCUS_PATH'
                                           else Path(value).is_dir()))
        checks['alpasim_package'] = importlib.util.find_spec('alpasim') is not None
    print(json.dumps(dict(profile=PROFILE, checks=checks, passed=all(checks.values()),
                          scope='runtime_prerequisites' if args.require_runtime else 'offline_evaluator'), indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
