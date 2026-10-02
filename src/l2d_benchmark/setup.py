"""Track-selective bootstrap and endpoint selection, with explicit runtime opt-in."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import build_opener, ProxyHandler
from .workspace import TRACKS, discover

EXTRAS = {'nuscenes': 'openloop', 'navsim': 'openloop', 'wod': 'openloop', 'carla': 'carla', 'realsim': 'realsim'}


def probe(url, trust_proxy=True):
    try:
        opener = build_opener() if trust_proxy else build_opener(ProxyHandler({}))
        with opener.open(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def reachable(url):
    return probe(url) or probe(url, trust_proxy=False)


def transport_environment(network, mode='auto'):
    env = os.environ.copy()
    proxy_keys = [key for key in env if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy')]
    direct = mode == 'direct'
    if mode == 'auto' and proxy_keys and not probe(network['PIP_INDEX_URL']):
        direct = probe(network['PIP_INDEX_URL'], trust_proxy=False)
    if direct:
        for key in proxy_keys:
            env.pop(key)
        env['NO_PROXY'] = env['no_proxy'] = '*'
    env.update(network)
    return env, 'direct' if direct else 'environment'


def sources(mode='auto'):
    hf_official = mode == 'official' or (mode == 'auto' and reachable('https://huggingface.co/api/models?limit=1'))
    pip_official = mode == 'official' or (mode == 'auto' and reachable('https://pypi.org/simple/pip/'))
    return {'HF_ENDPOINT': 'https://huggingface.co' if hf_official else 'https://hf-mirror.com',
            'PIP_INDEX_URL': 'https://pypi.org/simple' if pip_official else 'https://pypi.tuna.tsinghua.edu.cn/simple'}


def bootstrap(workspace, tracks, mode='auto', runtime=False, download=False, source_root=None, dry_run=False, package=None, transport='auto'):
    root = discover(workspace)
    network = sources(mode)
    env, transport = transport_environment(network, transport)
    sdk_source = Path(__file__).resolve().parents[2]
    editable = (sdk_source / 'pyproject.toml').is_file()
    # Install the same local checkout/wheel when provided, otherwise the released distribution.
    if package:
        package = str(Path(package).resolve())
    elif editable:
        package = str(sdk_source)
    else:
        from .installed_wheel import cached_wheel
        package = str(cached_wheel(root, dry_run))
    plan, blocked = [], []
    for track in tracks:
        backend = root / TRACKS[track]
        python_name = 'python3'
        interpreter = shutil.which(python_name) or python_name
        target = root / '.l2d/envs' / track
        python = str(target / 'bin/python')
        if track == 'carla' and not Path(python).is_file():
            conda = shutil.which('conda') or 'conda'
            channel = 'https://repo.anaconda.com/pkgs/main' if network['PIP_INDEX_URL'] == 'https://pypi.org/simple' else 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main'
            plan.append({'track': track, 'cwd': str(root), 'argv': [conda, 'create', '-y', '--override-channels', '-c', channel, '-p', str(target), 'python=3.11', 'pip', 'mkl-service']})
        elif track != 'carla' and not Path(python).is_file():
            plan.append({'track': track, 'cwd': str(root), 'argv': [interpreter, '-m', 'venv', str(target)]})
        install_target = package + '[' + EXTRAS[track] + ']' if not package.endswith('.whl') else package
        plan.append({'track': track, 'cwd': str(root), 'argv': [python, '-m', 'pip', 'install', '--index-url', network['PIP_INDEX_URL'], install_target]})
        if package.endswith('.whl'):
            # Wheel path extras cannot use a simple suffix; requirements are installed separately.
            import zipfile
            from email.parser import BytesParser
            if Path(package).is_file():
                with zipfile.ZipFile(package) as archive:
                    name = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
                    wheel_metadata = BytesParser().parsebytes(archive.read(name))
            else:
                from importlib.metadata import distribution
                wheel_metadata = distribution('language2drive').metadata
            requirements = [r.split(';')[0].strip() for r in wheel_metadata.get_all('Requires-Dist', []) if 'extra == "' + EXTRAS[track] + '"' in r]
            if requirements:
                plan.append({'track': track, 'cwd': str(root), 'argv': [python, '-m', 'pip', 'install', '--index-url', network['PIP_INDEX_URL'], *requirements]})
        if track in ('carla', 'realsim'):
            plan.append({'track': track, 'cwd': str(backend), 'argv': [python, '-m', 'pip', 'install', '--index-url', network['PIP_INDEX_URL'], '--no-deps', '-e', str(backend)]})
        if runtime:
            if track == 'carla':
                wheel = backend / 'carla_runtime/PythonAPI/carla/dist/carla-0.9.16-cp311-cp311-linux_x86_64.whl'
                runtime_requirements = ['dictor==0.1.12', 'ephem==4.1.5', 'networkx>=3', 'opencv-python>=4.7', 'psutil>=5', 'py-trees==0.8.3', 'pygame>=2', 'pyquaternion>=0.9', 'scipy>=1.10', 'six', 'tabulate', 'transforms3d', 'simple-watchdog-timer==0.1.1', 'pydot', 'pyparsing', 'requests', 'pexpect', 'ptyprocess', 'wcwidth', 'setuptools']
                plan.append({'track': track, 'cwd': str(backend), 'argv': [python, '-m', 'pip', 'install', '--index-url', network['PIP_INDEX_URL'], str(wheel), *runtime_requirements]})
            elif track == 'realsim':
                plan.append({'track': track, 'cwd': str(backend), 'argv': [python, str(backend / 'scripts/setup_runtime.py'), '--index-url', network['PIP_INDEX_URL'], '--install-rust']})
        if runtime and track == 'realsim':
            runtime_python = str(backend / '.runtime-env/venv/bin/python')
            runtime_uv = str(backend / '.runtime-env/bootstrap/bin/uv')
            # Native uv environments intentionally omit pip. Reuse their pinned
            # installer without resolving or changing frozen runtime dependencies.
            plan.append({'track': track, 'cwd': str(root), 'argv': [runtime_uv, '--no-config', 'pip', 'install', '--python', runtime_python, '--no-deps', package]})
        if download:
            hf = network['HF_ENDPOINT']
            # Never forward cached or environment credentials to an alternative host.
            if track in ('carla', 'nuscenes', 'realsim') and hf != 'https://huggingface.co':
                if track == 'carla' and runtime and not wheel.is_file():
                    plan = [step for step in plan if not (step['track'] == track and str(wheel) in step['argv'])]
                blocked.append({'track': track, 'reason': 'Benchmark resources currently require owner-granted HF access. Use --network official with authorized credentials; mirror downloads are anonymous only.'})
                continue
            if track == 'carla':
                argv = [python, str(backend / 'tools/download_resources.py'), '--groups', 'benchmark', 'carla', '--endpoint', hf]
            elif track == 'nuscenes':
                argv = [python, str(backend / 'scripts/download_assets.py'), '--endpoint', hf]
            elif track == 'navsim':
                argv = [python, str(backend / 'scripts/download_assets.py'), '--sensors', '--cameras-only']
            elif track == 'realsim':
                argv = [python, str(backend / 'scripts/hub_assets.py'), 'download', '--repo', 'Mike087/Language2Drive', '--directory', str(backend / 'assets'), '--endpoint', hf]
            elif source_root:
                argv = [python, str(backend / 'scripts/download_assets.py'), '--source-root', str(Path(source_root).resolve())]
            else:
                blocked.append({'track': track, 'reason': 'WOD-E2E requires licensed Waymo source files; supply --source-root. No model weights are downloaded by default.'})
                continue
            step = {'track': track, 'cwd': str(backend), 'argv': argv}
            if track == 'carla' and runtime:
                at = next(i for i, v in enumerate(plan) if v['track'] == track and any('carla-0.9.16' in a for a in v['argv']))
                plan.insert(at, step)
            else:
                plan.append(step)
            if track == 'realsim' and runtime:
                plan.append({'track': track, 'cwd': str(backend), 'argv': [python, str(backend / 'scripts/hub_assets.py'), 'scenes', '--bundle', str(backend / 'assets/langauge2drive.RealSim'), '--directory', str(backend / 'scenes'), '--endpoint', hf]})
                plan.append({'track': track, 'cwd': str(backend), 'argv': ['docker', 'pull', 'nvcr.io/nvidia/nre/nre-ga:26.04']})
    result = {'sources': network, 'transport': transport, 'commands': plan, 'resources_pending': blocked}
    if not dry_run:
        for step in plan:
            subprocess.run(step['argv'], cwd=step['cwd'], env=env, check=True)
        state = root / '.l2d/setup.json'
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps(result, indent=2))
    return result
