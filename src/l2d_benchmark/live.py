"""Host-side orchestration; native runtime and scoring authorities are preserved."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from .pipeline import environment, load_config, run
from .processes import Services
from .workspace import TRACKS


def check(config_path, selected):
    config = load_config(config_path)
    root = Path(config['workspace'])
    checks = []
    def record(name, valid, remedy):
        checks.append(dict(name=name, ready=bool(valid), remedy=None if valid else remedy))
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.free,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        record('NVIDIA driver', gpu.returncode == 0, gpu.stderr.strip() or 'Run on the GPU host, outside a sandbox that hides NVIDIA devices.')
        gpu_inventory = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired) as error:
        record('NVIDIA driver', False, str(error)); gpu_inventory = []
    for track in selected:
        cfg = config['tracks'][track]
        default = root / (TRACKS[track] + '/.runtime-env/venv/bin/python') if track == 'realsim' else root / '.l2d/envs' / track / 'bin/python'
        python = str(cfg.get('python', default))
        installed = Path(python).is_file()
        remedy = 'Run language2drive setup --tracks ' + track + (' --runtime' if track in ('carla', 'realsim') else '')
        record(track + ' interpreter', installed, remedy)
        if installed:
            modules = 'numpy, shapely, PIL'
            if track == 'carla':
                modules += ', carla, py_trees'
            elif track == 'realsim':
                modules += ', grpc, alpasim_runtime, alpasim_physics, alpasim_controller, language2drive_realsim'
            try:
                imported = subprocess.run([python, '-c', 'import ' + modules], capture_output=True,
                                          text=True, timeout=30, env=environment(root, track))
                ready = imported.returncode == 0
                detail = imported.stderr.strip()[-1200:]
            except (OSError, subprocess.TimeoutExpired) as error:
                ready, detail = False, str(error)
            record(track + ' runtime imports', ready, remedy + '. ' + detail)
        for key in ('sensor_root', 'sidecars', 'release_dir', 'contracts_dir', 'public_workload', 'sidecar', 'pcus', 'eval_config', 'bundle', 'scene_root', 'ego_hood_dir'):
            if key in cfg:
                record(track + ' ' + key, Path(cfg[key]).exists() and 'PATH/TO' not in cfg[key], 'Provide the authorized resource path in tracks.' + track + '.' + key)
        if track == 'carla':
            record('CARLA 0.9.16 binary', (root / TRACKS[track] / 'carla_runtime/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping').is_file(), 'Run setup --tracks carla --runtime --download')
            if cfg.get('managed_runtime'):
                record('CARLA release launcher authority', os.geteuid() == 0, 'The native release launcher requires a root-owned deployment and a non-root sim user. Use the native host guide or an already-running authorized server.')
                record('CARLA frozen runtime manifest', (root / TRACKS[track] / '.runtime/carla_runtime_assets.current.json').is_file(), 'Bootstrap the native server, then run l2d freeze-runtime-assets. See docs/GPU_EVALUATION.md.')
        if track == 'realsim' and cfg.get('managed_runtime'):
            try:
                result = subprocess.run(['docker', 'info', '--format', '{{json .Runtimes}}'], capture_output=True, text=True, timeout=15)
                valid = result.returncode == 0 and 'nvidia' in result.stdout
            except (OSError, subprocess.TimeoutExpired):
                valid = False
            record('RealSim Docker NVIDIA runtime', valid, 'Install NVIDIA Container Toolkit and authorize Docker access on the GPU host.')
    return dict(ready=all(row['ready'] for row in checks), checks=checks, gpu_inventory=gpu_inventory,
                gpu_experiments_executed=False)


@contextmanager
def carla_server(root, cfg, target):
    if not cfg.get('managed_runtime'):
        yield
        return
    if os.geteuid() != 0:
        raise RuntimeError('Native CARLA release launcher requires root; see docs/GPU_EVALUATION.md. An external server can use managed_runtime=false.')
    backend = root / TRACKS['carla']
    env = environment(root, 'carla')
    env['L2D_PYTHON'] = str(cfg.get('python', root / '.l2d/envs/carla/bin/python'))
    port = int(cfg.get('port', 2000))
    with Services(target / 'services') as services:
        services.start('carla', ['bash', str(backend / 'tools/start_external_carla.sh'),
                               '--port', str(port), '--gpu-rank', str(cfg.get('gpu', 0))],
                       backend, env, port, int(cfg.get('startup_timeout_s', 240)))
        yield


def launch(config_path, selected, output, adapter=None, model_config=None, model_python=None, dry_run=False):
    config = load_config(config_path)
    root, output = Path(config['workspace']), Path(output).resolve()
    model = config.get('model_service', {})
    adapter = adapter or model.get('adapter')
    if not adapter:
        raise ValueError('run requires --adapter or model_service.adapter; use evaluate for an external model service')
    python = model_python or model.get('python', sys.executable)
    parsed = urlparse(config.get('endpoint', 'http://127.0.0.1:8090'))
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('Managed model service requires an http://127.0.0.1:PORT endpoint')
    command = [str(python), '-m', 'l2d_benchmark.cli', 'serve', '--adapter', str(Path(adapter).resolve()), '--port', str(parsed.port or 8090)]
    model_config = model_config or model.get('config')
    if model_config:
        command.extend(['--config', str(Path(model_config).resolve())])
    if dry_run:
        plan = run(config_path, selected, output, True)
        plan['model_service'] = command
        plan['runtime_groups'] = {g: [t for t in ts if t in selected] for g, ts in
                                  {'carla': ['carla'], 'realsim': ['realsim'], 'openloop': ['nuscenes', 'navsim', 'wod']}.items()}
        return plan
    readiness = check(config_path, selected)
    if not readiness['ready']:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'preflight.json').write_text(json.dumps(readiness, indent=2))
        raise RuntimeError('GPU/resource preflight failed; see ' + str(output / 'preflight.json'))
    combined = dict(tracks={}, runtime_groups=['carla', 'realsim', 'openloop'], cross_track_aggregate=None)
    with Services(output / 'model-service') as services:
        services.start('model', command, root, environment(root, 'carla'), parsed.port or 8090)
        for track in selected:
            manager = carla_server(root, config['tracks'][track], output / track) if track == 'carla' else nullcontext()
            try:
                with manager:
                    report = run(config_path, [track], output)
                combined['tracks'].update(report['tracks'])
            except Exception as error:
                combined['tracks'][track] = dict(status='failed', error=str(error))
            (output / 'summary.json').write_text(json.dumps(combined, indent=2))
    return combined
