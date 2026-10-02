"""Coordinate isolated evaluator environments without changing legacy CLIs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .workspace import TRACKS, discover


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    workspace = Path(config.get('workspace', '.')).expanduser()
    if not workspace.is_absolute():
        workspace = path.parent / workspace
    root = discover(workspace)
    config['workspace'] = str(root)
    path_keys = {'python', 'sensor_root', 'release_dir', 'public_dir', 'sidecars', 'contracts_dir', 'public_workload', 'sidecar', 'pcus', 'asl_root', 'eval_config', 'usdz_glob', 'bundle', 'scene_root', 'ego_hood_dir', 'backend'}
    for cfg in config.get('tracks', {}).values():
        for key in path_keys.intersection(cfg):
            value = Path(cfg[key]).expanduser()
            if not value.is_absolute():
                cfg[key] = str(root / value)
    return config


def environment(root, track):
    backend = root / TRACKS[track]
    env = os.environ.copy()
    sdk_src = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = os.pathsep.join([sdk_src, str(backend / 'src'), str(backend), str(backend / 'language2drive/leaderboard'), str(backend / 'language2drive/scenario_runner'), env.get('PYTHONPATH', '')])
    return env


def commands(track, backend, cfg, output, endpoint):
    """Return the native scorer argv; no shell string interpolation."""
    if track == 'nuscenes':
        return [[str(backend / 'evaluate_predictions.py'), '--release-dir', str(cfg.get('release_dir', backend / 'releases/nuscenes-test-v1.0.0')),
                 '--predictions', str(output / 'predictions.jsonl'), '--output-dir', str(output / 'scores'), '--split', cfg.get('split', 'heldout'),
                 '--conditions', ','.join(cfg.get('conditions', ['canonical'])), '--wordings', ','.join(cfg.get('conditions', ['canonical'])), '--seed', str(cfg.get('seed', 0))]]
    if track == 'navsim':
        if cfg.get('scorer', 'corrected') != 'corrected':
            raise ValueError('Unified custom evaluations use corrected NAVSIM scoring; reproduce historical results with native script')
        return [[str(backend / 'navsim_offline/score_corrected.py'), '--sidecars', str(cfg['sidecars']), '--predictions', str(output / 'canonical.json'),
                 '--rewrite-predictions', str(output / 'rewrites.json'), '--output', str(output / 'score.json')]]
    if track == 'wod':
        return [[str(backend / 'scripts/evaluate_paper.py'), '--release-dir', str(cfg.get('release_dir', backend / 'prerelease/v0.1.0')),
                 '--contracts-dir', str(cfg.get('contracts_dir', backend / 'contracts/wod-e2e-p0-v0.1.0')),
                 '--model', cfg.get('model_name', 'custom') + '=' + str(output / 'predictions.jsonl'), '--output-dir', str(output / 'scores')]]
    if track == 'carla':
        bridge = Path(__file__).parent / 'bridges/carla.py'
        result = []
        for split in cfg.get('splits', ['dev', 'test']):
            target = output / split
            plan = target / 'run_plan.json'
            result.extend([['-m', 'language2drive.cli', 'run', '--agent-path', str(bridge), '--agent-config', str(output / 'agent.json'),
                           '--track', 'standardized', '--tier', cfg.get('tier', 'base'), '--split', split, '--seeds', str(cfg.get('seeds', '1')),
                           '--results', str(target / 'results'), '--plan-output', str(plan), '--execute', '--host', cfg.get('host', '127.0.0.1'), '--port', str(cfg.get('port', 2000)),
                           '--gpu-rank', str(cfg.get('gpu', 0)), '--traffic-manager-port', str(cfg.get('traffic_manager_port', 8000))],
                           ['-m', 'language2drive.cli', 'score', '--plan', str(plan), '--results', str(target / 'results'), '--output', str(target / 'score.json')]])
        return result
    return []


def run(config_path, selected, output, dry_run=False):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = discover(config.get('workspace'))
    output = Path(output).resolve()
    endpoint = config.get('endpoint', 'http://127.0.0.1:8090')
    if not dry_run:
        from .service import Client
        from . import __version__
        binding = {'config': config, 'service': Client(endpoint).identity(), 'sdk_version': __version__}
        path = output / 'campaign-binding.json'
        if path.exists() and json.loads(path.read_text()) != binding:
            raise ValueError('Campaign output belongs to a different model/configuration; choose a fresh directory')
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(binding, indent=2))
    report = {'model': config.get('model', {}), 'tracks': {}, 'cross_track_aggregate': None}
    for track in selected:
        cfg = config.get('tracks', {}).get(track)
        if cfg is None:
            raise ValueError(f'Missing tracks.{track} in {config_path}')
        backend = root / TRACKS[track]
        default_python = root / TRACKS[track] / '.runtime-env/venv/bin/python' if track == 'realsim' and cfg.get('managed_runtime') else root / '.l2d/envs' / track / 'bin/python'
        python = str(Path(cfg.get('python', default_python)).expanduser())
        target = output / track
        scorer = commands(track, backend, cfg, target, endpoint)
        worker = [python, '-m', 'l2d_benchmark.cli', '_worker', '--config', str(config_path), '--track', track, '--output', str(target)]
        all_commands = ([worker] if track != 'carla' else []) + [[python, *cmd] for cmd in scorer]
        if dry_run:
            report['tracks'][track] = {'commands': all_commands, 'cwd': str(backend)}
            continue
        target.mkdir(parents=True, exist_ok=True)
        if track == 'carla':
            (target / 'agent.json').write_text(json.dumps({'endpoint': endpoint, **cfg.get('controller', {})}, indent=2))
        try:
            for index, cmd in enumerate(all_commands):
                with (target / f'command-{index}.log').open('w') as log:
                    subprocess.run(cmd, cwd=backend, env=environment(root, track), stdout=log, stderr=subprocess.STDOUT, check=True)
            report['tracks'][track] = {'status': 'scored', 'artifacts': str(target), 'protocol': cfg.get('scorer', 'native'), 'score_files': [str(p) for p in target.rglob('*.json') if p.name in ('score.json', 'summary.json') or 'report' in p.name]}
        except (subprocess.CalledProcessError, OSError) as error:
            report['tracks'][track] = {'status': 'failed', 'error': str(error), 'artifacts': str(target)}
        output.mkdir(parents=True, exist_ok=True)
        (output / 'summary.json').write_text(json.dumps(report, indent=2))
    return report


def worker(config_path, track, output):
    config = load_config(config_path)
    root = discover(config.get('workspace'))
    cfg = config['tracks'][track]
    if track == 'realsim':
        from .realsim_campaign import campaign
        cfg = dict(cfg, endpoint=config.get('endpoint', 'http://127.0.0.1:8090'), backend=str(root / TRACKS[track]))
        cfg.setdefault('python', str(root / TRACKS[track] / '.runtime-env/venv/bin/python'))
        return campaign(cfg, output)
    from .runners import run_openloop
    return run_openloop(track, root / TRACKS[track], cfg, output, config.get('endpoint', 'http://127.0.0.1:8090'), config.get('model', {}))
