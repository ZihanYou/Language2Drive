"""Deploy each frozen RealSim execution group, retaining the 60-slot inventory."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import uuid
from .processes import Services


def frozen_groups(bundle, plan):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    from language2drive_realsim.alpasim_bridge import ALPASIM_COMMIT
    if manifest.get('alpasim_commit') != ALPASIM_COMMIT or manifest.get('scheduled_slots') != 60 or manifest.get('seed') != 1701:
        raise ValueError('Expected the pinned 60-slot seed-1701 portable RealSim bundle')
    groups = json.loads((bundle / 'runtime/groups.json').read_text())
    covered, group_ids = set(), set()
    expected = {r['pcu_id'] for r in plan['runs']}
    for group in groups:
        ids = set(group['pcu_ids'])
        if not ids or len(ids) != len(group['pcu_ids']) or covered.intersection(ids):
            raise ValueError('Missing or repeated PCUs in frozen execution groups')
        if group['id'] in group_ids or Path(group['id']).name != group['id']:
            raise ValueError('Invalid execution-group identity')
        group_ids.add(group['id']); covered.update(ids)
    if covered != expected or len(expected) != 30:
        raise ValueError('Frozen execution groups must cover exactly the scheduled 30 PCUs')
    return groups


def execute(cfg, plan, output):
    from language2drive_realsim.runtime_client import submit_run_plan
    bundle = Path(cfg['bundle'])
    groups = frozen_groups(bundle, plan)
    scenes = Path(cfg['scene_root']).resolve()
    hoods = Path(cfg['ego_hood_dir']).resolve()
    assets = json.loads((bundle / 'assets.json').read_text())
    for name in assets['dataset']['files']:
        if not (scenes / Path(name).name).is_file():
            raise FileNotFoundError('Missing frozen NuRec scene: ' + Path(name).name)
    if not hoods.is_dir():
        raise FileNotFoundError('Provide the official ego-hood assets; ego masking is retained')
    python = cfg['python']
    # SDK extracted workspace has the native Hydra template, with its existing defaults.
    backend = bundle.parents[1] if bundle.parent.name == 'assets' else Path(cfg['backend'])
    driver_template = backend / 'src/language2drive_realsim/configs/driver'
    base = int(cfg.get('service_base_port', 6005))
    if base < 1024 or base + 5 > 65535:
        raise ValueError('service_base_port must leave six unprivileged ports')
    ports = dict(renderer=base, physics=base+1, controller=base+2, driver=base+3, runtime=base+4, metrics=base+5)
    env = os.environ.copy()
    env.update(L2D_MODEL_ENDPOINT=cfg['endpoint'], L2D_REALSIM_SIDECAR=str(bundle / 'compiled/language.json'),
               CUDA_VISIBLE_DEVICES=str(cfg.get('gpu', 0)), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    returned = None
    for group in groups:
        target = Path(output) / 'runtime-groups' / group['id']
        target.mkdir(parents=True, exist_ok=False)
        user = deepcopy(group['user_config'])
        user['scene_provider']['usdz']['data_dir'] = str(scenes)
        user['simulation_config']['force_gt_frame_cache']['cache_dir'] = str(target / 'frame-cache')
        user['nr_workers'] = 1
        user['endpoints']['driver']['skip'] = False
        for name in ('renderer', 'physics', 'controller'):
            if user['endpoints'][name]['skip']:
                raise ValueError('Frozen closed-loop configuration cannot skip ' + name)
        user['prometheus']['url'] = None
        user['prometheus']['worker_ports'] = [ports['metrics']]
        network = {name: {'endpoints': [{'address': '127.0.0.1:' + str(ports[name]), 'managed': False}]} for name in ('renderer', 'physics', 'controller', 'driver')}
        network['trafficsim'] = {'endpoints': []}
        if not user['endpoints']['trafficsim']['skip']:
            raise ValueError('This bundle requires a traffic service; provide an external native deployment instead')
        for filename, value in [('user.json', user), ('network.json', network),
                                ('eval.json', json.loads((bundle / 'runtime/eval.json').read_text())),
                                ('controller.json', json.loads((bundle / 'runtime/controller.json').read_text()))]:
            (target / filename).write_text(json.dumps(value, indent=2))
        subset = dict(plan, runs=[r for r in plan['runs'] if r['pcu_id'] in group['pcu_ids']])
        subset['run_count'] = len(subset['runs'])
        (target / 'run_plan.json').write_text(json.dumps(subset, indent=2))
        container = 'l2d-nre-' + uuid.uuid4().hex[:16]
        commands = {
            'renderer': ['docker', 'run', '--rm', '--name', container, '--gpus', 'device=' + str(cfg.get('gpu', 0)),
                         '-p', f"127.0.0.1:{ports['renderer']}:6005", '-v', str(scenes) + ':/mnt/nre-data:ro',
                         '-v', str(hoods) + ':/mnt/ego-hoods:ro', '-e', 'HOME=/tmp',
                         '--entrypoint', '/app/internal/scripts/pycena/runtime/pycena_nrm_full', assets['renderer_image'],
                         'serve-grpc', '--host=0.0.0.0', '--port=6005', '--artifact-glob=/mnt/nre-data/*.usdz',
                         '--egocar-hood-dir=/mnt/ego-hoods', '--renderer=default', '--download-cache-dir=/tmp/nre-cache-dir',
                         '--cache-size=2', '--max-workers=1', '--enable-editing-actors'],
            'physics': [python, '-m', 'alpasim_physics.server', '--host', '127.0.0.1', '--port', str(ports['physics']), '--artifact-glob', str(scenes / '*.usdz'), '--use-ground-mesh=true', '--cache-size', '2'],
            'controller': [python, '-m', 'alpasim_controller.server', '--host', '127.0.0.1', '--port', str(ports['controller']), '--config', str(target / 'controller.json'), '--log_dir', str(target)],
            'driver': [str(Path(python).parent / 'l2d-alpasim-driver'), '--config-path', str(driver_template), '--config-name', 'l2d_unified', 'port=' + str(ports['driver']), 'output_dir=' + str(target / 'driver')],
            'runtime': [python, '-m', 'language2drive_realsim.host_runtime', '--user-config', str(target / 'user.json'), '--network-config', str(target / 'network.json'), '--eval-config', str(target / 'eval.json'), '--log-dir', str(target), '--serve', '--listen-address', '127.0.0.1:' + str(ports['runtime'])]}
        (target / 'launch.json').write_text(json.dumps(dict(commands=commands, gpu=cfg.get('gpu', 0), driver_binding='registered_static'), indent=2))
        try:
            with Services(target / 'services') as services:
                for name, command in commands.items():
                    services.start(name, command, backend, env, ports[name], int(cfg.get('startup_timeout_s', 300)))
                result = submit_run_plan(subset, '127.0.0.1:' + str(ports['runtime']), '127.0.0.1:' + str(ports['driver']), 1, use_registered_driver=True)
                (target / 'runtime_return.json').write_text(json.dumps(result, indent=2, allow_nan=False))
                if returned is None:
                    returned = dict(result, rollout_returns=[])
                for key in ('runtime_git_hash', 'driver_version', 'nre_version'):
                    if returned[key] != result[key]:
                        raise ValueError('Runtime authority changed between execution groups: ' + key)
                returned['rollout_returns'].extend(result['rollout_returns'])
        finally:
            # Only remove this campaign's uniquely named renderer, including failed starts.
            subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=30)
    ids = [r['session_uuid'] for r in returned['rollout_returns']]
    if len(ids) != 60 or set(ids) != {r['session_uuid'] for r in plan['runs']}:
        raise ValueError('Runtime returned an incomplete or duplicated 60-slot inventory')
    return returned
