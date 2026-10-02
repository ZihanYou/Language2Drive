"""Generate one evaluation configuration from the standard downloaded layout."""
import json
from pathlib import Path
from .workspace import TRACKS, discover


def configure(workspace, destination, gpu=0, **paths):
    root = discover(workspace)
    def chosen(name, default):
        return str(Path(paths.get(name) or default).expanduser().resolve())
    real = root / TRACKS['realsim']
    real_work = Path(chosen('realsim_work_dir', real / '.runtime-env'))
    bundle = Path(chosen('realsim_bundle', real / 'assets/langauge2drive.RealSim'))
    scenes = chosen('realsim_scenes', real / 'scenes')
    cfg = dict(workspace=str(root), endpoint='http://127.0.0.1:8090',
               model=dict(name='custom', seed=0), tracks={
        'carla': dict(managed_runtime=True, gpu=gpu, tier='base', splits=['dev', 'test'], seeds='1', host='127.0.0.1', port=2000),
        'realsim': dict(managed_runtime=True, gpu=gpu, python=str(real_work / 'venv/bin/python'),
                        bundle=str(bundle), scene_root=scenes, ego_hood_dir=chosen('ego_hood_dir', real / 'assets/ego-hoods'),
                        public_workload=str(bundle / 'compiled/public_workload.json'), sidecar=str(bundle / 'compiled/sidecar.json'),
                        pcus=str(bundle / 'pcus.json'), eval_config=str(bundle / 'runtime/eval.json'), usdz_glob=scenes+'/*.usdz', concurrency=1),
        'nuscenes': dict(sensor_root=chosen('nuscenes_root', root / TRACKS['nuscenes'] / 'assets/nuscenes'), split='heldout', conditions=['canonical'], seed=0),
        'navsim': dict(sensor_root=chosen('navsim_root', root / TRACKS['navsim'] / 'assets/openscene-v1.1/sensor_blobs/test'),
                       sidecars=chosen('navsim_sidecars', root / TRACKS['navsim'] / 'assets/private_sidecars'), scorer='corrected',
                       conditions=['canonical', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3']),
        'wod': dict(release_dir=chosen('wod_release', root / TRACKS['wod'] / 'prerelease/v0.1.0'),
                    contracts_dir=str(root / TRACKS['wod'] / 'contracts/wod-e2e-p0-v0.1.0'))})
    path = Path(destination).resolve()
    if path.exists():
        raise ValueError('Configuration already exists; preserve it or choose another --output')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2)+'\n')
    return dict(config=str(path), gpu=gpu, next='language2drive gpu-check --config ' + str(path))
