# Full GPU evaluation: three runtime groups, five backends

`carla`, `realsim` and `openloop` are the three runtime groups. The last group
expands to `nuscenes,navsim,wod`. All groups use the same adapter and retain their
native scoring protocol. A model may run GPU inference in its own Python environment.

## Install and provision once

```sh
pip install /path/to/language2drive-0.1.0-py3-none-any.whl
language2drive init --workspace ./l2d-workspace
language2drive setup --workspace ./l2d-workspace --tracks all --runtime --download
```

The installed wheel is reused automatically in isolated environments; no second
download from PyPI or `--package` is necessary. Extras `[carla]`, `[realsim]` and
`[openloop]` select Python dependencies. `setup --runtime --download` additionally
installs simulator environments, downloads accessible CARLA resources and the
RealSim bundle/scenes, and pulls the pinned NRE image. `--network auto` selects
official or mirror sources. Restricted HF resources require authorized access on
the official endpoint; anonymous mirrors cannot substitute for that access.

RealSim mirror provisioning exports the frozen lock and downloads registry
packages with their exact versions, environment markers and SHA256 hashes.
The original lock is preserved; native frozen synchronization still owns Git,
direct-URL and workspace dependencies. No dependency versions are re-resolved.

CARLA needs Conda on PATH. RealSim uses its native pinned Python 3.12 / AlpaSim
bootstrap, NVIDIA Container Toolkit and Docker. Host drivers are installed by the
host administrator. Supply Waymo source data with `setup --source-root ...`.
nuScenes test sensors and NAVSIM evaluator sidecars require their own authorized
resource paths. Official NRE ego-hood files must also be available; the runtime
keeps the frozen ego-mask configuration.

## Configure once

```sh
language2drive configure --workspace ./l2d-workspace --gpu 0 \
  --nuscenes-root /data/nuscenes \
  --navsim-root /data/openscene/sensor_blobs/test \
  --navsim-sidecars /data/l2d-navsim-sidecars \
  --wod-release /data/l2d-wod-release \
  --ego-hood-dir /data/nre-artifacts/ego-hoods \
  --output ./evaluation.json

language2drive gpu-check --config ./evaluation.json --tracks all
```

Bundle and scene paths default to the SDK download layout. Override them with
`--realsim-bundle` and `--realsim-scenes` to reuse resources. `gpu-check` checks
the host GPU and configured dependencies; it does not report experimental success.
Record immutable checkpoint/adapter revisions under `model` in the configuration.

## CARLA release deployment

The existing official CARLA launcher requires **root authority**, a non-root
`sim` account, root-owned launch evidence, and a simulator binary tree that `sim`
cannot replace. The unified SDK retains these requirements. The stock local
CARLA 0.9.15 installation cannot replace the pinned Language2Drive 0.9.16 runtime.

After deploying the downloaded runtime according to the [native guide](../language2drive.carla/README.md),
the deployment administrator first starts the bootstrap server:

```sh
# Deployment administrator, on the GPU host; use the installed CARLA Python.
export L2D_PYTHON=/absolute/workspace/.l2d/envs/carla/bin/python
bash language2drive.carla/tools/start_external_carla.sh --bootstrap --port 2100 --gpu-rank 0
# In another terminal, use the CARLA evaluator environment to freeze live assets.
l2d freeze-runtime-assets --host 127.0.0.1 --port 2100
```

Stop that bootstrap process after freezing. The `run` command can start the
official release server when run under the existing deployment authority, with
`tracks.carla.managed_runtime=true`. Alternatively, the administrator starts the
official server separately; set `managed_runtime=false` and provide its host/port.
Choose a fresh native release runtime directory/port for a new deployment;
the launcher deliberately refuses previous non-empty evidence directories.
Do not run an untrusted model adapter as root: use an already-running model
service and `language2drive evaluate` in the approved evaluator deployment.

## Launch and score

```sh
# One adapter, one model environment, all three groups / five backends.
language2drive run --config ./evaluation.json \
  --adapter /path/to/my_adapter.py --model-config /path/to/model.json \
  --model-python /path/to/model-env/bin/python \
  --tracks carla,realsim,openloop --output outputs/my-model

# Inspect commands before starting any process.
language2drive run --config ./evaluation.json \
  --adapter /path/to/my_adapter.py --tracks all --dry-run
```

For a model service started independently, `evaluate` uses the configured
endpoint; it does not manage CARLA. Managed RealSim deployment is selected by
the configuration and works in both `run` and `evaluate`.

RealSim starts the pinned renderer, physics, controller, SDK driver and native
runtime for **each frozen execution group**, sequentially. It retains the exact
30-PCU / 60-slot seed-1701 inventory and validates full group coverage. The driver
is registered before runtime startup so its identity is retained in real ASLs.
Only the exported instruction schedule enters the driver; evaluator contracts
are kept for scoring. Group configs preserve the frozen scene/timing/geometry
settings. Logs and launch arguments are saved under `runtime-groups/`.

Ports bind to loopback. Occupied ports fail without killing their owners.
Cleanup signals only subprocesses created by this campaign and removes only
its uniquely named renderer container. Existing training processes are untouched.

## Evidence and validation

Inspect `outputs/my-model/summary.json`, native score files, inference inventories
and real ASL receipts. Infrastructure failures remain failures; a synthetic
adapter test, successful installation or CUDA tensor operation is not a full
benchmark result. NAVSIM custom runs use the corrected scorer and are kept
separate from the historical manuscript profile.

```sh
python scripts/validate_gpu_campaign.py --output outputs/my-model
```

This validator requires all five tracks to be scored, the complete configured
open-loop inventory with no failed requests, and all 60 RealSim runtime outcomes.
It reports completeness only after real artifacts exist; it does not generate
predictions or grant a leaderboard submission approval.
