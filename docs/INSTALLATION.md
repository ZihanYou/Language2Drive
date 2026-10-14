# Installation and resource preparation

Install the coordinator in your model environment with `pip install -e .` or a release wheel. Select Python evaluator dependencies with `[openloop]`, `[carla]` or `[realsim]`; these extras intentionally do not impose model frameworks or GPU runtime versions.

From a checkout:

```sh
python -m pip install -e '.[openloop]'
sh scripts/bootstrap.sh --tracks nuscenes,navsim,wod --network auto
language2drive setup --tracks all --runtime --download --dry-run
```

From the published wheel:

```sh
python -m pip install https://github.com/ZihanYou/Language2Drive/releases/download/v0.1.0/language2drive-0.1.0-py3-none-any.whl
language2drive init --workspace /path/to/l2d-workspace
language2drive setup --workspace /path/to/l2d-workspace --tracks navsim
```

`init` verifies all bundled source hashes and refuses differing destination files. Original track paths and notices are preserved. The installed wheel is automatically reconstructed from its verified distribution files and reused for isolated environments; `--package` can explicitly override it. The wheel contains evaluator source; not checkpoint weights, simulator binaries, full datasets or model-specific vendor source.

## Source selection

`setup --network auto` probes official Hugging Face and PyPI independently with finite timeouts. It chooses `https://huggingface.co` or `https://hf-mirror.com`, and `https://pypi.org/simple` or `https://pypi.tuna.tsinghua.edu.cn/simple`. Use `--network official` / `mirror` to override. Mirrors do not bypass access restrictions. Current gated benchmark resources require the official HF endpoint and owner-granted credentials; the SDK reports those downloads as pending when a mirror is selected. NAVSIM sensor downloads retain the native ModelScope source.

Downloaders preserve their existing checksum and manifest validation. Resources are separate from model checkpoints; setup does not download every baseline checkpoint. A complete installation receipt is in `.l2d/setup.json`; inspect `resources_pending` and resource-specific output. Installation success does not itself establish dataset readiness.

## Track environments

| Selection | Environment | Additional resources |
|---|---|---|
| nuScenes / NAVSIM / WOD | `.l2d/envs/TRACK`, current Python 3.10+ | Official sensor images; frozen evaluator release/sidecars |
| CARLA | `.l2d/envs/carla`, standalone **Conda Python 3.11** prefix | CARLA-L2D runtime wheel/maps/resources, simulator process |
| RealSim CPU | `.l2d/envs/realsim`, Python 3.10+ | Frozen PCUs and traces for scoring |
| RealSim live | `language2drive.RealSim/.runtime-env/venv`, managed **Python 3.12** | Pinned AlpaSim, NuRec renderer, GPU/Docker and scene assets |

`--runtime` opts into simulator Python provisioning. CARLA downloads its runtime group before installing the bundled cp311 wheel when combined with `--download`; otherwise download it beforehand. RealSim invokes the existing locked runtime bootstrap with the selected PyPI index, then installs the SDK plugin into that environment. It preserves upstream package locks. OS drivers, renderer container access, Docker service permissions and external server configuration follow the native guides.

Full live RealSim bootstrapping can obtain its own Python 3.12 through uv; the SDK's preceding lightweight scoring environment does not require a separately preinstalled Python 3.12. CARLA's native evaluator requires a standalone Conda prefix; install Conda on PATH first. Setup creates its Python 3.11 environment automatically.

## Resources and current access

- CARLA: `Mike087/Language2Drive` benchmark/runtime manifests and fixed revision; substantial disk space is needed. Read [the native CARLA guide](../language2drive.carla/README.md).
- RealSim: authorized HF bundle and scenes. Its historical dataset prefix is `langauge2drive.RealSim` (spelling retained by native downloader). `setup --runtime --download` retrieves the bundle, its scene assets and the pinned renderer image after obtaining NVIDIA dataset access.
- nuScenes: native pinned replay bundle is downloaded by setup when authorized; source sensor images require official nuScenes test access and `sensor_root`. The replay bundle can reproduce frozen predictions without re-running model inference.
- NAVSIM: `--sensors --cameras-only` retrieves camera shards. Evaluator sidecars are a separate frozen resource; the toy sidecars are only for interface/scorer tests.
- WOD-E2E: provide `--source-root /path/to/licensed/Waymo` to native preparation. The SDK cannot obtain licensed raw validation assets for you.

## RealSim unified driver

The new Hydra template is `language2drive.RealSim/src/language2drive_realsim/configs/driver/l2d_unified.yaml`. Install both the RealSim integration and SDK into the pinned runtime environment. Set:

```sh
export L2D_MODEL_ENDPOINT=http://127.0.0.1:8090
export L2D_REALSIM_SIDECAR=/absolute/path/to/sidecar.private.json
```

From the RealSim track directory, launch the standalone configuration with the pinned Hydra entry point:

```sh
.runtime-env/venv/bin/l2d-alpasim-driver \
  --config-path "$PWD/src/language2drive_realsim/configs/driver" \
  --config-name l2d_unified port=50052 output_dir=/absolute/path/to/driver-output
```

The template disables waypoint commands, requests 10 Hz outputs and decodes cameras on CPU; your model remains in its own inference environment. It has no wizard configuration dependency. The native deployment guide explains the renderer/runtime launcher and Hydra config registration. Retain the `l2d_unified-driver-...` driver version authority in the ASL. The SDK checks this identity; unversioned drivers fail rather than earning scores.

In `examples/evaluation.json`, configure the runtime and driver addresses, ASL output root, evaluator YAML, USDZ glob, frozen public workload, private sidecar and frozen 30-PCU source. These are evaluator-side paths and never sent to `predict(request)`. Start the model service, CARLA server and AlpaSim stack, then `language2drive evaluate --tracks all --config ...` submits and scores the campaigns.

See [RealSim runtime setup](../language2drive.RealSim/docs/RUNTIME_SETUP.md) and [deployment](../language2drive.RealSim/docs/DEPLOYMENT.md) for host prerequisites. The simulator deployment remains explicit because it depends on the machine's GPU, containers and authorized resources.

See [the full GPU evaluation guide](GPU_EVALUATION.md) for `configure`, `gpu-check`, managed runtime groups and `run`.
