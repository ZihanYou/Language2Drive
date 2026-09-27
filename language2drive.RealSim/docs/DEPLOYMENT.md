# One-command deployment

## Prerequisites and default command

Use Linux or macOS with Python 3.10+ including `venv` and pip. Python 3.12 is
recommended. From the RealSim package directory:

```sh
sh scripts/setup.sh
```

The script performs these stages, stopping on any error:

1. Create or reuse this package's `.venv`; refuse a non-venv directory.
2. Install build tools and `.[test]` into that environment only.
3. Run the CPU regression suite (optional simulator integration is skipped if absent).
4. Score synthetic speed/lane/route examples into a new output directory.
5. Check evaluator imports and print the ready status and output locations.

No root privileges, credentials, GPU or model weights are needed for these steps.
Mirror downloads bypass inherited HTTP(S)/ALL proxy settings unless explicitly
requested. Existing experiments and previous demo outputs are not overwritten.

```sh
L2D_PYTHON=python3.12 sh scripts/setup.sh
sh scripts/setup.sh --venv /path/to/evaluator-env
sh scripts/setup.sh --index-url https://pypi.org/simple
sh scripts/setup.sh --keep-proxy
sh scripts/setup.sh --offline --wheelhouse /path/to/wheels
```

An offline wheelhouse must contain setuptools, wheel and their dependencies,
pytest, Shapely, NumPy and the dependencies of the test extra, built for the
target Python/platform. Offline mode never falls back to the network. The
`--skip-tests` option is intended only for a previously verified environment.

## Score an experiment

```sh
sh scripts/score.sh --pcus /path/to/frozen_pcus.json \
  --experiment /path/to/experiment.json --output scores.json
```

`scores.json` contains paper-scale `metrics`, `summary_components`, paired
identity checks, coverage and per-slot fractions. Trace paths are relative to
the experiment JSON or absolute paths. See [DATA_FORMAT](DATA_FORMAT.md).
To use a non-default environment, set `L2D_SCORE_PYTHON=/path/to/env/bin/python`.

## Closed-loop inference prerequisites

The CPU evaluator and the simulator/model environments are different. Obtain
the authorized assets and pinned simulator described in [RUNTIME_SETUP](RUNTIME_SETUP.md).
Install this integration package into that simulator environment, then check:

```sh
export L2D_PCUS_PATH=/path/to/frozen_pcus.json
export L2D_SCENE_ROOT=/path/to/scene-assets
export L2D_CHECKPOINT_ROOT=/path/to/model-checkpoints
python scripts/doctor.py --require-runtime
```

This checks Docker daemon access, the NVIDIA driver, asset paths and AlpaSim
import availability. It returns a nonzero exit status if anything is missing.
It does not download gated assets, accept third-party terms, install host drivers
or certify a renderer/model service as healthy. Follow the pinned runtime's
service startup and this package's compile/run interfaces to execute actual
rollouts. A successful CPU demo is not a closed-loop model result.

## Troubleshooting

- `venv`/`ensurepip` missing: install the OS Python-venv package, or choose a Python
  interpreter with venv support using `L2D_PYTHON`.
- Mirror unavailable: select an alternative `--index-url`, or prepare a wheelhouse.
- Optional AlpaSim test skipped: expected for CPU-only installation; use the
  simulator environment for those integration tests.
- Output already exists: choose a new output filename; scoring never replaces results.
- Incomplete record inventory: provide all 60 scheduled records, marking failed
  or unavailable executions explicitly. Do not drop unsuccessful arms.
