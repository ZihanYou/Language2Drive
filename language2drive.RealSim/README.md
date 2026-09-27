# Language2Drive RealSim

This source package provides paired language-conditioned driving evaluation:
PCU contracts, source compilation, simulator interfaces, trajectory verification,
physical-safety scoring, continuous paired semantic metrics, and supplemental
instruction-control diagnostics. No author identities or private service
endpoints are required to run the offline examples.

## One-command deployment (CPU evaluator)

```sh
sh scripts/setup.sh
```

This creates `.venv`, installs the package and test dependencies, runs regression
tests and a three-task synthetic demo, and checks the scoring interface. It uses
the Tsinghua PyPI mirror without inherited proxies by default. No sudo, Docker
installation, system Python changes or model downloads are performed. Re-running
is supported; outputs are written to fresh `artifacts/deployment-*` directories.

Use `L2D_PYTHON=python3.12 sh scripts/setup.sh` to select Python.
For air-gapped installation, use `sh scripts/setup.sh --offline --wheelhouse /path/to/wheels`.
Use `--index-url https://pypi.org/simple` to select another index, or explicitly
opt into the configured proxy with `--keep-proxy`. See [deployment guide](docs/DEPLOYMENT.md).

## Manual quick start (CPU, no model downloads)

Use Python 3.10 or newer; Python 3.12 is the recommended review environment.
From this directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
l2d-review --output demo.json demo
```

An installation-free alternative, if pytest is already available:

```sh
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m language2drive_realsim.review_cli demo
```

The demo constructs **synthetic** speed, lane and route examples. It tests the
software interface; its numbers are not experimental results from any model.
Scalars returned by `score_pair` are fractions; cohort reports include percent
fields. Always check the metric field name and scale before making a table.

The test extra installs pytest and Shapely (including its NumPy dependency).
The optional renderer protobuf integration tests are explicitly skipped when
the separately installed AlpaSim protobuf package is unavailable.

## Scoring experiment records

```sh
sh scripts/score.sh --output scores.json \
  --pcus path/to/frozen_pcus.json --experiment path/to/experiment.json
l2d-review --output diagnoses.json diagnose \
  --pcus path/to/frozen_pcus.json --experiment path/to/experiment.json
```

The `realsim-paper-1.0` scoring profile uses 30 PCUs, both arms, and seed 1701 (60 slots). Trace paths
are resolved relative to the experiment JSON. See [data contracts](docs/DATA_FORMAT.md)
and [metric definitions](docs/METRICS.md). Existing output files are not overwritten.
The low-level authoring/verifier library also supports other contract families
and legacy profiles; those are not silently substituted for the 30-PCU profile.

## Package contents

| Directory | Contents |
|---|---|
| `src/language2drive_realsim` | Compiler, validators, runtime bridge, model interfaces and scorers |
| `config` | Thresholds, upstream runtime pin and compatibility overlay |
| `tests` | CPU regression tests using synthetic fixtures |
| `docs` | Setup, data contracts, scoring and review-package instructions |

Full closed-loop inference additionally needs an authorized AlpaSim/NuRec
installation, scene assets, frozen PCU annotations and separately obtained model
checkpoints. These are not embedded in this **source-code** artifact. Neither are
experiment logs, score tables, machine environments or third-party source trees.
See [runtime setup](docs/RUNTIME_SETUP.md) for the integration boundary.

The paper-aligned entry point returns SISR, CFPS, SemISR, Sem-CFPS, SCS, PCS,
Safety, Progress, DrivingScore and L2D as numeric values on a 0–100 scale. It
retains unsuccessful scheduled executions in the denominator and computes L2D
from unrounded aggregate metrics. [Paper alignment](docs/PAPER_ALIGNMENT.md)
maps the implementation to the equations and specifies the aggregation order.
Safety in the physical-only profile means collision/off-road safety; red-light,
speed-limit and wrong-way violations are outside that profile.

中文说明：[部署及使用说明](docs/REVIEW_GUIDE_ZH.md)。
The original project code follows the parent repository's MIT license.
External dependencies retain their respective terms; see [NOTICE](NOTICE.md).
