# Language2Drive: NAVSIM open-loop track

This directory contains the `l2d-offline-navsim` 1.0.0 test inputs, an offline
semantic evaluator, and adapters for five released driving models. It accompanies
the NAVSIM experiment in *Language2Drive: Do Vision-Language-Action Models Really
Follow the Instructions in Autonomous Driving?* The setup uses fixed observations
from OpenScene/NAVSIM test logs and compares trajectories generated for paired
English instructions.

## Unified SDK / GPU inference

The refreshed release also supports the root project's one-adapter SDK. The
current manuscript baselines are **training-free**: existing checkpoints are
evaluated without Language2Drive training or fine-tuning. The official versioned
dataset and official model are planned releases on
[Hugging Face](https://huggingface.co/datasets/Mike087/Language2Drive/tree/main).

```sh
# From the Language2Drive workspace after installing its wheel or checkout.
language2drive setup --tracks navsim --download
language2drive configure --workspace . \
  --navsim-root /data/openscene/sensor_blobs/test \
  --navsim-sidecars /data/l2d-navsim-sidecars --output .l2d/evaluation.json
language2drive run --config .l2d/evaluation.json --tracks navsim \
  --adapter /path/to/my_adapter.py --model-python /path/to/model-env/bin/python \
  --output outputs/my-model
```

The adapter runs in your GPU model environment. NAVSIM still exports eight
rear-axle XY poses and evaluates **38 PCUs / 304 requests** with the corrected
scorer. `--tracks openloop` includes nuScenes and WOD-E2E; `--tracks all` also
includes CARLA and RealSim. [GPU setup and full evaluation](../docs/GPU_EVALUATION.md)
documents resources and native runtime requirements. Original baseline scripts
and the historical manuscript scorer below remain available.

## Test set

The test set has 38 paired counterfactual units (PCUs) from 19 recordings.
Each PCU has two instructions and three English paraphrases per instruction:
304 inference queries per model. The four tiers contain 14 atomic (A), 9
persistent (B), 11 scene-grounded (C), and 4 ordered or longer-horizon (D) PCUs.
Families cover lane and junction choices, passing or following, speed control,
object-related driving, local goals, named-landmark stopping, and event order.
The manifest and JSON inputs are in [`data/public`](data/public); a compact copy
is [`data/public_test_set.tar.gz`](data/public_test_set.tar.gz).

The public inputs provide the observed ego history, relative camera and LiDAR
paths, and instruction text. They contain no contract labels or future trajectory.
Original OpenScene sensor files and third-party checkpoints are downloaded from
their maintainers; they are not vendored in this repository.

## Get the assets

From this directory, run:

```sh
sh scripts/download_assets.sh
python3 scripts/check_public_data.py --sensors
```

The first command downloads and resumes the 32 official test shards per sensor
modality from the ModelScope OpenScene mirror, pinned official model repositories,
and pinned Hugging Face checkpoints through a mirror. It creates a local virtual
environment for the downloader. To start with camera-only assets or one model,
use `python3 scripts/download_assets.py --sensors --cameras-only` or
`python3 scripts/download_assets.py --models qwen_drive qwen_base` after
installing `huggingface-hub==0.36.0`. Run `--dry-run` to inspect the plan.
The download resumes after interruption. Set `L2D_NAVSIM_SENSOR_ROOT` to an
existing `openscene-v1.1/sensor_blobs/test` directory to reuse local sensors.
See [`docs/ASSETS.md`](docs/ASSETS.md) for provenance and disk layout.

## Generate trajectories

Install the official model dependencies in separate Python environments and
place them at `model_eval/envs/{qwen,legacy,onevl}`. The frozen software versions
and model revisions are recorded in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). Then run:

```sh
sh scripts/run_baselines.sh qwen_drive
sh scripts/run_baselines.sh autovla
sh scripts/run_baselines.sh drivema
sh scripts/run_baselines.sh recogdrive
sh scripts/run_baselines.sh onevl
```

Set `L2D_QWEN_PYTHON`, `L2D_LEGACY_PYTHON`, and `L2D_ONEVL_PYTHON` to override
environment paths. Each adapter uses only public inputs and writes resumable
JSONL and canonical/rewrite submissions under `model_eval/runs/`. GPU inference
requires enough memory for the corresponding official model. Adapter details,
language insertion points, and output conversions are in the protocol document.

## Score submissions

Submissions contain eight rear-axle poses in the anchor ego frame, at 0.5 s
intervals to 4 s. The schema and example are in
[`data/public/SUBMISSION_FORMAT.md`](data/public/SUBMISSION_FORMAT.md).
For exact reproduction of the NAVSIM numbers printed in the manuscript, use
the original scoring environment. The runner creates an isolated virtual
environment without Shapely and keeps the original behavior in which geometry
contract import errors are absorbed during scoring:

```sh
sh scripts/run_paper_legacy_scoring.sh \
  --sidecars /path/to/private_sidecars \
  --predictions /path/to/canonical.json \
  --rewrite-predictions /path/to/rewrites.json \
  --output /path/to/paper-legacy-scores.json
```

This is a historical reproduction profile, not a valid assessment of lane
geometry. For geometry-aware scoring, install
`python3 -m pip install -r navsim_offline/requirements-scorer.txt` and run
`python3 navsim_offline/score_corrected.py` with the same arguments. Keep the
two profiles and their results distinct.

Contract sidecars are held by the benchmark evaluator to preserve the test set.
The scorer source is public so its contract logic and aggregation can be audited;
official test labels are not included in the public inputs. `--rewrite-predictions`
accepts an object keyed by `canonical`, `paraphrase_1`, `paraphrase_2`, and
`paraphrase_3`, each containing a normal submission. The scorer reports binary
SemSISR and SemCFPS, graded SCS and PCS, family and tier breakdowns, and a
recording-cluster bootstrap. Metric definitions are in
[`docs/METRIC_SPEC.md`](docs/METRIC_SPEC.md).

To exercise the complete scorer without official labels, run the synthetic
speed-band example:

```sh
python3 navsim_offline/score_corrected.py \
  --sidecars examples/toy_sidecars \
  --predictions examples/toy_predictions.json \
  --rewrite-predictions examples/toy_rewrites.json \
  --output /tmp/l2d-toy-scores.json
```

It should report SemSISR and SemCFPS of `1.0`, SCS and PCS of `100.0`, and an
identity residual of zero.

This is a four-second, fixed-observation planning test. Its semantic scores do
not imply closed-loop driving success or counterfactual responses by other actors.
Code in this directory follows the repository [MIT license](../LICENSE);
OpenScene and model assets retain their respective upstream licenses.
