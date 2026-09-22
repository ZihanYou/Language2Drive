# Benchmark use and reproduction

All commands below run from `Language2Drive/language2drive.carla/`. The Git
repository contains the evaluation and adapter code. The Hugging Face dataset
`Mike087/Language2Drive`, under `language2drive.carla/`, contains the non-weight
resources. Model checkpoints are downloaded from their original publishers.

## 1. Evaluation definition

A paired counterfactual unit (PCU) contains two arms evaluated in the same
scene. Each arm has three English expressions. The source suite has 128 PCUs,
32 per layer, with 24 dev and 104 test PCUs.

| Layer | Task family |
|---|---|
| A | Atomic driving actions |
| B | Parameterized speed, lane and behavior constraints |
| C | Grounded semantic navigation, regions and parking targets |
| D | Multi-stage instructions with ordering, conditions and memory |

`config/benchmark_map_policy.json` defines the map set used by the execution
tiers. Small and Base are nested, balanced subsets; Large includes all 119
PCUs selected by that policy. Use the versioned ledgers without resampling.

| Tier | PCUs | Dev / seed | Test / seed | Both splits / seed |
|---|---:|---:|---:|---:|
| Small | 32 | 48 | 144 | 192 |
| Base | 64 | 84 | 300 | 384 |
| Large | 119 | 138 | 576 | 714 |

The default execution seed is 1. Specify `--seeds 1,2,3` for three seeds.
A full Base seed-1 workload consists of 384 core runs, 24 FDQ routes and six
Safety Priority cases, totaling **414 runs**. Report dev and test scopes
explicitly when using only one split.

- **Standardized**: public language context and declared sensors; evaluator
  contracts and private route targets remain evaluator inputs.
- **Native**: the native system interface, reported separately.
- **Oracle**: the privileged contract-aware agent for validating the suite.
- **FDQ**: driving quality on the frozen set of 24 CARLA scenarios.
- **Safety Priority**: six cases scored independently of A–D.

## 2. Environment and resources

Use Linux x86-64 with Python 3.11, an NVIDIA GPU, a compatible NVIDIA driver,
and Vulkan support. The base environment uses Torch 2.2.0 / CUDA 12.1 and
CARLA 0.9.16. GPU memory requirements depend on the chosen model. Qwen-Drive's
model-local overlay uses Torch 2.8 / CUDA 12.8.

Choose a working filesystem with room for the unpacked CARLA runtime (about
23 GB), resource archives, selected model weights, and rollout traces. The
runner reserves 4 GiB plus the next job's estimated transient storage.

```bash
git clone https://github.com/ZihanYou/Language2Drive.git
cd Language2Drive/language2drive.carla
conda create -n language2drive-carla python=3.11.15 pip mkl-service=2.5.2 -c conda-forge
conda activate language2drive-carla
python -m pip install 'huggingface-hub>=0.36,<1' 'zstandard>=0.23,<1'

python tools/download_resources.py --list
python tools/download_resources.py --groups benchmark carla vendor-language2drive_adapters
bash tools/setup_environment.sh
export L2D_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src:$PWD:$PWD/language2drive/leaderboard:$PWD/language2drive/scenario_runner:$PWD/carla_runtime/PythonAPI:$PWD/carla_runtime/PythonAPI/carla"
```

The resource downloader uses an immutable HF commit and verifies both size
and SHA-256 before extraction. Its default download endpoint is
`https://hf-mirror.com`; override it with `--endpoint` or `HF_ENDPOINT`.
The direct HF endpoint is the fallback when the mirror is unavailable.
Archives remain cached under `.runtime/downloads/` for resumable downloads.
`--verify-only` checks downloaded archives without extracting them.

```bash
python tools/download_resources.py --groups benchmark --verify-only
python model_zoo/preflight.py --code-only
python -m language2drive.cli --help
```

`environment.lock.yml` records the installed package versions. Use
`tools/setup_environment.sh` for the installation order, CARLA wheel, Torch
index, and compatible Hugging Face dependency. Download the chosen model's
overlay as shown in [model_zoo/README.md](../model_zoo/README.md), then obtain
its original weights and run the model preflight.

## 3. Prepare and register CARLA

The evaluator runs with administrative access to local process metadata;
CARLA runs as the dedicated, unprivileged Linux account `sim`. Use a checkout
location whose parent directories allow `sim` to traverse to the runtime.
The simulator files must be readable/executable but not writable by `sim`.

As administrator, prepare the server account and runtime:

```bash
id sim || useradd --system --no-create-home --shell /usr/sbin/nologin sim
chown -R root:root carla_runtime
chmod -R a+rX,go-w carla_runtime
runuser -u sim -- test -x "$PWD/carla_runtime/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
```

Install the system Vulkan/OpenGL runtime libraries required by CARLA for your
Linux distribution. The packaged runtime includes the benchmark's custom
maps; use this package and the included Python wheel together.

Register the installed map/runtime inventory once per installation. In an
administrator terminal with the evaluation environment active:

```bash
# Terminal A: temporary server for obtaining the local runtime witness.
./tools/start_external_carla.sh --bootstrap --port 2090 --gpu-rank 0
```

```bash
# Terminal B: wait for CARLA to accept connections, then freeze the inventory.
l2d freeze-runtime-assets --port 2090
```

Stop the temporary server with Ctrl-C in Terminal A. Start the evaluation
server; keep it running during planning and execution:

```bash
./tools/start_external_carla.sh --port 2000 --gpu-rank 0
```

The launcher fixes the rendering profile to Epic and writes a launch receipt.
The runner verifies the runtime, map inventory and process identity. A
running registered CARLA server is required even when generating a plan
without `--execute`.

Server state lives under `.runtime/carla-server-release-v1` (or a port-specific
sibling). For a new launch after stopping a previous server, move that old
state directory aside before starting the launcher again. Preserve evaluation
results and plans. Run `freeze-runtime-assets --replace` only after an
intentional runtime change; a changed runtime defines a new execution bundle.

## 4. Obtain one model

Example for OneVL:

```bash
python tools/download_resources.py --groups vendor-onevl
python tools/download_models.py onevl --list
python tools/download_models.py onevl
python model_zoo/preflight.py --model onevl
```

Model IDs are `alpamayo1.5`, `simlingo`, `lmdrive`, `onevl`, `impromptu-vla`,
`qwen-drive-1.0-4b`, and `senna`. The Senna result label is **Senna VLM bridge**.
Weights and processors stay in each model's local directories and are loaded
offline during evaluation. Configure gated-model access before downloading.

## 5. Smoke test and core evaluation

Keep the registered CARLA server running. A one-route smoke run exercises
model loading, sensors, closed-loop control and trace capture:

```bash
l2d run --agent onevl --track standardized --tier small --split dev \
  --seeds 1 --max-runs 1 --results results/onevl_smoke \
  --plan-output generated/onevl_smoke.plan.json --wall-timeout 1800 --execute
```

For Base seed 1, run both splits into the same model-specific results tree:

```bash
l2d run --agent onevl --track standardized --tier base --split dev \
  --seeds 1 --results results/onevl_base_s1 \
  --plan-output generated/onevl_base_dev.plan.json --wall-timeout 1800 --execute

l2d run --agent onevl --track standardized --tier base --split test \
  --seeds 1 --results results/onevl_base_s1 \
  --plan-output generated/onevl_base_test.plan.json --wall-timeout 1800 --execute
```

Omit `--execute` to save the plan only. For three-seed reproduction use
`--seeds 1,2,3`. Replace `base` with `small` or `large` to select another tier.
Use `--host`, `--port`, `--traffic-manager-port`, and `--gpu-rank` consistently
with the external server. Model-specific GPU configuration, including
SimLingo's `parallel_port_gpu_map`, must match the selected endpoint.

Each run stores the evaluator result, `trace.jsonl`, capture metadata and
`runner_command.json` under `results/.../<run_id>/`. The run ID binds the track,
PCU, arm, expression and seed. The plan binds code, agent configuration,
sidecar, route XML, runtime and dependency hashes.

## 6. Resume and select runs

Re-run the same command to continue pending work. Finalized results are reused
only when their execution inputs match. A recorded model failure is a valid
observed outcome. Use `--retry-finalized` explicitly when rerunning finalized
jobs, and use a separate results directory for a new model configuration.

`--route-entry-ids 0,1` selects exact XML route entries; `--max-runs N` caps the
pending jobs executed in that invocation. Plans retain the full requested
scope for scoring, so a capped smoke run does not become a full benchmark
result. Store separate plan filenames for different splits and workloads.

`--wall-timeout` controls the infrastructure timeout. It does not change the
simulator-time task deadlines. For long batches,
`--fast-session-validation` performs the full authority check at the start
and checks the server/process identity between routes.

## 7. FDQ, Safety Priority and oracle

```bash
l2d run-fdq --agent onevl --seeds 1 --results results/onevl_fdq_s1 \
  --plan-output generated/onevl_fdq.plan.json --wall-timeout 1800 --execute

l2d run-safety --agent onevl --results results/onevl_safety \
  --plan-output generated/onevl_safety.plan.json --wall-timeout 1800 --execute

l2d run --agent oracle --track oracle --split dev --seeds 1,2,3 \
  --results results/oracle --plan-output generated/oracle_dev.plan.json --execute
```

FDQ reports scenario success with DS/RC/IP diagnostics. Safety Priority reports
safe noncompliance, safe following, unsafe following and response-mode metrics.
The privileged oracle is a suite validation instrument and has its own result
namespace. Keep these workloads separate from the Standardized A–D comparison.

## 8. Score and generate a report

Score with the saved plan for the intended split/workload. This uses the
complete expected scope, rescoring recorded core traces with the same
contracts and validating execution receipts. CARLA does not need to be
running for offline scoring.

```bash
l2d score --plan generated/onevl_base_test.plan.json \
  --results results/onevl_base_s1 --output generated/onevl_base_test.score.json
l2d report --score generated/onevl_base_test.score.json \
  --output generated/onevl_base_test.md --json-output generated/onevl_base_test.report.json

l2d score --plan generated/onevl_fdq.plan.json \
  --results results/onevl_fdq_s1 --output generated/onevl_fdq.score.json
l2d score --plan generated/onevl_safety.plan.json \
  --results results/onevl_safety --output generated/onevl_safety.score.json
```

For the core comparison:

- **SISR**: strict per-run success, including the task's semantic and execution
  conditions.
- **SemISR**: per-run semantic instruction success.
- **CFPS**: both counterfactual arms succeed strictly for the same PCU,
  expression and seed.
- **Sem-CFPS**: both arms achieve semantic success for that same pairing.

Report A–D layer rates, the exact tier/split/seeds, expected and valid counts,
model/configuration identity and plan hash. Missing or invalid runs remain in
the expected denominator. A smoke run verifies integration; use the complete
declared scope for benchmark comparisons. Do not combine FDQ and semantic
success into an undocumented single score.

## 9. Rebuild the suite and develop an agent

The resource archive contains the frozen source snapshot, compiled XML,
private evaluator sidecars, ledgers, map graphs and annotations. To recompile
that exact source snapshot:

```bash
l2d validate
l2d build-suite --reuse-sources
l2d validate-supplementary
```

For dataset development, `l2d build-maps` rebuilds map graphs and
`l2d build-suite` regenerates source PCUs before compiling. Use a separate
checkout/results tree for a newly generated suite. `l2d visualize` renders
suite views; `l2d qualify` checks static oracle-path structure. Dynamic oracle
runs provide the corresponding simulator evidence.

Frozen source captures carry hashed provenance identifiers. The release's
`frozen_paths.py` resolves those identifiers to packaged resources, preserving
the original bytes and hashes. Live runtime inventories are generated on the
new host and are separate from those construction records.

A custom agent implements the leaderboard entry point, sensor declaration,
setup and `run_step`, and binds public context through
`set_language2drive_context`. Use the included adapters as examples:

```bash
l2d run --agent-path /path/to/agent.py --agent-config /path/to/config.json \
  --track standardized --tier small --split dev --seeds 1 --max-runs 1 \
  --results results/custom_smoke --plan-output generated/custom_smoke.plan.json --execute
```

The Standardized interface rejects dense global route injection. Keep private
sidecars and targets in the evaluator. Model inputs, output representation and
controller settings should be stated alongside the results.

## 10. Verification

```bash
python model_zoo/preflight.py --code-only
python -m pytest tests/test_core_driving_score.py tests/test_fdq_metrics.py \
  tests/test_contracts_metrics.py tests/test_vlm_waypoint_parser.py \
  tests/test_core.py tests/test_visualize_release.py tests/test_release_portability.py -q
```

Data validation uses the downloaded benchmark resources. GPU/CARLA integration
is exercised by the one-route smoke command, followed by the full workload.
Preserve the Git revision, resource manifest, model weight revisions, adapter
configuration, saved plans and result directories for each reproduced run.
