# Language2Drive · CARLA Track

Language-conditioned, closed-loop driving evaluation in CARLA. The benchmark
uses paired counterfactual units (PCUs): two instructions share an environment
and require different behaviors. Each arm has three language expressions.

The source suite contains **128 PCUs / 256 arms / 768 route entries**, organized
into four layers: atomic actions (A), parameterized constraints (B), grounded
semantic navigation (C), and multi-stage programs (D).

| Evaluation tier | PCUs | Dev runs / seed | Test runs / seed | Total runs / seed |
|---|---:|---:|---:|---:|
| Small | 32 | 48 | 144 | 192 |
| Base | 64 | 84 | 300 | 384 |
| Large | 119 | 138 | 576 | 714 |

The execution tiers follow the frozen map policy. FDQ provides 24 scenario
routes per seed; Safety Priority provides six independent requests.

## Get started

```bash
git clone https://github.com/ZihanYou/Language2Drive.git
cd Language2Drive/language2drive.carla
conda create -n language2drive-carla python=3.11.15 pip mkl-service=2.5.2 -c conda-forge
conda activate language2drive-carla
python -m pip install 'huggingface-hub>=0.36,<1' 'zstandard>=0.23,<1'
python tools/download_resources.py --groups benchmark carla vendor-language2drive_adapters
bash tools/setup_environment.sh
python model_zoo/preflight.py --code-only
```

Follow **[the benchmark and reproduction guide](docs/BENCHMARK.md)** for CARLA
startup, model setup, complete evaluation, resumption, scoring, and custom agents.

## Model zoo

This release contains adapters and dependency source for **Alpamayo 1.5,
SimLingo, LMDrive, OneVL, Impromptu-VLA, Qwen-Drive-1.0-4B, and Senna VLM
bridge**. [Model setup](model_zoo/README.md) documents their input/output
interfaces and pinned upstream repositories.

Model weights are obtained from their original publishers using
`python tools/download_models.py MODEL`. Neither this Git repository nor our
resource dataset redistributes model weights.

## Resources and layout

[Hugging Face resources](https://huggingface.co/datasets/Mike087/Language2Drive/tree/main/language2drive.carla)
contain CARLA, benchmark assets, and software dependency bundles. The downloader
supports `HF_ENDPOINT=https://hf-mirror.com` and verifies archive SHA-256 values
at an immutable revision recorded in `resources/manifest.json`.

```text
src/language2drive/  # suite construction, contracts, execution, scoring
language2drive/      # CARLA leaderboard and scenario runner
team_code/           # evaluator integration and privileged validation agent
model_zoo/           # seven model adapters, sources, pinned download definitions
tools/               # installation and resource utilities
tests/               # metric, contract, runner and release tests
tests_evaluator/     # evaluator integration tests
```

Original project code is covered by [LICENSE](LICENSE). Third-party components
retain their own licenses; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
