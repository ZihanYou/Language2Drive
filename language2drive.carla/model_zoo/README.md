# Model zoo

The adapters use the Standardized Track's public language context and declared
sensor interface. Evaluator contracts, semantic targets, and private oracle
paths are not supplied as model inputs.

| CLI model ID | Evaluation interface | Local dependency group |
|---|---|---|
| `alpamayo1.5` | Alpamayo 1.5 camera/history trajectory generation and controller | `vendor-alpamayo1.5` |
| `simlingo` | SimLingo native network, waypoint heads and PID | base environment |
| `lmdrive` | LMDrive visual encoder, language model, waypoint head and PID | `vendor-lmdrive` |
| `onevl` | OneVL-NAVSIM front camera and latent-prefix trajectory generation | `vendor-onevl` |
| `impromptu-vla` | Three-camera Impromptu-VLA trajectory generation | base environment |
| `qwen-drive-1.0-4b` | Qwen-Drive reasoning and planning heads | `vendor-qwen-drive-1.0-4b` |
| `senna` | **Senna VLM bridge**: high-level SPEED/PATH plan converted to waypoints | `vendor-senna` |

Always identify the Senna entry as **Senna VLM bridge** in result tables. The
camera and action interfaces are defined by each model's `adapter_config.*`.
Changes to those configurations define a new evaluation configuration.

## Installation

Activate the base environment described in [the reproduction guide](../docs/BENCHMARK.md).
All models use the shared dependency group; add the selected model's overlay:

```bash
python tools/download_resources.py --groups vendor-language2drive_adapters vendor-onevl
python tools/download_models.py onevl --list
python tools/download_models.py onevl
python model_zoo/preflight.py --model onevl
```

`--list` prints download locations without downloading weights. `WEIGHTS.json`
pins the original publishers and revisions. For gated repositories, complete
the publisher's access process and configure `HF_TOKEN` or Hugging Face login
locally. Use `--endpoint https://huggingface.co` to download directly.

The SimLingo downloader includes its InternVL2-1B backbone. LMDrive includes
its visual encoder, LMDrive checkpoint, LLaVA backbone and BERT resources.
OneVL downloads the two inference shards used by the trajectory adapter and
constructs the corresponding index. Alpamayo downloads its Cosmos processor
separately. None of these weights is hosted in our resource repository.

Overlays are loaded from `model_zoo/MODEL/vendor/` by the selected adapter.
Qwen-Drive carries its Torch 2.8 / CUDA 12.8 software stack there; the base CARLA
environment remains on its pinned Torch stack. The `.tar.zst` overlay is a
software dependency bundle, with no model checkpoints.

The Alpamayo adapter also requires FlashAttention for Torch 2.2 / CUDA 12.1:

```bash
python tools/download_resources.py --groups flash-attention
python -m pip install resources/wheels/flash_attn-*.whl
```

## Source verification

`SOURCES.json` records upstream commits and SHA-256 values for the adapted
source snapshot. Upstream license files remain alongside the source. The
Adapter configurations can be adjusted for the evaluation host and are bound
by each saved execution plan. The release includes the runtime source used by each adapter; no nested Git
checkout is required.

```bash
python model_zoo/preflight.py --code-only
python model_zoo/preflight.py --model simlingo --json
```

## Evaluation

Start and register CARLA as described in the main guide, then:

```bash
# Generate a plan (default behavior).
./model_zoo/run_base.sh simlingo
# Execute one smoke route.
L2D_EXECUTE=1 L2D_MAX_RUNS=1 L2D_SPLIT=dev ./model_zoo/run_small.sh simlingo
# Execute the Base test split.
L2D_EXECUTE=1 ./model_zoo/run_base.sh simlingo
```

Supported settings: `L2D_PYTHON`, `L2D_SPLIT`, `L2D_SEEDS`, `L2D_RESULTS`,
`L2D_PLAN`, `L2D_HOST`, `L2D_PORT`, `L2D_TM_PORT`, `L2D_GPU_RANK`,
`L2D_WALL_TIMEOUT`, `L2D_MAX_RUNS`, `L2D_EXECUTE`, and `L2D_PRINT_COMMAND`.
Use a distinct results directory for each model/configuration.
