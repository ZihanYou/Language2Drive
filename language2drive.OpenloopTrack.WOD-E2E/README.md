# Language2Drive Open-loop Track: WOD-E2E

This directory contains the WOD-E2E paired-instruction construction and evaluation code accompanying *Language2Drive: Do Vision-Language-Action Models Really Follow the Instructions in Autonomous Driving?* The prediction protocol and the four paired metrics follow Appendix E.2 and E.5 of the paper.

## One-command setup

Obtain WOD-E2E validation shards through the [Waymo Open Dataset access portal](https://waymo.com/open/data/e2e/) under its [dataset terms](https://waymo.com/open/terms/). Then, from this directory:

```sh
sh scripts/setup.sh --source-root /path/to/waymo/e2e/val
```

The command creates a conda environment using the Tsinghua mirrors, selects the 38 source shards listed in `config/paper_source_shards.json`, builds the 563-pair candidate registry and English P0 requests, freezes the 241-PCU evaluator contracts, and checks the release files and images against their hashes. It uses the locally supplied Waymo data; the raw images and references are not included in this code repository. Running the command again verifies the existing local assets.

To also download the three model checkpoint repositories through `https://hf-mirror.com`:

```sh
sh scripts/setup.sh --source-root /path/to/waymo/e2e/val --models all
```

Use `--models drivema`, `--models fast-ddrive`, or `--models qwen-drive` to choose one. Checkpoint downloads need enough disk space and separate model-specific inference environments. No GPU is used during asset preparation. See [model adapters](docs/MODELS.md).

## Evaluation

Each PCU contains the same recorded observation and two English instructions. An adapter returns one trajectory per request in the rear-axle ego frame, sampled every 0.25 s through 5 s (20 XY points). The evaluator reads frozen private contracts only after predictions have been produced.

```sh
python scripts/evaluate_paper.py \
  --release-dir prerelease/v0.1.0 \
  --contracts-dir contracts/wod-e2e-p0-v0.1.0 \
  --model MyModel=outputs/my_predictions.jsonl \
  --output-dir outputs/my_scores
```

`summary.json` reports `SemSISR`, `SemCFPS`, `SCS`, and `PCS` on a 0–100 scale, along with valid-output counts. The WOD-E2E track uses the paper's **P0 kinematic-proxy profile**: 76 STOP_GO and 165 SPEED_CHOICE PCUs, with one wording per PCU. Missing or invalid model outputs remain in the fixed 241-PCU denominator with zero credit. The 441-pair geometric diagnostic is a separate analysis with a different denominator. See [metric specification](docs/SCORING.md), [paper-to-code map](docs/PAPER_ALIGNMENT.md), and [file formats](docs/FORMATS.md).

## Construction and review

The toolkit includes metadata extraction, paired candidate construction, English instruction drafting, an eight-camera review page, model-vs-reference and A/B trajectory overlays, and PASS/FAIL review records. Run the local review service with `python scripts/serve_review.py --port 8787`; SSH port forwarding is described in [review workflow](docs/REVIEW.md). Source records, output predictions, contracts, and human decisions are local assets ignored by Git.

The four model integrations are DriveMA-2B, Fast-dDrive, Qwen-Drive-SFT, and Qwen-Drive-RL. DriveMA and Fast-dDrive use three current front views; Qwen has separate four-frame and genuine current-only input protocols. The natural-language prompt adaptation is explicit in the adapters. No model weights, third-party source trees, WOD sensor data, references, or personal credentials are committed here.

Code is covered by the repository [MIT license](../LICENSE). WOD data access and use follow the [Waymo terms](https://waymo.com/open/terms/); external models retain their own terms. [Third-party notices](THIRD_PARTY_NOTICES.md) list sources.
