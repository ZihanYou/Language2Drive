# Model adapters and inference

Keep inference environments separate from the CPU tools. Install the upstream
runtime versions compatible with the pinned checkpoints. DriveMA uses
Transformers 5.17.0 in the integration environment; Fast-dDrive uses 4.57.1.
Qwen-Drive uses its official repository requirements. CUDA/PyTorch and compiled
attention dependencies must match the target GPU environment. Packages can be
installed with `python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple`.

Check `nvidia-smi` for active processes and available memory before any GPU run.
Select an idle GPU with sufficient memory; if none is available, wait or select
another allocation. The Qwen launchers check for compute processes and require
at least 22,000 MiB free; this is a launch guard, not a universal memory bound.

## External checkpoints

After obtaining the WOD validation shards, the setup command can prepare the
local evaluation assets and fetch the complete external checkpoint repositories
in one invocation:

```sh
sh scripts/setup.sh --source-root /path/to/waymo/e2e/val --models all
```

The downloader uses `https://hf-mirror.com`, pins DriveMA and Fast-dDrive to
the recorded upstream commits, resolves a full Qwen checkpoint commit before
download, and records that revision locally. It also obtains the official Qwen
code at commit `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`. To download
models without rebuilding local WOD assets, run
`python scripts/download_assets.py --only-models --models all` in the setup
conda environment. Individual names are `drivema`, `fast-ddrive`, and
`qwen-drive`.

The generic helper supports manual checkpoint downloads as well:

```sh
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-download.txt
python scripts/download_checkpoint.py --repo zwc2003/DriveMA-2B --revision d5f4905d2032da943a7ae8f22d5b2b8efbe4b0f7 --output models/cache/drivema-2b
python scripts/download_checkpoint.py --repo Efficient-Large-Model/Fast-dDrive --revision 0fda81009f4efa58a2debbb48c0c09818e45341f --output models/cache/fast-ddrive
```

For Qwen model inference, install the separate upstream runtime from the
downloaded official repository. All helper downloads use the HF mirror;
checkpoint and repository terms remain upstream.

## DriveMA and Fast-dDrive

```sh
CUDA_VISIBLE_DEVICES=GPU_ID python scripts/model_adapters/drivema.py --inputs prerelease/v0.1.0/public/inputs.jsonl --workload prerelease/v0.1.0/public/workload.jsonl --image-root prerelease/v0.1.0/public --model models/cache/drivema-2b --output experiments/DriveMA/raw.jsonl --resume
CUDA_VISIBLE_DEVICES=GPU_ID python scripts/model_adapters/fast_ddrive.py --inputs prerelease/v0.1.0/public/inputs.jsonl --workload prerelease/v0.1.0/public/workload.jsonl --image-root prerelease/v0.1.0/public --model models/cache/fast-ddrive --output experiments/Fast-dDrive/raw.jsonl --resume
python scripts/model_adapters/trajectory_xy_to_submission.py --raw experiments/DriveMA/raw.jsonl --output experiments/DriveMA/predictions.jsonl --report experiments/DriveMA/conversion_report.json
python scripts/model_adapters/trajectory_xy_to_submission.py --raw experiments/Fast-dDrive/raw.jsonl --output experiments/Fast-dDrive/predictions.jsonl --report experiments/Fast-dDrive/conversion_report.json
```

Replace GPU_ID with the available physical index. Both adapters consume three
current front views, shared navigation/history and the English instruction.
They append the passenger instruction to the original model prompt; this is an
experimental instruction interface. Five native 1 Hz points are linearly
interpolated with the implicit current origin to 20 benchmark points.

Use `--shard-index K --num-shards N` for parallel requests; keep output filenames
separate. For deterministic decoding, `request_dedup.py prepare/expand` can
remove exact duplicate (input_id,instruction) calls and restore request IDs.
No reference-based best-of-K selection is performed.

## Qwen-Drive SFT/RL

```sh
python scripts/model_adapters/qwen_drive.py --inputs prerelease/v0.1.0/public/inputs.jsonl --workload prerelease/v0.1.0/public/workload.jsonl --image-root prerelease/v0.1.0/public --official-repo models/cache/Qwen-Drive-1.0-code --allow-temporal-jitter 0.1 --skip-incomplete --audit experiments/Qwen/input_audit.jsonl --output models/runs/qwen_drive/full_compatible_scenes.jsonl
bash scripts/model_adapters/run_qwen_drive_full.sh GPU_ID
```

Activate the Qwen conda environment before launching. `QWEN_PYTHON`, `QWEN_REPO`,
`QWEN_WEIGHTS`, `QWEN_RUN_DIR`, `QWEN_PUBLIC_DIR` and `QWEN_RUNTIME_CACHE` can
override runtime locations. Default locations are package-relative. The launchers
run SFT/RL with reasoning_planning, one sample, seed 42 and SDPA, then convert
50 native 10 Hz points to 20 benchmark points.

Missing four-frame camera history may use a separate current-only protocol:
`--allow-current-only --current-only-only --skip-incomplete`, with its own audit
and output files. Real current images are not repeated as fake historical frames.
Report native four-frame and current-only results separately. The supplied
current-only launcher expects unique scenes and a map produced by
`qwen_drive_dedup.py prepare`; use its CLI for explicit paths. The merge helper
preserves protocol and conversion audits.

To complete both Qwen protocols, first prepare current-only scenes and an exact
input-deduplication map in the Qwen runtime environment:

```sh
python scripts/model_adapters/qwen_drive.py --inputs prerelease/v0.1.0/public/inputs.jsonl --workload prerelease/v0.1.0/public/workload.jsonl --image-root prerelease/v0.1.0/public --official-repo models/cache/Qwen-Drive-1.0-code --allow-temporal-jitter 0.1 --allow-current-only --current-only-only --skip-incomplete --audit models/runs/qwen_drive/current_only_audit.jsonl --output models/runs/qwen_drive/current_only_scenes.jsonl
python scripts/model_adapters/qwen_drive_dedup.py prepare --scenes models/runs/qwen_drive/current_only_scenes.jsonl --workload prerelease/v0.1.0/public/workload.jsonl --unique-scenes models/runs/qwen_drive/current_only_unique_scenes.jsonl --mapping models/runs/qwen_drive/current_only_dedup_map.jsonl --report models/runs/qwen_drive/current_only_dedup_report.json
bash scripts/model_adapters/run_qwen_drive_current_only.sh GPU_ID
```

After both launchers finish, merge predictions per planner while preserving the
protocol split:

```sh
python scripts/model_adapters/merge_qwen_drive_protocols.py --workload prerelease/v0.1.0/public/workload.jsonl --native-scenes models/runs/qwen_drive/full_compatible_scenes.jsonl --current-only-scenes models/runs/qwen_drive/current_only_scenes.jsonl --native-predictions models/runs/qwen_drive/full_sft_predictions_4hz.jsonl --current-only-predictions models/runs/qwen_drive/current_only_sft_predictions_4hz.jsonl --output models/runs/qwen_drive/sft_merged_predictions_4hz.jsonl --report models/runs/qwen_drive/sft_merge_report.json
```

Repeat the merge with `rl` filenames for Qwen-Drive-RL. Use the merged file for
complete request coverage, and report four-frame/current-only results separately
using the evaluator's `by_camera_history_availability` output. Exact duplicates
may share one deterministic model call, while every request retains an output ID.

## Scoring a model

```sh
python scripts/score_full_prerelease.py --release-dir prerelease/v0.1.0 --predictions experiments/MODEL/predictions.jsonl --model-id MODEL --output-dir experiments/MODEL/geometry
python scripts/evaluate_paper.py --release-dir prerelease/v0.1.0 --contracts-dir contracts/wod-e2e-p0-v0.1.0 --model MODEL=experiments/MODEL/predictions.jsonl --output-dir experiments/MODEL/paper_metrics
```

See SCORING.md for the exact metric definitions and denominators.
