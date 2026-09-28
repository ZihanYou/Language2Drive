# Asset provenance and layout

`scripts/download_assets.sh` fetches assets from these upstream releases. It
does not require a Hugging Face token for public assets.

| Asset | Source | Destination |
|---|---|---|
| OpenScene v1.1 test camera and LiDAR shards | `OpenDriveLab/OpenScene` on ModelScope | `assets/openscene-v1.1/sensor_blobs/test` |
| Qwen-Drive, AutoVLA, DriveMA, ReCogDrive, OneVL source | Official GitHub repositories at the commits in `EXPERIMENT_PROTOCOL.md` | `model_eval/repos/` |
| Released model checkpoints | Official Hugging Face repositories at the revisions in `EXPERIMENT_PROTOCOL.md` | `model_eval/weights/` |

The downloader removes proxy environment variables, uses the ModelScope mirror
for sensors and `hf-mirror.com` for checkpoints, verifies sensor archive sizes
and hashes when exposed by the mirror, and records resumable completion markers
in `assets/navsim_test_downloads/completed/`. Checkpoint file sizes and model
revisions are checked before their manifest is marked complete. Repository code
is checked out at the exact recorded commit.

The sensors may require substantial disk space. `--cameras-only` limits the
sensor transfer for the five baseline adapters. Existing sensor data can be
reused by setting `L2D_NAVSIM_SENSOR_ROOT` to its test sensor root, followed by
`python3 scripts/check_public_data.py --sensors`.

The upstream assets are subject to their own access terms and licenses. This
repository distributes only the small benchmark input JSON and original
adapter/evaluator code.
