# Language2Drive OpenloopTrack · nuScenes

Paper-aligned paired instruction-following evaluation on **nuScenes v1.0-test**. Each paired counterfactual unit (PCU) uses the same observed context with two alternative driving instructions. The evaluator checks predicted trajectories against executable task contracts.

The release contains **75 PCUs**: 15 development and 60 heldout, split by driving log. Tasks are lane change versus lane keeping and approximately 15 versus 30 km/h scoped-speed requests. Predictions use a **3 s horizon, 0.25 s interval, 12 points**, in the anchor ego frame (+x forward, +y left).

## One-command setup and paper replay

The replay assets are stored in the **private** `Mike087/Language2Drive` dataset under `language2drive.OpenloopTrack.nScenes/`. Your Hugging Face account needs read access. Supply `HF_TOKEN` through your credential environment; it is never written to the package. With conda available, from this directory:

```sh
sh scripts/setup.sh --with-assets
sh scripts/reproduce.sh
```

The first command creates a local conda environment using the Tsinghua mirror, runs CPU tests, and downloads the checksum-pinned [replay bundle](https://huggingface.co/datasets/Mike087/Language2Drive/tree/main/language2drive.OpenloopTrack.nScenes). The second recomputes all four nuScenes rows of paper Table 9, checks their displayed values, and saves full metrics, intervals, PCU scores and provenance under `outputs/paper_replay/`. It needs **no GPU, model weights or raw sensor files**. The bundle is approximately 14 MB and includes scoring maps, evaluator contracts, and the frozen predictions.

If using an existing compatible conda environment:

```sh
python scripts/download_assets.py --prompt-token
python -m unittest discover -s tests -v
python scripts/reproduce_paper.py --check-paper --output-dir outputs/my_replay
```

Downloads default to **https://hf-mirror.com**, without environment proxies, and use your authorized read token for the private repository. `--prompt-token` reads the token without echo or persistence. For direct official access, select `--endpoint https://huggingface.co`. Cross-host download redirects strip the authorization header. All files are hash checked. Use `--archive /path/to/nuscenes-paper-1.0-replay.zip` for an offline installation. Downloading twice verifies the installed assets and avoids transferring them again. An existing result directory is never silently overwritten.

## Paper metrics

| Metric | Definition | Scale |
|---|---|---|
| SemSISR | Mean exclusive own-contract success over both arms | 0–100 |
| SemCFPS | Both arms succeed under the same paired wording | 0–100 |
| SCS | Mean own-arm graded completion | 0–100 |
| PCS | Mean weaker-arm completion within each pair | 0–100 |
| SRG | Difference of relative target-distance preferences | −100–100 |
| NBD | Normalized difference between paired trajectories | 0–100 |

The `nuscenes-paper-1.0` profile implements Appendix E.3, Eqs. (34)–(40). In particular, graded completion is temporal coverage multiplied by `min(1, endpoint_displacement / 8 m)`. The own/opposite exclusivity test applies to binary success. SRG uses distance preference, while NBD measures behavior-change magnitude.

The paper table uses **canonical heldout, seed 0**. The same replay also scores all three paraphrases and masked/shuffled controls. Four-wording averages are reported separately. Pairing occurs before wording averaging; PCUs then receive equal weight. Missing requested records are rejected; invalid trajectory payloads receive zero binary/completion credit. Conditional response diagnostics include valid-pair coverage.

See [metric specification](docs/METRICS.md), [paper-to-code mapping](docs/PAPER_ALIGNMENT.md), and [all numeric parameters](docs/evaluation_parameters.json).

## Reproduced paper table

| Model | SemSISR | SemCFPS | SCS | PCS |
|---|---:|---:|---:|---:|
| Impromptu-VLA | 15.83 | 0.00 | 20.21 | 0.00 |
| UniDriveVLA | 16.67 | 0.00 | 25.00 | 0.00 |
| OpenDriveVLA-0.5B | 13.33 | 3.33 | 19.06 | 4.30 |
| OmniDrive | 7.50 | 0.00 | 29.24 | 8.61 |

These values are computed from the provided normalized predictions. `config/paper_table9.json` stores the paper's rounded reference values only for the final assertion; it is not an input to any scoring kernel. Full-precision values are written to CSV/JSON before display rounding.

## Evaluate your model

An adapter exports `predict(public_input, instruction, seed)` and returns:

```python
{
    "coordinate_frame": "ego_at_anchor",
    "reference_point": "ego_pose_origin",
    "timestamps_s": [0.25 * i for i in range(1, 13)],
    "xy_m": predicted_xy,  # 12 finite [x, y] pairs in meters
}
```

The adapter receives public observations and the current instruction. Evaluator contracts and witnesses are not passed to it. See [input/output specification](docs/INPUT_OUTPUT.md).

Run the included CPU reference adapter:

```sh
python run_model.py \
  --release-dir releases/nuscenes-test-v1.0.0 \
  --split all --adapter-file constant_velocity_adapter.py \
  --conditions canonical,masked,shuffled,paraphrase_1,paraphrase_2,paraphrase_3 \
  --output outputs/constant_velocity.jsonl
python evaluate_predictions.py \
  --predictions outputs/constant_velocity.jsonl \
  --split heldout --output-dir outputs/constant_velocity_scores
```

Replace `--adapter-file` with your model adapter. For resumable inference use `model_eval/run_resumable.py` with the same arguments. Real model inference additionally requires nuScenes v1.0-test sensor data and the model's environment/checkpoint. Set `L2D_NUSCENES_ROOT=/path/to/nuscenes`; the directory must include `maps/expansion` for scoring. Obtain sensor data through the [nuScenes download portal](https://www.nuscenes.org/download). The frozen prediction replay does not run or download pretrained models.

Use `--conditions canonical` to score a complete canonical run. Use `--wordings canonical,paraphrase_1,paraphrase_2,paraphrase_3` to declare a four-wording headline. Controls are scored separately against the original arm contracts. For the published shuffled protocol, run inference on all 75 PCUs before heldout filtering.

## Build and review new tasks

Candidate construction, witness generation, map/footprint checks, review pages, authenticated SSH access, per-case submission and freeze tools are included. See [construction and review](docs/CONSTRUCTION.md). New human decisions create a new release; the supplied `nuscenes-test-v1.0.0` release remains fixed.

## Files and licenses

- `evaluate_predictions.py`, `model_eval/paper_metrics.py`: paper-aligned scoring and aggregation.
- `scripts/`: environment setup, asset installation and complete table replay.
- `releases/`: frozen public input and original release manifest.
- `config/`: pinned assets, scoring profile and paper-table reference.
- `docs/`, `tests/`: usage, complete parameters and CPU regressions.

Code follows the parent repository's MIT license. nuScenes maps and derived replay assets follow their separate data terms; see [NOTICE.md](NOTICE.md). No raw camera images, LiDAR scans, credentials, review-person identities, or model weights are stored in this code directory. The paper PDF is not redistributed.

中文说明：[部署与复算指南](docs/USAGE_ZH.md)。
