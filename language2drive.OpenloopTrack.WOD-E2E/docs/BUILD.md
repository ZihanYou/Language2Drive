# WOD-E2E local construction

The code package contains a frozen *source shard inventory*, not WOD records. Download WOD-E2E validation data from the official [Waymo access portal](https://waymo.com/open/data/e2e/) with your own registered access, then run:

```sh
sh scripts/setup.sh --source-root /path/to/waymo/e2e/val
```

The source directory must contain the 38 numbered TFRecord shard filenames and `val_sequence_name_to_scenario_cluster.json` listed in `config/paper_source_shards.json`. File sizes are checked before scanning. The script creates symlinks under ignored `assets/source_shards/`, so raw data are not copied into the repository. It then runs these stages in order:

1. `audit_local_data.py`: scan the numbered shards and extract rated reference metadata.
2. `summarize_candidates.py`: normalize the first 20 future points and construct full-window pairs.
3. `make_review_materials.py`: generate eight-camera imagery and paired trajectory review material.
4. `draft_semantic_tasks.py`: draft two English P0 instructions per describable pair.
5. `build_full_prerelease.py`: materialize public observations and instructions, plus a private candidate registry.
6. `freeze_paper_contracts.py`: select STOP_GO/SPEED_CHOICE pairs by two-way witness qualification and freeze private parameters.
7. `verify_assets.py`: hash-check each release file and JPEG and validate input/request references.

On the paper's source inventory, these stages yield 203 rated anchor scenes, 563 complete-window pairs, 441 pairs with English P0 requests (882 model queries), and 241 qualified motion-proxy PCUs (482 queries). The evaluation subset contains 76 STOP_GO and 165 SPEED_CHOICE PCUs. The default setup command checks these counts. Candidate extraction keeps all 563 pairs; qualification changes only the evaluator subset.

The local output layout is:

```text
artifacts/                         source inventory and normalized candidates
review/                            local review page, images and decisions
prerelease/v0.1.0/public/           observations, P0 workload and JPEGs
prerelease/v0.1.0/private/          rated references and candidate status
contracts/wod-e2e-p0-v0.1.0/       frozen evaluator parameters and hashes
```

`public/` is the only input permitted to model adapters. The evaluator uses `private/` and the frozen contracts separately. A paired request changes only the English instruction; public context remains fixed.

To verify an existing local build without rebuilding:

```sh
python scripts/verify_assets.py --release-dir prerelease/v0.1.0 --contracts-dir contracts/wod-e2e-p0-v0.1.0
```

The TFRecord metadata scanner skips JPEG payloads and does not check TFRecord CRCs. The frozen release manifest records hashes for all generated JSONL files, and the verifier hashes every exported JPEG. For complete raw-source validation, use the Waymo-provided data tooling.

## CPU baseline and geometry diagnostic

```sh
python scripts/run_public_cv_baseline.py --release-dir prerelease/v0.1.0 --output outputs/cv_predictions.jsonl
python scripts/score_full_prerelease.py --release-dir prerelease/v0.1.0 --predictions outputs/cv_predictions.jsonl --model-id cv --output-dir outputs/cv_geometry
python scripts/evaluate_paper.py --release-dir prerelease/v0.1.0 --contracts-dir contracts/wod-e2e-p0-v0.1.0 --model cv=outputs/cv_predictions.jsonl --output-dir outputs/cv_paper_metrics
```

The constant-velocity adapter uses public ego history and generates one trajectory per request. Geometry proximity is reported separately from the four paired instruction metrics.
