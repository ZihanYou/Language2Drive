# Observation, request and prediction formats

## Release layout

```text
prerelease/v0.1.0/
  manifest.json
  public/
    inputs.jsonl
    workload.jsonl
    image_manifest.jsonl
    images/
  private/
    candidates.jsonl
```

`inputs.jsonl` contains one observation per anchor: opaque `input_id`, source
frame/sequence ID, camera paths and hashes, calibration, past ego states, shared
route command and time-axis metadata. Current images contain eight views;
adapters use their documented subset. References, future state, candidate
family and scoring thresholds are absent from model inputs.

`workload.jsonl` contains one request per instruction:

```json
{"request_id":"R-example","input_id":"I-example","instruction":"Continue driving along the drivable route ahead.","instruction_language":"en"}
```

A candidate's two requests reference the same observation. Request IDs are
opaque to the model. The private registry maps them to A/B references and
contains draft status, reference XY arrays and original ratings. Unresolved
candidates remain in the registry with no generated request.

## Prediction interface

Each JSONL row is:

```text
{"request_id":"R-example","trajectory":[[x_1,y_1],...,[x_20,y_20]]}
```

The ellipsis illustrates the schema, not literal JSON. There must be exactly
20 finite XY points at 0.25, 0.50, ..., 5.00 seconds. Coordinates are in meters,
fixed at the anchor ego frame: rear-axle midpoint origin, x forward, y left.
The current origin (0,0) is implicit. Unknown and duplicate request IDs are
rejected; missing/invalid predictions are reported explicitly.

Model-native outputs remain available as separate run artifacts. DriveMA and
Fast-dDrive adapters write `xy_m`; convert with
`scripts/model_adapters/trajectory_xy_to_submission.py`. Qwen outputs use
`scripts/model_adapters/qwen_drive_postprocess.py`. Do not feed native formats
directly to the common scorer.

## Source protocol

Prediction horizon: 5 s; frequency: 4 Hz; 20 points. Rated arrays are truncated
to the first 20 points under `official_rfs_prefix20_v1`; short references are
excluded, not padded. Past ego state has 16 samples at 4 Hz. Qwen camera targets
are -1.5, -1.0, -0.5 and 0 s; nearest real frames may have <=0.1 s jitter, recorded
per frame. Missing history can use a separately tagged current-only protocol.

Normative upstream definitions: [data proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto),
[submission proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/end_to_end_driving_submission.proto),
and [RFS array processing](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py).
