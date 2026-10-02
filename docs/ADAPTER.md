# One adapter, five tracks

Implement `create_adapter(config) -> object` and `object.predict(request) -> dict`. Start it with `language2drive serve --adapter your_adapter.py --config model.json` in your model's existing environment. Only the lightweight SDK is needed there. Image decoding requires Pillow; your model can have its own torch/transformers versions.

`Request` contains `track`, `instruction`, `observation`, `output_spec` and `seed`. Read the output specification dynamically; do not assume one common frame anchor or time grid.

| Track | Horizon / samples | Reference point | Observation envelope |
|---|---|---|---|
| `carla` | 5 s / 20 at 4 Hz | `rear_axle` | Front/left/right RGB, speed, camera rig, timestamp, declared axle offset |
| `realsim` | 6 s / 60 at 10 Hz | `rig_origin` | Declared AlpaSim cameras, rectification, speed, acceleration and timestamp |
| `nuscenes` | 3 s / 12 at 4 Hz | `ego_pose_origin` | Observed camera history, ego history and anchor ego pose |
| `navsim` | 4 s / 8 at 2 Hz | `rear_axle` | Four observed frames per camera and ego history |
| `wod` | 5 s / 20 at 4 Hz | `rear_axle` | Front-camera history, past states, calibration and public time axis |

All outputs use `coordinate_frame="ego_at_anchor"`, +x forward / +y left, metres. Timestamps are seconds **after** the prediction anchor; there is no t=0 waypoint. Include `reference_point`, `timestamps_s` and `xy_m` exactly as shown in the root README. Return ordinary Python lists/numbers (convert tensor or NumPy outputs with `.tolist()`). The SDK rejects incompatible grids, wrong frames and NaN/Infinity. Resampling and rigid-frame conversion belong in your adapter and must be recorded in your system configuration.

```python
from l2d_benchmark.service import decode_image

# Each camera is an ordered list of observed frames, oldest first.
frames = request.observation["cameras"]["CAM_FRONT"]
images = [decode_image(frame["image"]) for frame in frames]
```

Camera names are native to the dataset. CARLA uses `front`, `left`, `right`; RealSim uses the configured logical IDs, e.g. `camera_front_wide_120fov`. Frames travel as lossless RGB PNG/base64 over the local service, so the model does not require evaluator filesystem access. Timestamps are present where the public source supplies them. For nuScenes, the evaluator reads `sensor_root/v1.0-test/calibrated_sensor.json` when present and attaches camera intrinsics/extrinsics to each frame (use `metadata_version` to select another public metadata directory). Without that public file, calibration is explicitly null; NAVSIM's source workload does not contain full calibration, so a model needing it must declare its public upstream calibration in the model configuration. No privileged route command is supplied.

## State, weights and randomness

The factory creates one independent **session state** per open-loop request / closed-loop rollout. Share immutable weights through a module-level cached model to avoid loading a checkpoint repeatedly; never share mutable recurrent memory across counterfactual arms. Stateful adapters must implement `create_adapter`; a module-level `predict` is suitable only for stateless models. An optional `close()` releases session resources and must not destroy shared weights that other sessions still use.

A closed-loop session calls the same adapter repeatedly as observations arrive. RealSim may interleave multiple sessions; the model service serializes inference while preserving separate state. Use `request.seed` for model inference randomness. CARLA standardized mode does **not** expose the evaluator's private simulator seed; its bridge uses an arm-independent `model_seed` (default 0).

The service's health response fingerprints the adapter file and model configuration. The inference ledger binds these fingerprints, the declared checkpoint revision, request identity and exact request content. Changing checkpoint files in place without changing their revision is unsupported; use an immutable checkpoint revision. Open-loop resumes reuse previously recorded failures as failures. Choose a fresh output directory for deliberate retries or model changes.

## Simulator bridges

CARLA uses the new bridge in `src/l2d_benchmark/bridges/carla.py`. It consumes only the standardized context, converts BGRA to RGB and maps trajectories to control using pure pursuit (wheelbase 2.8 m, max steering angle 0.7 rad), proportional speed control and a configurable rear-axle offset. This is a declared SDK controller, distinct from native paper baselines. Verify the axle offset against the evaluated vehicle. Camera extrinsics are in CARLA's forward/right convention; model XY is forward/left. Convert calibration explicitly.

RealSim uses the `alpasim.models` entry point `l2d_unified`. The sidecar-aware trusted driver supplies only the current language string and opaque session/timestamp to the bridge. The bridge anchors its 10 Hz trajectory to the actual observed pose and returns native `ModelPrediction`; an uncovered anchor timestamp fails instead of silently guessing a pose. Configure input cameras and rectification in `l2d_unified.yaml` and declare them in the model configuration. The session UUID stays in the bridge/service envelope, outside `Request`.

One adapter means one implementation for the public contract. Different camera rigs, frame origins and time grids remain explicit inputs. A model that cannot support one of those requirements should fail that track visibly rather than receive an invented score.

## Debugging

1. `language2drive adapter-check --adapter examples/constant_velocity.py` verifies the five output specifications without datasets.
2. Inspect `language2drive evaluate --config examples/evaluation.json --tracks navsim --dry-run`.
3. Read each track's `command-*.log`, inference ledger and native score artifact. Keep protocol and coverage beside metrics.

The synthetic example has no benchmark claim. Your real model may require cameras and therefore cannot run the camera-free adapter check without an explicitly configured synthetic input mode.
