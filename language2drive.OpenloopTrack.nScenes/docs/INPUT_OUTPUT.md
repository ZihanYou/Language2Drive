# Public inference and evaluation input

The frozen public JSONL contains PCU ID, task family, two instructions with three paraphrases each, and the shared observed input. The shared input includes current and historical sensor references, timestamps, ego poses and a canonical-JSON SHA-256. Each adapter invocation receives only `(public_input, instruction, seed)`.

Predictions are JSONL objects with `pcu_id`, `arm` (`arm_0` or `arm_1`), `condition`, integer `seed`, `public_input_hash`, `instruction`, and `trajectory`. The runner creates this envelope; the adapter returns only the trajectory. All selected identities must be unique and complete, and each input hash must match the frozen release. Multiple splits may share a file; scoring selects the requested split/condition/seed.

The trajectory object contains `coordinate_frame="ego_at_anchor"`, `reference_point="ego_pose_origin"`, `timestamps_s` and `xy_m`. The two arrays contain 12 future points at 0.25 s intervals through 3 s. No t=0 point is included. +x is forward, +y is left, positions are meters. The anchor uses the LIDAR_TOP keyframe's ego pose. Transform local points into global coordinates with the anchor yaw rotation and translation.

For a model with six native 0.5 s future positions, prepend the observed origin and linearly interpolate to 12 points; do not extrapolate beyond its predicted horizon. The interface validates timestamps, frame, reference point, point count and finite coordinates. Unparseable outputs can be represented by an empty/null trajectory payload, which is scored invalid rather than replaced.

Evaluator-only `private_sidecars.jsonl` retains contract lane tokens, speed bands, witnesses and qualification metadata. It is distributed in the replay asset bundle and loaded only after inference by the evaluator. Public runners never load or pass it to model adapters. For isolated model execution, provide only the public manifest, sensor files and model assets in that process.

Use `L2D_NUSCENES_ROOT` for the separately installed nuScenes root, `L2D_RELEASE_DIR` for an alternate frozen release, and `L2D_ARTIFACT_ROOT` for new construction outputs. For paper replay, the asset installer places maps under the default `data/nuscenes` directory. Camera/LiDAR observations remain in the upstream dataset and are not required for replaying supplied predictions.
