# Prediction submission format

Submit one UTF-8 JSON object keyed by every released `pcu_id`. Each PCU must
contain arms `"0"` and `"1"`. The canonical representation for each arm is:

```json
{
  "coordinate_frame": "ego_at_anchor",
  "reference_point": "rear_axle",
  "timestamps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
  "positions_xy": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                   [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
  "headings": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

`positions_xy` and `headings` are in metres and radians. Headings may be
omitted; the evaluator then derives them from successive positions. Every
number must be finite. Missing PCUs, missing arms, malformed trajectories, or
wrong coordinate metadata receive `O0_INVALID_OUTPUT` for the affected arm.

Minimal file structure:

```json
{
  "pcu_id_from_public_manifest": {
    "0": {"coordinate_frame": "ego_at_anchor", "reference_point": "rear_axle",
          "timestamps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
          "positions_xy": [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]},
    "1": {"coordinate_frame": "ego_at_anchor", "reference_point": "rear_axle",
          "timestamps": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
          "positions_xy": [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]}
  }
}
```

Score a submission from the track directory with evaluator-held sidecars:

```sh
python3 navsim_offline/score.py \
  --sidecars /private/path/private_sidecars \
  --predictions predictions.json \
  --output scores.json
```

Python and Shapely are required for semantic scoring.
