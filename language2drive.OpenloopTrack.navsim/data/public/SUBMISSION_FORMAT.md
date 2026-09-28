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

Reproduce the manuscript's historical scores from the track directory with
evaluator-held sidecars:

```sh
sh scripts/run_paper_legacy_scoring.sh \
  --sidecars /private/path/private_sidecars \
  --predictions predictions.json \
  --output scores.json
```

The legacy runner uses an isolated Python environment without Shapely. For
geometry-aware scoring, install `navsim_offline/requirements-scorer.txt` and
call `python3 navsim_offline/score_corrected.py` with the same arguments.
