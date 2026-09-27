# Candidate construction and human review

With full nuScenes v1.0-test metadata, camera/LiDAR keyframes and map expansion files installed:

```sh
export L2D_NUSCENES_ROOT=/path/to/nuscenes
export L2D_ARTIFACT_ROOT=/path/to/new_artifacts
python build_candidates.py
python serve_review.py --port 8765
```

The pipeline builds the anchor catalog, lane then speed candidates, witnesses, footprint/LiDAR evidence, eligibility checks, a provisional split and review pages. Construction thresholds are in `evaluation_parameters.json`. The released source inventory has 6,008 anchors / 150 scenes, 83 candidates, 82 automatically qualified cases and 75 human-approved PCUs. The full-data validator checks the fixed construction counts, witness contracts and disjoint log split.

Lane PCUs pair change with keep; speed PCUs pair the approximately 15/30 km/h requests. Witnesses establish feasible local alternatives during construction and are not unique imitation targets. Each scene contributes at most one selected PCU.

For remote review, run on the local machine:

```sh
ssh -N -L 8765:127.0.0.1:8765 user@host
```

Open `http://127.0.0.1:8765`; use Basic-auth username `reviewer` and the generated password in `$L2D_ARTIFACT_ROOT/audit/review_access_token`. The service binds to loopback. Each case collects reviewer alias, six judgments, decision and rationale; SQLite retains revision history and CSV export uses the latest decision. Approval requires positive observability, feasibility of both arms, mutual exclusivity, traffic clearance and map binding. Automatically rejected cases cannot be approved.

Freeze completed decisions to a fresh release:

```sh
python freeze_approved.py \
  --review-sheet "$L2D_ARTIFACT_ROOT/audit/review_sheet_submitted.csv" \
  --require-complete-review --version nuscenes-test-custom-v1 \
  --output releases/nuscenes-test-custom-v1
```

Keep personal reviewer records and service credentials outside public releases. Existing frozen inputs, splits and hashes are preserved when evaluating the supplied release.
