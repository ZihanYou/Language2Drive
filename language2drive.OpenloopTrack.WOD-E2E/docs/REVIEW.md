# Trajectory review and qualification

The generated interface displays English task instructions, eight camera views,
reference A/B XY paths and speed curves, a time slider, model/reference overlays
and model A/B comparison. Decisions are PENDING, PASS or FAIL, with reviewer
notes. Newly generated workspaces have no existing review decisions.

```sh
python scripts/serve_review.py --port 8787
```

On a remote machine, forward the loopback-only service from the local computer:

```sh
ssh -N -L 8787:127.0.0.1:8787 USER@HOST
```

Open `http://127.0.0.1:8787/` locally. Replace USER and HOST with your own access
configuration. This page contains private references; do not give it to models.

## Add model overlays and review ordering

```sh
python scripts/build_review_model_predictions.py --release-dir prerelease/v0.1.0 --model MODEL=experiments/MODEL/predictions.jsonl --score MODEL=experiments/MODEL/geometry/per_candidate.jsonl --output review/model_predictions.json
python scripts/build_model_review_priority.py --release-dir prerelease/v0.1.0 --model-score MODEL=experiments/MODEL/geometry/per_candidate.jsonl --output review/model_review_priority.csv
```

Add repeated model arguments for more models. Ranking changes review order,
not candidate qualification or score denominators. Decisions and appended event
history are stored separately from generated pages.

## Formal qualification procedure

First assess public observations without reference futures. Confirm both actions
are feasible and passenger-selectable, referents are observable, instructions
are mutually exclusive, the outcome is decidable within the horizon, and shared
navigation is compatible with both arms. Identify strong-interaction cases.

Then inspect private witnesses, freeze target regions/objects, time scope,
deadline, persistence, progress and tolerance parameters. Both witnesses must
pass their own predicate and fail the opposite predicate. Verify the original
semantic success sets are disjoint, rather than only adding an exclusion term
at scoring time. Use two independent reviewers and adjudicate disagreements;
unknown remains unresolved. A page PASS is an initial review decision.

After qualification, write equivalent English expressions and freeze development
and evaluation partitions grouped by sequence. Reviewers' names, notes and
source paths in generated files are user data and are not part of this code archive.
