# Input contracts

## Frozen PCUs

The source JSON contains a `pcus` list. A PCU includes:

- `pcu_id`, `task_family`, `decision_window` with start and deadline;
- `arms.R0` and `arms.R1`, their instructions and semantic/opposing contracts;
- evaluator-only `private_annotations`: gate segments, lane/scope polygons;
- `oracle_plans_by_seed`: frozen reference waypoints for DrivingScore;
- backend/scene bindings and review/qualification fields used by the compiler.

Here `private_annotations` means hidden from the **model**, not author identity.
Removing this field from scorer inputs would remove the semantic targets.
`schema.py` is the authoritative source schema validator; `compiler.py` separates
model-visible inputs from evaluator-side annotations. Synthetic source fixtures
in `tests/factories.py` exercise the complete schema, and `review_cli.toy_pairs`
contains small scorer-only geometry examples. Do not treat those as real scenes.

## Experiment document

```json
{
  "model_name": "model-alias",
  "configuration_id": "frozen-config-v1",
  "seed": 1701,
  "expected_runs": 60,
  "results": [
    {
      "pcu_id": "pcu-example",
      "arm_id": "R0",
      "seed": 1701,
      "execution_status": "completed",
      "evidence_valid": true,
      "origin": {"trace": "traces/example-r0.jsonl"},
      "components": {
        "own_contract": true,
        "opposing_contract": false,
        "progress": true,
        "deadline": true,
        "stability": true,
        "semantic_success": true
      }
    }
  ]
}
```

This illustrates **one record**, not a valid complete experiment. Supply all 60
distinct PCU/arm slots for scoring. Component values come from
`component_scoring.score_trace_components`, not manually entered success flags.
The `evidence_valid` flag must reflect upstream run/identity/trace validation.
Incomplete slots can be represented explicitly but cannot be counted as passes.

## Normalized trajectory (JSONL)

Each line is one sample with strictly increasing finite `time_s`:

```json
{"time_s": 0.0, "position": [0.0, 0.0], "speed_mps": 2.0, "progress_m": 0.0, "collision": false, "offroad": false, "render_valid": true}
```

The component verifier may additionally require event/region fields produced by
the runtime normalizer (`lane_ids`, `scopes`, `gates_crossed`, etc.). Coverage must
span the frozen scoring horizon. `collision`, `offroad`, `render_valid` are
booleans; invalid render or infrastructure-failure samples are rejected.
JSON does not permit NaN/Infinity. `alpasim_bridge.py` provides the ASL bridge;
raw coordinate files alone are not equivalent to a verified closed-loop run.

Use relative trace paths when sharing records. Exclude account names, absolute
host paths, checkpoint download tokens, scheduler logs and SSH commands from
review data. Preserve the PCU/arm/seed/configuration binding and metric evidence.
