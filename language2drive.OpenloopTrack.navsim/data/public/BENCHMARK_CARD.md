# L2D Offline NAVSIM Benchmark

## Scope

This benchmark measures whether changing only a natural-language instruction
changes a planner's four-second trajectory in the intended semantic direction.
The evaluation unit is a Paired Counterfactual Unit (PCU): both arms share the
same sensor history and non-language state while receiving mutually exclusive
English instructions.

The primary metrics are OL-SemSISR over individual arms and OL-SemCFPS over
paired arms. A PCU contributes to OL-SemCFPS only when both counterfactual arms
satisfy their own frozen contract and reject the opposite contract.

## Release contents

- `data/public_test_set.tar.gz`: compact public PCU inputs, instructions,
  paraphrases, sensor references, and the public manifest.
- `navsim_offline/`: scorer and contract verifier source. Contract
  sidecars are retained by the benchmark evaluator.
- `submission_schema.json`: canonical prediction JSON schema.
- `manifest.json`: selected PCU identifiers, coordinate frame, and version.

The camera and LiDAR files are distributed through OpenScene/NAVSIM and are not
duplicated in the public archive. Paths in each public input are relative to the
OpenScene v1.1 test sensor root.

## Model input and output

For each PCU, run the two arms independently. Each arm receives the same public
observation with only the instruction changed. Output eight rear-axle poses at
0.5-second intervals from 0.5 through 4.0 seconds in the anchor-local
`ego_at_anchor` frame. See `SUBMISSION_FORMAT.md` and
`submission_schema.json`.

## Metrics

- `ol_sem_sisr`: exclusive semantic success averaged across arms.
- `ol_sem_cfps`: fraction of PCUs for which both arms succeed; primary metric.
- Signed contract margin and semantic failure taxonomy.
- Three-second common-horizon results for eligible families.
- Tier and family breakdowns with recording-cluster bootstrap intervals.
- Trajectory validity and kinematic feasibility as separate planning-quality
  evidence.
- Optional masked, generic, shuffled, and paraphrase language controls.

## Qualification

Every released PCU passed source-split isolation, sensor availability, witness
replay, mutually exclusive contract checks, map-footprint checks, logged-actor
checks, 0.1-second continuous path collision review, public/private leakage
inspection, and recorded human review. Rejected PCUs are absent from both
release archives.

## Interpretation

The benchmark evaluates planning-level language control under fixed logged
observations. Logged-actor and map checks are conditional safety evidence. The
benchmark does not establish closed-loop recovery, counterfactual actor
responses, or long-horizon task completion.
