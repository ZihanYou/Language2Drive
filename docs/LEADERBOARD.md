# Leaderboard submissions

The project page currently displays manuscript baselines. Submit community evaluations through a repository issue using the benchmark-result template; maintainers review the artifacts before adding a row. There is no automatic upload from the SDK.

Include model name, immutable checkpoint/adapter revisions, track and release, scorer profile, split, wording set, model/evaluator seeds, camera rig, rectification, frame conversion, controller and scheduled inventory. Attach or link native score JSON, consolidated summary, coverage and failure reports, prediction/trace hashes and reproduction commands. Do not attach credentials or licensed raw data.

RealSim submissions require real ASL receipts, exact scene/start/seed binding, verified driver/runtime identity and paired non-language input equivalence. CARLA summaries require the relevant additional PSM diagnostic evaluation. Historical NAVSIM results and corrected-scoring results are separate tables. Strict closed-loop success and open-loop exclusive semantic success must retain their native labels.

Failed or missing scheduled executions remain in the denominator. A custom adapter/controller is a distinct system configuration. Do not pool scores across tracks, count toy fixtures as evaluated scenes or claim a paper reproduction from an interface test.
