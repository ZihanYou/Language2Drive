# Runtime integration

Use `sh scripts/setup_runtime.sh --install-rust` for an isolated installation from
the fixed upstream revision and lock. See [environment creation and troubleshooting](ENVIRONMENT_SETUP_ZH.md).
No pre-existing Python environment archive is required or distributed.

## Two environments

1. **Offline evaluator environment:** Python 3.10+, this package and pytest.
   Scoring and synthetic CPU regression need no Docker, GPU, network or checkpoint.
2. **Closed-loop environment:** the pinned upstream AlpaSim runtime, matching
   NuRec scene assets/renderer, GPU drivers and a separately installed model
   implementation. Follow the setup documentation distributed with that upstream
   revision. This package does not install NVIDIA system components or credentials.

The integration pins are in `config/alpasim_realsim.json` and `constants.py`:
AlpaSim `affc2eab209fa43bdfa2f26c0f8d437922d78a68`, dataset revision `26.01`,
NuRec version `26.1.112-59673f88`. The host nonlinear-controller overlay in
`config/runtime-compatibility.txt` specifies `casadi==3.7.2`.

Install this package into the upstream driver/runtime environment as appropriate:

```sh
python -m pip install -e '.[adapter]'
python -m pip install -r config/runtime-compatibility.txt
```

These commands add the integration package and overlay, not the entire simulator.
Retain upstream dependency locks. The SimLingo inference worker uses its own
upstream-prescribed environment and communicates through the length-framed
protocol in `simlingo_model.py`; its torch/transformers dependencies should not
be guessed from the CPU scoring environment.

## Driver registration

Installed package entry points register `l2d_alpamayo1_5`,
`l2d_alpamayo2_super`, `l2d_simlingo` and `l2d_privileged_oracle`.
Hydra templates are in `src/language2drive_realsim/configs/driver/`.
Checkpoint locations are deployment inputs; configure them through the
environment/configuration fields in the chosen template. Never embed tokens.
The included null-LiDAR configuration is an integration template and needs its
separate LMDrive adapter; this source artifact does not bundle every model backend.

The normal language-model interfaces disable waypoint route commands and use
the sidecar instruction schedule. Pass `L2D_REALSIM_SIDECAR` only to the trusted
driver wrapper. Keep geometric targets and oracle plans on the evaluator side.
Oracle and benchmark-model driver identities are checked separately.

## End-to-end order

1. Obtain scene assets and official checkpoints under their respective terms.
2. Supply the frozen reviewed PCU source. Validate scene/time bindings, target
   geometry, paired instruction issuance, rendering and oracle feasibility.
3. Compile with `compiler.compile_to_directory` or inspect `l2d-realsim compile --help`.
4. Start the renderer, runtime and selected driver with independent ports and
   explicit GPU allocation. Prefer local-only control-service access.
5. Build requests with `runtime_client.build_run_plan` and
   `simulation_request_payload`; submit via `submit_run_plan`. Select seed 1701
   explicitly for the single-seed model experiment. The general source compiler
   also supports multi-seed qualification; do not accidentally mix these runs.
6. Save raw run provenance; normalize ASL output with the bridge; verify paired
   prefix equivalence, execution/model identity and trace coverage. Evaluate
   semantic component predicates on the unchanged normalized trajectories.
7. Construct the 60-slot document described in `DATA_FORMAT.md` and run
   `l2d-score` or `sh scripts/score.sh`. Supplemental controls use
   `supplemental_diagnostics.make_runs` followed by the same runtime path.

Public interface help can be inspected without starting an experiment:

```sh
l2d-realsim --help
l2d-realsim compile --help
l2d-realsim run --help
l2d-review --help
```

Full end-to-end replay needs the actual frozen cohort and scene assets. This
source-only package does not pretend that synthetic examples reproduce the
paper's scene distribution or model scores.
