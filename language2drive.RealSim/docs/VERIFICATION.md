# Package verification

Verification performed on this source snapshot:

| Check | Result |
|---|---|
| Direct identity, endpoint and credential marker scan | No findings |
| Isolated Python 3.12 test environment without simulator packages | 346 passed, 1 optional module skipped |
| Clean Python 3.13 one-command deployment | Passed using the configured mirror, no inherited proxy |
| Repeated deployment with installed dependencies in offline mode | Passed |
| Paper Table 2 component recomputation | All eight model rows reproduced from their selected existing traces, within displayed rounding precision |
| Offline wheel build | Passed, no dependency downloads |
| Installed-wheel CLI imports and three-task synthetic demo | Passed in isolated Python 3.13 interpreter |
| Complete synthetic 30-PCU scoring entry point | Passed, all 60 slots covered |
| Main and release paper scorer | Identical implementation and metric regression tests |

The skipped module requires the external AlpaSim protobuf package. Paper-profile
tests cover hierarchical weighting, both individual/paired identities, zero
credit for unsuccessful scheduled slots, arm-specific route horizons and the
L2D formula. Existing versioned scorer tests remain in the suite. The deployment
script is also exercised for argument parsing, shell syntax and safe environment
selection. The selected experimental records and raw traces are not embedded
in the source distribution.

The archive contains only the manifest-listed UTF-8 source/configuration/docs.
ZIP entries have fixed timestamps and permissions, relative paths and empty
extra/comment fields; original filesystem ownership and modification times are
not copied. No raw experimental records or third-party binary assets are bundled.

These are software/package checks, not new model experiments or a guarantee
against identification through the research content itself.

## Core assets and runtime setup update

- The core bundle was uploaded to the private `Mike087/Language2Drive` dataset,
  under `langauge2drive.RealSim/`, commit
  `51867e4a15be8146910dbe3975d95a2e66c2de28`: 14 files, 9,150,460 bytes.
  It contains 30 PCUs, references to 23 scenes, 60 execution slots at seed 1701,
  and seven runtime groups. No Python environment archive was uploaded.
- The updated source passed all 374 tests in a Python 3.12 environment with
  the upstream simulator protobuf package available.
- Runtime import checks passed in the existing pinned upstream environment,
  including runtime/driver/controller/physics, the native Rust extension,
  Alpamayo imports, driver registration and the CasADi compatibility overlay.
- `setup_runtime.sh` creates a separate environment rather than copying that
  existing environment. The fixed resolution is supplied as
  `config/alpasim-runtime.uv.lock`, because the upstream Git repository does not
  track `uv.lock`. Git LFS assets are not automatically downloaded during setup.
- Environment checks do not constitute a new GPU closed-loop replay. A fresh
  installation can still require network access for uncached dependencies;
  its result is recorded locally in `setup-result.json`.
- A new isolated installation successfully bootstrapped Rust and checked out
  the pinned source, but stopped when direct GitHub access timed out fetching
  the locked `microsoft/mup` commit. This was a failed installation attempt,
  not a fresh-environment pass. Use reachable official endpoints or complete
  local caches; the existing-environment import pass above is a separate check.
