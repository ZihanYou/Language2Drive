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
