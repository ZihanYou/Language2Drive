# Release verification

The paper profile was validated in a newly created conda environment using the provided Tsinghua-mirror environment specification.

- 35 CPU tests passed, including binary contracts, temporal coverage, response direction, paired wording aggregation, invalid-output denominators, data hashes, asset installation and review submission/export.
- All four nuScenes Table 9 rows were recomputed from frozen predictions in the new environment and matched the paper's displayed values.
- For each model, 120 continuous aggregates (SCS, PCS, SRG, NBD and paired ADE over six conditions and four reporting groups), including counts and confidence intervals, matched the original frozen diagnostics after the declared scale conversion.
- Public inputs, evaluation contracts and maps retain their recorded SHA-256 identities. The download manifest pins the private dataset commit, archive checksum and all 11 asset files.
- Fresh local asset installation and repeat verification passed. Download transport supports authorized mirror/official HTTPS access, explicit proxy opt-in and an offline archive. No credential is embedded in source or data.
- The source release excludes model weights, raw sensor files, reviewer identities, private deployment paths and local experiment logs.

Run `python -m unittest discover -s tests -v` and `python scripts/reproduce_paper.py --check-paper --output-dir outputs/new_replay` to repeat the checks. Dataset read permission is required for the separate private replay assets.
