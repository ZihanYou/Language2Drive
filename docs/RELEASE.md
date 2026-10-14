# Maintainer release guide

## GitHub Pages

Public project page: [ZihanYou.github.io/Language2Drive](https://ZihanYou.github.io/Language2Drive/).

The site is a dependency-free static page in `docs/`. It includes 129 case videos across five backends, a 120-second English-captioned film, a six-map CARLA simulator atlas with 54 locations and 24 exploration videos, four interactive CARLA Scope diagrams, manuscript tables and a result submission builder. Local preview:

```sh
python scripts/serve_preview.py --port 18765
# Open http://localhost:18765
python scripts/check_site.py
# Optional real-browser verification (requires Playwright + Chromium)
python scripts/check_browser.py
```

The latest site and its media are checked into the repository, ready for deployment. Pushing changes to `docs/` on `main` validates and deploys the site through the Pages workflow. Manual deployment remains available. GitHub Pages must first be enabled with GitHub Actions as its source in the repository settings.

The leaderboard includes an interactive metric explorer (`metrics.js` / `metrics.css`) for all five tracks. Its examples explain strict and relaxed success, exclusive open-loop semantics, graded compliance, driving metrics and the L2D formula. Editing an example never changes the published result data. `scripts/check_browser.py` verifies the paired rules, aggregation formula, all reported closed-loop L2D rows, keyboard controls and mobile layout.

Official model papers are maintained in `docs/data/models.json`, shared by the leaderboard and case library through `models.js`. Aliases map displayed checkpoint names to their papers. Alpamayo entries distinguish the family paper from version-specific official model cards. When adding a model, update the registry and README table links; `scripts/check_site.py` checks coverage across every reported model and case.

1. In repository **Settings → Pages → Build and deployment**, select **GitHub Actions** as the source.
2. Open [Actions → Project page](https://github.com/ZihanYou/Language2Drive/actions/workflows/pages.yml), select **Run workflow**, and choose `main`.
3. Wait for the `github-pages` deployment to finish. Its environment URL is the published project page: `https://ZihanYou.github.io/Language2Drive/`.
4. Check the film, all five backend filters, Scope A–D animations and `submit.html` at the published address.

The workflow checks page assets before uploading `docs/`; no media rendering or simulator installation is needed during deployment. Assets use relative paths for the repository subpath. The result submission builder prepares a local packet and a GitHub issue draft. If the repository owner/name changes, update project URLs and social metadata. See [SHOWCASE.md](SHOWCASE.md) for media provenance, reproduction and local browser checks.

## Build the wheel

```sh
python -m pip install build
python scripts/build_release.py
python scripts/check_wheel.py dist/language2drive-0.1.0-py3-none-any.whl
```

Use the release builder rather than bare `python -m build`: it stages legacy backends, generates a per-file SHA256 inventory, retains licenses and includes adapter examples and Markdown guides while excluding website media, model weights and external model implementation trees. A normal editable install remains lightweight. The release builder uses tracked backend files plus the new unified driver template; commit future backend changes before creating a release. The SDK and backends are physically separate to avoid the existing CARLA `language2drive` module-name collision.

Artifacts are a `py3-none-any` wheel and its SHA256 file. GitHub tags `v*` run tests, build and attach artifacts to a release; keep the tag consistent with `pyproject.toml` and `l2d_benchmark.__version__`.

The initial wheel is published in [GitHub Release v0.1.0](https://github.com/ZihanYou/Language2Drive/releases/tag/v0.1.0). Install it directly from the release URL shown in [INSTALLATION.md](INSTALLATION.md).

## PyPI

[Language2Drive 0.1.0 is published on PyPI](https://pypi.org/project/language2drive/0.1.0/). Install the SDK with `pip install language2drive`; optional extras are `[openloop]`, `[carla]` and `[realsim]`. The GitHub Release wheel remains an alternative installation source.

The repository uses a PyPI Trusted Publisher with `pypi.yml` and the `pypi` environment. Version tags (`v*`) trigger **Publish PyPI**; manual publishing is also available after validation. It builds and checks the wheel before the authenticated publishing job; no token is stored in the repository.

When setting up a replacement repository, add a [Pending Trusted Publisher](https://pypi.org/manage/account/publishing/) with these exact values:

| Field | Value |
|---|---|
| PyPI project name | `language2drive` |
| GitHub owner | `ZihanYou` |
| Repository name | `Language2Drive` |
| Workflow filename | `pypi.yml` |
| Environment name | `pypi` |

The [0.1.0 publishing run](https://github.com/ZihanYou/Language2Drive/actions/runs/38504738065) completed successfully after the publisher was configured. For transient publishing failures, use **Re-run failed jobs** to reuse the verified build artifact.

## Paper and data metadata

The homepage, README and dataset card use the 22 paper authors and 19 displayed affiliations, recorded in `docs/data/paper.json`, with the requested Project Lead credit and affiliation selection. The current PDF and paper figures are copied into `docs/assets/`. The paper is published at [arXiv:2610.16634](https://arxiv.org/abs/2610.16634). Source-table hashes are in `docs/data/provenance.json`; framework asset provenance is in `docs/assets/framework/provenance.json`.

## Hugging Face dataset card

The published [dataset homepage](https://huggingface.co/datasets/Mike087/Language2Drive) is maintained from [huggingface/README.md](../huggingface/README.md). Its authors, cohort descriptions and training-free baseline tables follow the GitHub README. Keep download links aligned with the resource directories actually present on Hugging Face, including the historical `langauge2drive.RealSim` prefix. Publish this file as the dataset repository's root `README.md`; retain its component-specific license metadata.

Replace the Dataset coming-soon state when the intended public release and its immutable manifests are actually available. Update access rules, resource bindings, fixture checks and reproduction commands together. Licenses of third-party code and upstream datasets remain separate.

## Organization references

The information hierarchy was informed by the primary project repositories of [WorldLens](https://github.com/worldbench/WorldLens), [PAI-Bench](https://github.com/SHI-Labs/physical-ai-bench), [RoboSpatial](https://github.com/NVlabs/RoboSpatial), [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench), [COMFORT](https://github.com/sled-group/COMFORT) and the [RefAV project](https://cainand.github.io/RefAV/). The page uses its own editorial design and PCU interaction. References do not imply affiliation or a shared evaluation protocol.
