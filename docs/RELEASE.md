# Maintainer release guide

## GitHub Pages

The site is a dependency-free static page in `docs/`. It uses local assets, accessible result tabs, sortable manuscript tables, an explicitly conceptual PCU illustration and clipboard buttons. Local preview:

```sh
python -m http.server 8000 --directory docs
# Open http://localhost:8000
python scripts/check_site.py
# Optional real-browser verification (requires Playwright + Chromium)
python scripts/check_browser.py
```

In repository **Settings → Pages**, choose **GitHub Actions** as the source. Push the page changes to the default branch, or manually run **Project page**. The workflow validates local assets, uploads `docs/` and deploys it to `https://ZihanYou.github.io/Language2Drive/`. Relative asset paths work under the repository subpath. If the repository owner/name changes, update project URLs and social metadata. This change does not itself claim a live deployment.

## Build the wheel

```sh
python -m pip install build
python scripts/build_release.py
python scripts/check_wheel.py dist/language2drive-0.1.0-py3-none-any.whl
```

Use the release builder rather than bare `python -m build`: it stages legacy backends, generates a per-file SHA256 inventory, retains licenses and avoids bundling model weights or external model implementation trees. A normal editable install remains lightweight. The release builder uses tracked backend files plus the new unified driver template; commit future backend changes before creating a release. The SDK and backends are physically separate to avoid the existing CARLA `language2drive` module-name collision.

Artifacts are a `py3-none-any` wheel and its SHA256 file. GitHub tags `v*` run tests, build and attach artifacts to a release; keep the tag consistent with `pyproject.toml` and `l2d_benchmark.__version__`.

## PyPI

Package metadata and the `language2drive` CLI are ready. Public PyPI installation is **pending publication**. Check namespace availability and configure a PyPI Trusted Publisher with this repository, `pypi.yml` and the `pypi` environment. Set any desired maintainer review controls on that environment. Run **Publish PyPI** manually after validation. It builds and checks the wheel before the authenticated publishing job; no token is stored in the repository.

Once successfully published, `pip install language2drive` and optional extras work from the package index. Before publication, install the local/release wheel or checkout. Do not replace the pending status with an availability claim until installation from the public index has been verified.

## Paper and data metadata

The supplied manuscript has anonymous author metadata. Keep the provisional citation until authors and archival URL are finalized; do not invent authors, venue acceptance or an oral designation. Code-license ownership is not a substitute for paper authorship. The current PDF and paper figures are copied into `docs/assets/`; their provenance and source-table hashes are in `docs/data/provenance.json`.

Replace the Dataset coming-soon state when the intended public release and its immutable manifests are actually available. Update access rules, resource bindings, fixture checks and reproduction commands together. Licenses of third-party code and upstream datasets remain separate.

## Organization references

The information hierarchy was informed by the primary project repositories of [WorldLens](https://github.com/worldbench/WorldLens), [PAI-Bench](https://github.com/SHI-Labs/physical-ai-bench), [RoboSpatial](https://github.com/NVlabs/RoboSpatial), [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench), [COMFORT](https://github.com/sled-group/COMFORT) and the [RefAV project](https://cainand.github.io/RefAV/). The page uses its own editorial design and PCU interaction. References do not imply affiliation or a shared evaluation protocol.
