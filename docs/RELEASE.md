# Maintainer release guide

## GitHub Pages

The site is a dependency-free static page in `docs/`. It includes 73 case videos across five backends, a 120-second English-captioned film, four interactive CARLA Scope diagrams, manuscript tables and a result submission builder. Local preview:

```sh
python scripts/serve_preview.py --port 18765
# Open http://localhost:18765
python scripts/check_site.py
# Optional real-browser verification (requires Playwright + Chromium)
python scripts/check_browser.py
```

The latest site and its media are checked into the repository, ready for deployment. Pushing `main` runs validation; the Pages workflow remains manual-only.

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

Use the release builder rather than bare `python -m build`: it stages legacy backends, generates a per-file SHA256 inventory, retains licenses and avoids bundling model weights or external model implementation trees. A normal editable install remains lightweight. The release builder uses tracked backend files plus the new unified driver template; commit future backend changes before creating a release. The SDK and backends are physically separate to avoid the existing CARLA `language2drive` module-name collision.

Artifacts are a `py3-none-any` wheel and its SHA256 file. GitHub tags `v*` run tests, build and attach artifacts to a release; keep the tag consistent with `pyproject.toml` and `l2d_benchmark.__version__`.

## PyPI

Package metadata and the `language2drive` CLI are ready. Public PyPI installation is **pending publication**. Check namespace availability and configure a PyPI Trusted Publisher with this repository, `pypi.yml` and the `pypi` environment. Set any desired maintainer review controls on that environment. Run **Publish PyPI** manually after validation. It builds and checks the wheel before the authenticated publishing job; no token is stored in the repository.

Once successfully published, `pip install language2drive` and optional extras work from the package index. Before publication, install the local/release wheel or checkout. Do not replace the pending status with an availability claim until installation from the public index has been verified.

## Paper and data metadata

The homepage and README use the 22 authors and 20 affiliations from the supplied `Language2Drive_arxiv.pdf`, recorded in `docs/data/paper.json`. The current PDF and paper figures are copied into `docs/assets/`. The archival arXiv identifier remains pending. Source-table hashes are in `docs/data/provenance.json`; framework asset provenance is in `docs/assets/framework/provenance.json`.

Replace the Dataset coming-soon state when the intended public release and its immutable manifests are actually available. Update access rules, resource bindings, fixture checks and reproduction commands together. Licenses of third-party code and upstream datasets remain separate.

## Organization references

The information hierarchy was informed by the primary project repositories of [WorldLens](https://github.com/worldbench/WorldLens), [PAI-Bench](https://github.com/SHI-Labs/physical-ai-bench), [RoboSpatial](https://github.com/NVlabs/RoboSpatial), [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench), [COMFORT](https://github.com/sled-group/COMFORT) and the [RefAV project](https://cainand.github.io/RefAV/). The page uses its own editorial design and PCU interaction. References do not imply affiliation or a shared evaluation protocol.
