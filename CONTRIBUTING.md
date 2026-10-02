# Contributing

Keep the five legacy track entry points working. Add unified interfaces under `src/l2d_benchmark`, and document public observation fields, frames, time grids and native scorer profiles. Avoid exposing evaluator contracts or opposite-arm metadata to model adapters.

Before proposing SDK changes, run `python -m unittest discover -s tests -v`. For page changes, run `python scripts/check_site.py` and preview at desktop/mobile widths with `python -m http.server 8000 --directory docs`. For packaging changes, build with `scripts/build_release.py` and run `scripts/check_wheel.py` on the artifact.

Tests should distinguish interface fixtures from benchmark measurements. New leaderboard rows follow `docs/LEADERBOARD.md`; preserve actual values, failures and provenance. Keep third-party license notices with redistributed code. Do not add unverified paper author, venue or result claims.
