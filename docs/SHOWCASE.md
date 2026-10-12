# Project film and interactive evidence

The project preview combines the author's framework artwork, a 120-second film,
and 72 curated viewing copies from five author-provided evidence archives.
The local preview has no upload service. The site and all viewing assets are ready
for the manual GitHub Pages workflow; see [the deployment guide](RELEASE.md).

## Preview

```sh
# Run from the repository on the server.
python scripts/serve_preview.py --port 18765

# Run on your laptop, using your usual SSH host or alias.
ssh -N -L 18765:127.0.0.1:18765 youzh2507@server41
```

Open `http://localhost:18765/`. Direct entry points:

- `/#film`: English-captioned project film, chapter navigation and MP4 download.
- `/#framework`: guided framework tour, instruction selection and scope exploration.
- `/#examples`: backend/model/evidence filters, text search, same-PCU model switch,
  playback speed, case permalinks, downloads and source details.
- `/submit.html`: local result packet builder and prefilled GitHub issue.

The preview binds to loopback and supports byte-range requests for video seeking.
Animations respect reduced-motion preferences; the tour stops when it leaves
the viewport or the tab becomes hidden. Videos start on user interaction, and
starting one player pauses the other.

## Viewing-copy inventory

| Backend | Clips | Selection and source |
|---|---:|---|
| CARLA | 20 | All 16 formal task categories, all four scopes and 10 maps represented in the supplied archive; saved execution with camera pose re-renders |
| RealSim | 14 | Eight Alpamayo 1.5 examples and six privileged oracle examples, covering six task families |
| nuScenes | 20 | Four PCUs across scenes/locations, four models plus reference witnesses per PCU |
| NAVSIM | 7 | Selected task families; reference videos plus saved predictions from five models in a separate inspector |
| WOD-E2E | 11 | Paired benchmark targets from the October 12 target archive; five task families and ten scene clusters |

The inventory contains 50 backend/PCU combinations. Selection emphasizes task,
scene and model coverage; it is not an estimate of cohort performance. CARLA
debug runs are excluded. Formal CARLA cameras were re-rendered from saved poses;
the native model inputs were not retained. No inference was rerun for this page.
RealSim oracle runs illustrate feasibility and are labeled separately from models.
Open-loop recorded camera footage remains distinct from predicted trajectories.
WOD videos display paired benchmark targets: six instruction-defined schematics
and five frozen P0 contract-defined targets. The target type and source hashes
are retained in the downloadable case metadata.

The website shows allowlisted display metadata, selected viewing copies and
source/viewing-copy SHA-256 hashes. Hidden raw contracts, sensor bundles, credentials
and absolute source-machine paths are not exported. Viewing materials retain the
original authors', datasets' and models' terms; the repository's code license does
not replace them.

## Rebuild the gallery

Requires Python 3, Pillow, `ffmpeg` with libx264/libwebp, and `ffprobe`. The author archives
are local inputs, not resources fetched by visiting the website.

```sh
# Put the CARLA archive at /tmp/l2d-showcase/carla.zip.
python scripts/build_showcase.py \
  --source-root /path/to/case_vedio \
  --work-dir /tmp/l2d-showcase \
  --include-carla --workers 3
```

Required local archive names are recorded in `data/cases.json`. Without
`--include-carla`, the script builds the four locally supplied backends. It
extracts only selected videos and reads metadata directly from each ZIP. Viewing
copies use a unified 1920 × 1080, 12 fps, H.264 presentation with fast start.
`render_case.py` sets the typography, A/B instruction cards, camera windows and
metric plots. Camera crops retain the source movie’s timing. Full trajectories
are redrawn from the archived numeric records, with identical bounds and equal
XY scale across both arms. Long instructions are measured before rendering.

Every clip carries one of two explicit roles: **Model result** (executed model
trajectories or saved forecasts) and **Benchmark target** (reference trajectories
or reference-controller examples, including WOD instruction schematics and
contract-defined targets). Blue/orange identify instruction arms;
dashed gray denotes targets alongside model paths. WOD target videos are copied
byte-for-byte from `WOD-E2E_Benchmark_Targets_2026-10-12.zip`.
Source and output hashes stay in the case manifest; the raw
archives remain unchanged. Change `PRESENTATION_VERSION` to rebuild all outputs.
Sources are cached in the work directory with archive/member identities. Changing
the source archive or member refreshes the clip and poster automatically.

All 20 CARLA and eight RealSim Alpamayo 1.5 cases include paired CoT panels.
`reasoning.js` displays the latest recorded output for each arm at the video's
current time; desktop viewers can watch video and CoT side by side or expand the
video, and full transcripts support seeking and JSON download. Text is copied
verbatim from the archived CSVs, with source member hashes and native timestamps
in `data/reasoning/`. CARLA uses the archived agent-clock origins and replay speed;
RealSim uses the first shared camera timestamp and preserves the final reading hold.
The gallery builder exports these files automatically. To update just the CoT:

```sh
python scripts/build_reasoning.py --source-root /path/to/case_vedio \
  --carla-archive /tmp/l2d-showcase/carla.zip
```

Framework artwork in `assets/framework/` is extracted from the author's
`Framework.pptx`. The RealSim tile uses a NuRec front-camera JPEG from the
Alpamayo P02 run; `provenance.json` records that archive member and its hash,
alongside the embedded-image mapping for the other artwork. The unmodified full author figure remains available on the page.
The central paired animation is a conceptual SVG diagram. The four Scope cards
and detailed A/B explorer use native CARLA geometry and benchmark reference
paths from A-001, B-009, C-025 and D-001. `scopes.js` draws the roads, target
regions and paths at equal XY scale; animation follows normalized route
progression. Its display-only geometry is reproducible with:

```sh
python scripts/build_scope_figures.py --archive /tmp/l2d-showcase/carla.zip
```

The scope explorer supports scrubbing, playback, reduced motion and direct links
to each PCU’s model result. Backend buttons show clip counts and clear stale
model, role and search filters on a backend change.

Author names, order, affiliation indices and contribution markers in
`data/paper.json` follow page 1 of `Language2Drive_arxiv.pdf`. The homepage uses
short affiliation names with full names in tooltips. The paper download is an
exact copy of that supplied PDF; the README and homepage citations use its author list.

## Rebuild the film

```sh
python -m pip install Pillow
python scripts/build_demo.py --work-dir /tmp/l2d-film --force
```

The renderer uses Pillow for editable title cards and vector motion, and ffmpeg
for the real video excerpts. All four Scope chapters animate the same native
CARLA geometry and reference paths used by the website, with explicit benchmark
target labels. Case excerpts retain the gallery's unified A/B layout and role.
Its font paths are declared at the top of the script
(DejaVu Sans/Mono and URW P052 Roman on this host). `--force` rebuilds every segment
after source or layout edits; `--refresh-cases` rebuilds just the video segments.

Outputs under `assets/demo/`:

- `language2drive-film.mp4`: 1920 × 1080, 24 fps, 120 s, H.264. English captions
  are burned into the frames. The film is silent and has no synthetic narration.
- `captions.vtt`: optional English text track.
- `poster.webp`: project film cover.
- `storyboard.json`: timecodes, captions, source case identifiers, evidence roles,
  excerpt offsets, source viewing-copy hashes and Scope geometry provenance.

Chapter times are 00:00 (question), 00:12 (PCU), 00:32 (scopes), 00:48
(evidence), and 01:44 (evaluate and contribute). Source timing is retained within
excerpts; CARLA replay speed is shown in its case frames. The film uses actual
evidence excerpts, native CARLA reference geometry and original framework artwork.

## Check the result

```sh
python scripts/check_site.py
# Requires Playwright and its Chromium installation.
python scripts/check_browser.py
ffmpeg -v error -i docs/assets/demo/language2drive-film.mp4 -f null -
```

Checks cover local links, manifest inventory, source labels, manuscript tables,
desktop/mobile layout, real browser video playback, model switching, trajectories,
range requests, reduced motion, and submission validation/download/import.
No issue is posted and no remote website is deployed by these checks.
