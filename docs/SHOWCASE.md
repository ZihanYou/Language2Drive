# Project film and interactive evidence

The project preview combines the author's framework artwork, a 133-second English project film,
and 129 curated viewing copies and a six-map simulator atlas from author-provided evidence archives.
The local preview has no upload service. The site and all viewing assets are ready
for the automatic GitHub Pages workflow; see [the deployment guide](RELEASE.md).

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
- `/#simulator`: six custom CARLA maps, clickable locations with overview / road-forward / road-reverse photos, four tours per map, zoom / drag / keyboard navigation and synchronized camera routes.
- `/submit.html`: local result packet builder and prefilled GitHub issue.

The preview binds to loopback and supports byte-range requests for video seeking.
Animations respect reduced-motion preferences; the tour stops when it leaves
the viewport or the tab becomes hidden. Videos start on user interaction, and
starting one player pauses the other.

## Viewing-copy inventory

| Backend | Clips | Selection and source |
|---|---:|---|
| CARLA | 86 | All 16 formal task categories, all four scopes and 10 maps represented in the supplied archive; 20 saved model executions with camera pose re-renders, plus 33 target pairs in two camera views (66 clips) |
| RealSim | 14 | Eight Alpamayo 1.5 examples and six privileged oracle examples, covering six task families |
| nuScenes | 20 | Four PCUs across scenes/locations, four models plus reference witnesses per PCU |
| NAVSIM | 7 | Selected task families; reference videos plus saved predictions from five models in a separate inspector |
| WOD-E2E | 2 | One Stop / Go and one Speed Choice benchmark target, from the October 12 target archive |

The inventory contains 54 backend/PCU combinations. Selection emphasizes task,
scene and model coverage; it is not an estimate of cohort performance. CARLA
debug runs are excluded. Formal CARLA cameras were re-rendered from saved poses;
the native model inputs were not retained. No inference was rerun for this page.
RealSim oracle runs illustrate feasibility and are labeled separately from models.
Open-loop recorded camera footage remains distinct from predicted trajectories.
WOD videos display two paired frozen P0 contract-defined targets: Stop / Go
and Speed Choice. The target type and source hashes
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

Rebuild the 33 CARLA target pairs and six-map atlas from the October 13 archive:

```sh
python scripts/build_simulator.py \
  --archive /path/to/benchmark_target_exploration_20261013.zip \
  --workers 4
```

Use `--targets-only` or `--maps-only` for a partial rebuild. The complete gallery
builder also accepts `--carla-targets /path/to/benchmark_target_exploration_20261013.zip`.
Targets combine the two independent arm videos in the existing 1080p A/B design;
exterior and front views are independently selectable. Reference paths come from
the archived 30 Hz target trajectories. Source member hashes cover both arms.

The atlas reads the six exported CARLA lane graphs in their native world frame;
it does not convert OpenDRIVE coordinates. It contains 54 locations and 162
web-optimized photos, plus 24 full-length 45-second H.264 tours (two map tours
and two road-level drives per map). The map cursor follows the recorded camera
trajectory, with road-level viewpoint switches treated as cuts. `data/maps.json`
records source members, hashes and the measured geometry. Map thumbnails are
SVG exports of the same lane samples. Original frame ZIPs and task contracts
remain in the archive. This package includes map viewing media and geometry;
the simulator's cooked map binaries are installed separately through the
[CARLA runtime setup](INSTALLATION.md).

Every clip carries one of two explicit roles: **Model result** (executed model
trajectories or saved forecasts) and **Benchmark target** (reference trajectories
or reference-controller examples, including WOD contract-defined targets). Blue/orange identify instruction arms;
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

## Published project film

The homepage plays the author-provided **Language2Drive-en-v8.7.mp4**:
1920 × 1080, 30 fps, 133 seconds, H.264 video and AAC audio. The video and audio
streams are preserved; MP4 `faststart` places playback metadata at the beginning
of the file for quicker browser loading. The cover is a frame from this version.

Published files under `assets/demo/`:

- `language2drive-film.mp4`: the web-ready English film.
- `chapters.vtt`: chapter navigation aligned with the new edit.
- `poster.webp`: the updated film cover.
- `storyboard.json`: source filename and SHA-256, published video checksum,
  format metadata and chapter timing.

Chapters: 00:00 (question), 00:23 (PCU), 00:31 (four scopes), 00:41 (paired turns),
00:55 (scene-grounded parking), 01:05 (long-horizon programs), 01:15 (benchmark
coverage), 01:30 (main findings), and 01:54 (explore Language2Drive).
The player and download link use a version query to refresh cached copies.

## Render the earlier editable film

The existing renderer remains available for the earlier 120-second composition.
Its output goes to `assets/demo/generated/` so it can be developed independently
of the current homepage film.

```sh
python -m pip install Pillow
python scripts/build_demo.py --work-dir /tmp/l2d-film --force
```

The renderer uses Pillow for title cards and vector motion, and ffmpeg for case
excerpts. Its output includes the earlier film, text captions, cover and source
storyboard. `--output-dir` selects another destination; `--refresh-cases`
rebuilds only its case excerpts.

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
