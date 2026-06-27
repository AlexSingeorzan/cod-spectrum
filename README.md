# cod-spectrum

Always-on, evidence-backed analytics for CDL/Challengers broadcasts. The v0 pipeline discovers or registers a VOD, samples its HUD, extracts a score timeline, infers Hardpoint scoring-flow changes, computes an xMWP series, generates clips and reports, and stores the result behind a FastAPI service.

Every stored event is schema-validated with a timestamp, confidence, and evidence crop. CPU-only processing is the default. The included fixture is fully offline and deterministic.

## Roadmap and the universal event schema

The platform's direction and every planned subsystem are mapped in
[`docs/ROADMAP.md`](docs/ROADMAP.md). Its foundation is the **universal event
schema** ([`docs/EVENT_SCHEMA.md`](docs/EVENT_SCHEMA.md), `backend/app/events/`):
one `GameEvent` envelope carrying provenance, evidence, and confidence, wrapping a
typed payload, with **detected facts kept structurally separate from derived
insights** — an insight must cite the facts it came from, and a fact must carry
evidence. The pipeline emits these events: they persist to the `game_events` table
and the reports, dashboard, and API all read them (Phase 2, with byte-for-byte
report parity against the retired flat `events` table). See a real sample stream:

```bash
.venv/bin/python scripts/sample_events.py   # writes data/fixtures/sample_events.jsonl + an evaluation report
```

## Reproduce the vertical slice (under five minutes)

Prerequisites: Python 3.12 (3.11+ supported) and FFmpeg on `PATH`.

```bash
make setup
make fixture
make scorebar-ocr-dataset
make scorebar-ocr-eval
.venv/bin/python -m backend.app.workers.processor \
  --file data/videos/sample_hardpoint.mp4 \
  --hud-profile CDL_2026_1080p
make test
```

Use `make setup PYTHON_BOOTSTRAP=python3.11` when that is the installed interpreter.

The processor writes evidence crops to `data/crops/`, clips to `data/clips/`, JSON/Markdown/HTML to `data/reports/`, and structured rows to `cod_spectrum.db`. Running the same command again returns the existing report without duplicating events or clips.

Start the API and dashboard:

```bash
make serve
# dashboard: http://127.0.0.1:8000/
# OpenAPI:   http://127.0.0.1:8000/docs
```

## Always-on source monitoring

Edit `data/configs/sources.yaml`, then run:

```bash
make scheduler
# deterministic one-cycle check:
.venv/bin/python -m backend.app.workers.scheduler --once
```

The scheduler uses yt-dlp metadata extraction to discover recent entries and deduplicates on `(platform, video_id)`. A source with `download: false` is reference-only unless it points to a local file. Setting `download: true` is an explicit opt-in: the scheduler logs the action, rate-limits requests, downloads, and queues processing. See [LEGAL.md](LEGAL.md).

One-shot online processing is also an explicit download action:

```bash
.venv/bin/python -m backend.app.workers.processor \
  --url "https://www.youtube.com/watch?v=..." \
  --hud-profile CDL_2026_1080p \
  --ocr-engine tesseract \
  --mode hardpoint
```

## Pipeline and data model

Stages are `ingest → sample_extract → timeline → analytics → clips → report → store`. `processing_jobs` records each stage, attempts, logs, and retry time. Broadcast states are `discovered → downloading → downloaded → processing → processed`, with audited `failed` and `skipped` branches. Legal transitions are enforced.

The hierarchy is:

```text
source
└── broadcast (one VOD)
    ├── processing_jobs
    ├── match (one series)
    │   └── map (one game: hardpoint/snd/control/unknown)
    │       ├── game_events  (universal GameEvent envelopes)
    │       ├── clips
    │       └── model_outputs
    └── report
```

Tables: `sources`, `broadcasts`, `processing_jobs`, `matches`, `maps`, `game_events`, `clips`, `reports`, and `model_outputs`. Events are stored as universal `GameEvent` envelopes (see [`docs/EVENT_SCHEMA.md`](docs/EVENT_SCHEMA.md)). SQLite is the default; set `COD_SPECTRUM_DATABASE_URL` to a SQLAlchemy Postgres URL without changing application code.

## What is real and what is stubbed

- Real: fractional HUD cropping, sampling cadence, crop-change gating, score-event construction, map boundary heuristics, lead changes, scoring-flow break/retake inference with debounce, xMWP HeuristicV0, evidence persistence, FFmpeg clips, reports, scheduler, retry records, API, and dashboard.
- Stubbed: bundled scorebar OCR reads `data/fixtures/sample_scores.json` and marks its dependent events `is_placeholder=true`. It exists to make the complete flow deterministic and offline.
- Evaluated scorebar OCR baseline: `--ocr-engine cdl` uses `CdlScorebarOcrEngine`, a CPU k-NN digit-gallery model trained from human-verified LAT/VAN scorebar crops. It is versioned (`0.1.0-knn`) and confidence-capped by leave-one-crop-out evaluation. Current result: 21/21 operational gallery self-check, but only 10/21 leave-one-out exact score matches (`0.4762`) and 11/21 with temporal decoding (`0.5238`). This is not production-ready OCR.
- **Scoreboard kill counter (Phase 4, the kill spine)** — `PanelKillCounter` (`panel_kill_counter@0.1.0`) reads each player's running kills/deaths from the top team panels and emits `KillEvent` (attacker) + `DeathEvent` (victim) facts from **monotonic** increments. Scored against the **human-verified post-game card** it is **exact: 8/8 players, 0.0 mean kill error, team totals 106/79, and the 505 s checkpoint 73/61** (`make panel-eval`, offline from the cached readings). Tesseract reads the clean panel font reliably. This is the authoritative kill count + who; see **Scoreboard kill counter** below.
- Killfeed detection/content baseline (Phase 4): `KillfeedDetector` (`killfeed_classical@0.1.0`) localises kill-notification rows and a positional tracker collapses flicker into candidate kill onsets — `KillEvent` facts with evidence, confidence, and `identity_unread`. It is the **corroboration layer**, not the kill-count source: measured against panel-counter ground truth it runs at **~56% precision / ~80% recall**. `KillfeedSegmenter` (`killfeed_segmenter_classical@0.1.0`) now separates row crops into attacker/weapon-icon/victim/headshot evidence regions when the layout is clear; current readiness is **120/245** rows with all core boxes. `KillfeedContentReader` (`killfeed_content_knn@0.1.0`) is wired to train from labelled row crops and emit `KillEvent`/`DeathEvent`/kill-type `WeaponEvent`/`TradeEvent`, but the real LAT/VAN scaffold still has **0 content-labelled rows**, so it makes **no real content accuracy claim** yet. See **Killfeed detection and content** below.
- Kill-type recognition scaffold (Phase 5): `KillTypeRecognizer` compares independent killfeed icon crops with template and histogram nearest-neighbour baselines. The real dataset at `data/kill_type_dataset/` contains **120** icon crops and **0** labels, so `make kill-type-eval` reports **no accuracy claim**. The synthetic fixture proves `kill_type` event emission only. See **Kill-type recognition** below.
- Optional OCR experiment: `TesseractOcrEngine` is wired behind `--ocr-engine tesseract`. Install the Tesseract binary and run `.venv/bin/pip install -r requirements-ocr.txt`, then calibrate the scorebar profile before trusting output. It is intentionally not in the base environment.
- Deferred: real-labelled killfeed content accuracy, transition-card/mode classification, deep SnD/Control analytics, and minimap object detection.

No analytics are silently invented. Unknown-mode files skip Hardpoint break/retake and xMWP analytics. The sample is identified as Hardpoint by its filename; production runs should pass `--mode hardpoint` only when known or add a mode detector.

## HUD calibration and OCR

HUD profiles live in `data/configs/hud_profiles/` and use fractional coordinates, so one profile scales across resolutions. `CDL_2026_1080p` and `generic_1080p` are included.

```bash
.venv/bin/python -m backend.app.workers.processor \
  --file /path/to/local.mp4 \
  --hud-profile CDL_2026_1080p \
  --debug-crops
```

Inspect `data/crops/broadcast_<id>/debug/`, tune only the JSON region fractions, and rerun on a fresh/local test database. For the current `cdl` OCR baseline, the scorebar crop must contain the full broadcast scorebar geometry represented by `data/fixtures/scorebar_ocr/lat_van_hp`; the default synthetic fixture remains stub-backed. Label representative crops across teams, maps, resolutions, compression levels, and overlay revisions, then measure digit/score accuracy and temporal stability before enabling unattended processing. Expand next to match/hill timers, then killfeed rows.

Rebuild and evaluate the current scorebar OCR baseline:

```bash
make scorebar-ocr-dataset
make scorebar-ocr-eval
```

## Killfeed detection and content

`KillfeedDetector` is a classical, training-free localiser for the broadcast killfeed
(`attacker → weapon → victim` rows). It is honest about the split between what classical
CV can do and what needs a labelled model:

- **Detects** kill-notification rows in the verified killfeed region and tracks them
  positionally across frames into candidate **kill onsets** (timing/count), each a
  `KillEvent` fact with evidence + confidence and the tag `identity_unread`. The feed is
  semi-transparent over a moving scene, so per-frame detection flickers; positional
  tracking (not content-hashing) is what makes onset counting robust. It is a baseline —
  it over- and under-counts, which is why every onset is a labelable candidate, not a
  claimed kill.
- `KillfeedContentReader` is the label-trained content layer. It reads attacker,
  victim, kill_type, optional exact weapon, headshot, and trade fields from
  labelled row crops and expands them into `KillEvent`, `DeathEvent`, kill-type
  `WeaponEvent`, and `TradeEvent`. With no labels it abstains; it never fills
  names, kill types, or weapons from empty annotation slots.
- `KillfeedSegmenter` is the Stage B field splitter. It creates separate crop evidence
  for attacker text, weapon icon, victim text, and optional indicators. If the layout
  is not clear it returns `null` for that field instead of using fixed-layout guesses.

```bash
make killfeed-dataset   # build the annotation scaffold from the local VOD (2 fps)
make killfeed-eval      # detection precision/recall once annotations.jsonl is labelled
make killfeed-segments
make killfeed-segment-eval
make killfeed-segment-sample
make killfeed-content-eval
make killfeed-sample    # deterministic synthetic candidate-KillEvent stream
make killfeed-content-sample
```

The committed dataset (`data/killfeed_dataset/`, 245 unlabelled candidates from the
LAT/VAN Hardpoint) ships with empty label slots and a labelling guide in its README;
no kill identities are invented. `eval_killfeed.py` prints "no accuracy claim" until a
person labels `valid_kill` and adds missed kills as `detector="manual_added"`.
`eval_killfeed_content.py` separately reports `0` content-labelled rows and no
content-reader accuracy until attacker/victim/kill_type labels exist. The synthetic
fixture at `data/fixtures/killfeed_content_sample/` is only an event-contract sample,
not real broadcast accuracy. `eval_killfeed_segments.py` reports Stage B readiness:
`120/245` real row crops currently have attacker+weapon+victim segment boxes, which
is enough to start kill-type icon dataset work but not enough to claim OCR or classifier
accuracy. The Phase 5 recognizer design is in
[`docs/KILL_TYPE_RECOGNITION_DESIGN.md`](docs/KILL_TYPE_RECOGNITION_DESIGN.md).

## Kill-type recognition

`KillTypeRecognizer` is independent from player-name OCR. It consumes only
killfeed icon crops from Stage B segmentation and returns `kill_type=null` when
labels are missing or confidence is too low. It classifies coarse categories:
`gun`, `grenade`, `melee`, `fall_damage`, `suicide`, `environment`, `objective`,
`killstreak`, and `unknown`. Exact weapon names are optional future metadata;
downstream analytics consume `kill_type`.

Two CPU baselines are wired for comparison:

- `kill_type_icon_template_nn@0.2.0`
- `kill_type_icon_histogram_nn@0.2.0`

```bash
make kill-type-dataset  # build data/kill_type_dataset from Stage B segment crops
make kill-type-eval     # real dataset readiness; no accuracy claim until labelled
make kill-type-review   # validate/summarize annotation readiness
make kill-type-contact-sheet
make kill-type-sample   # synthetic kill_type event contract sample
```

Current real status: `120` icon crops, `0` labelled kill-type classes, no real
accuracy claim. The synthetic sample writes
`data/fixtures/kill_type_recognition_sample/` and demonstrates the `kill_type`
payload path with `weapon=null`.

Use `scripts/review_kill_type_dataset.py` for safe labelling instead of editing
JSONL directly. It can validate labels, list unreviewed crops, apply one reviewed
label, and write a contact sheet for visual review.

## Scoreboard kill counter

`PanelKillCounter` is the **kill spine**. The broadcast's top team panels show each
player's running kills/deaths — a clean, **monotonic** counter — so it answers *how
many kills, and who* with far more reliability than the semi-transparent feed:

- a player's `kills +N` → N `KillEvent`s by that player (attacker); `deaths +1` → a
  `DeathEvent` (victim). A kill and a death in the same step on opposite teams are paired.
- Reads use Tesseract (the clean panel font, not the stylised scorebar). The monotonic
  constraint is a strong error-corrector: a read is accepted only when it does not
  decrease, and an implausible jump must be confirmed by a second frame — a one-frame
  OCR glitch cannot invent a kill. It never counts kills from before it starts watching.

```bash
make panel-counter   # OCR both panels over the local VOD -> readings cache + events
make panel-eval      # score vs the verified post-game card + reconcile with the killfeed
```

Measured against the human-verified post-game card (`PLAYER_MAP_STATS`): **0.0 mean
kill error, 8/8 players exact, team totals 106/79, 505 s checkpoint 73/61.** The slow
OCR is cached to `data/panel_counter/readings.jsonl`, so `make panel-eval` re-scores
offline. The counter is also the ground truth that measures the killfeed detector
(above) at ~56% precision / ~80% recall via `reconcile_with_killfeed`.

## YOLO minimap next step

Collect and label minimap crops with player-arrow/object boxes, team/color, frame timestamp, HUD profile, and map. Broadcast minimaps usually expose the observed team’s information rather than a complete neutral ground truth, so training and outputs must preserve an `observed_team`/visibility field and must not infer hidden opponents. A future `MinimapDetector` interface should mirror `OcrEngine`: crop through the active HUD profile, return boxes with confidence and evidence, store raw results in `model_outputs`, and let a separate temporal service derive map events. GPU acceleration remains optional behind configuration.

## API

- `GET/POST /sources`
- `GET /broadcasts`, `GET /broadcasts/{id}`
- `POST /broadcasts/{id}/process`
- `GET /reports`, `GET /reports/{id}`
- `GET /clips/{id}`
- `GET /healthz`

## Current limitations

The sample’s OCR values are fixture-backed. Map boundaries rely on score reset/250 rather than transition cards. xMWP is an uncalibrated heuristic, and possible breaks/retakes are scoring-flow evidence—not kill or hill-control confirmation. Deep analytics are Hardpoint-only. Live platform behavior is not part of offline CI and can change independently. Downloading may violate platform terms or content rights; it remains disabled by default and is the operator’s responsibility.
