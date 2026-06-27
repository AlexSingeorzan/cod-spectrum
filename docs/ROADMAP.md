# COD Spectrum — Platform Roadmap

> The intelligence layer for Call of Duty. Breaking Point won the box score; we are
> building the layer that explains **why** matches are won and lost, backed by
> evidence and calibrated confidence at every step.

This document is the **Phase 0** deliverable: a complete, honest audit of the
repository and a map of every subsystem. Nothing in this repo is built before this
file exists, and every later phase is checked back against it.

---

## 1. North Star

A coach, analyst, broadcaster, or Activision should be able to ask:

> *"Why did OpTic lose P4?"*

and receive a timeline, clips, key kills, communications, rotations, spawn
analysis, pressure maps, decision mistakes, expected alternatives — **every claim
carrying its evidence and a confidence score.** No speculation, no magic numbers.

The product test for every feature is one question:

> **Would a professional coach make a better decision because this exists?**

If no, it does not get built.

---

## 2. Two kinds of data (the central principle)

Everything the platform produces is one of two things, and they are never mixed:

| | **Detected Fact** | **Derived Insight** |
|---|---|---|
| Answers | *What happened* | *Why it mattered* |
| Example | "Score changed 87→89 at 4:12" | "This kill raised LAT break probability by 42%" |
| Contains opinion? | Never | Always (a quantified judgement) |
| Must carry | Evidence (frame/crop), confidence | Citations to the exact facts it was derived from + confidence |
| Produced by | Vision / audio / OCR detectors, manual labels | Analytics / models over facts |

**A fact never contains an opinion. An insight always cites the facts that
produced it.** Nothing in the platform is allowed to be magic. This principle is
enforced *structurally* by the Phase 1 universal event schema (see
[`EVENT_SCHEMA.md`](EVENT_SCHEMA.md)), not just by convention.

Supporting invariants (already the discipline in the v0 code, now made universal):

- **Every event stores evidence** — frame index, video timestamp, frame path, crop path, source.
- **Every prediction stores confidence** — a number in `[0, 1]`, always visible.
- **Every model output stores its version** — name + version, so outputs are reproducible and comparable.
- **Every manual label stores its source** — who labelled it and from what.
- **Nothing silently falls back** — stub/placeholder data is flagged `is_placeholder` and never silently presented as real.

---

## 3. Current repository state (audit, 2026-06-27)

FastAPI + SQLAlchemy + SQLite, CPU-only, Python 3.12.13 in the local virtualenv.
`main` branch, **no git remote configured**. Baseline after the Phase 0/1/2 work
currently in this repository: **91 tests passing**.

### 3.1 What exists and is real

- **Ingestion pipeline** (`workers/processor.py`): idempotent, checkpointed stages
  `ingest → sample_extract → timeline → analytics → clips → report → store`, with
  per-stage `processing_jobs`, retry/backoff, and audited broadcast state machine
  (`state.py`, `models.py:BroadcastStatus`).
- **HUD cropping** (`services/hud.py`): resolution-independent fractional regions;
  profiles in `data/configs/hud_profiles/` (`CDL_2026_1080p`, `generic_1080p`).
- **Frame sampling** (`services/sampling.py`): cadence + crop-change gating to skip
  redundant OCR calls.
- **Score timeline + Hardpoint analytics** (`services/timeline.py`,
  `services/analytics.py`): score events, lead changes, map boundaries,
  scoring-flow break/retake inference with debounce, and `HeuristicV0` xMWP
  (uncalibrated sigmoid — explicitly a placeholder for a trained model).
- **Hill / gunfight / spawn breakdown** (`services/hardpoint_breakdown.py`): 60s
  hill-rotation windows, per-player K/D read from post-game cards (cross-checked),
  low-confidence spawn inference. Verified/derived/inferred are labelled separately.
- **Minimap** (`services/minimap.py`): `ClassicalMinimapDetector` (colour/shape CV,
  no training) → marker positions + occupancy heatmaps.
- **Coach view** (`services/coach_view.py`): momentum timeline, turning-point
  ranking, rotation timing — the first "explain why" surfaces.
- **What-If Lab** (`services/simulation.py`, `routes/sim.py`, `/lab`): transparent
  counterfactual Hardpoint model (remove-kill / prevent-spawn-flip / swap-duel),
  leave-one-out win-prob impact. Demo match is fully synthetic and labelled.
- **Real, evidence-backed match** (`services/real_match.py`): LAT vs Vancouver
  Surge, Hacienda Hardpoint (250–156), score timeline **read and visually verified
  frame-by-frame** from the official VOD, crops kept in `data/crops/lat_van_hp/`.
- **Source monitoring** (`sources/discovery.py`, `workers/scheduler.py`): yt-dlp
  metadata discovery, dedupe on `(platform, video_id)`, opt-in download.
- **Evidence + confidence persistence** everywhere; API (`routes/`) + dashboard (`ui.py`).
- **YOLO minimap dataset scaffold** (`scripts/build_minimap_dataset.py`,
  `data/minimap_dataset/`): model-assisted pre-labels, 6 sample frames. No model trained.

### 3.2 What is stubbed or deferred (no pretending)

- **Score OCR is not production-ready.** Default `StubOcrEngine` reads
  `data/fixtures/sample_scores.json`. `TesseractOcrEngine` is wired but **cannot
  read the stylised CDL scorebar font** reliably. Phase 3 adds
  `CdlScorebarOcrEngine` (`0.1.0-knn`), a CPU digit-gallery baseline trained from
  human-verified LAT/VAN scorebar crops. It reads the labeled gallery, but honest
  leave-one-crop-out evaluation is only `10/21` exact scores (`0.4762`) and
  temporal decoding is `11/21` (`0.5238`), so the stub remains the default.
- **No killfeed, weapon, position, spawn, or objective extraction from pixels.**
- **No audio pipeline at all** — caster/desk/player-comms intelligence is greenfield.
- **xMWP is uncalibrated**; break/retake are scoring-flow inferences, not kill/hill-control confirmation.
- **Deep analytics are Hardpoint-only**; SnD/Control are boundary-only.

### 3.3 The structural gap Phase 1 closes

The current `Event` model (`models.py`) is a **flat, score-centric SQL row** with
fixed columns (`score_a`, `score_b`, `hill_id`, `player`) and a **closed 13-value
`EventType` enum** (`schemas.py`). It cannot represent a kill's weapon, a position's
coordinates, or a comms transcript, and it has no way for an insight to cite the
facts behind it. **Phase 1 introduces a universal event envelope** that generalises
this — fact/insight separation, provenance, evidence, and typed payloads — built
*alongside* the existing model with an adapter, so nothing breaks.

---

## 4. The universal event architecture (Phase 1 foundation)

Everything in the platform becomes a **`GameEvent`**: a common envelope carrying
identity, provenance, evidence, and confidence, wrapping exactly one **typed
payload**. Detectors emit facts; analytics emit insights that cite facts. Storage,
reports, API, and dashboards all consume the same stream. Full spec in
[`EVENT_SCHEMA.md`](EVENT_SCHEMA.md). This is the contract every subsystem below
emits into — it was built first, on purpose, because it is the spine of the
platform.

---

## 5. Subsystem roadmap

Status legend: ✅ real · 🟡 partial / heuristic · 🔬 scaffold only · ⬜ not started.
Each module is **independent and loosely coupled** — models are never entangled.

### 5.1 Vision modules (each mirrors the `OcrEngine` Protocol: crop → detect → boxes+confidence+evidence → `model_outputs`)

| Module | Status | Emits (facts) | Coach question it serves |
|---|---|---|---|
| Score OCR | 🟡 stub + evaluated CDL baseline | `ScoreUpdateEvent` | What is the score over time? |
| Killfeed OCR | ⬜ | `KillEvent`, `DeathEvent`, `WeaponEvent`, `TradeEvent` | Who traded, who got isolated? |
| Weapon recognition | ⬜ | `WeaponEvent` | What archetypes/swaps were used? |
| Minimap detection | 🟡 classical | `PositionEvent` | Where was everyone? |
| Player tracking | 🔬 dataset only | `PositionEvent` (player-resolved) | Routes, crossfires, space created |
| Objective tracking | 🟡 score-inferred | `ObjectiveEvent`, `SpawnFlipEvent` | Hill/bomb control, capture progress |
| HUD recognition | ✅ regions | (frame state) | (enables every other module) |
| Camera-state detection | ⬜ | `TimelineEvent` (camera) | Which player/POV is shown? |
| Replay detection | ⬜ | `TimelineEvent` (replay) | Exclude replays from live telemetry |
| Listen-in detection | ⬜ | `TimelineEvent` (listen-in window) | When are player comms audible? |
| Commentary detection | ⬜ | `TimelineEvent` (caster) | Separate caster from player audio |
| Facecam detection | ⬜ | `TimelineEvent` (facecam) | Player reactions / tilt signals |
| Scoreboard detection | 🟡 post-game | (per-player stats) | Ground-truth K/D, accuracy |
| End-of-map stat detection | 🟡 manual | (map stat card) | Verify the box score automatically |

**Long-term CV targets** (all downstream of player tracking + objective tracking):
positions, movement paths, routes, crossfires, head glitches, sightlines, power
positions, flanks, spawn anchors/blocks, objective pressure, hill occupancy, bomb
routes/plants/defuses, capture progress, control-stack timing, utility usage,
weapon swaps/pickups, alive counts, vision cones, replay sequences, overlays.
Everything becomes structured `GameEvent`s.

> **Minimap honesty constraint:** broadcast minimaps usually show only the *observed*
> team's information. All position outputs must carry an `observed_team` / visibility
> field and must never infer hidden opponents.

### 5.2 Broadcast audio intelligence (greenfield)

The audio track is a dataset we currently ignore. Build a speech pipeline that
**separates streams** before transcribing: caster, desk analysis, player listen-ins,
crowd, game audio, music, adverts. Each segment is a `TimelineEvent` fact with its
audio-source label, confidence, and timestamp.

### 5.3 Player listen-in intelligence (a flagship differentiator)

No one in esports measures this. When the broadcast surfaces player comms:

1. Detect the listen-in window (vision + audio). 2. Diarise speakers. 3. Transcribe
with timestamps. 4. Align each sentence to game facts → `CommunicationEvent`s.

Then derive (as **insights**, each citing the comm facts + game facts):
callout timing/latency, communication density, dead-air %, overtalk %, redundant /
missed / false callouts, call accuracy & success, leadership distribution (who
initiates rotations / calls enemy locations / confirms kills / coordinates trades &
utility), comms before successful vs failed breaks vs collapses vs clutches, and
reaction time between call and execution.

Target insight examples: *"95% of successful rotations were initiated by Envoy."*
*"Communication density drops 41% during collapses."* *"LAT average response to
critical calls is 620ms."*

### 5.4 Tactical intelligence layer

Once facts (kills, positions, objectives, comms) exist, derive strategy-level
insights: break attempts, setup quality, rotation quality, pinch/collapse timing,
crossfire quality, map control, spawn pressure, route efficiency, space creation,
trade chains, lane control, tempo, risk profile, momentum, pressure, confidence.

### 5.5 Elite metrics (insights coaches cannot compute by hand)

Kill Value Added · Death Cost · Trade Damage Index · Pressure Without Kills · Spawn
Leverage · Hill Entry Quality · Break Probability Added · Rotation Expected Value ·
Map Control Equity · Collapse Detection · Invisible Impact Rating · Swing Event
Attribution · Space Created · Lane Denial Index · Crossfire Quality · Isolation Score
· Team Elasticity · Pressure Conversion Rate · Setup Stability · Route Predictability
· Information Advantage · Vision Advantage · Objective Leverage · Clutch Probability
Added.

**Communication-intelligence metrics:** Communication Quality Score, Leadership
Index, Decision/Response Latency, Confirmation Rate, Call Accuracy/Success,
Information Density, Dead-Air %, Overtalk %, Role Clarity, Strategic Diversity,
Emotional Stability, Pressure Response, Leadership Distribution, Initiative Score,
Trust Index (future).

Every metric ships with its definition, the facts it consumes, a confidence, and a
worked example — never a bare number.

### 5.6 Simulation / What-If

`/lab` already prototypes this. Grows into decision-quality and practice-priority
sims as real event data replaces the synthetic fixture.

### 5.7 Autonomous platform

End state: download schedules, track the CDL calendar, detect new matches, queue
ingestion, capture streams/VODs, run OCR + vision + speech, generate events →
insights → coach reports → APIs → dashboards → alerts, with no manual intervention.
Foundations exist (scheduler, discovery, queue, retry).

---

## 6. Machine-learning philosophy

Never jump straight to deep learning. For every model:

1. **Dataset** (labelled, versioned, with source per label).
2. **Annotation tooling** (turn the classical detectors into pre-labellers — already
   started for minimap).
3. **Evaluation harness** — precision, recall, confidence calibration, and a curated
   set of **failure examples** — *before* training.
4. **Versioned model** — every output records `model_name` + `model_version`.
5. Only then train; promote a model only when it beats the current baseline on the
   eval set.

Confidence is a first-class output, not an afterthought, and must be calibrated.

---

## 7. Phased plan

Each feature follows: **understand → design → implement → test → document →
evaluate → sample output → only then continue.** Never two major systems at once;
never skip tests; never invent data.

| Phase | Deliverable | Gate to exit |
|---|---|---|
| **0** ✅ | This roadmap | done — exists and audited |
| **1** ✅ | Universal event schema (envelope, fact/insight, provenance, evidence, typed payloads) + adapter | done — schema + tests + docs + sample output; suite green |
| **2** ✅ | Emit pipeline: score/break outputs persist as `GameEvent`s in `game_events`; report + API + dashboard read the unified stream; flat `events` table retired | done — byte-for-byte report parity (JSON/MD/HTML) vs pre-migration baseline |
| **3** ✅ | Score OCR baseline on labelled CDL scorebars (`CdlScorebarOcrEngine`) | done for baseline — dataset, eval, tests, docs, sample output; not promoted to default because LOO exact score accuracy is `0.4762` |
| **4** | Killfeed OCR → `KillEvent`/`DeathEvent`/`WeaponEvent`/`TradeEvent` | precision/recall on labelled killfeed set |
| **5** | Minimap → player-resolved `PositionEvent` (YOLO) with visibility discipline | mAP on labelled minimap set |
| **6** | Objective/spawn tracking → `ObjectiveEvent`/`SpawnFlipEvent` from pixels | agreement with verified hill timeline |
| **7** | Audio pipeline + listen-in → `CommunicationEvent` | diarisation + transcription eval |
| **8+** | Tactical layer, elite metrics, comms metrics, autonomous orchestration | each: defined, evidenced, evaluated |

Phases 3–7 are independent modules and may be reordered by value, but each consumes
the Phase 1 schema and each ships with its own dataset + evaluation.

---

## 8. Definition of done (every subsystem)

- [ ] Emits/consumes `GameEvent`s — facts and insights kept separate.
- [ ] Every output carries evidence, confidence, and (if model-produced) a version.
- [ ] Insights cite the exact facts they derive from.
- [ ] Dataset + evaluation (precision/recall/calibration + failure cases) for any model.
- [ ] Tests (unit + contract); placeholders flagged, never silently real.
- [ ] Documented, with a sample output a coach could read.
- [ ] Answers a real coach question.
