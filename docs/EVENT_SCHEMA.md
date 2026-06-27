# Universal Event Schema (Phase 1)

> Everything the platform observes or concludes becomes a **`GameEvent`** — one
> envelope, one typed payload, with provenance, evidence, and confidence built in.
> This is the spine every vision/audio/tactical module emits into.

Module: `backend/app/events/` · Schema version: **1.0** · Status: **wired into the
pipeline (Phase 2)** — events persist to the `game_events` table and the reports,
dashboard, and API read them, with byte-for-byte report parity against the retired
flat `events` table. Persistence bridge: `backend/app/services/event_store.py`.

---

## 1. Why this exists

The v0 `Event` (`models.py`) is a flat SQL row with score-specific columns and a
closed 13-value type enum. It cannot describe a kill's weapon, a player's position,
or a comms transcript, and an insight has no way to point at the facts it came from.

The universal schema fixes all of that **without touching the existing table**. It
is a Pydantic layer (`GameEvent`) plus a legacy adapter. The processor, DB, routes,
and tests are unchanged in Phase 1.

---

## 2. The two-layer model

```
GameEvent  (envelope — identity, provenance, evidence, confidence, citations)
└── payload: one typed body
       FACTS:    KillEvent · DeathEvent · WeaponEvent · TradeEvent ·
                 ObjectiveEvent · SpawnFlipEvent · PositionEvent ·
                 RotationEvent · CommunicationEvent · TimelineEvent ·
                 ScoreUpdateEvent · LeadChangeEvent · MapStartEvent · MapEndEvent
       INSIGHTS: InsightEvent
```

A payload class fixes its own `event_type` and `kind` (fact vs insight) as
class-level constants — they cannot be spoofed per-instance. The envelope reads them
back as computed fields so they appear in JSON and can be indexed.

`KillEvent` etc. are **payload bodies**, not envelopes. You always wrap one in a
`GameEvent`. (The naming mirrors the master spec's vocabulary; the envelope is the
single transport type.)

---

## 3. The envelope: `GameEvent`

| Field | Type | Notes |
|---|---|---|
| `event_id` | `str` | Auto-filled, **deterministic** content hash if omitted (no randomness). |
| `schema_version` | `str` | Defaults to `1.0`. |
| `broadcast_id` / `match_id` / `map_id` | `int \| None` | Hierarchy links (match the existing model ids). |
| `video_timestamp_seconds` | `float ≥ 0` | Canonical event time into the VOD. |
| `game_clock_seconds` | `float \| None` | In-game clock when known (≠ video time). |
| `confidence` | `float ∈ [0,1]` | Always present, always visible. |
| `provenance` | `Provenance` | Who/what produced it + model version. |
| `evidence` | `Evidence` | Frame/crop/timestamp/source. |
| `payload` | one typed body | Serialised with subclass fields (`SerializeAsAny`). |
| `derived_from` | `list[str]` | Event ids of the facts this was derived from. |
| `is_placeholder` | `bool` | Stub/fixture data, never silently real. |
| `tags` | `list[str]` | Free-form (`"verified"`, `"synthetic"`, …). |
| `event_type` *(computed)* | `str` | From the payload class. |
| `kind` *(computed)* | `fact \| insight` | From the payload class. |

### `Evidence`
`video_timestamp_seconds`, `frame_index?`, `frame_path?`, `crop_path?`,
`source_url?`, `thumbnail_b64?`. Helper `has_visual()` = a frame or crop exists.

### `Provenance`
`source: SourceKind`, `model_name?`, `model_version?`, `producer?`, `labeled_by?`,
`note?`.

`SourceKind` ∈ `human_verified · manual_label · model · heuristic · derived ·
synthetic · external`.

---

## 4. The invariants (enforced in code, tested)

These encode the master principle "facts are objective; insights cite their facts;
nothing is magic":

1. **Confidence bounded** — `0 ≤ confidence ≤ 1` (and on every nested metric).
2. **Facts carry visual evidence** — if `kind == fact`, `evidence.has_visual()` must
   be true (a frame or crop path). Continues the v0 evidence invariant.
3. **Insights must cite** — if `kind == insight`, `derived_from` must be non-empty.
   *An insight with no facts behind it is a validation error.*
4. **Facts must not cite** — if `kind == fact`, `derived_from` must be empty (a fact
   is an observation, not a derivation).
5. **Models declare a version** — if `provenance.source == model`, both
   `model_name` and `model_version` are required.
6. **Manual labels declare a labeller** — if `source ∈ {manual_label,
   human_verified}`, `labeled_by` is required.
7. **`event_type` / `kind` are derived from the payload**, never user-set, so they
   cannot disagree with the body.
8. **`event_id` is deterministic** — a SHA1 over identity + payload, so the same
   observation yields the same id and insight citations are reproducible.

---

## 5. Payload catalogue

All fact bodies; `InsightEvent` is the only insight.

| Payload | `event_type` | Key fields |
|---|---|---|
| `ScoreUpdateEvent` | `score_update` | `team_a/b`, `score_a/b`, `side_scored?` |
| `LeadChangeEvent` | `lead_change` | `new_leader_side`, `team_a/b`, `score_a/b` |
| `MapStartEvent` | `map_start` | `mode`, `map_name`, `team_a/b` |
| `MapEndEvent` | `map_end` | `mode`, `map_name`, `winner_side`, `score_a/b` |
| `KillEvent` | `kill` | `attacker`, `victim`, `*_team`, `weapon?`, `headshot?`, `is_trade?` |
| `DeathEvent` | `death` | `player`, `team`, `killer?`, `weapon?` |
| `WeaponEvent` | `weapon` | `player`, `team`, `weapon`, `action` (pickup/swap/use) |
| `TradeEvent` | `trade` | `dead_player`, `trading_player`, `*_team`, `trade_window_seconds`, kill ids |
| `ObjectiveEvent` | `objective` | `objective_type`, `action`, `hill_id?`, `side?`, `progress?` |
| `SpawnFlipEvent` | `spawn_flip` | `side`, `from_region?`, `to_region?`, `inferred`, `method?` |
| `PositionEvent` | `position` | `x`, `y` (0–1), `player?`, `team?`, `observed_team?`, `detector` |
| `RotationEvent` | `rotation` | `team`, `player?`, `from_region?`, `to_region?`, `hill_id?` |
| `CommunicationEvent` | `communication` | `transcript`, `speaker?`, `team?`, `callout_type?`, `targets`, `start/end_seconds`, `audio_source` |
| `TimelineEvent` | `timeline` | `marker_type` (replay/listen_in/camera/facecam/caster/pause/…), `label`, `end_seconds?`, `subject?` |
| `InsightEvent` | `insight` | `headline`, `explanation`, `metric?`, `value?`, `delta?`, `subject?` |

Every body also has an `attributes: dict` escape hatch for structured detail that
does not yet warrant a schema change, and a `raw_text?` for the originating OCR/ASR
string.

> **Position visibility discipline:** broadcast minimaps usually expose only the
> *observed* team. `PositionEvent.observed_team` records whose information this is;
> detectors must never invent hidden opponents.

---

## 6. Usage

```python
from backend.app.events import GameEvent, KillEvent, InsightEvent, Evidence, Provenance, SourceKind

kill = GameEvent(
    broadcast_id=1, map_id=1, video_timestamp_seconds=258.0, confidence=0.91,
    provenance=Provenance(source=SourceKind.MODEL, model_name="killfeed-ocr", model_version="0.1.0"),
    evidence=Evidence(video_timestamp_seconds=258.0, crop_path="data/crops/k_258.png"),
    payload=KillEvent(attacker="Envoy", attacker_team="LAT", victim="Pred", victim_team="VAN", weapon="SMG"),
)

insight = GameEvent(
    broadcast_id=1, map_id=1, video_timestamp_seconds=258.0, confidence=0.78,
    provenance=Provenance(source=SourceKind.DERIVED, producer="break-model@0.2"),
    evidence=Evidence(video_timestamp_seconds=258.0),       # insights need no visual
    derived_from=[kill.event_id],                            # MUST cite ≥1 fact
    payload=InsightEvent(headline="Lockout kill",
                         explanation="Envoy's pick prevented VAN's hill retake.",
                         metric="break_probability_added", delta=0.42),
)

GameEvent.model_validate_json(kill.model_dump_json())        # round-trips with payload type
```

Helpers: `events.to_jsonl(iterable)` / `events.from_jsonl(text)` for streams, and
`events.adapter.from_legacy_event(...)` to lift existing `Event` rows into envelopes.

---

## 7. What is wired, and what is not yet

Wired in Phase 2:

- `models.GameEventRecord` (`game_events` table) persists the full envelope as JSON
  plus denormalised columns for querying.
- The processor emits `GameEvent`s (via `events.from_legacy_event` over the score
  timeline / break-retake dicts) and stores them; the report, dashboard, and
  `/broadcasts/{id}` API read them back through `event_store.to_report_row`.
- The flat `events` table and `EventCreate` schema are removed — `game_events` is the
  single source of truth.

Not yet (later phases):

- **No new detectors.** Payloads for Kill/Position/Communication exist as the
  contract those future modules will fill; nothing emits them in the pipeline yet.
- xMWP still persists separately in `model_outputs` (not as events).

Evaluation of the schema is `tests/test_event_schema.py` and
`tests/test_event_store.py`, plus `scripts/sample_events.py`. Report parity is proven
by diffing the regenerated JSON/Markdown/HTML against the pre-migration baseline.
