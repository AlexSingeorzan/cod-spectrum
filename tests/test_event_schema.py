"""Contract tests for the universal event schema (docs/EVENT_SCHEMA.md).

These lock the principles the whole platform depends on: facts carry evidence,
insights cite their facts, models declare versions, confidence is bounded, and every
event round-trips through JSON with its concrete payload type intact.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.events import (
    CommunicationEvent,
    DeathEvent,
    Evidence,
    EventKind,
    GameEvent,
    InsightEvent,
    KillEvent,
    LeadChangeEvent,
    MapEndEvent,
    MapStartEvent,
    ObjectiveEvent,
    PositionEvent,
    Provenance,
    RotationEvent,
    ScoreUpdateEvent,
    SourceKind,
    SpawnFlipEvent,
    TimelineEvent,
    TradeEvent,
    WeaponEvent,
    from_jsonl,
    from_legacy_event,
    get_payload_class,
    payload_registry,
    to_jsonl,
)


# A representative, minimally-valid instance of every payload type.
SAMPLE_PAYLOADS = [
    ScoreUpdateEvent(team_a="LAT", team_b="VAN", score_a=85, score_b=63),
    LeadChangeEvent(team_a="LAT", team_b="VAN", score_a=85, score_b=63, new_leader_side="a"),
    MapStartEvent(team_a="LAT", team_b="VAN"),
    MapEndEvent(team_a="LAT", team_b="VAN", score_a=250, score_b=156, winner_side="a"),
    KillEvent(attacker="Envoy", attacker_team="LAT", victim="Pred", victim_team="VAN", weapon="SMG"),
    DeathEvent(player="Pred", team="VAN", killer="Envoy"),
    WeaponEvent(player="Envoy", team="LAT", weapon="SMG", action="swap"),
    TradeEvent(dead_player="Pred", trading_player="Cellium", trade_window_seconds=1.5),
    ObjectiveEvent(objective_type="hardpoint_hill", action="hill_contested", hill_id="P3"),
    SpawnFlipEvent(side="b", from_region="north", to_region="south"),
    PositionEvent(x=0.4, y=0.7, player="Envoy", observed_team="LAT"),
    RotationEvent(team="LAT", from_region="P3", to_region="P4"),
    CommunicationEvent(transcript="rotate now", speaker="Envoy", callout_type="rotate"),
    TimelineEvent(marker_type="replay", label="killcam replay"),
    InsightEvent(headline="Lockout kill", explanation="Prevented VAN retake", metric="break_prob_added", delta=0.42),
]


def _fact_evidence() -> Evidence:
    return Evidence(video_timestamp_seconds=258.0, frame_path="data/crops/x.jpg")


def _heuristic() -> Provenance:
    return Provenance(source=SourceKind.HEURISTIC, producer="test")


def make_event(payload, **overrides) -> GameEvent:
    kwargs = dict(
        broadcast_id=1, map_id=1, video_timestamp_seconds=258.0, confidence=0.9,
        provenance=_heuristic(), evidence=_fact_evidence(), payload=payload,
    )
    kwargs.update(overrides)
    return GameEvent(**kwargs)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_contains_every_payload_type():
    expected = {
        "score_update", "lead_change", "map_start", "map_end", "kill", "death",
        "weapon", "trade", "objective", "spawn_flip", "position", "rotation",
        "communication", "timeline", "insight",
    }
    assert set(payload_registry()) == expected


def test_only_insight_is_an_insight_kind():
    for event_type, cls in payload_registry().items():
        expected = EventKind.INSIGHT if event_type == "insight" else EventKind.FACT
        assert cls.KIND == expected, event_type


def test_get_payload_class_resolves_and_rejects():
    assert get_payload_class("kill") is KillEvent
    with pytest.raises(KeyError):
        get_payload_class("does_not_exist")


# --------------------------------------------------------------------------- #
# Core invariants
# --------------------------------------------------------------------------- #


def test_fact_requires_visual_evidence():
    with pytest.raises(ValidationError):
        make_event(
            ScoreUpdateEvent(team_a="A", team_b="B", score_a=1, score_b=0),
            evidence=Evidence(video_timestamp_seconds=1.0),  # no frame/crop
        )


def test_fact_must_not_cite_sources():
    with pytest.raises(ValidationError):
        make_event(
            ScoreUpdateEvent(team_a="A", team_b="B", score_a=1, score_b=0),
            derived_from=["abc123"],
        )


def test_insight_must_cite_at_least_one_fact():
    insight = InsightEvent(headline="x", explanation="y")
    with pytest.raises(ValidationError):
        make_event(insight, provenance=Provenance(source=SourceKind.DERIVED))
    # ...and succeeds once it cites a fact (evidence is optional for insights)
    ok = make_event(
        insight,
        provenance=Provenance(source=SourceKind.DERIVED),
        evidence=Evidence(video_timestamp_seconds=1.0),
        derived_from=["abc123"],
    )
    assert ok.kind == EventKind.INSIGHT


def test_model_provenance_requires_name_and_version():
    with pytest.raises(ValidationError):
        Provenance(source=SourceKind.MODEL, model_name="killfeed-ocr")  # missing version
    ok = Provenance(source=SourceKind.MODEL, model_name="killfeed-ocr", model_version="0.1.0")
    assert ok.model_version == "0.1.0"


def test_manual_and_verified_provenance_require_labeller():
    for source in (SourceKind.MANUAL_LABEL, SourceKind.HUMAN_VERIFIED):
        with pytest.raises(ValidationError):
            Provenance(source=source)
        assert Provenance(source=source, labeled_by="alex").labeled_by == "alex"


def test_confidence_is_bounded():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            make_event(KillEvent(attacker="x"), confidence=bad)


def test_position_coordinates_are_normalised():
    with pytest.raises(ValidationError):
        make_event(PositionEvent(x=1.2, y=0.5))


# --------------------------------------------------------------------------- #
# Identity, type derivation, serialisation
# --------------------------------------------------------------------------- #


def test_event_type_and_kind_are_derived_from_payload():
    event = make_event(KillEvent(attacker="Envoy"))
    assert event.event_type == "kill"
    assert event.kind == EventKind.FACT
    # they appear in the serialised envelope
    dumped = event.model_dump()
    assert dumped["event_type"] == "kill"
    assert dumped["kind"] == "fact"


def test_payload_is_source_of_truth_over_spoofed_top_level_type():
    raw = {
        "video_timestamp_seconds": 1.0, "confidence": 0.5,
        "provenance": {"source": "heuristic"},
        "evidence": {"video_timestamp_seconds": 1.0, "frame_path": "f.jpg"},
        "event_type": "score_update",                       # spoof attempt
        "payload": {"event_type": "kill", "attacker": "Envoy"},
    }
    event = GameEvent.model_validate(raw)
    assert event.event_type == "kill"
    assert isinstance(event.payload, KillEvent)


def test_event_id_is_deterministic_and_content_sensitive():
    a = make_event(KillEvent(attacker="Envoy", victim="Pred", weapon="SMG"))
    b = make_event(KillEvent(attacker="Envoy", victim="Pred", weapon="SMG"))
    c = make_event(KillEvent(attacker="Envoy", victim="Pred", weapon="AR"))
    assert a.event_id == b.event_id
    assert a.event_id != c.event_id


@pytest.mark.parametrize("payload", SAMPLE_PAYLOADS, ids=lambda p: p.EVENT_TYPE)
def test_every_event_type_round_trips_through_json(payload):
    overrides = {}
    if payload.KIND == EventKind.INSIGHT:
        overrides = {"derived_from": ["seed-fact-id"], "provenance": Provenance(source=SourceKind.DERIVED)}
    event = make_event(payload, **overrides)
    restored = GameEvent.model_validate_json(event.model_dump_json())
    assert type(restored.payload) is type(payload)
    assert restored.event_type == payload.EVENT_TYPE
    assert restored.event_id == event.event_id
    assert restored.payload.model_dump() == payload.model_dump()


def test_jsonl_stream_round_trips():
    events = [make_event(p, **({"derived_from": ["x"], "provenance": Provenance(source=SourceKind.DERIVED)}
                               if p.KIND == EventKind.INSIGHT else {})) for p in SAMPLE_PAYLOADS]
    text = to_jsonl(events)
    restored = from_jsonl(text)
    assert len(restored) == len(events)
    assert [e.event_type for e in restored] == [e.event_type for e in events]


# --------------------------------------------------------------------------- #
# Legacy adapter
# --------------------------------------------------------------------------- #


LEGACY_TYPES = [
    "map_start", "map_end", "score_update", "lead_change", "hill_change",
    "possible_break", "possible_retake", "killfeed_event_placeholder",
    "timeout_or_pause", "high_value_moment", "hill_summary", "gunfight", "spawn_flip",
]


def _legacy_row(event_type: str, **extra) -> dict:
    base = dict(
        broadcast_id=1, match_id=1, map_id=1, timestamp_seconds=146.0,
        event_type=event_type, team_a="LAT", team_b="VAN", player="VAN",
        opposing_player=None, score_a=34, score_b=38, hill_id="P2",
        confidence=0.88, raw_text="legacy", evidence_frame_path="data/crops/sb.jpg",
        clip_id=None, is_placeholder=False,
    )
    base.update(extra)
    return base


@pytest.mark.parametrize("event_type", LEGACY_TYPES)
def test_adapter_lifts_every_legacy_type_to_a_valid_fact(event_type):
    event = from_legacy_event(_legacy_row(event_type))
    assert event.kind == EventKind.FACT
    assert event.evidence.frame_path == "data/crops/sb.jpg"
    assert event.video_timestamp_seconds == 146.0
    assert 0.0 <= event.confidence <= 1.0
    # survives a JSON round-trip too
    assert GameEvent.model_validate_json(event.model_dump_json()).event_type == event.event_type


def test_adapter_maps_score_update_fields():
    event = from_legacy_event(_legacy_row("score_update"))
    assert isinstance(event.payload, ScoreUpdateEvent)
    assert event.payload.score_a == 34 and event.payload.score_b == 38


def test_adapter_flags_placeholder_as_synthetic():
    real = from_legacy_event(_legacy_row("score_update", is_placeholder=False))
    stub = from_legacy_event(_legacy_row("score_update", is_placeholder=True))
    assert real.provenance.source == SourceKind.HEURISTIC
    assert stub.provenance.source == SourceKind.SYNTHETIC and stub.is_placeholder is True


def test_adapter_maps_break_to_objective_fact():
    event = from_legacy_event(_legacy_row("possible_break"))
    assert isinstance(event.payload, ObjectiveEvent)
    assert event.payload.action == "possible_break"
    assert event.payload.hill_id == "P2"
