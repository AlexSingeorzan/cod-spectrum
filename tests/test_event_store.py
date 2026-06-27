"""Tests for the GameEvent persistence bridge (services/event_store.py).

These lock the Phase 2 contract: events persist as universal envelopes and project
back into the exact legacy dict shape the reports and API render, with no data loss.
"""
from __future__ import annotations

from sqlalchemy import select

from backend.app.events import GameEvent, from_legacy_event
from backend.app.models import GameEventRecord
from backend.app.services.event_store import (
    count_game_events,
    delete_game_events,
    load_game_events,
    record_to_game_event,
    store_game_events,
    to_report_row,
)


def legacy(event_type: str, **overrides) -> dict:
    base = dict(
        broadcast_id=1, match_id=1, map_id=1, timestamp_seconds=100.0, event_type=event_type,
        team_a="LAT", team_b="VAN", score_a=34, score_b=38, confidence=0.9,
        raw_text=None, evidence_frame_path="data/crops/x.jpg", is_placeholder=False,
    )
    base.update(overrides)
    return base


def _store_one(session, src: dict) -> GameEventRecord:
    store_game_events(session, [from_legacy_event(src)])
    session.commit()
    return session.scalars(select(GameEventRecord)).one()


def test_store_and_load_round_trip(session):
    events = [from_legacy_event(legacy("score_update"))]
    store_game_events(session, events)
    session.commit()
    loaded = load_game_events(session, 1)
    assert len(loaded) == 1
    assert isinstance(loaded[0], GameEvent)
    assert loaded[0].event_id == events[0].event_id
    assert loaded[0].evidence.has_visual()


def test_count_and_delete(session):
    store_game_events(session, [from_legacy_event(legacy("score_update")), from_legacy_event(legacy("lead_change"))])
    session.commit()
    assert count_game_events(session) == 2
    delete_game_events(session, 1)
    session.commit()
    assert count_game_events(session) == 0


def test_to_report_row_parity_for_score_event(session):
    record = _store_one(session, legacy("score_update", score_a=85, score_b=63))
    row = to_report_row(record)
    assert row["event_type"] == "score_update"
    assert (row["score_a"], row["score_b"]) == (85, 63)
    assert row["team_a"] == "LAT" and row["team_b"] == "VAN"
    assert row["evidence_frame_path"] == "data/crops/x.jpg"
    assert row["player"] is None and row["opposing_player"] is None and row["clip_id"] is None
    assert row["is_placeholder"] is False


def test_to_report_row_recovers_break_legacy_type_and_score(session):
    record = _store_one(session, legacy("possible_break", player="VAN", raw_text="flow changed"))
    row = to_report_row(record)
    assert row["event_type"] == "possible_break"          # recovered from ObjectiveEvent.action
    assert (row["score_a"], row["score_b"]) == (34, 38)    # recovered from payload attributes
    assert row["player"] == "VAN"
    assert row["raw_text"] == "flow changed"


def test_map_start_preserves_score(session):
    record = _store_one(session, legacy("map_start", score_a=16, score_b=11))
    row = to_report_row(record)
    assert row["event_type"] == "map_start"
    assert (row["score_a"], row["score_b"]) == (16, 11)


def test_clip_id_is_projected(session):
    record = _store_one(session, legacy("score_update"))
    record.clip_id = 5
    session.commit()
    assert to_report_row(record)["clip_id"] == 5


def test_placeholder_flag_round_trips(session):
    record = _store_one(session, legacy("score_update", is_placeholder=True))
    assert record.is_placeholder is True
    assert to_report_row(record)["is_placeholder"] is True
    assert record_to_game_event(record).is_placeholder is True
