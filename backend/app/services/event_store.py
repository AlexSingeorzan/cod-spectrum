"""Persistence bridge between the universal ``GameEvent`` schema and the database.

The ``game_events`` table stores the full validated envelope as JSON; this module is
the only place that translates between the ORM rows and ``GameEvent`` objects, plus a
``to_report_row`` projection that reproduces the legacy event-dict shape so reports,
the dashboard, and the API render exactly as before.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..events import GameEvent, ObjectiveEvent, TimelineEvent
from ..models import GameEventRecord


def store_game_events(session: Session, events: list[GameEvent]) -> list[GameEventRecord]:
    """Persist validated events; returns the created rows (flushed, with ids)."""
    records: list[GameEventRecord] = []
    for event in events:
        record = GameEventRecord(
            event_id=event.event_id,
            broadcast_id=event.broadcast_id,
            match_id=event.match_id,
            map_id=event.map_id,
            video_timestamp_seconds=event.video_timestamp_seconds,
            event_type=event.event_type,
            kind=event.kind.value,
            confidence=event.confidence,
            is_placeholder=event.is_placeholder,
            envelope=event.model_dump(mode="json"),
        )
        session.add(record)
        records.append(record)
    session.flush()
    return records


def record_to_game_event(record: GameEventRecord) -> GameEvent:
    return GameEvent.model_validate(record.envelope)


def load_game_event_records(session: Session, broadcast_id: int) -> list[GameEventRecord]:
    return list(session.scalars(
        select(GameEventRecord)
        .where(GameEventRecord.broadcast_id == broadcast_id)
        .order_by(GameEventRecord.video_timestamp_seconds, GameEventRecord.id)
    ).all())


def load_game_events(session: Session, broadcast_id: int) -> list[GameEvent]:
    return [record_to_game_event(record) for record in load_game_event_records(session, broadcast_id)]


def delete_game_events(session: Session, broadcast_id: int) -> None:
    session.execute(delete(GameEventRecord).where(GameEventRecord.broadcast_id == broadcast_id))


def count_game_events(session: Session) -> int:
    return session.scalar(select(func.count(GameEventRecord.id))) or 0


def _legacy_event_type(payload) -> str:
    """Recover the flat event_type the legacy pipeline used for this payload."""
    if isinstance(payload, ObjectiveEvent):
        return payload.action            # possible_break / possible_retake / hill_change / ...
    if isinstance(payload, TimelineEvent):
        return payload.label             # timeout_or_pause / high_value_moment / gunfight / ...
    return payload.EVENT_TYPE             # score_update / lead_change / map_start / map_end / ...


def to_report_row(record: GameEventRecord) -> dict[str, Any]:
    """Project a stored event into the legacy event-dict shape used by reports/API."""
    event = record_to_game_event(record)
    payload = event.payload
    attrs = payload.attributes
    player = payload.team if isinstance(payload, ObjectiveEvent) else getattr(payload, "player", None)
    return {
        "timestamp_seconds": event.video_timestamp_seconds,
        "event_type": _legacy_event_type(payload),
        "team_a": getattr(payload, "team_a", None) or attrs.get("team_a"),
        "team_b": getattr(payload, "team_b", None) or attrs.get("team_b"),
        "player": player,
        "opposing_player": getattr(payload, "opposing_player", None),
        "score_a": getattr(payload, "score_a", None) if getattr(payload, "score_a", None) is not None else attrs.get("score_a"),
        "score_b": getattr(payload, "score_b", None) if getattr(payload, "score_b", None) is not None else attrs.get("score_b"),
        "hill_id": getattr(payload, "hill_id", None),
        "confidence": event.confidence,
        "raw_text": payload.raw_text,
        "evidence_frame_path": event.evidence.frame_path or "",
        "clip_id": record.clip_id,
        "is_placeholder": event.is_placeholder,
    }
