"""Lift legacy ``Event`` rows (models.py / EventCreate) into universal ``GameEvent``s.

This proves the existing pipeline's output is fully expressible as universal events
and seeds the Phase 2 migration. It is a one-way lift and does not touch the database.
The events package stays DB-agnostic: the source is read by duck typing, so this
accepts either a SQLAlchemy ``Event`` or a plain dict / ``EventCreate``.
"""
from __future__ import annotations

from typing import Any

from .core import Evidence, GameEvent, Provenance, SourceKind
from .types import (
    KillEvent,
    LeadChangeEvent,
    MapEndEvent,
    MapStartEvent,
    ObjectiveEvent,
    ScoreUpdateEvent,
    SpawnFlipEvent,
    TimelineEvent,
)


def _get(src: Any, key: str, default: Any = None) -> Any:
    if isinstance(src, dict):
        return src.get(key, default)
    if hasattr(src, "model_dump"):  # pydantic EventCreate
        return src.model_dump().get(key, default)
    return getattr(src, key, default)


def _leader_side(score_a: int | None, score_b: int | None) -> str | None:
    if score_a is None or score_b is None or score_a == score_b:
        return None
    return "a" if score_a > score_b else "b"


def _build_payload(event_type: str, src: Any):
    team_a = _get(src, "team_a", "TEAM_A")
    team_b = _get(src, "team_b", "TEAM_B")
    score_a = _get(src, "score_a")
    score_b = _get(src, "score_b")
    player = _get(src, "player")
    hill_id = _get(src, "hill_id")
    raw_text = _get(src, "raw_text")

    if event_type == "score_update":
        return ScoreUpdateEvent(team_a=team_a, team_b=team_b, score_a=score_a or 0, score_b=score_b or 0, raw_text=raw_text)
    if event_type == "lead_change":
        return LeadChangeEvent(team_a=team_a, team_b=team_b, score_a=score_a or 0, score_b=score_b or 0,
                               new_leader_side=_leader_side(score_a, score_b) or "a", raw_text=raw_text)
    if event_type == "map_start":
        return MapStartEvent(team_a=team_a, team_b=team_b, raw_text=raw_text)
    if event_type == "map_end":
        return MapEndEvent(team_a=team_a, team_b=team_b, score_a=score_a or 0, score_b=score_b or 0,
                           winner_side=_leader_side(score_a, score_b), raw_text=raw_text)
    if event_type == "spawn_flip":
        return SpawnFlipEvent(team=player, hill_id=hill_id, raw_text=raw_text)
    if event_type == "killfeed_event_placeholder":
        return KillEvent(raw_text=raw_text)
    if event_type in {"timeout_or_pause", "high_value_moment", "gunfight"}:
        marker = {"timeout_or_pause": "pause", "high_value_moment": "high_value", "gunfight": "custom"}[event_type]
        return TimelineEvent(marker_type=marker, label=event_type, subject=player, raw_text=raw_text)
    # hill_change, possible_break, possible_retake, hill_summary -> objective facts
    if event_type in {"hill_change", "possible_break", "possible_retake", "hill_summary"}:
        return ObjectiveEvent(objective_type="hardpoint_hill", action=event_type, hill_id=hill_id,
                              team=player, raw_text=raw_text,
                              attributes={"score_a": score_a, "score_b": score_b})
    # Unknown legacy type: preserve it honestly as a custom timeline marker.
    return TimelineEvent(marker_type="custom", label=event_type, subject=player, raw_text=raw_text)


def from_legacy_event(src: Any) -> GameEvent:
    """Convert a legacy Event (ORM row, EventCreate, or dict) to a GameEvent fact."""
    event_type = _get(src, "event_type")
    if not event_type:
        raise ValueError("legacy event has no event_type")
    timestamp = float(_get(src, "timestamp_seconds", 0.0))
    frame_path = _get(src, "evidence_frame_path")
    confidence = float(_get(src, "confidence", 0.0))
    is_placeholder = bool(_get(src, "is_placeholder", False))

    payload = _build_payload(event_type, src)
    source = SourceKind.SYNTHETIC if is_placeholder else SourceKind.HEURISTIC
    return GameEvent(
        broadcast_id=_get(src, "broadcast_id"),
        match_id=_get(src, "match_id"),
        map_id=_get(src, "map_id"),
        video_timestamp_seconds=timestamp,
        confidence=max(0.0, min(1.0, confidence)),
        provenance=Provenance(source=source, producer=f"legacy-pipeline:{event_type}"),
        evidence=Evidence(video_timestamp_seconds=timestamp, frame_path=frame_path),
        payload=payload,
        is_placeholder=is_placeholder,
    )


__all__ = ["from_legacy_event"]
