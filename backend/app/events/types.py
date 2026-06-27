"""Concrete event payloads — the platform's event taxonomy.

Importing this module registers every payload type with the core registry. Each
class fixes its ``EVENT_TYPE`` and ``KIND``. All but ``InsightEvent`` are facts:
objective observations that must never carry an opinion. ``InsightEvent`` is the only
derived judgement, and (enforced by the envelope) must cite the facts it came from.
"""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from .core import EventKind, EventPayload, GameMode, Side


# ---------------------------------------------------------------------------
# Score domain (facts) — what the current pipeline already produces
# ---------------------------------------------------------------------------


class ScoreUpdateEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "score_update"
    KIND: ClassVar[EventKind] = EventKind.FACT

    team_a: str
    team_b: str
    score_a: int = Field(ge=0)
    score_b: int = Field(ge=0)
    side_scored: Side | None = None


class LeadChangeEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "lead_change"
    KIND: ClassVar[EventKind] = EventKind.FACT

    team_a: str
    team_b: str
    score_a: int = Field(ge=0)
    score_b: int = Field(ge=0)
    new_leader_side: Side


class MapStartEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "map_start"
    KIND: ClassVar[EventKind] = EventKind.FACT

    mode: GameMode = GameMode.UNKNOWN
    map_name: str = "unknown"
    team_a: str
    team_b: str


class MapEndEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "map_end"
    KIND: ClassVar[EventKind] = EventKind.FACT

    mode: GameMode = GameMode.UNKNOWN
    map_name: str = "unknown"
    team_a: str
    team_b: str
    score_a: int = Field(ge=0)
    score_b: int = Field(ge=0)
    winner_side: Side | None = None


# ---------------------------------------------------------------------------
# Combat domain (facts) — killfeed / weapon vision targets
# ---------------------------------------------------------------------------


class KillEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "kill"
    KIND: ClassVar[EventKind] = EventKind.FACT

    attacker: str | None = None
    attacker_team: str | None = None
    attacker_side: Side | None = None
    victim: str | None = None
    victim_team: str | None = None
    victim_side: Side | None = None
    weapon: str | None = None
    headshot: bool | None = None
    is_trade: bool | None = None


class DeathEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "death"
    KIND: ClassVar[EventKind] = EventKind.FACT

    player: str
    team: str | None = None
    side: Side | None = None
    killer: str | None = None
    weapon: str | None = None


class WeaponEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "weapon"
    KIND: ClassVar[EventKind] = EventKind.FACT

    player: str
    team: str | None = None
    weapon: str
    action: Literal["pickup", "swap", "use"] = "use"


class TradeEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "trade"
    KIND: ClassVar[EventKind] = EventKind.FACT

    dead_player: str
    dead_team: str | None = None
    trading_player: str
    trading_team: str | None = None
    trade_window_seconds: float = Field(ge=0)
    original_kill_event_id: str | None = None
    trade_kill_event_id: str | None = None


# ---------------------------------------------------------------------------
# Objective / space domain (facts)
# ---------------------------------------------------------------------------


class ObjectiveEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "objective"
    KIND: ClassVar[EventKind] = EventKind.FACT

    objective_type: Literal["hardpoint_hill", "snd_bomb", "control_zone"]
    # e.g. hill_active, hill_contested, hill_change, hill_summary, possible_break,
    # possible_retake, plant, defuse, capture
    action: str
    hill_id: str | None = None
    side: Side | None = None
    team: str | None = None
    progress: float | None = Field(default=None, ge=0, le=1)


class SpawnFlipEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "spawn_flip"
    KIND: ClassVar[EventKind] = EventKind.FACT

    side: Side | None = None
    team: str | None = None
    from_region: str | None = None
    to_region: str | None = None
    hill_id: str | None = None
    inferred: bool = True            # spawn flips are usually inferred, not read
    method: str | None = None


class PositionEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "position"
    KIND: ClassVar[EventKind] = EventKind.FACT

    x: float = Field(ge=0, le=1)     # normalised minimap coordinates
    y: float = Field(ge=0, le=1)
    player: str | None = None
    team: str | None = None
    side: Side | None = None
    detector: Literal["minimap", "tracking"] = "minimap"
    region: str | None = None
    # Broadcast minimaps usually show only the observed team. Record whose
    # information this is; never infer hidden opponents.
    observed_team: str | None = None


class RotationEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "rotation"
    KIND: ClassVar[EventKind] = EventKind.FACT

    team: str | None = None
    side: Side | None = None
    player: str | None = None
    from_region: str | None = None
    to_region: str | None = None
    hill_id: str | None = None


# ---------------------------------------------------------------------------
# Audio / broadcast domain (facts)
# ---------------------------------------------------------------------------


class CommunicationEvent(EventPayload):
    EVENT_TYPE: ClassVar[str] = "communication"
    KIND: ClassVar[EventKind] = EventKind.FACT

    transcript: str
    speaker: str | None = None
    team: str | None = None
    callout_type: str | None = None          # rotate, enemy_location, confirm, utility, ...
    targets: list[str] = Field(default_factory=list)
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    audio_source: Literal["player", "caster", "desk"] = "player"


class TimelineEvent(EventPayload):
    """Broadcast-state / narrative markers: replays, listen-in windows, camera state,
    facecam, caster segments, pauses, map transitions."""

    EVENT_TYPE: ClassVar[str] = "timeline"
    KIND: ClassVar[EventKind] = EventKind.FACT

    marker_type: Literal[
        "replay", "listen_in", "camera", "facecam", "caster",
        "commercial", "crowd", "pause", "map_transition", "high_value", "custom",
    ]
    label: str
    subject: str | None = None
    end_seconds: float | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Derived judgement (the only insight type)
# ---------------------------------------------------------------------------


class InsightEvent(EventPayload):
    """A derived judgement that explains *why* something mattered.

    The envelope requires a non-empty ``derived_from`` citing the facts behind it.
    """

    EVENT_TYPE: ClassVar[str] = "insight"
    KIND: ClassVar[EventKind] = EventKind.INSIGHT

    headline: str
    explanation: str
    metric: str | None = None
    value: float | None = None
    delta: float | None = None
    subject: str | None = None       # player/team the insight is about


__all__ = [
    "ScoreUpdateEvent", "LeadChangeEvent", "MapStartEvent", "MapEndEvent",
    "KillEvent", "DeathEvent", "WeaponEvent", "TradeEvent",
    "ObjectiveEvent", "SpawnFlipEvent", "PositionEvent", "RotationEvent",
    "CommunicationEvent", "TimelineEvent", "InsightEvent",
]
