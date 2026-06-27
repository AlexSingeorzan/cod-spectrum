"""Universal event schema for COD Spectrum (see docs/EVENT_SCHEMA.md).

Importing this package registers every payload type, so ``GameEvent`` can
deserialise any event by ``event_type``.
"""
from __future__ import annotations

from .adapter import from_legacy_event
from .core import (
    SCHEMA_VERSION,
    Evidence,
    EventKind,
    EventPayload,
    GameEvent,
    GameMode,
    Provenance,
    Side,
    SourceKind,
    from_jsonl,
    get_payload_class,
    payload_registry,
    register_payload,
    to_jsonl,
)
from .types import (
    CommunicationEvent,
    DeathEvent,
    InsightEvent,
    KillEvent,
    KillType,
    LeadChangeEvent,
    MapEndEvent,
    MapStartEvent,
    ObjectiveEvent,
    PositionEvent,
    RotationEvent,
    ScoreUpdateEvent,
    SpawnFlipEvent,
    TimelineEvent,
    TradeEvent,
    WeaponEvent,
)

__all__ = [
    "SCHEMA_VERSION",
    "GameEvent",
    "EventPayload",
    "EventKind",
    "SourceKind",
    "GameMode",
    "Side",
    "Evidence",
    "Provenance",
    "get_payload_class",
    "payload_registry",
    "register_payload",
    "to_jsonl",
    "from_jsonl",
    "from_legacy_event",
    # payloads
    "ScoreUpdateEvent",
    "LeadChangeEvent",
    "MapStartEvent",
    "MapEndEvent",
    "KillType",
    "KillEvent",
    "DeathEvent",
    "WeaponEvent",
    "TradeEvent",
    "ObjectiveEvent",
    "SpawnFlipEvent",
    "PositionEvent",
    "RotationEvent",
    "CommunicationEvent",
    "TimelineEvent",
    "InsightEvent",
]
