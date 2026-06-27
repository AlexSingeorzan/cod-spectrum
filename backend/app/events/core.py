"""Universal event envelope for COD Spectrum.

Everything the platform observes (facts) or concludes (insights) becomes a
``GameEvent``: one envelope carrying identity, provenance, evidence and confidence,
wrapping exactly one typed payload. The fact/insight separation and the evidence,
versioning, and citation rules from docs/ROADMAP.md are enforced here in code.

This module is intentionally decoupled from the database and the processing
pipeline. Wiring detectors and storage onto this schema is a later phase.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, computed_field, model_validator


SCHEMA_VERSION = "1.0"

# Team side within a single map; the human-readable label (e.g. "LAT") lives in the
# payload's team fields. "a"/"b" mirror the existing score_a/score_b convention.
Side = Literal["a", "b"]


class EventKind(str, Enum):
    """A detected observation, or a derived judgement about observations."""

    FACT = "fact"
    INSIGHT = "insight"


class SourceKind(str, Enum):
    """How an event was produced. Drives the provenance requirements."""

    HUMAN_VERIFIED = "human_verified"   # a person read it off the broadcast
    MANUAL_LABEL = "manual_label"       # a person annotated a dataset crop
    MODEL = "model"                     # a trained model emitted it (needs version)
    HEURISTIC = "heuristic"             # a rule/classical-CV detector emitted it
    DERIVED = "derived"                 # analytics over other events (insights)
    SYNTHETIC = "synthetic"             # generated demo/fixture data
    EXTERNAL = "external"               # external truth, e.g. a published box score


class GameMode(str, Enum):
    HARDPOINT = "hardpoint"
    SND = "snd"
    CONTROL = "control"
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    """Where an event can be seen in the source material. No event ships without it."""

    model_config = ConfigDict(extra="ignore")

    video_timestamp_seconds: float = Field(ge=0)
    frame_index: int | None = Field(default=None, ge=0)
    frame_path: str | None = None
    crop_path: str | None = None
    source_url: str | None = None
    thumbnail_b64: str | None = None

    def has_visual(self) -> bool:
        """True when a frame or crop image backs this event."""
        return bool((self.frame_path or "").strip() or (self.crop_path or "").strip())


class Provenance(BaseModel):
    """Who or what produced an event, and with which model version."""

    model_config = ConfigDict(extra="ignore")

    source: SourceKind
    model_name: str | None = None
    model_version: str | None = None
    producer: str | None = None        # service/pipeline that emitted it
    labeled_by: str | None = None       # person, for manual/verified sources
    note: str | None = None

    @model_validator(mode="after")
    def _require_source_metadata(self) -> "Provenance":
        if self.source == SourceKind.MODEL and not (self.model_name and self.model_version):
            raise ValueError("model-sourced events require both model_name and model_version")
        if self.source in {SourceKind.MANUAL_LABEL, SourceKind.HUMAN_VERIFIED} and not self.labeled_by:
            raise ValueError(f"{self.source.value} events require labeled_by")
        return self


# ---------------------------------------------------------------------------
# Payload base + registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type["EventPayload"]] = {}


class EventPayload(BaseModel):
    """Base class for every typed event body.

    Subclasses set ``EVENT_TYPE`` and ``KIND`` as class constants; they cannot be
    overridden per-instance, so a payload can never misreport what it is. Subclasses
    auto-register by ``EVENT_TYPE`` so the envelope can deserialise them.
    """

    model_config = ConfigDict(extra="ignore")

    EVENT_TYPE: ClassVar[str] = ""
    KIND: ClassVar[EventKind] = EventKind.FACT

    raw_text: str | None = None                       # originating OCR/ASR string
    attributes: dict[str, Any] = Field(default_factory=dict)  # structured escape hatch

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "EVENT_TYPE", ""):
            _REGISTRY[cls.EVENT_TYPE] = cls

    @computed_field  # surfaces in JSON; derived from the class, never user-set
    @property
    def event_type(self) -> str:
        return type(self).EVENT_TYPE

    @computed_field
    @property
    def kind(self) -> EventKind:
        return type(self).KIND


def register_payload(cls: type[EventPayload]) -> type[EventPayload]:
    """Explicit registration helper (subclasses also auto-register)."""
    if not cls.EVENT_TYPE:
        raise ValueError(f"{cls.__name__} must set EVENT_TYPE before registration")
    _REGISTRY[cls.EVENT_TYPE] = cls
    return cls


def get_payload_class(event_type: str) -> type[EventPayload]:
    try:
        return _REGISTRY[event_type]
    except KeyError as exc:
        raise KeyError(f"unknown event_type '{event_type}'; known: {sorted(_REGISTRY)}") from exc


def payload_registry() -> dict[str, type[EventPayload]]:
    """Snapshot of the registered payload types."""
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


class GameEvent(BaseModel):
    """The universal event: provenance + evidence + confidence + one typed payload."""

    model_config = ConfigDict(extra="ignore")

    event_id: str | None = None                       # deterministic content hash if omitted
    schema_version: str = SCHEMA_VERSION
    broadcast_id: int | None = None
    match_id: int | None = None
    map_id: int | None = None
    video_timestamp_seconds: float = Field(ge=0)
    game_clock_seconds: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    provenance: Provenance
    evidence: Evidence
    payload: SerializeAsAny[EventPayload]
    derived_from: list[str] = Field(default_factory=list)
    is_placeholder: bool = False
    tags: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def event_type(self) -> str:
        return self.payload.EVENT_TYPE

    @computed_field
    @property
    def kind(self) -> EventKind:
        return self.payload.KIND

    @model_validator(mode="before")
    @classmethod
    def _resolve_payload(cls, data: Any) -> Any:
        """Turn a serialised payload dict back into its concrete subclass."""
        if isinstance(data, dict):
            payload = data.get("payload")
            if isinstance(payload, dict):
                event_type = payload.get("event_type") or data.get("event_type")
                if not event_type:
                    raise ValueError("payload dict requires 'event_type' to resolve its type")
                payload_cls = get_payload_class(event_type)
                data = {**data, "payload": payload_cls.model_validate(payload)}
        return data

    @model_validator(mode="after")
    def _enforce_invariants(self) -> "GameEvent":
        if self.payload.KIND == EventKind.FACT:
            if self.derived_from:
                raise ValueError(f"fact '{self.event_type}' must not set derived_from (facts are observations, not derivations)")
            if not self.evidence.has_visual():
                raise ValueError(f"fact '{self.event_type}' requires evidence (frame_path or crop_path)")
        else:  # INSIGHT
            if not self.derived_from:
                raise ValueError(f"insight '{self.event_type}' must cite at least one source fact in derived_from")
        if not self.event_id:
            self.event_id = self._compute_id()
        return self

    def _compute_id(self) -> str:
        """SHA1 over identity + payload: same observation -> same id (no randomness)."""
        basis = {
            "schema": self.schema_version,
            "b": self.broadcast_id,
            "m": self.match_id,
            "mp": self.map_id,
            "t": round(self.video_timestamp_seconds, 3),
            "et": self.payload.EVENT_TYPE,
            "payload": self.payload.model_dump(mode="json"),
            "df": sorted(self.derived_from),
        }
        raw = json.dumps(basis, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def to_jsonl(events: Iterable[GameEvent]) -> str:
    """Serialise a stream of events to JSON Lines."""
    return "\n".join(event.model_dump_json() for event in events)


def from_jsonl(text: str) -> list[GameEvent]:
    """Parse a JSON Lines stream back into validated events."""
    return [GameEvent.model_validate_json(line) for line in text.splitlines() if line.strip()]
