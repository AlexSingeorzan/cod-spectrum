from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EventType = Literal[
    "map_start", "map_end", "score_update", "lead_change", "hill_change",
    "possible_break", "possible_retake", "killfeed_event_placeholder",
    "timeout_or_pause", "high_value_moment",
    # 60s hill-rotation pipeline (hardpoint_breakdown):
    "hill_summary", "gunfight", "spawn_flip",
]


class EventCreate(BaseModel):
    broadcast_id: int
    match_id: int | None = None
    map_id: int | None = None
    timestamp_seconds: float = Field(ge=0)
    event_type: EventType
    team_a: str
    team_b: str
    player: str | None = None
    opposing_player: str | None = None
    score_a: int | None = None
    score_b: int | None = None
    hill_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    raw_text: str | None = None
    evidence_frame_path: str
    clip_id: int | None = None
    is_placeholder: bool = False

    @field_validator("evidence_frame_path")
    @classmethod
    def evidence_is_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("events require evidence_frame_path")
        return value


class SourceCreate(BaseModel):
    name: str
    platform: str
    type: Literal["channel", "playlist", "video", "local"]
    url: str
    poll_minutes: int = Field(default=30, ge=1)
    download: bool = False


class SourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    platform: str
    source_type: str
    url: str
    poll_minutes: int
    download: bool
    enabled: bool


class BroadcastRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    platform: str
    video_id: str
    title: str
    source_url: str | None
    local_path: str | None
    status: str
    last_error: str | None


class ScoreObservation(BaseModel):
    timestamp_seconds: float
    score_a: int
    score_b: int
    confidence: float = Field(ge=0, le=1)
    evidence_frame_path: str
    raw_text: str | None = None


class XmwpPoint(BaseModel):
    timestamp_seconds: float
    score_a: int
    score_b: int
    probability_a: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_frame_path: str

    @field_validator("evidence_frame_path")
    @classmethod
    def xmwp_evidence_is_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("xMWP points require evidence_frame_path")
        return value


class EvidenceMetric(BaseModel):
    value: Any
    timestamp_seconds: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    evidence_frame_path: str

    @field_validator("evidence_frame_path")
    @classmethod
    def metric_evidence_is_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metrics require evidence_frame_path")
        return value


class RecommendedClip(BaseModel):
    id: int
    title: str
    start_seconds: float
    end_seconds: float
    timestamp_seconds: float
    confidence: float = Field(ge=0, le=1)
    evidence_frame_path: str
    file_path: str
    url: str


class ReportDocument(BaseModel):
    broadcast: dict
    match: dict
    map: dict
    timeline: list[dict]
    key_moments: list[dict]
    possible_breaks_retakes: list[dict]
    xmwp_timeline: list[XmwpPoint]
    recommended_clips: list[RecommendedClip]
    hardpoint_summary: dict[str, EvidenceMetric]
    data_confidence: float = Field(ge=0, le=1)
    data_confidence_evidence: EvidenceMetric
    known_limitations: list[str]
