from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Universal events live in backend/app/events/ (GameEvent). The flat EventCreate /
# EventType schema was retired in Phase 2; persistence is models.GameEventRecord.


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
