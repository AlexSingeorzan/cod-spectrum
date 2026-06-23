from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BroadcastStatus(str, enum.Enum):
    discovered = "discovered"
    downloading = "downloading"
    downloaded = "downloaded"
    processing = "processing"
    processed = "processed"
    failed = "failed"
    skipped = "skipped"


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    platform: Mapped[str] = mapped_column(String(50))
    source_type: Mapped[str] = mapped_column(String(30))
    url: Mapped[str] = mapped_column(Text, unique=True)
    poll_minutes: Mapped[int] = mapped_column(default=30)
    download: Mapped[bool] = mapped_column(default=False)
    enabled: Mapped[bool] = mapped_column(default=True)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Broadcast(Base):
    __tablename__ = "broadcasts"
    __table_args__ = (UniqueConstraint("platform", "video_id", name="uq_broadcast_platform_video"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"))
    platform: Mapped[str] = mapped_column(String(50))
    video_id: Mapped[str] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(String(500))
    source_url: Mapped[str | None] = mapped_column(Text)
    local_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[BroadcastStatus] = mapped_column(Enum(BroadcastStatus), default=BroadcastStatus.discovered)
    status_changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_error: Mapped[str | None] = mapped_column(Text)
    status_history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    matches: Mapped[list["Match"]] = relationship(back_populates="broadcast", cascade="all, delete-orphan")
    jobs: Mapped[list["ProcessingJob"]] = relationship(back_populates="broadcast", cascade="all, delete-orphan")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (UniqueConstraint("broadcast_id", "stage", name="uq_job_broadcast_stage"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"))
    stage: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30), default="pending")
    attempt_count: Mapped[int] = mapped_column(default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    logs: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    broadcast: Mapped[Broadcast] = relationship(back_populates="jobs")


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"))
    ordinal: Mapped[int] = mapped_column(default=1)
    team_a: Mapped[str] = mapped_column(String(100), default="TEAM_A")
    team_b: Mapped[str] = mapped_column(String(100), default="TEAM_B")
    broadcast: Mapped[Broadcast] = relationship(back_populates="matches")
    maps: Mapped[list["Map"]] = relationship(back_populates="match", cascade="all, delete-orphan")


class Map(Base):
    __tablename__ = "maps"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"))
    ordinal: Mapped[int] = mapped_column(default=1)
    mode: Mapped[str] = mapped_column(String(30), default="unknown")
    map_name: Mapped[str] = mapped_column(String(100), default="unknown")
    match: Mapped[Match] = relationship(back_populates="maps")
    events: Mapped[list["Event"]] = relationship(back_populates="map", cascade="all, delete-orphan")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_event_confidence"),
        CheckConstraint("length(evidence_frame_path) > 0", name="ck_event_evidence"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"))
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"))
    map_id: Mapped[int | None] = mapped_column(ForeignKey("maps.id"))
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    event_type: Mapped[str] = mapped_column(String(80))
    team_a: Mapped[str] = mapped_column(String(100))
    team_b: Mapped[str] = mapped_column(String(100))
    player: Mapped[str | None] = mapped_column(String(100))
    opposing_player: Mapped[str | None] = mapped_column(String(100))
    score_a: Mapped[int | None] = mapped_column(Integer)
    score_b: Mapped[int | None] = mapped_column(Integer)
    hill_id: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)
    raw_text: Mapped[str | None] = mapped_column(Text)
    evidence_frame_path: Mapped[str] = mapped_column(Text, nullable=False)
    clip_id: Mapped[int | None] = mapped_column(ForeignKey("clips.id"))
    is_placeholder: Mapped[bool] = mapped_column(default=False)
    map: Mapped[Map | None] = relationship(back_populates="events")


class Clip(Base):
    __tablename__ = "clips"
    id: Mapped[int] = mapped_column(primary_key=True)
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"))
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"))
    map_id: Mapped[int | None] = mapped_column(ForeignKey("maps.id"))
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    title: Mapped[str] = mapped_column(String(300))
    file_path: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    evidence_frame_path: Mapped[str] = mapped_column(Text)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"), unique=True)
    json_path: Mapped[str] = mapped_column(Text)
    markdown_path: Mapped[str] = mapped_column(Text)
    html_path: Mapped[str] = mapped_column(Text)
    data_confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelOutput(Base):
    __tablename__ = "model_outputs"
    id: Mapped[int] = mapped_column(primary_key=True)
    broadcast_id: Mapped[int] = mapped_column(ForeignKey("broadcasts.id"))
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"))
    map_id: Mapped[int | None] = mapped_column(ForeignKey("maps.id"))
    model_name: Mapped[str] = mapped_column(String(100))
    output_type: Mapped[str] = mapped_column(String(100))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    evidence_frame_path: Mapped[str] = mapped_column(Text)
    is_placeholder: Mapped[bool] = mapped_column(default=False)
