from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Broadcast, Clip, Event, Map, ProcessingJob, Report, Source
from ..queue import DatabaseQueue
from ..schemas import BroadcastRead, SourceCreate, SourceRead


router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/sources", response_model=list[SourceRead])
def list_sources(session: Session = Depends(get_db)):
    return session.scalars(select(Source).order_by(Source.id)).all()


@router.post("/sources", response_model=SourceRead, status_code=201)
def create_source(payload: SourceCreate, session: Session = Depends(get_db)):
    existing = session.scalar(select(Source).where(Source.url == payload.url))
    if existing:
        return existing
    source = Source(
        name=payload.name, platform=payload.platform, source_type=payload.type, url=payload.url,
        poll_minutes=payload.poll_minutes, download=payload.download,
    )
    session.add(source)
    session.commit()
    session.refresh(source)
    return source


@router.get("/broadcasts", response_model=list[BroadcastRead])
def list_broadcasts(session: Session = Depends(get_db)):
    return session.scalars(select(Broadcast).order_by(Broadcast.created_at.desc())).all()


@router.get("/broadcasts/{broadcast_id}")
def get_broadcast(broadcast_id: int, session: Session = Depends(get_db)):
    broadcast = session.get(Broadcast, broadcast_id)
    if not broadcast:
        raise HTTPException(404, "broadcast not found")
    jobs = session.scalars(select(ProcessingJob).where(ProcessingJob.broadcast_id == broadcast_id)).all()
    maps = session.scalars(select(Map).where(Map.broadcast_id == broadcast_id)).all()
    events = session.scalars(select(Event).where(Event.broadcast_id == broadcast_id).order_by(Event.timestamp_seconds)).all()
    return {
        "broadcast": BroadcastRead.model_validate(broadcast).model_dump(mode="json"),
        "status_history": broadcast.status_history,
        "jobs": [{"id": job.id, "stage": job.stage, "status": job.status, "attempt_count": job.attempt_count, "next_retry_at": job.next_retry_at, "logs": job.logs} for job in jobs],
        "maps": [{"id": item.id, "ordinal": item.ordinal, "mode": item.mode, "map_name": item.map_name} for item in maps],
        "events": [{"id": event.id, "timestamp_seconds": event.timestamp_seconds, "event_type": event.event_type, "score_a": event.score_a, "score_b": event.score_b, "confidence": event.confidence, "evidence_frame_path": event.evidence_frame_path, "clip_id": event.clip_id} for event in events],
    }


@router.post("/broadcasts/{broadcast_id}/process", status_code=202)
def enqueue_broadcast(broadcast_id: int, session: Session = Depends(get_db)):
    broadcast = session.get(Broadcast, broadcast_id)
    if not broadcast:
        raise HTTPException(404, "broadcast not found")
    if not broadcast.local_path:
        raise HTTPException(409, "broadcast is reference-only; provide a local file or enable source download")
    job = DatabaseQueue(session).enqueue(broadcast_id)
    session.commit()
    return {"job_id": job.id, "status": job.status}


@router.get("/reports")
def list_reports(session: Session = Depends(get_db)):
    return [{"id": item.id, "broadcast_id": item.broadcast_id, "data_confidence": item.data_confidence, "html_path": item.html_path} for item in session.scalars(select(Report).order_by(Report.created_at.desc())).all()]


@router.get("/reports/{report_id}")
def get_report(report_id: int, session: Session = Depends(get_db)):
    report = session.get(Report, report_id)
    if not report:
        raise HTTPException(404, "report not found")
    path = Path(report.html_path)
    if not path.exists():
        raise HTTPException(404, "report file missing")
    return FileResponse(path, media_type="text/html")


@router.get("/clips/{clip_id}")
def get_clip(clip_id: int, session: Session = Depends(get_db)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    path = Path(clip.file_path)
    if not path.exists():
        raise HTTPException(404, "clip file missing")
    return FileResponse(path, media_type="video/mp4", filename=path.name)

