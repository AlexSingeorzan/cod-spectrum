from __future__ import annotations

import json

from sqlalchemy import func, select

from backend.app.models import Broadcast, BroadcastStatus, Clip, GameEventRecord, ProcessingJob, Report
from backend.app.services.event_store import record_to_game_event
from backend.app.services.reports import render_html, render_markdown
from backend.app.workers.processor import process_local_file


def test_offline_processor_is_idempotent_and_complete(session, sample_video):
    first = process_local_file(sample_video, session=session)
    event_count = session.scalar(select(func.count(GameEventRecord.id)))
    clip_count = session.scalar(select(func.count(Clip.id)))
    second = process_local_file(sample_video, session=session)
    assert first.id == second.id
    assert session.scalar(select(func.count(GameEventRecord.id))) == event_count
    assert session.scalar(select(func.count(Clip.id))) == clip_count
    assert event_count >= 8
    assert clip_count >= 1
    broadcast = session.scalar(select(Broadcast))
    assert broadcast.status == BroadcastStatus.processed
    records = session.scalars(select(GameEventRecord)).all()
    assert {"map_start", "map_end", "score_update", "lead_change"}.issubset({item.event_type for item in records})
    assert all(item.confidence is not None for item in records)
    assert all(record_to_game_event(item).evidence.has_visual() for item in records)  # every event keeps evidence
    payload = json.loads(open(first.json_path).read())
    assert payload["xmwp_timeline"]
    assert all(point["evidence_frame_path"] for point in payload["xmwp_timeline"])
    assert all(metric["evidence_frame_path"] for metric in payload["hardpoint_summary"].values())
    assert payload["data_confidence_evidence"]["evidence_frame_path"]
    assert payload["known_limitations"]
    assert "Known limitations" in open(first.markdown_path).read()
    assert "Known limitations" in open(first.html_path).read()


def test_processor_resumes_from_completed_checkpoints(session, sample_video):
    process_local_file(sample_video, session=session)
    sample_job = session.scalar(select(ProcessingJob).where(ProcessingJob.stage == "sample_extract"))
    original_attempts = sample_job.attempt_count
    original_events = session.scalar(select(func.count(GameEventRecord.id)))
    broadcast = session.scalar(select(Broadcast))
    report = session.scalar(select(Report))
    session.delete(report)
    broadcast.status = BroadcastStatus.failed
    for stage in ("report", "store"):
        job = session.scalar(select(ProcessingJob).where(ProcessingJob.stage == stage))
        job.status = "pending"
    session.commit()

    resumed = process_local_file(sample_video, session=session)

    session.refresh(sample_job)
    session.refresh(broadcast)
    assert resumed.id is not None
    assert sample_job.attempt_count == original_attempts
    assert session.scalar(select(func.count(GameEventRecord.id))) == original_events
    assert broadcast.status == BroadcastStatus.processed
