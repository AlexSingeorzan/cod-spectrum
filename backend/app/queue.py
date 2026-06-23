from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ProcessingJob


class Queue(Protocol):
    """Replaceable processing queue; database-backed for v0."""

    def enqueue(self, broadcast_id: int, stage: str = "pipeline") -> ProcessingJob: ...


class DatabaseQueue:
    def __init__(self, session: Session):
        self.session = session

    def enqueue(self, broadcast_id: int, stage: str = "pipeline") -> ProcessingJob:
        job = self.session.scalar(select(ProcessingJob).where(
            ProcessingJob.broadcast_id == broadcast_id,
            ProcessingJob.stage == stage,
        ))
        if job is None:
            job = ProcessingJob(broadcast_id=broadcast_id, stage=stage, status="pending", logs="enqueued\n")
            self.session.add(job)
        elif job.status not in {"running", "completed"}:
            job.status = "pending"
            job.next_retry_at = None
            job.logs += f"re-enqueued at {datetime.now(timezone.utc).isoformat()}\n"
        return job

