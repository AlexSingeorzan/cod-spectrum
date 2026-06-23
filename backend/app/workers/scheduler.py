from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy import select

from ..config import ROOT, get_settings
from ..db import SessionLocal, init_db
from ..models import Broadcast, BroadcastStatus, ProcessingJob, Source
from ..queue import DatabaseQueue
from ..sources.discovery import YtDlpDiscoverer
from ..sources.discovery import DiscoveredVideo
from ..state import transition_broadcast
from .processor import download_url, process_local_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_sources(path: Path | None = None) -> list[dict]:
    config_path = path or get_settings().data_dir / "configs" / "sources.yaml"
    return yaml.safe_load(config_path.read_text()).get("sources", [])


def register_discovered_video(session, source: Source, video: DiscoveredVideo) -> tuple[Broadcast, bool]:
    """Idempotently register one discovered VOD."""
    broadcast = session.scalar(select(Broadcast).where(
        Broadcast.platform == video.platform, Broadcast.video_id == video.video_id,
    ))
    if broadcast is not None:
        return broadcast, False
    broadcast = Broadcast(
        source_id=source.id, platform=video.platform, video_id=video.video_id, title=video.title,
        source_url=video.url, local_path=video.local_path, status=BroadcastStatus.discovered,
        status_history=[{"from": None, "to": "discovered", "at": datetime.now(timezone.utc).isoformat()}],
    )
    session.add(broadcast)
    session.flush()
    return broadcast, True


def discover_and_enqueue(config_path: Path | None = None, force: bool = False) -> dict[str, int]:
    init_db()
    discoverer = YtDlpDiscoverer()
    stats = {"polled": 0, "discovered": 0, "deduplicated": 0, "enqueued": 0, "errors": 0}
    with SessionLocal() as session:
        for config in load_sources(config_path):
            if not config.get("enabled", True):
                logger.info("source disabled; skipping %s", config["name"])
                continue
            source = session.scalar(select(Source).where(Source.url == config["url"]))
            if source is None:
                source = Source(
                    name=config["name"], platform=config["platform"], source_type=config["type"], url=config["url"],
                    poll_minutes=config.get("poll_minutes", 30), download=config.get("download", False), enabled=True,
                )
                session.add(source)
                session.flush()
            if source.last_polled_at and not force:
                last_polled = source.last_polled_at
                if last_polled.tzinfo is None:
                    last_polled = last_polled.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - last_polled).total_seconds() / 60
                if elapsed < source.poll_minutes:
                    continue
            stats["polled"] += 1
            logger.info("polling source %s (%s)", source.name, source.url)
            try:
                videos = discoverer.discover(config)
                source.last_polled_at = datetime.now(timezone.utc)
                for video in videos:
                    broadcast, created = register_discovered_video(session, source, video)
                    if not created:
                        stats["deduplicated"] += 1
                        logger.info("deduplicated %s:%s", video.platform, video.video_id)
                        continue
                    stats["discovered"] += 1
                    logger.info("discovered %s (%s)", video.title, video.video_id)
                    if source.download and not broadcast.local_path:
                        logger.info("download explicitly enabled for source %s", source.name)
                        transition_broadcast(broadcast, BroadcastStatus.downloading)
                        downloaded = download_url(video.url, get_settings().data_dir / "videos")
                        broadcast.local_path = str(downloaded.resolve())
                        transition_broadcast(broadcast, BroadcastStatus.downloaded)
                    if broadcast.local_path:
                        DatabaseQueue(session).enqueue(broadcast.id)
                        stats["enqueued"] += 1
                    else:
                        logger.info("reference-only VOD recorded without processing: %s", video.url)
            except Exception as exc:
                stats["errors"] += 1
                logger.exception("source poll failed for %s: %s", source.name, exc)
            session.commit()
    return stats


def process_due_jobs() -> dict[str, int]:
    stats = {"processed": 0, "failed": 0}
    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        jobs = session.scalars(select(ProcessingJob).where(ProcessingJob.stage == "pipeline", ProcessingJob.status.in_(["pending", "failed"]))).all()
        for job in jobs:
            if job.next_retry_at:
                next_retry = job.next_retry_at if job.next_retry_at.tzinfo else job.next_retry_at.replace(tzinfo=timezone.utc)
                if next_retry > now:
                    continue
            broadcast = session.get(Broadcast, job.broadcast_id)
            if not broadcast or not broadcast.local_path or broadcast.status == BroadcastStatus.processed:
                continue
            job.status = "running"
            job.attempt_count += 1
            session.commit()
            try:
                process_local_file(Path(broadcast.local_path), session=session)
                job = session.get(ProcessingJob, job.id)
                job.status = "completed"
                job.logs += f"{datetime.now(timezone.utc).isoformat()} processed\n"
                stats["processed"] += 1
                session.commit()
            except Exception as exc:
                session.rollback()
                job = session.get(ProcessingJob, job.id)
                job.status = "failed"
                job.logs += f"{datetime.now(timezone.utc).isoformat()} {exc}\n"
                stats["failed"] += 1
                session.commit()
                logger.exception("processing failed for broadcast %s", broadcast.id)
    return stats


def run_cycle(force: bool = False) -> dict:
    discovery = discover_and_enqueue(force=force)
    processing = process_due_jobs()
    result = {"discovery": discovery, "processing": processing}
    logger.info("scheduler cycle complete: %s", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Always-on cod-spectrum source scheduler")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = parser.parse_args(argv)
    if args.once:
        run_cycle(force=True)
        return 0
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_cycle, "interval", minutes=1, id="source-monitor", max_instances=1, coalesce=True, next_run_time=datetime.now(timezone.utc))
    logger.info("scheduler started; polling due sources every minute")
    scheduler.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
