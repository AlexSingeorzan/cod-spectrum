from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import SessionLocal, init_db
from ..models import Broadcast, BroadcastStatus, Clip, Event, Map, Match, ModelOutput, ProcessingJob, Report
from ..schemas import EventCreate, ReportDocument, ScoreObservation, XmwpPoint
from ..services.analytics import HeuristicV0, choose_key_moments, detect_breaks_retakes, hardpoint_summary, xmwp_timeline
from ..services.clips import generate_clip
from ..services.hud import load_hud_profile
from ..services.ocr import build_ocr_engine
from ..services.reports import write_reports
from ..services.timeline import build_score_events, extract_score_observations
from ..state import transition_broadcast


STAGES = ("ingest", "sample_extract", "timeline", "analytics", "clips", "report", "store")


def local_video_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_local_broadcast(session: Session, path: Path) -> Broadcast:
    video_id = local_video_id(path)
    broadcast = session.scalar(select(Broadcast).where(Broadcast.platform == "local", Broadcast.video_id == video_id))
    if broadcast is None:
        broadcast = Broadcast(
            platform="local", video_id=video_id, title=path.stem.replace("_", " ").title(),
            source_url=None, local_path=str(path.resolve()), status=BroadcastStatus.discovered,
            status_history=[{"from": None, "to": "discovered", "at": datetime.now(timezone.utc).isoformat()}],
        )
        session.add(broadcast)
        session.flush()
    elif not broadcast.local_path:
        broadcast.local_path = str(path.resolve())
    return broadcast


def _job(session: Session, broadcast_id: int, stage: str) -> ProcessingJob:
    job = session.scalar(select(ProcessingJob).where(ProcessingJob.broadcast_id == broadcast_id, ProcessingJob.stage == stage))
    if job is None:
        job = ProcessingJob(broadcast_id=broadcast_id, stage=stage, status="pending", logs="")
        session.add(job)
        session.flush()
    return job


def _mark_stage(session: Session, broadcast_id: int, stage: str, status: str, log: str) -> None:
    job = _job(session, broadcast_id, stage)
    if status == "running":
        job.attempt_count += 1
    job.status = status
    job.logs += f"{datetime.now(timezone.utc).isoformat()} {log}\n"
    session.flush()


def _stage_completed(session: Session, broadcast_id: int, stage: str) -> bool:
    job = session.scalar(select(ProcessingJob).where(
        ProcessingJob.broadcast_id == broadcast_id, ProcessingJob.stage == stage,
    ))
    return bool(job and job.status == "completed")


def _event_payload(event: Event) -> dict:
    return {
        "timestamp_seconds": event.timestamp_seconds, "event_type": event.event_type,
        "team_a": event.team_a, "team_b": event.team_b, "player": event.player,
        "opposing_player": event.opposing_player, "score_a": event.score_a, "score_b": event.score_b,
        "hill_id": event.hill_id, "confidence": event.confidence, "raw_text": event.raw_text,
        "evidence_frame_path": event.evidence_frame_path, "clip_id": event.clip_id,
        "is_placeholder": event.is_placeholder,
    }


def _clear_incomplete_outputs(session: Session, broadcast_id: int) -> None:
    session.execute(delete(Event).where(Event.broadcast_id == broadcast_id))
    session.execute(delete(Clip).where(Clip.broadcast_id == broadcast_id))
    session.execute(delete(ModelOutput).where(ModelOutput.broadcast_id == broadcast_id))
    session.execute(delete(Report).where(Report.broadcast_id == broadcast_id))
    session.execute(delete(Map).where(Map.broadcast_id == broadcast_id))
    session.execute(delete(Match).where(Match.broadcast_id == broadcast_id))
    session.flush()


def process_local_file(
    path: Path,
    hud_profile: str = "CDL_2026_1080p",
    sample_fps: float = 1.0,
    debug_crops: bool = False,
    ocr_engine: str = "stub",
    mode: str = "auto",
    session: Session | None = None,
) -> Report:
    if not path.exists():
        raise FileNotFoundError(path)
    own_session = session is None
    session = session or SessionLocal()
    settings = get_settings()
    broadcast: Broadcast | None = None
    current_stage = "ingest"
    try:
        broadcast = register_local_broadcast(session, path)
        existing_report = session.scalar(select(Report).where(Report.broadcast_id == broadcast.id))
        if broadcast.status == BroadcastStatus.processed and existing_report:
            return existing_report
        if broadcast.status in {BroadcastStatus.discovered, BroadcastStatus.downloaded, BroadcastStatus.failed}:
            transition_broadcast(broadcast, BroadcastStatus.processing)
        match = session.scalar(select(Match).where(Match.broadcast_id == broadcast.id).order_by(Match.ordinal))
        if match is None:
            match = Match(broadcast_id=broadcast.id, ordinal=1, team_a="TEAM_A", team_b="TEAM_B")
            session.add(match)
            session.flush()
        game_map = session.scalar(select(Map).where(Map.broadcast_id == broadcast.id).order_by(Map.ordinal))
        if game_map is None:
            detected_mode = "hardpoint" if mode == "auto" and "hardpoint" in path.stem.lower() else mode
            if detected_mode == "auto":
                detected_mode = "unknown"
            game_map = Map(broadcast_id=broadcast.id, match_id=match.id, ordinal=1, mode=detected_mode, map_name="unknown")
            session.add(game_map)
            session.flush()
        if not _stage_completed(session, broadcast.id, "ingest"):
            _mark_stage(session, broadcast.id, "ingest", "completed", f"registered local file {path}")
        session.commit()

        current_stage = "sample_extract"
        observation_output = session.scalar(select(ModelOutput).where(
            ModelOutput.broadcast_id == broadcast.id, ModelOutput.output_type == "score_observations",
        ))
        if _stage_completed(session, broadcast.id, current_stage) and observation_output:
            observations = [ScoreObservation(**item) for item in observation_output.payload["observations"]]
            sampling_stats = observation_output.payload.get("sampling_stats", {})
        else:
            _mark_stage(session, broadcast.id, current_stage, "running", "sampling scorebar")
            session.commit()
            profile = load_hud_profile(hud_profile)
            observations, sampling_stats = extract_score_observations(
                path, profile, build_ocr_engine(ocr_engine), settings.data_dir / "crops" / f"broadcast_{broadcast.id}",
                sample_fps=sample_fps, change_threshold=settings.region_change_threshold, debug_crops=debug_crops,
            )
            if not observations:
                raise RuntimeError("no valid score observations extracted")
            session.execute(delete(ModelOutput).where(
                ModelOutput.broadcast_id == broadcast.id, ModelOutput.output_type == "score_observations",
            ))
            session.add(ModelOutput(
                broadcast_id=broadcast.id, match_id=match.id, map_id=game_map.id,
                model_name=f"{ocr_engine}_scorebar", output_type="score_observations",
                payload={"observations": [item.model_dump() for item in observations], "sampling_stats": sampling_stats},
                confidence=sum(item.confidence for item in observations) / len(observations),
                evidence_frame_path=observations[0].evidence_frame_path, is_placeholder=ocr_engine == "stub",
            ))
            _mark_stage(session, broadcast.id, current_stage, "completed", json.dumps(sampling_stats))
            session.commit()

        current_stage = "timeline"
        if not _stage_completed(session, broadcast.id, current_stage):
            _mark_stage(session, broadcast.id, current_stage, "running", "building score timeline")
            session.commit()
            core_payloads = build_score_events(observations, match.team_a, match.team_b, settings.hardpoint_target)
            for payload in core_payloads:
                payload["is_placeholder"] = ocr_engine == "stub"
                validated = EventCreate(broadcast_id=broadcast.id, match_id=match.id, map_id=game_map.id, **payload)
                session.add(Event(**validated.model_dump()))
            _mark_stage(session, broadcast.id, current_stage, "completed", f"created {len(core_payloads)} core events")
            session.commit()

        current_stage = "analytics"
        if not _stage_completed(session, broadcast.id, current_stage):
            _mark_stage(session, broadcast.id, current_stage, "running", "running hardpoint analytics")
            session.commit()
            derived = detect_breaks_retakes(observations, match.team_a, match.team_b, settings.break_debounce_seconds) if game_map.mode == "hardpoint" else []
            for payload in derived:
                payload["is_placeholder"] = ocr_engine == "stub"
                validated = EventCreate(broadcast_id=broadcast.id, match_id=match.id, map_id=game_map.id, **payload)
                session.add(Event(**validated.model_dump()))
            xmwp = xmwp_timeline(observations, HeuristicV0(settings.hardpoint_target, settings.xmwp_k)) if game_map.mode == "hardpoint" else []
            if xmwp:
                session.add(ModelOutput(
                    broadcast_id=broadcast.id, match_id=match.id, map_id=game_map.id,
                    model_name="HeuristicV0", output_type="xmwp_timeline",
                    payload={"points": [point.model_dump() for point in xmwp]},
                    confidence=sum(point.confidence for point in xmwp) / len(xmwp),
                    evidence_frame_path=observations[0].evidence_frame_path, is_placeholder=True,
                ))
            _mark_stage(session, broadcast.id, current_stage, "completed", f"created {len(derived)} break/retake events and {len(xmwp)} xMWP points")
            session.commit()
        xmwp_output = session.scalar(select(ModelOutput).where(
            ModelOutput.broadcast_id == broadcast.id, ModelOutput.output_type == "xmwp_timeline",
        ))
        xmwp = [XmwpPoint(**item) for item in xmwp_output.payload["points"]] if xmwp_output else []
        stored_events = session.scalars(select(Event).where(Event.broadcast_id == broadcast.id).order_by(Event.timestamp_seconds, Event.id)).all()
        event_payloads = [_event_payload(event) for event in stored_events]
        derived = [payload for payload in event_payloads if payload["event_type"] in {"possible_break", "possible_retake"}]
        key_moments = choose_key_moments(event_payloads)
        current_stage = "clips"
        if not _stage_completed(session, broadcast.id, current_stage):
            _mark_stage(session, broadcast.id, current_stage, "running", "generating key-moment clips")
            session.commit()
            unique_moments: list[dict] = []
            seen_timestamps: set[float] = set()
            for moment in key_moments:
                if moment["timestamp_seconds"] not in seen_timestamps:
                    unique_moments.append(moment)
                    seen_timestamps.add(moment["timestamp_seconds"])
                if len(unique_moments) >= 3:
                    break
            if not unique_moments:
                unique_moments = [event_payloads[-1]]
            for index, moment in enumerate(unique_moments, start=1):
                clip_path = settings.data_dir / "clips" / f"broadcast_{broadcast.id}_moment_{index}.mp4"
                start, end = generate_clip(path, clip_path, moment["timestamp_seconds"])
                clip = Clip(
                    broadcast_id=broadcast.id, match_id=match.id, map_id=game_map.id,
                    start_seconds=start, end_seconds=end, timestamp_seconds=moment["timestamp_seconds"], title=f"{moment['event_type'].replace('_', ' ').title()} at {moment['timestamp_seconds']:.0f}s",
                    file_path=str(clip_path), confidence=moment["confidence"], evidence_frame_path=moment["evidence_frame_path"],
                )
                session.add(clip)
                session.flush()
                for event in stored_events:
                    if event.timestamp_seconds == moment["timestamp_seconds"] and event.clip_id is None:
                        event.clip_id = clip.id
            _mark_stage(session, broadcast.id, current_stage, "completed", f"generated {len(unique_moments)} clips")
            session.commit()
        clips = session.scalars(select(Clip).where(Clip.broadcast_id == broadcast.id).order_by(Clip.id)).all()
        clip_payloads = [{
            "id": clip.id, "title": clip.title, "start_seconds": clip.start_seconds, "end_seconds": clip.end_seconds,
            "timestamp_seconds": clip.timestamp_seconds, "confidence": clip.confidence,
            "evidence_frame_path": clip.evidence_frame_path, "file_path": clip.file_path, "url": f"/clips/{clip.id}",
        } for clip in clips]
        stored_events = session.scalars(select(Event).where(Event.broadcast_id == broadcast.id).order_by(Event.timestamp_seconds, Event.id)).all()
        event_payloads = [_event_payload(event) for event in stored_events]
        derived = [payload for payload in event_payloads if payload["event_type"] in {"possible_break", "possible_retake"}]
        key_moments = choose_key_moments(event_payloads)

        current_stage = "report"
        report = session.scalar(select(Report).where(Report.broadcast_id == broadcast.id))
        if not _stage_completed(session, broadcast.id, current_stage) or report is None:
            _mark_stage(session, broadcast.id, current_stage, "running", "rendering JSON, Markdown, and HTML")
            session.commit()
            data_confidence = sum(event["confidence"] for event in event_payloads) / len(event_payloads)
            if sampling_stats.get("samples"):
                ocr_stability = 1.0 - (sampling_stats.get("stable_reuses", 0) / sampling_stats["samples"] * 0.05)
                data_confidence *= ocr_stability
            report_document = ReportDocument(
            broadcast={"id": broadcast.id, "title": broadcast.title, "platform": broadcast.platform, "video_id": broadcast.video_id},
            match={"id": match.id, "ordinal": 1, "team_a": match.team_a, "team_b": match.team_b},
            map={"id": game_map.id, "ordinal": 1, "mode": game_map.mode, "map_name": game_map.map_name},
            timeline=sorted(event_payloads, key=lambda item: (item["timestamp_seconds"], item["event_type"])),
            key_moments=key_moments,
            possible_breaks_retakes=derived,
            xmwp_timeline=xmwp,
            recommended_clips=clip_payloads,
            hardpoint_summary=hardpoint_summary(observations),
            data_confidence=round(data_confidence, 4),
            data_confidence_evidence={
                "value": round(data_confidence, 4), "timestamp_seconds": observations[-1].timestamp_seconds,
                "confidence": round(data_confidence, 4), "evidence_frame_path": observations[-1].evidence_frame_path,
            },
            known_limitations=[
                "The bundled sample uses deterministic fixture OCR; production scorebar OCR must be calibrated per HUD profile.",
                "Map boundaries are inferred from score reset/Hardpoint target and do not yet use transition-card detection.",
                "xMWP HeuristicV0 is uncalibrated and is explicitly a placeholder for a trained model.",
                "Break/retake events are scoring-flow inferences, not confirmation of hill control or kills.",
                "Deep analytics are implemented only for Hardpoint; SnD and Control are boundary-only in v0.",
                "Hill-by-hill attribution requires stable hill-timer OCR; v0 reports scoring-flow windows without inventing hill IDs.",
            ],
            )
            paths = write_reports(report_document, settings.data_dir / "reports", f"broadcast_{broadcast.id}")
            report = Report(
                broadcast_id=broadcast.id, json_path=str(paths["json"]), markdown_path=str(paths["markdown"]),
                html_path=str(paths["html"]), data_confidence=report_document.data_confidence,
            )
            session.add(report)
            _mark_stage(session, broadcast.id, current_stage, "completed", f"reports written to {settings.data_dir / 'reports'}")
            session.commit()

        current_stage = "store"
        broadcast = session.get(Broadcast, broadcast.id)
        if broadcast.status != BroadcastStatus.processed:
            transition_broadcast(broadcast, BroadcastStatus.processed)
        if not _stage_completed(session, broadcast.id, current_stage):
            _mark_stage(session, broadcast.id, current_stage, "completed", "pipeline checkpoints finalized")
        session.commit()
        return report
    except Exception as exc:
        session.rollback()
        if broadcast is not None:
            broadcast = session.get(Broadcast, broadcast.id)
            if broadcast and broadcast.status != BroadcastStatus.processed:
                if broadcast.status != BroadcastStatus.failed:
                    transition_broadcast(broadcast, BroadcastStatus.failed, str(exc))
                job = _job(session, broadcast.id, current_stage)
                job.status = "failed"
                job.attempt_count += 1
                job.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=min(3600, 2 ** job.attempt_count * 30))
                job.logs += f"{datetime.now(timezone.utc).isoformat()} ERROR {exc}\n"
                session.commit()
        raise
    finally:
        if own_session:
            session.close()


def download_url(url: str, output_dir: Path) -> Path:
    """Explicit opt-in download used only by --url processing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "%(id)s.%(ext)s")
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--sleep-requests", "1", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", template, "--print", "after_move:filepath", url]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {completed.stderr.strip()}")
    return Path(completed.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process a CoD broadcast into evidence-backed analytics")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--file", type=Path, help="local video path (fully offline)")
    inputs.add_argument("--url", help="video URL; explicitly downloads with yt-dlp")
    parser.add_argument("--hud-profile", default="CDL_2026_1080p")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--debug-crops", action="store_true")
    parser.add_argument("--ocr-engine", choices=("stub", "tesseract"), default="stub")
    parser.add_argument("--mode", choices=("auto", "hardpoint", "snd", "control", "unknown"), default="auto")
    args = parser.parse_args(argv)
    init_db()
    path = args.file or download_url(args.url, get_settings().data_dir / "videos")
    report = process_local_file(path.resolve(), args.hud_profile, args.fps, args.debug_crops, args.ocr_engine, args.mode)
    print(json.dumps({"report_id": report.id, "json": report.json_path, "markdown": report.markdown_path, "html": report.html_path}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
