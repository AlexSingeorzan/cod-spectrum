from __future__ import annotations

from pathlib import Path

import cv2

from ..schemas import ScoreObservation
from .hud import HudProfile, crop_region
from .ocr import OcrEngine, parse_score
from .sampling import crop_change_score, dump_debug_crops, iter_video_samples


def extract_score_observations(
    video_path: Path,
    profile: HudProfile,
    ocr: OcrEngine,
    output_dir: Path,
    sample_fps: float = 1.0,
    change_threshold: float = 2.0,
    debug_crops: bool = False,
) -> tuple[list[ScoreObservation], dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_crop = None
    previous_observation: ScoreObservation | None = None
    observations: list[ScoreObservation] = []
    stats = {"samples": 0, "ocr_calls": 0, "stable_reuses": 0}
    region = profile.regions["scorebar"]
    hints = profile.ocr_hints.get("scorebar", {})
    for sample in iter_video_samples(video_path, sample_fps):
        stats["samples"] += 1
        crop = crop_region(sample.frame, region)
        if debug_crops:
            dump_debug_crops(sample, profile, output_dir / "debug")
        changed = crop_change_score(previous_crop, crop) >= change_threshold
        if changed or previous_observation is None:
            stats["ocr_calls"] += 1
            evidence_path = output_dir / f"scorebar_{sample.index:05d}.jpg"
            cv2.imwrite(str(evidence_path), crop)
            result = ocr.read(crop, {**hints, "sample_index": sample.index, "timestamp_seconds": sample.timestamp_seconds})
            score = parse_score(result)
            if score is None:
                previous_crop = crop.copy()
                continue
            observation = ScoreObservation(
                timestamp_seconds=sample.timestamp_seconds,
                score_a=score[0],
                score_b=score[1],
                confidence=result.confidence,
                evidence_frame_path=str(evidence_path),
                raw_text=result.text,
            )
        else:
            stats["stable_reuses"] += 1
            observation = previous_observation.model_copy(update={"timestamp_seconds": sample.timestamp_seconds})
        observations.append(observation)
        previous_observation = observation
        previous_crop = crop.copy()
    return observations, stats


def build_score_events(
    observations: list[ScoreObservation],
    team_a: str = "TEAM_A",
    team_b: str = "TEAM_B",
    target: int = 250,
) -> list[dict]:
    if not observations:
        return []
    events: list[dict] = []
    first = observations[0]
    events.append(_event("map_start", first, team_a, team_b, min(first.confidence, 0.85)))
    previous = first
    last_nonzero_leader = _leader(first.score_a, first.score_b)
    for observation in observations[1:]:
        if (observation.score_a, observation.score_b) == (previous.score_a, previous.score_b):
            continue
        events.append(_event("score_update", observation, team_a, team_b, observation.confidence))
        leader = _leader(observation.score_a, observation.score_b)
        if leader and last_nonzero_leader and leader != last_nonzero_leader:
            events.append(_event("lead_change", observation, team_a, team_b, observation.confidence * 0.95))
        if leader:
            last_nonzero_leader = leader
        if max(observation.score_a, observation.score_b) >= target:
            events.append(_event("map_end", observation, team_a, team_b, min(observation.confidence, 0.9)))
            break
        previous = observation
    if not any(event["event_type"] == "map_end" for event in events):
        for previous, current in zip(observations, observations[1:]):
            if max(previous.score_a, previous.score_b) > 0 and current.score_a <= 1 and current.score_b <= 1:
                events.append(_event("map_end", previous, team_a, team_b, previous.confidence * 0.65))
                break
    return events


def _leader(score_a: int, score_b: int) -> str | None:
    if score_a == score_b:
        return None
    return "a" if score_a > score_b else "b"


def _event(event_type: str, observation: ScoreObservation, team_a: str, team_b: str, confidence: float) -> dict:
    return {
        "timestamp_seconds": observation.timestamp_seconds,
        "event_type": event_type,
        "team_a": team_a,
        "team_b": team_b,
        "score_a": observation.score_a,
        "score_b": observation.score_b,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "raw_text": observation.raw_text,
        "evidence_frame_path": observation.evidence_frame_path,
        "is_placeholder": False,
    }

