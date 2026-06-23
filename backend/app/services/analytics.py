from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ..schemas import ScoreObservation, XmwpPoint


class WinProbabilityModel(Protocol):
    """Computes a confidence-aware win probability from a score state."""

    def predict(self, score_a: int, score_b: int) -> float: ...


@dataclass(frozen=True)
class HeuristicV0:
    """Uncalibrated Hardpoint xMWP heuristic; replaceable by a trained model."""

    target: int = 250
    k: float = 0.15

    def predict(self, score_a: int, score_b: int) -> float:
        lead = score_a - score_b
        remaining = max(0, self.target - max(score_a, score_b))
        z = self.k * lead / math.sqrt(remaining + 1)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def xmwp_timeline(observations: list[ScoreObservation], model: WinProbabilityModel) -> list[XmwpPoint]:
    points: list[XmwpPoint] = []
    previous_score: tuple[int, int] | None = None
    for observation in observations:
        score = (observation.score_a, observation.score_b)
        if score == previous_score:
            continue
        points.append(XmwpPoint(
            timestamp_seconds=observation.timestamp_seconds,
            score_a=observation.score_a,
            score_b=observation.score_b,
            probability_a=model.predict(*score),
            confidence=observation.confidence,
            evidence_frame_path=observation.evidence_frame_path,
        ))
        previous_score = score
    return points


def detect_breaks_retakes(
    observations: list[ScoreObservation],
    team_a: str,
    team_b: str,
    debounce_seconds: float = 2.0,
) -> list[dict]:
    scoring_runs: list[dict] = []
    for previous, current in zip(observations, observations[1:]):
        delta_a = current.score_a - previous.score_a
        delta_b = current.score_b - previous.score_b
        if delta_a < 0 or delta_b < 0 or (delta_a > 0 and delta_b > 0):
            continue
        scoring_team = "a" if delta_a > 0 else "b" if delta_b > 0 else None
        if scoring_team is None:
            continue
        if scoring_runs and scoring_runs[-1]["team"] == scoring_team:
            scoring_runs[-1]["end"] = current.timestamp_seconds
            scoring_runs[-1]["points"] += max(delta_a, delta_b)
            scoring_runs[-1]["confidence_values"].append(current.confidence)
        else:
            scoring_runs.append({
                "team": scoring_team,
                "start": current.timestamp_seconds,
                "end": current.timestamp_seconds,
                "points": max(delta_a, delta_b),
                "confidence_values": [current.confidence],
                "observation": current,
            })
    events: list[dict] = []
    for previous_run, current_run in zip(scoring_runs, scoring_runs[1:]):
        duration = current_run["end"] - current_run["start"] + 1.0
        if previous_run["team"] == current_run["team"] or duration < debounce_seconds:
            continue
        observation: ScoreObservation = current_run["observation"]
        stability = min(1.0, duration / max(debounce_seconds * 2, 1.0))
        ocr_confidence = sum(current_run["confidence_values"]) / len(current_run["confidence_values"])
        confidence = round(ocr_confidence * (0.65 + 0.35 * stability), 4)
        lost_team = team_a if previous_run["team"] == "a" else team_b
        gained_team = team_a if current_run["team"] == "a" else team_b
        common = {
            "timestamp_seconds": observation.timestamp_seconds,
            "team_a": team_a,
            "team_b": team_b,
            "score_a": observation.score_a,
            "score_b": observation.score_b,
            "confidence": confidence,
            "raw_text": f"scoring flow changed from {lost_team} to {gained_team}; persisted {duration:.1f}s",
            "evidence_frame_path": observation.evidence_frame_path,
            "is_placeholder": False,
        }
        events.append({**common, "event_type": "possible_break", "player": lost_team})
        events.append({**common, "event_type": "possible_retake", "player": gained_team})
    return events


def hardpoint_summary(observations: list[ScoreObservation]) -> dict:
    if not observations:
        return {}
    changed = [observations[0]]
    for observation in observations[1:]:
        if (observation.score_a, observation.score_b) != (changed[-1].score_a, changed[-1].score_b):
            changed.append(observation)
    leads = [observation.score_a - observation.score_b for observation in changed]
    swings = [abs(current - previous) for previous, current in zip(leads, leads[1:])]
    biggest_index = max(range(len(swings)), key=swings.__getitem__) + 1 if swings else 0
    biggest_observation = changed[biggest_index]
    final = changed[-1]
    return {
        "final_score": {
            "value": {"team_a": final.score_a, "team_b": final.score_b},
            "timestamp_seconds": final.timestamp_seconds, "confidence": final.confidence,
            "evidence_frame_path": final.evidence_frame_path,
        },
        "biggest_swing": {
            "value": max(swings, default=0), "timestamp_seconds": biggest_observation.timestamp_seconds,
            "confidence": biggest_observation.confidence, "evidence_frame_path": biggest_observation.evidence_frame_path,
        },
        "score_states": {
            "value": len(changed), "timestamp_seconds": final.timestamp_seconds,
            "confidence": sum(item.confidence for item in changed) / len(changed),
            "evidence_frame_path": final.evidence_frame_path,
        },
    }


def choose_key_moments(events: list[dict]) -> list[dict]:
    preferred = {"lead_change", "possible_break", "possible_retake", "map_end"}
    moments = [event for event in events if event["event_type"] in preferred]
    return sorted(moments, key=lambda item: item["timestamp_seconds"])
