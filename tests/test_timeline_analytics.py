from __future__ import annotations

from backend.app.schemas import ScoreObservation
from backend.app.services.analytics import HeuristicV0, detect_breaks_retakes, hardpoint_summary, xmwp_timeline
from backend.app.services.timeline import build_score_events


def observation(timestamp: float, a: int, b: int, confidence: float = 0.95) -> ScoreObservation:
    return ScoreObservation(timestamp_seconds=timestamp, score_a=a, score_b=b, confidence=confidence, evidence_frame_path=f"{timestamp}.jpg")


def test_map_boundaries_and_lead_changes():
    values = [observation(0, 0, 0), observation(1, 5, 0), observation(2, 5, 8), observation(3, 250, 8)]
    events = build_score_events(values, target=250)
    types = [event["event_type"] for event in events]
    assert types[0] == "map_start"
    assert "lead_change" in types
    assert types[-1] == "map_end"
    assert all(event["evidence_frame_path"] for event in events)


def test_map_end_detected_on_nonzero_to_zero_reset():
    values = [observation(0, 0, 0), observation(1, 20, 10), observation(10, 0, 0)]
    events = build_score_events(values)
    assert any(event["event_type"] == "map_end" and event["timestamp_seconds"] == 1 for event in events)


def test_break_retake_debounce_accepts_persistent_flip_and_rejects_noise():
    persistent = [observation(0, 0, 0), observation(1, 2, 0), observation(2, 4, 0), observation(3, 4, 2), observation(4, 4, 4)]
    events = detect_breaks_retakes(persistent, "A", "B", debounce_seconds=2)
    assert [event["event_type"] for event in events] == ["possible_break", "possible_retake"]
    noisy = [observation(0, 0, 0), observation(1, 2, 0), observation(2, 2, 1), observation(3, 4, 1)]
    assert detect_breaks_retakes(noisy, "A", "B", debounce_seconds=2) == []


def test_xmwp_bounds_and_late_lead_weight():
    model = HeuristicV0()
    early = model.predict(20, 10)
    late = model.predict(240, 230)
    assert 0 < early < 1
    assert 0 < late < 1
    assert late > early > 0.5
    points = xmwp_timeline([observation(0, 20, 10)], model)
    assert points[0].evidence_frame_path == "0.jpg"


def test_hardpoint_metrics_each_carry_evidence():
    summary = hardpoint_summary([observation(0, 0, 0), observation(1, 5, 0)])
    assert summary
    assert all(metric["timestamp_seconds"] >= 0 for metric in summary.values())
    assert all(metric["confidence"] > 0 and metric["evidence_frame_path"] for metric in summary.values())
