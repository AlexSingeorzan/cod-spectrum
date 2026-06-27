from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.events import Evidence, GameEvent, Provenance, ScoreUpdateEvent, SourceKind
from backend.app.services.hud import crop_region, load_hud_profile, region_pixels
from backend.app.services.ocr import OcrEngine, StubOcrEngine


def test_hud_profile_fractional_crops_are_resolution_independent():
    profile = load_hud_profile("CDL_2026_1080p")
    region = profile.regions["scorebar"]
    assert region_pixels(region, 1920, 1080) == (768, 22, 384, 54)
    assert region_pixels(region, 1280, 720) == (512, 14, 256, 36)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert crop_region(frame, region).shape == (36, 256, 3)


def test_stub_ocr_is_deterministic_and_conforms_to_protocol():
    engine = StubOcrEngine()
    assert isinstance(engine, OcrEngine)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    first = engine.read(image, {"sample_index": 4})
    second = engine.read(image, {"sample_index": 4})
    assert first == second
    assert first.text == "6 5"
    assert first.is_placeholder is True


def test_event_evidence_invariant():
    base = dict(
        broadcast_id=1, video_timestamp_seconds=1.0, confidence=0.9,
        provenance=Provenance(source=SourceKind.HEURISTIC),
        payload=ScoreUpdateEvent(team_a="A", team_b="B", score_a=1, score_b=0),
    )
    with pytest.raises(ValidationError):
        GameEvent(**base, evidence=Evidence(video_timestamp_seconds=1.0))  # fact with no frame/crop
    event = GameEvent(**base, evidence=Evidence(video_timestamp_seconds=1.0, frame_path="crop.jpg"))
    assert event.evidence.frame_path == "crop.jpg"

