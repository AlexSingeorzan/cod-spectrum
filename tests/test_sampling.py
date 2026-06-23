from __future__ import annotations

import numpy as np

from backend.app.services.hud import load_hud_profile
from backend.app.services.ocr import StubOcrEngine
from backend.app.services.sampling import crop_change_score, iter_video_samples
from backend.app.services.timeline import extract_score_observations


def test_frame_sampling_count_and_cadence(sample_video):
    samples = list(iter_video_samples(sample_video, 1.0))
    assert len(samples) == 16
    assert [item.timestamp_seconds for item in samples[:3]] == [0.0, 1.0, 2.0]


def test_region_change_detection_skips_stable_crop(tmp_path, sample_video):
    profile = load_hud_profile("CDL_2026_1080p")
    observations, stats = extract_score_observations(
        sample_video, profile, StubOcrEngine(), tmp_path, sample_fps=1.0, change_threshold=2.0,
    )
    assert len(observations) == 16
    assert stats["stable_reuses"] >= 1
    stable = np.zeros((20, 20, 3), dtype=np.uint8)
    assert crop_change_score(stable, stable.copy()) == 0

