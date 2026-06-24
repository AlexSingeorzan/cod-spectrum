from __future__ import annotations

import cv2
import numpy as np

from backend.app.services import minimap as mm


def test_detector_localises_colored_markers():
    img = np.full((120, 120, 3), 18, np.uint8)          # dark minimap-ish
    cv2.circle(img, (30, 30), 5, (0, 0, 255), -1)        # red marker (BGR)
    cv2.circle(img, (85, 85), 5, (255, 255, 255), -1)    # white marker
    dets = mm.ClassicalMinimapDetector()._detect_crop(img)
    assert len(dets) >= 2
    assert {d.team for d in dets} == {"enemy", "observed"}
    # red marker classified enemy and located near where it was drawn
    red = next(d for d in dets if d.team == "enemy")
    assert abs(red.x - 30 / 120) < 0.08 and abs(red.y - 30 / 120) < 0.08


def test_detector_returns_serialisable_dicts():
    img = np.full((120, 120, 3), 18, np.uint8)
    cv2.circle(img, (40, 40), 5, (0, 0, 255), -1)
    full = {"regions": {"minimap": {"x": 0, "y": 0, "w": 1, "h": 1}}}
    out = mm.ClassicalMinimapDetector().detect(img, full)
    assert out and set(out[0]) == {"x", "y", "team", "confidence", "area"}
    assert all(0.0 <= d["x"] <= 1.0 and 0.0 <= d["y"] <= 1.0 for d in out)


def test_occupancy_heatmap_normalises_to_peak_one():
    fd = [
        [{"x": 0.1, "y": 0.1, "team": "observed", "confidence": 0.4, "area": 20}],
        [{"x": 0.1, "y": 0.1, "team": "observed", "confidence": 0.4, "area": 20}],
        [{"x": 0.9, "y": 0.9, "team": "enemy", "confidence": 0.5, "area": 20}],
    ]
    hm = mm.occupancy_heatmap(fd, size=8)
    assert hm.detections == 3 and hm.frames == 3
    assert max(max(r) for r in hm.grid) == 1.0          # the (0.1,0.1) cell hit twice -> peak


def test_saved_heatmaps_is_a_dict():
    assert isinstance(mm.saved_heatmaps(), dict)
