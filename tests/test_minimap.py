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
    assert out and set(out[0]) == {"x", "y", "w", "h", "team", "confidence", "area"}
    assert all(0.0 <= d["x"] <= 1.0 and 0.0 <= d["y"] <= 1.0 for d in out)


def test_read_frame_exposes_model_contract_and_visibility():
    img = np.full((120, 120, 3), 18, np.uint8)
    cv2.circle(img, (30, 30), 5, (0, 0, 255), -1)
    cv2.circle(img, (85, 85), 5, (255, 255, 255), -1)
    full = {"regions": {"minimap": {"x": 0, "y": 0, "w": 1, "h": 1}}}

    result = mm.ClassicalMinimapDetector().read_frame(
        img,
        full,
        frame_index=42,
        video_timestamp_seconds=12.5,
        frame_path="data/frames/test.png",
        crop_path="data/crops/minimap.png",
        observed_team="LAT",
    )

    assert result.accepted is True
    assert result.model_name == mm.MODEL_NAME
    assert result.model_version == mm.MODEL_VERSION
    assert result.training_dataset == mm.TRAINING_DATASET
    assert result.failure_reason is None
    assert result.fallback_used is False
    assert result.latency_ms >= 0
    assert {d.visibility for d in result.detections} == {"observed_team", "radar_visible_enemy"}
    assert all(d.observed_team == "LAT" for d in result.detections)
    assert result.as_dict()["detections"][0]["bbox_xywh_norm"]


def test_position_events_preserve_evidence_and_do_not_infer_hidden_opponents():
    img = np.full((120, 120, 3), 18, np.uint8)
    cv2.circle(img, (30, 30), 5, (0, 0, 255), -1)
    cv2.circle(img, (85, 85), 5, (255, 255, 255), -1)
    full = {"regions": {"minimap": {"x": 0, "y": 0, "w": 1, "h": 1}}}
    result = mm.ClassicalMinimapDetector().read_frame(
        img,
        full,
        frame_index=7,
        video_timestamp_seconds=88.0,
        frame_path="data/frames/mm_0007.png",
        crop_path="data/crops/mm_0007.png",
        observed_team="LAT",
    )

    events = mm.position_events_from_minimap_result(result, confidence_threshold=0.0, map_id=1)

    assert len(events) >= 2
    observed = next(e for e in events if e.payload.attributes["visibility"] == "observed_team")
    enemy = next(e for e in events if e.payload.attributes["visibility"] == "radar_visible_enemy")
    assert observed.event_type == "position"
    assert observed.payload.team == "LAT"
    assert observed.payload.observed_team == "LAT"
    assert observed.evidence.frame_index == 7
    assert observed.evidence.crop_path == "data/crops/mm_0007.png"
    assert observed.payload.attributes["bbox_xywh_norm"]
    assert enemy.payload.team is None
    assert enemy.payload.observed_team == "LAT"
    assert enemy.payload.attributes["team_marker"] == "enemy"


def test_position_events_abstain_without_visual_evidence_or_confidence():
    img = np.full((120, 120, 3), 18, np.uint8)
    cv2.circle(img, (30, 30), 5, (0, 0, 255), -1)
    full = {"regions": {"minimap": {"x": 0, "y": 0, "w": 1, "h": 1}}}
    result = mm.ClassicalMinimapDetector().read_frame(img, full, observed_team="LAT")

    assert mm.position_events_from_minimap_result(result, confidence_threshold=0.0) == []
    with_evidence = mm.ClassicalMinimapDetector().read_frame(
        img,
        full,
        crop_path="data/crops/mm.png",
        observed_team="LAT",
    )
    assert mm.position_events_from_minimap_result(with_evidence, confidence_threshold=0.99) == []


def test_read_frame_abstains_on_unreadable_input():
    result = mm.ClassicalMinimapDetector().read_frame(np.zeros((0, 0, 3), np.uint8))

    assert result.detections == []
    assert result.failure_reason == "frame_unreadable"


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
