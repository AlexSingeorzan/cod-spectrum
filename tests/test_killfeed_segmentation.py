from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.app.services.killfeed_segmentation import (
    CORE_FIELDS,
    KillfeedSegmenter,
    crop_segment,
)
from scripts.build_killfeed_segments import build_segments
from scripts.eval_killfeed_segments import evaluate


def _row_image() -> np.ndarray:
    image = np.zeros((38, 360, 3), np.uint8)
    image[:] = (28, 31, 34)
    white = (235, 235, 235)
    red = (60, 60, 235)
    cv2.putText(image, "NERO", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, white, 2)
    cv2.putText(image, "SMG", (135, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 2)
    cv2.circle(image, (222, 18), 7, white, 2)
    cv2.putText(image, "ABEZY", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, red, 2)
    return image


def _write_dataset(tmp_path: Path, image: np.ndarray) -> Path:
    dataset = tmp_path / "killfeed_dataset"
    rows_dir = dataset / "rows"
    rows_dir.mkdir(parents=True)
    cv2.imwrite(str(rows_dir / "kf_0000.png"), image)
    (dataset / "manifest.json").write_text(json.dumps({"detector": "synthetic"}))
    row = {
        "id": "kf_0000",
        "video_timestamp_seconds": 88.0,
        "row_image": "rows/kf_0000.png",
        "region_image": None,
        "box": [0, 0, 1, 1],
        "detector": "synthetic",
        "detector_confidence": 1.0,
        "color_hint": {},
        "label": {
            "valid_kill": None,
            "attacker": None,
            "attacker_team": None,
            "victim": None,
            "victim_team": None,
            "weapon": None,
            "headshot": None,
            "is_trade": None,
        },
        "label_source": "unlabeled",
        "labeled_by": None,
        "source_url": "synthetic://segmentation-test",
    }
    (dataset / "annotations.jsonl").write_text(json.dumps(row) + "\n")
    return dataset


def test_segmenter_splits_attacker_weapon_victim_and_headshot():
    image = _row_image()
    segmentation = KillfeedSegmenter().segment(image, sample_id="sample")

    assert segmentation.model_name == "killfeed_segmenter_classical"
    assert segmentation.model_version == "0.1.0"
    assert segmentation.fallback_used is False
    for field in CORE_FIELDS:
        segment = segmentation.segments[field]
        assert segment.box is not None
        assert segment.confidence >= 0.7
        assert crop_segment(image, segment) is not None
    assert segmentation.segments["headshot"].box is not None


def test_segmenter_returns_nulls_for_low_information_row():
    image = np.zeros((38, 360, 3), np.uint8)
    segmentation = KillfeedSegmenter().segment(image, sample_id="blank")

    assert segmentation.confidence == 0.0
    assert "missing_core_segments" in segmentation.failure_reason
    assert all(segmentation.segments[field].box is None for field in CORE_FIELDS)


def test_build_segments_writes_crops_and_eval_metrics(tmp_path):
    dataset = _write_dataset(tmp_path, _row_image())

    segments_path = build_segments(dataset)
    result = evaluate(dataset)

    assert segments_path.exists()
    rows = [json.loads(line) for line in segments_path.read_text().splitlines()]
    assert len(rows) == 1
    for field in CORE_FIELDS:
        crop_path = Path(rows[0]["segments"][field]["crop_path"])
        assert crop_path.exists()
    assert result["metrics"]["complete_core_segments"] == 1
    assert result["metrics"]["field_crop_counts"]["weapon"] == 1
