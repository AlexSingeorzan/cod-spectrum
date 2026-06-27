from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.app.events import SourceKind
from backend.app.services.kill_type_recognition import (
    KILL_TYPE_CATEGORIES,
    KillTypeRecognizer,
    kill_type_event_from_prediction,
)
from scripts.build_kill_type_dataset import build_kill_type_dataset
from scripts.eval_kill_type_recognition import evaluate


def _icon(kind: str) -> np.ndarray:
    image = np.zeros((34, 68, 3), np.uint8)
    image[:] = (18, 20, 22)
    white = (235, 235, 235)
    if kind == "gun":
        cv2.line(image, (9, 17), (54, 17), white, 3)
        cv2.line(image, (13, 19), (6, 25), white, 3)
        cv2.line(image, (33, 19), (43, 27), white, 2)
    elif kind == "grenade":
        cv2.circle(image, (34, 17), 9, white, 2)
        cv2.rectangle(image, (28, 6), (40, 10), white, -1)
        cv2.line(image, (41, 6), (49, 2), white, 2)
    elif kind == "killstreak":
        pts = np.array([[34, 5], [43, 24], [34, 20], [25, 24]], np.int32)
        cv2.fillPoly(image, [pts], white)
        cv2.circle(image, (34, 17), 12, white, 1)
    else:
        cv2.putText(image, kind[:2], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 2)
    return image


def _write_kill_type_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset = tmp_path / "kill_type_dataset"
    icons_dir = dataset / "icons"
    icons_dir.mkdir(parents=True)
    for row in rows:
        cv2.imwrite(str(dataset / row["icon_image"]), _icon(row["draw"]))
        row.pop("draw")
    (dataset / "manifest.json").write_text(json.dumps({
        "version": 1,
        "kind": "kill_type_icon_dataset",
        "dataset_id": "test_kill_type_dataset",
        "categories": list(KILL_TYPE_CATEGORIES),
        "label_status": "labeled",
    }))
    (dataset / "annotations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return dataset


def _kill_type_row(idx: int, kill_type: str | None, *, labelled: bool = True, draw: str = "gun") -> dict:
    return {
        "id": f"kill_type_{idx:04d}",
        "video_timestamp_seconds": float(idx),
        "icon_image": f"icons/kill_type_{idx:04d}.png",
        "source_crop_path": f"icons/kill_type_{idx:04d}.png",
        "source_row_image": f"rows/kf_{idx:04d}.png",
        "source_segments": "segments.jsonl",
        "source_url": "synthetic://kill-type-test",
        "segment_box": [0.25, 0.2, 0.2, 0.6],
        "segment_confidence": 0.9,
        "label": {
            "valid_kill_type": labelled,
            "kill_type": kill_type,
            "exact_weapon": None,
            "unclear": False if labelled else None,
        },
        "label_source": "manual_label" if labelled else "unlabeled",
        "labeled_by": "alex" if labelled else None,
        "draw": draw,
    }


def test_kill_type_categories_include_killstreak():
    assert "killstreak" in KILL_TYPE_CATEGORIES
    assert "unknown" in KILL_TYPE_CATEGORIES


def test_kill_type_dataset_builder_uses_only_referenced_segment_crops(tmp_path):
    killfeed = tmp_path / "killfeed_dataset"
    segment_dir = killfeed / "segments"
    segment_dir.mkdir(parents=True)
    referenced = segment_dir / "kf_0001_weapon.png"
    stale = segment_dir / "kf_stale_weapon.png"
    cv2.imwrite(str(referenced), _icon("gun"))
    cv2.imwrite(str(stale), _icon("grenade"))
    (killfeed / "annotations.jsonl").write_text(json.dumps({
        "id": "kf_0001",
        "source_url": "synthetic://source",
    }) + "\n")
    (killfeed / "segments.jsonl").write_text(json.dumps({
        "sample_id": "kf_0001",
        "video_timestamp_seconds": 12.0,
        "row_image": "rows/kf_0001.png",
        "model_name": "killfeed_segmenter_classical",
        "model_version": "0.1.0",
        "segments": {
            "weapon": {
                "box": [0.2, 0.1, 0.2, 0.8],
                "confidence": 0.8,
                "crop_path": str(referenced),
            }
        },
    }) + "\n")
    out = tmp_path / "kill_type_dataset"

    build_kill_type_dataset(killfeed, out)

    rows = [json.loads(line) for line in (out / "annotations.jsonl").read_text().splitlines()]
    icons = list((out / "icons").glob("*.png"))
    assert len(rows) == 1
    assert len(icons) == 1
    assert rows[0]["source_url"] == "synthetic://source"
    assert rows[0]["label"]["kill_type"] is None
    assert "killstreak" in json.loads((out / "manifest.json").read_text())["categories"]


def test_kill_type_recognizer_abstains_without_labels(tmp_path):
    dataset = _write_kill_type_dataset(
        tmp_path,
        [_kill_type_row(1, None, labelled=False, draw="gun")],
    )
    recognizer = KillTypeRecognizer(dataset)

    prediction = recognizer.read_row(json.loads((dataset / "annotations.jsonl").read_text()))

    assert prediction.kill_type is None
    assert prediction.as_dict()["kill_type"] is None
    assert prediction.failure_reason == "no_labeled_kill_type_templates"
    assert prediction.fallback_used is False


def test_reviewed_unknown_is_a_supported_kill_type_label(tmp_path):
    rows = [
        _kill_type_row(1, "unknown", draw="unknown"),
        _kill_type_row(2, "gun", draw="gun"),
    ]
    dataset = _write_kill_type_dataset(tmp_path, rows)

    recognizer = KillTypeRecognizer(dataset, confidence_threshold=0.0)

    assert recognizer.label_count == 2
    assert recognizer.classes == ["gun", "unknown"]


def test_kill_type_recognizer_reads_labelled_icon_and_emits_event(tmp_path):
    rows = [
        _kill_type_row(1, "gun", draw="gun"),
        _kill_type_row(2, "grenade", draw="grenade"),
        _kill_type_row(3, "killstreak", draw="killstreak"),
    ]
    dataset = _write_kill_type_dataset(tmp_path, rows)
    annotations = [json.loads(line) for line in (dataset / "annotations.jsonl").read_text().splitlines()]
    recognizer = KillTypeRecognizer(dataset, confidence_threshold=0.0)

    prediction = recognizer.read_row(annotations[2])
    event = kill_type_event_from_prediction(prediction, player="HyDra", team="LAT")

    assert prediction.kill_type == "killstreak"
    assert prediction.confidence > 0.0
    assert event is not None
    assert event.event_type == "weapon"
    assert event.payload.kill_type == "killstreak"
    assert event.payload.weapon is None
    assert event.payload.player == "HyDra"
    assert event.evidence.has_visual()
    assert event.provenance.source == SourceKind.MODEL


def test_kill_type_eval_reports_no_accuracy_without_labels(tmp_path):
    dataset = _write_kill_type_dataset(
        tmp_path,
        [_kill_type_row(1, None, labelled=False, draw="gun")],
    )

    result = evaluate(dataset)

    assert result["dataset"]["labelled_kill_type_icons"] == 0
    assert result["models"]["template"]["operational_gallery"] is None
    assert "no kill-type accuracy claim" in result["models"]["template"]["status"]


def test_kill_type_eval_compares_template_and_histogram_on_labelled_set(tmp_path):
    rows = [
        _kill_type_row(1, "gun", draw="gun"),
        _kill_type_row(2, "gun", draw="gun"),
        _kill_type_row(3, "grenade", draw="grenade"),
        _kill_type_row(4, "grenade", draw="grenade"),
        _kill_type_row(5, "killstreak", draw="killstreak"),
        _kill_type_row(6, "killstreak", draw="killstreak"),
    ]
    dataset = _write_kill_type_dataset(tmp_path, rows)

    result = evaluate(dataset)

    assert set(result["models"]) == {"template", "histogram"}
    assert result["models"]["template"]["operational_gallery"]["top1_accuracy"] == 1.0
    assert result["models"]["histogram"]["operational_gallery"]["top1_accuracy"] == 1.0
