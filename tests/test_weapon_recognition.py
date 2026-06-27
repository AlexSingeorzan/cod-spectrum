from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.app.events import SourceKind
from backend.app.services.weapon_recognition import (
    WeaponRecognizer,
    weapon_event_from_prediction,
)
from scripts.build_weapon_dataset import build_weapon_dataset
from scripts.eval_weapon_recognition import evaluate


def _icon(kind: str) -> np.ndarray:
    image = np.zeros((32, 64, 3), np.uint8)
    image[:] = (18, 20, 22)
    white = (235, 235, 235)
    if kind == "AR":
        cv2.line(image, (10, 15), (52, 15), white, 3)
        cv2.line(image, (12, 17), (5, 23), white, 3)
        cv2.line(image, (32, 17), (42, 25), white, 2)
    elif kind == "SMG":
        cv2.rectangle(image, (17, 10), (43, 17), white, -1)
        cv2.line(image, (25, 18), (21, 27), white, 2)
        cv2.line(image, (43, 14), (52, 14), white, 2)
    else:
        cv2.putText(image, kind[:2], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 2)
    return image


def _write_weapon_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset = tmp_path / "weapon_dataset"
    icons_dir = dataset / "icons"
    icons_dir.mkdir(parents=True)
    for row in rows:
        cv2.imwrite(str(dataset / row["icon_image"]), _icon(row["draw"]))
        row.pop("draw")
    (dataset / "manifest.json").write_text(json.dumps({
        "version": 1,
        "kind": "weapon_icon_dataset",
        "dataset_id": "test_weapon_dataset",
        "label_status": "labeled",
    }))
    (dataset / "annotations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return dataset


def _weapon_row(idx: int, weapon: str | None, *, labelled: bool = True, draw: str = "AR") -> dict:
    return {
        "id": f"weapon_{idx:04d}",
        "video_timestamp_seconds": float(idx),
        "icon_image": f"icons/weapon_{idx:04d}.png",
        "source_crop_path": f"icons/weapon_{idx:04d}.png",
        "source_row_image": f"rows/kf_{idx:04d}.png",
        "source_segments": "segments.jsonl",
        "source_url": "synthetic://weapon-test",
        "segment_box": [0.25, 0.2, 0.2, 0.6],
        "segment_confidence": 0.9,
        "label": {
            "valid_weapon": labelled,
            "weapon": weapon,
            "weapon_family": weapon,
            "unclear": False if labelled else None,
        },
        "label_source": "manual_label" if labelled else "unlabeled",
        "labeled_by": "alex" if labelled else None,
        "draw": draw,
    }


def test_weapon_dataset_builder_uses_only_referenced_segment_crops(tmp_path):
    killfeed = tmp_path / "killfeed_dataset"
    segment_dir = killfeed / "segments"
    segment_dir.mkdir(parents=True)
    referenced = segment_dir / "kf_0001_weapon.png"
    stale = segment_dir / "kf_stale_weapon.png"
    cv2.imwrite(str(referenced), _icon("AR"))
    cv2.imwrite(str(stale), _icon("SMG"))
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
    out = tmp_path / "weapon_dataset"

    build_weapon_dataset(killfeed, out)

    rows = [json.loads(line) for line in (out / "annotations.jsonl").read_text().splitlines()]
    icons = list((out / "icons").glob("*.png"))
    assert len(rows) == 1
    assert len(icons) == 1
    assert rows[0]["source_url"] == "synthetic://source"
    assert rows[0]["label"]["weapon"] is None


def test_weapon_recognizer_abstains_without_labels(tmp_path):
    dataset = _write_weapon_dataset(
        tmp_path,
        [_weapon_row(1, None, labelled=False, draw="AR")],
    )
    recognizer = WeaponRecognizer(dataset)

    prediction = recognizer.read_row(json.loads((dataset / "annotations.jsonl").read_text()))

    assert prediction.weapon is None
    assert prediction.failure_reason == "no_labeled_weapon_templates"
    assert prediction.fallback_used is False


def test_weapon_recognizer_reads_labelled_icon_and_emits_event(tmp_path):
    rows = [
        _weapon_row(1, "AR", draw="AR"),
        _weapon_row(2, "SMG", draw="SMG"),
    ]
    dataset = _write_weapon_dataset(tmp_path, rows)
    annotations = [json.loads(line) for line in (dataset / "annotations.jsonl").read_text().splitlines()]
    recognizer = WeaponRecognizer(dataset, confidence_threshold=0.0)

    prediction = recognizer.read_row(annotations[0])
    event = weapon_event_from_prediction(prediction, player="HyDra", team="LAT")

    assert prediction.weapon == "AR"
    assert prediction.confidence > 0.0
    assert event is not None
    assert event.event_type == "weapon"
    assert event.payload.weapon == "AR"
    assert event.payload.player == "HyDra"
    assert event.evidence.has_visual()
    assert event.provenance.source == SourceKind.MODEL


def test_weapon_eval_reports_no_accuracy_without_labels(tmp_path):
    dataset = _write_weapon_dataset(
        tmp_path,
        [_weapon_row(1, None, labelled=False, draw="AR")],
    )

    result = evaluate(dataset)

    assert result["dataset"]["labelled_weapon_icons"] == 0
    assert result["models"]["template"]["operational_gallery"] is None
    assert "no weapon-recognition accuracy claim" in result["models"]["template"]["status"]


def test_weapon_eval_compares_template_and_histogram_on_labelled_set(tmp_path):
    rows = [
        _weapon_row(1, "AR", draw="AR"),
        _weapon_row(2, "AR", draw="AR"),
        _weapon_row(3, "SMG", draw="SMG"),
        _weapon_row(4, "SMG", draw="SMG"),
    ]
    dataset = _write_weapon_dataset(tmp_path, rows)

    result = evaluate(dataset)

    assert set(result["models"]) == {"template", "histogram"}
    assert result["models"]["template"]["operational_gallery"]["top1_accuracy"] == 1.0
    assert result["models"]["histogram"]["operational_gallery"]["top1_accuracy"] == 1.0
