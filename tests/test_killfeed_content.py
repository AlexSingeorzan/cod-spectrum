from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.app.events import SourceKind
from backend.app.services.killfeed_content import (
    KillfeedContentReader,
    events_from_content_reads,
    manual_read_from_row,
)
from scripts.eval_killfeed_content import evaluate


def _row_image(attacker: str, victim: str, weapon: str, attacker_team: str = "LAT", victim_team: str = "VAN"):
    image = np.zeros((38, 360, 3), np.uint8)
    image[:] = (30, 34, 38)
    red, blue, white = (60, 60, 235), (235, 150, 40), (235, 235, 235)
    atk_color = red if attacker_team == "LAT" else blue
    vic_color = red if victim_team == "LAT" else blue
    cv2.putText(image, attacker, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, atk_color, 2)
    cv2.putText(image, weapon, (142, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
    cv2.putText(image, victim, (225, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, vic_color, 2)
    return image


def _annotation(
    idx: int,
    *,
    valid_kill=True,
    attacker="HyDra",
    attacker_team="LAT",
    victim="Lunarz",
    victim_team="VAN",
    kill_type="gun",
    weapon="SMG",
    headshot=False,
    is_trade=False,
    label_source="manual_label",
    labeled_by="alex",
):
    return {
        "id": f"kf_content_{idx:04d}",
        "video_timestamp_seconds": float(idx),
        "row_image": f"rows/kf_content_{idx:04d}.png",
        "region_image": None,
        "box": [0, 0, 1, 1],
        "detector": "synthetic_fixture",
        "detector_confidence": 1.0,
        "color_hint": {},
        "label": {
            "valid_kill": valid_kill,
            "attacker": attacker,
            "attacker_team": attacker_team,
            "victim": victim,
            "victim_team": victim_team,
            "kill_type": kill_type,
            "weapon": weapon,
            "headshot": headshot,
            "is_trade": is_trade,
        },
        "label_source": label_source,
        "labeled_by": labeled_by,
        "source_url": "synthetic://test",
    }


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset = tmp_path / "killfeed_content"
    rows_dir = dataset / "rows"
    rows_dir.mkdir(parents=True)
    for row in rows:
        label = row["label"]
        if label.get("attacker") and label.get("victim") and label.get("weapon"):
            image = _row_image(
                label["attacker"],
                label["victim"],
                label["weapon"],
                label.get("attacker_team") or "LAT",
                label.get("victim_team") or "VAN",
            )
        else:
            image = np.zeros((38, 360, 3), np.uint8)
        cv2.imwrite(str(dataset / row["row_image"]), image)
    (dataset / "manifest.json").write_text(json.dumps({
        "detector": "synthetic_fixture",
        "source_url": "synthetic://test",
        "label_status": "labeled",
    }))
    (dataset / "annotations.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return dataset


def test_content_reader_abstains_without_labels(tmp_path):
    row = _annotation(1, valid_kill=None, attacker=None, victim=None, weapon=None, label_source="unlabeled", labeled_by=None)
    dataset = _write_dataset(tmp_path, [row])

    reader = KillfeedContentReader(dataset)

    assert reader.label_count == 0
    assert reader.read_annotation(row) is None


def test_content_reader_reads_gallery_from_labeled_rows(tmp_path):
    rows = [
        _annotation(1, attacker="HyDra", victim="Lunarz", weapon="SMG"),
        _annotation(2, attacker="Mamba", attacker_team="VAN", victim="HyDra", victim_team="LAT", weapon="AR"),
    ]
    dataset = _write_dataset(tmp_path, rows)

    reader = KillfeedContentReader(dataset, confidence_threshold=0.0)
    read = reader.read_annotation(rows[0])

    assert read is not None
    assert read.attacker == "HyDra"
    assert read.victim == "Lunarz"
    assert read.kill_type == "gun"
    assert read.weapon == "SMG"
    assert read.source == "model"
    assert read.confidence == 1.0


def test_manual_reads_emit_kill_death_weapon_and_trade_events(tmp_path):
    rows = [
        _annotation(88, attacker="HyDra", attacker_team="LAT", victim="Lunarz", victim_team="VAN", weapon="SMG"),
        _annotation(91, attacker="Mamba", attacker_team="VAN", victim="HyDra", victim_team="LAT", weapon="AR", is_trade=True),
    ]
    dataset = _write_dataset(tmp_path, rows)
    reads = [manual_read_from_row(dataset, row) for row in rows]

    events = events_from_content_reads([read for read in reads if read is not None])

    assert [event.event_type for event in events] == [
        "kill", "death", "weapon", "kill", "death", "weapon", "trade",
    ]
    trade = events[-1]
    kill_type_events = [event for event in events if event.event_type == "weapon"]
    assert [event.payload.kill_type for event in kill_type_events] == ["gun", "gun"]
    assert trade.payload.dead_player == "Lunarz"
    assert trade.payload.trading_player == "Mamba"
    assert trade.payload.trade_window_seconds == 3.0
    assert all(event.evidence.has_visual() for event in events)
    assert all(event.provenance.source == SourceKind.MANUAL_LABEL for event in events)


def test_content_eval_reports_no_accuracy_without_labels(tmp_path):
    row = _annotation(1, valid_kill=None, attacker=None, victim=None, weapon=None, label_source="unlabeled", labeled_by=None)
    dataset = _write_dataset(tmp_path, [row])

    result = evaluate(dataset)

    assert result["dataset"]["content_labeled_rows"] == 0
    assert "no content-reader accuracy claim" in result["metrics"]["reader"]["status"]
    assert result["metrics"]["reader"]["operational_gallery"] is None


def test_content_eval_computes_gallery_and_leave_one_out(tmp_path):
    rows = [
        _annotation(1, attacker="HyDra", victim="Lunarz", weapon="SMG"),
        _annotation(2, attacker="HyDra", victim="Lunarz", weapon="SMG"),
    ]
    dataset = _write_dataset(tmp_path, rows)

    result = evaluate(dataset)

    reader = result["metrics"]["reader"]
    assert reader["status"] == "labeled"
    assert reader["operational_gallery"]["exact_accuracy"] == 1.0
    assert reader["leave_one_out"]["exact_accuracy"] == 1.0
