from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.services.kill_type_recognition import KILL_TYPE_CATEGORIES
from scripts.review_kill_type_dataset import (
    apply_label,
    main,
    prune_missing_crops,
    select_rows,
    summarize,
    validate_dataset,
    write_contact_sheet,
)


def _icon() -> np.ndarray:
    image = np.zeros((34, 68, 3), np.uint8)
    image[:] = (18, 20, 22)
    cv2.line(image, (9, 17), (54, 17), (235, 235, 235), 3)
    return image


def _row(idx: int, *, labelled: bool = False, bad: bool = False) -> dict:
    if labelled:
        label = {
            "valid_kill_type": True,
            "kill_type": "gun",
            "exact_weapon": None,
            "unclear": False,
        }
        return {
            "id": f"kf_{idx:04d}",
            "video_timestamp_seconds": float(idx),
            "icon_image": f"icons/kf_{idx:04d}.png",
            "segment_box": [0.2, 0.1, 0.2, 0.8],
            "segment_confidence": 0.8,
            "human_review_status": "reviewed",
            "label": label,
            "label_source": "manual_label",
            "labeled_by": "alex",
        }
    label = {
        "valid_kill_type": True if bad else None,
        "kill_type": None,
        "exact_weapon": None,
        "unclear": None,
    }
    return {
        "id": f"kf_{idx:04d}",
        "video_timestamp_seconds": float(idx),
        "icon_image": f"icons/kf_{idx:04d}.png",
        "segment_box": [0.2, 0.1, 0.2, 0.8],
        "segment_confidence": 0.8,
        "human_review_status": "unreviewed",
        "label": label,
        "label_source": "unlabeled",
        "labeled_by": None,
    }


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    dataset = tmp_path / "kill_type_dataset"
    icons = dataset / "icons"
    icons.mkdir(parents=True)
    for row in rows:
        cv2.imwrite(str(dataset / row["icon_image"]), _icon())
    (dataset / "manifest.json").write_text(json.dumps({
        "version": 1,
        "kind": "kill_type_icon_dataset",
        "dataset_id": "review_test",
        "categories": list(KILL_TYPE_CATEGORIES),
    }))
    (dataset / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    return dataset


def test_review_summary_counts_unreviewed_and_labelled_rows(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1), _row(2, labelled=True)])

    summary = summarize(dataset)

    assert summary["validation"]["ok"] is True
    assert summary["total_rows"] == 2
    assert summary["reviewed_rows"] == 1
    assert summary["unreviewed_rows"] == 1
    assert summary["labelled_training_rows"] == 1
    assert summary["kill_type_counts"] == {"gun": 1}


def test_review_validation_rejects_partial_unlabeled_rows(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1, bad=True)])

    report = validate_dataset(dataset)

    assert report.ok is False
    assert any("unlabeled rows must not contain partial labels" in error for error in report.errors)
    assert [row["id"] for row in select_rows(dataset, "invalid")] == ["kf_0001"]


def test_prune_missing_crops_removes_stale_rows_and_updates_manifest(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1), _row(2), _row(3, labelled=True)])
    (dataset / "icons" / "kf_0002.png").unlink()

    before = prune_missing_crops(dataset, dry_run=True)
    result = prune_missing_crops(dataset)

    rows = [
        json.loads(line)
        for line in (dataset / "annotations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads((dataset / "manifest.json").read_text())

    assert before["written"] is False
    assert before["pruned_ids"] == ["kf_0002"]
    assert result["written"] is True
    assert result["original_rows"] == 3
    assert result["kept_rows"] == 2
    assert result["pruned_ids"] == ["kf_0002"]
    assert [row["id"] for row in rows] == ["kf_0001", "kf_0003"]
    assert manifest["icon_count"] == 2
    assert manifest["label_status"] == "labeled"
    assert manifest["pruned_missing_icon_count"] == 1
    assert manifest["pruned_missing_icon_ids"] == ["kf_0002"]
    assert summarize(dataset)["validation"]["ok"] is True


def test_prune_missing_cli_preserves_existing_audit_when_no_rows_are_pruned(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1), _row(2)])
    audit_path = tmp_path / "pruned_missing_icons.json"
    audit_path.write_text(json.dumps({"previous": True}) + "\n")

    exit_code = main([
        "--dataset",
        str(dataset),
        "prune-missing",
        "--write-json",
        str(audit_path),
    ])

    assert exit_code == 0
    assert json.loads(audit_path.read_text()) == {"previous": True}


def test_apply_label_updates_one_row_and_keeps_jsonl_valid(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1), _row(2)])

    updated = apply_label(
        dataset,
        "kf_0001",
        reviewed_by="alex",
        valid=True,
        kill_type="grenade",
        unclear=False,
        exact_weapon=None,
        label_source="manual_label",
        reviewed_at="2026-06-27T12:00:00Z",
    )

    assert updated["human_review_status"] == "reviewed"
    assert updated["reviewed_at"] == "2026-06-27T12:00:00Z"
    assert updated["label"]["kill_type"] == "grenade"
    assert summarize(dataset)["kill_type_counts"] == {"grenade": 1}
    assert len(select_rows(dataset, "unreviewed")) == 1


def test_apply_label_rejects_guessy_invalid_combinations(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1)])

    with pytest.raises(ValueError, match="valid=true requires --kill-type"):
        apply_label(
            dataset,
            "kf_0001",
            reviewed_by="alex",
            valid=True,
            kill_type=None,
            unclear=False,
            exact_weapon=None,
            label_source="manual_label",
        )


def test_contact_sheet_writes_selected_rows(tmp_path):
    dataset = _write_dataset(tmp_path, [_row(1), _row(2), _row(3, labelled=True)])
    out = tmp_path / "sheet.png"

    write_contact_sheet(dataset, out, select_rows(dataset, "unreviewed"), columns=2)

    assert out.exists()
    assert out.stat().st_size > 0
    image = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[0] > 0 and image.shape[1] > 0
