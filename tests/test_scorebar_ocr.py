from __future__ import annotations

import json

import cv2

from backend.app.services.ocr import OcrEngine, build_ocr_engine
from backend.app.services.scorebar_ocr import CdlScorebarOcrEngine
from scripts.build_scorebar_ocr_dataset import build_dataset
from scripts.eval_scorebar_ocr import evaluate


def test_build_scorebar_dataset_from_verified_labels(tmp_path):
    dataset_dir = tmp_path / "scorebar_ocr"
    manifest_path = build_dataset(dataset_dir, labeled_by="alex")

    assert manifest_path.exists()
    assert len((dataset_dir / "digits.jsonl").read_text().splitlines()) == 107
    manifest = manifest_path.read_text()
    assert "human_verified" in manifest
    assert "lat_van_hp_0090" in manifest
    assert "lat_van_hp_0685" in manifest  # excluded: no scorebar crop at final timestamp


def test_scorebar_eval_reports_generalization_gap(tmp_path):
    dataset_dir = tmp_path / "scorebar_ocr"
    build_dataset(dataset_dir)

    result = evaluate(dataset_dir)

    operational = result["metrics"]["operational_gallery"]
    loo = result["metrics"]["leave_one_out"]
    temporal = result["metrics"]["leave_one_out_temporal"]
    assert operational["score_exact_accuracy"] == 1.0
    assert 0.0 < loo["score_exact_accuracy"] < operational["score_exact_accuracy"]
    assert loo["digit_accuracy"] >= 0.75
    assert temporal["score_exact_accuracy"] >= loo["score_exact_accuracy"]
    assert loo["failures"]


def test_cdl_scorebar_engine_reads_known_crop_with_eval_capped_confidence(tmp_path):
    dataset_dir = tmp_path / "scorebar_ocr"
    build_dataset(dataset_dir)
    result = evaluate(dataset_dir)
    (dataset_dir / "eval_results.json").write_text(json.dumps(result))

    engine = CdlScorebarOcrEngine(dataset_dir=dataset_dir, temporal=False)
    image = cv2.imread(str(dataset_dir / "crops" / "lat_van_hp_0146.png"), cv2.IMREAD_COLOR)
    read = engine.read(image)

    assert read.text == "34 38"
    assert read.confidence == result["metrics"]["leave_one_out"]["score_exact_accuracy"]
    assert read.is_placeholder is False
    assert read.boxes[0]["model_version"] == "0.1.0-knn"


def test_cdl_scorebar_engine_abstains_when_scorebar_absent(tmp_path):
    dataset_dir = tmp_path / "scorebar_ocr"
    build_dataset(dataset_dir)
    engine = CdlScorebarOcrEngine(dataset_dir=dataset_dir)
    image = cv2.imread("data/crops/lat_van_hp/sb_0678.png", cv2.IMREAD_COLOR)

    read = engine.read(image)

    assert read.is_placeholder is True
    assert read.text == ""
    assert read.confidence == 0.0


def test_build_ocr_engine_supports_cdl_backend():
    engine = build_ocr_engine("cdl")

    assert isinstance(engine, OcrEngine)
    assert getattr(engine, "model_version") == "0.1.0-knn"
