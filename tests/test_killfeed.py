from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend.app.events import EventKind, SourceKind
from backend.app.services import killfeed as kf
from scripts.build_killfeed_dataset import build_killfeed_dataset
from scripts.eval_killfeed import evaluate

ROOT = Path(__file__).resolve().parents[1]
REGION = kf.KILLFEED_REGION


def _killfeed_frame(rows: list[tuple[str, str, bool]], w: int = 1920, h: int = 1080) -> np.ndarray:
    """Synthetic broadcast frame with team-coloured kill rows in the killfeed region.

    Each row is (attacker, victim, red_attacker). Rows sit at stable vertical slots,
    like the real bottom-anchored feed. A non-uniform background stands in for the
    game world behind the semi-transparent feed.
    """
    frame = np.full((h, w, 3), (40, 46, 52), np.uint8)
    frame[:, : w // 2] = (60, 55, 48)
    rx0 = int(REGION["x"] * w)
    ry0 = int(REGION["y"] * h)
    rh = int(REGION["h"] * h)
    red, blue = (60, 60, 235), (235, 150, 40)            # BGR team colours
    for i, (atk, vic, red_attacker) in enumerate(rows):
        y = ry0 + int(0.42 * rh) + i * 40
        c_atk, c_vic = (red, blue) if red_attacker else (blue, red)
        cv2.putText(frame, atk, (rx0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_atk, 2)
        cv2.rectangle(frame, (rx0 + 150, y - 14), (rx0 + 180, y + 2), (240, 240, 240), -1)
        cv2.putText(frame, vic, (rx0 + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_vic, 2)
    return frame


def _white_band_frame(w: int = 1920, h: int = 1080) -> np.ndarray:
    """A desaturated full-width band in the killfeed region — i.e. a stats/HUD table,
    not a kill row. The team-colour filter must reject it."""
    frame = np.full((h, w, 3), (40, 46, 52), np.uint8)
    rx0, ry0 = int(REGION["x"] * w), int(REGION["y"] * h)
    rw, rh = int(REGION["w"] * w), int(REGION["h"] * h)
    y = ry0 + int(0.5 * rh)
    cv2.rectangle(frame, (rx0 + 6, y - 12), (rx0 + rw - 6, y + 8), (235, 235, 235), -1)
    return frame


# --- detector --------------------------------------------------------------------

def test_detector_localises_team_colored_rows():
    frame = _killfeed_frame([("ENVOY", "PRED", True), ("CELLIUM", "ABEZY", False)])
    rows = kf.KillfeedDetector().detect_rows(frame)
    assert len(rows) == 2
    assert all(0.0 <= r.x <= 1.0 and 0.0 <= r.y <= 1.0 for r in rows)
    assert all(0.0 < r.confidence <= 0.85 for r in rows)


def test_detector_quiet_on_empty_feed():
    assert kf.KillfeedDetector().detect_rows(_killfeed_frame([])) == []


def test_team_color_filter_rejects_desaturated_band():
    # This is the failure that the wrong (top-right stats) region produced: a
    # full-width white band with no team colour must not be read as a kill.
    frame = _white_band_frame()
    assert kf.KillfeedDetector(require_team_color=True).detect_rows(frame) == []
    assert len(kf.KillfeedDetector(require_team_color=False).detect_rows(frame)) >= 1


def test_detect_returns_serialisable_dicts():
    frame = _killfeed_frame([("ENVOY", "PRED", True)])
    out = kf.KillfeedDetector().detect(frame, {"regions": {"killfeed": REGION}})
    assert out and set(out[0]) == {"box", "ink_fill", "confidence", "left_hue", "right_hue"}


def test_killfeed_region_matches_hud_profile():
    # KILLFEED_REGION must stay in sync with the verified HUD profile region.
    profile = json.loads((ROOT / "data/configs/hud_profiles/CDL_2026_1080p.json").read_text())
    assert profile["regions"]["killfeed"] == REGION


# --- temporal tracking -----------------------------------------------------------

def test_tracker_dedupes_persistent_rows_into_one_onset_each():
    det = kf.KillfeedDetector()
    tracker = kf.KillfeedTracker()
    two = _killfeed_frame([("ENVOY", "PRED", True), ("CELLIUM", "ABEZY", False)])
    onsets = sum(len(tracker.update(k * 0.5, det.detect_rows(two))) for k in range(8))
    assert onsets == 2                                   # two slots, persisting != re-firing
    three = _killfeed_frame([("STORM", "RAID", True), ("ENVOY", "PRED", True), ("CELLIUM", "ABEZY", False)])
    assert len(tracker.update(4.0, det.detect_rows(three))) == 1   # one genuinely new slot


def test_row_returning_after_ttl_gap_is_a_new_onset():
    det = kf.KillfeedDetector()
    tracker = kf.KillfeedTracker(ttl_seconds=3.0)
    one = _killfeed_frame([("ENVOY", "PRED", True)])
    assert len(tracker.update(0.0, det.detect_rows(one))) == 1
    for k in range(10):                                  # 5s of empty feed (> ttl)
        tracker.update(0.5 + k * 0.5, det.detect_rows(_killfeed_frame([])))
    assert len(tracker.update(6.0, det.detect_rows(one))) == 1


def test_detect_kill_onsets_streams_in_time_order():
    frames = [(t, _killfeed_frame([("ENVOY", "PRED", True)] if t < 3 else [])) for t in [0, 1, 2, 5, 6]]
    onsets = list(kf.detect_kill_onsets(frames))
    assert [round(o.video_timestamp_seconds, 1) for o in onsets] == [0.0]


# --- event emission --------------------------------------------------------------

def test_onset_emits_valid_candidate_kill_event_with_unread_identity():
    det = kf.KillfeedDetector()
    onsets = kf.KillfeedTracker().update(212.4, det.detect_rows(_killfeed_frame([("ENVOY", "PRED", True)])))
    event = kf.onset_to_kill_event(
        onsets[0], crop_path="data/killfeed_dataset/rows/kf_0001_212.png",
        source_url="https://example/vod", broadcast_id=1, match_id=1, map_id=1,
    )
    assert event.event_type == "kill" and event.kind == EventKind.FACT
    assert event.evidence.has_visual() and event.derived_from == []    # fact invariants
    assert event.payload.attacker is None and event.payload.victim is None and event.payload.weapon is None
    assert 0.0 <= event.confidence <= 1.0
    assert event.provenance.source == SourceKind.HEURISTIC
    assert event.provenance.model_name == "killfeed_classical" and event.provenance.model_version
    assert "identity_unread" in event.tags


# --- dataset scaffold ------------------------------------------------------------

def test_build_dataset_writes_empty_label_scaffold(tmp_path):
    frames = [(float(t), _killfeed_frame([("ENVOY", "PRED", True), ("CELLIUM", "ABEZY", False)]))
              for t in range(6)]
    out = tmp_path / "killfeed_dataset"
    manifest = build_killfeed_dataset(frames, out, source_url="https://example/vod", video="data/videos/x.mp4")

    assert manifest["label_status"] == "unlabeled"
    assert manifest["onsets"] == 2                       # deduped across the 6 frames
    anns = [json.loads(line) for line in (out / "annotations.jsonl").read_text().splitlines()]
    assert len(anns) == 2
    for ann in anns:
        assert (out / ann["row_image"]).exists()
        assert ann["label_source"] == "unlabeled" and ann["labeled_by"] is None
        assert ann["source_url"] == "https://example/vod"
        assert all(v is None for v in ann["label"].values())   # nothing invented


# --- evaluation ------------------------------------------------------------------

def _write_labeled_dataset(tmp_path, rows) -> Path:
    d = tmp_path / "kf"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"detector": "killfeed_classical@0.1.0", "source_url": "u", "label_status": "labeled"}))
    (d / "annotations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


def _ann(i, valid, detector="killfeed_classical@0.1.0", attacker=None, victim=None, weapon=None):
    return {
        "id": f"kf_{i}", "video_timestamp_seconds": float(i), "row_image": f"rows/kf_{i}.png",
        "region_image": None, "box": [0, 0, 0, 0], "detector": detector, "detector_confidence": 0.6,
        "color_hint": {}, "label_source": "manual_label", "labeled_by": "alex", "source_url": "u",
        "label": {"valid_kill": valid, "attacker": attacker, "attacker_team": None, "victim": victim,
                  "victim_team": None, "weapon": weapon, "headshot": None, "is_trade": None},
    }


def test_eval_reports_unlabeled_when_no_labels(tmp_path):
    rows = [_ann(0, None), _ann(1, None)]
    for r in rows:
        r["label_source"] = "unlabeled"
        r["labeled_by"] = None
    result = evaluate(_write_labeled_dataset(tmp_path, rows))
    det = result["metrics"]["detection"]
    assert det["labeled"] == 0 and "no accuracy claim" in det["status"]
    assert "precision" not in det


def test_eval_computes_detection_precision_recall(tmp_path):
    rows = [
        _ann(0, True, attacker="ENVOY", victim="PRED", weapon="SMG"),
        _ann(1, True, attacker="ABEZY", victim="LUNARZ"),
        _ann(2, False), _ann(3, False),                 # two false positives
        _ann(4, True, detector="manual_added", attacker="CELLIUM", victim="MAMBA"),  # one missed kill
    ]
    result = evaluate(_write_labeled_dataset(tmp_path, rows))
    det = result["metrics"]["detection"]
    assert (det["true_positives"], det["false_positives"], det["false_negatives"]) == (2, 2, 1)
    assert det["precision"] == 0.5 and det["recall"] == round(2 / 3, 4)
    content = result["metrics"]["content_readiness"]
    assert content["valid_kills"] == 3 and content["with_weapon"] == 1
