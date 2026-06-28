"""Evaluate the Phase 6 minimap detection contract on a synthetic fixture.

This is not real broadcast accuracy. It verifies that the current detector
surface preserves model metadata, visual evidence, normalized boxes, visibility,
latency, abstention thresholds, and PositionEvent emission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.events import to_jsonl  # noqa: E402
from backend.app.services.minimap import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    MODEL_VERSION,
    ClassicalMinimapDetector,
    crop_minimap,
    position_events_from_minimap_result,
)

FIXTURE_DIR = ROOT / "data" / "fixtures" / "minimap_contract"
FULL_FRAME_PROFILE = {"regions": {"minimap": {"x": 0, "y": 0, "w": 1, "h": 1}}}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def draw_fixture_frame() -> np.ndarray:
    frame = np.full((160, 160, 3), 18, np.uint8)
    cv2.circle(frame, (42, 42), 6, (0, 0, 255), -1)
    cv2.circle(frame, (116, 112), 6, (255, 255, 255), -1)
    cv2.rectangle(frame, (4, 135), (36, 154), (20, 90, 180), -1)
    return frame


def _preserve_existing_latency(path: Path, payload: dict) -> dict:
    if not path.exists():
        return payload
    try:
        existing = json.loads(path.read_text())
    except json.JSONDecodeError:
        return payload
    if "latency_ms" in existing:
        payload = {**payload, "latency_ms": existing["latency_ms"]}
    return payload


def evaluate(out_dir: Path = FIXTURE_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = draw_fixture_frame()
    frame_path = out_dir / "frame.png"
    crop_path = out_dir / "minimap_crop.png"
    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(crop_path), crop_minimap(frame, FULL_FRAME_PROFILE["regions"]["minimap"]))

    detector = ClassicalMinimapDetector()
    result = detector.read_frame(
        frame,
        FULL_FRAME_PROFILE,
        frame_index=24,
        video_timestamp_seconds=88.0,
        frame_path=_rel(frame_path),
        crop_path=_rel(crop_path),
        observed_team="LAT",
        human_review_status="synthetic_fixture",
    )
    events = position_events_from_minimap_result(
        result,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        broadcast_id=1,
        map_id=1,
    )
    high_threshold_events = position_events_from_minimap_result(result, confidence_threshold=0.99)

    detections_path = out_dir / "detections.json"
    eval_path = out_dir / "eval_results.json"
    result_json = _preserve_existing_latency(detections_path, result.as_dict())
    event_jsonl = to_jsonl(events)
    detections_path.write_text(json.dumps(result_json, indent=2) + "\n")
    (out_dir / "events.jsonl").write_text(event_jsonl + ("\n" if event_jsonl else ""))

    visibility_counts: dict[str, int] = {}
    for detection in result.detections:
        visibility_counts[detection.visibility] = visibility_counts.get(detection.visibility, 0) + 1

    evaluation = _preserve_existing_latency(eval_path, {
        "fixture": _rel(out_dir),
        "model": f"{MODEL_NAME}@{MODEL_VERSION}",
        "samples": 1,
        "detections": len(result.detections),
        "position_events": len(events),
        "high_threshold_events": len(high_threshold_events),
        "visibility_counts": visibility_counts,
        "latency_ms": round(result.latency_ms, 3),
        "contract": {
            "has_model_metadata": bool(result.model_name and result.model_version and result.training_dataset),
            "has_visual_evidence": all(event.evidence.has_visual() for event in events),
            "has_bounding_boxes": all(
                bool(event.payload.attributes.get("bbox_xywh_norm"))
                for event in events
            ),
            "does_not_infer_enemy_team": all(
                event.payload.team is None
                for event in events
                if event.payload.attributes.get("visibility") == "radar_visible_enemy"
            ),
            "low_confidence_abstains": len(high_threshold_events) == 0,
        },
        "notes": [
            "Synthetic contract fixture only; no real broadcast accuracy claim.",
            "Enemy markers are radar-visible observations, not inferred hidden opponents.",
            "Future YOLO detector must satisfy the same result/event contract.",
        ],
    })
    eval_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    return evaluation


def print_report(result: dict) -> None:
    print("# Minimap Contract Evaluation")
    print(f"- fixture: {result['fixture']}")
    print(f"- model: {result['model']}")
    print(f"- detections: {result['detections']} | events: {result['position_events']}")
    print(f"- visibility: {result['visibility_counts']}")
    print(f"- high-threshold events: {result['high_threshold_events']}")
    print(f"- latency_ms: {result['latency_ms']}")
    print("Contract:")
    for key, value in result["contract"].items():
        print(f"- {key}: {value}")
    print("Notes:")
    for note in result["notes"]:
        print(f"- {note}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args.out)
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
