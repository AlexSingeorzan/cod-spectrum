"""Sample output for Phase 4 Stage B killfeed segmentation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.killfeed_segmentation import (  # noqa: E402
    CORE_FIELDS,
    KillfeedSegmenter,
    write_segment_crops,
)

SAMPLE_DIR = ROOT / "data" / "fixtures" / "killfeed_segmentation_sample"


def _draw_sample_row() -> np.ndarray:
    image = np.zeros((38, 360, 3), np.uint8)
    image[:] = (28, 31, 34)
    red = (60, 60, 235)
    white = (235, 235, 235)
    cv2.putText(image, "NERO", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, white, 2)
    cv2.putText(image, "SMG", (135, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 2)
    cv2.circle(image, (222, 18), 7, white, 2)
    cv2.putText(image, "ABEZY", (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, red, 2)
    return image


def main() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    row_path = SAMPLE_DIR / "row.png"
    image = _draw_sample_row()
    cv2.imwrite(str(row_path), image)

    segmenter = KillfeedSegmenter()
    segmentation = segmenter.segment(image, sample_id="segmentation_sample_0001")
    segmentation = write_segment_crops(image, segmentation, SAMPLE_DIR / "segments")
    payload = segmentation.as_dict()
    for segment in payload["segments"].values():
        if segment.get("crop_path"):
            segment["crop_path"] = str(Path(segment["crop_path"]).resolve().relative_to(ROOT))
    payload["row_image"] = str(row_path.relative_to(ROOT))
    payload["notes"] = [
        "Synthetic sample verifies the Stage B field-box contract only.",
        "It is not real broadcast segmentation accuracy.",
    ]
    (SAMPLE_DIR / "segments.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("=== killfeed segmentation sample ===")
    print(f"row: {row_path.relative_to(ROOT)}")
    for field in CORE_FIELDS + ("headshot",):
        segment = payload["segments"][field]
        print(
            f"- {field:<8} box={segment['box']} conf={segment['confidence']} "
            f"crop={segment['crop_path']}"
        )
    print(f"wrote {Path(SAMPLE_DIR / 'segments.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
