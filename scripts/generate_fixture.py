from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    readings = json.loads((ROOT / "data/fixtures/sample_scores.json").read_text())["readings"]
    output = ROOT / "data/videos/sample_hardpoint.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 fixture")
    for index, reading in enumerate(readings):
        frame = np.full((height, width, 3), (28, 35, 42), dtype=np.uint8)
        cv2.putText(frame, "COD SPECTRUM SYNTHETIC HARDPOINT", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (170, 180, 190), 3)
        cv2.rectangle(frame, (768, 22), (1152, 76), (8, 10, 12), -1)
        cv2.rectangle(frame, (768, 22), (960, 76), (165, 72, 40), 3)
        cv2.rectangle(frame, (960, 22), (1152, 76), (48, 110, 184), 3)
        cv2.putText(frame, f"{reading['score_a']:03d}", (810, 63), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)
        cv2.putText(frame, f"{reading['score_b']:03d}", (1005, 63), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3)
        cv2.putText(frame, f"T+{index:02d}s", (895, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (220, 220, 220), 2)
        writer.write(frame)
    writer.release()
    print(f"wrote {output} ({len(readings)} seconds)")


if __name__ == "__main__":
    main()

