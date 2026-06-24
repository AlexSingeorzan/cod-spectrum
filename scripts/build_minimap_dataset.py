"""Model-assisted YOLO dataset builder for minimap player detection.

Pre-seeds bounding boxes with the existing ClassicalMinimapDetector so a human
only CORRECTS boxes (fast) instead of drawing every one from scratch, then writes
a YOLO-format dataset. This is the bridge from the classical detector to a trained
model — the README's "YOLO minimap next step".

Classes: 0 = observed_player, 1 = enemy_player (radar-visible).

Usage (from repo root):
  .venv/bin/python scripts/build_minimap_dataset.py --vod data/videos/lat_van.mp4 --n 300

Output:
  data/minimap_dataset/images/*.png   minimap crops
  data/minimap_dataset/labels/*.txt   YOLO pre-labels  (class cx cy w h, normalised)
  data/minimap_dataset/data.yaml      (tracked in git)

Then correct the pre-labels in any YOLO tool (Label Studio / Roboflow / labelImg),
split images+labels into train/ and val/, and train:
  yolo detect train data=data/minimap_dataset/data.yaml model=yolov8n.pt imgsz=256 epochs=120

The trained model drops in behind the same `MinimapDetector` interface — only
`detect()` changes; the heatmap, spawn derivation and persistence stay identical.
Honesty rule carried through: the broadcast minimap only shows the observed team +
radar-visible enemies, so never label hidden opponents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from backend.app.services.minimap import ClassicalMinimapDetector, crop_minimap  # noqa: E402

CLASS_ID = {"observed": 0, "enemy": 1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vod", type=Path, required=True, help="local VOD path")
    ap.add_argument("--n", type=int, default=300, help="number of frames to sample")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "minimap_dataset")
    ap.add_argument("--start", type=float, default=80.0, help="skip intro (seconds)")
    ap.add_argument("--end", type=float, default=690.0, help="end of Hardpoint map (seconds)")
    args = ap.parse_args(argv)
    if not args.vod.exists():
        raise SystemExit(f"VOD not found: {args.vod} (large VODs are gitignored — download first)")

    img_dir, lbl_dir = args.out / "images", args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.vod))
    detector = ClassicalMinimapDetector()
    span = max(1, args.n - 1)
    times = [args.start + (args.end - args.start) * i / span for i in range(args.n)]
    images = boxes = 0
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        stem = f"mm_{i:04d}_{int(t)}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), crop_minimap(frame))
        lines = [
            f"{CLASS_ID[d['team']]} {d['x']:.6f} {d['y']:.6f} {max(d['w'], 0.03):.6f} {max(d['h'], 0.03):.6f}"
            for d in detector.detect(frame)
        ]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        images += 1
        boxes += len(lines)
    cap.release()
    print(f"wrote {images} images and {boxes} pre-seeded boxes -> {args.out}")
    print("next: correct labels, split images/labels into train/ + val/, then")
    print("  yolo detect train data=data/minimap_dataset/data.yaml model=yolov8n.pt imgsz=256 epochs=120")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
