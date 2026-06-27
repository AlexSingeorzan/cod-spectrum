"""Build the Phase 3 scorebar OCR dataset from verified scorebar crops.

The default source is ``backend.app.services.real_match.VERIFIED``: manually
verified LAT/VAN scorebar labels with evidence crops in ``data/crops/lat_van_hp``.
No labels are inferred by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.real_match import SOURCE_URL, VERIFIED  # noqa: E402
from backend.app.services.scorebar_ocr import (  # noqa: E402
    default_dataset_dir,
    region,
    scorebar_present,
    slice_digits,
    white_mask,
)


def build_dataset(out_dir: Path, labeled_by: str = "alex") -> Path:
    crop_dir = ROOT / "data" / "crops" / "lat_van_hp"
    dataset_crop_dir = out_dir / "crops"
    glyph_dir = out_dir / "glyphs"
    dataset_crop_dir.mkdir(parents=True, exist_ok=True)
    glyph_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    excluded: list[dict] = []
    digit_rows: list[dict] = []

    for timestamp, score_a, score_b in VERIFIED:
        image_path = crop_dir / f"sb_{timestamp:04d}.png"
        sample_id = f"lat_van_hp_{timestamp:04d}"
        if not image_path.exists():
            excluded.append({
                "sample_id": sample_id,
                "timestamp_seconds": timestamp,
                "reason": "evidence_crop_missing",
                "score_a": score_a,
                "score_b": score_b,
            })
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or not scorebar_present(image):
            excluded.append({
                "sample_id": sample_id,
                "timestamp_seconds": timestamp,
                "reason": "scorebar_not_present",
                "score_a": score_a,
                "score_b": score_b,
            })
            continue

        dataset_image_path = dataset_crop_dir / f"{sample_id}.png"
        cv2.imwrite(str(dataset_image_path), image)

        sample = {
            "sample_id": sample_id,
            "timestamp_seconds": float(timestamp),
            "image_path": dataset_image_path.relative_to(out_dir).as_posix(),
            "score_a": score_a,
            "score_b": score_b,
            "source_url": SOURCE_URL,
            "label_source": "human_verified",
            "labeled_by": labeled_by,
        }
        samples.append(sample)

        for side, score in (("left", score_a), ("right", score_b)):
            digits = str(score)
            glyphs = slice_digits(white_mask(region(image, side)), len(digits))
            if len(glyphs) != len(digits):
                excluded.append({
                    "sample_id": sample_id,
                    "timestamp_seconds": timestamp,
                    "reason": f"{side}_glyph_count_mismatch",
                    "expected_digits": len(digits),
                    "actual_glyphs": len(glyphs),
                })
                continue
            for index, (digit, glyph) in enumerate(zip(digits, glyphs)):
                glyph_path = glyph_dir / f"{sample_id}_{side}_{index}_{digit}.png"
                cv2.imwrite(str(glyph_path), (glyph * 255).astype("uint8"))
                digit_rows.append({
                    "sample_id": sample_id,
                    "timestamp_seconds": float(timestamp),
                    "side": side,
                    "digit_index": index,
                    "label": int(digit),
                    "path": glyph_path.relative_to(out_dir).as_posix(),
                    "source_url": SOURCE_URL,
                    "label_source": "human_verified",
                    "labeled_by": labeled_by,
                })

    manifest = {
        "version": 1,
        "model_family": "cdl_scorebar_knn",
        "description": "Human-verified LAT/VAN scorebar crops for Phase 3 OCR evaluation.",
        "source_url": SOURCE_URL,
        "label_source": "human_verified",
        "labeled_by": labeled_by,
        "samples": samples,
        "excluded": excluded,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out_dir / "digits.jsonl").open("w") as handle:
        for row in digit_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out_dir / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=default_dataset_dir())
    parser.add_argument("--labeled-by", default="alex")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_dataset(args.out, labeled_by=args.labeled_by)
    payload = json.loads(manifest.read_text())
    digit_count = sum(1 for line in (args.out / "digits.jsonl").read_text().splitlines() if line.strip())
    print(f"wrote {len(payload['samples'])} scorebar samples and {digit_count} digit glyphs to {manifest}")
    if payload["excluded"]:
        print(f"excluded {len(payload['excluded'])} labels; see manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
