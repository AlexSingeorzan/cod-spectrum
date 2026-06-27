"""Build Phase 4 Stage B killfeed field segments.

Reads a killfeed annotation dataset, segments each row crop into attacker, weapon,
victim, and optional indicator boxes, writes segment crops, and records the result
in ``segments.jsonl``. This does not label or classify anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.killfeed_content import load_annotation_rows  # noqa: E402
from backend.app.services.killfeed_segmentation import (  # noqa: E402
    CORE_FIELDS,
    KillfeedSegmenter,
    write_segment_crops,
)


DEFAULT_DATASET = ROOT / "data" / "killfeed_dataset"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_segments(dataset_dir: Path, *, write_crops: bool = True) -> Path:
    _manifest, rows = load_annotation_rows(dataset_dir)
    segmenter = KillfeedSegmenter()
    segments_dir = dataset_dir / "segments"
    output_rows = []

    for row in rows:
        row_path = Path(row["row_image"])
        if not row_path.is_absolute():
            row_path = dataset_dir / row_path
        image = cv2.imread(str(row_path), cv2.IMREAD_COLOR)
        if image is None:
            output_rows.append({
                "sample_id": row["id"],
                "row_image": row["row_image"],
                "video_timestamp_seconds": row["video_timestamp_seconds"],
                "failure_reason": "row_image_unreadable",
                "segments": {},
            })
            continue

        segmentation = segmenter.segment(image, sample_id=row["id"])
        if write_crops:
            segmentation = write_segment_crops(image, segmentation, segments_dir)
        payload = segmentation.as_dict()
        payload.update({
            "row_image": row["row_image"],
            "video_timestamp_seconds": row["video_timestamp_seconds"],
            "detector": row.get("detector"),
            "detector_confidence": row.get("detector_confidence"),
            "human_review_status": "unreviewed",
        })
        for segment in payload["segments"].values():
            if segment.get("crop_path"):
                segment["crop_path"] = _rel(Path(segment["crop_path"]))
        output_rows.append(payload)

    output_path = dataset_dir / "segments.jsonl"
    output_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in output_rows) + "\n")

    complete = sum(
        1
        for row in output_rows
        if all(row.get("segments", {}).get(field, {}).get("box") for field in CORE_FIELDS)
    )
    print(
        f"segmented {len(output_rows)} killfeed rows -> {_rel(output_path)} "
        f"({complete} with attacker+weapon+victim boxes)"
    )
    if write_crops:
        print(f"segment crops: {_rel(segments_dir)}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--no-crops", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_segments(args.dataset, write_crops=not args.no_crops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
