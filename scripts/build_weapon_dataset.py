"""Build the Phase 5 weapon-icon dataset from Stage B killfeed segments.

This copies only weapon crops referenced by ``segments.jsonl``. Unreferenced files
in the segment directory are ignored, so stale generated crops cannot silently
enter the classifier dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.weapon_recognition import default_dataset_dir  # noqa: E402

DEFAULT_KILLFEED_DATASET = ROOT / "data" / "killfeed_dataset"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _rooted(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_weapon_dataset(
    killfeed_dataset: Path,
    out_dir: Path,
    *,
    clean_icons: bool = True,
) -> Path:
    segments_path = killfeed_dataset / "segments.jsonl"
    if not segments_path.exists():
        raise FileNotFoundError(f"missing Stage B segments file: {segments_path}")

    segment_rows = _load_jsonl(segments_path)
    annotation_lookup = {}
    annotations_path = killfeed_dataset / "annotations.jsonl"
    if annotations_path.exists():
        annotation_lookup = {row["id"]: row for row in _load_jsonl(annotations_path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    icons_dir = out_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    if clean_icons:
        for old in icons_dir.glob("*.png"):
            old.unlink()

    rows: list[dict] = []
    skipped = []
    for row in segment_rows:
        weapon_segment = row.get("segments", {}).get("weapon") or {}
        crop_path = weapon_segment.get("crop_path")
        if not crop_path:
            skipped.append({"sample_id": row.get("sample_id"), "reason": "no_weapon_segment"})
            continue
        source_crop = _rooted(crop_path)
        if not source_crop.exists():
            skipped.append({"sample_id": row.get("sample_id"), "reason": "weapon_crop_missing"})
            continue

        sample_id = str(row["sample_id"])
        icon_name = f"{sample_id}_weapon.png"
        destination = icons_dir / icon_name
        shutil.copyfile(source_crop, destination)
        rows.append({
            "id": sample_id,
            "video_timestamp_seconds": row.get("video_timestamp_seconds"),
            "icon_image": f"icons/{icon_name}",
            "source_crop_path": _rel(source_crop),
            "source_row_image": row.get("row_image"),
            "source_segments": _rel(segments_path),
            "source_url": annotation_lookup.get(sample_id, {}).get("source_url"),
            "segment_box": weapon_segment.get("box"),
            "segment_confidence": weapon_segment.get("confidence"),
            "segmenter": f"{row.get('model_name')}@{row.get('model_version')}",
            "detector": row.get("detector"),
            "detector_confidence": row.get("detector_confidence"),
            "human_review_status": "unreviewed",
            "label": {
                "valid_weapon": None,
                "weapon": None,
                "weapon_family": None,
                "unclear": None,
            },
            "label_source": "unlabeled",
            "labeled_by": None,
        })

    manifest = {
        "version": 1,
        "kind": "weapon_icon_dataset",
        "dataset_id": "lat_van_hp_weapon_icons_v0",
        "source_dataset": _rel(killfeed_dataset),
        "source_segments": _rel(segments_path),
        "icon_count": len(rows),
        "skipped_count": len(skipped),
        "label_status": "unlabeled",
        "label_fields": {
            "valid_weapon": "true/false after human review",
            "weapon": "weapon class, or unknown for visible-but-unidentifiable",
            "weapon_family": "optional AR/SMG/SNIPER/PISTOL/etc.",
            "unclear": "true when crop is too ambiguous for training/eval",
        },
        "honesty_note": (
            "Weapon icons are cropped evidence only. No classifier accuracy is "
            "claimed until these rows are human-labelled."
        ),
        "evaluation_metrics": {
            "real_labelled_accuracy": None,
            "reason": "0 labelled real weapon icons",
        },
        "skipped": skipped[:50],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    print(f"wrote {len(rows)} weapon icon crops to {_rel(out_dir / 'annotations.jsonl')}")
    if skipped:
        print(f"skipped {len(skipped)} rows without usable weapon crops")
    return out_dir / "annotations.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--killfeed-dataset", type=Path, default=DEFAULT_KILLFEED_DATASET)
    ap.add_argument("--out", type=Path, default=ROOT / default_dataset_dir())
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_weapon_dataset(args.killfeed_dataset, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
