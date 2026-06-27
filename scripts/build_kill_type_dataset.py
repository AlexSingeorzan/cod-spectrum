"""Build the Phase 5 coarse kill-type dataset from Stage B killfeed segments.

The source crop is still the killfeed icon segment, but the label target is a
coarse ``kill_type``:

gun, grenade, melee, fall_damage, suicide, environment, objective, killstreak,
unknown.

Exact weapon names are optional future metadata and are not required for the
analytics contract.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.kill_type_recognition import (  # noqa: E402
    KILL_TYPE_CATEGORIES,
    default_dataset_dir,
)

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


def build_kill_type_dataset(
    killfeed_dataset: Path,
    out_dir: Path,
    *,
    clean_icons: bool = True,
) -> Path:
    segments_path = killfeed_dataset / "segments.jsonl"
    if not segments_path.exists():
        raise FileNotFoundError(f"missing Stage B segments file: {segments_path}")

    segment_rows = _load_jsonl(segments_path)
    annotations_path = killfeed_dataset / "annotations.jsonl"
    annotation_lookup = {}
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
        icon_segment = row.get("segments", {}).get("weapon") or {}
        crop_path = icon_segment.get("crop_path")
        if not crop_path:
            skipped.append({"sample_id": row.get("sample_id"), "reason": "no_icon_segment"})
            continue
        source_crop = _rooted(crop_path)
        if not source_crop.exists():
            skipped.append({"sample_id": row.get("sample_id"), "reason": "icon_crop_missing"})
            continue

        sample_id = str(row["sample_id"])
        icon_name = f"{sample_id}_kill_type_icon.png"
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
            "segment_box": icon_segment.get("box"),
            "segment_confidence": icon_segment.get("confidence"),
            "segmenter": f"{row.get('model_name')}@{row.get('model_version')}",
            "detector": row.get("detector"),
            "detector_confidence": row.get("detector_confidence"),
            "human_review_status": "unreviewed",
            "label": {
                "valid_kill_type": None,
                "kill_type": None,
                "exact_weapon": None,
                "unclear": None,
            },
            "label_source": "unlabeled",
            "labeled_by": None,
        })

    manifest = {
        "version": 1,
        "kind": "kill_type_icon_dataset",
        "dataset_id": "lat_van_hp_kill_type_icons_v0",
        "source_dataset": _rel(killfeed_dataset),
        "source_segments": _rel(segments_path),
        "icon_count": len(rows),
        "skipped_count": len(skipped),
        "label_status": "unlabeled",
        "categories": list(KILL_TYPE_CATEGORIES),
        "label_fields": {
            "valid_kill_type": "true/false after human review",
            "kill_type": "one of categories; unknown means reviewed visible cause outside/indistinguishable from supported classes",
            "exact_weapon": "optional future metadata such as MCW/Jackal PDW/C9/AMES",
            "unclear": "true when crop is too ambiguous for training/eval",
        },
        "honesty_note": (
            "Icon crops are evidence only. No kill-type classifier accuracy is "
            "claimed until these rows are human-labelled."
        ),
        "evaluation_metrics": {
            "real_labelled_accuracy": None,
            "reason": "0 labelled real kill-type icons",
        },
        "skipped": skipped[:50],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    (out_dir / "README.md").write_text(
        "# Kill-Type Icon Dataset\n\n"
        "Phase 5 coarse kill-type dataset generated from Stage B killfeed icon segments.\n\n"
        f"- Source segments: `{_rel(segments_path)}`\n"
        "- Icon crops: `icons/`\n"
        "- Annotation file: `annotations.jsonl`\n"
        f"- Current icon crops: `{len(rows)}`\n"
        "- Current labelled kill-type icons: `0`\n"
        "- Current real kill-type accuracy: no claim\n\n"
        "Build and evaluate:\n\n"
        "```bash\n"
        "make kill-type-dataset\n"
        "make kill-type-eval\n"
        "```\n\n"
        "The dataset intentionally starts with empty labels:\n\n"
        "```json\n"
        "{\n"
        "  \"valid_kill_type\": null,\n"
        "  \"kill_type\": null,\n"
        "  \"exact_weapon\": null,\n"
        "  \"unclear\": null\n"
        "}\n"
        "```\n\n"
        "Only human-reviewed rows with `valid_kill_type=true`, a non-null "
        "`kill_type`, and `label_source != \"unlabeled\"` are used for "
        "training/evaluation. Use `unknown` only when a reviewer can see a valid "
        "kill cause but cannot assign one of the named classes; use `unclear=true` "
        "for ambiguous crops. `exact_weapon` is optional future metadata; "
        "downstream analytics consume `kill_type`.\n"
    )
    print(f"wrote {len(rows)} kill-type icon crops to {_rel(out_dir / 'annotations.jsonl')}")
    if skipped:
        print(f"skipped {len(skipped)} rows without usable icon crops")
    return out_dir / "annotations.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--killfeed-dataset", type=Path, default=DEFAULT_KILLFEED_DATASET)
    ap.add_argument("--out", type=Path, default=ROOT / default_dataset_dir())
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_kill_type_dataset(args.killfeed_dataset, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
