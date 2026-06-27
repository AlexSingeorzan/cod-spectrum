"""Sample output for Phase 5 weapon recognition.

The real weapon dataset is unlabelled, so this script builds a tiny synthetic
labelled fixture to prove the classifier/evaluation/event contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.events import to_jsonl  # noqa: E402
from backend.app.services.weapon_recognition import (  # noqa: E402
    MODEL_NAMES,
    MODEL_VERSION,
    WeaponRecognizer,
    weapon_event_from_prediction,
)
from scripts.eval_weapon_recognition import evaluate  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "fixtures" / "weapon_recognition_sample"


def _draw_icon(weapon: str, variant: int) -> np.ndarray:
    image = np.zeros((32, 64, 3), np.uint8)
    image[:] = (20, 22, 24)
    white = (235, 235, 235)
    y = 15 + (variant % 2)
    x_shift = variant
    if weapon == "AR":
        cv2.line(image, (11 + x_shift, y), (49 + x_shift, y), white, 3)
        cv2.line(image, (9 + x_shift, y + 2), (3 + x_shift, y + 7), white, 3)
        cv2.line(image, (31 + x_shift, y + 2), (39 + x_shift, y + 10), white, 2)
    elif weapon == "SMG":
        cv2.rectangle(image, (17 + x_shift, y - 5), (42 + x_shift, y + 2), white, -1)
        cv2.line(image, (25 + x_shift, y + 3), (22 + x_shift, y + 11), white, 2)
        cv2.line(image, (42 + x_shift, y - 2), (50 + x_shift, y - 2), white, 2)
    elif weapon == "SNIPER":
        cv2.line(image, (6 + x_shift, y), (56 + x_shift, y), white, 2)
        cv2.circle(image, (30 + x_shift, y - 5), 4, white, 2)
        cv2.line(image, (13 + x_shift, y + 2), (9 + x_shift, y + 9), white, 2)
    else:
        cv2.putText(image, weapon[:2], (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 2)
    return image


def build_sample_dataset() -> Path:
    icons_dir = SAMPLE_DIR / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    weapons = ["AR", "SMG", "SNIPER"]
    for weapon in weapons:
        for variant in range(2):
            sample_id = f"weapon_{weapon.lower()}_{variant}"
            icon_name = f"{sample_id}.png"
            cv2.imwrite(str(icons_dir / icon_name), _draw_icon(weapon, variant))
            rows.append({
                "id": sample_id,
                "video_timestamp_seconds": float(90 + len(rows)),
                "icon_image": f"icons/{icon_name}",
                "source_crop_path": f"icons/{icon_name}",
                "source_row_image": "synthetic://weapon-recognition-demo",
                "source_segments": "synthetic://weapon-recognition-demo",
                "source_url": "synthetic://weapon-recognition-demo",
                "segment_box": [0.35, 0.2, 0.2, 0.6],
                "segment_confidence": 1.0,
                "segmenter": "synthetic_fixture",
                "detector": "synthetic_fixture",
                "detector_confidence": 1.0,
                "human_review_status": "reviewed",
                "label": {
                    "valid_weapon": True,
                    "weapon": weapon,
                    "weapon_family": weapon,
                    "unclear": False,
                },
                "label_source": "manual_label",
                "labeled_by": "synthetic_fixture",
            })

    manifest = {
        "version": 1,
        "kind": "weapon_icon_dataset",
        "dataset_id": "synthetic_weapon_icons_v0",
        "icon_count": len(rows),
        "label_status": "labeled",
        "content": f"{MODEL_NAMES['template']}@{MODEL_VERSION}",
        "honesty_note": "Synthetic fixture for event-contract verification only.",
    }
    (SAMPLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (SAMPLE_DIR / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    (SAMPLE_DIR / "README.md").write_text(
        "# Weapon Recognition Sample\n\n"
        "Synthetic labelled fixture for the Phase 5 weapon recognizer. It proves the "
        "event contract only; it is not real broadcast accuracy.\n"
    )
    return SAMPLE_DIR


def main() -> int:
    dataset = build_sample_dataset()
    eval_result = evaluate(dataset)
    (dataset / "eval_results.json").write_text(json.dumps(eval_result, indent=2) + "\n")

    rows = [
        json.loads(line)
        for line in (dataset / "annotations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    recognizer = WeaponRecognizer(dataset, approach="template", confidence_threshold=0.0)
    prediction = recognizer.read_row(rows[0])
    event = weapon_event_from_prediction(prediction, player="HyDra", team="LAT")
    events = [event] if event is not None else []
    (dataset / "events.jsonl").write_text(to_jsonl(events) + ("\n" if events else ""))

    print("=== synthetic weapon recognition sample ===")
    print(f"dataset: {dataset.relative_to(ROOT)}")
    print(f"prediction: weapon={prediction.weapon} conf={prediction.confidence:.2f} "
          f"model={prediction.model_name}@{prediction.model_version}")
    if event is not None:
        print(f"event: {event.event_type} player={event.payload.player} weapon={event.payload.weapon} "
              f"evidence={event.evidence.crop_path}")
    print()
    for approach, metrics in eval_result["models"].items():
        gallery = metrics["operational_gallery"]
        loo = metrics["leave_one_out"]
        if gallery is None:
            print(f"{approach}: {metrics['status']}")
        else:
            print(f"{approach}: gallery={gallery['top1_accuracy']} loo={loo['top1_accuracy']} "
                  f"loo_abstain={loo['abstentions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
