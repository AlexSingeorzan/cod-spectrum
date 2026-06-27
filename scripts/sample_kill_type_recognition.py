"""Sample output for Phase 5 coarse kill-type recognition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.events import to_jsonl  # noqa: E402
from backend.app.services.kill_type_recognition import (  # noqa: E402
    KILL_TYPE_CATEGORIES,
    MODEL_NAMES,
    MODEL_VERSION,
    KillTypeRecognizer,
    kill_type_event_from_prediction,
)
from scripts.eval_kill_type_recognition import evaluate  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "fixtures" / "kill_type_recognition_sample"
SAMPLE_TYPES = ["gun", "grenade", "melee", "killstreak", "environment"]


def _draw_icon(kill_type: str, variant: int) -> np.ndarray:
    image = np.zeros((34, 68, 3), np.uint8)
    image[:] = (20, 22, 24)
    white = (235, 235, 235)
    x_shift = variant
    if kill_type == "gun":
        cv2.line(image, (9 + x_shift, 17), (54 + x_shift, 17), white, 3)
        cv2.line(image, (13 + x_shift, 19), (6 + x_shift, 25), white, 3)
        cv2.line(image, (33 + x_shift, 19), (43 + x_shift, 27), white, 2)
    elif kill_type == "grenade":
        cv2.circle(image, (34 + x_shift, 17), 9, white, 2)
        cv2.rectangle(image, (28 + x_shift, 6), (40 + x_shift, 10), white, -1)
        cv2.line(image, (41 + x_shift, 6), (49 + x_shift, 2), white, 2)
    elif kill_type == "melee":
        cv2.line(image, (18 + x_shift, 27), (48 + x_shift, 7), white, 4)
        cv2.line(image, (20 + x_shift, 7), (48 + x_shift, 27), white, 2)
    elif kill_type == "killstreak":
        pts = np.array([[34 + x_shift, 5], [43 + x_shift, 24], [34 + x_shift, 20], [25 + x_shift, 24]], np.int32)
        cv2.fillPoly(image, [pts], white)
        cv2.circle(image, (34 + x_shift, 17), 12, white, 1)
    elif kill_type == "environment":
        cv2.rectangle(image, (15 + x_shift, 10), (52 + x_shift, 24), white, 2)
        cv2.line(image, (17 + x_shift, 24), (50 + x_shift, 10), white, 2)
    else:
        cv2.putText(image, kill_type[:2], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 2)
    return image


def build_sample_dataset() -> Path:
    icons_dir = SAMPLE_DIR / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for kill_type in SAMPLE_TYPES:
        for variant in range(2):
            sample_id = f"kill_type_{kill_type}_{variant}"
            icon_name = f"{sample_id}.png"
            cv2.imwrite(str(icons_dir / icon_name), _draw_icon(kill_type, variant))
            rows.append({
                "id": sample_id,
                "video_timestamp_seconds": float(90 + len(rows)),
                "icon_image": f"icons/{icon_name}",
                "source_crop_path": f"icons/{icon_name}",
                "source_row_image": "synthetic://kill-type-recognition-demo",
                "source_segments": "synthetic://kill-type-recognition-demo",
                "source_url": "synthetic://kill-type-recognition-demo",
                "segment_box": [0.35, 0.2, 0.2, 0.6],
                "segment_confidence": 1.0,
                "segmenter": "synthetic_fixture",
                "detector": "synthetic_fixture",
                "detector_confidence": 1.0,
                "human_review_status": "reviewed",
                "label": {
                    "valid_kill_type": True,
                    "kill_type": kill_type,
                    "exact_weapon": None,
                    "unclear": False,
                },
                "label_source": "manual_label",
                "labeled_by": "synthetic_fixture",
            })

    manifest = {
        "version": 1,
        "kind": "kill_type_icon_dataset",
        "dataset_id": "synthetic_kill_type_icons_v0",
        "icon_count": len(rows),
        "label_status": "labeled",
        "categories": list(KILL_TYPE_CATEGORIES),
        "content": f"{MODEL_NAMES['template']}@{MODEL_VERSION}",
        "honesty_note": "Synthetic fixture for event-contract verification only.",
    }
    (SAMPLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (SAMPLE_DIR / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    (SAMPLE_DIR / "README.md").write_text(
        "# Kill-Type Recognition Sample\n\n"
        "Synthetic labelled fixture for the Phase 5 kill-type recognizer. It proves "
        "the event contract only; it is not real broadcast accuracy.\n"
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
    recognizer = KillTypeRecognizer(dataset, approach="template", confidence_threshold=0.0)
    prediction = recognizer.read_row(rows[3])  # a grenade row
    event = kill_type_event_from_prediction(prediction, player="HyDra", team="LAT")
    events = [event] if event is not None else []
    (dataset / "events.jsonl").write_text(to_jsonl(events) + ("\n" if events else ""))

    print("=== synthetic kill-type recognition sample ===")
    print(f"dataset: {dataset.relative_to(ROOT)}")
    print(f"prediction: kill_type={prediction.kill_type} conf={prediction.confidence:.2f} "
          f"model={prediction.model_name}@{prediction.model_version}")
    if event is not None:
        print(f"event: {event.event_type} player={event.payload.player} "
              f"kill_type={event.payload.kill_type} weapon={event.payload.weapon} "
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
