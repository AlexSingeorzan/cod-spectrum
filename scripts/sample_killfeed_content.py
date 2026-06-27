"""Sample output for the Phase 4 killfeed content reader.

The real LAT/VAN killfeed dataset is still unlabelled, so this script uses a small
synthetic labelled fixture to show the event contract: killfeed content reads can
emit KillEvent, DeathEvent, WeaponEvent and TradeEvent. It also prints the real
dataset's current label/readiness status.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.events import EventKind, to_jsonl  # noqa: E402
from backend.app.services.killfeed_content import (  # noqa: E402
    MODEL_NAME,
    MODEL_VERSION,
    KillfeedContentReader,
    events_from_content_reads,
)
from scripts.eval_killfeed_content import evaluate  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "fixtures" / "killfeed_content_sample"
REAL_DATASET = ROOT / "data" / "killfeed_dataset"


SAMPLE_ROWS = [
    {
        "id": "kf_content_0001",
        "t": 88.0,
        "attacker": "HyDra",
        "attacker_team": "LAT",
        "victim": "Lunarz",
        "victim_team": "VAN",
        "weapon": "SMG",
        "headshot": False,
        "is_trade": False,
    },
    {
        "id": "kf_content_0002",
        "t": 91.0,
        "attacker": "Mamba",
        "attacker_team": "VAN",
        "victim": "HyDra",
        "victim_team": "LAT",
        "weapon": "AR",
        "headshot": True,
        "is_trade": True,
    },
    {
        "id": "kf_content_0003",
        "t": 96.0,
        "attacker": "aBeZy",
        "attacker_team": "LAT",
        "victim": "Nero",
        "victim_team": "VAN",
        "weapon": "SMG",
        "headshot": False,
        "is_trade": False,
    },
]


def _draw_row(row: dict) -> np.ndarray:
    image = np.zeros((38, 360, 3), np.uint8)
    image[:] = (30, 34, 38)
    red, blue, white = (60, 60, 235), (235, 150, 40), (235, 235, 235)
    atk_color = red if row["attacker_team"] == "LAT" else blue
    vic_color = red if row["victim_team"] == "LAT" else blue
    cv2.putText(image, row["attacker"], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, atk_color, 2)
    cv2.putText(image, row["weapon"], (142, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
    if row["headshot"]:
        cv2.circle(image, (210, 18), 7, white, 2)
    cv2.putText(image, row["victim"], (225, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, vic_color, 2)
    return image


def build_sample_dataset() -> Path:
    rows_dir = SAMPLE_DIR / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    for row in SAMPLE_ROWS:
        row_path = rows_dir / f"{row['id']}.png"
        cv2.imwrite(str(row_path), _draw_row(row))
        annotations.append({
            "id": row["id"],
            "video_timestamp_seconds": row["t"],
            "row_image": f"rows/{row['id']}.png",
            "region_image": None,
            "box": [0.0, 0.0, 1.0, 1.0],
            "detector": "synthetic_fixture",
            "detector_confidence": 1.0,
            "color_hint": {},
            "label": {
                "valid_kill": True,
                "attacker": row["attacker"],
                "attacker_team": row["attacker_team"],
                "victim": row["victim"],
                "victim_team": row["victim_team"],
                "weapon": row["weapon"],
                "headshot": row["headshot"],
                "is_trade": row["is_trade"],
            },
            "label_source": "manual_label",
            "labeled_by": "synthetic_fixture",
            "source_url": "synthetic://killfeed-content-demo",
        })

    manifest = {
        "version": 1,
        "kind": "killfeed_content_sample",
        "detector": "synthetic_fixture",
        "content_reader": f"{MODEL_NAME}@{MODEL_VERSION}",
        "source_url": "synthetic://killfeed-content-demo",
        "label_status": "labeled",
        "labeled_by": "synthetic_fixture",
        "honesty_note": "Synthetic fixture for event-contract verification only; not real broadcast accuracy.",
    }
    (SAMPLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (SAMPLE_DIR / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n"
    )
    return SAMPLE_DIR


def main() -> int:
    dataset = build_sample_dataset()
    reader = KillfeedContentReader(dataset_dir=dataset, confidence_threshold=0.0)
    annotations = [
        json.loads(line)
        for line in (dataset / "annotations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    reads = [reader.read_annotation(row) for row in annotations]
    events = events_from_content_reads([read for read in reads if read is not None])
    (dataset / "events.jsonl").write_text(to_jsonl(events) + "\n")

    eval_result = evaluate(dataset)
    (dataset / "eval_results.json").write_text(json.dumps(eval_result, indent=2) + "\n")

    print("=== synthetic killfeed content event stream ===")
    for event in events:
        p = event.payload
        if event.event_type == "kill":
            detail = f"{p.attacker} -> {p.victim} ({p.weapon})"
        elif event.event_type == "death":
            detail = f"{p.player} killed_by={p.killer} weapon={p.weapon}"
        elif event.event_type == "weapon":
            detail = f"{p.player} used {p.weapon}"
        else:
            detail = f"{p.trading_player} traded {p.dead_player} in {p.trade_window_seconds}s"
        print(f"~ {event.video_timestamp_seconds:5.1f}s  {event.kind.value:<4} {event.event_type:<7} "
              f"conf={event.confidence:.2f}  {detail}")
    facts = [event for event in events if event.kind == EventKind.FACT]
    print(f"\nsynthetic facts: {len(facts)}  | wrote {Path(dataset / 'events.jsonl').relative_to(ROOT)}")
    print(f"synthetic eval: gallery exact="
          f"{eval_result['metrics']['reader']['operational_gallery']['exact_matches']}/"
          f"{eval_result['metrics']['reader']['operational_gallery']['samples']}")

    print("\n=== real committed killfeed dataset ===")
    if (REAL_DATASET / "annotations.jsonl").exists():
        real = evaluate(REAL_DATASET)
        print(f"candidates: {real['dataset']['candidates']}")
        print(f"content-labelled rows: {real['dataset']['content_labeled_rows']}")
        print(f"reader status: {real['metrics']['reader']['status']}")
        print("next: label attacker/victim/weapon fields in data/killfeed_dataset/annotations.jsonl")
    else:
        print("not built yet — run `make killfeed-dataset` with the local VOD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
