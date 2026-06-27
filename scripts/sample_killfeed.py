"""Phase 4 sample output: killfeed detection -> candidate KillEvents.

Deterministic and offline. It builds synthetic killfeed frames, runs the *real*
classical detector + positional tracker, and emits the candidate ``KillEvent``
stream — each a fact with evidence + confidence and ``identity_unread`` (the detector
localises kills; it does not read names yet). Then it summarises the real committed
dataset and prints the honest evaluation status.

Run: .venv/bin/python scripts/sample_killfeed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.events import EventKind, to_jsonl  # noqa: E402
from backend.app.services.killfeed import (  # noqa: E402
    KILLFEED_REGION,
    KillfeedDetector,
    KillfeedTracker,
    onset_to_kill_event,
)
from scripts.eval_killfeed import evaluate, load_annotations  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "fixtures" / "killfeed_sample"
OUT_JSONL = SAMPLE_DIR / "sample_killfeed_events.jsonl"
DATASET_DIR = ROOT / "data" / "killfeed_dataset"


def _killfeed_frame(rows, w=1920, h=1080):
    frame = np.full((h, w, 3), (40, 46, 52), np.uint8)
    frame[:, : w // 2] = (60, 55, 48)
    rx0, ry0, rh = int(KILLFEED_REGION["x"] * w), int(KILLFEED_REGION["y"] * h), int(KILLFEED_REGION["h"] * h)
    red, blue = (60, 60, 235), (235, 150, 40)
    for i, (atk, vic, red_attacker) in enumerate(rows):
        y = ry0 + int(0.42 * rh) + i * 40
        c_atk, c_vic = (red, blue) if red_attacker else (blue, red)
        cv2.putText(frame, atk, (rx0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_atk, 2)
        cv2.rectangle(frame, (rx0 + 150, y - 14), (rx0 + 180, y + 2), (240, 240, 240), -1)
        cv2.putText(frame, vic, (rx0 + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_vic, 2)
    return frame


def synthetic_event_stream() -> list:
    """Detector + tracker over a scripted synthetic sequence -> candidate KillEvents."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    detector, tracker = KillfeedDetector(), KillfeedTracker()
    # t<3: one kill on screen; at t=4 a second distinct kill appears and persists.
    timeline = [(t, [("ENVOY", "PRED", True)]) for t in (1.0, 1.5, 2.0)]
    timeline += [(t, [("ENVOY", "PRED", True), ("CELLIUM", "ABEZY", False)]) for t in (4.0, 4.5, 5.0)]

    region_crop = None
    events = []
    for t, rows in timeline:
        frame = _killfeed_frame(rows)
        h, w = frame.shape[:2]
        rx0, ry0 = int(KILLFEED_REGION["x"] * w), int(KILLFEED_REGION["y"] * h)
        region_crop = frame[ry0:ry0 + int(KILLFEED_REGION["h"] * h), rx0:rx0 + int(KILLFEED_REGION["w"] * w)]
        rch, rcw = region_crop.shape[:2]
        for onset in tracker.update(t, detector.detect_rows(frame)):
            box = onset.row.box_list()
            x0, y0 = int(box[0] * rcw), int(box[1] * rch)
            x1, y1 = int((box[0] + box[2]) * rcw), int((box[1] + box[3]) * rch)
            crop_name = f"rows/kf_{len(events):02d}_{int(t)}.png"
            (SAMPLE_DIR / "rows").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(SAMPLE_DIR / crop_name), region_crop[y0:y1, x0:x1])
            event = onset_to_kill_event(
                onset, crop_path=str((SAMPLE_DIR / crop_name).relative_to(ROOT)),
                source_url="synthetic://killfeed-demo", broadcast_id=1, match_id=1, map_id=1,
            )
            # Synthetic input -> flag the demo so it is never mistaken for real telemetry.
            events.append(event.model_copy(update={
                "is_placeholder": True, "tags": event.tags + ["synthetic"],
            }))
    return events


def main() -> int:
    events = synthetic_event_stream()
    OUT_JSONL.write_text(to_jsonl(events) + "\n")

    print("=== synthetic candidate KillEvent stream (detector + tracker, identity unread) ===")
    for e in events:
        p = e.payload
        ident = f"{p.attacker or '?'} -> {p.victim or '?'} ({p.weapon or '?'})"
        print(f"~ {e.video_timestamp_seconds:5.1f}s  {e.kind.value:<6} kill   conf={e.confidence:.2f}  "
              f"identity={ident:<14} evidence={Path(e.evidence.crop_path).name}  [{','.join(e.tags)}]")
    facts = [e for e in events if e.kind == EventKind.FACT]
    visual = [e for e in facts if e.evidence.has_visual()]
    unread = [e for e in facts if e.payload.attacker is None]
    print(f"\ncandidate kills: {len(events)}  | facts with evidence: {len(visual)}/{len(facts)}  "
          f"| identity unread: {len(unread)}/{len(facts)}")
    print(f"wrote {OUT_JSONL.relative_to(ROOT)}")

    print("\n=== real committed dataset (data/killfeed_dataset) ===")
    if (DATASET_DIR / "annotations.jsonl").exists():
        manifest, rows = load_annotations(DATASET_DIR)
        print(f"source: {manifest.get('source_url')}")
        print(f"candidates: {len(rows)}  | label_status: {manifest.get('label_status')}")
        det = evaluate(DATASET_DIR)["metrics"]["detection"]
        print(f"detection eval: {det['status']}")
        print("-> label annotations.jsonl, then `make killfeed-eval` for precision/recall.")
    else:
        print("not built yet — run `make killfeed-dataset` with the local VOD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
