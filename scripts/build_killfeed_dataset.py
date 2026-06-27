"""Killfeed annotation-dataset builder — Phase 4, deliverable 2 (scaffold).

Turns the classical ``KillfeedDetector`` into a *pre-labeller*: it runs over a VOD,
finds candidate kill onsets, saves the row crop (and a region crop for context) for
each, and writes an ``annotations.jsonl`` with EMPTY label slots plus full source
metadata. A human then fills attacker / victim / weapon and marks each row
``valid_kill`` true/false — and adds any kills the detector MISSED as rows with
``detector="manual_added"`` so recall is measurable.

Nothing here invents a label. The crops and timestamps are real detections; every
identity field starts ``null`` until a person reads it (mirrors the minimap dataset
scaffold and the master rule "every manual label stores its source").

Usage (from repo root):
  .venv/bin/python scripts/build_killfeed_dataset.py --vod data/videos/lat_van.mp4

Then label ``data/killfeed_dataset/annotations.jsonl`` and evaluate with
``scripts/eval_killfeed.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from backend.app.services.killfeed import (  # noqa: E402
    KILLFEED_REGION,
    MODEL_NAME,
    MODEL_VERSION,
    KillfeedDetector,
    KillfeedTracker,
    crop_killfeed,
)

DEFAULT_OUT = ROOT / "data" / "killfeed_dataset"

# Empty label slot a human fills in. valid_kill gates detection precision/recall;
# the identity fields feed the future content reader (DeathEvent/WeaponEvent/Trade).
EMPTY_LABEL = {
    "valid_kill": None,        # True = real kill, False = false positive
    "attacker": None, "attacker_team": None,
    "victim": None, "victim_team": None,
    "weapon": None, "headshot": None, "is_trade": None,
}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_killfeed_dataset(
    frames: Iterable[tuple[float, "cv2.Mat"]],
    out_dir: Path,
    *,
    source_url: str | None = None,
    video: Path | str | None = None,
    window: tuple[float, float] | None = None,
    sample_fps: float | None = None,
    region: dict | None = None,
    detector: KillfeedDetector | None = None,
    tracker: KillfeedTracker | None = None,
    save_region: bool = True,
) -> dict:
    """Run detection over ``frames`` ((timestamp, BGR frame) pairs) and write the
    annotation scaffold. Returns the manifest dict. Pure function of its inputs so
    it is testable without a real VOD."""
    region = region or KILLFEED_REGION
    detector = detector or KillfeedDetector()
    tracker = tracker or KillfeedTracker()

    rows_dir = out_dir / "rows"
    regions_dir = out_dir / "regions"
    rows_dir.mkdir(parents=True, exist_ok=True)
    if save_region:
        regions_dir.mkdir(parents=True, exist_ok=True)

    annotations: list[dict] = []
    frames_seen = 0
    for timestamp, frame in frames:
        frames_seen += 1
        region_crop = crop_killfeed(frame, region)
        rch, rcw = region_crop.shape[:2]
        for onset in tracker.update(timestamp, detector.detect_rows(frame, region)):
            row = onset.row
            idx = len(annotations)
            sample_id = f"kf_{idx:04d}_{int(round(timestamp))}"
            x0, y0 = int(row.x * rcw), int(row.y * rch)
            x1, y1 = int((row.x + row.w) * rcw), int((row.y + row.h) * rch)
            row_rel = f"rows/{sample_id}.png"
            cv2.imwrite(str(out_dir / row_rel), region_crop[y0:y1, x0:x1])
            region_rel = None
            if save_region:
                region_rel = f"regions/{sample_id}.png"
                cv2.imwrite(str(out_dir / region_rel), region_crop)
            annotations.append({
                "id": sample_id,
                "video_timestamp_seconds": round(timestamp, 3),
                "row_image": row_rel,
                "region_image": region_rel,
                "box": row.box_list(),
                "detector": f"{MODEL_NAME}@{MODEL_VERSION}",
                "detector_confidence": round(row.confidence, 3),
                "color_hint": {"left_hue": row.left_hue, "right_hue": row.right_hue},
                "label": dict(EMPTY_LABEL),
                "label_source": "unlabeled",
                "labeled_by": None,
                "source_url": source_url,
            })

    manifest = {
        "version": 1,
        "kind": "killfeed_detection_dataset",
        "detector": f"{MODEL_NAME}@{MODEL_VERSION}",
        "source_url": source_url,
        "video": _rel(Path(video)) if video else None,
        "window_seconds": list(window) if window else None,
        "sample_fps": sample_fps,
        "frames_sampled": frames_seen,
        "onsets": len(annotations),
        "label_status": "unlabeled",
        "labeled_by": None,
        "honesty_note": (
            "Detections are unverified classical-CV candidates. attacker/victim/weapon "
            "are NOT read by the detector — every label starts null. Fill labels by hand "
            "(valid_kill true/false + identities), and add any MISSED kills as rows with "
            "detector='manual_added' so recall is measurable, before making any accuracy claim."
        ),
        "label_schema": {
            "valid_kill": "true=real kill, false=false positive (required for detection eval)",
            "attacker/victim": "gamertag as shown (feeds DeathEvent / content reader)",
            "attacker_team/victim_team": "team tag e.g. LAT/VAN",
            "weapon": "weapon name or icon class (feeds WeaponEvent)",
            "headshot": "bool if a headshot marker is shown",
            "is_trade": "bool if this kill traded a recent teammate death (feeds TradeEvent)",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out_dir / "annotations.jsonl").open("w") as handle:
        for row in annotations:
            handle.write(json.dumps(row) + "\n")
    return manifest


def iter_vod_frames(vod: Path, start: float, end: float, sample_fps: float) -> Iterator[tuple[float, "cv2.Mat"]]:
    cap = cv2.VideoCapture(str(vod))
    try:
        step = 1.0 / sample_fps
        t = start
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                yield t, frame
            t += step
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vod", type=Path, default=ROOT / "data" / "videos" / "lat_van.mp4")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", type=float, default=80.0, help="skip intro (seconds)")
    ap.add_argument("--end", type=float, default=690.0, help="end of Hardpoint map (seconds)")
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--source-url", default=None, help="VOD URL, stored on each label")
    args = ap.parse_args(argv)
    if not args.vod.exists():
        raise SystemExit(f"VOD not found: {args.vod} (large VODs are gitignored — download first)")

    manifest = build_killfeed_dataset(
        iter_vod_frames(args.vod, args.start, args.end, args.sample_fps),
        args.out,
        source_url=args.source_url,
        video=args.vod,
        window=(args.start, args.end),
        sample_fps=args.sample_fps,
    )
    print(f"sampled {manifest['frames_sampled']} frames -> {manifest['onsets']} candidate "
          f"kill onsets in {_rel(args.out)}/")
    print("next: label annotations.jsonl (valid_kill + identities; add missed kills as "
          "detector='manual_added'), then: .venv/bin/python scripts/eval_killfeed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
