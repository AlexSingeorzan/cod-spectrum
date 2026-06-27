"""Run the scoreboard K/D kill counter over a VOD — Phase 4 (the kill spine).

Does the slow part once: OCR-reads both team panels frame by frame and CACHES the
per-player K/D readings to ``readings.jsonl``. The counting logic + evaluation then
run offline over that cache (``eval_panel_counter.py``), so they are fast and
reproducible without re-OCR or even the VOD.

Also emits the ``KillEvent`` / ``DeathEvent`` stream (with evidence panel crops at
each kill frame) and a killfeed reconciliation summary.

Usage:
  .venv/bin/python scripts/run_panel_counter.py --vod data/videos/lat_van.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from backend.app.events import Evidence, to_jsonl  # noqa: E402
from backend.app.services.hud import load_hud_profile  # noqa: E402
from backend.app.services.killfeed import KillfeedDetector, KillfeedTracker  # noqa: E402
from backend.app.services.panel_counter import (  # noqa: E402
    KD_PANEL_A,
    KD_PANEL_B,
    MODEL_NAME,
    MODEL_VERSION,
    PanelKillCounter,
    read_panels,
    reconcile_with_killfeed,
)
from backend.app.services.real_match import MAP_NAME, SOURCE_URL, TEAM_A, TEAM_B  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "panel_counter"
# Slot -> verified gamertag, in panel row order (LAT top-left, VAN top-right).
NAMES = {"a1": "Scrap", "a2": "HyDra", "a3": "aBeZy", "a4": "Nium",
         "b1": "Craze", "b2": "Mamba", "b3": "Lunarz", "b4": "Nero"}


def _crop_panels(frame, regions):
    h, w = frame.shape[:2]
    ra, rb = regions.get("kd_panel_a", KD_PANEL_A), regions.get("kd_panel_b", KD_PANEL_B)
    def crop(r):
        return frame[int(r["y"] * h):int((r["y"] + r["h"]) * h), int(r["x"] * w):int((r["x"] + r["w"]) * w)]
    return crop(ra), crop(rb)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vod", type=Path, default=ROOT / "data" / "videos" / "lat_van.mp4")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--hud-profile", default="CDL_2026_1080p")
    ap.add_argument("--start", type=float, default=80.0)
    ap.add_argument("--end", type=float, default=690.0)
    ap.add_argument("--sample-fps", type=float, default=1.0)
    args = ap.parse_args(argv)
    if not args.vod.exists():
        raise SystemExit(f"VOD not found: {args.vod} (large VODs are gitignored — download first)")

    profile = load_hud_profile(args.hud_profile)
    regions = profile.regions
    crop_dir = args.out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    counter = PanelKillCounter(team_a=TEAM_A, team_b=TEAM_B, names=NAMES)
    kdet, ktrack = KillfeedDetector(), KillfeedTracker()

    cap = cv2.VideoCapture(str(args.vod))
    readings_log: list[dict] = []
    events = []
    kill_times: list[float] = []
    kf_times: list[float] = []
    step = 1.0 / args.sample_fps
    t = args.start
    while t <= args.end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            t += step
            continue
        readings = read_panels(frame, profile)
        readings_log.append({"t": round(t, 3), "a": readings["a"], "b": readings["b"]})
        crop_path = None
        new = counter.update(t, readings, Evidence(video_timestamp_seconds=t, source_url=SOURCE_URL,
                                                   crop_path="pending"))
        if new:
            crop_path = f"crops/panels_{int(round(t * 1000)):08d}.png"
            left, right = _crop_panels(frame, regions)
            stacked = cv2.hconcat([cv2.resize(left, (220, 240)), cv2.resize(right, (220, 240))])
            cv2.imwrite(str(args.out / crop_path), stacked)
            for e in new:
                e.evidence.crop_path = crop_path
                events.append(e)
                if e.event_type == "kill":
                    kill_times.append(t)
        for o in ktrack.update(t, kdet.detect_rows(frame, regions.get("killfeed"))):
            kf_times.append(o.video_timestamp_seconds)
        t += step
    cap.release()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "readings.jsonl").write_text("\n".join(json.dumps(r) for r in readings_log) + "\n")
    (args.out / "events.jsonl").write_text(to_jsonl(events) + "\n" if events else "")
    rec = reconcile_with_killfeed(kill_times, kf_times)
    final = readings_log[-1] if readings_log else {"a": [], "b": []}
    manifest = {
        "detector": f"{MODEL_NAME}@{MODEL_VERSION}", "ocr_backend": "tesseract",
        "source_url": SOURCE_URL, "map": MAP_NAME, "team_a": TEAM_A, "team_b": TEAM_B,
        "names": NAMES, "window_seconds": [args.start, args.end], "sample_fps": args.sample_fps,
        "frames": len(readings_log), "kill_events": len(kill_times), "death_events": len(events) - len(kill_times),
        "counter_total_kills": counter.total_kills(),
        "reconciliation": rec.as_dict(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"frames {len(readings_log)} | kill events {len(kill_times)} | counter total {counter.total_kills()}")
    print(f"final readout: a={final['a']} b={final['b']}")
    print(f"killfeed reconciliation: {rec.as_dict()}")
    print(f"cached readings + events -> {args.out.relative_to(ROOT)}/  (run scripts/eval_panel_counter.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
