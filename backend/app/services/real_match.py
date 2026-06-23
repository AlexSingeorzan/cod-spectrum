"""Real LAT vs Vancouver Surge analysis — Map 1 (Hacienda Hardpoint).

Scores here were read and VISUALLY VERIFIED, frame by frame, from the official
Call of Duty League VOD (https://youtu.be/6pyPLwdrzq4 — "Major IV Qualifiers,
Week 3 Day 2"), sampled at ~28-second intervals across the Hardpoint map. The
final 250-156 is confirmed on the post-game stat card. This is NOT automated OCR
output: tesseract could not read the stylised CDL scorebar font reliably, so the
honest path was human verification with the cropped scorebar kept as evidence
(data/crops/lat_van_hp/).

Scope: Map 1 only. Maps 2 (S&D) and 3 (Overload) use different scoreboards and
are out of scope for this 0-250 Hardpoint model. xMWP is the uncalibrated
HeuristicV0. There is no kill/spawn telemetry here — this is scoreboard truth.
"""
from __future__ import annotations

import base64
from functools import lru_cache

from ..config import get_settings
from .analytics import HeuristicV0

SOURCE_URL = "https://youtu.be/6pyPLwdrzq4"
SOURCE_TITLE = "LA Thieves vs Vancouver Surge — Major IV Qualifiers, W3D2"
MAP_NAME = "Hacienda"
MODE = "Hardpoint"
TARGET = 250
TEAM_A, TEAM_B = "LAT", "VAN"

# (vod_time_seconds, LAT_score, VAN_score) — verified from the broadcast scorebar.
VERIFIED: list[tuple[int, int, int]] = [
    (90, 16, 11), (118, 34, 12), (146, 34, 38), (174, 39, 47), (202, 52, 58),
    (230, 61, 63), (258, 85, 63), (286, 99, 64), (314, 104, 71), (342, 120, 71),
    (370, 120, 95), (398, 129, 102), (426, 138, 109), (454, 150, 117), (482, 164, 121),
    (510, 190, 121), (538, 201, 128), (566, 215, 132), (594, 223, 132), (622, 232, 141),
    (650, 239, 156), (685, 250, 156),
]


def _thumb(t: int) -> str:
    path = get_settings().data_dir / "crops" / "lat_van_hp" / f"sb_{t:04d}.png"
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return ""


@lru_cache(maxsize=1)
def analysis() -> dict:
    model = HeuristicV0(TARGET, 0.15)
    points = [
        {"t": t, "a": a, "b": b, "prob_a": round(model.predict(a, b), 4)}
        for t, a, b in VERIFIED
    ]
    intervals = []
    for (t0, a0, b0), (t1, a1, b1) in zip(VERIFIED, VERIFIED[1:]):
        da, db = a1 - a0, b1 - b0
        intervals.append({"t": t1, "t0": t0, "da": da, "db": db, "net": da - db})

    leader, lead_changes = None, []
    for t, a, b in VERIFIED:
        cur = TEAM_A if a > b else TEAM_B if b > a else None
        if cur and cur != leader:
            lead_changes.append({"t": t, "to": cur, "a": a, "b": b})
            leader = cur
    van_window = [lc["t"] for lc in lead_changes if lc["to"] == TEAM_B][:1]
    van_lead = [van_window[0], next((lc["t"] for lc in lead_changes if lc["to"] == TEAM_A and lc["t"] > van_window[0]), VERIFIED[-1][0])] if van_window else None

    swings = sorted(intervals, key=lambda i: abs(i["net"]), reverse=True)
    key_moments = [
        {"t": 146, "kind": "VAN", "title": "Vancouver break", "detail": "VAN scores 26 unanswered (12 → 38) to seize the lead.", "thumb": _thumb(146)},
        {"t": 258, "kind": "LAT", "title": "Thieves retake", "detail": "LAT answers with 24 unanswered (61 → 85) to reclaim the hill.", "thumb": _thumb(258)},
        {"t": 510, "kind": "LAT", "title": "Map-breaking hold", "detail": "LAT runs 164 → 190 while VAN is stuck on 121 — win probability jumps past 0.79.", "thumb": _thumb(510)},
        {"t": 650, "kind": "LAT", "title": "Closed out", "detail": "VAN stalls on 156; LAT closes 239 → 250.", "thumb": _thumb(650)},
    ]
    return {
        "meta": {
            "team_a": TEAM_A, "team_b": TEAM_B, "map_name": MAP_NAME, "mode": MODE,
            "target": TARGET, "final_a": VERIFIED[-1][1], "final_b": VERIFIED[-1][2],
            "source_url": SOURCE_URL, "source_title": SOURCE_TITLE,
            "samples": len(VERIFIED), "duration": VERIFIED[-1][0],
        },
        "points": points,
        "intervals": intervals,
        "lead_changes": lead_changes,
        "van_lead": van_lead,
        "top_swings": swings[:6],
        "key_moments": key_moments,
    }
