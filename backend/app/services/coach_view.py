"""Match-story coach view — charts that answer coach questions, not stat dumps.

Three signature views, all from the verified LAT-VAN Map 1 data:
  1. Momentum timeline   — xMWP with breaks/retakes/spawn-flips/hill ticks on ONE
                           line ("why the map turned").
  2. Turning points      — per-hill win-probability delta, ranked ("why we won").
  3. Rotation timing      — per-hill control speed proxy ("what to practise"),
                           honest that true rotate-time + league/opponent baselines
                           need positional tracking and a multi-map corpus.

Win probability is the uncalibrated HeuristicV0; a useful, well-known property
shows through: an early Hardpoint lead barely moves win%, so the model correctly
says VAN's P1-P2 lead mattered far less than LAT's P3 retake and P7 break.
"""
from __future__ import annotations

from . import hardpoint_breakdown as hb
from . import real_match
from .analytics import HeuristicV0

TEAM_A, TEAM_B = "LAT", "VAN"
_MODEL = HeuristicV0(250, 0.15)


def _hill_narrative(h) -> str:
    loser_pts = h.db if h.winner == TEAM_A else h.da
    if abs(h.margin) >= 28 and loser_pts <= 8:
        return f"{h.winner} lockout — held {TEAM_B if h.winner == TEAM_A else TEAM_A} to {loser_pts}"
    if abs(h.margin) >= 18:
        return f"{h.winner} controlled the hill {h.da}-{h.db}"
    if abs(h.margin) <= 4:
        return f"contested hill ({h.da}-{h.db})"
    return f"{h.winner} edged it {h.da}-{h.db}"


def analysis() -> dict:
    hills = hb.HillSegmenter(names=hb.HILL_NAMES).segment(hb.SCORE_AT_HILL_BOUNDARY)
    hb_payload = hb.analysis()
    rm = real_match.analysis()

    momentum = [
        {"t": t, "a": a, "b": b, "p": round(_MODEL.predict(a, b), 4)}
        for t, a, b in real_match.VERIFIED
    ]
    hill_ticks = [{"t": h.t_start, "hill": h.index, "name": h.name} for h in hills]
    hill_ticks.append({"t": hb.SCORE_AT_HILL_BOUNDARY[-1][0], "hill": None, "name": "final"})

    events = []
    for lc in rm["lead_changes"][1:]:
        kind = "retake" if lc["to"] == TEAM_A else "break"
        events.append({"t": lc["t"], "type": kind, "team": lc["to"], "label": f"{lc['to']} {kind}"})
    for sf in hb_payload["spawn_flips"]:
        events.append({"t": sf["t"], "type": "spawn", "team": sf["team_locked"],
                       "sev": sf["severity"], "label": f"{sf['team_locked']} spawn {sf['severity']}"})

    turning = []
    for h in hills:
        p0, p1 = _MODEL.predict(h.a_start, h.b_start), _MODEL.predict(h.a_end, h.b_end)
        turning.append({
            "hill": h.index, "name": h.name, "t_start": h.t_start, "t_end": h.t_end,
            "winner": h.winner, "margin": h.margin, "da": h.da, "db": h.db,
            "win_before": round(p0 * 100), "win_after": round(p1 * 100),
            "dwin": round((p1 - p0) * 100, 1), "narrative": _hill_narrative(h),
        })
    turning_ranked = sorted(turning, key=lambda r: abs(r["dwin"]), reverse=True)

    rotation = [{
        "hill": h.index, "name": h.name, "lat": h.da, "van": h.db,
        "first_control": h.winner, "control_a": h.control_a,
        "lat_rate": round(h.da / 60, 2), "van_rate": round(h.db / 60, 2),
    } for h in hills]

    return {
        "team_a": TEAM_A, "team_b": TEAM_B,
        "momentum": momentum,
        "hill_ticks": hill_ticks,
        "events": events,
        "turning_points": turning_ranked,
        "rotation": rotation,
        "rotation_note": (
            "Proxy = points scored per hill (control rate). True rotation time "
            "(seconds from hill spawn to first contact) and league/opponent "
            "baselines require minimap positional tracking and a multi-map corpus "
            "— both are roadmap, not yet measured here."
        ),
    }
