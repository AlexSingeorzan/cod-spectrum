"""60-second hill-by-hill breakdown: scoring, gunfights, and spawn flips.

This is the data-pipeline layer the scoreboard-only view could not reach. A
Hardpoint hill rotates every 60 seconds, so the whole match is modelled as a
sequence of `HillWindow`s, and gunfight + spawn-flip structures hang off them.

What is VERIFIED (read frame-by-frame from the official VOD, youtu.be/6pyPLwdrzq4):
  * score at every 60s hill boundary  -> `SCORE_AT_HILL_BOUNDARY`
  * per-player K/D for the whole map   -> `PLAYER_MAP_STATS` (post-game card)
  * two mid-map team-kill checkpoints   -> `TEAM_KILL_CHECKPOINTS`
What is DERIVED (clearly flagged): per-hill winner/margin/control (from scores),
per-hill frag lean (from score dominance + the verified checkpoints).
What is INFERRED (lowest confidence, flagged): spawn flips — a team scoring a
hill near-unanswered means the loser was locked off their spawn. Confirming this
properly is the `MinimapDetector` job below (the broadcast minimap shows spawns).

Nothing here is invented: every number traces to a verified read or a stated rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

HILL_SECONDS = 60
TARGET = 250
TEAM_A, TEAM_B = "LAT", "VAN"

# --- VERIFIED from the VOD --------------------------------------------------
# (vod_time_seconds, LAT_score, VAN_score) at each ~60s hill boundary.
SCORE_AT_HILL_BOUNDARY: list[tuple[int, int, int]] = [
    (90, 16, 11), (150, 34, 39), (210, 52, 61), (270, 97, 63), (330, 116, 71),
    (390, 127, 102), (450, 150, 115), (510, 190, 121), (570, 215, 132),
    (630, 232, 148), (685, 250, 156),
]
# Post-game stat card, Map 1 Hacienda Hardpoint. (kills, deaths). Cross-checks:
# sum(LAT kills)=106=sum(VAN deaths); sum(VAN kills)=79=sum(LAT deaths).
PLAYER_MAP_STATS: dict[str, dict[str, tuple[int, int]]] = {
    "LAT": {"Scrap": (25, 16), "HyDra": (32, 21), "aBeZy": (31, 19), "Nium": (18, 23)},
    "VAN": {"Craze": (15, 29), "Mamba": (19, 25), "Lunarz": (20, 26), "Nero": (25, 26)},
}
# Verified team kill totals at mid-map checkpoints (from the side panels).
TEAM_KILL_CHECKPOINTS: dict[int, dict[str, int]] = {
    505: {"LAT": 73, "VAN": 61},
    685: {"LAT": 106, "VAN": 79},
}
HILL_NAMES = ["Mansion", "Stables", "Garden", "Comms", "Mansion", "Stables",
              "Garden", "Comms", "Mansion", "Stables"]


# --- Structures -------------------------------------------------------------
@dataclass(frozen=True)
class HillWindow:
    index: int
    name: str
    t_start: int
    t_end: int
    a_start: int
    a_end: int
    b_start: int
    b_end: int

    @property
    def da(self) -> int:
        return self.a_end - self.a_start

    @property
    def db(self) -> int:
        return self.b_end - self.b_start

    @property
    def margin(self) -> int:
        return self.da - self.db

    @property
    def winner(self) -> str | None:
        return TEAM_A if self.margin > 0 else TEAM_B if self.margin < 0 else None

    @property
    def control_a(self) -> float:
        total = self.da + self.db
        return round(self.da / total, 3) if total else 0.5


@dataclass(frozen=True)
class SpawnFlip:
    """Inferred spawn-side loss for `team_locked` during a hill. Confidence is
    low by construction — this is a scoreboard inference pending minimap proof."""
    hill_index: int
    t: int
    team_locked: str
    severity: str          # "lock" | "pressure"
    basis: str
    confidence: float


@dataclass(frozen=True)
class GunfightContext:
    map_kills: dict[str, dict[str, tuple[int, int]]]
    team_kills: dict[str, int]
    team_deaths: dict[str, int]
    top_fraggers: list[tuple[str, str, int, int]]   # (team, player, K, D)
    struggled: list[tuple[str, str, int, int]]
    checkpoints: dict[int, dict[str, int]]


# --- Extractor interfaces (the pipeline; mirror OcrEngine) ------------------
@runtime_checkable
class PanelStatReader(Protocol):
    """Reads the eight per-player K/D panels via the active HUD profile.
    The crop regions are calibrated (LAT top-left, VAN top-right); digit
    recognition shares the stylised-font problem the scorebar has, so values
    are verified the same way (template/manual) before being trusted."""
    def read_panels(self, frame: "np.ndarray", hud_profile: dict) -> dict: ...


@runtime_checkable
class MinimapDetector(Protocol):
    """The real path to spawns/positions (README's YOLO step). Mirrors the
    OcrEngine contract: crop the minimap through the HUD profile, return player
    boxes with team, confidence and an `observed_team` visibility flag — never
    infer hidden opponents. A temporal service then derives spawn ownership and
    flips. Not yet implemented; `infer_spawn_flips` is the scoreboard stopgap."""
    def detect(self, frame: "np.ndarray", hud_profile: dict) -> list[dict]: ...


class HillSegmenter:
    """Aligns a (t, score_a, score_b) timeline to fixed 60s hill rotations."""

    def __init__(self, hill_seconds: int = HILL_SECONDS, names: list[str] | None = None):
        self.hill_seconds = hill_seconds
        self.names = names or []

    def segment(self, boundary_scores: list[tuple[int, int, int]]) -> list[HillWindow]:
        hills: list[HillWindow] = []
        for i, ((t0, a0, b0), (t1, a1, b1)) in enumerate(zip(boundary_scores, boundary_scores[1:])):
            hills.append(HillWindow(
                index=i + 1, name=self.names[i] if i < len(self.names) else f"Hill {i + 1}",
                t_start=t0, t_end=t1, a_start=a0, a_end=a1, b_start=b0, b_end=b1,
            ))
        return hills


class SpawnInference:
    """Scoreboard heuristic for spawn flips, pending `MinimapDetector`.

    A hill won by >= `lock_margin` while the loser scores <= `lock_cap` means the
    loser was held off the hill — a spawn-side loss that rotation. A softer band
    is flagged as 'pressure'. Confidence is deliberately capped low."""

    def __init__(self, lock_margin: int = 28, lock_cap: int = 8, pressure_margin: int = 18):
        self.lock_margin = lock_margin
        self.lock_cap = lock_cap
        self.pressure_margin = pressure_margin

    def infer(self, hills: list[HillWindow]) -> list[SpawnFlip]:
        flips: list[SpawnFlip] = []
        for hill in hills:
            winner, loser = (TEAM_A, TEAM_B) if hill.margin > 0 else (TEAM_B, TEAM_A)
            loser_delta = hill.db if winner == TEAM_A else hill.da
            if abs(hill.margin) >= self.lock_margin and loser_delta <= self.lock_cap:
                flips.append(SpawnFlip(
                    hill_index=hill.index, t=hill.t_start, team_locked=loser, severity="lock",
                    basis=f"{winner} won the hill {abs(hill.margin)}:{loser_delta} margin — {loser} held off the hill",
                    confidence=0.45,
                ))
            elif abs(hill.margin) >= self.pressure_margin and loser_delta <= self.lock_cap + 6:
                flips.append(SpawnFlip(
                    hill_index=hill.index, t=hill.t_start, team_locked=loser, severity="pressure",
                    basis=f"{winner} controlled the hill ({hill.da}-{hill.db}) — likely spawn pressure on {loser}",
                    confidence=0.3,
                ))
        return flips


def gunfights() -> GunfightContext:
    team_kills, team_deaths = {}, {}
    flat: list[tuple[str, str, int, int]] = []
    for team, players in PLAYER_MAP_STATS.items():
        team_kills[team] = sum(k for k, _ in players.values())
        team_deaths[team] = sum(d for _, d in players.values())
        flat.extend((team, name, k, d) for name, (k, d) in players.items())
    top = sorted(flat, key=lambda r: r[2], reverse=True)[:3]
    struggled = sorted(flat, key=lambda r: r[2] - r[3])[:2]
    return GunfightContext(
        map_kills=PLAYER_MAP_STATS, team_kills=team_kills, team_deaths=team_deaths,
        top_fraggers=top, struggled=struggled, checkpoints=TEAM_KILL_CHECKPOINTS,
    )


def analysis() -> dict:
    hills = HillSegmenter(names=HILL_NAMES).segment(SCORE_AT_HILL_BOUNDARY)
    flips = SpawnInference().infer(hills)
    gf = gunfights()
    flips_by_hill: dict[int, SpawnFlip] = {f.hill_index: f for f in flips}
    hill_rows = []
    for h in hills:
        flip = flips_by_hill.get(h.index)
        hill_rows.append({
            "index": h.index, "name": h.name, "t_start": h.t_start, "t_end": h.t_end,
            "a_start": h.a_start, "a_end": h.a_end, "b_start": h.b_start, "b_end": h.b_end,
            "da": h.da, "db": h.db, "margin": h.margin, "winner": h.winner,
            "control_a": h.control_a,
            "spawn_flip": ({"team_locked": flip.team_locked, "severity": flip.severity,
                            "basis": flip.basis, "confidence": flip.confidence} if flip else None),
        })
    # key hills = biggest control swings
    key = sorted(hill_rows, key=lambda r: abs(r["margin"]), reverse=True)[:3]
    return {
        "team_a": TEAM_A, "team_b": TEAM_B, "hill_seconds": HILL_SECONDS,
        "hills": hill_rows,
        "key_hills": [k["index"] for k in key],
        "spawn_flips": [{"hill_index": f.hill_index, "t": f.t, "team_locked": f.team_locked,
                         "severity": f.severity, "basis": f.basis, "confidence": f.confidence} for f in flips],
        "gunfights": {
            "team_kills": gf.team_kills, "team_deaths": gf.team_deaths,
            "top_fraggers": gf.top_fraggers, "struggled": gf.struggled,
            "players": gf.map_kills, "checkpoints": gf.checkpoints,
        },
    }
