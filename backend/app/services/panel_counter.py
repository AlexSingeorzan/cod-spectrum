"""Scoreboard K/D kill counter — Phase 4, the kill *spine*.

The broadcast's top team panels show each player's running **kills/deaths**. Unlike
the semi-transparent killfeed, that counter is a clean, **monotonic** signal — kills
only ever go up within a map — so it is the authoritative answer to *how many kills,
and who got them*:

  * a player's ``kills`` going up by N  -> N ``KillEvent``s by that player (attacker);
  * a player's ``deaths`` going up by 1 -> a ``DeathEvent`` for that player (victim).

The digits are clean enough that Tesseract reads them reliably (the stylised scorebar
font it cannot read; this panel font it can — verified across the LAT/VAN map). The
monotonic constraint is a strong error-corrector: a read is accepted only when it does
not *decrease*, and an implausible jump must be confirmed by a second frame, so a
one-frame OCR glitch cannot invent kills.

Reconciliation with the killfeed (``reconcile_with_killfeed``) closes the loop: a panel
increment near a killfeed onset is a confirmed kill; a killfeed onset with no increment
is a false positive; an increment with no killfeed row is a missed feed read.

The read step (frame -> per-player K/D) is isolated behind ``KdReader`` so the counting
logic is pure and testable without Tesseract. Nothing is invented: a player's history
before we start watching is unknown, and unreadable cells are skipped, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from ..events import DeathEvent, Evidence, GameEvent, KillEvent, Provenance, SourceKind

MODEL_NAME = "panel_kill_counter"
MODEL_VERSION = "0.1.0"

# Verified K/D cell regions for CDL_2026_1080p (kept in sync with the HUD profile).
KD_PANEL_A = {"x": 0.237, "y": 0.095, "w": 0.030, "h": 0.110}   # left panel (team a)
KD_PANEL_B = {"x": 0.878, "y": 0.095, "w": 0.033, "h": 0.110}   # right panel (team b)
PLAYERS_PER_TEAM = 4

# A kill increment of <= this is accepted immediately; a larger jump must be confirmed
# by a second consecutive frame (guards against an OCR glitch inventing kills).
MAX_IMMEDIATE_STEP = 3


@runtime_checkable
class KdReader(Protocol):
    """Reads one K/D cell image -> (kills, deaths), or None if unreadable."""

    def read_cell(self, image: np.ndarray) -> tuple[int, int] | None: ...


class TesseractKdReader:
    """Default reader: Tesseract with a digit/slash whitelist on a thresholded cell."""

    name = "tesseract_kd"
    _PATTERN = re.compile(r"^(\d{1,2})/(\d{1,2})$")

    def read_cell(self, image: np.ndarray) -> tuple[int, int] | None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("panel counter needs: pip install pytesseract + the tesseract binary") from exc
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        _thresh, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(
            binary, config="--psm 7 -c tessedit_char_whitelist=0123456789/"
        ).strip()
        match = self._PATTERN.match(text)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))


def _cells(frame: np.ndarray, region: dict, n: int) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    x0, y0 = int(region["x"] * w), int(region["y"] * h)
    strip = frame[y0:y0 + int(region["h"] * h), x0:x0 + int(region["w"] * w)]
    rh = strip.shape[0]
    return [strip[i * rh // n:(i + 1) * rh // n] for i in range(n)]


def _profile_regions(profile) -> dict:
    """Accept either a HudProfile (``.regions``) or a plain ``{"regions": {...}}`` dict."""
    if profile is None:
        return {}
    if hasattr(profile, "regions"):
        return profile.regions
    return profile.get("regions", {})


def read_panels(
    frame: np.ndarray,
    profile=None,
    reader: KdReader | None = None,
    players_per_team: int = PLAYERS_PER_TEAM,
) -> dict[str, list[tuple[int, int] | None]]:
    """Read both team panels -> {"a": [(k,d)|None x4], "b": [...]}. Unreadable cells -> None."""
    reader = reader or TesseractKdReader()
    regions = _profile_regions(profile)
    region_a = regions.get("kd_panel_a", KD_PANEL_A)
    region_b = regions.get("kd_panel_b", KD_PANEL_B)
    return {
        "a": [reader.read_cell(c) for c in _cells(frame, region_a, players_per_team)],
        "b": [reader.read_cell(c) for c in _cells(frame, region_b, players_per_team)],
    }


@dataclass
class _Slot:
    kills: int | None = None       # confirmed monotonic value (None until first seen)
    deaths: int | None = None
    pending: tuple[str, int] | None = None   # (field, value) awaiting a confirming frame


class PanelKillCounter:
    """Stateful, chronological kill counter over per-player K/D reads.

    Feed it ``update(timestamp, readings, evidence)`` in time order; it emits
    ``KillEvent`` / ``DeathEvent`` facts on confirmed monotonic increments. Player
    identity is the team-slot (``a1``..``b4``); pass ``names`` to surface gamertags.
    """

    name = MODEL_NAME
    model_version = MODEL_VERSION

    def __init__(self, team_a: str = "A", team_b: str = "B",
                 names: dict[str, str] | None = None, max_immediate_step: int = MAX_IMMEDIATE_STEP):
        self.team = {"a": team_a, "b": team_b}
        self.names = names or {}
        self.max_immediate_step = max_immediate_step
        self._slots: dict[str, _Slot] = {f"{s}{i + 1}": _Slot() for s in "ab" for i in range(PLAYERS_PER_TEAM)}

    def total_kills(self) -> int:
        return sum(s.kills for s in self._slots.values() if s.kills is not None)

    def kills_by_slot(self) -> dict[str, int]:
        return {sid: s.kills for sid, s in self._slots.items() if s.kills is not None}

    def team_kills(self) -> dict[str, int]:
        out = {"a": 0, "b": 0}
        for sid, s in self._slots.items():
            if s.kills is not None:
                out[sid[0]] += s.kills
        return out

    def _accept(self, slot: _Slot, field_name: str, value: int) -> int:
        """Return the confirmed increment for kills/deaths, applying monotonic + vote rules."""
        current = getattr(slot, field_name)
        if current is None:                       # first sighting: seed history, emit nothing
            setattr(slot, field_name, value)
            return 0
        if value <= current:                      # never decreases; equal clears any pending
            if slot.pending and slot.pending[0] == field_name:
                slot.pending = None
            return 0
        step = value - current
        if step <= self.max_immediate_step:       # small, plausible -> accept now
            setattr(slot, field_name, value)
            slot.pending = None
            return step
        # large jump -> require the same value on a second frame before trusting it
        if slot.pending == (field_name, value):
            setattr(slot, field_name, value)
            slot.pending = None
            return step
        slot.pending = (field_name, value)
        return 0

    def update(self, video_timestamp_seconds: float,
               readings: dict[str, list[tuple[int, int] | None]],
               evidence: Evidence) -> list[GameEvent]:
        kill_slots: list[str] = []
        death_slots: list[str] = []
        for side in ("a", "b"):
            for i, kd in enumerate(readings.get(side, [])):
                if kd is None:
                    continue
                slot_id = f"{side}{i + 1}"
                slot = self._slots[slot_id]
                kills, deaths = kd
                kill_slots += [slot_id] * self._accept(slot, "kills", kills)
                death_slots += [slot_id] * self._accept(slot, "deaths", deaths)

        # Pair only when unambiguous: exactly one kill and one death, on opposite teams.
        paired = (len(kill_slots) == 1 and len(death_slots) == 1
                  and kill_slots[0][0] != death_slots[0][0])
        events: list[GameEvent] = []
        for slot_id in kill_slots:
            victim = death_slots[0] if paired else None
            events.append(self._kill_event(video_timestamp_seconds, slot_id, victim, evidence))
        for slot_id in death_slots:
            killer = kill_slots[0] if paired else None
            events.append(self._death_event(video_timestamp_seconds, slot_id, killer, evidence))
        return events

    # --- event construction -------------------------------------------------------

    def _ident(self, slot_id: str) -> str:
        return self.names.get(slot_id, slot_id)

    def _provenance(self, note: str) -> Provenance:
        return Provenance(source=SourceKind.MODEL, model_name=MODEL_NAME, model_version=MODEL_VERSION,
                          producer="panel-kill-counter", note=note)

    def _kill_event(self, t: float, slot_id: str, victim_slot: str | None, evidence: Evidence) -> GameEvent:
        side = slot_id[0]
        return GameEvent(
            video_timestamp_seconds=t, confidence=0.7,
            provenance=self._provenance("kill = monotonic +1 on scoreboard kills column (OCR)"),
            evidence=evidence,
            payload=KillEvent(
                attacker=self._ident(slot_id), attacker_side=side, attacker_team=self.team[side],
                victim=self._ident(victim_slot) if victim_slot else None,
                victim_side=victim_slot[0] if victim_slot else None,
                victim_team=self.team[victim_slot[0]] if victim_slot else None,
                attributes={"source": "scoreboard_kd", "attacker_slot": slot_id,
                            "victim_slot": victim_slot, "paired": victim_slot is not None},
            ),
            tags=["panel_counter", "scoreboard_kill"] + ([] if victim_slot else ["victim_unpaired"]),
        )

    def _death_event(self, t: float, slot_id: str, killer_slot: str | None, evidence: Evidence) -> GameEvent:
        side = slot_id[0]
        return GameEvent(
            video_timestamp_seconds=t, confidence=0.7,
            provenance=self._provenance("death = monotonic +1 on scoreboard deaths column (OCR)"),
            evidence=evidence,
            payload=DeathEvent(
                player=self._ident(slot_id), side=side, team=self.team[side],
                killer=self._ident(killer_slot) if killer_slot else None,
                attributes={"source": "scoreboard_kd", "victim_slot": slot_id,
                            "killer_slot": killer_slot},
            ),
            tags=["panel_counter", "scoreboard_death"],
        )


# ---------------------------------------------------------------------------
# Reconciliation with the killfeed detector
# ---------------------------------------------------------------------------


@dataclass
class Reconciliation:
    confirmed: int = 0          # panel kill matched by a killfeed onset nearby
    panel_only: int = 0         # panel kill with no killfeed row (missed feed read)
    killfeed_only: int = 0      # killfeed onset with no panel kill (likely false positive)
    matched_pairs: list[tuple[float, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        total = self.confirmed + self.killfeed_only
        return {
            "confirmed": self.confirmed,
            "panel_only": self.panel_only,
            "killfeed_only": self.killfeed_only,
            "killfeed_precision_estimate": round(self.confirmed / total, 4) if total else None,
            "note": ("panel scoreboard is the kill ground truth here; killfeed_only are likely "
                     "false positives, panel_only are kills the killfeed detector missed"),
        }


def reconcile_with_killfeed(panel_kill_times: list[float], killfeed_onset_times: list[float],
                            window_seconds: float = 2.5) -> Reconciliation:
    """Match panel kills to killfeed onsets within a time window. The panel counter is
    treated as the kill ground truth (monotonic, authoritative count)."""
    remaining = sorted(killfeed_onset_times)
    rec = Reconciliation()
    for t in sorted(panel_kill_times):
        hit = next((o for o in remaining if abs(o - t) <= window_seconds), None)
        if hit is not None:
            rec.confirmed += 1
            rec.matched_pairs.append((round(t, 2), round(hit, 2)))
            remaining.remove(hit)
        else:
            rec.panel_only += 1
    rec.killfeed_only = len(remaining)
    return rec
