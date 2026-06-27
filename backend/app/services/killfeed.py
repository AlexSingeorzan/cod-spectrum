"""Killfeed detection (classical CV) — Phase 4, deliverable 1.

This is the honest first layer of killfeed analysis. It is deliberately split into
what classical CV can actually do reliably versus what needs a trained, *labelled*
model — and it never crosses that line.

  * What this DOES (a real, evidence-backed fact candidate): it LOCALISES killfeed
    entries — the horizontal kill-notification rows in the broadcast's top-right
    killfeed region — frame by frame, and a positional tracker collapses a row that
    persists across many frames into a single *kill onset* with a timestamp.
    That yields kill **timing and count** facts: "a kill notification appeared at
    3:32", backed by the crop, with a confidence. Coaches care about kill timing
    against the hill clock even before names are read.

  * What this explicitly does NOT do: it does not read the attacker, victim, or
    weapon. Those are stylised, team-coloured names + weapon icons that need a
    labelled dataset and a trained reader (Phase 4, deliverable 2). Emitted
    ``KillEvent``s therefore carry ``attacker=victim=kill_type=weapon=None`` and the tag
    ``identity_unread``. We never invent a name we did not read.

Upgrade path (documented, not claimed done): the dataset scaffold
(``scripts/build_killfeed_dataset.py``) turns these detections into an annotation
set; once rows are human-labelled, a content reader fills attacker/victim/kill_type
and optional exact weapon metadata, and the same onsets unlock ``DeathEvent`` /
``WeaponEvent`` / ``TradeEvent``. The
detector interface, evidence persistence and onset tracking stay identical.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from ..events import Evidence, GameEvent, KillEvent, Provenance, SourceKind

MODEL_NAME = "killfeed_classical"
MODEL_VERSION = "0.1.0"

# Killfeed region in the CDL_2026 1080p broadcast (fractions of frame). This is the
# LEFT-CENTRE feed (attacker -> weapon -> victim), VERIFIED against data/videos/
# lat_van.mp4 — the HUD profile's original top-right "killfeed" box was actually the
# opponent stats panel. Callers should pass the HUD profile (single source of truth);
# this is the fallback, kept in sync with data/configs/hud_profiles/CDL_2026_1080p.json.
KILLFEED_REGION = {"x": 0.012, "y": 0.39, "w": 0.235, "h": 0.19}

# A killfeed row is wide and short. These bounds are expressed as fractions of the
# killfeed-region height/width and gate noise (icons, score popups) out of the
# horizontal-band search. They are detector parameters, not learned weights.
_MIN_ROW_H = 0.045
_MAX_ROW_H = 0.230
_MIN_ROW_W = 0.34          # a real entry spans attacker + icon + victim
_BAND_INK_THRESHOLD = 0.05  # min fraction of "ink" pixels per scanline to be in a row


@dataclass(frozen=True)
class KillfeedRow:
    """One detected kill-notification row, localised within the killfeed region.

    Coordinates are normalised to the killfeed crop (0..1). ``left_hue`` /
    ``right_hue`` are low-confidence team colour hints (attacker on the left, victim
    on the right) for the annotation scaffold — they are never turned into a name
    here. ``y_center`` is what the tracker keys on to follow a row across frames.
    """

    x: float
    y: float
    w: float
    h: float
    ink_fill: float
    confidence: float
    left_hue: int | None = None
    right_hue: int | None = None

    @property
    def y_center(self) -> float:
        return self.y + self.h / 2

    def box_list(self) -> list[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.w, 4), round(self.h, 4)]

    def as_dict(self) -> dict:
        return {
            "box": self.box_list(),
            "ink_fill": round(self.ink_fill, 4),
            "confidence": round(self.confidence, 3),
            "left_hue": self.left_hue,
            "right_hue": self.right_hue,
        }


@dataclass(frozen=True)
class KillOnset:
    """The first frame at which a distinct killfeed row appeared = a candidate kill."""

    video_timestamp_seconds: float
    row: KillfeedRow

    @property
    def confidence(self) -> float:
        return self.row.confidence


def crop_killfeed(frame: np.ndarray, region: dict | None = None) -> np.ndarray:
    region = region or KILLFEED_REGION
    h, w = frame.shape[:2]
    x0, y0 = int(region["x"] * w), int(region["y"] * h)
    return frame[y0:y0 + int(region["h"] * h), x0:x0 + int(region["w"] * w)]


def _ink_mask(crop: np.ndarray) -> np.ndarray:
    """Binary mask of killfeed 'ink': saturated team-coloured names + bright icons.

    This is the only background-invariant signal we can lean on: killfeed text is
    either strongly saturated (team colours) or near-white (weapon icon / white
    text). Muted game backgrounds fall away. It is imperfect on bright/saturated
    scenery — which is exactly why detections are *measured* against labels, never
    assumed correct.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    saturated = (s > 90) & (v > 110)
    bright = (s < 60) & (v > 205)
    mask = (saturated | bright).astype(np.uint8) * 255
    # Bridge characters within a row into a connected horizontal band.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))


def _dominant_hue(hsv_slice: np.ndarray) -> int | None:
    if hsv_slice.size == 0:
        return None
    h, s, v = cv2.split(hsv_slice)
    sel = (s > 90) & (v > 110)
    if int(sel.sum()) < 5:
        return None
    return int(np.median(h[sel]))


def _row_confidence(ink_fill: float, width_frac: float, height_frac: float) -> float:
    """Modest confidence from how 'row-like' a band is. Capped: this is an
    unverified classical detector, so it must not emit near-certainty."""
    density = min(1.0, ink_fill / 0.45)
    width = min(1.0, (width_frac - _MIN_ROW_W) / (1.0 - _MIN_ROW_W)) if width_frac > _MIN_ROW_W else 0.0
    # rows are short relative to the region; penalise tall blobs
    shape = 1.0 - min(1.0, max(0.0, (height_frac - _MIN_ROW_H) / (_MAX_ROW_H - _MIN_ROW_H)) * 0.5)
    score = 0.5 * density + 0.35 * width + 0.15 * shape
    return round(min(0.85, max(0.0, score)), 4)


def _runs_above(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where a 1-D signal exceeds a threshold."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(values):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


class KillfeedDetector:
    """Classical, training-free killfeed row localiser.

    ``detect`` mirrors the project's detector contract (frame + optional HUD profile
    -> list of plain dicts with boxes + confidence), so a trained model can later
    drop in behind the same call.
    """

    name = MODEL_NAME
    model_version = MODEL_VERSION

    def __init__(self, require_team_color: bool = True):
        # Kill notifications carry team-coloured names; a desaturated full-width band
        # is almost always a stats/HUD table, not the feed. On by default; the only
        # broadcast-general assumption the detector makes, and it is tunable.
        self.require_team_color = require_team_color

    def detect(self, frame: np.ndarray, hud_profile: dict | None = None) -> list[dict]:
        region = (hud_profile or {}).get("regions", {}).get("killfeed", KILLFEED_REGION)
        return [row.as_dict() for row in self.detect_rows(frame, region)]

    def detect_rows(self, frame: np.ndarray, region: dict | None = None) -> list[KillfeedRow]:
        crop = crop_killfeed(frame, region)
        return self._detect_crop(crop)

    def _detect_crop(self, crop: np.ndarray) -> list[KillfeedRow]:
        ch, cw = crop.shape[:2]
        if ch < 8 or cw < 8:
            return []
        mask = _ink_mask(crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        proj = mask.sum(axis=1) / (255.0 * cw)           # ink fraction per scanline
        proj = np.convolve(proj, np.ones(3) / 3.0, mode="same")  # light vertical smooth

        rows: list[KillfeedRow] = []
        for y0, y1 in _runs_above(proj, _BAND_INK_THRESHOLD):
            height_frac = (y1 - y0) / ch
            if not (_MIN_ROW_H <= height_frac <= _MAX_ROW_H):
                continue
            band = mask[y0:y1]
            cols = np.where(band.sum(axis=0) > 0)[0]
            if cols.size == 0:
                continue
            x0, x1 = int(cols.min()), int(cols.max()) + 1
            width_frac = (x1 - x0) / cw
            if width_frac < _MIN_ROW_W:
                continue
            hsv_band = hsv[y0:y1, x0:x1]
            mid = (x1 - x0) // 2
            left_hue = _dominant_hue(hsv_band[:, :mid])
            right_hue = _dominant_hue(hsv_band[:, mid:])
            if self.require_team_color and left_hue is None and right_hue is None:
                continue
            ink_fill = float(band[:, x0:x1].mean() / 255.0)
            confidence = _row_confidence(ink_fill, width_frac, height_frac)
            rows.append(KillfeedRow(
                x=x0 / cw, y=y0 / ch, w=width_frac, h=height_frac,
                ink_fill=ink_fill, confidence=confidence,
                left_hue=left_hue, right_hue=right_hue,
            ))
        return rows


class KillfeedTracker:
    """Collapses flickering per-frame row detections into candidate kill onsets.

    A broadcast killfeed is semi-transparent over a moving scene, so the classical
    detector blinks the same row in and out frame to frame; a content hash of the
    tiny crop is not stable enough to dedupe it (measured: nominally-identical rows
    drift ~0.4 of bits). *Position* is stable, though — a kill occupies a vertical
    slot in the feed for a few seconds. So we track active slots with a time-to-live:
    a detected row near an active slot REFRESHES it (the same kill persisting, even
    through a one-frame dropout); a row in a free slot starts a new slot = a candidate
    kill onset.

    Best-effort, not ground truth: two near-simultaneous kills sharing a slot can be
    merged (under-count) and a faded row can be missed. ``eval_killfeed.py`` quantifies
    exactly this against human labels, and the annotation workflow lets the labeller
    add missed kills + mark false positives. No kill identity is read here.
    """

    def __init__(self, ttl_seconds: float = 3.0, y_tolerance: float = 0.10,
                 min_confidence: float = 0.45, position_ema: float = 0.4):
        self.ttl_seconds = ttl_seconds
        self.y_tolerance = y_tolerance
        self.min_confidence = min_confidence
        self.position_ema = position_ema
        self._tracks: list[dict] = []   # {"y": float, "t": float}

    def update(self, video_timestamp_seconds: float, rows: Iterable[KillfeedRow]) -> list[KillOnset]:
        t = video_timestamp_seconds
        self._tracks = [tr for tr in self._tracks if t - tr["t"] <= self.ttl_seconds]
        onsets: list[KillOnset] = []
        for row in sorted(rows, key=lambda r: r.confidence, reverse=True):
            if row.confidence < self.min_confidence:
                continue
            yc = row.y_center
            best: dict | None = None
            best_dist = self.y_tolerance
            for track in self._tracks:
                dist = abs(track["y"] - yc)
                if dist < best_dist:
                    best, best_dist = track, dist
            if best is not None:                   # same slot persisting -> not a new kill
                best["t"] = t
                best["y"] = (1 - self.position_ema) * best["y"] + self.position_ema * yc
                continue
            self._tracks.append({"y": yc, "t": t})
            onsets.append(KillOnset(video_timestamp_seconds=t, row=row))
        return onsets


def detect_kill_onsets(
    frames: Iterable[tuple[float, np.ndarray]],
    *,
    region: dict | None = None,
    detector: KillfeedDetector | None = None,
    tracker: KillfeedTracker | None = None,
) -> Iterator[KillOnset]:
    """Stream (timestamp, frame) pairs -> distinct kill onsets, in time order."""
    detector = detector or KillfeedDetector()
    tracker = tracker or KillfeedTracker()
    for timestamp, frame in frames:
        rows = detector.detect_rows(frame, region)
        yield from tracker.update(timestamp, rows)


def onset_to_kill_event(
    onset: KillOnset,
    *,
    crop_path: str,
    frame_path: str | None = None,
    source_url: str | None = None,
    broadcast_id: int | None = None,
    match_id: int | None = None,
    map_id: int | None = None,
) -> GameEvent:
    """Wrap a kill onset as a candidate ``KillEvent`` fact.

    Identity is unread on purpose: attacker/victim/kill_type/weapon stay ``None``
    and the event is tagged ``identity_unread``. Evidence (the row crop) is required by the
    envelope, so a candidate kill can never exist without something to look at.
    """
    row = onset.row
    payload = KillEvent(
        attacker=None, attacker_team=None,
        victim=None, victim_team=None,
        weapon=None, headshot=None, is_trade=None,
        attributes={
            "identity": "unread",
            "detector": f"{MODEL_NAME}@{MODEL_VERSION}",
            "row_box": row.box_list(),
            "row_ink_fill": round(row.ink_fill, 4),
            "left_color_hue_hint": row.left_hue,
            "right_color_hue_hint": row.right_hue,
        },
    )
    return GameEvent(
        broadcast_id=broadcast_id, match_id=match_id, map_id=map_id,
        video_timestamp_seconds=onset.video_timestamp_seconds,
        confidence=onset.confidence,
        provenance=Provenance(
            source=SourceKind.HEURISTIC,
            model_name=MODEL_NAME, model_version=MODEL_VERSION,
            producer="killfeed-detector",
            note="killfeed row localised; attacker/victim/kill_type/weapon not read",
        ),
        evidence=Evidence(
            video_timestamp_seconds=onset.video_timestamp_seconds,
            crop_path=crop_path, frame_path=frame_path, source_url=source_url,
        ),
        payload=payload,
        tags=["killfeed", "candidate", "identity_unread"],
    )
